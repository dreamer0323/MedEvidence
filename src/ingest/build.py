"""建库：parse → chunk → 过滤 → 去重（已向量化的块跳过）→ 写 Milvus Lite。

设计取舍都写在 plan/09-决策记录.md「建库：逐块 text_hash + 文本清洗」一节：

  · 幂等的判据是 **`text_hash(chunk.embed_input)`**，不是文章 id、也不是"文件存在"。
    文本变了 hash 就变 → 自动重算；文本没变 → 一分钱不花。
  · 首次建库没有"可跳过"的块，hash 的价值在第二次以后（清洗/修 parser 时）。
  · 过滤（样板节）只在这里做，不污染 chunk_index 锚点。

用法::

    python -m src.ingest.build --dry-run          # 只统计，不连库、不调 API
    python -m src.ingest.build --limit 20         # 先跑 20 篇试水
    python -m src.ingest.build                    # 全量（可反复跑，幂等）
    python -m src.ingest.build --drop             # 重建集合（数据全丢）
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path

from src.config import MILVUS_URI, RAW_DIR, setup_logging
from src.embedding import EmbeddingClient
from src.ingest.chunk import Chunk, build_chunks
from src.ingest.parse_pmc import parse_article
from src.vectorstore import MAX_SECTION_LEN, ChunkRow, MilvusStore

logger = logging.getLogger(__name__)

# 一次读多少块再统一处理：1000 块 × 1024 维 float32 ≈ 4MB，内存和往返都合适
BATCH_CHUNKS = 1000

# 样板节规则：规则名 → 关键词（小写子串）。
# 匹配对象是 **path 全链 + 末级标题** —— 只看末级标题会漏掉
# `Supplementary information` 下面那些没有标题的附件条目（实测 144 个空标题节里 143 个）。
#
# 为什么做成表而不是一串 if：每条规则单独记账，跑完知道是哪一条丢了多少；
# 加/删一条不用改逻辑。规则依据见 plan/09 与 scratch/audit_titles.py。
BOILERPLATE_RULES: dict[str, tuple[str, ...]] = {
    "conflict": ("conflict of interest", "conflicts of interest", "competing interest",
                 "declaration of interest", "disclosure"),
    "contribution": ("author contribution", "authors' contribution", "authors’ contribution",
                     "credit authorship", "contributors"),
    "funding": ("funding",),
    "ethics": ("ethic", "consent"),
    "acknowledgments": ("acknowledg",),
    "supplementary": ("supplement", "supporting information", "additional file"),
    "source data": ("source data",),
    "reporting summary": ("reporting summary",),
    "availability": ("availability", "data sharing", "data accessibility"),
    "abbreviation": ("abbreviation",),
}


def boilerplate_rule(chunk: Chunk) -> str | None:
    """命中则返回规则名，否则 None。

    ⚠️ `Section.path` **已经包含末级标题**（parse 阶段 `here = path + [title]`），
    所以这里只拼 path；再拼一次 section 会得到 `Methods / Statistical Analysis /
    Statistical Analysis` —— 不影响匹配，但会让日志和调试看起来像有 bug。
    """
    haystack = " / ".join(chunk.path or [chunk.section]).lower()
    for rule, keywords in BOILERPLATE_RULES.items():
        if any(k in haystack for k in keywords):
            return rule
    return None


def chunk_pk(chunk: Chunk) -> str:
    """稳定的行身份：`{pmc_id}:{chunk_index}`。

    ⚠️ 不是内容哈希。内容相同的两段文本在不同文章里是**两块**，
    各自要有自己的引用；用内容当身份会让第二篇检索不到（见 plan/09 主键方案 C）。
    """
    return f"{chunk.pmc_id}:{chunk.chunk_index}"


def text_hash(chunk: Chunk) -> str:
    """哈希对象 = 喂给 embedding 的字符串，不是入库的 text。

    ⚠️ 为什么是 embed_input：向量由它算出来，它没变，向量就还能用。
    用 text 当哈希会漏掉"正文没变、标题变了"的情况（标题进了 embed_input）。
    """

    embed_raw_text = chunk.embed_input          # @property，不是方法：别加括号
    embed_str = embed_raw_text.encode("utf-8")   # 采用 utf-8 编码，与 embedding 一致

    return hashlib.sha256(embed_str).hexdigest()


def to_row(chunk: Chunk, vector: list[float]) -> ChunkRow:
    """Chunk → 存储层契约。**全项目唯一的结构转换点**（见 docs/data-model.md）。

    `section` 超长时截断并打 warning：它只是展示字段，完整层级在 `path` 里。
    实测全量只有 1 个节超长（261 字符），为它重建整个集合不划算。
    ⚠️ `text` 不许这样处理——正文截断就是丢证据，超长必须抛错。
    """
    section = chunk.section
    if len(section) > MAX_SECTION_LEN:
        logger.warning(
            "section 超长截断：%s chunk=%d %d→%d（完整层级在 path）",
            chunk.pmc_id, chunk.chunk_index, len(section), MAX_SECTION_LEN,
        )
        section = section[:MAX_SECTION_LEN]

    return ChunkRow(
        pk=chunk_pk(chunk),
        vector=vector,
        text=chunk.text,
        pmc_id=chunk.pmc_id,
        section=section,
        chunk_index=chunk.chunk_index,
        path=" / ".join(chunk.path),   # Milvus 列是 VARCHAR，列表要拼成字符串
        source_type=chunk.source_type,
        text_hash=text_hash(chunk),
    )


def iter_articles(raw_dir: Path, limit: int | None, stats: Counter | None = None):
    """按文档序产出 `(pmc_id, [要入库的 chunk])`。

    对账必须以**篇**为单位：要判断"库里多出来的 pk"，必须先知道这一篇的完整 pk 集合。
    """
    stats = Counter() if stats is None else stats
    files = sorted(raw_dir.glob("*.xml"))
    if limit:
        files = files[:limit]

    for path in files:
        try:
            article = parse_article(path.read_bytes())
        except Exception as err:                       # noqa: BLE001
            stats["parse_failed"] += 1
            logger.warning("%s 解析失败：%s", path.name, err)
            continue

        kept: list[Chunk] = []
        for chunk in build_chunks(article):
            rule = boilerplate_rule(chunk)
            if rule:
                stats[f"dropped:{rule}"] += 1
                continue
            kept.append(chunk)

        stats["articles"] += 1
        stats["chunks"] += len(kept)
        yield article.pmc_id, kept


def plan_reconciliation(chunks: list[Chunk], existing: dict[str, str]):
    """对账（纯函数，好测）：返回 `(要向量化的块, 要删除的 pk, 跳过数)`。

    · 库里没有这个 pk → 新块，要向量化
    · 有 pk 但 hash 不同 → 文本变了，要重新向量化（upsert 覆盖）
    · 有 pk 且 hash 相同 → 跳过，一分钱不花
    · 库里有、当前没有 → 影子块 / 孤儿，删除
    """
    current = {chunk_pk(c): text_hash(c) for c in chunks}
    to_embed = [c for c in chunks if existing.get(chunk_pk(c)) != current[chunk_pk(c)]]
    stale = [pk for pk in existing if pk not in current]
    return to_embed, stale, len(chunks) - len(to_embed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedEvidence 建库")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（试水用）")
    parser.add_argument("--drop", action="store_true", help="重建集合（数据全丢）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不连库、不调 API")
    args = parser.parse_args(argv)

    setup_logging()
    logger.info("MILVUS_URI = %s", MILVUS_URI)

    stats = Counter()
    articles = iter_articles(Path(RAW_DIR), args.limit, stats)

    if args.dry_run:
        for _pmc_id, _chunks in articles:
            pass
        print(f"文章 {stats['articles']}（解析失败 {stats['parse_failed']}）")
        print(f"入库块 {stats['chunks']}")
        for key in sorted(k for k in stats if k.startswith("dropped:")):
            print(f"  过滤 {key[8:]:16s} {stats[key]}")
        return 0

    client = EmbeddingClient()
    api_texts = 0                 # 实际送去 embedding 的条数（幂等验收用）
    pending: list[Chunk] = []     # 需要向量化的块（新增 + 变更）
    stale: list[str] = []         # 库里多余、要删除的 pk

    with MilvusStore() as store:
        store.ensure_collection(drop=args.drop)

        # 阶段一：按篇对账，决定"谁要花钱、谁要删"（不调 API）
        for pmc_id, chunks in articles:
            existing = store.rows_by_pmc(pmc_id)
            todo, gone, skipped = plan_reconciliation(chunks, existing)
            pending.extend(todo)
            stale.extend(gone)
            stats["skipped"] += skipped
            stats["new"] += sum(1 for c in todo if chunk_pk(c) not in existing)
            stats["updated"] += len(todo) - sum(
                1 for c in todo if chunk_pk(c) not in existing
            )

        # 阶段二：批量向量化 + upsert（钱花在这里）
        for start in range(0, len(pending), BATCH_CHUNKS):
            batch = pending[start : start + BATCH_CHUNKS]
            vectors = client.embed(
                [c.embed_input for c in batch], desc=f"第 {start // BATCH_CHUNKS + 1} 批"
            )
            api_texts += len(batch)
            rows = [to_row(c, v) for c, v in zip(batch, vectors)]
            stats["upserted"] += store.upsert(rows)

        # 阶段三：清理影子块 / 孤儿行
        if stale:
            stats["deleted"] += store.delete_pks(stale)

        print(f"库内条数     : {store.count()}")
        print(f"新增         : {stats['new']}")
        print(f"更新（覆盖） : {stats['updated']}")
        print(f"删除（孤儿） : {stats['deleted']}")
        print(f"跳过（未变） : {stats['skipped']}")
        print(f"embedding 条数: {api_texts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
