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
from src.vectorstore import ChunkRow, MilvusStore

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


def text_hash(chunk: Chunk) -> str:
    """哈希对象 = 喂给 embedding 的字符串，不是入库的 text。

    ⚠️ 为什么是 embed_input：向量由它算出来，它没变，向量就还能用。
    用 text 当哈希会漏掉"正文没变、标题变了"的情况（标题进了 embed_input）。
    """

    embed_raw_text = chunk.embed_input          # @property，不是方法：别加括号
    embed_str = embed_raw_text.encode("utf-8")   # 采用 utf-8 编码，与 embedding 一致

    return hashlib.sha256(embed_str).hexdigest()


def to_row(chunk: Chunk, vector: list[float]) -> ChunkRow:
    """Chunk → 存储层契约。**全项目唯一的结构转换点**（见 docs/data-model.md）。"""
    return ChunkRow(
        vector=vector,
        text=chunk.text,
        pmc_id=chunk.pmc_id,
        section=chunk.section,
        chunk_index=chunk.chunk_index,
        path=" / ".join(chunk.path),   # Milvus 列是 VARCHAR，列表要拼成字符串
        source_type=chunk.source_type,
        text_hash=text_hash(chunk),
    )


def collect_chunks(raw_dir: Path, limit: int | None, stats: Counter | None = None):
    """按文档序产出要入库的 chunk。过滤在这里，统计写进调用方传进来的 Counter。

    ⚠️ 统计必须由调用方持有：生成器跑完就没了，内部的 Counter 外面拿不到。
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

        stats["articles"] += 1
        for chunk in build_chunks(article):
            rule = boilerplate_rule(chunk)
            if rule:
                stats[f"dropped:{rule}"] += 1
                continue
            stats["chunks"] += 1
            yield chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedEvidence 建库")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（试水用）")
    parser.add_argument("--drop", action="store_true", help="重建集合（数据全丢）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不连库、不调 API")
    args = parser.parse_args(argv)

    setup_logging()
    logger.info("MILVUS_URI = %s", MILVUS_URI)

    stats = Counter()
    chunks = collect_chunks(Path(RAW_DIR), args.limit, stats)

    if args.dry_run:
        for _ in chunks:
            pass
        print(f"文章 {stats['articles']}（解析失败 {stats['parse_failed']}）")
        print(f"入库块 {stats['chunks']}")
        for key in sorted(k for k in stats if k.startswith("dropped:")):
            print(f"  过滤 {key[8:]:16s} {stats[key]}")
        return 0

    client = EmbeddingClient()
    api_texts = 0                 # 实际送去 embedding 的条数（幂等验收用）

    with MilvusStore() as store:
        store.ensure_collection(drop=args.drop)

        batch: list[Chunk] = []

        def flush() -> None:
            nonlocal api_texts
            if not batch:
                return
            hashes = [text_hash(c) for c in batch]
            existing = store.existing_hashes(hashes)
            todo = [c for c, h in zip(batch, hashes) if h not in existing]
            stats["skipped"] += len(batch) - len(todo)
            if todo:
                vectors = client.embed(
                    [c.embed_input for c in todo], desc=f"第 {stats['flushed'] + 1} 批"
                )
                api_texts += len(todo)
                rows = [to_row(c, v) for c, v in zip(todo, vectors)]
                stats["inserted"] += store.insert(rows)
            stats["flushed"] += 1
            batch.clear()

        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= BATCH_CHUNKS:
                flush()
        flush()

        print(f"库内条数     : {store.count()}")
        print(f"本次插入     : {stats['inserted']}")
        print(f"已存在跳过   : {stats['skipped']}")
        print(f"embedding 条数: {api_texts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
