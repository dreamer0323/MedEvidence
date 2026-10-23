"""Section → Chunk：把解析出来的结构落成真正要入库的块。

策略已在 plan/09-决策记录.md「分块策略：结构化分块 + 硬上限兜底」定死，
这里只实现，不重新做选择：

  1. 一个 Section 默认就是一块（结构切分，overlap = 0）
  2. 只有超过硬上限才降级：按段落（"\\n"）贪心聚合 → 单段仍超限 → 硬切
  3. embed_input = 标题 + 正文；入库的 text 是纯正文（方案 C）

**8192 这两个字有歧义，别混说**（2026-10-21 更正，见 09）：
  Milvus 的 VARCHAR(8192) 数的是**字符**，BGE-M3 的 8192 数的是 **token**。
  本语料 CJK 占比 0.006%，token:char ≈ 0.25 → 8192 字符 ≈ 2048 token，
  离模型上限还有四倍。**这条墙只有 Milvus 在逼。**

这里**不做**"哪些块不该入库"的过滤 —— 那是建库步骤的决定（04 #13 仍未决）。
本模块只保证三件事：不丢字、不重复、每块 <= CHUNK_HARD_LIMIT。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.ingest.parse_pmc import ABSTRACT_TITLE, Article, Section

# 防溢出的墙，不是切分目标（命名见 09：不叫 CHUNK_MAX_CHARS）
CHUNK_HARD_LIMIT = 8192

# 硬切时允许的句末最早位置：再往前切，尾巴会被反复再切，碎块变多
SENTENCE_FLOOR = CHUNK_HARD_LIMIT // 2

SOURCE_SEC = "sec"
SOURCE_ABSTRACT = "abstract"


@dataclass
class Chunk:
    """入库的最小单位。

    `text` 与 `embed_input` 是**两根不同的字符串**，这不是冗余：
    入库的 text 必须和原文对得上（引用要能回指），喂 embedding 的字符串
    可以带标题前缀补上下文（09 的方案 C）。只存一个，另一个就会漂。
    """

    pmc_id: str                     # 唯一标识
    section: str                    # 末级标题；完整层级在 path
    path: list[str]                 # 节路径，从根到当前，如 ['Methods', 'Patients']
    source_type: str                # SOURCE_SEC / SOURCE_ABSTRACT ，来源类型
    text: str                       # 纯正文 —— 只有它入库
    chunk_index: int = 0            # 同一篇内从 0 起，文档排序

    @property
    def embed_input(self) -> str:
        """喂 embedding 的字符串：标题 + 正文。标题为空时不加前缀。"""
        if not self.section:
            return self.text
        return f"{self.section}\n{self.text}"


# ---- 对外接口 --------------------------------------------------------------

def build_chunks(article: Article) -> list[Chunk]:
    """一篇 Article → 按文档序排好的 Chunk 列表。

    ⚠️ 不在这里过滤任何块：`chunk_index` 是"同一篇内第几块"的溯源锚点，
    在这里删块会让锚点和 Milvus 里存的对不上。过滤留给建库步骤。
    """
    chunks: list[Chunk] = []

    for sec in article.sections:
        for text in split_section(sec.text):
            chunks.append(
                Chunk(
                    pmc_id=article.pmc_id,
                    section=sec.title,
                    path=list(sec.path),
                    source_type=_source_type(sec),
                    text=text,
                )
            )

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def split_section(text: str) -> list[str]:
    """一段 Section 正文 → 每块 <= CHUNK_HARD_LIMIT 的文本列表。

    未超限：**原样返回一块**，不做任何改写 —— 改写会让引用对不上原文。
    超限：先按段落边界贪心聚合（段落边界来自 `_text_of` 插入的 "\\n"），
          单段仍超限才交给 `_split_oversize` 硬切。

    拼接后必须与入参逐字相等 —— 这条守恒式由 tests/test_chunk.py 断言。
    实现上把每个段落连同它前面的 "\\n" 当成一个不可拆的单元（`unit`），
    这样聚合时**根本不会出现"分隔符归谁"的问题**，守恒式由构造保证。
    """
    if len(text) <= CHUNK_HARD_LIMIT:
        return [text]

    pieces: list[str] = []
    buf = ""

    for i, para in enumerate(text.split("\n")):
        unit = para if i == 0 else "\n" + para

        # 单段（含分隔符）自己就超限：聚合救不了它，只能硬切 
        if len(unit) > CHUNK_HARD_LIMIT:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_split_oversize(unit))
            continue

        if not buf:
            buf = unit
        elif len(buf) + len(unit) <= CHUNK_HARD_LIMIT:
            buf += unit
        else:
            pieces.append(buf)
            buf = unit

    if buf:
        pieces.append(buf)
    return pieces


def _split_oversize(text: str) -> list[str]:
    """最后一级兜底：单个段落自己就超过 CHUNK_HARD_LIMIT 时的硬切。

    TODO(human)：约 5-10 行，把 text 切成若干块，每块 <= CHUNK_HARD_LIMIT。

    规则已定（2026-10-21，见 09），三层从便宜到兜底：
      1. 在 `[SENTENCE_FLOOR, CHUNK_HARD_LIMIT]` 窗口里**从后往前找句末**：
         "." 后面跟空格 + 大写字母（或 "." 后接引号）→ 找到就切在句号之后
      2. 窗口里没有句末 → 取窗口内最后一个空格
      3. 仍没有 → 直接切在 CHUNK_HARD_LIMIT

    数据（全语料实测，真正走到这里的只有 6 个块）：最近句号 7737–8178
    → 第 1 层吃掉 6/6，第 2、3 层当前触发 0 次，买的是不变式不是效果。

    硬性不变量（测试会验）：
      · "".join(返回值) == text —— 逐字相等，不许丢字、不许重复
      · 每块 <= CHUNK_HARD_LIMIT，且无句号的 2 万字符段落也要收敛
    """
    pieces: list[str] = [] # 返回值
    rest = text # 剩余文本
    lo,hi = SENTENCE_FLOOR, CHUNK_HARD_LIMIT

    while len(rest)>hi:
        #───── 阶段 A：搜索断点─────
        cut = 0

        for i in range(hi-1,lo-2,-1):
            if rest[i] != ".":
                continue
            a = rest[i+1:i+2] 
            b = rest[i+2:i+3]

            if (a==" " and b.isupper()) or a in ('"', "'"):
                cut = i+1
                break   

        if not cut:
            for j in range(hi-1,lo-2,-1):
                if rest[j] == " ":
                    cut = j+1
                    break

        if not cut:
            cut = hi 

        #───── 阶段 B：切块─────
        pieces.append(rest[:cut]) 
        rest = rest[cut:]
        
    #───── 阶段 C：处理剩余文本─────
    if rest:
        pieces.append(rest)
    return pieces

# ---- 内部零件 --------------------------------------------------------------

def _source_type(sec: Section) -> str:
    """Abstract 在 parse 阶段也是一个 Section，靠 path 把它认回来。"""
    if sec.title == ABSTRACT_TITLE and sec.path == [ABSTRACT_TITLE]:
        return SOURCE_ABSTRACT
    return SOURCE_SEC


if __name__ == "__main__":
    # 自检：python -m src.ingest.chunk [raw目录] [篇数]
    import logging
    import sys
    from pathlib import Path

    from src.config import RAW_DIR, setup_logging
    from src.ingest.parse_pmc import parse_article

    setup_logging()
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(RAW_DIR)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    total = 0
    for f in sorted(raw.glob("*.xml"))[:limit]:
        art = parse_article(f.read_bytes())
        chunks = build_chunks(art)
        total += len(chunks)
        print(
            f"{f.name:20s} {art.pmc_id:12s} "
            f"sections={len(art.sections):3d} chunks={len(chunks):3d}"
        )
    print(f"\n前 {limit} 篇共 {total} 块")
