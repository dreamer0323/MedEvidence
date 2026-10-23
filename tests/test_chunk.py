"""chunk.py 的验收测试：只验不变量，不验实现细节。

为什么不用"chunk 条数对不对"当验收：条数对不上说明有问题，但条数**对得上**
不代表没问题 —— 一个块被拆成两块、另一个块被丢掉，条数可能刚好抵消。

真正的验收是两条：
  1. 守恒式：某个 Section 的所有 chunk 拼接 == 原 Section 文本（逐字）
  2. 契约：喂 embedding 的字符串有标题，入库的 text 没有
"""
import pytest

from src.ingest.chunk import (
    CHUNK_HARD_LIMIT,
    SOURCE_ABSTRACT,
    SOURCE_SEC,
    Chunk,
    build_chunks,
    split_section,
)
from src.ingest.parse_pmc import ABSTRACT_TITLE, Article, Section


def make_article(*sections) -> Article:
    return Article(
        pmc_id="PMC999",
        title="demo",
        journal="J",
        year=2024,
        abstract="",
        sections=list(sections),
    )


# ---------------------------------------------------------------- 基本形状

def test_short_section_is_one_untouched_chunk():
    text = "We stained the slides. Then we counted cells."
    art = make_article(Section(title="Methods", text=text, level=1, path=["Methods"]))

    chunks = build_chunks(art)

    assert len(chunks) == 1
    assert chunks[0].text == text          # 不超限就不许改写
    assert chunks[0].source_type == SOURCE_SEC
    assert chunks[0].path == ["Methods"]


def test_abstract_is_marked_by_source_type():
    art = make_article(
        Section(title=ABSTRACT_TITLE, text="Background. Methods.", level=1, path=[ABSTRACT_TITLE])
    )
    assert build_chunks(art)[0].source_type == SOURCE_ABSTRACT


def test_chunk_index_is_document_order_within_article():
    long_text = "Sentence about tissue. " * 500
    art = make_article(
        Section(title="Results", text=long_text, level=1, path=["Results"]),
        Section(title="Discussion", text="Short.", level=1, path=["Discussion"]),
    )
    chunks = build_chunks(art)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].section == "Discussion"


def test_section_metadata_is_carried_to_every_piece():
    long_text = "Sentence about tissue. " * 500
    art = make_article(
        Section(title="Results", text=long_text, level=2, path=["Methods", "Results"])
    )
    for c in build_chunks(art):
        assert c.pmc_id == "PMC999"
        assert c.section == "Results"
        assert c.path == ["Methods", "Results"]


# ---------------------------------------------------------------- 契约

def test_embed_input_has_title_but_text_stays_pure():
    c = Chunk(
        pmc_id="PMC1",
        section="EntropyMasker",
        path=["Methods", "EntropyMasker"],
        source_type=SOURCE_SEC,
        text="We developed a thresholding method.",
    )
    assert c.embed_input.startswith("EntropyMasker")
    assert c.embed_input.endswith(c.text)
    assert "EntropyMasker" not in c.text


def test_empty_title_adds_no_prefix():
    c = Chunk(pmc_id="PMC1", section="", path=[], source_type=SOURCE_SEC, text="Body.")
    assert c.embed_input == c.text


# ---------------------------------------------------------------- 超限降级

def test_paragraph_packing_prefers_paragraph_boundaries():
    """三个 ~5000 字的段落：两两装不下，应在段落边界断开，不触发硬切。"""
    p = "Cells were counted under the microscope. " * 115   # ~4800 字，无换行
    text = "\n".join([p, p, p])
    pieces = split_section(text)
    assert [piece.strip() for piece in pieces] == [p.strip()] * 3


def test_oversize_split_conserves_every_character():
    text = "The value was 3.6.5 and the cells grew. " * 400   # ~16000 字，单段
    pieces = split_section(text)

    assert len(pieces) > 1
    assert "".join(pieces) == text                              # 丢字/重复/错位一起排除
    assert all(len(piece) <= CHUNK_HARD_LIMIT for piece in pieces)


def test_paragraph_without_any_period_still_converges():
    """无句号也要收敛：第 2 层空格 / 第 3 层硬切兜底，不许死循环或超限。"""
    text = "x" * (CHUNK_HARD_LIMIT * 2 + 500)
    pieces = split_section(text)

    assert "".join(pieces) == text
    assert all(len(piece) <= CHUNK_HARD_LIMIT for piece in pieces)


def test_no_piece_crosses_the_hard_limit_on_mixed_text():
    text = "\n".join(
        ["Short intro."] + ["A fairly long paragraph about histopathology. " * 200] * 2
    )
    pieces = split_section(text)

    assert "".join(pieces) == text
    assert all(len(piece) <= CHUNK_HARD_LIMIT for piece in pieces)
    assert all(piece for piece in pieces)                        # 不产空块


@pytest.mark.parametrize("size", [0, 1, CHUNK_HARD_LIMIT - 1, CHUNK_HARD_LIMIT])
def test_text_at_or_below_limit_is_untouched(size):
    text = "a" * size
    assert split_section(text) == [text]
