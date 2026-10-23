"""build.py 的两块纯逻辑：样板节过滤 + text_hash。不连库、不调 API。"""
from __future__ import annotations

from src.ingest.build import boilerplate_rule, text_hash
from src.ingest.chunk import SOURCE_SEC, Chunk


def mk(path: list[str], section: str = "", text: str = "Body.") -> Chunk:
    return Chunk(
        pmc_id="PMC1", section=section, path=path, source_type=SOURCE_SEC, text=text
    )


def test_untitled_child_under_supplementary_is_dropped():
    """144 个空标题节里 143 个是这种 —— 末级标题为空，只能靠 path 命中。"""
    c = mk(["Supplementary information"], section="", text="ESM 1(DOCX 1963 kb)")
    assert boilerplate_rule(c) == "supplementary"


def test_supporting_information_variant_is_dropped():
    """sec-type 口径漏掉的 174 个标题变体，靠补词覆盖。"""
    c = mk(["Supporting Information"], section="Supporting Information")
    assert boilerplate_rule(c) == "supplementary"


def test_normal_methods_section_is_kept():
    c = mk(["Methods", "Statistical Analysis"], section="Statistical Analysis")
    assert boilerplate_rule(c) is None


def test_conflict_of_interest_is_dropped():
    c = mk(["Conflict of Interest"], section="Conflict of Interest")
    assert boilerplate_rule(c) == "conflict"


def test_abbreviation_is_dropped_from_retrieval():
    """检索库丢；术语词典另走一遍抽取（plan/09）。"""
    c = mk(["Abbreviations"], section="Abbreviations")
    assert boilerplate_rule(c) == "abbreviation"


def test_text_hash_depends_on_embed_input_not_just_text():
    """同样的正文、不同的标题 → embed_input 不同 → hash 必须不同。
    否则"只改了标题前缀"这类改动会漏掉重向量化。"""
    a = mk(["Methods"], section="Methods", text="We did X.")
    b = mk(["Results"], section="Results", text="We did X.")
    assert a.text == b.text
    assert text_hash(a) != text_hash(b)


def test_text_hash_is_stable_for_identical_input():
    assert text_hash(mk(["Methods"], "Methods", "We did X.")) == text_hash(
        mk(["Methods"], "Methods", "We did X.")
    )
