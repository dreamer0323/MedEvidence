"""`_text_of` 的直连测试 —— 只测这一个零件，不经过整篇解析。

为什么单独开一个文件：`_walk_sections` 还没写，`parse_article` 会先炸在那里，
于是 `test_parse_pmc.py` 里"每条都经过 parse_article"的写法**没法给你即时反馈**。
这个文件让 `_text_of` 一写完就能红转绿。

规则本身（哪条对、为什么）的规格仍在 `test_parse_pmc.py`；这里只是把入口换近一点。
"""
from __future__ import annotations

from lxml import etree

from src.ingest import parse_pmc


def node(xml: str):
    return etree.fromstring(xml)


def test_bibr_xref_dropped_entirely():
    """⚠️ 真实坑：不整个丢掉，正文会黏进一串孤零零的参考文献编号。"""
    p = node('<p>read by two pathologists'
             '<xref rid="b1" ref-type="bibr"><sup>1</sup></xref> in a blinded manner.</p>')
    assert parse_pmc._text_of(p) == "read by two pathologists in a blinded manner."


def test_aff_xref_dropped_entirely():
    p = node('<p>patients<xref rid="aff1" ref-type="aff"><sup>a</sup></xref> from two centers.</p>')
    assert parse_pmc._text_of(p) == "patients from two centers."


def test_sup_becomes_caret():
    """⚠️ 绝不能变成 109 —— 那是数字被改写，本项目的头号禁忌。"""
    p = node("<p>expression was 10<sup>9</sup> fold higher.</p>")
    assert parse_pmc._text_of(p) == "expression was 10^9 fold higher."


def test_sub_becomes_underscore():
    p = node("<p>We used H<sub>2</sub>O<sub>2</sub>.</p>")
    assert parse_pmc._text_of(p) == "We used H_2O_2."


def test_italic_keeps_text_drops_tag():
    p = node("<p>the <italic>t</italic> test.</p>")
    assert parse_pmc._text_of(p) == "the t test."


def test_whitespace_collapsed_to_single_space():
    p = node("<p>a\n   b\t\tc </p>")
    assert parse_pmc._text_of(p) == "a b c"


def test_works_on_container_node_not_just_p():
    """入参可能是 <abstract> / <caption> / <title>，得能递归下去。"""
    a = node("<abstract><p>Background: 10<sup>9</sup> fold higher.</p></abstract>")
    assert parse_pmc._text_of(a) == "Background: 10^9 fold higher."


def test_original_tree_is_not_mutated():
    """原地改会污染后续遍历 —— 必须深拷贝后再动。"""
    p = node('<p>x<xref rid="b1" ref-type="bibr"><sup>1</sup></xref>y</p>')
    parse_pmc._text_of(p)
    assert len(p.findall(".//xref")) == 1
    assert len(p.findall(".//sup")) == 1


def test_block_children_get_newline_separator():
    """⚠️ 真实坑：块级子元素之间不插分隔符会黏成一坨。

    实测 data/raw 20 篇里 5 篇中招，例如
    <sec>Background</sec><sec>Histopathology image analysis …</sec>
    会拼成 "BackgroundHistopathology image analysis …"。
    这类文本喂给 LLM 是幻觉的来源，所以必须切开。
    """
    a = node("<abstract><p>First para.</p><p>Second para.</p></abstract>")
    assert parse_pmc._text_of(a) == "First para.\nSecond para."


def test_structured_abstract_keeps_sec_titles_on_own_lines():
    """结构化摘要（Background/Methods/…）的 <title> 保留，但独占一行。"""
    a = node("<abstract><title>Abstract</title>"
             "<sec><title>Background</title><p>Histopathology is a standard.</p></sec>"
             "</abstract>")
    assert parse_pmc._text_of(a) == "Abstract\nBackground\nHistopathology is a standard."


def test_source_newline_vs_structural_newline_are_distinguishable():
    """原文的换行 → 空格；块级结构给的换行 → 换行。这两个不能混。"""
    a = node("<abstract><p>a\n   b</p><p>c</p></abstract>")
    assert parse_pmc._text_of(a) == "a b\nc"


# ---- 2026-10-21 清洗规则（见 plan/09「建库：逐块 text_hash + 文本清洗」）------

def test_tex_math_source_is_dropped_but_mathml_is_kept():
    """⚠️ 实测：LaTeX 源码长在 tex-math < alternatives < (inline|disp)-formula < p，
    全语料 1883 块 / 7.05% 中招。要丢的是**源码**，同层的 MathML 是正文，不能一起丢。"""
    p = node(
        '<p>where <inline-formula><alternatives>'
        '<tex-math>\\documentclass[12pt]{minimal}</tex-math>'
        '<mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML"><mml:mi>E</mml:mi></mml:math>'
        '</alternatives></inline-formula> is energy.</p>'
    )
    out = parse_pmc._text_of(p)
    assert "documentclass" not in out
    assert out == "where E is energy."


def test_tex_math_without_mathml_leaves_no_gap():
    """没有 MathML 备选时，丢掉 tex-math 后剩两个空格 —— 必须被压成一个。"""
    p = node(
        '<p>value <inline-formula><alternatives>'
        '<tex-math>\\begin{document}x^2\\end{document}</tex-math>'
        '</alternatives></inline-formula> here.</p>'
    )
    assert parse_pmc._text_of(p) == "value here."


def test_empty_brackets_left_by_xref_removal_are_cleaned():
    """⚠️ 实测：4594 块 / 17.20% 有 `[]`。原文 `3D CRANI [<xref>1</xref>]`
    删掉 xref 后剩 `3D CRANI []` —— 方括号是原文字符，编号在 xref 里。"""
    p = node('<p>3D CRANI [<xref rid="b1" ref-type="bibr">1</xref>] was used.</p>')
    assert parse_pmc._text_of(p) == "3D CRANI was used."


def test_table_marker_brackets_are_not_touched():
    """`[TABLE ...]` 是 _table_text 自己插的标记，不是残留，不能一起清掉。"""
    p = node("<p>[TABLE Table 1: Baseline] cell | value</p>")
    assert parse_pmc._text_of(p) == "[TABLE Table 1: Baseline] cell | value"


def test_literal_empty_brackets_in_source_are_kept():
    """⚠️ 原文自带的 `[]` 不是 xref 残留（实测 9 处，如 R 代码 `Table[]`）。
    全局正则会把它们一起删掉 —— 只许删「xref 脱落处」的那一对。"""
    p = node("<p>inp_df &lt;- Table[] %&gt;% select(x)</p>")
    assert "Table[]" in parse_pmc._text_of(p)
