"""解析层测试 —— 全部离线，用一份仿真的 JATS 片段当输入。

这层的 bug 是"错得看不出来"：少一个字段不报错，只是检索不到；
把参考文献编号黏进正文也不报错，只是句子读着别扭 —— 而大模型会拿着
读着别扭的句子去补全（幻觉的来源）。所以规格必须写死在测试里。

**本文件是规格，不是实现。** `src/ingest/parse_pmc.py` 由你写。
"""
from __future__ import annotations

import pytest
from lxml import etree

from src.ingest import parse_pmc

# --------------------------------------------------------------- 仿真输入
# 结构照着真实 PMC 全文搭的（嵌套 sec / 表格 / bibr+aff 的 xref / sup / italic / ref-list）

ARTICLE = """<article article-type="research-article" xml:lang="en">
  <front>
    <journal-meta><journal-title>Brain Pathology</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmcaid">13660408</article-id>
      <article-id pub-id-type="pmcid">PMC13660408</article-id>
      <article-id pub-id-type="publisher-id">OTO270312</article-id>
      <title-group><article-title>Digital pathology of <italic>BRAF</italic> in thyroid cancer</article-title></title-group>
      <pub-date pub-type="epub"><year>2025</year></pub-date>
      <abstract><p>Background: expression was 10<sup>9</sup> fold higher.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Methods</title>
      <p>All sections were read by two independent pathologists<xref rid="b1" ref-type="bibr"><sup>1</sup></xref> in a blinded manner.</p>
      <sec>
        <title>Statistical Analysis</title>
        <p>We used H<sub>2</sub>O<sub>2</sub> and the <italic>t</italic> test.</p>
      </sec>
      <sec>
        <title>Patients</title>
        <p>We enrolled patients<xref rid="aff1" ref-type="aff"><sup>a</sup></xref> from two centers.</p>
        <table-wrap>
          <label>Table 1</label>
          <caption><p>Baseline characteristics</p></caption>
          <table>
            <thead><tr><th>Variable</th><th>Group A</th><th>Group B</th></tr></thead>
            <tbody>
              <tr><td>Median age</td><td>62</td><td>58</td></tr>
              <tr><td>HER2 positive</td><td>23%</td><td>31%</td></tr>
            </tbody>
          </table>
        </table-wrap>
      </sec>
    </sec>
    <sec>
      <title>Results</title>
      <p>The positive rate was 23%.</p>
    </sec>
    <sec>
      <title>   </title>
      <p></p>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="b1"><element-citation><person-group><surname>Ogura</surname>
      <given-names>JH</given-names></person-group><article-title>Laryngectomy</article-title>
      </element-citation></ref>
    </ref-list>
  </back>
</article>"""


def as_bytes(xml: str = ARTICLE) -> bytes:
    return xml.encode("utf-8")


def sections_by_path(article) -> dict[tuple[str, ...], str]:
    return {tuple(s.path): s.text for s in article.sections}


# --------------------------------------------------------------- 根形状容忍


def test_parses_a_bare_article_root():
    """模块抓下来的文件，根元素就是 <article>（实测）。"""
    art = parse_pmc.parse_article(as_bytes())
    assert art.pmc_id == "PMC13660408"


def test_tolerates_a_pmc_articleset_wrapper():
    """手工抓的那批根是 <pmc-articleset>，也要能吃 —— data/raw 真混过两种形状。"""
    wrapped = f'<?xml version="1.0" ?><pmc-articleset>{ARTICLE}</pmc-articleset>'
    art = parse_pmc.parse_article(as_bytes(wrapped))
    assert art.pmc_id == "PMC13660408"


def test_rejects_when_not_exactly_one_article():
    """0 篇（非 OA 全文）或多篇都要当场报错，不许静默返回空文章。"""
    with pytest.raises(ValueError):
        parse_pmc.parse_article(b'<?xml version="1.0" ?><pmc-articleset></pmc-articleset>')


# --------------------------------------------------------------- 元数据


def test_pmc_id_comes_from_pmcid_not_publisher_id():
    """⚠️ 最容易抓错的一处：<article id="..."> 是出版社稿号，不是 PMC ID。"""
    art = parse_pmc.parse_article(as_bytes())
    assert art.pmc_id == "PMC13660408"
    assert "OTO270312" not in art.pmc_id


def test_metadata_fields():
    art = parse_pmc.parse_article(as_bytes())
    assert art.journal == "Brain Pathology"
    assert art.year == 2025
    assert "BRAF" in art.title          # 斜体只脱标签，文字要留


# --------------------------------------------------------------- 章节结构


def test_only_leaf_sections_plus_parents_own_text():
    """只留叶子节；父节自己的正文够长（≥50 字）才单独成块。

    实测依据：父节 Methods 标 6435 字，子节加起来 6428 —— 父子都入库 = 同一段存两遍。
    """
    art = parse_pmc.parse_article(as_bytes())
    got = sections_by_path(art)

    # 叶子节
    assert ("Methods", "Statistical Analysis") in got
    assert ("Methods", "Patients") in got
    assert ("Results",) in got
    # 父节 Methods 自己那段有 70+ 字 → 单独成块，path 只到它自己
    assert ("Methods",) in got
    # 父节不该把子节内容吞进来
    assert "H2O2" not in got[("Methods",)] and "H_2O_2" not in got[("Methods",)]


def test_abstract_is_its_own_section():
    art = parse_pmc.parse_article(as_bytes())
    got = sections_by_path(art)
    assert ("Abstract",) in got
    assert "10^9" in got[("Abstract",)]


def test_empty_section_is_dropped():
    art = parse_pmc.parse_article(as_bytes())
    assert () not in sections_by_path(art)
    assert all(s.text.strip() for s in art.sections)


def test_section_level_reflects_nesting():
    art = parse_pmc.parse_article(as_bytes())
    lv = {tuple(s.path): s.level for s in art.sections}
    assert lv[("Results",)] == 1
    assert lv[("Methods", "Patients")] == 2


# --------------------------------------------------------------- 内联标签


def test_reference_xref_is_dropped_entirely():
    """⚠️ 真实坑：bibr xref 若不整个丢掉，正文会黏进一串孤零零的数字。"""
    art = parse_pmc.parse_article(as_bytes())
    text = sections_by_path(art)[("Methods",)]
    assert "blinded manner" in text
    assert "pathologists1" not in text
    assert "pathologists 1" not in text


def test_affiliation_xref_is_dropped_entirely():
    art = parse_pmc.parse_article(as_bytes())
    text = sections_by_path(art)[("Methods", "Patients")]
    assert "from two centers" in text
    assert "patientsa" not in text


def test_sup_and_sub_become_caret_and_underscore():
    """10<sup>9</sup> → 10^9（**绝不能变成 109**，那是数字被改写）。"""
    art = parse_pmc.parse_article(as_bytes())
    text = sections_by_path(art)[("Methods", "Statistical Analysis")]
    assert "H_2O_2" in text
    assert "H2O2" not in text


def test_italic_keeps_text_drops_tag():
    art = parse_pmc.parse_article(as_bytes())
    text = sections_by_path(art)[("Methods", "Statistical Analysis")]
    assert "t test" in text          # 斜体标签没了，字母 t 还在


# --------------------------------------------------------------- 表格


def test_table_is_kept_as_marked_block():
    """表格整块保留、就地插在它所在章节的文本流里，行内单元格用 | 连接。"""
    art = parse_pmc.parse_article(as_bytes())
    text = sections_by_path(art)[("Methods", "Patients")]

    assert "[TABLE Table 1: Baseline characteristics]" in text
    assert "[/TABLE]" in text
    assert "Variable | Group A | Group B" in text
    assert "Median age | 62 | 58" in text      # 数字必须和它的标签在同一行
    # 表格必须整块出现在**同一个** section 的文本里，不许被拆到别处
    assert text.count("[TABLE") == 1


def test_ref_list_never_reaches_sections():
    """参考文献列表整体剔除：实测 surname/given-names/string-name 出现 47+ 次。"""
    art = parse_pmc.parse_article(as_bytes())
    joined = " ".join(s.text for s in art.sections)
    assert "Ogura" not in joined
    assert "Laryngectomy" not in joined
    assert not any("Reference" in s.title for s in art.sections)
