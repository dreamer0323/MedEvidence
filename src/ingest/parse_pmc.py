"""PMC 全文 XML → 结构化的 Article / Section。

只做解析，不做分块（分块在 chunk.py）、不做向量化。

**规格在 tests/test_parse_pmc.py —— 那个文件是契约，这个文件是实现。**
本文件里 `_text_of`（本轮你写）和 `_walk_sections`（下一轮你写）是"要用自己的判断填"的地方。

三处真实坑（测试里都有对应断言，别绕过它们）：
  1. lxml 的 itertext() 是**递归**的 → 用错地方父节会吞掉子节的全部文本
  2. <xref ref-type="bibr"> 不整个丢掉，正文会黏进一串孤零零的参考文献编号
  3. <sup> 直接取文本会把 10^9 变成 109 —— 数字被改写，这是本项目的头号禁忌
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree
import copy


from src.config import RAW_DIR, setup_logging

logger = logging.getLogger(__name__)

# ---- 规格常量（测试里断言的就是这些）--------------------------------------
MIN_OWN_TEXT_CHARS = 50      # 父节自己的正文短于这个就不单独成块
ABSTRACT_TITLE = "Abstract"
TABLE_OPEN = "[TABLE {}]"    # {} = "Table 1: Baseline characteristics"
TABLE_CLOSE = "[/TABLE]"
CELL_SEP = " | "
BIBR_AFF_REF_TYPES = ("bibr", "aff")   # 这两种 xref 整个元素丢掉

# 块级标签：在文本流里这几个必须独占一行，不能和相邻文字黏在一起。
# 依据（实测 data/raw 20 篇）：abstract 里是 <p>/<sec>/<title> 并列时，
# 不插分隔符会拼成 "BackgroundHistopathology image analysis..." —— 20 篇里 5 篇中招。
BLOCK_TAGS = frozenset({
    "p", "sec", "title", "fig", "caption", "table-wrap",
    "list", "list-item", "disp-formula", "boxed-text", "graphic",
})

# 原文里的连续空白（含换行/制表符）压成一个空格。
# 注意用 sub 而不是 split()+join()：split() 会**去掉**首尾空格，
# 而 "the " / " test." 这种首尾空格正是内联标签拼接时的粘合剂
# （`<p>the <italic>t</italic> test.</p>` 去掉就变成 "thettest."）。
_WS_RUN = re.compile(r"\s+")


# ---- 数据结构 --------------------------------------------------------------

@dataclass
class Section:
    title: str                      # 节标题，如 "Methods"
    text: str                       # 节正文，包含所有段落
    level: int                      # 1 = 顶层 sec，2 = 嵌套
    path: list[str] = field(default_factory=list)   # 节路径，从根到当前，如 ['Methods', 'Patients']


@dataclass
class Article:
    pmc_id: str                     # 必须是 'PMC13660408' 这种，不是出版社稿号
    title: str
    journal: str                    # 期刊名称，如 "Nature"
    year: int | None                # 发表年份，如 2023
    abstract: str                   # 与 sections 里的 Abstract 节同源，便于直接取用
    sections: list[Section] = field(default_factory=list) # 所有 sec 节，按层级排序


# ---- 内部零件 --------------------------------------------------------------

def _one_article(root) -> etree._Element:
    """从任意根元素里取出**唯一**那篇 <article>。

    容忍两种形状：根就是 <article>，或者根是 <pmc-articleset>。
    ⚠️ 篇数 != 1 必须抛 ValueError —— 不许静默返回空文章（0 篇 = 非 OA 全文）。
    """
    arts = [root] if root.tag == "article" else root.findall(".//article")
    if len(arts) != 1:
        raise ValueError(
            f"期望恰好 1 篇 <article>，实际 {len(arts)} 篇（0 篇 = 非 OA 全文 / id 无效）"
        )
    return arts[0]


def _text_of(node) -> str:
    """任意节点 → 一行干净文本。**本文件的核心，也是最容易写错的地方。**

    入参不一定是 <p>：也可能是 <title> / <abstract> / <caption> / <td>，
    所以要能往下递归，不能只看 node.text。

    规则（逐条都有测试断言，见 tests/test_text_of.py）：
      · <xref ref-type="bibr"> / ref-type="aff" → **整个元素丢掉**（连里面的 <sup> 一起）
      · <sup> 的文本加 "^" 前缀、<sub> 加 "_" 前缀 → 10^9 / H_2O_2
      · 其它内联标签（<italic> 等）→ 丢标签、留文字
      · 块级子元素（<p>/<sec>/<title>/<fig> …）各自独占一行 → 段落间是 "\n"
      · <tex-math> 里的 LaTeX 源码 → 整个元素丢掉（同层的 MathML 保留）
      · 被方括号包住的 xref（`[<xref>1</xref>]`）→ 括号对一起删；
        原文自带的 `[]` 与 `[TABLE ...]` 不动
      · 连续空白压成一个空格（只压原文里的），首尾去空、空行丢弃

    ⚠️ 不要直接对原树动手 —— 原地改会污染后续遍历，先深拷贝。
    可用工具：copy.deepcopy、element.getparent().remove(element)、
    etree.strip_tags(element, "*") 批量脱标签、element.insert(0, ...) 插前缀。
    """
    # 1. 先深拷贝，绝不污染原树
    node = copy.deepcopy(node)

    # 2. 剔除 ref-type="bibr" / "aff" 的整个 xref 元素
    #    注意：xpath 找到的是后代节点，一定要通过 getparent 移除
    #    ⚠️ tail 陷阱：xref 后面那截正文挂在 xref.tail 上（不在父节点的 .text 里），
    #       直接 remove 会把那半句话一起带走 —— "patients from two centers." 变 "patients"。
    #       删之前先把 tail 交给"前一个兄弟"（没有前兄弟就交给父节点的 .text）。
    #       xpath 按文档序返回，连续多个 xref 也安全：先删的把 tail 交给前兄弟，
    #       后删的再接到同一个前兄弟后面，顺序不变。
    xrefs = node.xpath('.//xref[@ref-type="bibr" or @ref-type="aff"]')
    for xref in xrefs:
        parent = xref.getparent()
        prev = xref.getprevious()
        prev_text = (prev.tail if prev is not None else parent.text) or ""
        tail = xref.tail or ""

        # 若 xref 被方括号包着（`[<xref>1</xref>]`），连这一对括号一起删。
        # ⚠️ 只删 xref 脱落处的这一对：原文自带的 `[]` 有 9 处（R 代码 `Table[]`、
        #    原文举例 `Figure []`），全局正则会把它们误伤（2026-10-21 实测后修正）。
        #    多引用 `[<xref>1</xref>, <xref>2</xref>]` 条件不成立 —— 括号保留。
        if prev_text.rstrip().endswith("[") and tail.lstrip().startswith("]"):
            prev_text = prev_text.rstrip()[:-1]
            tail = tail.lstrip()[1:]

        if prev is not None:
            prev.tail = prev_text + tail
        else:
            parent.text = prev_text + tail
        parent.remove(xref)

    # 3. 处理上下标前缀
    #    直接修改 .text 属性可以安全地在首位插入文本节点
    for el in node.xpath('.//sup'):
        el.text = '^' + (el.text or '')
    for el in node.xpath('.//sub'):
        el.text = '_' + (el.text or '')

    # 4. 归一化**原文里**的连续空白 → 单个空格
    #    必须在第 5 步插结构换行**之前**做，否则原文的换行和第 5 步插的换行
    #    混在一起，第 7 步就分不清哪个该压、哪个该留。
    #    这里不 strip，原因见 _WS_RUN 的注释。
    for el in node.iter():
        if el.text:
            el.text = _WS_RUN.sub(" ", el.text)
        if el.tail:
            el.tail = _WS_RUN.sub(" ", el.tail)

    # 5. 块级子元素后面插结构换行（必须在 strip_tags **之前**插）
    #    插 tail 而不是 text：strip_tags 会把子孙的 text 和 tail 一起并进 node.text，
    #    所以插在谁身上最终都会落到那一个字符串里。
    #    用 iterdescendants 而不是 iter：node 自己若是 <p>，不该凭空多一个尾部空行。
    for el in node.iterdescendants():
        if el.tag in BLOCK_TAGS:
            el.tail = (el.tail or "") + "\n"

    # 5.5 丢弃 <tex-math> 里的 LaTeX 源码（第 5–6 步之间，脱标签之前）。
    #     实测位置：tex-math < alternatives < (inline|disp)-formula < p，
    #     全语料 1883 块 / 7.05% 中招（见 plan/09）。同层的 MathML 是正文，必须保留。
    #     ⚠️ 和删 xref 同一个 tail 陷阱：tex-math 后面的文字挂在它的 .tail 上。
    for tex in node.xpath(".//tex-math"):
        parent = tex.getparent()
        prev = tex.getprevious()
        if tex.tail:
            if prev is not None:
                prev.tail = (prev.tail or "") + tex.tail
            else:
                parent.text = (parent.text or "") + tex.tail
        parent.remove(tex)

    # 6. 批量剥离所有剩余的内联标签（如 italic），只保留文字
    etree.strip_tags(node, "*")

    # 7. 切行、逐行压空白 + 去首尾、丢空行
    #    逐行再压一次是因为 strip_tags 拼接时可能撞出连续空格
    #    （父节点尾部空格 + 子节点头部空格 → 两个空格）
    #    方括号对已在第 2 步随 xref 一起删掉，删完留下的双空格在这里被压掉。
    return "\n".join(
        " ".join(line.split())
        for line in (node.text or "").split("\n")
        if line.strip()
    )


def _own_text(sec) -> str:
    """只取这一节**自己那层**的文本，不含子节。

    ⚠️ `sec.itertext()` 会把子节的文本一起抓出来（实测：父节 Methods 标 6435 字，
    三个子节加起来 6428 字 → 父节自己的正文只有 7 字）。
    """
    parts = []
    for child in sec:
        if child.tag == "p":
            parts.append(_text_of(child))
        elif child.tag == "table-wrap":
            parts.append(_table_text(child))
    return "\n".join(p for p in parts if p.strip())


def _table_text(table_wrap) -> str:
    """<table-wrap> → 带标记的文本块。格式在测试里写死了：

        [TABLE Table 1: Baseline characteristics]
        Variable | Group A | Group B
        Median age | 62 | 58
        [/TABLE]

    要点：行内单元格用 CELL_SEP 连接、行间换行；**空单元格保留占位**（否则列会错位）；
    单元格内部的 <sup>/<sub> 同样按 _text_of 的规则处理。
    """
    label = (table_wrap.findtext("label") or "Table").strip()
    caption = table_wrap.find("caption")
    caption_text = _text_of(caption) if caption is not None else ""
    head = TABLE_OPEN.format(f"{label}: {caption_text}" if caption_text else label)

    lines = [head]
    for tr in table_wrap.iter("tr"):
        cells = [_text_of(cell) for cell in tr if cell.tag in ("td", "th")]
        lines.append(CELL_SEP.join(cells))
    lines.append(TABLE_CLOSE)
    return "\n".join(lines)


def _walk_sections(parent, path: list[str], level: int, out: list[Section]) -> None:
    """深度优先遍历 <sec>，按规格产出 Section。

    参数语义：parent 是容器（<body> 或某个 <sec>），path 是**容器自己的**路径，
    level 是 parent 的**直接子 <sec>** 的层级（所以 <body> 调进来时 level=1）。

    规则：
      · 这一节自己的正文够长（≥ MIN_OWN_TEXT_CHARS）→ 单独成一个 Section，path 到它自己
      · 短于阈值 → **不新造块**，并入递归后第一个子块的开头。
        两种做法都不丢信息，但独立成块会造出一批十几字的碎片块 ——
        它们不承载可回答的内容，却会以噪声形式挤进 top-k，占掉本该给子块的名额。
      · 短正文且**无子块**（叶子节 / 子节全空）→ 只能自己成块，否则这段字就丢了
      · 叶子节 → 一个 Section；父节的标题只进 path，**不进**子节的 text
      · 空节丢弃
      · <table-wrap> 已经由 _own_text 就地拼进文本流，这里不用再管
    """
    for sec in parent.findall("sec"):
        title_node = sec.find("title")
        title = (_text_of(title_node) if title_node is not None else "").strip()
        here = path + [title] if title else list(path)
        own = _own_text(sec)

        # 够长 → 自己先出块（文档序：父块排在子块之前），再往下递归
        if len(own) >= MIN_OWN_TEXT_CHARS:
            out.append(Section(title=title, text=own, level=level, path=here))
            _walk_sections(sec, here, level + 1, out)
            continue

        # 短正文：先递归，再决定它去哪。
        # mark 是递归前的位置 —— 递归后 out[mark] 就是文档序上第一个子块。
        mark = len(out)
        _walk_sections(sec, here, level + 1, out)

        if not own:
            continue                       # 空节丢弃
        if len(out) > mark:
            out[mark].text = own + "\n" + out[mark].text    # 并入第一个子块
        else:
            out.append(Section(title=title, text=own, level=level, path=here))


# ---- 对外接口 --------------------------------------------------------------

def _article_id(art) -> str:
    """取 PMC ID。⚠️ <article id="oto270312"> 是**出版社稿号**，不是 PMC ID。"""
    meta = art.find("front/article-meta")
    for pub_id_type in ("pmcid", "pmcaid"):
        node = meta.find(f'article-id[@pub-id-type="{pub_id_type}"]')
        if node is not None and (node.text or "").strip():
            if pub_id_type != "pmcid":
                logger.warning("缺 pmcid，退回 pmcaid=%s（不是标准展示形式）", node.text)
            return node.text.strip()
    logger.warning("这篇文章没有任何 PMC 系 ID → pmc_id 置空（不允许自己拼 'PMC'+数字）")
    return ""


def parse_article(article_bytes: bytes) -> Article:
    """XML 字节 → Article。规格见 tests/test_parse_pmc.py。"""
    root = etree.fromstring(article_bytes)
    art = _one_article(root)
    meta = art.find("front/article-meta")

    title_node = meta.find("title-group/article-title")
    title = _text_of(title_node) if title_node is not None else ""

    journal_node = art.find("front/journal-meta/journal-title")
    journal = _text_of(journal_node) if journal_node is not None else ""

    year_text = meta.findtext("pub-date/year")
    year = int(year_text) if year_text and year_text.strip().isdigit() else None

    abstract = ""
    sections: list[Section] = []
    abstract_node = meta.find("abstract")
    if abstract_node is not None:
        abstract = _text_of(abstract_node)
        if abstract:
            sections.append(
                Section(title=ABSTRACT_TITLE, text=abstract, level=1, path=[ABSTRACT_TITLE])
            )

    body = art.find("body")
    if body is not None:
        _walk_sections(body, [], 1, sections)

    return Article(
        pmc_id=_article_id(art),
        title=title,
        journal=journal,
        year=year,
        abstract=abstract,
        sections=sections,
    )


def main() -> None:
    """人眼验收：python -m src.ingest.parse_pmc [目录]

    测试过了不等于解析对了 —— 真实数据里一定有 fixture 没覆盖的怪东西。
    失败**不中断**，逐篇记账并打印失败率（这个数字本身就是产出）。
    """
    setup_logging()
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    files = sorted(out_dir.glob("*.xml"))
    ok = fail = 0
    for f in files:
        try:
            art = parse_article(f.read_bytes())
        except Exception as err:                       # noqa: BLE001
            fail += 1
            logger.warning("%s 解析失败：%s", f.name, err)
            continue
        ok += 1
        print(f"\n{f.name}  {art.pmc_id}  ({art.year})  {art.journal}")
        print(f"  title: {art.title[:70]}")
        for s in art.sections:
            print(f"  {'  ' * (s.level - 1)}· {' / '.join(s.path)}  ({len(s.text)} 字)")

    print(f"\n共 {len(files)} 篇：成功 {ok} / 失败 {fail}")
    print(f"成功率：{ok / len(files):.2%}")
    print(f"失败率：{1-ok / len(files):.2%}")


if __name__ == "__main__":
    main()
