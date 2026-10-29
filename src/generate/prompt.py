"""把检索结果组装成 Prompt，并校验答案里的引用编号。

契约（见 plan/09「生成链路：证据块级编号 + L2 引用校验」）：

  · 证据用 `[1]..[n]` 编号，模型只写编号、不写 PMC ID —— 程序负责映射回 Hit。
    理由：模型编 PMC ID 编得很像真的，编短编号一眼就能查出来。
  · 编号在 `build_prompt` 里建立一次；`verify_citations` 必须拿到**同一份** evidence。
    只要中间过滤 / 去重 / 重排过而没有同步，编号就会和 Hit 错位 —— 这是这段代码
    最容易写错的地方。
  · 越界编号 = 模型在编，不是"警告"。
  · L2 只做两件事：范围校验 + 句子级无引用检测。**抓不到"编号合法但内容不符"的
    假引用**（那是 L3 的活，第 7 周有数据再决定）。

写法约定：本模块刻意不用推导式和链式调用，一律用显式循环再配注释 ——
这段代码要能一眼讲给别人听，省下的几行不值得多花理解成本。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieve.search import Hit

SYSTEM_PROMPT = """你是医学文献证据助手，只根据用户提供的【证据】回答。

规则：
1. 每一个事实性陈述后面必须标注来源编号，格式为 [n]，例如 [1]；多个来源写成 [1,3]。
2. 只能使用【证据】里出现过的编号。不要编造编号，不要输出 PMC ID。
3. 证据不足以回答时，直接说「现有证据不足以回答这个问题」，不要用背景知识补全。
4. 区分证据里的结论与推测：原文没有的因果解释不要加。
5. 用中文回答；医学术语保留英文原词，必要时在括号里给出。
"""

# 引用标记：[1] / [ 1 ] / [1,3] / [1，3] / [1-5] / [1–5] / [1-3,5] 都算。
# `-` 和 en dash `–` 都可能是区间 —— 真实模型两种都写过（见 plan/04 #24）。
_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*[-–,，]\s*\d+)*)\s*\]")
_CITE_SPLIT_RE = re.compile(r"[,，]")
_RANGE_SPLIT_RE = re.compile(r"\s*[-–]\s*")

# 区间最多展开多少个编号。防止模型写 [1-9999999] 把内存打爆。
_RANGE_EXPAND_LIMIT = 100

# 粗切句子：中文句末标点直接切；英文 `. ! ?` 只在后面跟空白时切（否则 0.95 会被切开）；换行也切。
# 已知坑：`vs.` / `e.g.` 这类缩写后跟空格仍会误切。误切只会让 unsupported 偏多
# （保守），不会漏报。量化误切率见 plan/04 #22。
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+|\n+")

# ---- Markdown 结构 ----
# 切句之前必须先擦掉，否则 `## 标题` / `**1.` 会被切成"没有引用的句子"。
# 见 plan/04 #25：IHC 那条答案曾因此被判成 11 句裸奔。
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_BOLD_LINE_RE = re.compile(r"^\s*(?:\*\*|__).+?(?:\*\*|__)\s*$", re.MULTILINE)  # 整行加粗 = 小标题
_BOLD_RE = re.compile(r"(\*\*|__)")
_LIST_MARKER_RE = re.compile(r"^\s{0,3}(?:[-*+]|\d+\.)\s+", re.MULTILINE)

# ---- 拒答模板 ----
# 命中即视为"模型选择了不答"。这类句子本来就不该有引用，不能算裸奔。
_REFUSAL_MARKERS = (
    "现有证据不足",
    "现有语料中没有找到",
    "无法回答这个问题",
)


# --------------------------------------------------------------- 小工具


def _extract_citation_numbers(answer: str) -> list[int]:
    """把答案里的引用编号抠出来，保留重复；区间会展开成一个个编号。

    "A [1]。B [1,3]。C [2-4]。" → [1, 1, 3, 2, 3, 4]
    去重交给调用方 —— "这个编号出现了几次"以后可能有用。
    """
    numbers: list[int] = []

    groups = _CITE_RE.findall(answer)
    for group in groups:
        # group 形如 "1" / "1,3" / "1-5" / "1-3,5"，先按逗号拆成若干段
        pieces = _CITE_SPLIT_RE.split(group)
        for piece in pieces:
            # 每一段可能是单个编号 "3"，也可能是区间 "1-5"
            bounds = _RANGE_SPLIT_RE.split(piece)

            if len(bounds) == 1:
                number = int(bounds[0].strip())
                numbers.append(number)
                continue

            start = int(bounds[0].strip())
            end = int(bounds[1].strip())
            expanded = _expand_range(start, end)
            for number in expanded:
                numbers.append(number)

    return numbers


def _expand_range(start: int, end: int) -> list[int]:
    """把区间 [start, end] 展开成编号列表。

    上限保护：跨度超过 _RANGE_EXPAND_LIMIT 时只记两端点 ——
    模型写 [1-9999999] 时不至于把内存打爆。
    倒序区间（如 [5-1]）不是合法写法，直接忽略。
    """
    if end < start:
        return []

    if end - start > _RANGE_EXPAND_LIMIT:
        return [start, end]

    expanded: list[int] = []
    for number in range(start, end + 1):
        expanded.append(number)
    return expanded


def _split_sentences(answer: str) -> list[str]:
    """粗切句子，丢掉空串。"""
    sentences: list[str] = []

    pieces = _SENTENCE_SPLIT_RE.split(answer)
    for piece in pieces:
        sentence = piece.strip()
        if sentence:
            sentences.append(sentence)

    return sentences


def _normalize_for_sentences(answer: str) -> str:
    """擦掉 Markdown 排版，只留下能当作事实陈述的正文。

    必须在切句**之前**做：`**1.` 这类标记会被句号切成独立"句子"，
    然后被误判成"没有引用的事实句"。见 plan/04 #25。
    """
    text = _CODE_FENCE_RE.sub("", answer)
    text = _HEADING_LINE_RE.sub("", text)   # `## 标题` 整行删掉
    text = _BOLD_LINE_RE.sub("", text)      # 整行加粗 = 小标题，整行删掉
    text = _BOLD_RE.sub("", text)           # 行内残留的加粗标记只删符号
    text = _LIST_MARKER_RE.sub("", text)    # `-` / `1.` 这类列表符号
    return text


def is_refusal(answer: str) -> bool:
    """答案是不是拒答。

    第 4 周用它**单独算拒答率** —— 拒答不是"没标引用"，是"选择了不答"，
    不能混进 unsupported（plan/04 #25）。
    """
    head = answer.strip()[:120]
    for marker in _REFUSAL_MARKERS:
        if marker in head:
            return True
    return False


def _is_refusal_sentence(sentence: str) -> bool:
    for marker in _REFUSAL_MARKERS:
        if marker in sentence:
            return True
    return False


def _merge_lead_in_sentences(sentences: list[str]) -> list[str]:
    """把以冒号结尾的引导句并入下一句。

    为什么不是直接丢掉：引导句自己也可能带事实 ——
    "该药物 2023 年获批：" 这句话如果被丢掉，就等于给它免检。
    合并之后再看**整组**有没有引用：
      · 有引用 → 前一句确实只是引子，整组通过
      · 没引用 → 两句一起标成 unsupported
    末尾没有下一句的引导句原样保留（同样要接受检查）。
    """
    merged: list[str] = []
    pending = ""

    for sentence in sentences:
        if pending:
            sentence = pending + " " + sentence
            pending = ""

        if sentence.endswith("：") or sentence.endswith(":"):
            pending = sentence
            continue

        merged.append(sentence)

    if pending:
        merged.append(pending)

    return merged


def _sentences_for_check(answer: str, merge_lead_in: bool) -> list[str]:
    """归一化 + 切句 + 可选的引导句合并 —— 分子分母共用这一条路径。"""
    normalized = _normalize_for_sentences(answer)
    sentences = _split_sentences(normalized)

    if merge_lead_in:
        sentences = _merge_lead_in_sentences(sentences)

    return sentences


def unsupported_sentences(answer: str, merge_lead_in: bool = True) -> list[str]:
    """返回一个引用编号都没有的句子 —— 模型"自由发挥"的部分。

    三类句子不算：
      · 拒答 / 元陈述（本来就不该有引用）
      · Markdown 结构（标题、加粗标记），已在归一化阶段去掉
      · 引导句，已在切句后并入下一句（见 `_merge_lead_in_sentences`）

    `merge_lead_in=False` 用于算对照指标（不合并、原样计入）。
    """
    unsupported: list[str] = []

    for sentence in _sentences_for_check(answer, merge_lead_in):
        if _CITE_RE.search(sentence) is not None:
            continue
        if _is_refusal_sentence(sentence):
            continue
        unsupported.append(sentence)

    return unsupported


def count_sentences(answer: str, merge_lead_in: bool = True) -> int:
    """有效句数：分子分母必须用同一种切法，否则比率没有意义。"""
    count = 0

    for sentence in _sentences_for_check(answer, merge_lead_in):
        if _is_refusal_sentence(sentence):
            continue
        count += 1

    return count


# --------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class CitationReport:
    """一次答案的引用体检结果。"""

    used: tuple[int, ...]         # 答案里实际引用、且合法的编号（升序去重）
    invalid: tuple[int, ...]      # 越界编号（n < 1 或 n > 证据条数）
    unsupported: tuple[str, ...]  # 一个引用都没有的句子 —— L2 要抓的"自由发挥"
    uncited: tuple[int, ...]      # 给了证据、但答案里一次都没用上的编号

    @property
    def ok(self) -> bool:
        """没有越界编号就算通过。unsupported 算不算失败由调用方按场景决定。"""
        return len(self.invalid) == 0


# --------------------------------------------------------------- 组装 Prompt


def build_prompt(question: str, evidence: list[Hit]) -> str:
    """问题 + 编号证据 → 一次对话的 user message。

    证据按传入顺序编号（约定：调用方先按检索分降序排好，这里不再排序）。
    编号从 1 起，与 verify_citations 的判定口径一致。
    """
    blocks: list[str] = []

    for index, hit in enumerate(evidence, start=1):
        header = f"[{index}] 来源：{hit.cite()} · 节：{hit.path}"
        body = hit.text.strip()
        blocks.append(header + "\n" + body)

    if blocks:
        evidence_text = "\n\n".join(blocks)
    else:
        evidence_text = "（无）"

    question = question.strip()
    return f"【证据】\n{evidence_text}\n\n【问题】\n{question}"


# --------------------------------------------------------------- 校验引用


def verify_citations(answer: str, evidence: list[Hit]) -> CitationReport:
    """答案里引用了哪些证据、有没有越界、哪些句子没挂引用、哪些证据没被用上。

    分四步做，每一步只干一件事：

      第 1 步  抠出答案里所有编号（保留重复）
      第 2 步  按 1..total 分成"合法"和"越界"两类
      第 3 步  找出给了、但一次都没被引用的证据
      第 4 步  找出一个引用都没有的句子（逐句判，不能看全局）
    """
    total = len(evidence)

    # 第 1 步：抠编号
    numbers = _extract_citation_numbers(answer)

    # 第 2 步：分两类。同一个编号重复出现时只记一次。
    used: list[int] = []
    invalid: list[int] = []

    for number in numbers:
        if 1 <= number <= total:
            if number not in used:
                used.append(number)
        else:
            if number not in invalid:
                invalid.append(number)

    used.sort()
    invalid.sort()

    # 第 3 步：给了、但答案里一次都没用上的证据
    uncited: list[int] = []

    for number in range(1, total + 1):
        if number not in used:
            uncited.append(number)

    # 第 4 步：一个引用都没有的句子（引导句会先并入下一句）
    unsupported = unsupported_sentences(answer)

    return CitationReport(
        used=tuple(used),
        invalid=tuple(invalid),
        unsupported=tuple(unsupported),
        uncited=tuple(uncited),
    )
