"""生成链路：Prompt 组装 + 引用校验 + 空证据不调模型。"""
from __future__ import annotations

from src.generate.client import NO_EVIDENCE_ANSWER, generate
from src.generate.prompt import build_prompt, is_refusal, verify_citations
from src.retrieve.search import Hit


def make_hit(pmc_id: str, section: str, text: str, chunk_index: int = 0) -> Hit:
    return Hit(
        text=text,
        pmc_id=pmc_id,
        section=section,
        chunk_index=chunk_index,
        path=section,
        source_type="sec",
        score=1.0,
    )


EVIDENCE = [
    make_hit("PMC111", "Results / Survival", "HER2 评分准确率为 95%。", 0),
    make_hit("PMC222", "Methods", "纳入了 200 例患者。", 0),
    make_hit("PMC333", "Discussion", "该方法的局限是样本量小。", 0),
]


# --------------------------------------------------------------- build_prompt

def test_build_prompt_numbers_from_one():
    prompt = build_prompt("HER2 的准确率是多少？", EVIDENCE)
    assert "[1] 来源：PMC111 · 节：Results / Survival" in prompt
    assert "[2] 来源：PMC222 · 节：Methods" in prompt
    assert "[3] 来源：PMC333 · 节：Discussion" in prompt
    assert "HER2 的准确率是多少？" in prompt
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]")


def test_build_prompt_empty_evidence():
    assert "（无）" in build_prompt("随便问", [])


# --------------------------------------------------------------- verify_citations

def test_single_citation():
    report = verify_citations("HER2 评分准确率为 95% [1]。", EVIDENCE)
    assert report.used == (1,)
    assert report.invalid == ()
    assert report.unsupported == ()
    assert report.uncited == (2, 3)
    assert report.ok


def test_multi_and_spaced_citation():
    report = verify_citations("准确率为 95% [1,3]，样本 200 例 [ 2 ]。", EVIDENCE)
    assert report.used == (1, 2, 3)
    assert report.invalid == ()
    assert report.unsupported == ()


def test_out_of_range_is_invalid_not_used():
    report = verify_citations("编的 [0]，另一个编的 [99]。", EVIDENCE)
    assert report.used == ()
    assert report.invalid == (0, 99)
    assert not report.ok


def test_duplicate_citation_counted_once():
    report = verify_citations("结论 A [1]。结论 B [1]。", EVIDENCE)
    assert report.used == (1,)


def test_sentence_without_citation_is_unsupported():
    report = verify_citations("准确率 95% [1]。该研究还发现年龄与预后相关。", EVIDENCE)
    assert report.used == (1,)
    assert report.unsupported == ("该研究还发现年龄与预后相关。",)


def test_no_citation_at_all():
    report = verify_citations("我不知道。", EVIDENCE)
    assert report.used == ()
    assert report.unsupported == ("我不知道。",)
    assert report.uncited == (1, 2, 3)


def test_range_citation_is_expanded():
    report = verify_citations("结论 [1-3]。", EVIDENCE)
    assert report.used == (1, 2, 3)
    assert report.uncited == ()
    assert report.unsupported == ()


def test_en_dash_range_citation():
    report = verify_citations("结论 [1–2]。", EVIDENCE)
    assert report.used == (1, 2)


def test_range_mixed_with_list():
    report = verify_citations("结论 [1-2,3]。", EVIDENCE)
    assert report.used == (1, 2, 3)


def test_out_of_range_range_is_invalid():
    report = verify_citations("结论 [2-9]。", EVIDENCE)
    assert report.used == (2, 3)
    assert report.invalid == (4, 5, 6, 7, 8, 9)


def test_huge_range_does_not_explode():
    report = verify_citations("结论 [1-9999999]。", EVIDENCE)
    assert report.used == (1,)
    assert report.invalid == (9999999,)


def test_markdown_headings_are_not_unsupported():
    answer = "## 结论\n\n**1. 要点**\n准确率 95% [1]。"
    report = verify_citations(answer, EVIDENCE)
    assert report.unsupported == ()
    assert report.used == (1,)


def test_inline_bold_with_citation_is_kept():
    answer = "- **Patch 水平**：准确率 95% [1]。"
    report = verify_citations(answer, EVIDENCE)
    assert report.unsupported == ()
    assert report.used == (1,)


def test_refusal_sentence_is_not_unsupported():
    report = verify_citations("现有证据不足以回答这个问题。", EVIDENCE)
    assert report.unsupported == ()
    assert report.used == ()


def test_is_refusal_detects_template():
    assert is_refusal("现有证据不足以回答这个问题。")
    assert is_refusal(NO_EVIDENCE_ANSWER)
    assert not is_refusal("准确率为 95% [1]。")


def test_is_refusal_known_false_positive():
    # 已知代价：事实句里恰好包含完整模板串，也会被误判成拒答（plan/09 已记）
    assert is_refusal("该研究解释了什么情况下模型无法回答这个问题 [1]。")


def test_lead_in_merged_with_cited_next():
    answer = "准确率如下：\nPatch 水平 96.49% [1]。"
    report = verify_citations(answer, EVIDENCE)
    assert report.unsupported == ()


def test_lead_in_with_uncited_next_is_flagged():
    answer = "该药物 2023 年获批：\n随后扩大了适应症。"
    report = verify_citations(answer, EVIDENCE)
    assert len(report.unsupported) == 1


def test_trailing_colon_without_next_is_kept():
    answer = "该药物 2023 年获批："
    report = verify_citations(answer, EVIDENCE)
    assert len(report.unsupported) == 1


def test_empty_evidence_has_no_uncited():
    report = verify_citations(NO_EVIDENCE_ANSWER, [])
    assert report.used == ()
    assert report.invalid == ()
    assert report.uncited == ()


# --------------------------------------------------------------- generate

class _ExplodingClient:
    """证据为空时如果被调用就炸 —— 这是 D1 决策的回归测试。"""

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                raise AssertionError("证据为空时不该调用 LLM")


def test_generate_skips_llm_when_no_evidence():
    answer = generate("随便问", [], client=_ExplodingClient(), model="stub")
    assert answer.text == NO_EVIDENCE_ANSWER
    assert answer.citations.used == ()


class _StubClient:
    """固定返回一句带引用的答案。"""

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                message = type("Message", (), {"content": "准确率为 95% [1]。"})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()


def test_generate_parses_citations_from_model_output():
    answer = generate("准确率？", EVIDENCE, client=_StubClient(), model="stub")
    assert answer.text == "准确率为 95% [1]。"
    assert answer.citations.used == (1,)
    assert answer.citations.invalid == ()
    assert answer.model == "stub"
