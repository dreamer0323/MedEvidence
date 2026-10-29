"""评测指标：命中判定、Recall@5、MRR、汇总。"""
from __future__ import annotations

from src.eval.metrics import EvalRecord, first_hit_rank, reciprocal_rank, summarize
from src.retrieve.search import Hit


def make_hit(pmc_id: str, section: str, score: float = 1.0) -> Hit:
    return Hit(
        text="x",
        pmc_id=pmc_id,
        section=section,
        chunk_index=0,
        path=section,
        source_type="sec",
        score=score,
    )


HITS = [
    make_hit("PMC1", "Introduction", 0.9),
    make_hit("PMC2", "Abstract", 0.8),
    make_hit("PMC1", "Abstract", 0.7),
    make_hit("PMC3", "Results", 0.6),
    make_hit("PMC4", "Methods", 0.5),
    make_hit("PMC5", "Conclusion", 0.4),
]


def test_rank_one_when_first_hit_matches():
    assert first_hit_rank(HITS, ["PMC1"], ["Introduction"]) == 1


def test_rank_three():
    assert first_hit_rank(HITS, ["PMC1"], ["Abstract"]) == 3


def test_section_must_match_exactly():
    # PMC2 有 Abstract，但 gold 要的是 Results
    assert first_hit_rank(HITS, ["PMC2"], ["Results"]) is None


def test_pmc_must_match():
    assert first_hit_rank(HITS, ["PMC9"], ["Abstract"]) is None


def test_empty_gold_returns_none():
    assert first_hit_rank(HITS, [], []) is None
    assert first_hit_rank(HITS, ["PMC1"], []) is None


def test_returns_first_not_last():
    # Introduction 在第 1 位、Abstract 在第 3 位，两个都命中 → 取 1
    assert first_hit_rank(HITS, ["PMC1"], ["Abstract", "Introduction"]) == 1


def test_recall_at_5_excludes_rank_six():
    rank = first_hit_rank(HITS, ["PMC5"], ["Conclusion"])
    assert rank == 6

    record = EvalRecord(
        id="x",
        question="q",
        has_answer=True,
        rank=rank,
        refused=False,
        unsupported=0,
        sentence_count=1,
        unsupported_raw=0,
        sentence_count_raw=1,
    )
    assert record.recall_at_5 == 0.0
    assert reciprocal_rank(HITS, ["PMC5"], ["Conclusion"]) == 1 / 6


def test_summarize_rates():
    records = [
        EvalRecord("a", "q", True, 1, False, 0, 2, 0, 2),
        EvalRecord("b", "q", True, 6, False, 1, 4, 2, 5),
        EvalRecord("c", "q", False, None, True, 0, 0, 0, 0),
        EvalRecord("d", "q", True, None, True, 0, 0, 0, 0),
    ]
    summary = summarize(records)

    assert summary["n"] == 4
    # 分母 = 3 道有答案的题（a/b/d），负例 c 不进 Recall 分母
    assert summary["recall_at_5"] == 1 / 3
    assert summary["recall_n"] == 3
    assert summary["mrr"] == (1 + 1 / 6) / 2
    assert summary["mrr_n"] == 2
    assert summary["unsupported_rate"] == 1 / 6
    assert summary["uncited_raw_rate"] == 2 / 7
    assert summary["correct_refusal_rate"] == 1.0
    assert summary["wrong_refusal_rate"] == 1 / 3


def test_recall_denominator_excludes_negative_questions():
    records = [
        EvalRecord("a", "q", True, 1, False, 0, 1, 0, 1),
        EvalRecord("b", "q", False, None, True, 0, 0, 0, 0),
    ]
    summary = summarize(records)

    assert summary["recall_at_5"] == 1.0
    assert summary["recall_n"] == 1
