"""评测指标：全部用代码判定，不用 LLM-as-judge（见 plan/09「评测体系」）。

口径（2026-10-26 本人拍板 A / A）：

  · **命中** = Top-K 里存在某条 Hit，`pmc_id` 在 `gold_pmc_ids` 里，
    **并且** `section` 精确等于 `gold_sections` 中的某一项
  · **Recall@K** = Top-K 内命中的题数 / 总题数（分母含负例）
  · **MRR** = 只在 `rank` 存在的题上算 `1/rank`，再平均（没命中的题不进分母）
  · `section` 是**末级标题**（"Abstract" / "Survival analysis"），不是完整 `path`
"""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieve.search import Hit

DEFAULT_K = 5


def first_hit_rank(
    hits: list[Hit],
    gold_pmc_ids: list[str],
    gold_sections: list[str],
) -> int | None:
    """返回第一个命中 gold 的 Hit 的排名（从 1 起）；没有命中返回 None。

    TODO(human)：约 8 行。验收见 tests/test_eval.py：

      · 命中条件：`hit.pmc_id` 在 `gold_pmc_ids` 里，**并且**
        `hit.section` 精确等于 `gold_sections` 里的某一项
      · 返回**第一个**命中的位置（1 起），不是最后一个
      · `gold_pmc_ids` 或 `gold_sections` 为空 → 直接返回 None（负例没有 gold）
      · 一条都没命中 → None

    提示：`for rank, hit in enumerate(hits, start=1):`
    """
    if not hits:
        return None
    if not gold_pmc_ids or not gold_sections:
        return None

    for rank, hit in enumerate(hits, start=1):
        if hit.pmc_id in gold_pmc_ids and hit.section in gold_sections:
            return rank
        
    return None


def hit_at_k(
    hits: list[Hit],
    gold_pmc_ids: list[str],
    gold_sections: list[str],
    k: int = DEFAULT_K,
) -> bool:
    """Top-K 里有没有命中。"""
    rank = first_hit_rank(hits[:k], gold_pmc_ids, gold_sections)
    if rank is None:
        return False
    return True


def reciprocal_rank(
    hits: list[Hit],
    gold_pmc_ids: list[str],
    gold_sections: list[str],
) -> float | None:
    """1 / 第一个命中的排名；没命中返回 None（调用方决定要不要进分母）。"""
    rank = first_hit_rank(hits, gold_pmc_ids, gold_sections)
    if rank is None:
        return None
    return 1.0 / rank


@dataclass(frozen=True)
class EvalRecord:
    """一条评测样本跑完后的记录。"""

    id: str
    question: str
    has_answer: bool        # 语料里到底有没有答案（人工标注）
    rank: int | None        # 第一个命中的排名（在 top_k 内）
    refused: bool           # 模型有没有拒答
    unsupported: int        # 没有引用的事实句数（引导句已合并）
    sentence_count: int     # 有效句数（拒答时为 0）
    unsupported_raw: int    # 不合并引导句时的裸奔句数（对照用）
    sentence_count_raw: int # 不合并引导句时的有效句数（对照用）

    @property
    def recall_at_5(self) -> float:
        if self.rank is None:
            return 0.0
        if self.rank <= DEFAULT_K:
            return 1.0
        return 0.0


def summarize(records: list[EvalRecord]) -> dict:
    """把逐条记录汇总成 baseline 数字。

    每个指标的分母都在注释里写死 —— 换分母必须改这里，不能偷偷改。
    """
    if not records:
        raise ValueError("评测集为空，无法汇总")

    total = len(records)

    recall_sum = 0.0    # Recall@5 的分子, 只算「语料里有答案」的题
    recall_n = 0        # Recall@5 的分母
    rr_sum = 0.0      # MRR 的分子  
    rr_count = 0      # MRR 的分母

    unsupported_sum = 0
    sentence_sum = 0
    unsupported_raw_sum = 0
    sentence_raw_sum = 0

    should_refuse = 0
    refused_correctly = 0
    should_answer = 0
    refused_wrongly = 0

    for record in records:
        # Recall 的分母只算「语料里有答案」的题 —— 负例永远命中不了，
        # 算进分母会把分数硬性封顶（见 plan/09 口径修正）。
        if record.has_answer:
            recall_sum += record.recall_at_5
            recall_n += 1

        if record.rank is not None:
            rr_sum += 1.0 / record.rank
            rr_count += 1

        # 拒答的答案没有事实句，不能进 unsupported 的分母
        if not record.refused:
            unsupported_sum += record.unsupported
            sentence_sum += record.sentence_count
            unsupported_raw_sum += record.unsupported_raw
            sentence_raw_sum += record.sentence_count_raw

        if record.has_answer:
            should_answer += 1
            if record.refused:
                refused_wrongly += 1
        else:
            should_refuse += 1
            if record.refused:
                refused_correctly += 1

    if recall_n == 0:
        recall_at_5 = None
    else:
        recall_at_5 = recall_sum / recall_n

    if rr_count == 0:
        mrr = 0.0
    else:
        mrr = rr_sum / rr_count

    if sentence_sum == 0:
        unsupported_rate = 0.0
    else:
        unsupported_rate = unsupported_sum / sentence_sum

    # 对照指标：不合并引导句时的裸奔率。和主指标的差 = 合并规则掩盖了多少。
    if sentence_raw_sum == 0:
        uncited_raw_rate = 0.0
    else:
        uncited_raw_rate = unsupported_raw_sum / sentence_raw_sum

    if should_refuse == 0:
        correct_refusal_rate = None
    else:
        correct_refusal_rate = refused_correctly / should_refuse

    if should_answer == 0:
        wrong_refusal_rate = None
    else:
        wrong_refusal_rate = refused_wrongly / should_answer

    return {
        "n": total,
        "recall_at_5": recall_at_5,
        "recall_n": recall_n,
        "mrr": mrr,
        "mrr_n": rr_count,
        "unsupported_rate": unsupported_rate,
        "uncited_raw_rate": uncited_raw_rate,
        "correct_refusal_rate": correct_refusal_rate,
        "wrong_refusal_rate": wrong_refusal_rate,
    }
