"""跑评测集：python -m src.eval.run [N]

N 是条数上限，不填就跑全部。
逐条 retrieve + generate，落盘明细（一次运行一个文件）和汇总（baseline.json）。

钱花在 generate 上：先跑 3 条确认链路，再放开全量。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from src.config import EVAL_DATASET, EVAL_RESULTS_DIR, ensure_dirs, setup_logging
from src.eval.manifest import file_sha256, git_commit, read_manifest
from src.eval.metrics import EvalRecord, first_hit_rank, summarize
from src.eval.validate import validate_dataset
from src.generate.client import generate
from src.generate.prompt import count_sentences, is_refusal, unsupported_sentences
from src.retrieve import retrieve

# 检索 10 条：Recall@5 只看前 5，MRR 需要看到第 6-10 名
TOP_K = 10


def load_dataset(path: Path) -> list[dict]:
    """
    加载评测集，返回一个 dict 列表。
    每个 dict 包含 id, question, has_answer, gold_pmc_ids, gold_sections, tags。
    """
    items: list[dict] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return items


def describe_hits(hits) -> list[dict]:
    """把 Top-K 压成可落盘的形状（不带正文，避免结果文件过大）。"""
    rows: list[dict] = []

    for rank, hit in enumerate(hits, start=1):
        rows.append({
            "rank": rank,
            "pmc_id": hit.pmc_id,
            "section": hit.section,
            "score": round(hit.score, 4),
        })

    return rows


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None  # 不填就跑全部

    ensure_dirs()
    setup_logging()

    items = load_dataset(EVAL_DATASET)
    if limit is not None:
        items = items[:limit]

    # 先校验再花钱：gold 写错的题会被当成检索失败，白调一周
    errors = validate_dataset(items)
    if errors:
        print("评测集校验失败：")
        for message in errors:
            print(f"  ✗ {message}")
        return 1

    records: list[EvalRecord] = []
    details: list[dict] = []

    for item in items:
        hits = retrieve(item["question"], top_k=TOP_K)
        answer = generate(item["question"], hits)

        rank = first_hit_rank(hits, item["gold_pmc_ids"], item["gold_sections"])
        refused = is_refusal(answer.text)

        # 主指标：引导句并入下一句
        unsupported_count = len(answer.citations.unsupported)
        sentence_count = count_sentences(answer.text)

        # 对照指标：不合并、原样计入，用来看合并规则掩盖了多少
        unsupported_raw = len(unsupported_sentences(answer.text, merge_lead_in=False))
        sentence_count_raw = count_sentences(answer.text, merge_lead_in=False)

        records.append(EvalRecord(
            id=item["id"],
            question=item["question"],
            has_answer=item["has_answer"],
            rank=rank,
            refused=refused,
            unsupported=unsupported_count,
            sentence_count=sentence_count,
            unsupported_raw=unsupported_raw,
            sentence_count_raw=sentence_count_raw,
        ))

        details.append({
            "id": item["id"],
            "question": item["question"],
            "has_answer": item["has_answer"],
            "tags": item.get("tags", []),
            "rank": rank,
            "refused": refused,
            "sentence_count": sentence_count,
            "unsupported": list(answer.citations.unsupported),
            "used": list(answer.citations.used),
            "invalid": list(answer.citations.invalid),
            "uncited": list(answer.citations.uncited),
            "answer": answer.text,
            "model": answer.model,
            "top_hits": describe_hits(hits),
        })

        print(f"[{item['id']}] rank={rank} refused={refused} "
              f"unsupported={len(answer.citations.unsupported)}")

    summary = summarize(records)

    # 结果版本化：没有这个块，下周的数字就没法和这周比
    manifest = read_manifest()
    summary["_versions"] = {
        "dataset_sha256": file_sha256(EVAL_DATASET),
        "corpus_chunks": manifest.get("n_chunks"),
        "git_commit": git_commit(),
        "llm_model": details[-1]["model"] if details else "",
        "retriever": f"vector(top_k={TOP_K})",
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    detail_path = EVAL_RESULTS_DIR / f"run-{stamp}.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = EVAL_RESULTS_DIR / "baseline.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细：{detail_path}")
    print(f"汇总：{summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
