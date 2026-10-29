"""混合检索：向量 + BM25，用 RRF 融合。

为什么要混合：两路的失败模式不同 ——
  · 向量（稠密）：擅长语义近似，但会把 `LH510` 泛化成"扫描仪"
  · BM25（稀疏）：擅长精确词面，但同义改写（`four stages` vs `four-stage pipeline`）救不了

为什么用 RRF 而不是分数加权：余弦相似度是 0~1，BM25 分数可以到 50+，
两者**不可比**，加权只能拍脑袋。RRF 只看排名，绕开分数校准。

两路各取 Top-N（默认 20），融合后取 Top-K。N 取 20 是因为 baseline 里
正确答案常落在 6-10 名（见 plan/09 的失败分解）。
"""
from __future__ import annotations

from dataclasses import replace

from src.retrieve.bm25 import BM25Index
from src.retrieve.search import DEFAULT_TOP_K, Hit, retrieve
from src.vectorstore import MilvusStore

DEFAULT_N_EACH = 20
RRF_K = 60


def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """把多路排名融合成一个排名，返回 [(id, 分数)]，按分数降序。

      · 输入：若干「已经按相关性排好序」的 id 列表，列表内部不重复
      · 每个 id 的分数 = Σ_路 `1 / (k + rank)`，rank **从 1 起**
      · 只出现在一路里的 id 也要保留（分数只算那一路）
      · 输出按分数**降序**；分数相同时按 id 升序，保证结果稳定

    """
    scores: dict[str, float] = {}
    # 累加分数
    for rank_list in ranked_lists:
        rank=1
        for doc_id in rank_list:
            current_score = 1 / (k + rank)

            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += current_score
            rank += 1
    
    result_list = list(scores.items())
    def sort_rule(item):
        doc_id = item[0]
        score = item[1]
        return (-score, doc_id)
        
    result_list.sort(key=sort_rule)
    return result_list



def _pk_of(hit: Hit) -> str:
    """Hit 没有 pk 字段，按建库规则拼回去：`{pmc_id}:{chunk_index}`。"""
    return f"{hit.pmc_id}:{hit.chunk_index}"


def _fetch_hits(store: MilvusStore, pks: list[str]) -> dict[str, Hit]:
    """按 pk 批量取回 Hit（供只在 BM25 一路出现的候选使用）。"""
    if not pks:
        return {}

    quoted = ", ".join(f'"{pk}"' for pk in pks)
    rows = store.client.query(
        collection_name=store.collection,
        filter=f"pk in [{quoted}]",
        output_fields=[
            "pk", "text", "pmc_id", "section", "chunk_index", "path", "source_type",
        ],
        limit=len(pks),
    )

    found: dict[str, Hit] = {}
    for row in rows:
        found[row["pk"]] = Hit(
            text=row["text"],
            pmc_id=row["pmc_id"],
            section=row["section"],
            chunk_index=row["chunk_index"],
            path=row["path"],
            source_type=row["source_type"],
            score=0.0,      # 这里没有检索分；融合分在下面统一替换
        )

    return found


def hybrid_retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    store: MilvusStore | None = None,
    embedder=None,
    bm25: BM25Index | None = None,
    n_each: int = DEFAULT_N_EACH,
) -> list[Hit]:
    """向量 + BM25 各取 n_each 条，RRF 融合后返回 top_k 条。

    `Hit.score` 返回的是 **RRF 融合分**，不是余弦相似度 —— 两者量纲不同，别混用。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空：空向量查询是 bug，不是合法输入")

    own_store = store is None
    store = store or MilvusStore()

    try:
        store.ensure_collection()

        vector_hits = retrieve(query, top_k=n_each, store=store, embedder=embedder)
        bm25 = bm25 or BM25Index.load_or_build(store)
        keyword_hits = bm25.search(query, top_k=n_each)

        vector_pks: list[str] = []
        for hit in vector_hits:
            vector_pks.append(_pk_of(hit))

        keyword_pks: list[str] = []
        for pk, _score in keyword_hits:
            keyword_pks.append(pk)

        fused = rrf_fuse([vector_pks, keyword_pks])

        known: dict[str, Hit] = {}
        for hit in vector_hits:
            known[_pk_of(hit)] = hit

        missing: list[str] = []
        for pk, _score in fused:
            if pk not in known:
                missing.append(pk)

        if missing:
            known.update(_fetch_hits(store, missing))

        results: list[Hit] = []
        for pk, fused_score in fused[:top_k]:
            hit = known.get(pk)
            if hit is None:
                continue
            results.append(replace(hit, score=fused_score))

        return results

    finally:
        if own_store:
            store.close()
