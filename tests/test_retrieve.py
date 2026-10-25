"""检索模块的单测：桩 embedding + 临时 Milvus，不调 API、不花钱。

断言的是**行为**不是实现：条数、字段、排序、边界、异常。
"""
from __future__ import annotations

import pytest

from src.config import EMBEDDING_DIM
from src.retrieve import retrieve
from src.vectorstore import ChunkRow, MilvusStore


def onehot(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def row(pk: str, i: int, text: str, pmc_id: str = "PMC1", section: str = "Methods") -> ChunkRow:
    return ChunkRow(
        pk=pk, vector=onehot(i), text=text, pmc_id=pmc_id, section=section,
        chunk_index=int(pk.split(":")[1]), path=section, source_type="sec",
        text_hash=f"h-{pk}",
    )


ROWS = [
    row("PMC1:0", 0, "alpha"),
    row("PMC1:1", 1, "beta"),
    row("PMC2:0", 2, "gamma", pmc_id="PMC2", section="Results"),
]


class StubEmbedder:
    """返回固定向量，并记下被传入的 query —— 用来验证"没加前缀"。"""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


@pytest.fixture
def store(tmp_path):
    with MilvusStore(uri=str(tmp_path / "t.db")) as s:
        s.ensure_collection()
        s.upsert(ROWS)
        yield s


def test_returns_topk_with_expected_fields(store):
    hits = retrieve("alpha", top_k=2, store=store, embedder=StubEmbedder(onehot(0)))

    assert len(hits) == 2
    assert hits[0].text == "alpha"
    assert hits[0].pmc_id == "PMC1"
    assert hits[0].path == "Methods"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_results_are_sorted_by_score_desc(store):
    query = [0.0] * EMBEDDING_DIM
    query[0], query[1] = 0.8, 0.6          # 同时靠近 alpha 和 beta

    hits = retrieve("q", top_k=3, store=store, embedder=StubEmbedder(query))

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].text == "alpha"


def test_empty_query_raises(store):
    with pytest.raises(ValueError, match="不能为空"):
        retrieve("   ", store=store, embedder=StubEmbedder(onehot(0)))


def test_top_k_larger_than_corpus_returns_all(store):
    hits = retrieve("q", top_k=99, store=store, embedder=StubEmbedder(onehot(0)))
    assert len(hits) == len(ROWS)


def test_min_score_filters_low_similarity(store):
    hits = retrieve(
        "q", top_k=3, min_score=0.5, store=store, embedder=StubEmbedder(onehot(0))
    )
    assert len(hits) == 1
    assert hits[0].text == "alpha"


def test_query_is_passed_through_without_prefix(store):
    stub = StubEmbedder(onehot(0))
    retrieve("乳腺癌 IHC 判读", store=store, embedder=stub)
    assert stub.calls == ["乳腺癌 IHC 判读"]      # 没有标题前缀，没有改写


def test_injected_store_is_not_closed(store):
    retrieve("alpha", store=store, embedder=StubEmbedder(onehot(0)))
    assert store.count() == len(ROWS)             # 注入的库由调用方负责关闭
