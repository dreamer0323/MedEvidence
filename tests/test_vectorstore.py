"""向量库测试：用随机向量，不调 embedding API —— 所以不需要密钥、不花钱。

为什么值得为封装写测试：它下面压着 Milvus，Milvus 的报错信息对不上你的字段名，
出问题时你分不清是"分块写错了"还是"库用错了"。这层测试把两者切开。
"""
from __future__ import annotations

import random

import pytest

from src.config import EMBEDDING_DIM
from src.vectorstore.milvus_store import MilvusStore

COLLECTION = "chunks"


def _vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(EMBEDDING_DIM)]


def _row(seed: int, pmc_id: str, text: str, section: str = "Results", idx: int = 0):
    return {
        "vector": _vec(seed),
        "text": text,
        "pmc_id": pmc_id,
        "section": section,
        "chunk_index": idx,
    }


@pytest.fixture
def store(tmp_path):
    # 每个用例一个独立的 .db 目录，避免互相污染，也避免踩 Windows 的文件锁
    with MilvusStore(uri=str(tmp_path / "test.db")) as s:
        s.ensure_collection()
        yield s


def test_insert_count_and_search_roundtrip(store):
    rows = [
        _row(1, "PMC111", "乳腺浸润性导管癌的免疫组化判读"),
        _row(2, "PMC222", "全切片图像的染色差异"),
        _row(3, "PMC333", "FISH 检测 HER2 扩增"),
    ]
    assert store.insert(rows) == 3
    assert store.count() == 3

    hits = store.search(rows[0]["vector"], top_k=3)

    assert len(hits) == 3
    # 用第 0 行的向量去查，它自己必须排第一，且相似度为 1
    assert hits[0]["pmc_id"] == "PMC111"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    # 返回结构必须是抹平过的，上层不该看到 Milvus 的嵌套
    assert set(hits[0]) == {"id", "score", "text", "pmc_id", "section", "chunk_index"}


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()          # 第二次调用不该炸，也不该清空
    store.insert([_row(1, "PMC111", "x")])
    store.ensure_collection()
    assert store.count() == 1


def test_insert_rejects_missing_field(store):
    row = _row(1, "PMC111", "x")
    del row["pmc_id"]
    with pytest.raises(ValueError, match="pmc_id"):
        store.insert([row])
    assert store.count() == 0           # 校验失败不该留下半截数据


def test_insert_rejects_wrong_dimension(store):
    row = _row(1, "PMC111", "x")
    row["vector"] = row["vector"][:512]
    with pytest.raises(ValueError, match="维度"):
        store.insert([row])


def test_insert_rejects_overlong_text(store):
    """超长 chunk 必须在写入端炸掉 —— 静默截断会让引用内容不完整。"""
    row = _row(1, "PMC111", "长" * 9000)
    with pytest.raises(ValueError, match="分块参数"):
        store.insert([row])


def test_search_rejects_wrong_query_dimension(store):
    with pytest.raises(ValueError, match="维度"):
        store.search([0.1] * 8)


def test_insert_empty_list_is_noop(store):
    assert store.insert([]) == 0
