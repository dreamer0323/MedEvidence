"""向量库测试：用随机向量，不调 embedding API —— 所以不需要密钥、不花钱。

为什么值得为封装写测试：它下面压着 Milvus，Milvus 的报错信息对不上你的字段名，
出问题时你分不清是"分块写错了"还是"库用错了"。这层测试把两者切开。
"""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

from src.config import EMBEDDING_DIM
from src.vectorstore.milvus_store import ChunkRow, MilvusStore

COLLECTION = "chunks"


def _vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(EMBEDDING_DIM)]


def _row(seed: int, pmc_id: str, text: str, section: str = "Results", idx: int = 0):
    return ChunkRow(
        vector=_vec(seed),
        text=text,
        pmc_id=pmc_id,
        section=section,
        chunk_index=idx,
        path=f"Methods / {section}",
        source_type="sec",
        text_hash=f"{seed:016x}",
    )


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

    hits = store.search(rows[0].vector, top_k=3)

    assert len(hits) == 3
    # 用第 0 行的向量去查，它自己必须排第一，且相似度为 1
    assert hits[0]["pmc_id"] == "PMC111"
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
    # 返回结构必须是抹平过的，上层不该看到 Milvus 的嵌套
    assert set(hits[0]) == {
        "id", "score", "text", "pmc_id", "section", "chunk_index", "path", "source_type",
    }


def test_existing_hashes_returns_only_those_in_store(store):
    store.insert([_row(1, "PMC1", "a"), _row(2, "PMC2", "b")])
    assert store.existing_hashes(["0000000000000001", "ffffffffffffffff"]) == {
        "0000000000000001"
    }


def test_existing_hashes_empty_input_returns_empty_set(store):
    """空列表必须提前返回 —— 空 filter 在 Milvus 里等于扫全库。"""
    assert store.existing_hashes([]) == set()


def test_ensure_collection_is_idempotent(store):
    store.ensure_collection()          # 第二次调用不该炸，也不该清空
    store.insert([_row(1, "PMC111", "x")])
    store.ensure_collection()
    assert store.count() == 1


def test_chunk_row_requires_every_field():
    """缺字段在**构造期**就报错，不再靠 insert 里的运行时校验（类型化边界的意义）。"""
    with pytest.raises(TypeError):
        ChunkRow(vector=_vec(1), text="x")        # 缺 pmc_id / path / text_hash …


def test_insert_rejects_wrong_dimension(store):
    row = replace(_row(1, "PMC111", "x"), vector=_vec(1)[:512])
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
