"""Embedding 客户端测试。

分两档：
  - 不联网的：校验逻辑（空串拦截、顺序对齐）—— 用假 config，永远能跑
  - 联网的：真调一次 API —— 没配密钥就 skip，不阻塞本地开发
"""
from __future__ import annotations

import os

import pytest

from src.config import EMBEDDING_DIM, EmbeddingConfig
from src.embedding.client import EmbeddingClient


class _FakeEmbedding:
    """最小可用的假 embedding 服务：按入参顺序返回固定向量。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]):
        self.calls.append(list(input))
        data = [_Item(i, [0.01 * (i + 1)] * EMBEDDING_DIM) for i in range(len(input))]
        return _Resp(data)


class _Item:
    def __init__(self, index: int, embedding: list[float]):
        self.index = index
        self.embedding = embedding


class _Resp:
    def __init__(self, data):
        self.data = data


@pytest.fixture
def offline_client(monkeypatch):
    """构造一个不发请求的 client（base_url 指向本地黑洞也没关系，因为被替换了）。"""
    cfg = EmbeddingConfig(
        api_key="fake-key",
        base_url="http://127.0.0.1:1/v1",
        model="fake-model",
        provider="fake",
        batch_size=2,
    )
    client = EmbeddingClient(cfg)
    fake = _FakeEmbedding()
    monkeypatch.setattr(client._client, "embeddings", fake)
    return client, fake


def test_empty_input_returns_empty(offline_client):
    client, fake = offline_client
    assert client.embed([]) == []
    assert fake.calls == []


def test_blank_text_is_rejected_with_index(offline_client):
    """空串会被 API 拒掉，必须提前拦并报出下标 —— 否则万条数据里没法定位。"""
    client, _ = offline_client
    with pytest.raises(ValueError, match="第 1 条"):
        client.embed(["正常文本", "   "])


def test_auto_batching_and_order(offline_client):
    """batch_size=2、5 条输入 → 3 次请求，且向量与文本顺序严格对齐。"""
    client, fake = offline_client
    texts = [f"文本{i}" for i in range(5)]

    vectors = client.embed(texts)

    assert len(vectors) == 5
    assert [len(c) for c in fake.calls] == [2, 2, 1]
    # 假服务按 batch 内下标生成向量，第 0 条永远是 0.01 起头的那个
    assert vectors[0][0] == pytest.approx(0.01)
    assert vectors[2][0] == pytest.approx(0.01)


HAS_KEY = bool(os.getenv("EMBEDDING_API_KEY", "").strip())


@pytest.mark.skipif(not HAS_KEY, reason="未配置 EMBEDDING_API_KEY，跳过联网测试")
def test_real_api_returns_expected_dimension():
    """真调一次，确认模型名、base_url、维度三件事都对上。"""
    vec = EmbeddingClient().embed_one("乳腺浸润性导管癌 免疫组化 判读")
    assert len(vec) == EMBEDDING_DIM
