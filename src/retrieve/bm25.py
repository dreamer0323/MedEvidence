"""BM25 关键词检索：混合检索里的"稀疏"一路。

设计（见 plan/09「混合检索」）：

  · 用 `rank_bm25` 在**进程内**建索引 —— 不引入 Elasticsearch（2C2G 跑不动 JVM），
    也不改 Milvus schema（改 schema 要重建 collection，等于重新 embedding 2.6 万块）。
  · 索引缓存到 `data/bm25_index.pkl`；集合行数变了就重建。
  · 本模块只负责 `query → [(pk, score)]`，取回 Hit 的活交给 `hybrid`。

实测（2026-10-27，26,230 块）：token 6.5M，分词 32.5s + 建索引 13.5s，峰值内存 515 MB。
⚠️ 2C2G 服务器上跑不了；v0 走本地部署。上 2G 机器的替代方案是 SQLite FTS5
（同算法、磁盘 106 MB、建索引 6.6s、查询 120-210ms），见 plan/09 的部署 fallback。

分词：小写 → 只留字母数字（连字符被拆开）→ 去停用词。
  · `Whole-Slide` → `whole` `slide`；`LH510` 保持完整 —— 这正是要 BM25 救的实体名。
  · 不做词干化：医学术语（`staining` / `stained`）分开影响不大，
    而词干化会把 `HER2` 这类缩写搞乱。
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import DATA_DIR
from src.vectorstore import MilvusStore

INDEX_PATH = DATA_DIR / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 只去掉最高频的功能词。BM25 的 IDF 本来就会压低它们，去掉主要是省内存
# （2.6 万块 × 平均 247 token，停用词约占 30-40%）。
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "these", "they", "this", "those", "to", "was", "were",
    "we", "with",
})


def tokenize(text: str) -> list[str]:
    """英文/数字分词，去停用词。**查询和文档必须用同一个函数**。"""
    tokens: list[str] = []

    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        tokens.append(token)

    return tokens


class BM25Index:
    """进程内 BM25 索引。`pks[i]` 和 `tokenized_docs[i]` 一一对应。"""

    def __init__(self, pks: list[str], tokenized_docs: list[list[str]]) -> None:
        self.pks = pks
        self.tokenized_docs = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs)

    @classmethod
    def load_or_build(cls, store: MilvusStore) -> "BM25Index":
        """有缓存且行数没变就复用；否则重建并写缓存。"""
        total = store.count()
        cached = cls._load_cache()

        if cached is not None and cached["count"] == total:
            return cls(cached["pks"], cached["docs"])

        index = cls._build(store)
        index._save_cache(total)
        return index

    @classmethod
    def _build(cls, store: MilvusStore) -> "BM25Index":
        pks: list[str] = []
        docs: list[list[str]] = []

        offset = 0
        page_size = 16384

        while True:
            rows = store.client.query(
                collection_name=store.collection,
                filter="pk != ''",
                output_fields=["pk", "text"],
                limit=page_size,
                offset=offset,
            )
            if not rows:
                break

            for row in rows:
                pks.append(row["pk"])
                docs.append(tokenize(row["text"]))

            offset += len(rows)

        return cls(pks, docs)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """返回 [(pk, bm25_score)]，按分数降序。"""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[tuple[str, float]] = []
        for i in order:
            results.append((self.pks[i], float(scores[i])))

        return results

    @staticmethod
    def _load_cache() -> dict | None:
        if not INDEX_PATH.exists():
            return None
        with INDEX_PATH.open("rb") as f:
            return pickle.load(f)

    def _save_cache(self, count: int) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"count": count, "pks": self.pks, "docs": self.tokenized_docs}
        with INDEX_PATH.open("wb") as f:
            pickle.dump(payload, f)
