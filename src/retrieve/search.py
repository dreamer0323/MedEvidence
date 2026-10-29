"""在线检索：query → 向量 → Milvus → 带引用的命中列表。

这一步**不生成答案**（第 3 周才接生成），只把"证据"找回来。
设计取舍见 plan/09「检索模块」：

  · query 不加标题前缀 —— 标题是文档侧技巧，BGE-M3 是对称模型
  · k=10 是先验起点，不是结论；第 4 周用评测集调
  · 暂不设分数阈值，但留 `min_score` 参数位
  · 同一节的多个块不合并：合并会丢 chunk_index，引用变模糊
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embedding import EmbeddingClient
from src.vectorstore import MilvusStore

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class Hit:
    """一条检索结果 —— 上层（组装 prompt / 生成引用）只需要这些。"""

    text: str
    pmc_id: str
    section: str
    chunk_index: int
    path: str
    source_type: str
    score: float

    def cite(self) -> str:
        """引用标记的**唯一**格式来源，别在别处再拼一遍。"""
        return self.pmc_id if self.pmc_id.startswith("PMC") else f"PMC{self.pmc_id}"


def _to_hit(row: dict) -> Hit:
    return Hit(
        text=row["text"],
        pmc_id=row["pmc_id"],
        section=row["section"],
        chunk_index=row["chunk_index"],
        path=row["path"],
        source_type=row["source_type"],
        score=row["score"],
    )


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
    *,
    store: MilvusStore | None = None,
    embedder: EmbeddingClient | None = None,
) -> list[Hit]:
    """按自然语言查询返回 Top-K 证据块。

    store / embedder 可注入：单测用桩 + 临时 Milvus，不调 API、不花钱。
    不注入时按默认配置构造，并在结束时关闭自己打开的库连接。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空：空向量查询是 bug，不是合法输入")
    if top_k <= 0:
        raise ValueError(f"top_k 必须为正，收到 {top_k}")

    own_store = store is None
    store = store or MilvusStore()
    embedder = embedder or EmbeddingClient()

    try:
        store.ensure_collection()          # 已存在则 load（Lite 新进程是 released）
        vector = embedder.embed_one(query)
        rows = store.search(vector, top_k=top_k)
        hits = [_to_hit(row) for row in rows]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits
    finally:
        if own_store:
            store.close()
            
def describe(top_k: int = DEFAULT_TOP_K) -> str:
    """这一路检索的身份 —— 版本块里的 retriever 只能从这里取。"""
    return f"vector(top_k={top_k})"