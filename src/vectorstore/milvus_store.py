"""Milvus Lite 封装。

**为什么是 Lite 不是 standalone**：见 plan/09-决策记录.md 第 1 条（服务器可用内存
只剩 826Mi / 默认不开认证还暴露公网）。关键前提是 **Lite 与 standalone 的 API 完全
一致** —— 所以这个文件里不许出现任何 Lite 专有的写法，将来迁回 standalone 只改
`MILVUS_URI` 一个常量。

## 表结构（面试要能解释每一列为什么在）

    id          INT64    主键，auto_id      —— 用自增而不是 uuid：省空间，且写入顺序≈
                                               物理顺序，Lite 场景下扫描更快
    vector      FLOAT_VECTOR(1024)          —— BGE-M3 dense 输出
    text        VARCHAR(8192)               —— chunk 正文，检索后直接进 prompt
    pmc_id      VARCHAR(64)                 —— 溯源：回到 PMC 原文
    section     VARCHAR(256)                —— 溯源：命中的是 Introduction 还是 Results
    chunk_index INT64                       —— 溯源：同一篇内第几个 chunk

为什么存 pmc_id + section + chunk_index 而不是只存 pmc_id：
**这个项目的卖点是"答案带引用"**，而引用能不能指对，取决于检索结果里有没有
足够定位回原文的字段。三个字段拼起来 =「PMC12345 的 Results 第 2 块」，
面试官问"你怎么保证引用不指错"时，答案就在这张表里。

**不存全文**：全文在 data/raw/ 的 XML 里，向量库里重复存一遍是浪费 ——
而且会诱使你"检索到 chunk 后拿整篇喂给 LLM"，把 20 篇全文塞进上下文。

## 度量方式为什么用 COSINE

BGE-M3 的 dense 向量没做归一化，此时 COSINE 与 IP 在数学上等价；但 COSINE 对
"将来某天换了归一化的模型"这件事不敏感 —— 少一个隐含假设，少一类线上事故。
（如果要用 IP，就必须保证写入前做过 L2 归一化，这个假设很容易被后来的人破坏。）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pymilvus import DataType, MilvusClient

from src.config import EMBEDDING_DIM, MILVUS_URI

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"

# VARCHAR 上限 65535。取 8192 是个软约束：它逼着分块阶段就把长度卡住，
# 而不是等插入时炸在 Milvus 里。超长 chunk 会直接抛错，不会静默截断。
MAX_TEXT_LEN = 8192
MAX_ID_LEN = 64
MAX_SECTION_LEN = 256
# 完整章节路径实测最长拼出 1125 字符，留一倍余量
MAX_PATH_LEN = 2048
MAX_SOURCE_TYPE_LEN = 16
MAX_HASH_LEN = 64

# 检索默认带回的字段：text 进 prompt，其余用于生成引用标记
OUTPUT_FIELDS = ["text", "pmc_id", "section", "chunk_index", "path", "source_type"]

@dataclass(frozen=True)
class ChunkRow:
    """Milvus 一行的类型化契约 —— **不属于** parse / chunk 层。

    为什么要有这个类：`insert()` 以前收裸 dict，`path` 该传字符串还是列表
    只能靠运行时撞运气（曾经把 list 传进 VARCHAR 列，Milvus 才报错）。
    类型化之后，边界上的错在**构造那一刻**就暴露，而不是到建库中途。

    只做浅转换（`to_dict` 不复制 1024 维向量），避免每批多拷 4MB。
    """

    vector: list[float]
    text: str
    pmc_id: str
    section: str
    chunk_index: int
    path: str                       # " / ".join(path)，字符串不是列表
    source_type: str
    text_hash: str                  # sha256(chunk.embed_input)

    def to_dict(self) -> dict:
        """转成 pymilvus 要的 dict。浅拷贝：vector 传引用。"""
        return {
            "vector": self.vector,
            "text": self.text,
            "pmc_id": self.pmc_id,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "path": self.path,
            "source_type": self.source_type,
            "text_hash": self.text_hash,
        }


class MilvusStore:
    """Milvus Lite 的一层薄封装：建集合 / 插入 / 检索 / 计数。

    用法::

        with MilvusStore() as store:
            store.ensure_collection()
            store.insert(rows)
            hits = store.search(query_vector, top_k=20)
    """

    def __init__(
        self,
        uri: str = MILVUS_URI,
        collection: str = COLLECTION_NAME,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        self.uri = uri
        self.collection = collection
        self.dim = dim
        # ⚠️ Windows 上 Lite 会锁住文件：同一个 uri 开第二个 client 会报
        #    "database is locked"。所以别在循环里反复 new，复用一个实例，
        #    或者用完 close()。
        self.client = MilvusClient(uri=uri)

    # ------------------------------------------------------------ 建表

    def ensure_collection(self, drop: bool = False) -> None:
        """建集合 + 索引。已存在则跳过（除非 drop=True 重建）。

        drop=True 是**重建库**的唯一正确姿势。注意它不是"清空数据"：
        重新建集合后必须重新灌数据，用之前先想清楚。
        """
        if drop and self.has_collection():
            logger.warning("drop 集合 %s（数据会全部丢失）", self.collection)
            self.client.drop_collection(self.collection)

        if self.has_collection():
            logger.info("集合 %s 已存在，跳过建表", self.collection)
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("text", DataType.VARCHAR, max_length=MAX_TEXT_LEN)
        schema.add_field("pmc_id", DataType.VARCHAR, max_length=MAX_ID_LEN)
        schema.add_field("section", DataType.VARCHAR, max_length=MAX_SECTION_LEN)
        schema.add_field("chunk_index", DataType.INT64)
        # 溯源：完整章节路径（实测最长 1125 字符）与来源类型（sec / abstract）
        schema.add_field("path", DataType.VARCHAR, max_length=MAX_PATH_LEN)
        schema.add_field("source_type", DataType.VARCHAR, max_length=MAX_SOURCE_TYPE_LEN)
        # 增量建库的内容指纹：哈希对象是 chunk.embed_input（见 plan/09）
        # 不建标量索引：2.8 万条的量级，过滤查询全扫也不过几十毫秒
        schema.add_field("text_hash", DataType.VARCHAR, max_length=MAX_HASH_LEN)

        index_params = self.client.prepare_index_params()
        # AUTOINDEX 让 Milvus 按数据量自己挑索引算法。
        # 1000 篇这个量级（约万级 chunk）连 HNSW 都跑不满，手工调索引参数
        # 属于自娱自乐 —— 记进决策记录时这条要写清楚"为什么不做"。
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info("集合 %s 已建好（dim=%d, metric=COSINE）", self.collection, self.dim)

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection)

    # ------------------------------------------------------------ 写入

    def insert(self, rows: list[ChunkRow], *, batch_size: int = 1000) -> int:
        """写入 chunk 行，返回写入条数。

        字段是否存在由 `ChunkRow` 在构造时保证；这里只校验**长度和维度**——
        这些是数据本身的问题，Milvus 的报错对应不回你的数据源。
        """
        if not rows:
            return 0

        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            for j, row in enumerate(batch):
                if len(row.vector) != self.dim:
                    raise ValueError(
                        f"第 {i + j} 行向量维度 {len(row.vector)} != {self.dim}"
                    )
                if len(row.text) > MAX_TEXT_LEN:
                    raise ValueError(
                        f"第 {i + j} 行 text 长度 {len(row.text)} 超过 {MAX_TEXT_LEN}。"
                        f"该回去改分块参数，不是在写入端截断。"
                    )
                if len(row.path) > MAX_PATH_LEN:
                    raise ValueError(
                        f"第 {i + j} 行 path 长度 {len(row.path)} 超过 {MAX_PATH_LEN}"
                    )
                if len(row.source_type) > MAX_SOURCE_TYPE_LEN:
                    raise ValueError(f"第 {i + j} 行 source_type 不合法：{row.source_type}")
                if len(row.text_hash) > MAX_HASH_LEN:
                    raise ValueError(f"第 {i + j} 行 text_hash 长度超过 {MAX_HASH_LEN}")
            total += len(
                self.client.insert(self.collection, [r.to_dict() for r in batch])
            )

        logger.info("写入 %d 行到 %s", total, self.collection)
        return total

    def existing_hashes(self, hashes: list[str], *, batch_size: int = 1000) -> set[str]:
        """给定一批 text_hash，返回其中**已经在库里**的那些。

        增量建库的核心：在这个集合里的块，不再调用 embedding API（省的是真金白银）。

        TODO(human)：5-10 行。

        约束：
          · 用 `self.client.query(collection_name=..., filter=..., output_fields=["text_hash"],
            consistency_level="Strong")`
          · 空列表**提前返回空集合** —— 空 filter 等于扫全库
          · Milvus 的 `in` 表达式有长度上限，必须按 batch_size 分批
          · 表达式里的每个值都要带引号（hash 是十六进制、不含引号，但别依赖这个假设）
          · 返回 set，调用方要 O(1) 判断
        """
        

        if not hashes:
            return set()

        found: set[str] = set()
        for start in range(0, len(hashes), batch_size):
            part = hashes[start : start + batch_size]
            # 表达式里的字符串值必须带引号；hash 是十六进制，本身不含引号
            expr = "text_hash in [" + ", ".join(f'"{h}"' for h in part) + "]"
            rows = self.client.query(
                collection_name=self.collection,
                filter=expr, 
                output_fields=["text_hash"],# 只返回 text_hash，其他字段不返回
                consistency_level="Strong", # 强一致性：Milvus Lite 默认最终一致，刚灌完库立刻检索可能查不到，
            )
            found.update(row["text_hash"] for row in rows)
        return found

    # ------------------------------------------------------------ 检索

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        *,
        filter_expr: str = "",
    ) -> list[dict]:
        """向量检索，返回按相似度降序的扁平 dict 列表。

        返回结构刻意抹平了 Milvus 的嵌套：调用方（检索/重排链路）不该知道
        Milvus 的返回长什么样 —— 将来换库才不用改上层。
        """
        if len(query_vector) != self.dim:
            raise ValueError(f"查询向量维度 {len(query_vector)} != {self.dim}")

        raw = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr,
            output_fields=OUTPUT_FIELDS,
            # 强一致性：Milvus Lite 默认最终一致，刚灌完库立刻检索可能查不到，
            # 那是最容易被误判成"分块有问题"的一类 bug
            consistency_level="Strong",
        )
        if not raw:
            return []

        return [
            {
                "id": hit["id"],
                "score": hit["distance"],          # COSINE 相似度，越大越像
                "text": hit["entity"]["text"],
                "pmc_id": hit["entity"]["pmc_id"],
                "section": hit["entity"]["section"],
                "chunk_index": hit["entity"]["chunk_index"],
                "path": hit["entity"]["path"],
                "source_type": hit["entity"]["source_type"],
            }
            for hit in raw[0]
        ]

    # ------------------------------------------------------------ 维护

    def count(self) -> int:
        """当前集合里的 chunk 条数 —— 交付物「能 print 出条数」用的就是它。"""
        res = self.client.query(
            collection_name=self.collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(res[0]["count(*)"])

    def drop(self) -> None:
        if self.has_collection():
            self.client.drop_collection(self.collection)
            logger.warning("已删除集合 %s", self.collection)

    def close(self) -> None:
        """释放文件锁。Windows 上不关就删不掉、也开不了第二个。"""
        self.client.close()

    def __enter__(self) -> "MilvusStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


if __name__ == "__main__":
    # 自检：python -m src.vectorstore.milvus_store
    # 只建表 + 计数，不写入、不调 API —— 用来确认"库能连上"
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    store = MilvusStore()
    try:
        print(f"uri = {store.uri}")
        store.ensure_collection()
        print(f"集合 {store.collection} 存在，当前 {store.count()} 条")
    finally:
        store.close()
