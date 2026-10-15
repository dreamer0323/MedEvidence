"""Milvus Lite 封装（嵌入式向量库 —— 见 plan/09-决策记录.md 第 1 条）。"""
from src.vectorstore.milvus_store import COLLECTION_NAME, MilvusStore

__all__ = ["MilvusStore", "COLLECTION_NAME"]
