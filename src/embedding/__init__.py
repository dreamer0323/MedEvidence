"""BGE-M3 向量化封装（走 API，不本地部署 —— 见 plan/09-决策记录.md 第 3 条）。"""
from src.embedding.client import EmbeddingClient

__all__ = ["EmbeddingClient"]
