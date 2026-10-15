"""BGE-M3 向量化客户端（SiliconFlow，OpenAI 兼容接口）。

这里只做三件事，多一件都不要加：

  1. 自动分批 —— 调用方丢 10 万条进来，不用关心 API 单次上限
  2. 失败重试 —— 网络抖动不该让整个建库任务从头再来
  3. 维度断言 —— 输出维度和 EMBEDDING_DIM 不一致时当场炸

第 3 条是重点。向量维度不匹配的代价不是"报错"，是**静默写入一坨检索不出
东西的数据**，等你发现时已经建完库了。
"""
from __future__ import annotations

import logging
import time

from openai import OpenAI

from src.config import (
    EMBEDDING_DIM,
    EmbeddingConfig,
    get_embedding_config,
)

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """把一批文本变成一批向量。

    用法::

        client = EmbeddingClient()
        vectors = client.embed(["乳腺癌 IHC 判读标准", "WSI 的染色差异"])
        assert len(vectors[0]) == EMBEDDING_DIM
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        # 缺 key 会在这里（构造时）就抛 ConfigError，而不是等发请求
        self.config = config or get_embedding_config()
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=timeout,
        )

    # ------------------------------------------------------------ 对外接口

    def embed(self, texts: list[str], *, desc: str = "embedding") -> list[list[float]]:
        """批量向量化，返回顺序与入参**严格一致**。

        desc 只用于日志，建库时传个"第 3 批 1000 篇"之类，好定位卡在哪。
        """
        if not texts:
            return []

        # 空串会被 API 直接拒掉，且在 10 万条里很难定位 → 提前拦，报出下标
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"{desc}: 第 {i} 条文本为空，API 会拒绝这类输入")

        vectors: list[list[float]] = []
        batch_size = self.config.batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for bi in range(total_batches):
            batch = texts[bi * batch_size : (bi + 1) * batch_size]
            logger.info("%s: 批次 %d/%d（%d 条）", desc, bi + 1, total_batches, len(batch))
            vectors.extend(self._embed_batch(batch))

        return vectors

    def embed_one(self, text: str) -> list[float]:
        """单条向量化 —— 在线检索链路用这个（query 只有一条）。"""
        return self.embed([text], desc="embed_query")[0]

    # ------------------------------------------------------------ 内部实现

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                )
            except Exception as err:  # noqa: BLE001 —— 见下方注释
                last_err = err
                # ⚠️ 这里把所有异常一视同仁地重试，是**建库脚本**的偷懒做法：
                #    建库是一次性任务，宁可多等几秒也不愿中途崩。
                #    400（参数错）和 429（限流）其实该分开处理 ——
                #    第 11 周的"降级链路"会把它换成分类重试 + 指数退避。
                if attempt == self.max_retries:
                    break
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "embedding 第 %d 次失败（%s），%ds 后重试", attempt, err, backoff
                )
                time.sleep(backoff)
                continue

            # 返回顺序不保证和入参一致，按 index 排回来 ——
            # 否则向量和文本错位，检索出来的引用会张冠李戴
            items = sorted(resp.data, key=lambda d: d.index)
            out = [item.embedding for item in items]

            if len(out) != len(batch):
                raise RuntimeError(f"API 返回 {len(out)} 条，请求了 {len(batch)} 条")
            for vec in out:
                if len(vec) != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"{self.config.model} 返回 {len(vec)} 维，"
                        f"但 EMBEDDING_DIM={EMBEDDING_DIM}。"
                        f"换模型就要同步改建集合时的 dimension，并重建库。"
                    )
            return out

        raise RuntimeError(f"embedding 重试 {self.max_retries} 次仍失败") from last_err


if __name__ == "__main__":
    # 自检：python -m src.embedding.client
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

    client = EmbeddingClient()
    t0 = time.time()
    vec = client.embed_one("乳腺浸润性导管癌的免疫组化判读")
    dt = time.time() - t0

    print(f"model      = {client.config.model}")
    print(f"维度        = {len(vec)}（期望 {EMBEDDING_DIM}）")
    print(f"前 4 维     = {[round(x, 4) for x in vec[:4]]}")
    print(f"单条耗时    = {dt * 1000:.0f} ms")
