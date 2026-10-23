"""集中配置层。

约定：**除本文件外，任何模块都不许直接读 os.environ。**
这样"配置从哪来"只有一个答案 —— 面试问"密钥怎么管的"能一句话答完，
本地 → 云服务器换环境也只改这一处。

两个设计取舍：

1. 路径常量在 import 时就算好：纯字符串运算，不会失败。
2. 密钥在**调用时**才校验（fail fast），不是 import 时 —— 否则一台没配 `.env`
   的机器连 `import src.config` 都会炸，单测都跑不起来。缺 key 应该在你真正
   要发请求的那一刻报错，而不是在导入阶段。
"""
from __future__ import annotations 

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import dotenv

# --------------------------------------------------------------- 路径

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # esearch / efetch 下来的原始 XML
PROCESSED_DIR = DATA_DIR / "processed"  # 解析分块后的 jsonl
LOG_DIR = PROJECT_ROOT / "log"

# 显式指定 .env 路径，不依赖"当前工作目录恰好是项目根"
# ⚠️ 必须在下面读 `MILVUS_URI` 之前加载，否则 .env 里的值读不到
dotenv.load_dotenv(PROJECT_ROOT / ".env", override=False)

# ⚠️ Milvus Lite 的 uri 传进来后，建出来的是一个**同名目录**，不是一个文件。
# 所以别用 `Path(MILVUS_URI).exists()` 判断"库建好没有"，用 has_collection()。
# 本地默认 data/medevidence.db；云端部署用环境变量指向服务器上的路径。
# Lite 是**嵌入式**的：建库进程必须和 .db 在同一台机器上，不能远程连。
MILVUS_URI = os.getenv("MILVUS_URI", "").strip() or str(DATA_DIR / "medevidence.db")

# --------------------------------------------------------------- 常量

# BGE-M3 的 dense 向量维度，写死在这里。
# 建集合时的 dimension 必须和 embedding 输出严格一致，不一致会直接插不进去 ——
# 写死是为了让不一致在**第一次插入**时就炸，而不是等到检索结果莫名其妙。
EMBEDDING_DIM = 1024

# 单次 embedding 请求塞几条文本。取小了慢、取大了可能被 API 拒。
# ⚠️ SiliconFlow 对 bge-m3 的真实批量上限尚未实测 → 记进 plan/04-怀疑清单.md
DEFAULT_EMBEDDING_BATCH_SIZE = 16


class ConfigError(RuntimeError):
    """配置缺失 / 不合法。带上变量名，别让人去猜。"""


def require_env(name: str, default: str | None = None) -> str:
    """取一个必填环境变量；没有就抛错，并且把"该做什么"写在错误里。"""
    value = os.getenv(name, default)
    if value is None or not value.strip():
        raise ConfigError(
            f"缺少环境变量 {name}。\n"
            f"  做法：cp .env.example .env，然后填上真实值（.env 已被 gitignore）"
        )
    return value.strip()


def optional_env(name: str, default: str) -> str:
    """取一个选填环境变量，**空字符串按未设置处理**。

    为什么必须这样：`.env.example` 里为了让人看见"有这项"，会写成
    `EMBEDDING_BATCH_SIZE=`（等号后面空的）。如果直接 `os.getenv(name, default)`，
    拿到的是 `""` 而不是 None，默认值不生效 —— 再 `int("")` 就炸了。
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _mask(secret: str) -> str:
    """日志/终端里只露尾巴，避免密钥跟着截图流出去。"""
    return "*" * 8 + secret[-4:] if len(secret) > 4 else "****"


# --------------------------------------------------------------- NCBI 抓取

# NCBI E-utilities 的"礼貌参数"：官方要求标识调用方（tool + email）。不填也能跑，
# 但被限流/封 IP 时你没有申诉依据。
# 有 API key → 10 req/s；没有 → 3 req/s。抓取脚本里的 sleep 应该**由这个速度推导**，
# 而不是随手写一个 0.4（写死了，换 key 那天就变成限流器）。
NCBI_EMAIL = optional_env("NCBI_EMAIL", "")
NCBI_API_KEY = optional_env("NCBI_API_KEY", "")
NCBI_REQUESTS_PER_SECOND = 10 if NCBI_API_KEY else 3
NCBI_MIN_INTERVAL = 1.0 / NCBI_REQUESTS_PER_SECOND


# --------------------------------------------------------------- 配置对象

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    provider: str


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    provider: str
    batch_size: int


def get_llm_config() -> LLMConfig:
    return LLMConfig(
        api_key=require_env("LLM_API_KEY"),
        base_url=require_env("LLM_BASE_URL"),
        model=require_env("LLM_MODEL_NAME"),
        provider=optional_env("LLM_PROVIDER", "deepseek"),
    )


def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        api_key=require_env("EMBEDDING_API_KEY"),
        base_url=require_env("EMBEDDING_BASE_URL"),
        model=require_env("EMBEDDING_MODEL_NAME"),
        provider=optional_env("EMBEDDING_PROVIDER", "siliconflow"),
        batch_size=int(optional_env("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))),
    )


# --------------------------------------------------------------- 杂项

def ensure_dirs() -> None:
    """建好所有输出目录。离线建库前调一次即可，幂等。"""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO, log_file: str | None = "medevidence.log") -> None:
    """统一日志：控制台 + 文件。

    文件默认落在 log/ 下（已被 gitignore）。多个入口脚本重复调用是安全的。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        ensure_dirs()
        handlers.append(logging.FileHandler(LOG_DIR / log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,          # 允许重复调用覆盖，省得各入口自己判断
    )


if __name__ == "__main__":
    # 自检：python -m src.config
    # 目的是"一眼看出环境对不对"，所以密钥只打尾巴。
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"MILVUS_URI   = {MILVUS_URI}")
    print(f"EMBEDDING_DIM= {EMBEDDING_DIM}")
    print(f"NCBI 频率上限 = {NCBI_REQUESTS_PER_SECOND} req/s（间隔 {NCBI_MIN_INTERVAL:.2f}s）")
    print("-" * 46)
    for name in (
        "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME", "LLM_PROVIDER",
        "EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL_NAME",
        "EMBEDDING_PROVIDER", "EMBEDDING_BATCH_SIZE",
        "NCBI_EMAIL", "NCBI_API_KEY",
    ):
        raw = os.getenv(name)
        if raw is None:
            shown = "❌ 未设置"
        elif "KEY" in name:
            shown = _mask(raw)
        else:
            shown = raw
        print(f"{name:24s} = {shown}")
