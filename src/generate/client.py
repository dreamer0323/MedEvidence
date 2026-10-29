"""DeepSeek 生成：Prompt → 答案 + 引用体检。

这一步是**样板**：真正有判断的在 `prompt.py`（证据怎么编号、答案怎么校验）。
这里只保证三件事：空证据不花钱、temperature=0、超时有上限。
"""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.config import LLMConfig, get_llm_config
from src.generate.prompt import (
    SYSTEM_PROMPT,
    CitationReport,
    build_prompt,
    verify_citations,
)
from src.retrieve.search import Hit

# 检索为空时的固定拒答。**不调模型** —— 空检索是确定性事实，不需要 LLM 判断。
# 这条同时也是第 11 周降级链路的起点（见 plan/09）。
NO_EVIDENCE_ANSWER = "现有语料中没有找到相关证据，无法回答这个问题。"


@dataclass(frozen=True)
class Answer:
    text: str
    citations: CitationReport
    model: str


def _make_client(cfg: LLMConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60.0)


def generate(
    question: str,
    evidence: list[Hit],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
) -> Answer:
    """检索结果 → 答案。

    client / model 可注入：单测传桩，不读 .env、不调 API、不花钱。
    两个都传时完全不碰配置层。
    """
    if not evidence:
        return Answer(
            text=NO_EVIDENCE_ANSWER,
            citations=verify_citations("", []),
            model=model or "",
        )

    cfg: LLMConfig | None = None    # 配置层
    if client is None or model is None:
        cfg = get_llm_config()
    if client is None:
        client = _make_client(cfg)
    model = model or cfg.model

    resp = client.chat.completions.create( 
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, evidence)},
        ],
        # 0 不是为了"更聪明"，是为了**可复现**：评测集要比较不同检索策略，
        # 采样噪声会让 A/B 的差异分不清是策略带来的还是随机带来的。
        temperature=0.0,
    )
    text = (resp.choices[0].message.content or "").strip()
    return Answer(text=text, citations=verify_citations(text, evidence), model=model)
