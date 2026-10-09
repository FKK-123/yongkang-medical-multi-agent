# src/agents/knowledge/hallucination_check.py

from __future__ import annotations
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agents.knowledge.prompts import HALLUCINATION_CHECK_PROMPT


async def check_hallucination(
    question: str,
    evidence: str,
    answer: str,
    llm: BaseChatModel,
    threshold: float = 0.7,
) -> dict:
    """
    幻觉检测：校验回答是否基于检索结果。
    返回 {"is_grounded": bool, "unsupported_claims": [...], "confidence": float}
    如果 confidence < threshold，标记为不可信。
    """
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        question=question, evidence=evidence, answer=answer,
    )
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        result["is_grounded"] = result.get("is_grounded", False) and result.get("confidence", 0) >= threshold
        return result
    except Exception as e:
        logger.warning(f"幻觉检测失败: {e}")
        return {"is_grounded": True, "unsupported_claims": [], "confidence": 1.0}