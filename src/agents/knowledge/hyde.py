# src/agents/knowledge/hyde.py

from __future__ import annotations
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.agents.knowledge.prompts import HYDE_PROMPT


async def generate_hyde_embedding(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
) -> list[float]:
    """
    HyDE（Hypothetical Document Embeddings）：
    1. LLM 生成假设性回答
    2. 对假设回答做向量化
    3. 用假设回答的向量去检索（比原始问题的向量召回率更高）
    """
    prompt = HYDE_PROMPT.format(question=question)
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt)])
        hypothetical_doc = response.content.strip()
        logger.debug(f"HyDE 假设文档: {hypothetical_doc[:100]}...")
        return await embedding_model.aembed_query(hypothetical_doc)
    except Exception as e:
        logger.warning(f"HyDE 生成失败，回退到原始查询向量: {e}")
        return await embedding_model.aembed_query(question)