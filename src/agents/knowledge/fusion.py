# src/agents/knowledge/fusion.py

from __future__ import annotations
import asyncio
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.knowledge.prompts import FUSION_PROMPT
from src.agents.knowledge.doc_rag import search_docs_raw, format_doc_context
from src.agents.knowledge.graph_rag import search_graph_raw
from src.agents.knowledge.hallucination_check import check_hallucination

import json


async def multi_channel_search(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    neo4j_driver: AsyncDriver,
    db_session: AsyncSession | None = None,
    channels: list[str] | None = None,
    role: str = "patient",
) -> str:
    """
    多通道并行检索 → 结果融合 → 幻觉检测 → 返回最终回答。
    channels: 指定使用哪些通道 ["doc_rag", "graph_rag", "nl2sql"]，默认 doc_rag + graph_rag
    """
    if channels is None:
        channels = ["doc_rag", "graph_rag"] #默认文档和图知识库两路并行查询

    tasks = {}
    if "doc_rag" in channels:
        tasks["doc_rag"] = search_docs_raw(
            question, embedding_model, milvus_client,
        )
    if "graph_rag" in channels:
        tasks["graph_rag"] = search_graph_raw(
            question, neo4j_driver, llm,
        )
    if "nl2sql" in channels and db_session:
        from src.agents.knowledge.nl2sql import search_sql
        tasks["nl2sql"] = search_sql(question, llm, db_session)

    results = {}
    gathered = await asyncio.gather(
        *tasks.values(), return_exceptions=True,
    )
    for key, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            logger.warning(f"通道 {key} 检索失败: {result}")
            results[key] = None
        else:
            results[key] = result

    source_parts = []
    evidence_parts = []

    doc_hits = results.get("doc_rag")
    if doc_hits:
        ctx = format_doc_context(doc_hits)
        source_parts.append(f"### 文档检索结果\n{ctx}")
        evidence_parts.append(ctx[:1000])

    graph_records = results.get("graph_rag")
    if graph_records:
        graph_str = json.dumps(graph_records, ensure_ascii=False, indent=2)
        source_parts.append(f"### 知识图谱检索结果\n{graph_str}")
        evidence_parts.append(graph_str[:1000])

    sql_answer = results.get("nl2sql")
    if sql_answer and isinstance(sql_answer, str):
        source_parts.append(f"### 运营数据查询结果\n{sql_answer}")
        evidence_parts.append(sql_answer[:1000])

    if not source_parts:
        return "所有检索通道均未找到与您问题相关的信息。"

    sources = "\n\n".join(source_parts)
    prompt = FUSION_PROMPT.format(
        question=question, sources=sources, role=role,
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    answer = response.content

    evidence = "\n".join(evidence_parts)
    #幻觉检测
    hal_result = await check_hallucination(question, evidence, answer, llm)
    if not hal_result["is_grounded"]:
        claims = "、".join(hal_result.get("unsupported_claims", []))
        answer += f"\n\n⚠️ 提示：以下内容未在检索结果中找到充分依据，请谨慎参考：{claims}"

    return answer