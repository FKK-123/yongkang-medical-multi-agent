# src/agents/knowledge/prescription_review.py

from __future__ import annotations
import asyncio
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

from src.agents.knowledge.prompts import (
    PRESCRIPTION_PARSE_PROMPT, PRESCRIPTION_REPORT_PROMPT,
)
from src.agents.knowledge.doc_rag import search_docs_raw, format_doc_context
from src.agents.knowledge.graph_rag import search_graph_raw


async def _parse_prescription(question: str, llm: BaseChatModel) -> dict:
    """从自然语言中提取处方结构化信息。"""
    prompt = PRESCRIPTION_PARSE_PROMPT.format(question=question)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        return json.loads(content)
    except Exception as e:
        logger.warning(f"处方解析失败: {e}")
        return {"drugs": [], "patient_info": {"allergies": [], "diseases": []}}


async def _check_dosage(
    drug_name: str,
    dosage: str | None,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> dict:
    """剂量校验：查说明书推荐剂量范围。"""
    query = f"{drug_name} 用法用量 推荐剂量"
    hits = await search_docs_raw(
        query, embedding_model, milvus_client,
        top_k=5, rerank_top_k=2, doc_type="drug_instruction",
    )
    context = format_doc_context(hits) if hits else "未找到说明书"
    return {
        "type": "dosage",
        "drug": drug_name,
        "input_dosage": dosage,
        "reference": context[:500],
        "source": hits[0]["doc_name"] if hits else "无",
    }


async def _check_interaction(
    drug_names: list[str],
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> dict:
    """配伍校验：查药物间相互作用。"""
    if len(drug_names) < 2:
        return {"type": "interaction", "drugs": drug_names, "result": "单药无需配伍校验", "records": []}

    query = f"{'、'.join(drug_names)} 之间是否有药物相互作用或配伍禁忌"
    records = await search_graph_raw(query, neo4j_driver, llm)
    return {
        "type": "interaction",
        "drugs": drug_names,
        "records": records[:10],
        "result": json.dumps(records[:10], ensure_ascii=False) if records else "知识图谱中未发现配伍禁忌记录",
    }


async def _check_allergy(
    drug_names: list[str],
    allergies: list[str],
    neo4j_driver: AsyncDriver,
    llm: BaseChatModel,
) -> dict:
    """
    过敏校验（成分级匹配）：
    1. 字符串匹配：药品名 vs 过敏史
    2. GraphRAG 成分级匹配：查药品所属类别，与过敏史交叉比对
    """
    conflicts = []
    allergy_set = {a.lower() for a in allergies}

    for drug in drug_names:
        if drug.lower() in allergy_set:
            conflicts.append({"drug": drug, "allergy": drug, "level": "exact_match"})
            continue
        for allergy in allergies:
            if allergy.lower() in drug.lower() or drug.lower() in allergy.lower():
                conflicts.append({"drug": drug, "allergy": allergy, "level": "name_match"})
                break

    if allergies and drug_names:
        query = (
            f"查询以下药品的药物类别和成分信息：{'、'.join(drug_names)}。"
            f"患者对以下物质过敏：{'、'.join(allergies)}。"
            f"判断是否存在交叉过敏风险。"
        )
        records = await search_graph_raw(query, neo4j_driver, llm)
        if records:
            for r in records[:10]:
                r_str = json.dumps(r, ensure_ascii=False).lower()
                for allergy in allergies:
                    if allergy.lower() in r_str:
                        drug_in_record = next(
                            (d for d in drug_names if d.lower() in r_str), None,
                        )
                        if drug_in_record and not any(
                            c["drug"] == drug_in_record for c in conflicts
                        ):
                            conflicts.append({
                                "drug": drug_in_record,
                                "allergy": allergy,
                                "level": "component_match",
                                "evidence": json.dumps(r, ensure_ascii=False)[:200],
                            })

    return {
        "type": "allergy",
        "allergies": allergies,
        "conflicts": conflicts,
        "result": (
            f"发现过敏风险：{json.dumps(conflicts, ensure_ascii=False)}"
            if conflicts else "未发现过敏冲突"
        ),
    }


async def _check_duplicate(drug_list: list[dict]) -> dict:
    """重复用药检查：同一处方中是否有同类药。"""
    categories = {}
    for d in drug_list:
        name = d.get("name", "")
        if name:
            categories.setdefault(name, []).append(name)
    duplicates = {k: v for k, v in categories.items() if len(v) > 1}
    return {
        "type": "duplicate",
        "duplicates": duplicates,
        "result": f"重复用药：{duplicates}" if duplicates else "未发现重复用药",
    }


async def review_prescription(
    question: str,
    llm: BaseChatModel,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    neo4j_driver: AsyncDriver,
) -> str:
    """
    处方审核完整流程：
    解析处方 → 并行校验（剂量/配伍/过敏/重复）→ LLM 生成审核报告。
    """
    prescription = await _parse_prescription(question, llm)
    drugs = prescription.get("drugs", [])
    patient = prescription.get("patient_info", {})

    if not drugs:
        return "未能从您的描述中识别出具体药品信息，请提供药品名称、剂量等详细信息。"

    drug_names = [d["name"] for d in drugs if d.get("name")]
    allergies = patient.get("allergies", []) or []

    tasks = [
        _check_interaction(drug_names, neo4j_driver, llm),
        _check_allergy(drug_names, allergies, neo4j_driver, llm),
        _check_duplicate(drugs),
    ]
    for d in drugs:
        if d.get("name"):
            tasks.append(_check_dosage(
                d["name"], d.get("dosage"),
                embedding_model, milvus_client,
            ))

    check_results = await asyncio.gather(*tasks)

    results_str = json.dumps(
        list(check_results), ensure_ascii=False, indent=2, default=str,
    )
    prescription_str = json.dumps(prescription, ensure_ascii=False, indent=2)

    prompt = PRESCRIPTION_REPORT_PROMPT.format(
        prescription=prescription_str, check_results=results_str,
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content