# src/agents/inquiry/neo4j_queries.py

from __future__ import annotations
from loguru import logger
from neo4j import AsyncDriver
from src.agents.inquiry.state import CandidateDisease


async def query_candidate_diseases(
    confirmed_symptoms: list[str],
    neo4j_driver: AsyncDriver,
    top_k: int = 10,
) -> list[CandidateDisease]:
    """
    根据已确认症状列表，从 Neo4j 查询候选疾病。
    按基础置信度（命中症状数 / 该疾病总症状数）降序排列，取 Top K。
    """
    if not confirmed_symptoms:
        return []

    cypher = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE s.name IN $confirmed_symptoms
    WITH d, collect(s.name) AS matched_symptoms, count(s) AS matched_count
    MATCH (d)-[:HAS_SYMPTOM]->(all_s:Symptom)
    WITH d, matched_symptoms, matched_count, count(all_s) AS total_symptoms
    ORDER BY toFloat(matched_count) / total_symptoms DESC
    LIMIT $top_k
    RETURN
        d.name AS disease,
        matched_symptoms,
        matched_count,
        total_symptoms,
        toFloat(matched_count) / total_symptoms AS base_confidence
    """

    async with neo4j_driver.session() as session:
        result = await session.run(
            cypher,
            confirmed_symptoms=confirmed_symptoms,
            top_k=top_k,
        )
        records = await result.data()

    candidates = []
    for r in records:
        candidates.append(CandidateDisease(
            name=r["disease"],
            base_confidence=round(r["base_confidence"], 4),
            confidence=round(r["base_confidence"], 4),  # 初始值，后续加权调整
            matched_symptoms=r["matched_symptoms"],
            all_symptoms=[],   # 由 enrich_candidate_details 补充
            department="",
            checks=[],
            complications=[],
        ))

    logger.debug(f"Neo4j 候选疾病: {[c.name for c in candidates]}")
    return candidates


async def enrich_candidate_details(
    candidates: list[CandidateDisease],
    neo4j_driver: AsyncDriver,
) -> list[CandidateDisease]:
    """
    补充候选疾病的完整信息：全部症状、建议科室、建议检查、并发症。
    在首次查到候选疾病后调用一次，后续轮次复用缓存在 state 里的数据。
    """
    if not candidates:
        return candidates

    disease_names = [c.name for c in candidates]

    # 查全部症状
    symptoms_cypher = """
    MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(s.name) AS symptoms
    """
    # 查科室
    dept_cypher = """
    MATCH (d:Disease)-[:BELONGS_TO]->(dept:Department)
    WHERE d.name IN $names
    RETURN d.name AS disease, dept.name AS department
    """
    # 查检查项目
    check_cypher = """
    MATCH (d:Disease)-[:NEED_CHECK]->(c:Check)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(c.name) AS checks
    """
    # 查并发症
    comp_cypher = """
    MATCH (d:Disease)-[:ACOMPANY_WITH]->(comp:Disease)
    WHERE d.name IN $names
    RETURN d.name AS disease, collect(comp.name) AS complications
    """

    async with neo4j_driver.session() as session:
        symptoms_result = await (await session.run(symptoms_cypher, names=disease_names)).data()
        dept_result     = await (await session.run(dept_cypher,     names=disease_names)).data()
        check_result    = await (await session.run(check_cypher,    names=disease_names)).data()
        comp_result     = await (await session.run(comp_cypher,     names=disease_names)).data()

    # 构建查找字典
    symptoms_map    = {r["disease"]: r["symptoms"]      for r in symptoms_result}
    dept_map        = {r["disease"]: r["department"]    for r in dept_result}
    check_map       = {r["disease"]: r["checks"]        for r in check_result}
    comp_map        = {r["disease"]: r["complications"] for r in comp_result}

    for c in candidates:
        c.all_symptoms  = symptoms_map.get(c.name, [])
        c.department    = dept_map.get(c.name, "")
        c.checks        = check_map.get(c.name, [])
        c.complications = comp_map.get(c.name, [])

    return candidates


async def get_pending_symptoms(
    candidates: list[CandidateDisease],
    confirmed_symptoms: list[str],
    denied_symptoms: list[str],
    asked_symptoms: list[str],
) -> list[tuple[str, int]]:
    """
    从候选疾病的症状集合中，提取尚未确认/否认/追问过的症状。
    返回 [(症状名, 出现在几个候选疾病中)]，按出现次数升序排列。
    出现次数越少 = 区分度越高，优先追问。
    """
    already_known = set(confirmed_symptoms) | set(denied_symptoms) | set(asked_symptoms)
    symptom_count: dict[str, int] = {}
    for c in candidates:
        for s in c.all_symptoms:
            if s not in already_known:
                symptom_count[s] = symptom_count.get(s, 0) + 1

    # 区分度高（出现次数少）的排前面
    sorted_symptoms = sorted(symptom_count.items(), key=lambda x: x[1])
    return sorted_symptoms