# src/agents/inquiry/symptom_normalizer.py

from __future__ import annotations

import json
from loguru import logger
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from neo4j import AsyncDriver
from pymilvus import MilvusClient

# ── 第一层：结构化输出模型 ────────────────────────────────────────────────
class SymptomsOutput(BaseModel):
    """LLM 结构化输出的 Schema。with_structured_output 会强制 LLM 按此格式返回。"""
    symptoms: list[str] = Field(
        description="从用户描述中提取并标准化后的医学症状术语列表，无症状时为空列表"
    )


# ── 第一层 Prompt ────────────────────────────────────────────────────────
SYMPTOM_EXTRACT_PROMPT = """你是医疗术语标准化专家。

任务：从用户的描述中提取所有症状，并将每个症状转换为标准医学术语。

标准化规则（不限于此，尽量标准化）：
- 发烧/烧/低烧/高烧 → 发热
- 肚子疼/肚痛/腹部疼痛/肚子不舒服 → 腹痛
- 头晕/头晕眼花/天旋地转 → 眩晕
- 喘不上气/憋气/气短/胸闷喘气 → 呼吸困难
- 拉肚子/跑肚/稀便/大便不成形 → 腹泻
- 心跳快/心慌/心跳加速/心跳不规律 → 心悸
- 浑身没劲/没力气/疲惫/全身乏力 → 乏力
- 嗓子疼/喉咙疼/咽喉痛 → 咽痛
- 胸口疼/胸部疼痛/前胸痛 → 胸痛
- 恶心想吐/想呕吐/胃部不适 → 恶心
- 头疼/头部疼痛/偏头痛 → 头痛
- 流鼻涕/鼻涕/鼻塞流涕 → 流涕
- 咳嗽/干咳/咳痰 → 咳嗽

用户描述：{user_input}

请提取所有症状并标准化后填入 symptoms 字段。如果没有明确症状，symptoms 填空列表。"""


async def extract_and_normalize_symptoms(
    user_input: str,
    llm: BaseChatModel,
) -> list[str]:
    """
    第一层：LLM 提取 + 标准化。
    使用 with_structured_output 强制 LLM 按 SymptomsOutput Schema 返回，
    无需手动解析 JSON，彻底消除格式错误风险。
    """
    structured_llm = llm.with_structured_output(SymptomsOutput)
    prompt = SYMPTOM_EXTRACT_PROMPT.format(user_input=user_input)
    try:
        result: SymptomsOutput = await structured_llm.ainvoke(
            [SystemMessage(content=prompt)]
        )
        symptoms = [s.strip() for s in result.symptoms if s.strip()]
        logger.debug(f"LLM 提取症状: {symptoms}")
        return symptoms
    except Exception as e:
        # with_structured_output 在极少数情况下仍可能失败（如模型不支持 function calling）
        # 此时静默降级，返回空列表，由后续层处理
        logger.warning(f"LLM 结构化输出失败，降级为空列表: {e}")
        return []


async def match_symptoms_in_neo4j(
    symptoms: list[str],
    neo4j_driver: AsyncDriver,
) -> tuple[list[str], list[str]]:
    """第二层：Neo4j 精确匹配。返回 (命中列表, 未命中列表)。"""
    if not symptoms:
        return [], []
    async with neo4j_driver.session() as session:
        result = await session.run(
            "MATCH (s:Symptom) WHERE s.name IN $names RETURN s.name AS name",
            names=symptoms,
        )
        records = await result.data()
        matched_set = {r["name"] for r in records}
    matched = [s for s in symptoms if s in matched_set]
    unmatched = [s for s in symptoms if s not in matched_set]
    logger.debug(f"Neo4j 精确匹配: 命中={matched}, 未命中={unmatched}")
    return matched, unmatched


SIMILARITY_THRESHOLD = 0.85  # 低于此值视为真正的图谱外症状


async def semantic_match_symptoms(
    unmatched_symptoms: list[str],
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> tuple[dict[str, str], list[str]]:
    """
    第三层：Milvus 语义相似度兜底。
    返回 (mapped={用户原词: 图谱标准词}, still_unmatched=真正图谱外症状)。
    """
    if not unmatched_symptoms:
        return {}, []
    # 批量向量化，减少 API 调用次数
    query_embeddings = await embedding_model.aembed_documents(unmatched_symptoms)
    mapped: dict[str, str] = {}
    still_unmatched: list[str] = []
    for symptom, query_vec in zip(unmatched_symptoms, query_embeddings):
        try:
            results = milvus_client.search(
                collection_name="symptom_index",
                data=[query_vec],
                limit=1,
                output_fields=["name"],
            )
            if results and results[0]:
                top_hit = results[0][0]
                score = top_hit["distance"]  # COSINE 相似度，越高越相似
                if score >= SIMILARITY_THRESHOLD:
                    std_name = top_hit["entity"]["name"]
                    mapped[symptom] = std_name
                    logger.debug(f"语义映射: '{symptom}' → '{std_name}' (score={score:.3f})")
                else:
                    still_unmatched.append(symptom)
            else:
                still_unmatched.append(symptom)
        except Exception as e:
            logger.warning(f"Milvus 语义匹配失败 '{symptom}': {e}")
            still_unmatched.append(symptom)
    return mapped, still_unmatched



#组装三层管线工作流
async def normalize_symptoms(
    user_input: str,
    llm: BaseChatModel,
    neo4j_driver: AsyncDriver,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> dict:
    """
    完整三层症状标准化流水线入口。

    Returns:
        matched      : 直接命中 Neo4j 的标准症状列表
        mapped       : {用户原词: 图谱标准词}，语义兜底后的映射
        unmatched    : 真正的图谱外症状（供医生参考）
        all_standard : matched + mapped.values()，用于后续 Neo4j 查询
    """
    normalized = await extract_and_normalize_symptoms(user_input, llm)
    if not normalized:
        return {"matched": [], "mapped": {}, "unmatched": [], "all_standard": []}

    matched, unmatched_after_exact = await match_symptoms_in_neo4j(normalized, neo4j_driver)
    mapped, still_unmatched = await semantic_match_symptoms(
        unmatched_after_exact, embedding_model, milvus_client
    )
    all_standard = matched + list(mapped.values())
    return {
        "matched": matched,
        "mapped": mapped,
        "unmatched": still_unmatched,
        "all_standard": all_standard,
    }


# ── 反向处理：标准术语 → 口语（追问时使用） ──────────────────────────────
SYMPTOM_HUMANIZE_PROMPT = """将以下医学症状术语转换为患者容易理解的口语表达。

症状列表：{symptoms}

要求：
- 简洁易懂，避免专业术语
- 可以加括号补充解释，帮助患者理解
- 输出 JSON 数组，与输入顺序一一对应

示例：
输入：["发热", "心悸", "呼吸困难"]
输出：["发烧", "心跳加速或心慌", "喘不上气或胸闷"]"""


async def humanize_symptoms(
    symptoms: list[str],
    llm: BaseChatModel,
) -> list[str]:
    """
    将标准医学术语转换为患者易懂的口语（用于追问时的友好表达）。
    失败时静默降级，直接返回原术语，不影响主流程。
    """
    if not symptoms:
        return []
    prompt = SYMPTOM_HUMANIZE_PROMPT.format(
        symptoms=json.dumps(symptoms, ensure_ascii=False)
    )
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content.strip())
        # 长度不一致说明 LLM 输出有问题，退回原词
        return result if len(result) == len(symptoms) else symptoms
    except Exception:
        return symptoms