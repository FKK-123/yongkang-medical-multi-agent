# src/agents/inquiry/confidence.py

from __future__ import annotations
from src.agents.inquiry.state import CandidateDisease, PatientContext


def apply_context_weights(
    candidates: list[CandidateDisease],
    patient_context: PatientContext,
    denied_symptoms: list[str],
) -> list[CandidateDisease]:
    """
    在基础置信度上叠加上下文权重，返回重新排序后的候选疾病列表。

    权重规则（来自设计文档）：
      +0.15  用户有该疾病的既往病史（PostgreSQL medical_history）
      +0.10  长期记忆中有相关记录（Milvus long_term_memories）
      +0.05  用户年龄/性别与该疾病易感人群匹配（Patient 表 easy_get 字段）
      -0.20  用户明确否认该疾病的核心症状
             （核心症状 = 只属于该疾病、不属于其他候选疾病的独有症状）
    """
    # 计算每个症状出现在几个候选疾病中（用于判断"核心症状"）
    symptom_disease_count: dict[str, int] = {}
    for c in candidates:
        for s in c.all_symptoms:
            symptom_disease_count[s] = symptom_disease_count.get(s, 0) + 1

    for c in candidates:
        score = c.base_confidence

        # +0.15 既往病史命中
        for history_item in patient_context.medical_history:
            if c.name in history_item:
                score += 0.15
                break

        # +0.10 长期记忆命中
        for memory in patient_context.long_term_memories:
            if c.name in memory:
                score += 0.10
                break

        # +0.05 易感人群匹配（简化实现：暂用年龄段粗匹配，后续可接 easy_get 字段）
        # 此处预留接口，实际匹配逻辑需结合 Disease.easy_get 字段内容
        # score += 0.05  # 暂时注释，等 easy_get 字段接入后启用

        # -0.20 核心症状被否认
        # 核心症状 = 只属于该疾病（symptom_disease_count == 1）且被用户否认
        for s in c.all_symptoms:
            if s in denied_symptoms and symptom_disease_count.get(s, 0) == 1:
                score -= 0.20
                break  # 一个核心症状被否认就扣分，不叠加

        # 置信度限制在 [0, 1]
        c.confidence = round(max(0.0, min(1.0, score)), 4)

    # 按最终置信度降序排列
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates


def check_convergence(
    candidates: list[CandidateDisease],
    current_round: int,
    max_rounds: int = 10,
) -> tuple[bool, bool]:
    """
    判断问诊是否可以收敛（输出结论）。

    Returns:
        (should_conclude, force_conclude)
        should_conclude : True = 可以输出结论
        force_conclude  : True = 是因为达到轮次上限被迫结束（需在结论中标注"信息不足"）
    """
    # 达到轮次上限，强制结束
    if current_round >= max_rounds:
        return True, True

    if not candidates:
        return False, False

    top1 = candidates[0].confidence

    # 条件1：Top1 置信度 ≥ 70%
    if top1 >= 0.70:
        return True, False

    # 条件2：Top1 与 Top2 置信度差值 ≥ 30%（Top1 明显领先）
    if len(candidates) >= 2:
        top2 = candidates[1].confidence
        if top1 - top2 >= 0.30:
            return True, False

    return False, False