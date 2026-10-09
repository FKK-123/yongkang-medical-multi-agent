# src/agents/inquiry/db_queries.py

from __future__ import annotations
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from src.modules.medical.model import Patient, Consultation
from src.agents.inquiry.state import PatientContext


async def load_patient_context(
    patient_id: int | None,
    db: AsyncSession | None = None,
    recent_consultations_limit: int = 5,
) -> PatientContext:
    """
    从 PostgreSQL 加载患者上下文：基本信息 + 近期问诊记录。
    patient_id 为 None 时返回空上下文。
    db 为 None 时自动创建 session。
    """
    if patient_id is None:
        return PatientContext()

    from src.infra.database import AsyncSessionLocal
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        # 查患者基本信息
        patient_result = await db.execute(
            select(Patient).where(Patient.id == patient_id)
        )
        patient = patient_result.scalar_one_or_none()
        if not patient:
            logger.warning(f"患者 ID {patient_id} 不存在")
            return PatientContext()

        # 解析既往病史和过敏史（数据库中用逗号分隔存储）
        medical_history = (
            [h.strip() for h in patient.medical_history.split(",") if h.strip()]
            if patient.medical_history else []
        )
        allergy_history = (
            [a.strip() for a in patient.allergy_history.split(",") if a.strip()]
            if patient.allergy_history else []
        )

        # 查近期问诊记录（取最近 N 条）
        consultations_result = await db.execute(
            select(Consultation)
            .where(Consultation.patient_id == patient_id)
            .order_by(desc(Consultation.created_at))
            .limit(recent_consultations_limit)
        )
        consultations = consultations_result.scalars().all()

        # 把历史诊断结论加入 medical_history（用于置信度加权）
        for c in consultations:
            if c.diagnosis and c.diagnosis not in medical_history:
                medical_history.append(c.diagnosis)

        return PatientContext(
            patient_id=patient_id,
            age=patient.age,
            gender=patient.gender,
            allergy_history=allergy_history,
            medical_history=medical_history,
        )
    except Exception as e:
        logger.error(f"加载患者上下文失败: {e}")
        return PatientContext()
    finally:
        if _own_session:
            await db.close()


async def save_consultation_record(
    patient_id: int | None,
    session_id: str,
    chief_complaint: str,
    diagnosis: str,
    department_name: str,
    urgency_level: str,
    db: AsyncSession | None = None,
) -> int | None:
    """
    将本次问诊结果保存到 PostgreSQL consultations 表。
    返回新记录的 ID，失败时返回 None。
    db 为 None 时（如从 Supervisor 工具链调用）自动创建 session。
    """
    from src.infra.database import AsyncSessionLocal

    # db 为 None 时自己创建 session，用完后关闭
    _own_session = db is None
    if _own_session:
        db = AsyncSessionLocal()

    try:
        # 查科室 ID（通过科室名查）
        from src.modules.medical.model import Department
        dept_result = await db.execute(
            select(Department).where(Department.name == department_name)
        )
        dept = dept_result.scalar_one_or_none()
        dept_id = dept.id if dept else None

        consultation = Consultation(
            patient_id=patient_id,  # 未登录用户为 None，存 NULL
            department_id=dept_id,
            chief_complaint=chief_complaint,
            diagnosis=diagnosis,
            urgency_level=urgency_level,
            session_id=session_id,
        )
        db.add(consultation)
        await db.commit()
        await db.refresh(consultation)
        logger.info(f"问诊记录已保存，ID={consultation.id}")
        return consultation.id
    except Exception as e:
        await db.rollback()
        logger.error(f"保存问诊记录失败: {e}")
        return None
    finally:
        if _own_session:
            await db.close()