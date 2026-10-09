# src/agents/knowledge/feedback.py

from __future__ import annotations
from datetime import datetime
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def save_feedback(
        db: AsyncSession,
        user_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
        intent: str = "",
        channels: str = "",
) -> int:
    """
    保存用户对知识问答的反馈（点赞/踩）。
    rating: 1=有用, -1=无用, 0=未评价
    返回反馈记录 ID。
    """

    # 写入 knowledge_feedback 表
    result = await db.execute(
        text("""
             INSERT INTO knowledge_feedback
             (user_id, question, answer_preview, rating, comment, intent, channels, created_at)
             VALUES (:user_id, :question, :answer_preview, :rating, :comment, :intent, :channels,
                     :created_at) RETURNING id
             """),
        {
            "user_id": user_id,
            "question": question[:500],
            "answer_preview": answer[:500],
            "rating": rating,
            "comment": comment[:1000],
            "intent": intent,
            "channels": channels,
            "created_at": datetime.now(),
        },
    )
    await db.commit()
    row = result.fetchone()
    feedback_id = row[0] if row else 0
    logger.bind(audit=True).info(
        f"knowledge_feedback | user={user_id} | rating={rating} | "
        f"question={question[:60]} | comment={comment[:60]}"
    )
    return feedback_id


async def get_feedback_stats(db: AsyncSession) -> dict:
    """获取反馈统计数据。"""

    # 返回 {"total": N, "positive": N, "negative": N}
    result = await db.execute(text("""
                                   SELECT COUNT(*) AS total,
                                          COUNT(*)    FILTER (WHERE rating = 1) AS positive, COUNT(*) FILTER (WHERE rating = -1) AS negative
                                   FROM knowledge_feedback
                                   """))
    row = result.fetchone()
    if row:
        return {"total": row[0], "positive": row[1], "negative": row[2]}
    return {"total": 0, "positive": 0, "negative": 0}