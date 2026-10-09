# src/agents/knowledge/notification.py

from __future__ import annotations
from datetime import datetime
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def notify_doc_update(
    db: AsyncSession,
    doc_name: str,
    doc_type: str,
    category: str,
    action: str = "upload",
) -> int:
    """
    记录知识文档更新通知。
    文档上传/更新/删除时调用，写入通知表供相关科室查询。
    """
    result = await db.execute(
        text("""
            INSERT INTO knowledge_notifications
                (doc_name, doc_type, category, action, is_read, created_at)
            VALUES
                (:doc_name, :doc_type, :category, :action, false, :created_at)
            RETURNING id
        """),
        {
            "doc_name": doc_name,
            "doc_type": doc_type,
            "category": category,
            "action": action,
            "created_at": datetime.now(),
        },
    )
    await db.commit()
    row = result.fetchone()
    nid = row[0] if row else 0
    logger.info(f"知识更新通知 | action={action} | doc={doc_name} | category={category}")
    return nid


async def get_unread_notifications(
    db: AsyncSession,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """获取未读的知识更新通知。"""
    if category:
        result = await db.execute(
            text("""
                SELECT id, doc_name, doc_type, category, action, created_at
                FROM knowledge_notifications
                WHERE is_read = false AND category = :category
                ORDER BY created_at DESC LIMIT :limit
            """),
            {"category": category, "limit": limit},
        )
    else:
        result = await db.execute(
            text("""
                SELECT id, doc_name, doc_type, category, action, created_at
                FROM knowledge_notifications
                WHERE is_read = false
                ORDER BY created_at DESC LIMIT :limit
            """),
            {"limit": limit},
        )
    rows = result.fetchall()
    return [
        {
            "id": r[0], "doc_name": r[1], "doc_type": r[2],
            "category": r[3], "action": r[4],
            "created_at": str(r[5]),
        }
        for r in rows
    ]


async def mark_notifications_read(db: AsyncSession, ids: list[int]) -> None:
    """标记通知为已读。"""
    if not ids:
        return
    await db.execute(
        text("UPDATE knowledge_notifications SET is_read = true WHERE id = ANY(:ids)"),
        {"ids": ids},
    )
    await db.commit()