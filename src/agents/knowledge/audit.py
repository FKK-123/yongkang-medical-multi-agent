# src/agents/knowledge/audit.py

from __future__ import annotations
import time
from loguru import logger


class QueryAuditLog:
    """查询审计日志记录器。记录每次知识检索的完整链路。"""

    @staticmethod
    def log(
        user_id: str,
        role: str,
        question: str,
        intent: str,
        channels: list[str],
        answer_preview: str,
        duration_ms: float,
        hallucination_check: dict | None = None,
    ) -> None:
        logger.bind(audit=True).info(
            "knowledge_query | "
            f"user={user_id} | role={role} | intent={intent} | "
            f"channels={channels} | duration={duration_ms:.0f}ms | "
            f"grounded={hallucination_check.get('is_grounded', 'N/A') if hallucination_check else 'N/A'} | "
            f"question={question[:80]} | "
            f"answer={answer_preview[:80]}"
        )


class Timer:
    """简单计时器，用于测量检索耗时。"""

    def __init__(self):
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        pass

    @property
    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) * 1000