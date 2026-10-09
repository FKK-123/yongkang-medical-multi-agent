from __future__ import annotations
import json
import traceback
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.infra.database import get_db
from src.infra.redis_cache import get_checkpointer_redis
from src.agents.supervisor_agent import get_supervisor_agent, UserContext
from src.agents.inquiry.graph import run_inquiry, build_inquiry_deps
from src.agents.inquiry.state import InquiryState, InquiryPhase
from src.agents.workers.inquiry_agent import handle_handoff

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    patient_id: int | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


def _make_keys(user_id: str, session_id: str) -> tuple[str, str]:
    """生成 Redis 键名。thread_id 与 Supervisor checkpointer 保持一致。"""
    thread_id = f"{user_id}:{session_id}"
    return f"inquiry_active:{thread_id}", f"inquiry_state:{thread_id}"


async def _run_inquiry_turn(
    message: str,
    thread_id: str,
    redis,
    db,
) -> str:
    """
    执行一轮问诊对话（路由层直接调用，绕过 Supervisor）。
    从 Redis 恢复状态 → 执行 InquiryGraph → 保存新状态 → 返回回复。
    """
    active_key = f"inquiry_active:{thread_id}"
    state_key  = f"inquiry_state:{thread_id}"

    # 从 Redis 反序列化恢复上一轮状态
    raw = await redis.get(state_key)
    existing_state = InquiryState.model_validate_json(raw) if raw else None

    deps = build_inquiry_deps(db_session=db)
    reply, new_state = await run_inquiry(
        user_message=message,
        thread_id=thread_id,
        deps=deps,
        existing_state=existing_state,
    )

    # 问诊结束：清除 Redis 标记，触发挂号移交
    if new_state.phase in (InquiryPhase.HANDOFF, InquiryPhase.END):
        await redis.delete(active_key, state_key)
        if new_state.phase == InquiryPhase.HANDOFF and new_state.handoff_payload:
            handoff_reply = await handle_handoff(new_state.handoff_payload)
            return f"{reply}\n\n---\n{handoff_reply}"
        return reply

    # 问诊继续：更新 Redis 状态，重置 TTL
    await redis.set(state_key,  new_state.model_dump_json(), ex=3600)
    await redis.set(active_key, "1",                         ex=3600)
    return reply


# ── 非流式接口 ────────────────────────────────────────────────────────────
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    非流式对话接口。
    路由层判断是否有活跃问诊：有则直接走 InquiryGraph，无则走 Supervisor。
    """
    try:
        redis = get_checkpointer_redis()
        active_key, state_key = _make_keys(req.user_id, req.session_id)
        thread_id = f"{req.user_id}:{req.session_id}"
        active_key = f"inquiry_active:{thread_id}"

        # ── 问诊进行中：直接走 InquiryGraph ──
        if await redis.exists(active_key):
            reply = await _run_inquiry_turn(req.message, thread_id, redis, db)
            return ChatResponse(reply=reply, session_id=req.session_id)

        # ── 无活跃问诊：走 Supervisor ──
        agent = await get_supervisor_agent()
        config = {"configurable": {"thread_id": thread_id}}

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config=config,
            context=UserContext(user_id=req.user_id, session_id=req.session_id),
        )
        reply = result["messages"][-1].content
        return ChatResponse(reply=reply, session_id=req.session_id)

    except Exception as e:
        logger.exception(f"chat 接口异常")
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── 流式接口（SSE） ───────────────────────────────────────────────────────
@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    流式对话接口（Server-Sent Events）。
    问诊进行中时，InquiryGraph 的回复以流式推送；
    Supervisor 回复同样以流式推送。

    客户端接收格式：
        data: {"type": "token",  "content": "..."}
        data: {"type": "done",   "session_id": "..."}
        data: {"type": "error",  "message": "..."}
    """
    async def event_generator():
        try:
            redis = get_checkpointer_redis()
            thread_id = f"{req.user_id}:{req.session_id}"
            active_key = f"inquiry_active:{thread_id}"

            # ── 问诊进行中：InquiryGraph 非流式执行，结果整体推送 ──
            # （InquiryGraph 内部多次调用 LLM，流式拆分复杂度高，
            #   此处以整体推送为主，后续可按节点拆分优化）
            if await redis.exists(active_key):
                reply = await _run_inquiry_turn(req.message, thread_id, redis, db)
                data = json.dumps({"type": "token", "content": reply}, ensure_ascii=False)
                yield f"data: {data}\n\n"

            else:
                # ── 无活跃问诊：Supervisor 流式推送 ──
                agent = await get_supervisor_agent()
                config = {"configurable": {"thread_id": thread_id}}
                async for chunk in agent.astream(
                    {"messages": [{"role": "user", "content": req.message}]},
                    config=config,
                    stream_mode="messages",
                ):
                    if isinstance(chunk, tuple):
                        msg_chunk, _ = chunk
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            data = json.dumps(
                                {"type": "token", "content": msg_chunk.content},
                                ensure_ascii=False,
                            )
                            yield f"data: {data}\n\n"

            done_data = json.dumps(
                {"type": "done", "session_id": req.session_id}, ensure_ascii=False
            )
            yield f"data: {done_data}\n\n"

        except Exception as e:
            logger.exception(f"chat/stream 接口异常")
            error_data = json.dumps({"type": "error", "message": traceback.format_exc()}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
