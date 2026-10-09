import uuid

from src.agents.supervisor_agent import chat_endpoint

import pytest

@pytest.mark.asyncio
async def test_supervisor_agent():
    result = await chat_endpoint("001","001","你好，我叫凡永康")
    print(result)


import pytest
from src.agents.supervisor_agent import chat_endpoint


@pytest.mark.asyncio
async def test_agent_memory():
    """验证短期记忆：同一 thread_id 下第二轮能记住第一轮的内容"""
    # 第一轮：自我介绍
    reply1 = await chat_endpoint("1123", "ATDAAS", "你好，我叫雷丰阳")
    print(f"\n第一轮回复：{reply1}")

    # 第二轮：考察记忆
    reply2 = await chat_endpoint("1123", "ATDAAS", "我是谁？")
    print(f"第二轮回复：{reply2}")

    # 断言：第二轮回复应包含名字
    assert "雷丰阳" in reply2, f"Agent 应该记住用户名字，实际回复：{reply2}"


def new_session() -> str:
    """每次测试生成唯一 session_id，防止不同测试/不同执行轮次的 checkpoint 互相污染。"""
    return uuid.uuid4().hex[:8]


async def test_agent_store():
    """验证长期记忆：跨会话记忆
    第一个 session 写入病史 → 第二个全新 session 能检索到
    """
    # ── 第一轮：用新会话写入病史 ──────────────────────────────────────────
    session_1 = new_session()
    resp1 = await chat_endpoint("1123", session_1, "你好，我叫张三，有糖尿病史")
    print(f"\n[写入] 回复：{resp1}")

    # ── 第二轮：换一个全新会话，查询病史（跨会话长期记忆） ───────────────
    session_2 = new_session()  # 与 session_1 不同，短期记忆里没有上文
    resp2 = await chat_endpoint("1123", session_2, "我有什么病史呢？我叫什么？")
    print(f"[检索] 回复：{resp2}")

    assert "糖尿病" in resp2, f"长期记忆应能跨会话检索到糖尿病史，实际回复：{resp2}"