# src/agents/worker_tools.py

from langchain_core.tools import tool
# from src.agents.workers.inquiry_agent import get_inquiry_agent
from src.agents.workers.report_agent import get_report_agent
from src.agents.workers.drug_agent import get_drug_agent
from src.agents.workers.knowledge_agent import get_knowledge_agent
from src.agents.workers.operation_agent import get_operation_agent


# src/agents/worker_tools.py

from langchain_core.tools import tool
from src.agents.workers.report_agent import get_report_agent
from src.agents.workers.drug_agent import get_drug_agent
from src.agents.workers.knowledge_agent import get_knowledge_agent
from src.agents.workers.operation_agent import get_operation_agent


from dataclasses import dataclass

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agents.workers.report_agent import get_report_agent
from src.agents.workers.drug_agent import get_drug_agent
from src.agents.workers.knowledge_agent import get_knowledge_agent
from src.agents.workers.operation_agent import get_operation_agent

from src.agents.inquiry.graph import run_inquiry, InquiryDeps, build_inquiry_deps
from src.agents.inquiry.state import InquiryPhase
from src.agents.workers.inquiry_agent import handle_handoff
from src.infra.redis_cache import get_checkpointer_redis
from src.core.config import get_settings

settings = get_settings()

@dataclass
class UserContext:  # 用户上下文
    user_id: str
    session_id: str

# 仅第一轮会发起调用
@tool
async def call_inquiry_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """
    启动智慧问诊流程。
    适用场景：患者首次描述症状、询问挂哪个科室时。
    后续多轮对话由系统自动路由，无需再次调用此工具。

    Args:
        message    : 患者描述的症状或问诊需求
    """
    # 获取当前会话 ID 和用户 ID
    session_id = runtime.context.session_id
    user_id = runtime.context.user_id
    print("🔧工具调用 call_inquiry_agent :", session_id, message)

    deps = build_inquiry_deps()

    # 首轮：初始化空状态，执行第一轮问诊; 调用工作流
    reply, new_state = await run_inquiry(
        user_message=message,
        thread_id=f"{user_id}:{session_id}",
        deps=deps,
        existing_state=None,  # 首轮，无历史状态
        user_id=user_id,
    )

    # 首轮即收敛（精确症状/急症等）：触发挂号移交，不设活跃标记
    if new_state.phase == InquiryPhase.END:
        if new_state.handoff_payload:
            handoff_reply = await handle_handoff(new_state.handoff_payload)
            return f"{reply}\n\n---\n{handoff_reply}"
        return reply

    # 问诊未结束：保存状态，设置活跃标记，等待后续轮次
    redis = get_checkpointer_redis()
    state_key = f"inquiry_state:{user_id}:{session_id}"
    await redis.set(
        state_key,
        new_state.model_dump_json(),
        ex=3600,  # 1 小时过期
    )

    active_key = f"inquiry_active:{user_id}:{session_id}"
    await redis.set(active_key, "1", ex=3600)

    return reply


@tool
async def call_report_agent(message: str) -> str:
    """
    调用报告解读Agent，解读患者的检验报告或影像报告。
    适用场景：患者上传了报告需要解读、询问报告中某项指标含义时。
    message: 报告内容描述或患者的具体问题。
    """
    agent = get_report_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_drug_agent(message: str) -> str:
    """
    调用药物Agent，进行药物推荐、药物交互检测或处方审查。
    适用场景：询问用什么药、多种药物能否同服、处方是否安全时。
    message: 患者的用药问题或处方信息。
    """
    agent = get_drug_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_knowledge_agent(message: str) -> str:
    """
    调用知识问答Agent，回答医学知识类问题。
    适用场景：询问疾病知识、治疗方案、医学术语解释、文献检索时。
    message: 患者或医生的医学知识问题。
    """
    agent = get_knowledge_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_operation_agent(message: str) -> str:
    """
    调用运营数据Agent，查询医院运营统计数据。
    适用场景：运营人员询问就诊量、收入、科室排名等运营数据时。
    message: 运营人员的数据查询需求（自然语言）。
    """
    agent = get_operation_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_inquiry_agent,
    call_report_agent,
    call_drug_agent,
    call_knowledge_agent,
    call_operation_agent,
]
@tool
async def call_report_agent(message: str) -> str:
    """
    调用报告解读Agent，解读患者的检验报告或影像报告。
    适用场景：患者上传了报告需要解读、询问报告中某项指标含义时。
    message: 报告内容描述或患者的具体问题。
    """
    agent = get_report_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_drug_agent(message: str) -> str:
    """
    调用药物Agent，进行药物推荐、药物交互检测或处方审查。
    适用场景：询问用什么药、多种药物能否同服、处方是否安全时。
    message: 患者的用药问题或处方信息。
    """
    agent = get_drug_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_knowledge_agent(message: str) -> str:
    """
    调用知识问答Agent，回答医学知识类问题。
    适用场景：询问疾病知识、治疗方案、医学术语解释、文献检索时。
    message: 患者或医生的医学知识问题。
    """
    agent = get_knowledge_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


@tool
async def call_operation_agent(message: str) -> str:
    """
    调用运营数据Agent，查询医院运营统计数据。
    适用场景：运营人员询问就诊量、收入、科室排名等运营数据时。
    message: 运营人员的数据查询需求（自然语言）。
    """
    agent = get_operation_agent()
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content


# 所有 Worker 工具列表，供 Supervisor 使用
WORKER_TOOLS = [
    call_inquiry_agent,
    call_report_agent,
    call_drug_agent,
    call_knowledge_agent,
    call_operation_agent,
]