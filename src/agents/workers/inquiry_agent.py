import uuid

from langchain.agents import create_agent
from src.core.config import get_llm, get_settings
from src.agents.inquiry.state import InquiryHandoffPayload

settings = get_settings()

INQUIRY_WORKER_PROMPT = """你是天宫医疗的挂号助手。

你已经收到了智能问诊的结论，现在需要帮助患者完成挂号预约。

问诊结论：
{handoff_payload}

你的职责：
1. 向患者确认挂号信息（科室、时间偏好）
2. 生成问诊单摘要（供医生参考）
3. 完成预约挂号（调用挂号工具，待接入）

请用温和、专业的语气与患者沟通。"""


def create_inquiry_worker_agent():
    llm = get_llm(temperature=0.3)
    # 当前阶段：无工具，纯 LLM 推理
    # 未来可添加：预约挂号工具、问诊单生成工具、排班查询工具
    tools = []
    return create_agent(model=llm, tools=tools)


_inquiry_worker_agent = None

def get_inquiry_worker_agent():
    global _inquiry_worker_agent
    if _inquiry_worker_agent is None:
        _inquiry_worker_agent = create_inquiry_worker_agent()
    return _inquiry_worker_agent


# Compatibility name used by earlier versions of worker_tools.py.
get_inquiry_agent = get_inquiry_worker_agent

#生产环境
# async def handle_handoff(payload: InquiryHandoffPayload) -> str:
#     """
#     接收问诊移交数据包，执行挂号流程。
#     返回给用户的挂号确认消息。
#     """
#     agent = get_inquiry_worker_agent()
#     prompt = INQUIRY_WORKER_PROMPT.format(
#         handoff_payload=payload.model_dump_json(indent=2, ensure_ascii=False)
#     )
#     result = await agent.ainvoke(
#         {"messages": [{"role": "user", "content": prompt}]}
#     )
#     return result["messages"][-1].content

# ── 结构化输出 Schema ─────────────────────────────────────────────────────
from pydantic import Field
from pydantic import BaseModel

class AppointmentResult(BaseModel):
    """LLM 生成的模拟挂号预约结果"""
    doctor_name: str = Field(description="医生姓名及职称，如：张伟（主任医师）")
    department: str = Field(description="就诊科室")
    date: str = Field(description="就诊日期，格式 YYYY-MM-DD，取明天或后天")
    time_slot: str = Field(description="就诊时段，如：08:00-12:00 或 14:00-17:30")
    fee: float = Field(description="挂号费（元），主任医师 50 元，其他 30 元")
    queue_number: int = Field(description="就诊序号，1~30 之间的整数")


# ── 核心函数 ──────────────────────────────────────────────────────────────

async def mock_create_appointment(payload: InquiryHandoffPayload) -> AppointmentResult:
    """
    用 LLM 结构化输出生成一条模拟预约数据。
    输入问诊结论，输出符合场景的预约信息，无需维护任何模拟数据库。
    """
    llm = get_llm(temperature=0.7)  # 适当随机，让每次生成的医生/时间略有不同
    structured_llm = llm.with_structured_output(AppointmentResult)

    prompt = f"""根据以下问诊结论，生成一条合理的挂号预约信息。

问诊结论：
- 主诊断：{payload.primary_disease}
- 建议科室：{payload.department}
- 确认症状：{', '.join(payload.confirmed_symptoms[:5])}

要求：
- 医生姓名用中文常见姓名，附上职称
- 日期取明天或后天（今天是 {__import__('datetime').date.today()}）
- 时段选上午或下午
- 挂号费主任医师 50 元，其他 30 元"""

    return await structured_llm.ainvoke(prompt)

# 测试环境用，不依赖真实挂号系统
async def handle_handoff(payload: InquiryHandoffPayload) -> str:
    """
    接收问诊移交数据包，生成模拟挂号结果并返回确认文案。
    """
    result = await mock_create_appointment(payload)
    appointment_id = f"APT{uuid.uuid4().hex[:8].upper()}"

    return (
        f"已为您成功预约挂号！\n\n"
        f"预约号：{appointment_id}\n"
        f"就诊科室：{result.department}\n"
        f"就诊医生：{result.doctor_name}\n"
        f"就诊时间：{result.date} {result.time_slot}\n"
        f"就诊序号：第 {result.queue_number} 号\n"
        f"挂号费：{result.fee} 元\n\n"
        f"请提前 15 分钟到诊室门口等候，祝您早日康复。"
    )
