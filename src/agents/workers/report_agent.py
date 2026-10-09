# src/agents/workers/report_agent.py

from langchain.agents import create_agent
from src.core.config import get_llm

REPORT_SYSTEM_PROMPT = """你是天宫医疗的报告解读助手。

你的职责：
1. 解读患者上传的检验报告、影像报告
2. 提取关键异常指标，用通俗语言解释含义
3. 标注需要重点关注的项目
4. 给出初步建议（是否需要复查、就诊等）

回复格式：
- 报告类型：xxx
- 关键发现：xxx
- 异常指标：xxx（正常范围：xxx，当前值：xxx）
- 建议：xxx

注意：解读结果仅供参考，最终诊断以医生为准。"""


def create_report_agent():
    llm = get_llm(temperature=0.1)

    # 当前阶段：模拟解读
    # 未来可添加：Qwen-VL 视觉模型工具、历史报告对比工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=REPORT_SYSTEM_PROMPT,
        name="report_agent",
    )


_report_agent = None

def get_report_agent():
    global _report_agent
    if _report_agent is None:
        _report_agent = create_report_agent()
    return _report_agent