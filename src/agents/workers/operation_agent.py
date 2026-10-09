# src/agents/workers/operation_agent.py

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.infra.neo4j_client import get_neo4j_driver
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.tools import KnowledgeDeps, build_knowledge_tools

settings = get_settings()

OPERATION_SYSTEM_PROMPT = """你是天宫医疗的运营数据助手。

## 你的职责
1. 根据运营人员的自然语言问题，查询统计数据
2. 生成数据报表和趋势分析
3. 辅助医院管理层做运营决策

## 你的工具
- search_knowledge_sql：查询运营数据库中的统计数据（问诊量、药品库存、科室排名等）

## 安全规则
- 只允许查询聚合统计数据，严禁返回患者个人信息
- 所有查询结果需脱敏处理
- 如果问题涉及患者隐私，拒绝回答并说明原因

回复格式：
- 数据摘要：xxx
- 关键指标：xxx
- 趋势分析：xxx
- 决策建议：xxx

注意：数据查询结果仅供内部运营参考。"""


def create_operation_agent(db_session=None):
    llm = ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.2,
    )
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    deps = KnowledgeDeps(
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
        db_session=db_session,
    )
    knowledge_tools = build_knowledge_tools(deps)
    tools = [t for t in knowledge_tools if t.name == "search_knowledge_sql"]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=OPERATION_SYSTEM_PROMPT,
        name="operation_agent",
    )


_operation_agent = None

def get_operation_agent():
    global _operation_agent
    if _operation_agent is None:
        _operation_agent = create_operation_agent()
    return _operation_agent