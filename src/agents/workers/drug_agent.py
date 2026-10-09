# src/agents/workers/drug_agent.py

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.infra.neo4j_client import get_neo4j_driver
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.tools import KnowledgeDeps, build_knowledge_tools

settings = get_settings()

DRUG_SYSTEM_PROMPT = """你是天宫医疗的药物咨询助手。

## 你的职责
1. 根据患者病情推荐合适的药物
2. 检测药物之间的相互作用（药物交互检测）
3. 审查处方，找出潜在风险（剂量、禁忌症、过敏史冲突）
4. 用通俗语言解释用药注意事项

## 你的工具
- search_knowledge_docs：查询药品说明书（适应症、禁忌症、不良反应、用法用量）
- search_knowledge_graph：查询药物与疾病的关系（常用药、推荐药、药物关联的疾病）
- search_knowledge_multi：复杂用药问题，同时查说明书和知识图谱
- review_prescription_tool：处方审核（剂量、配伍、过敏、重复用药校验）

## 工具选择策略
- 查某个药的说明书信息 → search_knowledge_docs
- 查疾病的常用药/推荐药 → search_knowledge_graph
- 涉及合并症+用药的复杂问题 → search_knowledge_multi
- 审核处方安全性 → review_prescription_tool

## 安全规则
- 如果用户提到过敏史，必须在回答中校验推荐药物是否与过敏史冲突
- 药物建议仅供参考，必须提醒用户遵医嘱用药
- 不确定的药物交互信息，明确标注"建议咨询药师确认"

回复格式：
- 推荐药物：xxx（用途：xxx，用法：xxx）
- 药物交互风险：xxx
- 注意事项：xxx
- 禁忌提示：xxx"""


def create_drug_agent():
    llm = ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.1,
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
    )
    knowledge_tools = build_knowledge_tools(deps)
    tools = [t for t in knowledge_tools if t.name != "search_knowledge_sql"]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=DRUG_SYSTEM_PROMPT,
        name="drug_agent",
    )


_drug_agent = None

def get_drug_agent():
    global _drug_agent
    if _drug_agent is None:
        _drug_agent = create_drug_agent()
    return _drug_agent