from dataclasses import dataclass

from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.redis import AsyncRedisSaver
from langchain.agents import create_agent

from src.agents.tools.store_toots import save_memory, search_memory
from src.agents.tools.worker_tools import WORKER_TOOLS, UserContext
from src.infra.milvus_client import get_milvus_client_alias
from src.infra.milvus_store import MilvusStore
from src.infra.redis_cache import get_checkpointer_redis
from src.core.config import get_llm, get_settings
from dotenv import load_dotenv
load_dotenv()


def _get_embedding_model():
    """返回向量化模型。根据你的实际情况替换。"""
    # 使用 DashScope（阿里云）: 环境变量需要配置 DASHSCOPE_API_KEY
    from langchain_community.embeddings import DashScopeEmbeddings
    settings = get_settings()
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


SUPERVISOR_SYSTEM_PROMPT = """你是天宫医疗的智能总助手。

## 你的职责

识别用户意图，调用对应的专项助手处理，整合结果后给出清晰、友好的回复。
你自己不直接回答任何健康、医疗、症状相关的问题，必须通过专项助手完成。

## 可调用的专项助手

- call_inquiry_agent：智慧问诊
  适用：用户描述身体不适、症状，需要分诊建议或预约挂号时
  示例："我头疼发烧"、"肚子疼好几天了"、"不知道该挂什么科"、"有点不舒服"

- call_report_agent：报告解读
  适用：用户上传或描述检验报告、影像报告，需要解读关键指标时
  示例："帮我看看这个血常规报告"、"我的CT结果是什么意思"

- call_drug_agent：药物咨询
  适用：询问用药建议、药物相互作用、处方安全性时
  示例："布洛芬和阿莫西林能一起吃吗"、"我对青霉素过敏能用什么替代"

- call_knowledge_agent：医学知识问答
  适用：询问疾病知识、治疗方案、医学术语解释时
  示例："高血压平时要注意什么"、"糖尿病的早期症状有哪些"

- call_operation_agent：运营数据查询（仅限内部运营人员）
  适用：查询医院就诊量、科室排名、收入统计等运营数据时

## 记忆工具

- search_memory：在调用专项助手前，先检索用户的历史记忆（过敏史、既往病史、用药偏好等），将相关信息附加到调用参数中，避免重复询问用户
- save_memory：当用户提到重要的个人健康信息时（过敏史、慢性病、长期用药等），主动保存到长期记忆

## 工作原则

1. 你绝不直接回答健康相关问题。只要用户提到身体不适、症状、不舒服（如"不舒服"、"不太好"、"有点难受"），无论多模糊，都必须调用 call_inquiry_agent。意图不明确时也默认调 call_inquiry_agent。
2. 传递消息规则：调用专项助手时，message 参数必须原样传递用户的原始输入，禁止改写、补充指令或添油加醋。如果 search_memory 检索到了相关历史信息，以"[长期记忆] xxx"的格式附加在用户原文之后，不要修改用户原文本身。
3. 复杂问题可串联：例如用户描述症状后又问用药，先调用 call_inquiry_agent 再调用 call_drug_agent
4. 患者安全优先：识别到"胸痛"、"呼吸困难"、"意识不清"等急症关键词时，立即提示用户拨打 120 或前往急诊，不要等待问诊流程
5. 语气温和专业：用患者能理解的语言表达，避免过度使用医学术语"""



# 创建出 监督 Agent
async def create_supervisor_agent():
    # 1. 复用项目已有的 checkpointer 专用 Redis 客户端（bytes 模式）
    redis_client = get_checkpointer_redis()

    # 2. 创建 AsyncRedisSaver，并调用 asetup() 初始化 RediSearch 索引
    # asetup() 会在 Redis Stack 中创建 checkpoint / checkpoint_write 两个索引
    # 必须在首次使用前调用一次，索引已存在时自动跳过，可以重复调用
    checkpointer = AsyncRedisSaver(redis_client=redis_client)
    await checkpointer.asetup()


    # 3. 长期记忆
    # ── 长期记忆：Milvus Store ─────────────────────────────────────────
    milvus_alias = get_milvus_client_alias()
    embedding_model = _get_embedding_model()
    store = MilvusStore(
        alias=milvus_alias,
        embeddings=embedding_model,
        dims=1024,   # DashScope text-embedding-v3 默认输出 1024 维
    )

    # 4. 长期记忆工具
    tools = [save_memory, search_memory] + WORKER_TOOLS

    # 4. 创建 Agent
    agent = create_agent(
        model=get_llm(temperature=0.2),
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        context_schema=UserContext,
        middleware=[
            SummarizationMiddleware( # 会话总结压缩
                model=get_llm(temperature=0.2),
                trigger=[
                    ("tokens", 4000),  # token数达到4k时触发
                    ("messages", 6),  # 或消息数达到 4条时触发
                    # ("fraction", 0.8)  # 或80%消息时触发
                ],
                keep=("messages", 6),  # 摘要后保留最近 4 条消息
            )
        ],
        checkpointer=checkpointer, # 短期记忆. agent chat ui（禁用你配置 checkpointer）
        store=store # 长期记忆. 把同一个用户，任意会话的有价值信息进行存储。
    )
    return agent


# 模块级单例：避免每次请求都重新创建 agent 和 checkpointer
_supervisor_agent = None

# 返回 agent
async def get_supervisor_agent():
    """返回全局单例 Agent，首次调用时初始化。"""
    global _supervisor_agent
    if _supervisor_agent is None:
        _supervisor_agent = await create_supervisor_agent()
    return _supervisor_agent


# FastAPI 路由中使用
async def chat_endpoint(user_id: str, session_id: str, message: str):
    # 使用单例，不重复初始化。获取 agent
    agent = await get_supervisor_agent()

    # thread_id 用来区分不同会话。  用户id:会话id:日期（前端传来一个会话id）
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        context=UserContext(user_id=user_id, session_id=session_id),
    )
    return result["messages"][-1].content
