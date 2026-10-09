from __future__ import annotations
import json
from typing import Any

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pymilvus import MilvusClient

from src.infra.milvus_client import get_milvus_client_alias
from src.infra.neo4j_client import get_neo4j_driver
from src.core.config import get_settings
from src.core.config import get_llm
from src.agents.inquiry.state import (
    InquiryState, InquiryPhase, InquiryHandoffPayload, PatientContext
)
from src.agents.inquiry.symptom_normalizer import (
    normalize_symptoms, humanize_symptoms
)
from src.agents.inquiry.neo4j_queries import (
    query_candidate_diseases, enrich_candidate_details, get_pending_symptoms
)
from src.agents.inquiry.confidence import apply_context_weights, check_convergence
from src.agents.inquiry.prompts import (
    CLARIFY_PROMPT, ASK_SYMPTOMS_PROMPT, PARSE_ANSWER_PROMPT,
    CONCLUSION_PROMPT, EMERGENCY_CHECK_PROMPT
)

settings = get_settings()

# ── 依赖注入容器（在 graph 编译时注入，避免全局单例） ──────────────────────
class InquiryDeps:
    def __init__(self, llm, neo4j_driver, embedding_model, milvus_client, db_session):
        self.llm = llm
        self.neo4j_driver = neo4j_driver
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.db_session = db_session


# ════════════════════════════════════════════════════════════════════════
# 节点函数（每个节点接收 state，返回 state 的部分更新）
# ════════════════════════════════════════════════════════════════════════

async def node_load_context(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点①：加载患者上下文。
    从 PostgreSQL 加载既往病史，从 Milvus 加载长期记忆。
    仅在第一轮（round==0）执行，后续轮次跳过。
    """
    if state.round > 0:
        logger.debug("节点①加载患者上下文 非首轮，跳过上下文加载 (round={})", state.round)
        return {}  # 非首轮，不重复加载

    logger.info("节点①加载患者上下文 开始加载患者上下文 | patient_id={} session_id={}",
                state.patient_context.patient_id, state.session_id)

    from src.agents.inquiry.db_queries import load_patient_context
    # user_id 即 patients.id，前端传来的是字符串，转 int 后查患者档案
    patient_id = int(state.patient_context.patient_id) if state.patient_context.patient_id else None
    # 从HIS查询患者信息和就诊记录
    patient_ctx = await load_patient_context(
        patient_id=patient_id,
        db=deps.db_session,
    )

    ## TODO 从Milvus加载长期记忆


    merged_ctx = PatientContext(
        patient_id=state.patient_context.patient_id,  # 保持 str，贯穿整个流程
        age=patient_ctx.age,
        gender=patient_ctx.gender,
        allergy_history=patient_ctx.allergy_history,
        medical_history=patient_ctx.medical_history,
        long_term_memories=state.patient_context.long_term_memories,
    )
    logger.info("节点①加载患者上下文 上下文加载完成 | age={} gender={} medical_history={} allergy={}",
                merged_ctx.age, merged_ctx.gender,
                len(merged_ctx.medical_history), len(merged_ctx.allergy_history))
    return {"patient_context": merged_ctx}


async def node_check_emergency(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点②：急症识别。
    仅在第一轮执行。识别到急症时直接跳转到 CONCLUDE 阶段，
    并在 candidate_diseases 中放入一个特殊的"急诊"标记。
    """
    if state.round > 0:
        logger.debug("节点②急症识别 非首轮，跳过急症检查")
        return {}

    logger.info("节点②急症识别 开始急症识别检查")
    last_user_msg = ""
    for msg in reversed(state.messages):  # 历史消息： 1-2-3-4-5-6
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        return {}

    prompt = EMERGENCY_CHECK_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        result = json.loads(content)
        if result.get("is_emergency"):
            logger.warning("节点②急症识别 ⚠️ 检测到急症！原因: {}", result.get('reason', ''))
            emergency_reply = (
                "⚠️ 根据您描述的症状，这可能是紧急情况！\n\n"
                f"原因：{result.get('reason', '存在急症风险')}\n\n"
                "**请立即前往最近医院的急诊科就诊，或拨打 120 急救电话。**\n\n"
                "不要等待，请立即行动！"
            )
            return {
                "phase": InquiryPhase.END,
                "messages": [AIMessage(content=emergency_reply)],
            }
        else:
            logger.info("节点②急症识别 未检测到急症，继续正常问诊流程")
    except Exception as e:
        logger.warning(f"急症识别解析失败: {e}")
    return {}


async def node_extract_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点③：症状标准化。
    从最新的用户消息中提取症状，经三层标准化后合并到 confirmed_symptoms。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg:
        logger.warning("节点③症状标准化 未找到用户消息，跳过症状提取")
        return {}

    logger.info("节点③症状标准化 开始提取症状 | 用户输入: {!r}", last_user_msg[:80])

    # 等待标准化三层管线执行
    result = await normalize_symptoms(
        user_input=last_user_msg,
        llm=deps.llm,
        neo4j_driver=deps.neo4j_driver,
        embedding_model=deps.embedding_model,
        milvus_client=deps.milvus_client,
    )

    new_confirmed = list(set(state.confirmed_symptoms) | set(result["all_standard"]))
    new_unmatched = list(set(state.unmatched_symptoms) | set(result["unmatched"]))

    logger.info("节点③症状标准化 提取完成 | 标准化症状={} 未匹配={} 累计确认={}",
                result["all_standard"], result["unmatched"], new_confirmed)

    if new_confirmed:
        return {
            "confirmed_symptoms": new_confirmed,
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.GRAPH_QUERY, # 接下来进入哪个阶段
        }
    else:
        logger.info("节点③症状标准化 无法提取明确症状，进入澄清流程")
        return {
            "unmatched_symptoms": new_unmatched,
            "phase": InquiryPhase.CLARIFY,
        }


async def node_clarify(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点④：澄清模糊描述。
    当用户描述没有明确症状时，引导用户进一步描述。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    logger.info("节点④澄清模糊描述 用户描述模糊，发起澄清引导 | round={}", state.round)
    prompt = CLARIFY_PROMPT.format(user_input=last_user_msg)
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])
    logger.debug("节点④澄清模糊描述 澄清回复已生成，等待用户下一轮输入")
    return {
        "round": state.round + 1,
        "messages": [AIMessage(content=response.content)],
    }


async def \
        node_query_neo4j(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑤：查询 Neo4j 候选疾病。
    用已确认症状查候选疾病，补充详情，应用上下文权重。
    """
    logger.info("节点⑤查询候选疾病 查询候选疾病 | 确认症状={}", state.confirmed_symptoms)
    candidates = await query_candidate_diseases(
        confirmed_symptoms=state.confirmed_symptoms,
        neo4j_driver=deps.neo4j_driver,
    )
    if candidates:
        candidates = await enrich_candidate_details(candidates, deps.neo4j_driver)
        candidates = apply_context_weights(
            candidates, state.patient_context, state.denied_symptoms
        )
        logger.info("节点⑤查询候选疾病 候选疾病 top5: {}",
                    [(c.name, round(c.confidence, 3)) for c in candidates[:5]])
    else:
        logger.warning("节点⑤查询候选疾病 未找到候选疾病，将强制结束问诊")

    # 收敛判断（要么超10轮，要么有候选疾病，要么没有更多有用信息）
    should_conclude, force_conclude = check_convergence(candidates, state.round)
    logger.info("节点⑤查询候选疾病 收敛判断 | should_conclude={} force_conclude={} round={}",
                should_conclude, force_conclude, state.round)

    if should_conclude:
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": force_conclude,
        }
    elif not candidates:
        # 没有候选疾病（症状太罕见），直接结束
        return {
            "candidate_diseases": [],
            "phase": InquiryPhase.CONCLUDE,
            "force_conclude": True,
        }
    else:
        return {
            "candidate_diseases": candidates,
            "phase": InquiryPhase.SYMPTOM_CONFIRM,
        }


async def node_ask_symptoms(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑥：生成追问话术。
    从候选疾病中选出区分度最高的 ≤3 个症状，口语化后追问用户。
    """
    logger.info("节点⑥生成追问话术 生成追问 | round={} 已问症状={}", state.round, state.asked_symptoms)
    pending = await get_pending_symptoms(
        candidates=state.candidate_diseases,
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        asked_symptoms=state.asked_symptoms,
    )

    # 取区分度最高的前 3 个（出现次数最少的）
    top_symptoms = [s for s, _ in pending[:3]]

    if not top_symptoms:
        # 没有更多可问的症状，直接收敛
        logger.info("节点⑥生成追问话术 无更多可追问症状，直接进入结论")
        return {"phase": InquiryPhase.CONCLUDE}

    logger.info("节点⑥生成追问话术 本轮追问症状: {}", top_symptoms)
    # 口语化
    human_symptoms = await humanize_symptoms(top_symptoms, deps.llm)

    prompt = ASK_SYMPTOMS_PROMPT.format(
        symptoms_to_ask="\n".join(f"- {s}" for s in human_symptoms)
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    return {
        "round": state.round + 1,
        "asked_symptoms": state.asked_symptoms + top_symptoms,
        "pending_ask_symptoms": top_symptoms,  # 记录本轮问了哪些，供下一节点解析
        "messages": [AIMessage(content=response.content)],
    }


async def node_parse_answer(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑦：解析用户对追问的回答。
    判断用户确认/否认了哪些症状，更新 confirmed_symptoms 和 denied_symptoms。
    """
    last_user_msg = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    if not last_user_msg or not state.pending_ask_symptoms:
        logger.debug("节点⑦解析用户回答 无用户消息或无待解析症状，跳过")
        return {}

    logger.info("节点⑦解析用户回答 解析用户回答 | 待解析症状={} 用户回答={!r}",
                state.pending_ask_symptoms, last_user_msg[:80])

    prompt = PARSE_ANSWER_PROMPT.format(
        asked_symptoms=", ".join(state.pending_ask_symptoms),
        user_answer=last_user_msg,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1].lstrip("json").strip()
        parsed = json.loads(content)
        new_confirmed = list(
            set(state.confirmed_symptoms) | set(parsed.get("confirmed", []))
        )
        new_denied = list(
            set(state.denied_symptoms) | set(parsed.get("denied", []))
        )
        logger.info("节点⑦解析用户回答 解析结果 | 新增确认={} 新增否认={}",
                    parsed.get("confirmed", []), parsed.get("denied", []))
    except Exception as e:
        logger.warning(f"解析用户回答失败: {e}")
        new_confirmed = state.confirmed_symptoms
        new_denied = state.denied_symptoms

    return {
        "confirmed_symptoms": new_confirmed,
        "denied_symptoms": new_denied,
        "pending_ask_symptoms": [],
    }


async def node_conclude(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑧：生成诊断结论。
    整合所有信息，生成用户可读的结论文案，并构建 HandoffPayload。
    """
    candidates = state.candidate_diseases
    if not candidates:
        logger.warning("节点⑧生成诊断结论 无候选疾病，返回无结果提示")
        no_result_msg = (
            "根据您描述的症状，我暂时无法在知识库中找到匹配的疾病。\n\n"
            "这可能是因为症状较为罕见，或者需要更多信息。\n"
            "建议您直接前往医院就诊，由医生进行面诊。"
        )
        return {
            "phase": InquiryPhase.END,
            "messages": [AIMessage(content=no_result_msg)],
        }

    top1 = candidates[0]
    suspected = [c.name for c in candidates[1:5]]
    logger.info("节点⑧生成诊断结论 生成诊断结论 | 主诊断={} 置信度={:.3f} 科室={} 疑似={}",
                top1.name, top1.confidence, top1.department, suspected)

    prompt = CONCLUSION_PROMPT.format(
        confirmed_symptoms="、".join(state.confirmed_symptoms) or "暂无",
        primary_disease=top1.name,
        confidence=top1.confidence,
        suspected_diseases="、".join(suspected) if suspected else "无",
        department=top1.department or "综合内科",
        checks="、".join(top1.checks[:5]) if top1.checks else "暂无",
        force_conclude=state.force_conclude,
    )
    response = await deps.llm.ainvoke([SystemMessage(content=prompt)])

    # 构建移交数据包（挂号数据包）
    handoff = InquiryHandoffPayload(
        patient_id=state.patient_context.patient_id,
        patient_context=state.patient_context.model_dump(),
        confirmed_symptoms=state.confirmed_symptoms,
        denied_symptoms=state.denied_symptoms,
        unmatched_symptoms=state.unmatched_symptoms,
        primary_disease=top1.name,
        primary_confidence=top1.confidence,
        suspected_diseases=suspected,
        department=top1.department or "综合内科",
        recommended_checks=top1.checks[:5],
        total_rounds=state.round,
        session_id=state.session_id,
    )

    return {
        "phase": InquiryPhase.HANDOFF,
        "handoff_payload": handoff,
        "messages": [AIMessage(content=response.content)],
    }


async def node_save_record(state: InquiryState, deps: InquiryDeps) -> dict:
    """
    节点⑨：保存问诊记录到 PostgreSQL。
    无论用户是否同意挂号都执行。
    """
    if not state.handoff_payload:
        logger.warning("节点⑨保存问诊记录 无 handoff_payload，跳过保存")
        return {"phase": InquiryPhase.END}

    from src.agents.inquiry.db_queries import save_consultation_record
    payload = state.handoff_payload
    chief_complaint = "、".join(state.confirmed_symptoms[:5])
    # user_id 即 patients.id，转 int 后写入外键
    patient_id = int(payload.patient_id) if payload.patient_id else None

    logger.info("节点⑨保存问诊记录 保存问诊记录 | patient_id={} diagnosis={} department={}",
                patient_id, payload.primary_disease, payload.department)

    await save_consultation_record(
        patient_id=patient_id,
        session_id=payload.session_id,
        chief_complaint=chief_complaint,
        diagnosis=payload.primary_disease,
        department_name=payload.department,
        urgency_level="normal",
        db=deps.db_session,
    )
    logger.info("节点⑨保存问诊记录 问诊记录保存完成，流程结束")
    return {"phase": InquiryPhase.END}



# src/agents/inquiry/graph.py （续）

# ════════════════════════════════════════════════════════════════════════
# 路由函数（决定下一个节点）
# ════════════════════════════════════════════════════════════════════════

def route_dispatcher(state: InquiryState) -> str:
    """入口分发：首轮走完整流程，后续轮次根据上一轮状态决定入口。"""
    if state.round == 0:
        return "load_context"
    # 上一轮是追问症状（pending_ask_symptoms 非空）→ 解析用户回答
    if state.pending_ask_symptoms:
        return "parse_answer"
    # 上一轮是澄清引导（pending_ask_symptoms 为空）→ 提取新症状
    return "extract_symptoms"


def route_after_emergency(state: InquiryState) -> str:
    """急症检查后的路由：急症直接结束，否则提取症状。"""
    if state.phase == InquiryPhase.END:
        return END
    return "extract_symptoms"


def route_after_extract(state: InquiryState) -> str:
    """症状提取后的路由：有症状查图谱，没症状澄清。"""
    if state.phase == InquiryPhase.GRAPH_QUERY:
        return "query_neo4j"
    return "clarify"


def route_after_neo4j(state: InquiryState) -> str:
    """Neo4j 查询后的路由：收敛则结论，否则追问。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return "ask_symptoms"


def route_after_ask(state: InquiryState) -> str:
    """追问后的路由：没有更多症状可问则直接结论。"""
    if state.phase == InquiryPhase.CONCLUDE:
        return "conclude"
    return END  # 等待用户回答（下一轮消息进来后从 parse_answer 继续）


def route_after_parse(state: InquiryState) -> str:
    """解析回答后：重新查 Neo4j 更新候选疾病。"""
    return "query_neo4j"


def route_after_conclude(state: InquiryState) -> str:
    """结论后：保存记录，结束。"""
    if state.phase == InquiryPhase.END:
        return END
    return "save_record"


# ════════════════════════════════════════════════════════════════════════
# 图的组装
# ════════════════════════════════════════════════════════════════════════

def build_inquiry_graph(deps: InquiryDeps):
    """
    构建并编译问诊 StateGraph。
    deps 通过闭包注入到每个节点函数中。
    """
    # 用闭包把 deps 绑定到节点函数
    async def _load_context(state):    return await node_load_context(state, deps)
    async def _check_emergency(state): return await node_check_emergency(state, deps)
    async def _extract_symptoms(state):return await node_extract_symptoms(state, deps)
    async def _clarify(state):         return await node_clarify(state, deps)
    async def _query_neo4j(state):     return await node_query_neo4j(state, deps)
    async def _ask_symptoms(state):    return await node_ask_symptoms(state, deps)
    async def _parse_answer(state):    return await node_parse_answer(state, deps)
    async def _conclude(state):        return await node_conclude(state, deps)
    async def _save_record(state):     return await node_save_record(state, deps)

    #有状态图
    graph = StateGraph(InquiryState)

    # 注册节点（dispatcher 是无操作的分发节点，仅用于路由）
    graph.add_node("dispatcher",       lambda state: {})
    graph.add_node("load_context",     _load_context)
    graph.add_node("check_emergency",  _check_emergency)
    graph.add_node("extract_symptoms", _extract_symptoms)
    graph.add_node("clarify",          _clarify)
    graph.add_node("query_neo4j",      _query_neo4j)
    graph.add_node("ask_symptoms",     _ask_symptoms)
    graph.add_node("parse_answer",     _parse_answer)
    graph.add_node("conclude",         _conclude)
    graph.add_node("save_record",      _save_record)

    # 入口：dispatcher 根据 round 分发到首轮流程或后续轮次流程
    graph.set_entry_point("dispatcher")

    # 注册边（固定边）
    graph.add_edge("load_context", "check_emergency")
    graph.add_edge("clarify", END)          # 澄清后等待用户回复
    graph.add_edge("save_record", END)

    # 注册条件边（路由）
    graph.add_conditional_edges("dispatcher",       route_dispatcher,
                                 {"load_context": "load_context", "parse_answer": "parse_answer",
                                  "extract_symptoms": "extract_symptoms"})
    graph.add_conditional_edges("check_emergency",  route_after_emergency,
                                 {"extract_symptoms": "extract_symptoms", END: END})
    graph.add_conditional_edges("extract_symptoms", route_after_extract,
                                 {"query_neo4j": "query_neo4j", "clarify": "clarify"})
    graph.add_conditional_edges("query_neo4j",      route_after_neo4j,
                                 {"conclude": "conclude", "ask_symptoms": "ask_symptoms"})
    graph.add_conditional_edges("ask_symptoms",     route_after_ask,
                                 {"conclude": "conclude", END: END})
    graph.add_conditional_edges("parse_answer",     route_after_parse,
                                 {"query_neo4j": "query_neo4j"})
    graph.add_conditional_edges("conclude",         route_after_conclude,
                                 {"save_record": "save_record", END: END})
    return graph.compile()


# ════════════════════════════════════════════════════════════════════════
# 对外接口：供 call_inquiry_agent 工具调用
# ════════════════════════════════════════════════════════════════════════

async def run_inquiry(
    user_message: str,
    thread_id: str,
    deps: InquiryDeps,
    existing_state: InquiryState | None = None,
    user_id: str | None = None,
    long_term_memories: list[str] | None = None,
) -> tuple[str, InquiryState]:
    """
    执行一轮问诊对话。

    Args:
        user_message      : 用户本轮输入
        thread_id        : thread_id（用于关联 Redis 和 PostgreSQL 记录）
        deps              : 依赖注入容器
        existing_state    : 上一轮的状态（多轮对话时传入）
        user_id           : 来自 UserContext 的用户 ID，贯穿整个问诊流程
        long_term_memories: Supervisor 从 Milvus 检索到的长期记忆摘要

    Returns:
        (assistant_reply, new_state)
    """
    graph = build_inquiry_graph(deps)

    if existing_state is None:
        # 首轮：初始化状态
        state = InquiryState(
            session_id=thread_id,
            patient_context=PatientContext(
                patient_id=user_id,
                long_term_memories=long_term_memories or [],
            ),
        )
    else:
        state = existing_state

    # 追加用户消息
    state.messages.append(HumanMessage(content=user_message))

    logger.info("▶ run_inquiry 开始 | session={} round={} entry={} user_id={}",
                thread_id, state.round,
                "load_context" if state.round == 0 else "parse_answer", user_id)

    # 执行图（dispatcher 节点会根据 round 自动路由到正确的入口）
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(state, config=config)

    new_state = InquiryState(**result) if isinstance(result, dict) else result

    # 取最后一条 AI 消息作为回复
    reply = ""
    for msg in reversed(new_state.messages):
        if isinstance(msg, AIMessage):
            reply = msg.content
            break

    logger.info("◀ run_inquiry 完成 | session={} phase={} round={} reply_len={}",
                thread_id, new_state.phase, new_state.round, len(reply))
    return reply, new_state
def build_inquiry_deps(db_session=None) -> InquiryDeps:
    """
    构建问诊依赖注入容器的工厂函数。
    供 call_inquiry_agent 工具和 FastAPI 路由共用，避免重复代码。
    """
    llm = get_llm(temperature=0.3)
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()  # 确保连接已建立
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return InquiryDeps(
        llm=llm,
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
    )

def build_inquiry_deps(db_session=None) -> InquiryDeps:
    """
    构建问诊依赖注入容器的工厂函数。
    供 call_inquiry_agent 工具和 FastAPI 路由共用，避免重复代码。
    """
    llm = get_llm(temperature=0.3)
    embedding_model = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()  # 确保连接已建立
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return InquiryDeps(
        llm=llm,
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        db_session=db_session,
    )

def route_dispatcher(state: InquiryState) -> str:
    if state.round == 0:
        return "load_context"
    return "parse_answer"
