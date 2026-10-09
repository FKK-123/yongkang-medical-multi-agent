from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

def get_user_id(runtime: ToolRuntime):
    # 从 thread_id 中提取用户ID
    if runtime.config.get("configurable"):
        thread_id = runtime.config.get("configurable").get("thread_id")
        if thread_id:

            return thread_id.split(":")[0]
    return None

@tool
async def save_memory(
    content: str,
    runtime: ToolRuntime,
) -> str:
    """
    将重要信息保存到长期记忆中。
    当用户提到重要的个人信息、偏好、病史、过敏史、药物史、手术史等需要跨会话记住的内容时调用。

    Args:
        content: 要记住的内容，用一句话描述
    """

    user_id = get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"

    import time
    key = f"memory_{int(time.time())}"
    await runtime.store.aput(
        namespace=("users", user_id, "memories"),
        key=key,
        value={"content": content, "timestamp": time.time()},
    )
    return f"已记住：{content}"


@tool
async def search_memory(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """
    从长期记忆中检索与问题相关的历史信息。
    当需要回忆用户之前说过的内容、历史病情、偏好等时调用。

    Args:
        query: 检索关键词或问题
    """

    user_id = get_user_id(runtime)
    if not user_id:
        return "无法获取用户ID。"
    # BaseStore.asearch 的 namespace_prefix 是 positional-only 参数。
    results = await runtime.store.asearch(
        ("users", user_id, "memories"),
        query=query,
        limit=5,
    )
    if not results:
        return "没有找到相关记忆。"

    memories = [f"- {item.value['content']}" for item in results]
    return "相关历史记忆：\n" + "\n".join(memories)
