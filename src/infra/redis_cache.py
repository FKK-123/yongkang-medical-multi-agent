import redis.asyncio as redis
from src.core.config import get_settings

settings = get_settings()

# 模块级别创建连接池（应用启动时初始化一次）
redis_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=True,
    encoding="utf-8",
)

# 模块级别创建 Redis 客户端实例（复用连接池）; _ 开头认为是 私有变量，不暴露出去
_redis_client = redis.Redis(connection_pool=redis_pool)

# RedisSaver（checkpointer）专用连接池：必须 decode_responses=False（bytes 模式）
# 与业务连接池共享相同的地址配置，但保持独立的连接池，互不干扰
_checkpointer_pool = redis.ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD or None,
    decode_responses=False,  # RedisSaver 要求 bytes，不能 decode
)

_checkpointer_client = redis.Redis(connection_pool=_checkpointer_pool)

async def get_redis_client() -> redis.Redis:
    """
    FastAPI Depends 注入用。
    直接返回模块级别的 Redis 客户端实例，不需要每次创建新实例。
    连接池会自动管理连接的获取和归还。
    """
    return _redis_client


def get_checkpointer_redis() -> redis.Redis:
    """
    返回供 RedisSaver（LangGraph checkpointer）专用的 Redis 客户端。
    decode_responses=False，以 bytes 模式运行，与业务 Redis 客户端隔离。
    """
    return _checkpointer_client