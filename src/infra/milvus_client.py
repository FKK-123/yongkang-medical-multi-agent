from loguru import logger
from pymilvus import connections, utility

from src.core.config import get_settings

settings = get_settings()
MILVUS_ALIAS = "yongkang_milvus"


def get_milvus_client_alias() -> str:
    """
    返回 Milvus 连接别名。
    若连接不存在会自动创建，供业务层复用。
    """
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    return MILVUS_ALIAS


def get_milvus_dependency() -> str:
    """
    FastAPI Depends 注入用。
    返回可复用的 Milvus 连接别名。
    """
    return get_milvus_client_alias()


def check_milvus_health() -> bool:
    """
    检查 Milvus 连通性。
    通过拉取集合列表验证连接是否可用。
    """
    alias = get_milvus_client_alias()
    collections = utility.list_collections(using=alias)
    logger.info(f"Milvus 连接成功，当前集合数量：{len(collections)}")
    return True


def close_milvus_client() -> None:
    """关闭 Milvus 连接。"""
    connections.disconnect(alias=MILVUS_ALIAS)
