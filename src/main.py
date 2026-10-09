from fastapi import Depends, FastAPI
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import text

from src.infra.minio_client import ensure_bucket_exists, get_minio_client
from src.infra.milvus_client import (
    check_milvus_health,
    close_milvus_client,
    get_milvus_dependency,
)
from src.infra.neo4j_client import (
    check_neo4j_health,
    close_neo4j_driver,
    get_neo4j_driver,
)
from src.core.config import get_settings
from src.middlewares.logging import LoggingMiddleware
from src.core.exceptions import register_exception_handlers
from src.core.logger import setup_logger
from src.infra.database import engine
from src.infra.redis_cache import get_redis_client
from loguru import logger



# 使用上下文管理器感知项目的生命周期
from contextlib import asynccontextmanager
from src.api.routers.chat import router as chat_router
from src.api.routers.knowledge import router as knowledge_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger() # 配置日志组件
    settings = get_settings()
    logger.info(f"{settings.APP_NAME} 启动.. | 使用环境： {settings.APP_ENV}")
    # 应用启动时执行
    # 确保 MinIO 桶的存在
    try:
        ensure_bucket_exists()
    except Exception as e:
        logger.error(f"minio 桶创建失败，文件上传功能将无法使用：{e}")
    try:
        check_milvus_health()
    except Exception as e:
        logger.error(f"milvus 连接失败，向量检索功能将无法使用：{e}")
    try:
        await check_neo4j_health()
    except Exception as e:
        logger.error(f"neo4j 连接失败，图谱功能将无法使用：{e}")

    yield
    # 应用关闭时执行
    # 关闭数据库连接池
    await close_neo4j_driver()
    close_milvus_client()
    await engine.dispose()
    logger.info(f"{settings.APP_NAME} 关闭.. ")


def create_app() -> FastAPI:
    settings = get_settings()

    # 创建应用
    # 注册生命周期管理器
    app = FastAPI(title=settings.APP_NAME, 
                  version="1.0.0",
                  debug=settings.APP_DEBUG,
                  lifespan=lifespan,
                  )


    # 注册中间件
    app.add_middleware(LoggingMiddleware)
    # 注册跨域中间件
    app.add_middleware(CORSMiddleware,
                       allow_origins=["http://localhost:3000","http://localhost:5173"],
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"])



    # 注册异常处理器
    register_exception_handlers(app)

    # 注册路由
    app.include_router(chat_router)
    app.include_router(knowledge_router)
    # app.include_router(user_router, prefix="/api/v1")


    return app

# 创建fastapi 应用
app = create_app()

# 健康检查路由； 能访问通，就代表应用启动
@app.get("/health")
async def root():
    # prometheus 规范约束，返回的 status 必须是 ok，代表健康
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps(milvus_alias: str = Depends(get_milvus_dependency)):
    """检查核心依赖可用性（PostgreSQL / Redis / MinIO / Milvus / Neo4j）。"""
    settings = get_settings()
    result = {
        "status": "ok",
        "dependencies": {
            "postgres": {"ok": True, "error": ""},
            "redis": {"ok": True, "error": ""},
            "minio": {"ok": True, "error": ""},
            "milvus": {"ok": True, "error": ""},
            "neo4j": {"ok": True, "error": ""},
        },
    }

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        result["dependencies"]["postgres"] = {"ok": False, "error": str(e)}

    try:
        redis_client = await get_redis_client()
        await redis_client.ping()
    except Exception as e:
        result["dependencies"]["redis"] = {"ok": False, "error": str(e)}

    try:
        minio_client = get_minio_client()
        minio_client.bucket_exists(settings.MINIO_BUCKET)
    except Exception as e:
        result["dependencies"]["minio"] = {"ok": False, "error": str(e)}

    try:
        if not milvus_alias:
            raise RuntimeError("milvus alias is empty")
        check_milvus_health()
    except Exception as e:
        result["dependencies"]["milvus"] = {"ok": False, "error": str(e)}

    try:
        neo4j_driver = get_neo4j_driver()
        await neo4j_driver.verify_connectivity()
    except Exception as e:
        result["dependencies"]["neo4j"] = {"ok": False, "error": str(e)}

    if any(not item["ok"] for item in result["dependencies"].values()):
        result["status"] = "degraded"

    return result
