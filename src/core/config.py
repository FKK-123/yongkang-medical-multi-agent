# 读取环境变量
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "yongkang-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "medical"
    DB_PASSWORD: str = "medical123"
    DB_NAME: str = "medical_db"

    #minerU
    MINERU_BACKEND: str = ""
    MINERU_API_URL: str = ""
    MINERU_TIMEOUT: int = 22

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medical123"

    # 模型
    DASHSCOPE_API_KEY: str = "sk-ws-H.PRMLIPY.US1S.MEYCIQCUAMmtrrQyRm0Bnp1lHY-jugqbTrQqa-nArUAkI5Ho2gIhALCnPsUk2-Azq1GBbZWweAF4e3f_r-N45jIzyc0Ws1z0"
       # 聊天模型
    BASE_URL_CHAT: str = "https://api.deepseek.com"
    DEEPSEEK_API_KEY: str = "sk-8363b6642218431484308528376517c3"
    CHAT_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    VL_MODEL: str = "qwen-vl"

    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # 指定环境变量文件
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# 保存到内存缓存中。以后直接获取。
@lru_cache  # lru 把对象实例保存到内存中。这是一种单例的实现
def get_settings() -> Settings:
    return Settings()

def get_llm(temperature: float = 0.3):
    """Create the configured chat model used by worker agents."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    # DashScope's compatible endpoint uses the OpenAI API protocol.  Using
    # ChatDeepSeek here would ignore BASE_URL_CHAT and send qwen-* models to
    # the DeepSeek endpoint.
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=temperature,
    )
