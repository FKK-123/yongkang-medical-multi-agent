# scripts/init_symptom_index.py
"""
症状向量索引初始化脚本。
首次部署时执行一次，之后 Neo4j 新增症状时增量执行。

用法：
    python scripts/init_symptom_index.py
"""

import asyncio
import sys
import os

from langchain_community.embeddings import DashScopeEmbeddings

# 把项目根目录加入 Python 路径，确保能 import src 下的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility, MilvusClient
)
from neo4j import AsyncGraphDatabase
from src.core.config import get_settings

settings = get_settings()

COLLECTION_NAME = "symptom_index"
EMBEDDING_DIM = 1024  # text-embedding-v3 维度，与 MilvusStore 保持一致
MILVUS_ALIAS = "symptom_init"


def ensure_symptom_collection(alias: str) -> Collection:
    """
    确保 symptom_index collection 存在。
    已存在则直接返回，不存在则创建 Schema + 向量索引。
    """
    if utility.has_collection(COLLECTION_NAME, using=alias):
        print(f"[INFO] collection '{COLLECTION_NAME}' 已存在，跳过创建。")
        col = Collection(COLLECTION_NAME, using=alias)
        col.load()
        return col

    fields = [
        # 用症状名作为主键，天然去重，更新时直接 upsert
        FieldSchema(name="id",        dtype=DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema(name="name",      dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="Neo4j Symptom nodes vector index")
    col = Collection(COLLECTION_NAME, schema=schema, using=alias)
    col.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    col.load()
    print(f"[INFO] collection '{COLLECTION_NAME}' 创建成功。")
    return col


async def fetch_all_symptoms(neo4j_driver) -> list[str]:
    """从 Neo4j 取出所有 Symptom 节点的名称。"""
    async with neo4j_driver.session() as session:
        result = await session.run("MATCH (s:Symptom) RETURN s.name AS name")
        records = await result.data()
    names = [r["name"] for r in records if r["name"]]
    print(f"[INFO] 从 Neo4j 获取到 {len(names)} 个症状节点。")
    return names


def get_embedding_model():
    """
    返回 Embedding 模型实例。
    项目使用 DeepSeek 兼容的 OpenAI 接口 + text-embedding-v3。
    """
    return DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
    )


async def build_symptom_index():
    """主流程：全量构建症状向量索引。"""
    # 1. 连接 Milvus
    connections.connect(
        alias=MILVUS_ALIAS,
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    col = ensure_symptom_collection(MILVUS_ALIAS)

    # 2. 连接 Neo4j，获取所有症状名
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    symptom_names = await fetch_all_symptoms(neo4j_driver)
    await neo4j_driver.close()

    if not symptom_names:
        print("[WARN] Neo4j 中没有 Symptom 节点，请先导入医疗数据。")
        return

    # 3. 批量向量化（每批 100 个，避免单次请求过大）
    embedding_model = get_embedding_model()
    batch_size = 100
    all_data = []
    for i in range(0, len(symptom_names), batch_size):
        batch = symptom_names[i: i + batch_size]
        embeddings = await embedding_model.aembed_documents(batch)
        for name, emb in zip(batch, embeddings):
            all_data.append({"id": name, "name": name, "embedding": emb})
        print(f"[INFO] 已向量化 {min(i + batch_size, len(symptom_names))}/{len(symptom_names)}")

    # 4. 写入 Milvus（upsert 语义：先删同名旧记录，再插入）
    # pymilvus 的 Collection.upsert 在 v2.x 中等价于 delete + insert
    col.upsert(all_data)
    col.flush()
    print(f"[INFO] 症状索引构建完成，共写入 {len(all_data)} 条记录。")

    # 5. 断开连接
    connections.disconnect(alias=MILVUS_ALIAS)


if __name__ == "__main__":
    asyncio.run(build_symptom_index())