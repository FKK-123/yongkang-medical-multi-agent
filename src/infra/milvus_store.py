import json
import asyncio
import time
from typing import Any, Iterable
from langgraph.store.base import (
    BaseStore, Item, SearchItem,
    GetOp, PutOp, SearchOp, ListNamespacesOp,
    Op, Result,
)
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)
from langchain_core.embeddings import Embeddings

# 集合名
COLLECTION_NAME = "agent_long_term_memory"


# 保证集合存在
def _ensure_collection(alias: str, dims: int) -> Collection:
    """确保 Milvus Collection 存在，不存在则创建。alias 为 Milvus 数据库名。默认是 default。"""
    if utility.has_collection(COLLECTION_NAME, using=alias):
        return Collection(COLLECTION_NAME, using=alias)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="key", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="value_json", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dims),
        FieldSchema(name="created_at", dtype=DataType.DOUBLE),
    ]
    schema = CollectionSchema(fields, description="Agent long-term memory")
    collection = Collection(COLLECTION_NAME, schema=schema, using=alias)

    # 创建向量索引
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    collection.load()
    return collection


class MilvusStore(BaseStore):
    """
    基于 Milvus 的长期记忆 Store，支持语义相似度检索。

    用法：
        store = MilvusStore(
            alias="_milvus",
            embeddings=your_embedding_model,
            dims=1536,
        )
        agent = create_agent(model=llm, tools=tools, store=store)
    """

    def __init__(
        self,
        alias: str,
        embeddings: Embeddings,
        dims: int = 1024,
    ):
        self._alias = alias
        self._embeddings = embeddings
        self._dims = dims
        self._collection = _ensure_collection(alias, dims)

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #

    def _ns_to_str(self, namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    def _make_id(self, namespace: tuple[str, ...], key: str) -> str:
        return f"{self._ns_to_str(namespace)}::{key}"

    def _embed_text(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)

    async def _aembed_text(self, text: str) -> list[float]:
        return await self._embeddings.aembed_query(text)

    def _item_from_hit(self, hit: dict) -> SearchItem:
        value = json.loads(hit["value_json"])
        ns_str = hit["namespace"]
        namespace = tuple(ns_str.split("/")) if ns_str else ()
        return SearchItem(
            namespace=namespace,
            key=hit["key"],
            value=value,
            created_at=hit.get("created_at"),
            updated_at=hit.get("created_at"),
            score=hit.get("score"),
        )

    # ------------------------------------------------------------------ #
    # BaseStore 必须实现的两个核心方法：batch / abatch
    # 其余 get/put/search/delete 都由 BaseStore 默认实现调用 batch
    # ------------------------------------------------------------------ #

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """同步批量执行操作。
        注意：使用 asyncio.run() 而不是 get_event_loop().run_until_complete()。
        因为 LangGraph 在执行 sync tool 时会放入 thread pool，
        而 thread pool 的线程没有 event loop，get_event_loop() 会抛 RuntimeError。
        asyncio.run() 每次都创建独立的新 event loop，在任意线程中都能安全运行。
        """
        return asyncio.run(self.abatch(list(ops)))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """异步批量执行操作（核心实现）。"""
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                await self._aput(op.namespace, op.key, op.value)
                results.append(None)
            elif isinstance(op, SearchOp):
                items = await self._asearch(op.namespace_prefix, op.query, op.limit)
                results.append(items)
            elif isinstance(op, ListNamespacesOp):
                results.append([])  # 简化实现
            else:
                results.append(None)
        return results

    # ------------------------------------------------------------------ #
    # 具体操作实现
    # ------------------------------------------------------------------ #

    async def _aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any] | None,
    ) -> None:
        """写入或删除一条记忆。"""
        doc_id = self._make_id(namespace, key)

        # value 为 None 表示删除
        if value is None:
            expr = f'id == "{doc_id}"'
            self._collection.delete(expr)
            return

        # 将 value 序列化为文本用于生成向量
        text_for_embed = json.dumps(value, ensure_ascii=False)
        embedding = await self._aembed_text(text_for_embed)

        # 先删除旧记录（upsert 语义）
        self._collection.delete(f'id == "{doc_id}"')

        # 插入新记录
        self._collection.insert([{
            "id": doc_id,
            "namespace": self._ns_to_str(namespace),
            "key": key,
            "value_json": text_for_embed,
            "embedding": embedding,
            "created_at": time.time(),
        }])
        self._collection.flush()

    async def _aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Item | None:
        """精确查询一条记忆。"""
        doc_id = self._make_id(namespace, key)
        results = self._collection.query(
            expr=f'id == "{doc_id}"',
            output_fields=["id", "namespace", "key", "value_json", "created_at"],
        )
        if not results:
            return None
        row = results[0]
        return Item(
            namespace=namespace,
            key=key,
            value=json.loads(row["value_json"]),
            created_at=row.get("created_at"),
            updated_at=row.get("created_at"),
        )

    async def _asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: str | None,
        limit: int = 10,
    ) -> list[SearchItem]:
        """语义相似度检索记忆。"""
        ns_filter = self._ns_to_str(namespace_prefix)

        if not query:
            # 无查询词时按 namespace 前缀列举
            results = self._collection.query(
                expr=f'namespace like "{ns_filter}%"',
                output_fields=["id", "namespace", "key", "value_json", "created_at"],
                limit=limit,
            )
            return [self._item_from_hit(r) for r in results]

        # 有查询词时做向量相似度检索
        query_vec = await self._aembed_text(query)
        search_results = self._collection.search(
            data=[query_vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=limit,
            expr=f'namespace like "{ns_filter}%"' if ns_filter else None,
            output_fields=["id", "namespace", "key", "value_json", "created_at"],
        )

        items = []
        for hit in search_results[0]:
            entity = hit.entity
            value = json.loads(entity.get("value_json", "{}"))
            ns_str = entity.get("namespace", "")
            namespace = tuple(ns_str.split("/")) if ns_str else ()
            items.append(SearchItem(
                namespace=namespace,
                key=entity.get("key", ""),
                value=value,
                created_at=entity.get("created_at"),
                updated_at=entity.get("created_at"),
                score=hit.score,
            ))
        return items