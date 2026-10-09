# src/agents/knowledge/doc_ingestion.py

from __future__ import annotations
import hashlib
from loguru import logger
from langchain_core.embeddings import Embeddings
from pymilvus import (
    MilvusClient, DataType,
)

from src.agents.knowledge.doc_rag import COLLECTION_NAME

EMBEDDING_DIM = 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def ensure_knowledge_collection(milvus_client: MilvusClient) -> None:
    """确保 knowledge_docs collection 存在。"""
    if milvus_client.has_collection(COLLECTION_NAME):
        return

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
    schema.add_field("category", DataType.VARCHAR, max_length=100)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"Collection '{COLLECTION_NAME}' 创建成功")


async def _parse_with_mineru(file_path: str, file_name: str) -> str | None:
    """尝试用 MinerU 解析文档，失败则返回 None。"""
    try:
        from src.agents.knowledge.mineru_client import parse_document
        md_text = await parse_document(file_path, file_name)
        if md_text and len(md_text.strip()) > 10:
            logger.info(f"MinerU 解析成功: {file_name} ({len(md_text)} chars)")
            return md_text
    except Exception as e:
        logger.warning(f"MinerU 解析失败，回退到 LlamaIndex: {e}")
    return None


async def _parse_with_llamaindex(file_path: str) -> list:
    """用 LlamaIndex 解析文档（兜底方案）。"""
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_files=[file_path])
    return reader.load_data()


def _split_markdown(md_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    对 MinerU 输出的 Markdown 做智能分块。
    按段落（双换行）分割，再按 chunk_size 合并。
    """
    paragraphs = md_text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
            else:
                current = para
                continue
            current = ""

    if current:
        chunks.append(current)

    return chunks


async def ingest_file(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
) -> int:
    """
    导入单个文档到 Milvus。
    优先使用 MinerU 解析（高精度），失败回退到 LlamaIndex。
    """
    from llama_index.core.node_parser import SentenceSplitter

    ensure_knowledge_collection(milvus_client)
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]

    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )

    # 优先 MinerU，失败回退 LlamaIndex
    md_text = await _parse_with_mineru(file_path, doc_name)

    if md_text:
        chunks = _split_markdown(md_text)
        texts = chunks
        pages = [0] * len(chunks)
    else:
        documents = await _parse_with_llamaindex(file_path)
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(documents)
        if not nodes:
            logger.warning(f"文档 '{doc_name}' 解析后无内容")
            return 0
        texts = [node.get_content() for node in nodes]
        pages = []
        for node in nodes:
            page = node.metadata.get("page_label", 0)
            try:
                pages.append(int(page))
            except (ValueError, TypeError):
                pages.append(0)

    if not texts:
        return 0

    batch_size = 50
    all_data = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings = await embedding_model.aembed_documents(batch_texts)

        for j, (text_content, emb) in enumerate(zip(batch_texts, embeddings)):
            chunk_idx = i + j
            all_data.append({
                "id": f"{doc_id}_{chunk_idx}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "category": category,
                "page_number": pages[chunk_idx] if chunk_idx < len(pages) else 0,
                "chunk_index": chunk_idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

    milvus_client.insert(collection_name=COLLECTION_NAME, data=all_data)
    logger.info(f"文档 '{doc_name}' 导入完成，共 {len(all_data)} 个分块")
    return len(all_data)


async def ingest_drug_instructions(
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session,
) -> int:
    """从 PostgreSQL drug_details 表导入药品说明书到 Milvus。"""
    from sqlalchemy import select
    from src.modules.medical.model import Drug, DrugDetail
    from llama_index.core.node_parser import SentenceSplitter

    ensure_knowledge_collection(milvus_client)

    result = await db_session.execute(
        select(Drug.name, DrugDetail.full_instruction)
        .join(DrugDetail, Drug.id == DrugDetail.drug_id)
        .where(DrugDetail.full_instruction.isnot(None))
    )
    rows = result.all()

    if not rows:
        logger.warning("drug_details 表中无说明书数据")
        return 0

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    total = 0

    for drug_name, instruction in rows:
        if not instruction or not instruction.strip():
            continue

        doc_id = hashlib.md5(f"drug_{drug_name}".encode()).hexdigest()[:16]
        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
        )

        from llama_index.core import Document
        doc = Document(text=instruction, metadata={"drug_name": drug_name})
        nodes = splitter.get_nodes_from_documents([doc])
        texts = [n.get_content() for n in nodes]

        if not texts:
            continue

        embeddings = await embedding_model.aembed_documents(texts)
        data = []
        for idx, (text_content, emb) in enumerate(zip(texts, embeddings)):
            data.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": f"{drug_name}说明书",
                "doc_type": "drug_instruction",
                "category": "药剂科",
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

        milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
        total += len(data)

    logger.info(f"药品说明书导入完成，共 {total} 个分块")
    return total