from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag.common.minio_client import get_object
from rag.document import store
from rag.document.chunker import chunk
from rag.document.parser import parse

if TYPE_CHECKING:
    # 仅类型注解需要;运行期不导入,避免 pipeline ↔ worker 循环导入
    from rag.worker import WorkerCtx

logger = logging.getLogger(__name__)


async def ingest_document(ctx: WorkerCtx, document_id: str) -> None:
    """web 投递、worker 执行的入库编排。失败置 failed 并 re-raise 供重试。"""
    pool = ctx["pg"]
    minio = ctx["minio"]
    bucket = ctx["bucket"]
    embedding = ctx["embedding"]
    settings = ctx["settings"]

    await store.set_status(pool, document_id, "processing")
    try:
        doc = await store.get_document(pool, document_id)
        if doc is None:
            raise ValueError(f"document not found: {document_id}")

        data = await get_object(minio, bucket, doc["object_key"])
        text = parse(data, doc["content_type"])
        chunks = await chunk(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("切块结果为空,无可入库内容")

        embedded: list[tuple[int, str, list[float]]] = []
        batch = settings.embedding_batch_size
        index = 0
        for i in range(0, len(chunks), batch):
            window = chunks[i : i + batch]
            vectors = await embedding.embed(window)
            for text_piece, vector in zip(window, vectors):
                embedded.append((index, text_piece, vector))
                index += 1

        await store.store_chunks_and_complete(
            pool, document_id, doc["knowledge_base_id"], embedded
        )
    except Exception as e:
        logger.exception("文档入库失败: %s", document_id)
        await store.set_status(pool, document_id, "failed", error=str(e)[:500])
        raise
