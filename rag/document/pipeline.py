from __future__ import annotations

import asyncio
from typing import TypedDict

from minio import Minio
from psycopg_pool import AsyncConnectionPool

from rag.common.logging import get_logger
from rag.common.minio_client import get_object
from rag.config import Settings, SplitStrategy
from rag.document import store
from rag.document.chunker import chunk
from rag.document.parser import parse
from rag.models.embedding import EmbeddingModel

logger = get_logger()


class _IngestDeps(TypedDict):
    """`ingest_document` 所需依赖,由 arq worker ctx 注入。"""
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    settings: Settings
    redis: object  # arq 框架自动注入 ArqRedis,仅用于 enqueue_job


def _parse_and_chunk(
    data: bytes,
    content_type: str,
    strategy: SplitStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, dict[str, str]]]:
    """CPU 密集段(PDF 解析 + 切块)合并到一次调用,由调用方 to_thread 整体 offload。

    统一归一化为 (text, metadata):
    - list[str] 策略 → metadata 为空 {}
    - paragraph_semantic → text 取 content,章节标题存入 metadata["title"]
    """
    text = parse(data, content_type)
    chunks = chunk(strategy, text, chunk_size, chunk_overlap)
    normalized: list[tuple[str, dict[str, str]]] = []
    for c in chunks:
        if isinstance(c, dict):
            normalized.append((c["content"], {"title": c["title"]} if c["title"] else {}))
        else:
            normalized.append((c, {}))
    return normalized


async def ingest_document(ctx: _IngestDeps, document_id: str) -> None:
    """web 投递、worker 执行的入库编排。失败置 failed 并 re-raise 供重试。"""
    pool = ctx["pg"]
    minio = ctx["minio"]
    bucket = ctx["bucket"]
    embedding = ctx["embedding"]
    settings = ctx["settings"]

    # 原子领取:并发/重复投递时只有一个任务能拿到,其余跳过,避免重复入库
    if not await store.claim_for_processing(pool, document_id):
        logger.info("文档非待处理状态或已被其他任务领取,跳过: %s", document_id)
        return
    try:
        doc = await store.get_document(pool, document_id)
        if doc is None:
            raise ValueError(f"document not found: {document_id}")

        data = await get_object(minio, bucket, doc["object_key"])
        # parse + chunk 是纯 CPU(GIL 活),合并到一次线程池调用,避免阻塞 event loop
        chunks = await asyncio.to_thread(
            _parse_and_chunk,
            data,
            doc["content_type"],
            settings.SPLIT_STRATEGY,
            settings.CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("切块结果为空,无可入库内容")

        embedded: list[tuple[int, str, list[float], dict[str, str]]] = []
        batch = settings.EMBEDDING_BATCH_SIZE
        index = 0
        for i in range(0, len(chunks), batch):
            window = chunks[i : i + batch]
            vectors = await embedding.embed([text for text, _ in window])
            for (text_piece, meta), vector in zip(window, vectors):
                embedded.append((index, text_piece, vector, meta))
                index += 1

        await store.store_chunks_and_complete(
            pool, document_id, doc["knowledge_base_id"], embedded
        )

        # 向量入库已完成;实体图抽取为 best-effort 增强,投递独立任务(阶段一)。
        if settings.ENABLE_ENTITY_EXTRACTION:
            await ctx["redis"].enqueue_job("extract_document_entities", document_id)
        else:
            await store.set_graph_status(pool, document_id, "skipped")
    except asyncio.CancelledError:
        # arq job_timeout 通过 cancel 中断(CancelledError 属 BaseException,
        # 不被下方 except Exception 捕获)。必须置 failed 后 re-raise,否则文档
        # 永久卡在 processing:cron 退避重试只扫 failed,claim 的 stale 回收也无人触发。
        logger.warning("文档入库被取消(疑似超时): %s", document_id)
        await store.set_status(pool, document_id, "failed", error="cancelled/timeout")
        raise
    except Exception as e:
        logger.exception("文档入库失败: %s", document_id)
        await store.set_status(pool, document_id, "failed", error=str(e)[:500])
        raise
