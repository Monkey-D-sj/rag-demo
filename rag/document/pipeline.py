from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from minio import Minio
from psycopg_pool import AsyncConnectionPool

from rag.agent.cache import clear_semantic_cache
from rag.common.logging import get_logger
from rag.common.minio_client import get_object, get_object_to_file
from rag.config import Settings
from rag.document.chunker import SplitStrategy, chunk
from rag.document import store
from rag.document.parser import parse
from rag.document.table_extractor import (
    TableBlock,
    extract_tables,
    generate_table_summary,
)
from rag.models.base import ChatModel
from rag.models.embedding import EmbeddingModel

from rag.document import REGULATION_KB_ID

logger = get_logger()


class _IngestDeps(TypedDict, total=False):
    """`ingest_document` 所需依赖,由 arq worker ctx 注入。"""
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    settings: Settings
    llm: ChatModel  # 可选，用于表格摘要生成；缺失时降级为规则摘要
    task_publisher: object  # RabbitMQ 任务发布器，仅用于投递实体抽取任务


def _parse_and_chunk(
    data: str | bytes,
    content_type: str,
    strategy: SplitStrategy,
    chunk_size: int,
    chunk_overlap: int,
    doc_title: str = "",
) -> tuple[str, list[tuple[str, dict[str, str]]]]:
    """CPU 密集段(PDF 解析 + 切块)合并到一次调用,由调用方 to_thread 整体 offload。

    返回 (full_text, chunks) — full_text 用于 parent-child retrieval 的 document_content 存储。
    """
    full_text = parse(data, content_type)
    chunks: list[tuple[str, dict[str, str]]] = chunk(strategy, full_text, chunk_size, chunk_overlap)
    if doc_title:
        doc_title = doc_title.rsplit(".", 1)[0]  # strip extension
        chunks = [
            (f"《{doc_title}》{text}", {**meta, "document_title": doc_title})
            for text, meta in chunks
        ]
        full_text = f"《{doc_title}》{full_text}"
    return full_text, chunks


async def _extract_and_summarize_tables(
    data: bytes,
    content_type: str,
    llm: ChatModel | None = None,
) -> list[TableBlock]:
    """提取文档中的表格，并（可选）用 LLM 生成自然语言摘要。

    LLM 摘要失败时静默降级为规则摘要（`TableBlock.meta` 已包含），
    不阻塞表格入库。
    """
    # CPU 密集段（PDF 表格解析）offload 到线程池
    blocks = await asyncio.to_thread(extract_tables, data, content_type)
    if not blocks:
        return []

    if llm is not None:
        for tb in blocks:
            try:
                summary = await generate_table_summary(
                    llm, tb.markdown, tb.headers, tb.caption
                )
                if summary:
                    # 覆盖规则摘要为 LLM 生成的更高质版本
                    tb.meta["table_summary"] = summary
            except Exception:
                logger.warning(
                    "表格摘要生成失败（列: %s），降级为规则摘要",
                    tb.headers, exc_info=True,
                )

    return blocks


# 知识库 → 切分策略（硬编码，每个 KB 固定策略）
_KB_STRATEGY: dict[str, SplitStrategy] = {
    REGULATION_KB_ID: SplitStrategy.paragraph_semantic,  # 法规：章→条逐级切分
}


async def _ingest(ctx: _IngestDeps, document_id: str) -> None:
    """入库编排核心逻辑。失败置 failed 并 re-raise 供重试。"""
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

        strategy = _KB_STRATEGY.get(
            str(doc["knowledge_base_id"]), SplitStrategy.paragraph_semantic,
        )

        content_type = doc["content_type"]
        src: str | bytes
        tmp_path: str | None = None

        if content_type in ("pdf", "docx"):
            # PDF/DOCX:流式下载到临时文件,解析库延迟加载,大文件不占内存
            tmp_path = await get_object_to_file(minio, bucket, doc["object_key"])
            src = tmp_path
        else:
            # TXT/MD:文件小,内存加载即可
            src = await get_object(minio, bucket, doc["object_key"])

        try:
            # parse + chunk 是纯 CPU(GIL 活),合并到一次线程池调用,避免阻塞 event loop
            full_text, chunks = await asyncio.to_thread(
                _parse_and_chunk,
                src,
                content_type,
                strategy,
                settings.CHUNK_SIZE,
                settings.CHUNK_OVERLAP,
                doc["filename"],
            )
            if not chunks:
                raise ValueError("切块结果为空,无可入库内容")

            # 存储解析后的全文，供 parent-child retrieval 使用
            await store.set_document_content(pool, document_id, full_text)

            # ── 表格提取（best-effort，不影响正文入库） ──
            table_blocks = await _extract_and_summarize_tables(
                src,
                content_type,
                llm=ctx.get("llm"),
            )
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning("删除临时文件失败: %s", tmp_path)

        # 将表格块追加到 chunk 列表（表格 chunk_index 紧跟正文之后）
        table_start_index = len(chunks)
        for tb in table_blocks:
            chunks.append((tb.embed_text, tb.meta))

        if table_blocks:
            logger.info(
                "文档 %s 附加 %d 个表格 chunk（chunk_index %d-%d）",
                document_id,
                len(table_blocks),
                table_start_index,
                table_start_index + len(table_blocks) - 1,
            )

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

        # 入库成功,历史缓存答案可能已过时,整表失效(best-effort)
        try:
            await clear_semantic_cache(pool)
        except Exception:  # noqa: BLE001 - 缓存失效失败不影响入库结果
            logger.warning("清空语义缓存失败", exc_info=True)

        # 向量入库已完成;实体图抽取为 best-effort 增强,投递独立任务(阶段一)。
        if settings.ENABLE_ENTITY_EXTRACTION:
            await ctx["task_publisher"].enqueue("extract_document_entities", document_id)
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


async def ingest_document(ctx: _IngestDeps, document_id: str) -> None:
    """入库入口：_ingest 内部按 KB 查 _KB_STRATEGY 决定切分策略。"""
    await _ingest(ctx, document_id)
