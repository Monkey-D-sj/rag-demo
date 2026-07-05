from __future__ import annotations

import asyncio
from typing import TypedDict

from neo4j import AsyncDriver
from psycopg_pool import AsyncConnectionPool

from rag.common.logging import get_logger
from rag.config import Settings
from rag.document import store
from rag.document.entity_extraction import ExtractionResult, extract_entities
from rag.graph.aggregate import aggregate
from rag.graph.store import purge_document, write_graph
from rag.models.base import ChatModel

logger = get_logger()


class _ExtractDeps(TypedDict):
    """`extract_document_entities` 所需依赖,由 arq worker ctx 注入。"""
    pg: AsyncConnectionPool
    neo4j: AsyncDriver
    llm: ChatModel
    settings: Settings

# demo 为中文语料(如西游记);显式指定中文输出,避免抽取默认语言。
_LANGUAGE = "中文"


async def extract_document_entities(ctx: _ExtractDeps, document_id: str) -> None:
    """独立 arq 任务:抽取整篇文档实体/关系并写入 Neo4j。

    best-effort(阶段一):失败置 graph_status=failed 记日志,不 re-raise、不重试。
    与向量入库 status 独立,不影响已 done 的检索能力。
    """
    pool = ctx["pg"]
    driver = ctx["neo4j"]
    llm = ctx["llm"]
    database = ctx["settings"].NEO4J_DATABASE

    if not await store.claim_graph_processing(pool, document_id):
        logger.info("图抽取非待处理状态或已被领取,跳过: %s", document_id)
        return

    try:
        logger.info("开始抽取文档 %s 实体/关系", document_id)
        logger.info("开始从数据库获取文档 %s 的所有 chunk", document_id)
        chunks = await store.get_chunks_for_graph(pool, document_id)
        logger.info("文档 %s 有 %d 个 chunk", document_id, len(chunks))

        # 并发抽取:Semaphore 限制同时在飞的 LLM 请求数,防限流。
        # 单 chunk 失败仅记日志并跳过(返回 None),不拖垮整篇;顺序无关,aggregate 按 name 合并。
        concurrency = ctx["settings"].GRAPH_EXTRACT_CONCURRENCY
        sem = asyncio.BoundedSemaphore(concurrency)

        async def _extract_one(row) -> tuple[str, ExtractionResult] | None:
            chunk_uid = f"{document_id}:{row['chunk_index']}"
            async with sem:
                try:
                    result = await extract_entities(
                        llm, row["text"],
                        chapter_context=row.get("title"),
                        language=_LANGUAGE,
                    )
                except Exception:  # noqa: BLE001 - 单 chunk 失败跳过,best-effort
                    logger.exception("chunk %s 抽取失败,已跳过", chunk_uid)
                    return None
                return (chunk_uid, result)

        results = await asyncio.gather(*(_extract_one(row) for row in chunks))
        extracted = [r for r in results if r is not None]

        failed = len(chunks) - len(extracted)
        if chunks and not extracted:
            # 全部 chunk 抽取失败(疑似 LLM 整体不可用):不 purge 旧图,
            # 抛出走 except 分支标 failed,由 cron 退避重试自愈。
            raise RuntimeError(f"全部 {len(chunks)} 个 chunk 抽取失败")
        if failed:
            logger.warning(
                "文档 %s 有 %d/%d 个 chunk 抽取失败,已跳过",
                document_id, failed, len(chunks),
            )

        entities, relations = aggregate(extracted)

        await purge_document(driver, database, document_id)
        await write_graph(driver, database, document_id, entities, relations)

        await store.set_graph_status(pool, document_id, "done")
        logger.info(
            "图抽取完成: %s 实体 %d 关系 %d",
            document_id, len(entities), len(relations),
        )
    except Exception as e:  # noqa: BLE001 - best-effort,阶段一不 re-raise
        logger.exception("图抽取失败: %s", document_id)
        await store.set_graph_status(pool, document_id, "failed", error=str(e)[:500])
