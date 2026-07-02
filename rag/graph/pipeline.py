from __future__ import annotations

from typing import TYPE_CHECKING

from rag.common.logging import get_logger
from rag.document import store
from rag.document.entity_extraction import extract_entities
from rag.graph.aggregate import aggregate
from rag.graph.store import purge_document, write_graph

if TYPE_CHECKING:
    from rag.worker.main import WorkerCtx

logger = get_logger()

# demo 为中文语料(如西游记);显式指定中文输出,避免抽取默认语言。
_LANGUAGE = "中文"


async def extract_document_entities(ctx: "WorkerCtx", document_id: str) -> None:
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
        chunks = await store.get_chunks_for_graph(pool, document_id)
        extracted: list[tuple[str, object]] = []
        for row in chunks:
            chunk_uid = f"{document_id}:{row['chunk_index']}"
            result = await extract_entities(
                llm, row["text"],
                chapter_context=row.get("title"),
                language=_LANGUAGE,
            )
            extracted.append((chunk_uid, result))

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
