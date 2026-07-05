from arq.connections import RedisSettings
from arq.cron import cron
from arq.worker import func, run_worker
from minio import Minio
from neo4j import AsyncDriver
from psycopg_pool import AsyncConnectionPool
from typing import TypedDict

from rag.common.logging import get_logger, setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import Settings, get_settings
from rag.db import create_pg_pool
from rag.db.neo4j import create_neo4j_driver, ensure_graph_constraints
from rag.document import store
from rag.document.pipeline import ingest_document
from rag.graph.pipeline import extract_document_entities
from rag.models.base import ChatModel
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel

logger = get_logger()


class WorkerCtx(TypedDict):
    settings: Settings
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    neo4j: AsyncDriver
    llm: ChatModel

async def on_startup(ctx: dict) -> None:
    setup_logging()
    settings = get_settings()
    settings.check_required()
    ctx["settings"] = settings
    ctx["pg"] = await create_pg_pool(settings)
    ctx["minio"] = create_minio_client(settings)
    ctx["bucket"] = settings.MINIO_BUCKET
    ctx["embedding"] = EmbeddingModel(settings)
    ctx["neo4j"] = create_neo4j_driver(settings)
    ctx["llm"] = NormalModel(settings)
    await ensure_graph_constraints(ctx["neo4j"], settings.NEO4J_DATABASE)


async def on_shutdown(ctx: dict) -> None:
    await ctx["pg"].close()
    await ctx["neo4j"].close()


async def retry_failed_documents(ctx: dict) -> None:
    """自愈 cron:每 5 min 扫描 failed 文档,按指数退避重新入库。

    退避公式: backoff_base * 2^retry_count 秒
    retry_count 达 max_retry_rounds 后放弃(真·死信),需人工通过 API retry 介入。
    """
    settings: Settings = ctx["settings"]
    ids = await store.claim_failed_for_retry(
        ctx["pg"], settings.MAX_RETRY_ROUNDS, settings.RETRY_BACKOFF_BASE
    )
    for doc_id in ids:
        try:
            await ingest_document(ctx, doc_id)
        except Exception:
            logger.exception("自愈重试失败: %s", doc_id)


class WorkerSettings:
    functions = [func(ingest_document), func(extract_document_entities, timeout=900)]
    on_startup = on_startup
    on_shutdown = on_shutdown
    cron_jobs = [
        cron(retry_failed_documents, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55})
    ]
    max_tries = 3
    job_timeout = 300
    # 入库以 CPU 为主(PDF 解析/切块, GIL 串行),单进程高并发无益反增 GIL 抖动;
    # 控制在低并发,吞吐靠多开 worker 进程横向扩展。
    max_jobs = 4
    redis_settings = RedisSettings(
        host=get_settings().REDIS_HOST,
        port=get_settings().REDIS_PORT,
        database=get_settings().ARQ_REDIS_DB,
        password=get_settings().REDIS_PASSWORD,
    )


def run() -> None:
    """``rag-worker`` console_scripts 入口 —— 启动 arq worker。

    也可直接 ``arq rag.worker.main.WorkerSettings`` 启动。
    """
    import sys

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    from rag.common.logging import setup_logging

    setup_logging()
    run_worker(WorkerSettings)
