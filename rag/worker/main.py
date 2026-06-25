import logging

from arq.connections import RedisSettings
from arq.cron import cron
from minio import Minio
from psycopg_pool import AsyncConnectionPool
from typing import TypedDict

from rag.common.logging import setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import Settings, get_settings
from rag.db import create_pg_pool
from rag.document import store
from rag.document.pipeline import ingest_document
from rag.models.embedding import EmbeddingModel

logger = logging.getLogger(__name__)

_settings = get_settings()


class WorkerCtx(TypedDict):
    settings: Settings
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel

async def on_startup(ctx: dict) -> None:
    setup_logging()
    settings = get_settings()
    settings.check_required()
    ctx["settings"] = settings
    ctx["pg"] = await create_pg_pool(settings)
    ctx["minio"] = create_minio_client(settings)
    ctx["bucket"] = settings.minio_bucket
    ctx["embedding"] = EmbeddingModel(settings)


async def on_shutdown(ctx: dict) -> None:
    await ctx["pg"].close()


async def retry_failed_documents(ctx: dict) -> None:
    """自愈 cron:每 5 min 扫描 failed 文档,按指数退避重新入库。

    退避公式: backoff_base * 2^retry_count 秒
    retry_count 达 max_retry_rounds 后放弃(真·死信),需人工通过 API retry 介入。
    """
    settings: Settings = ctx["settings"]
    ids = await store.claim_failed_for_retry(
        ctx["pg"], settings.max_retry_rounds, settings.retry_backoff_base
    )
    for doc_id in ids:
        try:
            await ingest_document(ctx, doc_id)
        except Exception:
            logger.exception("自愈重试失败: %s", doc_id)


class WorkerSettings:
    functions = [ingest_document]
    on_startup = on_startup
    on_shutdown = on_shutdown
    cron_jobs = [
        cron(retry_failed_documents, minute="*/5")  # 每 5 分钟扫描一次
    ]
    max_tries = 3
    job_timeout = 300
    # 入库以 CPU 为主(PDF 解析/切块, GIL 串行),单进程高并发无益反增 GIL 抖动;
    # 控制在低并发,吞吐靠多开 worker 进程横向扩展。
    max_jobs = 4
    redis_settings = RedisSettings(
        host=_settings.redis_host,
        port=_settings.redis_port,
        database=_settings.arq_redis_db,
        password=_settings.redis_password,
    )
