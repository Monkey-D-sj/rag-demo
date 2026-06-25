from typing import TypedDict

from arq.connections import RedisSettings
from minio import Minio
from psycopg_pool import AsyncConnectionPool

from rag.common.logging import setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import get_settings
from rag.db import create_pg_pool
from rag.document.pipeline import ingest_document
from rag.models.embedding import EmbeddingModel
from rag.config import Settings

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


class WorkerSettings:
    functions = [ingest_document]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_tries = 3
    job_timeout = 300
    redis_settings = RedisSettings(
        host=_settings.redis_host,
        port=_settings.redis_port,
        database=_settings.arq_redis_db,
        password=_settings.redis_password,
    )
