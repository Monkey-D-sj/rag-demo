from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from rag.api.common.error_handlers import register_error_handlers
from rag.api.modules import register_modules
from rag.common.logging import setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import get_settings
from rag.db import create_pg_pool, create_redis_client
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()

    # ------ 初始化 pg -------
    pool = await create_pg_pool(settings)
    app.state.pg = pool

    # ------ 初始化 redis -------
    app.state.redis = create_redis_client(settings)

    # ------ 初始化 agent 依赖单例 -------
    embedding = EmbeddingModel(settings)
    app.state.memory_manager = MemoryManager(
        long_term=PgVectorLongTermMemory(pool, embedding),
        short_term=RedisShortTermMemory(app.state.redis),
    )
    app.state.llm = NormalModel(settings)
    app.state.retriever = KnowledgeRetriever(pool, embedding)

    # ------ 初始化对象存储与任务队列 -------
    app.state.minio = create_minio_client(settings)
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=settings.redis_host,
            port=settings.redis_port,
            database=settings.arq_redis_db,
            password=settings.redis_password,
        )
    )

    yield

    # ------ 关闭资源 -------
    await pool.close()
    await app.state.redis.aclose()
    await app.state.arq_pool.aclose()


app = FastAPI(lifespan=lifespan)
register_modules(app)
register_error_handlers(app)
