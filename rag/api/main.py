from contextlib import asynccontextmanager
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from rag.api.common.error_handlers import register_error_handlers
from rag.api.modules import register_modules
from rag.common.logging import get_logger, setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import get_settings
from rag.db import create_pg_pool, create_redis_client
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel

logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.check_required()

    # ------ 初始化 pg -------
    logger.info("初始化 pg 数据库连接池")
    pool = await create_pg_pool(settings)
    app.state.pg = pool
    logger.info("pg 数据库连接池初始化完成")

    # ------ 初始化 redis -------
    logger.info("初始化 redis 连接池")
    app.state.redis = create_redis_client(settings)
    logger.info("redis 连接池初始化完成")

    # ------ 初始化 agent 依赖单例 -------
    logger.info("初始化 agent 依赖单例")
    logger.info("初始化嵌入模型")
    embedding = EmbeddingModel(settings)
    logger.info("嵌入模型初始化完成")
    logger.info("初始化内存管理器")
    app.state.memory_manager = MemoryManager(
        long_term=PgVectorLongTermMemory(pool, embedding),
        short_term=RedisShortTermMemory(app.state.redis),
    )
    logger.info("内存管理器初始化完成")
    logger.info("初始化 LLM 模型")
    app.state.llm = NormalModel(settings)
    logger.info("LLM 模型初始化完成")
    logger.info("初始化知识检索器")
    app.state.retriever = KnowledgeRetriever(pool, embedding)
    logger.info("知识检索器初始化完成")

    # ------ 初始化对象存储与任务队列 -------
    logger.info("初始化对象存储与任务队列")
    logger.info("初始化 minio 客户端")
    app.state.minio = create_minio_client(settings)
    logger.info("minio 客户端初始化完成")
    logger.info("初始化 arq 任务队列")
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            database=settings.ARQ_REDIS_DB,
            password=settings.REDIS_PASSWORD,
        )
    )

    yield

    # ------ 关闭资源 -------
    logger.info("关闭资源")
    logger.info("关闭 pg 数据库连接池")
    await pool.close()
    logger.info("关闭 pg 数据库连接池完成")
    logger.info("关闭 redis 连接池")
    await app.state.redis.aclose()
    logger.info("关闭 redis 连接池完成")
    logger.info("关闭 arq 任务队列")
    await app.state.arq_pool.aclose()
    logger.info("关闭 arq 任务队列完成")
    logger.info("关闭 minio 客户端")
    await app.state.minio.close()
    logger.info("关闭 minio 客户端完成")

def start_app() -> FastAPI:
    logger.info("rag-demo 启动")
    app = FastAPI(lifespan=lifespan)
    logger.info("注册模块")
    register_modules(app)
    logger.info("模块注册完成")
    logger.info("注册错误处理函数")
    register_error_handlers(app)
    logger.info("错误处理函数注册完成")

    return app
