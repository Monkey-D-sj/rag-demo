import sys

if sys.platform == "win32":
    import asyncio

    # psycopg 异步只支持 SelectorEventLoop。本模块是 ASGI 入口,reload 开启时
    # uvicorn 的 worker 子进程也会导入它(而不会执行 rag/__main__.py),在此设定
    # policy 才能同时覆盖 reload 开/关两种进程模型。须在任何事件循环创建前执行。
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import AsyncExitStack, asynccontextmanager
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
from rag.db.neo4j import create_neo4j_driver
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel

logger = get_logger()


def _close(name: str, closer):
    """包装资源关闭回调,附带日志。供 AsyncExitStack 登记使用。"""

    async def _callback():
        logger.info("关闭 %s", name)
        await closer()
        logger.info("关闭 %s 完成", name)

    return _callback


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.check_required()

    logger.info("--------------------------------------------------------")
    logger.info("---------------------  初始化依赖项  ---------------------")
    logger.info("--------------------------------------------------------")

    # AsyncExitStack:每创建一个资源就登记其关闭回调,
    # 启动中途失败时按 LIFO 逆序回滚已创建的资源,正常退出时优雅关闭。
    async with AsyncExitStack() as stack:
        # ------ 初始化 pg -------
        logger.info("初始化 pg 数据库连接池")
        pool = await create_pg_pool(settings)
        stack.push_async_callback(_close("pg 数据库连接池", pool.close))
        app.state.pg = pool
        logger.info("pg 数据库连接池初始化完成")

        # ------ 初始化 redis -------
        logger.info("初始化 redis 连接池")
        app.state.redis = create_redis_client(settings)
        stack.push_async_callback(_close("redis 连接池", app.state.redis.aclose))
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
        # minio 为同步 SDK,内部 urllib3 连接池随对象回收,无需显式关闭
        app.state.minio = create_minio_client(settings)
        logger.info("minio 客户端初始化完成")
        logger.info("初始化 neo4j 图数据库")
        app.state.neo4j = create_neo4j_driver(settings)
        stack.push_async_callback(_close("neo4j 图数据库", app.state.neo4j.close))
        logger.info("neo4j 图数据库初始化完成")
        logger.info("初始化 arq 任务队列")
        app.state.arq_pool = await create_pool(
            RedisSettings(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                database=settings.ARQ_REDIS_DB,
                password=settings.REDIS_PASSWORD,
            )
        )
        stack.push_async_callback(_close("arq 任务队列", app.state.arq_pool.aclose))
        logger.info("arq 任务队列初始化完成")
        logger.info("---------------------------------------------------------")
        logger.info("---------------------  初始化依赖项完成  -------------------")
        logger.info("---------------------------------------------------------")

        yield

        # 退出 async with 时,stack 按 LIFO 逆序执行已登记的关闭回调
        logger.info("--------------------------------------------------------")
        logger.info("---------------------  关闭资源  -------------------------")
        logger.info("--------------------------------------------------------")

    logger.info("--------------------------------------------------------")
    logger.info("---------------------  关闭资源完成  ---------------------")
    logger.info("--------------------------------------------------------")



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
