"""RabbitMQ consumer for document ingestion and graph extraction tasks."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import TypedDict

from minio import Minio
from neo4j import AsyncDriver
from psycopg_pool import AsyncConnectionPool

from rag.agent.cache import purge_expired
from rag.common.logging import get_logger, setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import Settings, get_settings
from rag.db import create_pg_pool, create_redis_client
from rag.db.neo4j import create_neo4j_driver, ensure_graph_constraints
from rag.document import store
from rag.document.pipeline import ingest_document
from rag.governance import create_guard
from rag.graph.pipeline import extract_document_entities
from rag.models.base import ChatModel
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel
from rag.tasks import (
    DocumentTask,
    RabbitMQTaskPublisher,
    TaskPublisher,
    decode_document_task,
    declare_task_queues,
)

logger = get_logger()


class WorkerCtx(TypedDict):
    settings: Settings
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    neo4j: AsyncDriver | None
    llm: ChatModel
    redis: object
    guard: object | None
    task_publisher: TaskPublisher


async def on_startup(ctx: dict) -> None:
    setup_logging()
    settings = get_settings()
    ctx["settings"] = settings
    ctx["pg"] = await create_pg_pool(settings)
    ctx["minio"] = create_minio_client(settings)
    ctx["bucket"] = settings.MINIO_BUCKET
    ctx["redis"] = create_redis_client(settings)
    ctx["guard"] = create_guard(ctx["redis"], ctx["pg"], source="worker")
    ctx["embedding"] = EmbeddingModel(settings, guard=ctx["guard"])
    if settings.NEO4J_ENABLED:
        ctx["neo4j"] = create_neo4j_driver(settings)
        await ensure_graph_constraints(ctx["neo4j"], settings.NEO4J_DATABASE)
    else:
        ctx["neo4j"] = None
    ctx["llm"] = NormalModel(settings, guard=ctx["guard"])
    ctx["task_publisher"] = await RabbitMQTaskPublisher.connect(settings)


async def on_shutdown(ctx: dict) -> None:
    closers = (
        ("RabbitMQ", ctx.get("task_publisher"), "close"),
        ("PostgreSQL", ctx.get("pg"), "close"),
        ("Neo4j", ctx.get("neo4j"), "close"),
        ("Redis", ctx.get("redis"), "aclose"),
    )
    for name, resource, method in closers:
        if resource is None:
            continue
        try:
            await getattr(resource, method)()
        except Exception:  # noqa: BLE001 - 继续释放其余资源
            logger.warning("关闭 %s 失败", name, exc_info=True)


async def retry_failed_documents(ctx: dict) -> None:
    """每五分钟扫描 failed/stalled 文档并重新投递，状态领取保持幂等。"""
    settings: Settings = ctx["settings"]
    failed_ids = await store.claim_failed_for_retry(
        ctx["pg"], settings.MAX_RETRY_ROUNDS, settings.RETRY_BACKOFF_BASE
    )
    stalled_ids = await store.find_stalled_documents(
        ctx["pg"], settings.STALE_DOC_SECONDS
    )
    for document_id in dict.fromkeys([*failed_ids, *stalled_ids]):
        try:
            await ctx["task_publisher"].enqueue("ingest_document", document_id)
        except Exception:
            logger.exception("自愈重试投递失败: %s", document_id)


async def purge_semantic_cache(ctx: dict) -> None:
    """每小时物理删除过期语义缓存行。"""
    settings: Settings = ctx["settings"]
    try:
        await purge_expired(ctx["pg"], settings.SEMANTIC_CACHE_TTL_HOURS)
    except Exception:  # noqa: BLE001 - 清理失败等下一轮
        logger.warning("语义缓存过期清理失败", exc_info=True)


async def _extract_wrapper(ctx: dict, document_id: str) -> None:
    """neo4j 未启用时跳过实体抽取，避免运行时连接错误。"""
    if ctx.get("neo4j") is None:
        logger.info("neo4j 未启用，跳过实体抽取: %s", document_id)
        return
    await extract_document_entities(ctx, document_id)


def _timeout_for(task_name: str) -> int:
    return 900 if task_name == "extract_document_entities" else 300


async def _execute_task(ctx: dict, task: DocumentTask) -> None:
    async with asyncio.timeout(_timeout_for(task.task_name)):
        if task.task_name == "ingest_document":
            await ingest_document(ctx, task.document_id)
        else:
            await _extract_wrapper(ctx, task.document_id)


async def handle_message(ctx: dict, message) -> None:
    """Handle one delivery with manual acknowledgement and bounded retries."""
    try:
        task = decode_document_task(message.body, message.headers)
    except ValueError:
        logger.error("丢弃格式非法的 RabbitMQ 任务消息", exc_info=True)
        await message.reject(requeue=False)
        return

    try:
        await _execute_task(ctx, task)
    except asyncio.CancelledError:
        # 进程关闭时不要确认消息；RabbitMQ 会将未确认消息重新投递。
        raise
    except Exception:
        settings: Settings = ctx["settings"]
        if task.retry_count + 1 < settings.RABBITMQ_MAX_ATTEMPTS:
            logger.exception(
                "任务失败，重新投递（%d/%d）: %s %s",
                task.retry_count + 1,
                settings.RABBITMQ_MAX_ATTEMPTS - 1,
                task.task_name,
                task.document_id,
            )
            try:
                await ctx["task_publisher"].enqueue(
                    task.task_name, task.document_id, retry_count=task.retry_count + 1
                )
            except Exception:
                logger.exception("任务重投失败，将原消息放回队列: %s", task.document_id)
                await message.nack(requeue=True)
            else:
                await message.ack()
        else:
            logger.exception("任务重试用尽，投递死信队列: %s", task.document_id)
            await message.reject(requeue=False)
    else:
        await message.ack()


async def _run_periodically(ctx: dict, stop_event: asyncio.Event, interval_seconds: int, callback) -> None:
    """Run a maintenance callback on a relative interval until shutdown."""
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            await callback(ctx)


def _install_stop_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            # Windows 的 Proactor loop 不支持 add_signal_handler；KeyboardInterrupt
            # 仍会让 asyncio.run 取消任务并进入 finally。
            pass


async def _run_worker() -> None:
    ctx: dict = {}
    stop_event = asyncio.Event()
    periodic_tasks: list[asyncio.Task] = []
    queue = None
    consumer_tag = None
    consumer_channel = None

    try:
        await on_startup(ctx)
        settings: Settings = ctx["settings"]
        publisher: RabbitMQTaskPublisher = ctx["task_publisher"]
        consumer_channel = await publisher.connection.channel()
        await consumer_channel.set_qos(prefetch_count=settings.RABBITMQ_PREFETCH_COUNT)
        queue = await declare_task_queues(consumer_channel, settings)
        consumer_tag = await queue.consume(lambda message: handle_message(ctx, message))
        periodic_tasks = [
            asyncio.create_task(_run_periodically(ctx, stop_event, 300, retry_failed_documents)),
            asyncio.create_task(_run_periodically(ctx, stop_event, 3600, purge_semantic_cache)),
        ]
        _install_stop_handlers(stop_event)
        logger.info(
            "RabbitMQ worker 已启动: queue=%s prefetch=%d",
            settings.RABBITMQ_QUEUE,
            settings.RABBITMQ_PREFETCH_COUNT,
        )
        await stop_event.wait()
    finally:
        if queue is not None and consumer_tag is not None:
            with contextlib.suppress(Exception):
                await queue.cancel(consumer_tag)
        for task in periodic_tasks:
            task.cancel()
        if periodic_tasks:
            await asyncio.gather(*periodic_tasks, return_exceptions=True)
        if consumer_channel is not None:
            with contextlib.suppress(Exception):
                await consumer_channel.close()
        await on_shutdown(ctx)


def run() -> None:
    """``rag-worker`` console_scripts 入口。"""
    from rag.common.platform import setup_windows_loop

    setup_windows_loop()
    asyncio.run(_run_worker())
