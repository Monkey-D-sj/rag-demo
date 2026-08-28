"""RabbitMQ task publishing primitives shared by the API and worker."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from rag.config import Settings


TaskName = Literal["ingest_document", "extract_document_entities"]
_TASK_NAMES = frozenset({"ingest_document", "extract_document_entities"})


class TaskPublisher(Protocol):
    async def enqueue(
        self,
        task_name: TaskName,
        document_id: str,
        *,
        retry_count: int = 0,
    ) -> None: ...


@dataclass(frozen=True)
class DocumentTask:
    task_name: TaskName
    document_id: str
    retry_count: int = 0


def decode_document_task(body: bytes, headers: dict | None = None) -> DocumentTask:
    """Validate a broker message before it is handed to application code."""
    try:
        payload = json.loads(body)
        task_name = payload["task_name"]
        document_id = payload["document_id"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("任务消息不是有效 JSON") from exc

    if task_name not in _TASK_NAMES or not isinstance(document_id, str) or not document_id:
        raise ValueError("任务消息字段非法")

    raw_retry_count = (headers or {}).get("x-retry-count", 0)
    try:
        retry_count = int(raw_retry_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("任务消息重试计数非法") from exc
    if retry_count < 0:
        raise ValueError("任务消息重试计数非法")

    return DocumentTask(task_name, document_id, retry_count)


async def declare_task_queues(channel, settings: Settings):
    """Idempotently declare the durable work queue and its dead-letter queue."""
    await channel.declare_queue(settings.RABBITMQ_DLQ, durable=True)
    return await channel.declare_queue(
        settings.RABBITMQ_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": settings.RABBITMQ_DLQ,
        },
    )


class RabbitMQTaskPublisher:
    """Persistent publisher with broker confirms via an aio-pika robust connection."""

    def __init__(self, connection, channel, queue_name: str):
        self.connection = connection
        self.channel = channel
        self.queue_name = queue_name

    @classmethod
    async def connect(cls, settings: Settings) -> "RabbitMQTaskPublisher":
        import aio_pika

        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
        channel = await connection.channel(publisher_confirms=True)
        await declare_task_queues(channel, settings)
        return cls(connection, channel, settings.RABBITMQ_QUEUE)

    async def enqueue(
        self,
        task_name: TaskName,
        document_id: str,
        *,
        retry_count: int = 0,
    ) -> None:
        import aio_pika

        if task_name not in _TASK_NAMES:
            raise ValueError(f"未知任务: {task_name}")
        if retry_count < 0:
            raise ValueError("retry_count 不能为负数")

        body = json.dumps(
            {"task_name": task_name, "document_id": document_id},
            separators=(",", ":"),
        ).encode()
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                message_id=uuid.uuid4().hex,
                headers={"x-retry-count": retry_count},
            ),
            routing_key=self.queue_name,
        )

    async def close(self) -> None:
        await self.connection.close()
