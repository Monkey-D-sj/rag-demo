"""Chat SSE 流式协议 SDK。

提供 ChatStream 类——async context manager + async iterable，
封装 SSE 格式化、事件校验、自动收尾。

生产者-消费者模式：生产者（业务代码 / LangGraph workflow）通过
status()/message()/error()/send_event() 入队；消费者（FastAPI
StreamingResponse）通过 __aiter__ 出队。调用 close() 发送 [DONE] 并
终止迭代。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from pydantic import BaseModel

from rag.agent.type import StreamEventType


# ── 事件模型 ────────────────────────────────────────────


class ChatStatus(BaseModel):
    type: StreamEventType = StreamEventType.STATUS
    data: str


class ChatMessage(BaseModel):
    type: StreamEventType = StreamEventType.MESSAGE
    data: str


class ChatError(BaseModel):
    type: StreamEventType = StreamEventType.ERROR
    data: str


class ChatCitations(BaseModel):
    type: StreamEventType = StreamEventType.CITATIONS
    data: list[dict]


ChatEvent = ChatStatus | ChatMessage | ChatError | ChatCitations

# 事件类型 → Pydantic 模型映射
_EVENT_MODELS: dict[StreamEventType, type[BaseModel]] = {
    StreamEventType.STATUS: ChatStatus,
    StreamEventType.MESSAGE: ChatMessage,
    StreamEventType.ERROR: ChatError,
    StreamEventType.CITATIONS: ChatCitations,
}


# ── JSON 序列化辅助 ──────────────────────────────────────


def _json_default(obj: object) -> str:
    """处理 json.dumps 无法序列化的常见类型（UUID、datetime 等）。"""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── ChatStream ──────────────────────────────────────────


class ChatStream:
    """Chat SSE 流式响应封装。

    直接使用（手动管理 + 后台生产者）:

        stream = ChatStream()

        async def _produce():
            try:
                async for event in invoke(...):
                    await stream.send_event(event)
            except Exception as e:
                await stream.error(str(e))
            finally:
                stream.close()

        asyncio.create_task(_produce())
        async for sse_line in stream:
            yield sse_line

    async with 语法（适合直接调用 status/message/error 的场景）:

        async with ChatStream() as stream:
            await stream.status("检索记忆中…")
            ...
        # __aexit__ 自动 close()
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False

    # ── 类型化发送方法 ─────────────────────────────────

    async def status(self, text: str) -> None:
        """发送阶段提示事件。"""
        event = ChatStatus(data=text)
        await self._enqueue(event)

    async def message(self, text: str) -> None:
        """发送逐字 token 事件。"""
        event = ChatMessage(data=text)
        await self._enqueue(event)

    async def error(self, text: str) -> None:
        """发送错误事件。"""
        event = ChatError(data=text)
        await self._enqueue(event)

    async def send_event(self, raw: dict) -> None:
        """适配层：将 LangGraph writer 产出的原始 dict 转为 SSE 事件入队。

        已知类型（status / message / error）通过 Pydantic 校验后序列化；
        未知类型静默跳过。
        """
        raw_type = raw.get("type", "")
        try:
            event_type = StreamEventType(raw_type)
        except ValueError:
            return
        model_cls = _EVENT_MODELS.get(event_type)
        if model_cls is None:
            return
        event = model_cls(**raw)
        await self._enqueue(event)

    # ── 流生命周期 ─────────────────────────────────────

    def close(self) -> None:
        """关闭流：入队 [DONE] + sentinel，终止 __aiter__。

        幂等：多次调用只生效一次。
        """
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait("data: [DONE]\n\n")
        self._queue.put_nowait(None)  # sentinel

    # ── 内部 ───────────────────────────────────────────

    async def _enqueue(self, event: ChatEvent) -> None:
        """将事件序列化为 SSE 行并入队。"""
        payload = json.dumps(
            event.model_dump(), ensure_ascii=False, default=_json_default
        )
        await self._queue.put(f"data: {payload}\n\n")

    # ── Async Context Manager ──────────────────────────

    async def __aenter__(self) -> "ChatStream":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> None:
        if exc is not None:
            await self.error(str(exc))
        self.close()

    # ── Async Iterable（供 FastAPI StreamingResponse 消费）──

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
