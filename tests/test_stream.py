"""ChatStream 单元测试。"""

from __future__ import annotations

import json

from rag.agent.type import StreamEventType, stream_event
from rag.api.common.stream import ChatStream, ChatStatus, ChatMessage, ChatError


def _parse_sse(text: str) -> list[dict | str]:
    """将 SSE 文本解析为事件列表。"""
    result: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            result.append("[DONE]")
        else:
            result.append(json.loads(payload))
    return result


# ── 事件模型校验 ─────────────────────────────────────────

def test_chat_status_model():
    event = ChatStatus(data="检索记忆中...")
    d = event.model_dump()
    assert d == stream_event(StreamEventType.STATUS, "检索记忆中...")


def test_chat_message_model():
    event = ChatMessage(data="关务")
    d = event.model_dump()
    assert d == stream_event(StreamEventType.MESSAGE, "关务")


def test_chat_error_model():
    event = ChatError(data="超时")
    d = event.model_dump()
    assert d == stream_event(StreamEventType.ERROR, "超时")


# ── ChatStream 正常流程 ─────────────────────────────────

async def test_stream_status_and_message():
    async with ChatStream() as stream:
        await stream.status("检索中...")
        await stream.message("你好")

    # 消费 SSE 输出
    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == stream_event(StreamEventType.STATUS, "检索中...")
    assert events[1] == stream_event(StreamEventType.MESSAGE, "你好")
    assert events[2] == "[DONE]"


async def test_stream_error_then_done():
    async with ChatStream() as stream:
        await stream.error("出错了")

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == stream_event(StreamEventType.ERROR, "出错了")
    assert events[1] == "[DONE]"


async def test_stream_auto_error_on_exception():
    """__aexit__ 收到异常时自动发送 error + [DONE]。"""
    from rag.common.exception import friendly_message

    stream = ChatStream()
    try:
        async with stream:
            raise RuntimeError("llm down")
    except RuntimeError:
        pass  # 异常仍会传播

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == stream_event(StreamEventType.ERROR, friendly_message(RuntimeError("llm down")))
    assert events[-1] == "[DONE]"


async def test_stream_empty():
    """空流只有 [DONE]。"""
    stream = ChatStream()
    async with stream:
        pass  # 什么都不发

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)
    assert events == ["[DONE]"]


# ── send_event 适配层 ────────────────────────────────────

async def test_send_event_known_types():
    async with ChatStream() as stream:
        await stream.send_event(stream_event(StreamEventType.STATUS, "s1"))
        await stream.send_event(stream_event(StreamEventType.MESSAGE, "m1"))
        await stream.send_event(stream_event(StreamEventType.ERROR, "e1"))

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == stream_event(StreamEventType.STATUS, "s1")
    assert events[1] == stream_event(StreamEventType.MESSAGE, "m1")
    assert events[2] == stream_event(StreamEventType.ERROR, "e1")
    assert events[3] == "[DONE]"


async def test_send_event_skips_unknown():
    """未知事件类型（如旧的 'update'）被静默跳过。"""
    async with ChatStream() as stream:
        await stream.send_event({"type": "update", "node": "x", "data": {}})
        await stream.send_event(stream_event(StreamEventType.STATUS, "ok"))

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert len(events) == 2
    assert events[0] == stream_event(StreamEventType.STATUS, "ok")
    assert events[1] == "[DONE]"


async def test_send_event_missing_type():
    """无 type 字段的事件被跳过。"""
    async with ChatStream() as stream:
        await stream.send_event({"data": "no type"})
        await stream.send_event(stream_event(StreamEventType.STATUS, "ok"))

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events == [stream_event(StreamEventType.STATUS, "ok"), "[DONE]"]


# ── done 不重复 ────────────────────────────────────────────

async def test_done_not_duplicated():
    """多次调用 error/status 后再正常退出，[DONE] 只发一次。"""
    async with ChatStream() as stream:
        await stream.status("s")
        await stream.status("s2")

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)
    done_count = sum(1 for e in events if e == "[DONE]")
    assert done_count == 1
