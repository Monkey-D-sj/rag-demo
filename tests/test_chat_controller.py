import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.chat.controller as controller_mod
import rag.api.modules.chat.service as service_mod
from rag.agent.type import StreamEventType, stream_event
from rag.api.dependencies.agent import get_llm, get_memory_manager, get_retriever
from rag.common.exception import friendly_message


async def _fake_invoke(session_id, query, context, config=None):
    yield stream_event(StreamEventType.STATUS, "检索记忆中...")
    yield stream_event(StreamEventType.STATUS, "深度思考中")
    yield stream_event(StreamEventType.STATUS, "检索知识库中...")
    yield stream_event(StreamEventType.STATUS, "生成回答中")
    yield stream_event(StreamEventType.MESSAGE, "关务")
    yield stream_event(StreamEventType.MESSAGE, "信息")


async def _boom_invoke(session_id, query, context, config=None):
    yield stream_event(StreamEventType.STATUS, "检索记忆中...")
    raise RuntimeError("llm down")


def _parse_sse(text: str) -> list[str]:
    return [
        line[len("data: ") :]
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _client(monkeypatch, fake_invoke):
    monkeypatch.setattr(service_mod, "invoke", fake_invoke)
    app = FastAPI()
    app.include_router(controller_mod.chat_router)
    app.dependency_overrides[get_memory_manager] = lambda: object()
    app.dependency_overrides[get_llm] = lambda: object()
    app.dependency_overrides[get_retriever] = lambda: object()
    app.state.pg = None  # controller 传 pool=request.app.state.pg
    return TestClient(app)


def test_chat_controller_streams_chain(monkeypatch):
    client = _client(monkeypatch, _fake_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]

    # 验证事件顺序
    types = [e["type"] for e in events]
    assert types == [StreamEventType.STATUS, StreamEventType.STATUS, StreamEventType.STATUS, StreamEventType.STATUS, StreamEventType.MESSAGE, StreamEventType.MESSAGE]

    assert events[4] == stream_event(StreamEventType.MESSAGE, "关务")
    assert events[5] == stream_event(StreamEventType.MESSAGE, "信息")


def test_chat_controller_emits_error_frame(monkeypatch):
    client = _client(monkeypatch, _boom_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]
    assert events[-1] == stream_event(StreamEventType.ERROR, friendly_message(RuntimeError("llm down")))
