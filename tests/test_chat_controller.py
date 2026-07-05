import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.chat.controller as controller_mod
import rag.api.modules.chat.service as service_mod
from rag.api.dependencies.agent import get_llm, get_memory_manager, get_retriever


async def _fake_invoke(session_id, query, context):
    yield {"type": "status", "data": "检索记忆中..."}
    yield {"type": "status", "data": "深度思考中"}
    yield {"type": "status", "data": "检索知识库中..."}
    yield {"type": "status", "data": "生成回答中"}
    yield {"type": "message", "data": "关务"}
    yield {"type": "message", "data": "信息"}


async def _boom_invoke(session_id, query, context):
    yield {"type": "status", "data": "检索记忆中..."}
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
    return TestClient(app)


def test_chat_controller_streams_chain(monkeypatch):
    client = _client(monkeypatch, _fake_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]

    # 不应包含 update 事件
    assert all(e["type"] in ("status", "message", "error") for e in events)

    # 验证事件顺序
    types = [e["type"] for e in events]
    assert types == ["status", "status", "status", "status", "message", "message"]

    assert events[4] == {"type": "message", "data": "关务"}
    assert events[5] == {"type": "message", "data": "信息"}


def test_chat_controller_emits_error_frame(monkeypatch):
    client = _client(monkeypatch, _boom_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]
    assert events[-1] == {"type": "error", "data": "llm down"}
