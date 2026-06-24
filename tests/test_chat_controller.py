from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.chat.controller as controller_mod
from rag.api.dependence.agent import get_llm, get_memory_manager


async def _fake_invoke(session_id, query, context):
    yield {"recall_memory": {"context": "c"}}
    yield {"handle_query": {"rewrite_query": "rw"}}


def test_chat_controller_runs_chain(monkeypatch):
    monkeypatch.setattr(controller_mod, "invoke", _fake_invoke)

    app = FastAPI()
    app.include_router(controller_mod.chat_router)
    app.dependency_overrides[get_memory_manager] = lambda: object()
    app.dependency_overrides[get_llm] = lambda: object()

    client = TestClient(app)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    assert resp.json()["chunks"][-1] == {"handle_query": {"rewrite_query": "rw"}}
