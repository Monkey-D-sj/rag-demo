from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.document.controller as ctrl
import rag.api.modules.document.service as svc
from rag.api.dependencies.db import get_pg
from rag.api.dependencies.storage import get_minio, get_task_publisher
from rag.api.common.error_handlers import register_error_handlers


class _FakeTaskPublisher:
    def __init__(self):
        self.jobs = []

    async def enqueue(self, name, document_id, **kwargs):
        self.jobs.append((name, document_id, kwargs))


def _build_app():
    app = FastAPI()
    app.include_router(ctrl.document_router)
    register_error_handlers(app)
    app.dependency_overrides[get_pg] = lambda: object()
    return app


def _app(task_publisher):
    app = _build_app()
    app.dependency_overrides[get_minio] = lambda: object()
    app.dependency_overrides[get_task_publisher] = lambda: task_publisher
    return app


def test_upload_rejects_unknown_type():
    task_publisher = _FakeTaskPublisher()
    client = TestClient(_app(task_publisher))
    resp = client.post(
        "/documents/", files={"file": ("a.exe", b"x", "application/octet-stream")},
        data={"knowledge_base_id": "00000000-0000-0000-0000-000000000002"},
    )
    assert resp.status_code == 400
    assert task_publisher.jobs == []


def test_upload_happy_path_enqueues(monkeypatch):
    created = {}

    async def fake_put(client, bucket, key, data, content_type):
        created["key"] = key
        created["data"] = data

    async def fake_create_document(pool, **kw):
        created.update(kw)
        return "doc-1"

    monkeypatch.setattr(svc, "put_object", fake_put)
    monkeypatch.setattr(svc.store, "create_document", fake_create_document)

    task_publisher = _FakeTaskPublisher()
    client = TestClient(_app(task_publisher))
    resp = client.post(
        "/documents/", files={"file": ("note.txt", b"hello", "text/plain")},
        data={"knowledge_base_id": "00000000-0000-0000-0000-000000000002"},
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body == {"document_id": "doc-1", "status": "pending"}
    assert task_publisher.jobs == [("ingest_document", "doc-1", {})]
    assert created["content_type"] == "txt"
    assert created["data"] == b"hello"


def test_get_status_returns_404_when_missing(monkeypatch):
    async def fake_get(pool, doc_id):
        return None

    monkeypatch.setattr(svc.store, "get_document", fake_get)

    client = TestClient(_build_app())
    resp = client.get("/documents/missing-id")
    assert resp.status_code == 404


def test_get_status_returns_doc(monkeypatch):
    async def fake_get(pool, doc_id):
        return {
            "id": "doc-1", "filename": "a.txt", "status": "done",
            "chunk_count": 3, "error": None,
        }

    monkeypatch.setattr(svc.store, "get_document", fake_get)

    client = TestClient(_build_app())
    resp = client.get("/documents/doc-1")
    assert resp.status_code == 200
    assert resp.json() == {
        "document_id": "doc-1", "filename": "a.txt", "status": "done",
        "chunk_count": 3, "error": None, "graph_status": None, "graph_error": None,
    }
