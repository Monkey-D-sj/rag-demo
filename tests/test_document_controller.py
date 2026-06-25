from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.document.controller as ctrl
import rag.api.modules.document.service as svc
from rag.api.dependence.db import get_pg
from rag.api.dependence.storage import get_arq_pool, get_minio
from rag.api.common.error_handlers import register_error_handlers


class _FakeArq:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, name, *args):
        self.jobs.append((name, args))


def _build_app():
    app = FastAPI()
    app.include_router(ctrl.document_router)
    register_error_handlers(app)
    app.dependency_overrides[get_pg] = lambda: object()
    return app


def _app(arq):
    app = _build_app()
    app.dependency_overrides[get_minio] = lambda: object()
    app.dependency_overrides[get_arq_pool] = lambda: arq
    return app


def test_upload_rejects_unknown_type():
    arq = _FakeArq()
    client = TestClient(_app(arq))
    resp = client.post(
        "/documents/", files={"file": ("a.exe", b"x", "application/octet-stream")}
    )
    assert resp.status_code == 400
    assert arq.jobs == []


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

    arq = _FakeArq()
    client = TestClient(_app(arq))
    resp = client.post(
        "/documents/", files={"file": ("note.txt", b"hello", "text/plain")}
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body == {"document_id": "doc-1", "status": "pending"}
    assert arq.jobs == [("ingest_document", ("doc-1",))]
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
        "chunk_count": 3, "error": None,
    }
