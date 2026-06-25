from types import SimpleNamespace

import pytest

import rag.document.pipeline as pipe


class _FakeEmbedding:
    def __init__(self):
        self.batches = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _settings(batch=2):
    return SimpleNamespace(chunk_size=800, chunk_overlap=100, embedding_batch_size=batch)


async def test_ingest_happy_path_batches_and_completes(monkeypatch):
    statuses = []
    completed = {}

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k1", "content_type": "txt", "knowledge_base_id": "kb1"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        completed["embedded"] = embedded
        completed["kb"] = kb

    async def fake_get_object(client, bucket, key):
        return b"ignored-by-fake-parse"

    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "full text")
    async def fake_chunk(text, size, overlap):
        return ["a", "b", "c"]

    monkeypatch.setattr(pipe, "chunk", fake_chunk)

    emb = _FakeEmbedding()
    ctx = {
        "pg": None, "minio": None, "bucket": "b",
        "embedding": emb, "settings": _settings(batch=2),
    }

    await pipe.ingest_document(ctx, "d1")

    assert statuses[0] == ("processing", None)
    assert emb.batches == [["a", "b"], ["c"]]
    assert completed["kb"] == "kb1"
    assert completed["embedded"] == [
        (0, "a", [0.0, 0.0, 0.0, 0.0]),
        (1, "b", [0.0, 0.0, 0.0, 0.0]),
        (2, "c", [0.0, 0.0, 0.0, 0.0]),
    ]


async def test_ingest_empty_chunks_marks_failed(monkeypatch):
    statuses = []

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb"}

    async def fake_get_object(*a, **k):
        return b"data"

    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "")
    async def fake_chunk_empty(text, size, overlap):
        return []

    monkeypatch.setattr(pipe, "chunk", fake_chunk_empty)

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _FakeEmbedding(), "settings": _settings()}

    with pytest.raises(ValueError):
        await pipe.ingest_document(ctx, "d1")
    assert statuses[0] == ("processing", None)
    assert any(s == "failed" and e is not None for s, e in statuses)
