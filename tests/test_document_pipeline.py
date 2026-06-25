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
    claimed = []

    async def fake_claim(pool, doc_id):
        claimed.append(doc_id)
        return True

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k1", "content_type": "txt", "knowledge_base_id": "kb1"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        completed["embedded"] = embedded
        completed["kb"] = kb

    async def fake_get_object(client, bucket, key):
        return b"ignored-by-fake-parse"

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "full text")
    monkeypatch.setattr(pipe, "chunk", lambda text, size, overlap: ["a", "b", "c"])

    emb = _FakeEmbedding()
    ctx = {
        "pg": None, "minio": None, "bucket": "b",
        "embedding": emb, "settings": _settings(batch=2),
    }

    await pipe.ingest_document(ctx, "d1")

    assert claimed == ["d1"]
    assert statuses == []  # happy path 不写 set_status(只在失败时写)
    assert emb.batches == [["a", "b"], ["c"]]
    assert completed["kb"] == "kb1"
    assert completed["embedded"] == [
        (0, "a", [0.0, 0.0, 0.0, 0.0]),
        (1, "b", [0.0, 0.0, 0.0, 0.0]),
        (2, "c", [0.0, 0.0, 0.0, 0.0]),
    ]


async def test_ingest_skips_when_not_claimed(monkeypatch):
    calls = []

    async def fake_claim(pool, doc_id):
        return False

    async def fail_if_called(*a, **k):
        calls.append(a)
        raise AssertionError("未领取时不应继续处理")

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fail_if_called)
    monkeypatch.setattr(pipe.store, "set_status", fail_if_called)

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _FakeEmbedding(), "settings": _settings()}

    await pipe.ingest_document(ctx, "d1")  # 不抛、直接返回
    assert calls == []


async def test_ingest_empty_chunks_marks_failed(monkeypatch):
    statuses = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb"}

    async def fake_get_object(*a, **k):
        return b"data"

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "")
    monkeypatch.setattr(pipe, "chunk", lambda text, size, overlap: [])

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _FakeEmbedding(), "settings": _settings()}

    with pytest.raises(ValueError):
        await pipe.ingest_document(ctx, "d1")
    assert len(statuses) == 1
    assert statuses[0][0] == "failed" and statuses[0][1] is not None
