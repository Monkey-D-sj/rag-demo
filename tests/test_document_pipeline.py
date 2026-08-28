import asyncio
from types import SimpleNamespace

import pytest

import rag.document.pipeline as pipe


class _FakeEmbedding:
    def __init__(self):
        self.batches = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _settings(batch=2, graph=False):
    return SimpleNamespace(
        CHUNK_SIZE=800, CHUNK_OVERLAP=100,
        EMBEDDING_BATCH_SIZE=batch, ENABLE_ENTITY_EXTRACTION=graph,
    )


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
        return {"object_key": "k1", "content_type": "txt", "knowledge_base_id": "kb1", "filename": "test.pdf"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        completed["embedded"] = embedded
        completed["kb"] = kb

    async def fake_set_graph_status(pool, doc_id, status, error=None):
        pass

    async def fake_get_object(client, bucket, key):
        return b"ignored-by-fake-parse"

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe.store, "set_graph_status", fake_set_graph_status)
    async def fake_set_doc_content(pool, doc_id, full_text):
        pass
    monkeypatch.setattr(pipe.store, "set_document_content", fake_set_doc_content)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "full text")
    monkeypatch.setattr(
        pipe, "chunk",
        lambda strategy, text, size, overlap: [("a", {}), ("b", {}), ("c", {})],
    )

    emb = _FakeEmbedding()
    ctx = {
        "pg": None, "minio": None, "bucket": "b",
        "embedding": emb, "settings": _settings(batch=2), "redis": None,
    }

    await pipe.ingest_document(ctx, "d1")

    assert claimed == ["d1"]
    assert statuses == []  # happy path 不写 set_status(只在失败时写)
    assert emb.batches == [["《test》a", "《test》b"], ["《test》c"]]
    assert completed["kb"] == "kb1"
    # filename "test.pdf" 去扩展名后作为标题前缀写入 text 与 meta
    assert completed["embedded"] == [
        (0, "《test》a", [0.0, 0.0, 0.0, 0.0], {"document_title": "test"}),
        (1, "《test》b", [0.0, 0.0, 0.0, 0.0], {"document_title": "test"}),
        (2, "《test》c", [0.0, 0.0, 0.0, 0.0], {"document_title": "test"}),
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
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb", "filename": "test.txt"}

    async def fake_get_object(*a, **k):
        return b"data"

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: [])

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _FakeEmbedding(), "settings": _settings(), "redis": None}

    with pytest.raises(ValueError):
        await pipe.ingest_document(ctx, "d1")
    assert len(statuses) == 1
    assert statuses[0][0] == "failed" and statuses[0][1] is not None


async def test_ingest_cancellation_marks_failed_and_reraises(monkeypatch):
    """arq job_timeout 通过 CancelledError 中断任务:必须置 failed 后 re-raise,
    否则文档永久卡在 processing(cron 只扫 failed)。"""
    statuses = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb", "filename": "test.txt"}

    async def fake_get_object(*a, **k):
        return b"data"

    class _CancelledEmbedding:
        async def embed(self, texts):
            raise asyncio.CancelledError

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    async def fake_set_doc_content(pool, doc_id, full_text):
        pass
    monkeypatch.setattr(pipe.store, "set_document_content", fake_set_doc_content)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: [("a", {})])

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _CancelledEmbedding(), "settings": _settings(batch=8), "redis": None}

    with pytest.raises(asyncio.CancelledError):
        await pipe.ingest_document(ctx, "d1")
    assert len(statuses) == 1
    assert statuses[0][0] == "failed"


async def test_ingest_enqueues_graph_task_when_enabled(monkeypatch):
    enqueued = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb", "filename": "test.txt"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        pass

    async def fake_get_object(*a, **k):
        return b"data"

    class _TaskPublisher:
        async def enqueue(self, name, document_id, **kwargs):
            enqueued.append((name, (document_id,)))

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    async def fake_set_doc_content(pool, doc_id, full_text):
        pass
    monkeypatch.setattr(pipe.store, "set_document_content", fake_set_doc_content)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: [("a", {})])

    ctx = {"pg": None, "minio": None, "bucket": "b", "embedding": _FakeEmbedding(),
           "settings": _settings(graph=True), "task_publisher": _TaskPublisher()}

    await pipe.ingest_document(ctx, "d1")
    assert ("extract_document_entities", ("d1",)) in enqueued


async def test_ingest_clears_semantic_cache_on_success(monkeypatch):
    """入库成功后必须调用 clear_semantic_cache(best-effort)。"""
    calls = []

    async def _fake_clear(pool):
        calls.append(pool)

    async def fake_claim(pool, doc_id):
        return True

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k1", "content_type": "txt", "knowledge_base_id": "kb1", "filename": "test.pdf"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        pass

    async def fake_set_graph_status(pool, doc_id, status, error=None):
        pass

    async def fake_get_object(client, bucket, key):
        return b"ignored-by-fake-parse"

    async def fake_set_doc_content(pool, doc_id, full_text):
        pass

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe.store, "set_graph_status", fake_set_graph_status)
    monkeypatch.setattr(pipe.store, "set_document_content", fake_set_doc_content)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "full text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: [("a", {})])
    monkeypatch.setattr(pipe, "clear_semantic_cache", _fake_clear)

    ctx = {
        "pg": "fake-pool", "minio": None, "bucket": "b",
        "embedding": _FakeEmbedding(), "settings": _settings(batch=2), "redis": None,
    }

    await pipe.ingest_document(ctx, "d1")

    assert calls == ["fake-pool"]


async def test_ingest_marks_graph_skipped_when_disabled(monkeypatch):
    skipped = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb", "filename": "test.txt"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        pass

    async def fake_get_object(*a, **k):
        return b"data"

    async def fake_set_graph_status(pool, doc_id, status, error=None):
        skipped.append((doc_id, status))

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe.store, "set_graph_status", fake_set_graph_status)
    async def fake_set_doc_content(pool, doc_id, full_text):
        pass
    monkeypatch.setattr(pipe.store, "set_document_content", fake_set_doc_content)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: [("a", {})])

    ctx = {"pg": None, "minio": None, "bucket": "b", "embedding": _FakeEmbedding(),
           "settings": _settings(), "redis": None}

    await pipe.ingest_document(ctx, "d1")
    assert ("d1", "skipped") in skipped
