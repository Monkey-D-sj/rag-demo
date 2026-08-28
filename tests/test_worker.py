from types import SimpleNamespace

import rag.worker.main as wm
from rag.tasks import DocumentTask


class _FakeMessage:
    def __init__(self, body, headers=None):
        self.body = body
        self.headers = headers or {}
        self.actions = []

    async def ack(self):
        self.actions.append(("ack",))

    async def nack(self, *, requeue):
        self.actions.append(("nack", requeue))

    async def reject(self, *, requeue):
        self.actions.append(("reject", requeue))


async def test_retry_cron_rescues_stalled_documents(monkeypatch):
    published = []

    async def fake_claim_failed(pool, max_rounds, backoff):
        return ["f1"]

    async def fake_find_stalled(pool, stale_after_seconds):
        return ["s1"]

    async def fake_find_undelivered(pool):
        return []

    async def fake_set_delivery_status(pool, doc_id, status):
        pass

    class _Publisher:
        async def enqueue(self, name, document_id, **kwargs):
            published.append((name, document_id))

    monkeypatch.setattr(wm.store, "claim_failed_for_retry", fake_claim_failed)
    monkeypatch.setattr(wm.store, "find_stalled_documents", fake_find_stalled)
    monkeypatch.setattr(wm.store, "find_undelivered_documents", fake_find_undelivered)
    monkeypatch.setattr(wm.store, "set_delivery_status", fake_set_delivery_status)

    ctx = {
        "pg": None,
        "task_publisher": _Publisher(),
        "settings": SimpleNamespace(
            MAX_RETRY_ROUNDS=10, RETRY_BACKOFF_BASE=60, STALE_DOC_SECONDS=900
        ),
    }
    await wm.retry_failed_documents(ctx)

    assert set(published) == {("ingest_document", "f1"), ("ingest_document", "s1")}


async def test_retry_cron_relays_undelivered_documents(monkeypatch):
    """delivery_status='pending' 的文档由 relay 补投，confirm 成功置 sent。"""
    published = []
    sent = []

    async def fake_claim_failed(pool, max_rounds, backoff):
        return []

    async def fake_find_stalled(pool, stale_after_seconds):
        return []

    async def fake_find_undelivered(pool):
        return ["u1"]

    async def fake_set_delivery_status(pool, doc_id, status):
        sent.append((doc_id, status))

    class _Publisher:
        async def enqueue(self, name, document_id, **kwargs):
            published.append((name, document_id))

    monkeypatch.setattr(wm.store, "claim_failed_for_retry", fake_claim_failed)
    monkeypatch.setattr(wm.store, "find_stalled_documents", fake_find_stalled)
    monkeypatch.setattr(wm.store, "find_undelivered_documents", fake_find_undelivered)
    monkeypatch.setattr(wm.store, "set_delivery_status", fake_set_delivery_status)

    ctx = {
        "pg": None,
        "task_publisher": _Publisher(),
        "settings": SimpleNamespace(
            MAX_RETRY_ROUNDS=10, RETRY_BACKOFF_BASE=60, STALE_DOC_SECONDS=900
        ),
    }
    await wm.retry_failed_documents(ctx)

    assert published == [("ingest_document", "u1")]
    assert sent == [("u1", "sent")]


def test_worker_task_timeouts_match_previous_arq_limits():
    assert wm._timeout_for("ingest_document") == 300
    assert wm._timeout_for("extract_document_entities") == 900


async def test_handle_message_acknowledges_success(monkeypatch):
    async def fake_execute(ctx, task):
        assert task == DocumentTask("ingest_document", "doc-1")

    monkeypatch.setattr(wm, "_execute_task", fake_execute)
    message = _FakeMessage(b'{"task_name":"ingest_document","document_id":"doc-1"}')

    await wm.handle_message({}, message)

    assert message.actions == [("ack",)]


async def test_handle_message_republishes_before_ack(monkeypatch):
    async def fake_execute(ctx, task):
        raise RuntimeError("boom")

    class _Publisher:
        def __init__(self):
            self.tasks = []

        async def enqueue(self, name, document_id, **kwargs):
            self.tasks.append((name, document_id, kwargs))

    publisher = _Publisher()
    monkeypatch.setattr(wm, "_execute_task", fake_execute)
    message = _FakeMessage(b'{"task_name":"ingest_document","document_id":"doc-1"}')
    ctx = {
        "task_publisher": publisher,
        "settings": SimpleNamespace(RABBITMQ_MAX_ATTEMPTS=3),
    }

    await wm.handle_message(ctx, message)

    assert publisher.tasks == [("ingest_document", "doc-1", {"retry_count": 1})]
    assert message.actions == [("ack",)]


async def test_handle_message_dead_letters_after_final_attempt(monkeypatch):
    async def fake_execute(ctx, task):
        raise RuntimeError("boom")

    monkeypatch.setattr(wm, "_execute_task", fake_execute)
    message = _FakeMessage(
        b'{"task_name":"ingest_document","document_id":"doc-1"}',
        {"x-retry-count": 2},
    )
    ctx = {"settings": SimpleNamespace(RABBITMQ_MAX_ATTEMPTS=3)}

    await wm.handle_message(ctx, message)

    assert message.actions == [("reject", False)]


async def test_handle_message_dead_letters_invalid_payload():
    message = _FakeMessage(b"not-json")

    await wm.handle_message({}, message)

    assert message.actions == [("reject", False)]
