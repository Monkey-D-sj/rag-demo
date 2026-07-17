from types import SimpleNamespace

import rag.worker.main as wm
from rag.worker.main import WorkerSettings, retry_failed_documents


async def test_retry_cron_rescues_stalled_documents(monkeypatch):
    """自愈 cron 除 failed 外,还必须回收卡死的 pending/processing 文档
    (enqueue 丢失、worker 超时被取消未置 failed 等状态机盲区)。"""
    ingested = []

    async def fake_claim_failed(pool, max_rounds, backoff):
        return ["f1"]

    async def fake_find_stalled(pool, stale_after_seconds):
        return ["s1"]

    async def fake_ingest(ctx, doc_id):
        ingested.append(doc_id)

    monkeypatch.setattr(wm.store, "claim_failed_for_retry", fake_claim_failed)
    monkeypatch.setattr(wm.store, "find_stalled_documents", fake_find_stalled)
    monkeypatch.setattr(wm, "ingest_document", fake_ingest)

    ctx = {
        "pg": None,
        "settings": SimpleNamespace(
            MAX_RETRY_ROUNDS=10, RETRY_BACKOFF_BASE=60, STALE_DOC_SECONDS=900
        ),
    }
    await retry_failed_documents(ctx)

    assert set(ingested) == {"f1", "s1"}


def test_worker_registers_ingest_function():
    fn_names = [f.name for f in WorkerSettings.functions]
    assert "ingest_document" in fn_names


def test_worker_registers_dlq_cron_job():
    assert len(WorkerSettings.cron_jobs) == 1
    cron = WorkerSettings.cron_jobs[0]
    assert cron.coroutine is retry_failed_documents
    # cron.minute is a set, e.g. {0, 5, 10, ..., 55}
    assert 0 in cron.minute
    assert 55 in cron.minute


def test_worker_retry_and_timeout_configured():
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.job_timeout == 300


def test_worker_uses_arq_redis_db():
    assert WorkerSettings.redis_settings.database == 1


def test_worker_registers_extract_function():
    fn_names = [f.name for f in WorkerSettings.functions]
    assert "extract_document_entities" in fn_names


def test_extract_document_entities_timeout():
    fx = next(f for f in WorkerSettings.functions if f.name == "extract_document_entities")
    assert fx.timeout_s == 900
