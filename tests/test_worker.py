from rag.document.pipeline import ingest_document
from rag.worker.main import WorkerSettings, retry_failed_documents


def test_worker_registers_ingest_function():
    assert ingest_document in WorkerSettings.functions


def test_worker_registers_dlq_cron_job():
    assert len(WorkerSettings.cron_jobs) == 1
    cron = WorkerSettings.cron_jobs[0]
    assert cron.coroutine is retry_failed_documents
    assert cron.minute == "*/5"


def test_worker_retry_and_timeout_configured():
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.job_timeout == 300


def test_worker_uses_arq_redis_db():
    assert WorkerSettings.redis_settings.database == 1
