from rag.document.pipeline import ingest_document
from rag.worker.main import WorkerSettings


def test_worker_registers_ingest_function():
    assert ingest_document in WorkerSettings.functions


def test_worker_retry_and_timeout_configured():
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.job_timeout == 300


def test_worker_uses_arq_redis_db():
    assert WorkerSettings.redis_settings.database == 1
