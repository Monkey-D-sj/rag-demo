from rag.worker.main import WorkerSettings, retry_failed_documents


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
