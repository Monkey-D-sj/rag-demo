from rag.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.internal")
    monkeypatch.setenv("PG_PORT", "6000")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    s = Settings()
    assert s.pg_host == "db.internal"
    assert s.pg_port == 6000
    assert s.pg_async_dsn == "postgresql://rag:secret@db.internal:6000/rag_memory"
    assert s.pg_sync_url == "postgresql+psycopg://rag:secret@db.internal:6000/rag_memory"


def test_settings_defaults(monkeypatch):
    for k in ("PG_HOST", "PG_PORT", "REDIS_HOST"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.pg_host == "localhost"
    assert s.redis_port == 6379
    assert s.embedding_dim == 1024


def test_settings_has_log_defaults():
    from rag.config import Settings

    s = Settings()
    assert s.log_level == "INFO"
    assert s.log_format == "text"
    assert s.log_file is None
    assert s.log_file_max_bytes == 10 * 1024 * 1024
    assert s.log_file_backup_count == 5


def test_settings_has_document_ingestion_defaults():
    s = Settings()
    assert s.minio_endpoint == "localhost:9000"
    assert s.minio_bucket == "rag-documents"
    assert s.minio_secure is False
    assert s.arq_redis_db == 1
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.embedding_batch_size == 16
    assert s.max_upload_mb == 20
