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
    assert s.minio_access_key == "minioadmin"
    assert s.minio_secret_key == "minioadmin"
    assert s.arq_redis_db == 1
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.embedding_batch_size == 16
    assert s.max_upload_mb == 20


def test_settings_has_dlq_defaults():
    s = Settings()
    assert s.max_retry_rounds == 10
    assert s.retry_backoff_base == 60


def test_settings_has_graph_defaults():
    s = Settings()
    assert s.ENABLE_ENTITY_EXTRACTION is False
    assert s.NEO4J_URI == "bolt://localhost:7687"
    assert s.NEO4J_USER == "neo4j"
    assert s.NEO4J_DATABASE == "neo4j"


def test_check_required_ignores_neo4j_when_extraction_disabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("ENABLE_ENTITY_EXTRACTION", "false")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    Settings().check_required()  # 不抛


def test_check_required_needs_neo4j_when_extraction_enabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("ENABLE_ENTITY_EXTRACTION", "true")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    import pytest
    with pytest.raises(ValueError):
        Settings().check_required()
