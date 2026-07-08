from rag.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.internal")
    monkeypatch.setenv("PG_PORT", "6000")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    s = Settings()
    assert s.PG_HOST == "db.internal"
    assert s.PG_PORT == 6000
    assert s.PG_ASYNC_DSN == "postgresql://rag:secret@db.internal:6000/rag_memory"
    assert s.PG_SYNC_URL == "postgresql+psycopg://rag:secret@db.internal:6000/rag_memory"


def test_settings_defaults():
    s = Settings()
    assert s.PG_HOST == "localhost"
    assert s.REDIS_PORT == 6379
    assert s.EMBEDDING_DIM == 1024


def test_settings_has_log_defaults():
    s = Settings()
    assert s.LOG_LEVEL == "INFO"
    assert s.LOG_FORMAT == "text"
    assert s.LOG_FILE is None
    assert s.LOG_FILE_MAX_BYTES == 10 * 1024 * 1024
    assert s.LOG_FILE_BACKUP_COUNT == 5


def test_settings_has_document_ingestion_defaults():
    s = Settings()
    assert s.MINIO_ENDPOINT == "localhost:9000"
    assert s.MINIO_BUCKET == "rag-documents"
    assert s.MINIO_SECURE is False
    assert s.MINIO_ACCESS_KEY == "minioadmin"
    assert s.MINIO_SECRET_KEY == "minioadmin"
    assert s.ARQ_REDIS_DB == 1
    assert s.CHUNK_SIZE == 800
    assert s.CHUNK_OVERLAP == 100
    assert s.EMBEDDING_BATCH_SIZE == 10
    assert s.MAX_UPLOAD_MB == 20


def test_settings_has_dlq_defaults():
    s = Settings()
    assert s.MAX_RETRY_ROUNDS == 10
    assert s.RETRY_BACKOFF_BASE == 60


def test_settings_has_graph_defaults():
    s = Settings()
    assert s.NEO4J_ENABLED is False
    assert s.ENABLE_ENTITY_EXTRACTION is False
    assert s.NEO4J_URI == "bolt://localhost:7687"
    assert s.NEO4J_USER == "neo4j"
    assert s.NEO4J_DATABASE == "neo4j"


def test_settings_has_retriever_defaults():
    s = Settings()
    assert s.RETRIEVER_CANDIDATE_MULTIPLIER == 2
    assert s.RETRIEVER_VEC_SIMILARITY_THRESHOLD == 0.5


def test_check_required_ignores_neo4j_when_neo4j_disabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("NEO4J_ENABLED", "false")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    Settings().check_required()  # 不抛


def test_check_required_needs_neo4j_when_neo4j_enabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("NEO4J_ENABLED", "true")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    import pytest
    with pytest.raises(ValueError):
        Settings().check_required()
