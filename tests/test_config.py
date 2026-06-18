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
