from unittest.mock import MagicMock

from rag.agent.cache.manager import (
    SemanticCache,
    clear_semantic_cache,
    purge_expired,
)


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeCursor:
    """模拟 psycopg cursor,记录全部 execute 调用,fetchone 按序出队。"""

    def __init__(self, rows=None):
        self.calls = []          # [(sql, params), ...]
        self._rows = list(rows or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))

    async def fetchone(self):
        return self._rows.pop(0) if self._rows else None


def _make_cache(cur, monkeypatch, recorder=None):
    monkeypatch.setattr(
        "rag.agent.cache.manager.get_cursor", lambda p: cur
    )
    return SemanticCache(
        MagicMock(), _FakeEmbedding(),
        threshold=0.95, ttl_hours=168, recorder=recorder,
    )


async def test_lookup_hit_above_threshold(monkeypatch):
    row = {"id": 1, "answer": "答案A", "citations": [{"index": 1}], "similarity": 0.97}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    hit = await cache.lookup("孙悟空是谁")

    assert hit == {"answer": "答案A", "citations": [{"index": 1}]}
    # 第二条 SQL 是 hit_count 自增
    assert any("hit_count" in sql for sql, _ in cur.calls)


async def test_lookup_miss_below_threshold(monkeypatch):
    row = {"id": 1, "answer": "答案A", "citations": [], "similarity": 0.90}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    assert await cache.lookup("孙悟空是谁") is None
    # 未命中不得自增 hit_count
    assert not any("hit_count" in sql for sql, _ in cur.calls)


async def test_lookup_empty_table_returns_none(monkeypatch):
    cur = _FakeCursor(rows=[])
    cache = _make_cache(cur, monkeypatch)
    assert await cache.lookup("孙悟空是谁") is None


async def test_lookup_db_error_degrades_to_none(monkeypatch):
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache = _make_cache(_BoomCursor(), monkeypatch)
    assert await cache.lookup("孙悟空是谁") is None  # 不抛异常


async def test_lookup_hit_records_usage(monkeypatch):
    row = {"id": 1, "answer": "A", "citations": [], "similarity": 0.99}
    recorder = MagicMock()
    cache = _make_cache(_FakeCursor(rows=[row]), monkeypatch, recorder=recorder)

    await cache.lookup("q", session_id="s1")

    assert recorder.record.call_count == 1
    rec = recorder.record.call_args[0][0]
    assert rec.call_type == "semantic_cache"
    assert rec.status == "cache_hit"
    assert rec.session_id == "s1"


async def test_store_inserts_when_no_near_duplicate(monkeypatch):
    cur = _FakeCursor(rows=[None])  # 查重无结果
    cache = _make_cache(cur, monkeypatch)

    await cache.store("q", "answer", [{"index": 1}])

    assert any("INSERT INTO semantic_cache" in sql for sql, _ in cur.calls)


async def test_store_skips_near_duplicate(monkeypatch):
    row = {"id": 1, "answer": "old", "citations": [], "similarity": 0.98}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    await cache.store("q", "answer", [])

    assert not any("INSERT INTO" in sql for sql, _ in cur.calls)


async def test_store_db_error_swallowed(monkeypatch):
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache = _make_cache(_BoomCursor(), monkeypatch)
    await cache.store("q", "a", [])  # 不抛异常即通过


async def test_clear_and_purge_sql(monkeypatch):
    cur = _FakeCursor()
    monkeypatch.setattr("rag.agent.cache.manager.get_cursor", lambda p: cur)

    await clear_semantic_cache(MagicMock())
    await purge_expired(MagicMock(), ttl_hours=168)

    assert any("DELETE FROM semantic_cache" in sql and "created_at" not in sql
               for sql, _ in cur.calls)
    assert any("DELETE FROM semantic_cache" in sql and "created_at" in sql
               for sql, _ in cur.calls)


def test_protocol_conformance():
    from rag.agent.type import SemanticCacheProtocol

    cache = SemanticCache(MagicMock(), _FakeEmbedding(), threshold=0.95, ttl_hours=1)
    assert isinstance(cache, SemanticCacheProtocol)
