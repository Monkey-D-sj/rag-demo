from unittest.mock import AsyncMock, MagicMock

from rag.agent.cache.manager import (
    SemanticCache,
    clear_semantic_cache,
    purge_expired,
)


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeCursor:
    """模拟 psycopg cursor，记录全部 execute 调用，fetchone 按序出队。"""

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
    """注入 _FakeCursor 并返回 (SemanticCache, cursor) 以便检查 SQL 调用。"""
    monkeypatch.setattr("rag.agent.cache.manager.get_cursor", lambda p: cur)
    cache = SemanticCache(
        MagicMock(), _FakeEmbedding(),
        threshold=0.95, ttl_hours=168, recorder=recorder,
    )
    return cache, cur


# ── lookup ──

async def test_lookup_hit(monkeypatch):
    """相似度 >= threshold 返回缓存结果，增 hit_count 并记录用量。"""
    row = {"id": 1, "answer": "答案A", "citations": [{"index": 1}], "similarity": 0.97}
    recorder = MagicMock()
    cache, cur = _make_cache(_FakeCursor(rows=[row]), monkeypatch, recorder=recorder)

    hit = await cache.lookup("孙悟空是谁", session_id="s1")

    assert hit == {"answer": "答案A", "citations": [{"index": 1}]}
    assert any("hit_count" in sql for sql, _ in cur.calls)

    rec = recorder.record.call_args[0][0]
    assert rec.call_type == "semantic_cache"
    assert rec.status == "cache_hit"


async def test_lookup_miss_below_threshold(monkeypatch):
    """相似度低于阈值返回 None，不增 hit_count。"""
    row = {"id": 1, "answer": "答案A", "citations": [], "similarity": 0.90}
    cache, cur = _make_cache(_FakeCursor(rows=[row]), monkeypatch)

    assert await cache.lookup("孙悟空是谁") is None
    assert not any("hit_count" in sql for sql, _ in cur.calls)


async def test_lookup_db_error_degrades_to_none(monkeypatch):
    """DB 异常降级为 None，不抛给调用方。"""
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache, _ = _make_cache(_BoomCursor(), monkeypatch)
    assert await cache.lookup("孙悟空是谁") is None


# ── store ──

async def test_store_inserts_when_no_near_duplicate(monkeypatch):
    cache, cur = _make_cache(_FakeCursor(rows=[None]), monkeypatch)
    await cache.store("q", "answer", [{"index": 1}])
    assert any("INSERT INTO semantic_cache" in sql for sql, _ in cur.calls)


async def test_store_skips_near_duplicate(monkeypatch):
    cache, cur = _make_cache(
        _FakeCursor(rows=[{"id": 1, "answer": "old", "citations": [], "similarity": 0.98}]),
        monkeypatch,
    )
    await cache.store("q", "answer", [])
    assert not any("INSERT INTO" in sql for sql, _ in cur.calls)


async def test_store_db_error_swallowed(monkeypatch):
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache, _ = _make_cache(_BoomCursor(), monkeypatch)
    await cache.store("q", "a", [])  # 不抛异常即通过


# ── 管理操作 ──

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
