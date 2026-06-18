import pytest
from contextlib import asynccontextmanager

from rag.config import get_settings
from rag.db.redis import create_redis_client
from rag.db.postgres import get_cursor
from rag.memory.adapters.short_term_redis import RedisShortTermMemory


@asynccontextmanager
async def get_cursor_cleanup(pool, session_id):
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM long_term_memories WHERE metadata->>'session_id' = %(sid)s",
            {"sid": session_id},
        )
    yield


@pytest.mark.integration
async def test_short_term_roundtrip():
    client = create_redis_client(get_settings())
    adapter = RedisShortTermMemory(client)
    sid = "test-session-stm"
    try:
        await adapter.clear(sid)
        await adapter.add(sid, "hello")
        await adapter.add(sid, "world")
        recent = await adapter.get_recent(sid, n=10)
        assert [r["text"] for r in recent] == ["hello", "world"]
    finally:
        await adapter.clear(sid)
        await client.aclose()


from rag.memory.manager import MemoryManager


class _FakeLong:
    def __init__(self):
        self.calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.calls.append((session_id, query, top_k, filters))
        return [{"text": "L"}]


class _FakeShort:
    def __init__(self):
        self.added = []

    async def add(self, session_id, text, metadata=None):
        self.added.append((session_id, text))

    async def get_recent(self, session_id, n=10):
        return [{"text": "S"}]


async def test_manager_search_passes_args_in_order():
    long = _FakeLong()
    mgr = MemoryManager(long_term=long, short_term=_FakeShort())
    out = await mgr.search("sid1", "q1")
    assert out == [{"text": "L"}]
    assert long.calls == [("sid1", "q1", 5, None)]


async def test_manager_add_message_writes_short_term():
    short = _FakeShort()
    mgr = MemoryManager(long_term=_FakeLong(), short_term=short)
    await mgr.add_message("sid1", "hi")
    assert short.added == [("sid1", "hi")]


import numpy as np

from rag.db.postgres import create_pg_pool
from rag.models.embedding import EmbeddingModel
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory


class _StubEmbedding(EmbeddingModel):
    def __init__(self):
        pass  # 跳过真实 client

    async def embed(self, texts):
        return [np.array([0.01] * 1024, dtype=np.float32) for _ in texts]


@pytest.mark.integration
async def test_long_term_add_then_search():
    pool = await create_pg_pool(get_settings())
    adapter = PgVectorLongTermMemory(pool, _StubEmbedding())
    sid = "test-session-ltm"
    try:
        mid = await adapter.add(sid, "关税申报流程")
        assert mid
        results = await adapter.search(sid, "关税")
        assert any(r["text"] == "关税申报流程" for r in results)
    finally:
        async with get_cursor_cleanup(pool, sid):
            pass
        await pool.close()
