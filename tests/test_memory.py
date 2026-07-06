import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from contextlib import asynccontextmanager

from rag.config import get_settings
from rag.db.redis import create_redis_client
from rag.db.postgres import create_pg_pool, get_cursor
from rag.agent.memory import MemoryManager
from rag.models.embedding import EmbeddingModel


@asynccontextmanager
async def get_cursor_cleanup(pool, session_id):
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM long_term_memories WHERE metadata->>'session_id' = %(sid)s",
            {"sid": session_id},
        )
    yield


# ── 短期记忆单元测试 ──

async def test_add_message_pushes_to_redis():
    redis = MagicMock()
    pipe = MagicMock()
    pipe.execute = AsyncMock()
    redis.pipeline.return_value = pipe
    mgr = MemoryManager(pool=None, embedding=None, redis=redis)

    await mgr.add_message("sid1", "hello", {"role": "user"})

    # 验证 redis pipeline 调用
    key = "session:sid1:messages"
    pipe.rpush.assert_called_once()
    assert pipe.rpush.call_args[0][0] == key
    payload = json.loads(pipe.rpush.call_args[0][1])
    assert payload["text"] == "hello"
    assert payload["metadata"] == {"role": "user"}
    pipe.execute.assert_awaited_once()


async def test_get_recent_messages_reads_redis():
    redis = MagicMock()
    redis.lrange = AsyncMock(return_value=[
        json.dumps({"text": "a"}),
        json.dumps({"text": "b"}),
    ])
    mgr = MemoryManager(pool=None, embedding=None, redis=redis)

    result = await mgr.get_recent_messages("sid1", n=2)

    redis.lrange.assert_awaited_once_with("session:sid1:messages", -2, -1)
    assert [r["text"] for r in result] == ["a", "b"]


async def test_get_recent_messages_returns_empty_when_n_zero():
    redis = MagicMock()
    mgr = MemoryManager(pool=None, embedding=None, redis=redis)
    assert await mgr.get_recent_messages("sid1", n=0) == []


async def test_short_term_noop_when_redis_is_none():
    mgr = MemoryManager(pool=None, embedding=None, redis=None)
    # 不应抛异常
    await mgr.add_message("sid1", "hi")
    assert await mgr.get_recent_messages("sid1") == []
    await mgr.clear_session("sid1")


# ── 长期记忆单元测试 ──


class _FakeEmbedding(EmbeddingModel):
    def __init__(self):
        pass

    async def embed(self, texts):
        return [np.array([0.01] * 1024, dtype=np.float32) for _ in texts]


class _FakeCursor:
    """模拟 psycopg cursor，记录最后一次 execute 调用。"""
    def __init__(self):
        self.last_sql = ""
        self.last_params = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, sql, params=None):
        self.last_sql = sql
        self.last_params = params or {}

    async def fetchall(self):
        return [{"text": "mock_result", "similarity": 0.95}]


async def test_search_builds_correct_query(monkeypatch):
    pool = MagicMock()
    mgr = MemoryManager(pool=pool, embedding=_FakeEmbedding())

    fake_cur = _FakeCursor()
    monkeypatch.setattr(
        "rag.agent.memory.manager.get_cursor",
        lambda p: fake_cur,
    )

    await mgr.search("sid1", "关税", top_k=3, filters={"type": "doc"})

    assert "top_k" in fake_cur.last_params
    assert fake_cur.last_params["top_k"] == 3
    assert "filter_0" in fake_cur.last_params
    assert fake_cur.last_params["filter_0"] == "doc"


async def test_add_returns_memory_id(monkeypatch):
    pool = MagicMock()
    mgr = MemoryManager(pool=pool, embedding=_FakeEmbedding())

    fake_cur = _FakeCursor()
    monkeypatch.setattr(
        "rag.agent.memory.manager.get_cursor",
        lambda p: fake_cur,
    )

    mid = await mgr.add("sid1", "测试文本")
    assert mid  # UUID 字符串
    assert len(mid) == 36


# ── 集成测试 ──

@pytest.mark.integration
async def test_short_term_roundtrip():
    client = create_redis_client(get_settings())
    mgr = MemoryManager(pool=None, embedding=None, redis=client)
    sid = "test-session-stm"
    try:
        await mgr.clear_session(sid)
        await mgr.add_message(sid, "hello")
        await mgr.add_message(sid, "world")
        recent = await mgr.get_recent_messages(sid, n=10)
        assert [r["text"] for r in recent] == ["hello", "world"]
    finally:
        await mgr.clear_session(sid)
        await client.aclose()


@pytest.mark.integration
async def test_long_term_add_then_search():
    pool = await create_pg_pool(get_settings())
    mgr = MemoryManager(pool=pool, embedding=_FakeEmbedding())
    sid = "test-session-ltm"
    try:
        mid = await mgr.add(sid, "关税申报流程")
        assert mid
        results = await mgr.search(sid, "关税")
        assert any(r["text"] == "关税申报流程" for r in results)
    finally:
        async with get_cursor_cleanup(pool, sid):
            pass
        await pool.close()
