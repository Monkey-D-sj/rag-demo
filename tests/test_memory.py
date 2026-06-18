import pytest

from rag.config import get_settings
from rag.db.redis import create_redis_client
from rag.memory.adapters.short_term_redis import RedisShortTermMemory


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
