import pytest

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.db.redis import create_redis_client


@pytest.mark.integration
async def test_get_cursor_returns_dict_row():
    pool = await create_pg_pool(get_settings())
    try:
        async with get_cursor(pool) as cur:
            await cur.execute("SELECT 1 AS one")
            row = await cur.fetchone()
            assert row["one"] == 1
    finally:
        await pool.close()


@pytest.mark.integration
async def test_redis_ping():
    client = create_redis_client(get_settings())
    try:
        assert await client.ping() is True
    finally:
        await client.aclose()
