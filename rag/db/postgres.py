from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from rag.config import Settings


async def _configure(conn) -> None:
    """每条连接注册 pgvector，使 Python list 可作为 vector 绑定。"""
    await register_vector_async(conn)


async def create_pg_pool(settings: Settings) -> AsyncConnectionPool:
    """创建并打开异步连接池（不在 import 期调用）。"""
    pool = AsyncConnectionPool(
        conninfo=settings.pg_async_dsn,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
        open=False,
        configure=_configure,
    )
    await pool.open()
    return pool


@asynccontextmanager
async def get_cursor(pool: AsyncConnectionPool):
    """获取 dict-row 异步游标；块正常结束自动 commit，异常 rollback。"""
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            yield cur
