"""会话元数据 CRUD，沿用 rag/document/store.py 的 get_cursor 模式。"""

from psycopg_pool import AsyncConnectionPool

from rag.db.postgres import get_cursor


async def create_session(pool: AsyncConnectionPool) -> dict:
    """创建新会话，返回 {session_id, title, created_at, updated_at}。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            INSERT INTO sessions (title)
            VALUES ('新会话')
            RETURNING id AS session_id, title, created_at, updated_at
            """
        )
        return await cur.fetchone()


async def list_sessions(pool: AsyncConnectionPool) -> list[dict]:
    """列出所有会话，按最近活跃排序。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT id AS session_id, title, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            """
        )
        return await cur.fetchall()


async def get_session(pool: AsyncConnectionPool, session_id: str) -> dict | None:
    """获取单个会话。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT id AS session_id, title, created_at, updated_at FROM sessions WHERE id = %(id)s",
            {"id": session_id},
        )
        return await cur.fetchone()


async def ensure_session(pool: AsyncConnectionPool, session_id: str) -> dict:
    """幂等获取/创建会话：存在则返回，不存在则 INSERT。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT id AS session_id, title, created_at, updated_at FROM sessions WHERE id = %(id)s",
            {"id": session_id},
        )
        row = await cur.fetchone()
        if row:
            return row
        await cur.execute(
            """
            INSERT INTO sessions (id, title)
            VALUES (%(id)s, '新会话')
            RETURNING id AS session_id, title, created_at, updated_at
            """,
            {"id": session_id},
        )
        return await cur.fetchone()


async def set_title(pool: AsyncConnectionPool, session_id: str, title: str) -> None:
    """更新会话标题。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE sessions SET title = %(title)s, updated_at = now()
            WHERE id = %(id)s
            """,
            {"id": session_id, "title": title},
        )


async def touch_session(pool: AsyncConnectionPool, session_id: str) -> None:
    """更新会话的 updated_at 时间戳。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "UPDATE sessions SET updated_at = now() WHERE id = %(id)s",
            {"id": session_id},
        )


async def delete_session(pool: AsyncConnectionPool, session_id: str) -> bool:
    """删除会话，返回是否成功。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM sessions WHERE id = %(id)s RETURNING id",
            {"id": session_id},
        )
        return await cur.fetchone() is not None
