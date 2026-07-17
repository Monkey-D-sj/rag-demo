import time

from pgvector import Vector
from psycopg.types.json import Json

from rag.common.logging import get_logger
from rag.db.postgres import get_cursor
from rag.governance.usage import CallRecord

logger = get_logger()

_LOOKUP_SQL = """
SELECT id, answer, citations,
       1 - (embedding <=> %(vec)s) AS similarity
FROM semantic_cache
WHERE created_at > now() - make_interval(hours => %(ttl)s)
ORDER BY embedding <=> %(vec)s
LIMIT 1
"""

_INSERT_SQL = """
INSERT INTO semantic_cache (question, answer, citations, embedding)
VALUES (%(question)s, %(answer)s, %(citations)s, %(vec)s)
"""


async def clear_semantic_cache(pool) -> None:
    """整表清空。文档入库成功后调用:答案可能引用任意 KB,局部失效不正确。"""
    async with get_cursor(pool) as cur:
        await cur.execute("DELETE FROM semantic_cache")


async def purge_expired(pool, ttl_hours: int) -> None:
    """物理删除过期行,worker cron 定期调用。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM semantic_cache"
            " WHERE created_at <= now() - make_interval(hours => %(ttl)s)",
            {"ttl": ttl_hours},
        )


class SemanticCache:
    """答案级语义缓存:以 rewrite_query 的 embedding 为 key,pgvector 相似度命中。

    lookup/store 内部吞掉一切异常(缓存故障绝不阻断主链路):
    lookup 失败返回 None(降级为未命中),store 失败仅记 warning。
    """

    def __init__(
        self,
        pool,
        embedding,
        *,
        threshold: float = 0.95,
        ttl_hours: int = 168,
        recorder=None,
    ) -> None:
        self._pool = pool
        self._embedding = embedding
        self._threshold = threshold
        self._ttl_hours = ttl_hours
        self._recorder = recorder

    async def _embed_query(self, query: str) -> Vector:
        return Vector((await self._embedding.embed([query]))[0])

    async def _nearest_by_vec(self, cur, vec: Vector) -> dict | None:
        await cur.execute(_LOOKUP_SQL, {"vec": vec, "ttl": self._ttl_hours})
        return await cur.fetchone()

    async def lookup(self, query: str, session_id: str | None = None) -> dict | None:
        start = time.monotonic()
        try:
            async with get_cursor(self._pool) as cur:
                vec = await self._embed_query(query)
                row = await self._nearest_by_vec(cur, vec)
                if row is None or row["similarity"] < self._threshold:
                    return None
                await cur.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1"
                    " WHERE id = %(id)s",
                    {"id": row["id"]},
                )
        except Exception:  # noqa: BLE001 - 缓存故障降级为未命中
            logger.warning("语义缓存查询失败,降级为未命中", exc_info=True)
            return None

        if self._recorder is not None:
            self._recorder.record(CallRecord(
                call_type="semantic_cache",
                model="semantic-cache",
                status="cache_hit",
                attempts=1,
                latency_ms=int((time.monotonic() - start) * 1000),
                session_id=session_id,
            ))
        logger.info("语义缓存命中: similarity=%.4f", row["similarity"])
        return {"answer": row["answer"], "citations": row["citations"]}

    async def store(self, query: str, answer: str, citations: list) -> None:
        try:
            async with get_cursor(self._pool) as cur:
                vec = await self._embed_query(query)
                row = await self._nearest_by_vec(cur, vec)
                if row is not None and row["similarity"] >= self._threshold:
                    return  # 已有近重复条目,跳过插入防膨胀
                # 并发未命中可能同时通过查重并各插一行,属已知可容忍竞态:lookup 取最近邻,TTL/清空兜底增长
                await cur.execute(_INSERT_SQL, {
                    "question": query,
                    "answer": answer,
                    "citations": Json(citations),
                    "vec": vec,
                })
        except Exception:  # noqa: BLE001 - 回写失败不拖垮回答
            logger.warning("语义缓存回写失败", exc_info=True)
