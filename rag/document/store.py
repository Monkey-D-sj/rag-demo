import datetime
import uuid

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from rag.db.postgres import get_cursor


async def create_document(
    pool: AsyncConnectionPool,
    *,
    knowledge_base_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    content_hash: str,
    object_key: str,
) -> str:
    document_id = str(uuid.uuid4())
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            INSERT INTO documents
                (id, knowledge_base_id, filename, content_type,
                 size_bytes, content_hash, object_key, status)
            VALUES
                (%(id)s, %(kb)s, %(fn)s, %(ct)s,
                 %(sz)s, %(hash)s, %(key)s, 'pending')
            """,
            {
                "id": document_id,
                "kb": knowledge_base_id,
                "fn": filename,
                "ct": content_type,
                "sz": size_bytes,
                "hash": content_hash,
                "key": object_key,
            },
        )
    return document_id


async def search_chunks(
    pool: AsyncConnectionPool,
    embedding: list[float],
    knowledge_base_id: str,
    top_k: int = 5,
) -> list[dict]:
    """按余弦相似度从知识库召回最相关的 chunk。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT id, document_id, chunk_index, text,
                   1 - (embedding <=> %(emb)s) AS similarity
            FROM document_chunks
            WHERE knowledge_base_id = %(kb)s
            ORDER BY embedding <=> %(emb)s
            LIMIT %(k)s
            """,
            {"emb": embedding, "kb": knowledge_base_id, "k": top_k},
        )
        return await cur.fetchall()


async def get_document(pool: AsyncConnectionPool, document_id: str) -> dict | None:
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT * FROM documents WHERE id = %(id)s", {"id": document_id}
        )
        return await cur.fetchone()


async def list_documents(
    pool: AsyncConnectionPool,
    *,
    knowledge_base_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """分页列出文档(按创建时间倒序),返回 (当前页行, 满足过滤条件的总数)。

    用 COUNT(*) OVER() 窗口函数在同一次查询里带出总数,省去额外的 count 查询。
    """
    conditions = []
    params: dict = {"limit": limit, "offset": offset}
    if knowledge_base_id is not None:
        conditions.append("knowledge_base_id = %(kb)s")
        params["kb"] = knowledge_base_id
    if status is not None:
        conditions.append("status = %(status)s")
        params["status"] = status
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with get_cursor(pool) as cur:
        await cur.execute(
            f"""
            SELECT id, knowledge_base_id, filename, content_type,
                   size_bytes, status, chunk_count, error,
                   graph_status, graph_error,
                   created_at, updated_at,
                   COUNT(*) OVER() AS total
            FROM documents
            {where}
            ORDER BY created_at DESC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            params,
        )
        rows = await cur.fetchall()

    total = rows[0]["total"] if rows else 0
    return rows, total


async def claim_for_processing(
    pool: AsyncConnectionPool, document_id: str, *, stale_after_seconds: int = 600
) -> bool:
    """原子领取:置为 processing 并返回是否领取成功。

    可领取条件:
    - 文档处于 pending/failed;或
    - 处于 processing 但 updated_at 已超过 stale_after_seconds(worker 中途硬崩、
      没走 failed 分支导致状态卡死),视为陈旧任务可被回收重跑。

    并发或重复投递(重试、双提交、arq 重跑)时,行级锁保证只有一个任务领取成功,
    其余拿到 False 直接跳过,避免多个任务同时对同一文档重复入库(删/插 chunk 打架)。

    注意:stale_after_seconds 必须大于 worker 的 job_timeout(默认 300),
    否则可能误回收仍在运行的任务。
    """
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET status = 'processing', updated_at = now()
            WHERE id = %(id)s
              AND (
                status IN ('pending', 'failed')
                OR (
                    status = 'processing'
                    AND updated_at < now() - make_interval(secs => %(stale)s)
                )
              )
            RETURNING id
            """,
            {"id": document_id, "stale": stale_after_seconds},
        )
        return await cur.fetchone() is not None


async def claim_failed_for_retry(
    pool: AsyncConnectionPool, max_retry_rounds: int, backoff_base: int
) -> list[str]:
    """扫描并原子领取应重试的失败文档,返回文档 ID 列表。

    单事务内完成:
    1. FOR UPDATE SKIP LOCKED 锁候选行
    2. 按指数退避过滤(backoff_base * 2^retry_count 秒)
    3. retry_count += 1, status 重置 pending

    多个 cron 并发安全;只返回实际更新成功的 id。
    """
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # 1. 锁所有 candidate（避免并发 cron 抢同一行）
            await cur.execute(
                """
                SELECT id, retry_count, updated_at
                FROM documents
                WHERE status = 'failed' AND retry_count < %(max_rounds)s
                FOR UPDATE SKIP LOCKED
                """,
                {"max_rounds": max_retry_rounds},
            )
            candidates = await cur.fetchall()

        # 2. Python 侧过滤退避（带时区的指数退避）
        now = datetime.datetime.now(datetime.timezone.utc)
        eligible: list[str] = []
        for row in candidates:
            backoff_s = backoff_base * (2 ** row["retry_count"])
            # updated_at 来自 DB 是 aware datetime;确保比较双方都有 tz
            ts = row["updated_at"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            if ts + datetime.timedelta(seconds=backoff_s) < now:
                eligible.append(row["id"])

        # 3. 原子更新
        if eligible:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE documents
                    SET retry_count = retry_count + 1,
                        status = 'pending',
                        error = NULL,
                        updated_at = now()
                    WHERE id = ANY(%(ids)s)
                    """,
                    {"ids": eligible},
                )

    return eligible


async def force_retry(pool: AsyncConnectionPool, document_id: str) -> None:
    """强制重试:清零 retry_count、清除错误、重置为 pending。

    与 claim_failed_for_retry 不同,本函数不检查退避时间也不递增计数,
    是运维级别的"从头再来",适合根因修复后批量恢复。
    """
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET retry_count = 0, status = 'pending', error = NULL, updated_at = now()
            WHERE id = %(id)s
            """,
            {"id": document_id},
        )


async def set_status(
    pool: AsyncConnectionPool, document_id: str, status: str, *, error: str | None = None
) -> None:
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET status = %(s)s, error = %(e)s, updated_at = now()
            WHERE id = %(id)s
            """,
            {"s": status, "e": error, "id": document_id},
        )


async def store_chunks_and_complete(
    pool: AsyncConnectionPool,
    document_id: str,
    knowledge_base_id: str,
    embedded: list[tuple[int, str, list[float], dict[str, str]]],
) -> None:
    """单事务:删旧 chunk → 插新 chunk → 置 done + chunk_count。可安全重放。

    embedded 每项为 (chunk_index, text, embedding, metadata);metadata 存入 JSONB 列
    (如 paragraph_semantic 策略的章节标题 {"title": ...})。
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %(id)s",
                {"id": document_id},
            )
            if embedded:
                await cur.executemany(
                    """
                    INSERT INTO document_chunks
                        (document_id, knowledge_base_id, chunk_index, text, embedding, metadata)
                    VALUES
                        (%(doc)s, %(kb)s, %(idx)s, %(text)s, %(emb)s, %(meta)s)
                    """,
                    [
                        {
                            "doc": document_id,
                            "kb": knowledge_base_id,
                            "idx": chunk_index,
                            "text": text,
                            "emb": embedding,
                            "meta": Jsonb(metadata),
                        }
                        for chunk_index, text, embedding, metadata in embedded
                    ],
                )
            await cur.execute(
                """
                UPDATE documents
                SET status = 'done', chunk_count = %(n)s, updated_at = now()
                WHERE id = %(id)s
                """,
                {"n": len(embedded), "id": document_id},
            )


async def get_chunks_for_graph(
    pool: AsyncConnectionPool, document_id: str
) -> list[dict]:
    """读取文档所有 chunk 供实体抽取:chunk_index、text、章节标题(metadata.title)。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT chunk_index,
                   text,
                   metadata ->> 'title' AS title
            FROM document_chunks
            WHERE document_id = %(id)s
            ORDER BY chunk_index
            """,
            {"id": document_id},
        )
        return await cur.fetchall()


async def claim_graph_processing(
    pool: AsyncConnectionPool, document_id: str
) -> bool:
    """原子领取图抽取:graph_status pending/failed → processing,返回是否成功。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET graph_status = 'processing', updated_at = now()
            WHERE id = %(id)s AND graph_status IN ('pending', 'failed')
            RETURNING id
            """,
            {"id": document_id},
        )
        return await cur.fetchone() is not None


async def set_graph_status(
    pool: AsyncConnectionPool,
    document_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """设置 graph_status(与向量入库 status 独立)。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET graph_status = %(s)s, graph_error = %(e)s, updated_at = now()
            WHERE id = %(id)s
            """,
            {"s": status, "e": error, "id": document_id},
        )
