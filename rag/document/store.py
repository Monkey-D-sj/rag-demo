import uuid

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


async def get_document(pool: AsyncConnectionPool, document_id: str) -> dict | None:
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT * FROM documents WHERE id = %(id)s", {"id": document_id}
        )
        return await cur.fetchone()


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
    embedded: list[tuple[int, str, list[float]]],
) -> None:
    """单事务:删旧 chunk → 插新 chunk → 置 done + chunk_count。可安全重放。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %(id)s",
                {"id": document_id},
            )
            for chunk_index, text, embedding in embedded:
                await cur.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, knowledge_base_id, chunk_index, text, embedding)
                    VALUES
                        (%(doc)s, %(kb)s, %(idx)s, %(text)s, %(emb)s)
                    """,
                    {
                        "doc": document_id,
                        "kb": knowledge_base_id,
                        "idx": chunk_index,
                        "text": text,
                        "emb": embedding,
                    },
                )
            await cur.execute(
                """
                UPDATE documents
                SET status = 'done', chunk_count = %(n)s, updated_at = now()
                WHERE id = %(id)s
                """,
                {"n": len(embedded), "id": document_id},
            )
