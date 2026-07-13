# 默认知识库 ID,须与 alembic 0002 迁移中的 DEFAULT_KB_ID 保持一致
DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001"


async def ensure_kb_partition(pool, kb_id: str) -> None:
    """为新知识库创建 document_chunks 分区（幂等：已存在则跳过）。

    在 knowledge_bases 表新增 KB 后调用，确保 chunk 数据路由到专属分区
    而非落入 DEFAULT 分区。
    """
    from rag.db.postgres import get_cursor

    safe = kb_id.replace("-", "_")
    partition_name = f"dchunks_kb_{safe}"

    async with get_cursor(pool) as cur:
        # 幂等：分区已存在则跳过
        await cur.execute(
            """
            SELECT 1 FROM pg_class
            WHERE relname = %(name)s AND relkind = 'r'
            """,
            {"name": partition_name},
        )
        if await cur.fetchone():
            return

        await cur.execute(
            f"""
            CREATE TABLE {partition_name} PARTITION OF document_chunks
                FOR VALUES IN (%(kb_id)s)
            """,
            {"kb_id": kb_id},
        )
