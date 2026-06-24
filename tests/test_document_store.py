import pytest

from rag.config import get_settings
from rag.db import create_pg_pool
from rag.document import DEFAULT_KB_ID, store


@pytest.mark.integration
async def test_store_lifecycle_and_idempotent_replace():
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=DEFAULT_KB_ID,
            filename="a.txt",
            content_type="txt",
            size_bytes=5,
            content_hash="hash1",
            object_key="k1",
        )

        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "pending"
        assert doc["filename"] == "a.txt"

        await store.set_status(pool, doc_id, "processing")
        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "processing"

        emb = [0.0] * get_settings().embedding_dim
        await store.store_chunks_and_complete(
            pool, doc_id, DEFAULT_KB_ID, [(0, "c0", emb), (1, "c1", emb)]
        )
        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "done"
        assert doc["chunk_count"] == 2

        # 幂等:重放只覆盖,不累加
        await store.store_chunks_and_complete(
            pool, doc_id, DEFAULT_KB_ID, [(0, "only", emb)]
        )
        doc = await store.get_document(pool, doc_id)
        assert doc["chunk_count"] == 1

        assert await store.get_document(pool, "00000000-0000-0000-0000-0000000000ff") is None
    finally:
        await pool.close()
