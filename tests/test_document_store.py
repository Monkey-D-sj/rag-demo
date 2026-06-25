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


@pytest.mark.integration
async def test_claim_for_processing_dedup_and_stale_reclaim():
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=DEFAULT_KB_ID,
            filename="c.txt",
            content_type="txt",
            size_bytes=1,
            content_hash="hash-claim",
            object_key="kc",
        )

        # pending → 领取成功,置 processing
        assert await store.claim_for_processing(pool, doc_id) is True
        assert (await store.get_document(pool, doc_id))["status"] == "processing"

        # 新鲜 processing → 重复领取被拒
        assert await store.claim_for_processing(pool, doc_id) is False

        # 陈旧 processing(stale=0 强制判旧)→ 可回收重跑
        assert (
            await store.claim_for_processing(pool, doc_id, stale_after_seconds=0)
            is True
        )

        # done → 不可领取
        emb = [0.0] * get_settings().embedding_dim
        await store.store_chunks_and_complete(pool, doc_id, DEFAULT_KB_ID, [(0, "x", emb)])
        assert await store.claim_for_processing(pool, doc_id) is False
    finally:
        await pool.close()
