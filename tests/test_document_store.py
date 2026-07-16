import pytest

from rag.config import get_settings
from rag.db import create_pg_pool
from rag.document import BOOK_KB_ID, store


@pytest.mark.integration
async def test_store_lifecycle_and_idempotent_replace():
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=BOOK_KB_ID,
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

        emb = [0.0] * get_settings().EMBEDDING_DIM
        await store.store_chunks_and_complete(
            pool, doc_id, BOOK_KB_ID, [(0, "c0", emb, {}), (1, "c1", emb, {})]
        )
        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "done"
        assert doc["chunk_count"] == 2

        # 幂等:重放只覆盖,不累加
        await store.store_chunks_and_complete(
            pool, doc_id, BOOK_KB_ID, [(0, "only", emb, {})]
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
            knowledge_base_id=BOOK_KB_ID,
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
        emb = [0.0] * get_settings().EMBEDDING_DIM
        await store.store_chunks_and_complete(pool, doc_id, BOOK_KB_ID, [(0, "x", emb, {})])
        assert await store.claim_for_processing(pool, doc_id) is False
    finally:
        await pool.close()


@pytest.mark.integration
async def test_find_stalled_documents_covers_pending_and_processing():
    """卡死回收:超龄的 pending(enqueue 丢失)与 processing(超时未置 failed)
    都应被找回;done 不应出现。"""
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=BOOK_KB_ID,
            filename="stall.txt",
            content_type="txt",
            size_bytes=1,
            content_hash="hash-stall",
            object_key="ks",
        )

        # 新鲜 pending:未超龄,不算卡死
        assert doc_id not in await store.find_stalled_documents(
            pool, stale_after_seconds=600
        )
        # 阈值 0 强制判旧 → pending 被找回
        assert doc_id in await store.find_stalled_documents(pool, stale_after_seconds=0)

        # processing 超龄同样被找回
        await store.claim_for_processing(pool, doc_id)
        assert doc_id in await store.find_stalled_documents(pool, stale_after_seconds=0)

        # done 之后不再出现
        emb = [0.0] * get_settings().EMBEDDING_DIM
        await store.store_chunks_and_complete(
            pool, doc_id, BOOK_KB_ID, [(0, "x", emb, {})]
        )
        assert doc_id not in await store.find_stalled_documents(
            pool, stale_after_seconds=0
        )
    finally:
        await pool.close()
