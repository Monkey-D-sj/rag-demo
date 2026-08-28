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
        assert doc["delivery_status"] == "pending"
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


@pytest.mark.integration
async def test_get_documents_content_batch_and_slim_search_rows():
    """get_documents_content 按 doc_id 批量取全文；检索行不再携带 document_content。"""
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=BOOK_KB_ID,
            filename="content.txt",
            content_type="txt",
            size_bytes=1,
            content_hash="hash-content",
            object_key="kco",
        )
        await store.set_document_content(pool, doc_id, "父文档全文")
        emb = [0.0] * get_settings().EMBEDDING_DIM
        await store.store_chunks_and_complete(
            pool, doc_id, BOOK_KB_ID, [(0, "c0", emb, {})]
        )

        # 批量补查：命中返回 {id: content}，未知 id 不出现在结果里
        contents = await store.get_documents_content(pool, [doc_id])
        assert contents == {doc_id: "父文档全文"}
        assert await store.get_documents_content(
            pool, ["00000000-0000-0000-0000-0000000000ff"]
        ) == {}
        assert await store.get_documents_content(pool, []) == {}

        # 检索行瘦身：两路 SQL 都不再携带全文
        vec_rows = await store.search_chunks(pool, emb, [BOOK_KB_ID], top_k=5)
        assert vec_rows
        assert all("document_content" not in r for r in vec_rows)

        bm25_rows = await store.search_chunks_bm25(pool, "c0", [BOOK_KB_ID], top_k=5)
        assert all("document_content" not in r for r in bm25_rows)
    finally:
        await pool.close()


async def test_get_chunks_by_uids_empty_short_circuit(monkeypatch):
    monkeypatch.setattr(
        "rag.document.store.get_cursor",
        lambda p: (_ for _ in ()).throw(AssertionError("不应建立游标")),
    )
    from rag.document.store import get_chunks_by_uids

    assert await get_chunks_by_uids(object(), []) == []


async def test_get_chunks_by_uids_builds_unnest_join(monkeypatch):
    class _Cur:
        def __init__(self):
            self.sql = ""
            self.params = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def execute(self, sql, params=None):
            self.sql = sql
            self.params = params or {}

        async def fetchall(self):
            return [{"id": "c1", "document_id": "d1", "chunk_index": 3, "text": "t"}]

    cur = _Cur()
    monkeypatch.setattr("rag.document.store.get_cursor", lambda p: cur)
    from rag.document.store import get_chunks_by_uids

    rows = await get_chunks_by_uids(object(), [("d1", 3), ("d2", 0)])

    assert rows[0]["id"] == "c1"
    assert "unnest" in cur.sql
    assert cur.params["docs"] == ["d1", "d2"]
    assert cur.params["idxs"] == [3, 0]


async def test_find_undelivered_documents_sql(monkeypatch):
    """投递 relay 扫描：只挑 delivery_status='pending' 且未 done 的文档。"""

    class _Cur:
        def __init__(self):
            self.sql = ""
            self.params = {}
            self.rows = [{"id": "d1"}, {"id": "d2"}]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def execute(self, sql, params=None):
            self.sql = sql
            self.params = params or {}

        async def fetchall(self):
            return self.rows

    cur = _Cur()
    monkeypatch.setattr("rag.document.store.get_cursor", lambda p: cur)
    from rag.document.store import find_undelivered_documents

    ids = await find_undelivered_documents(object())

    assert ids == ["d1", "d2"]
    assert "delivery_status = 'pending'" in cur.sql
    assert "status <> 'done'" in cur.sql


async def test_set_delivery_status_sql(monkeypatch):
    class _Cur:
        def __init__(self):
            self.sql = ""
            self.params = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def execute(self, sql, params=None):
            self.sql = sql
            self.params = params or {}

    cur = _Cur()
    monkeypatch.setattr("rag.document.store.get_cursor", lambda p: cur)
    from rag.document.store import set_delivery_status

    await set_delivery_status(object(), "doc-1", "sent")

    assert "UPDATE documents SET delivery_status" in cur.sql
    assert cur.params == {"s": "sent", "id": "doc-1"}
