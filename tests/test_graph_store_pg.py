import uuid

import pytest

pytestmark = pytest.mark.integration

from rag.config import get_settings  # noqa: E402
from rag.db import create_pg_pool  # noqa: E402
from rag.document import store  # noqa: E402


@pytest.fixture
async def pool():
    p = await create_pg_pool(get_settings())
    yield p
    await p.close()


async def _mk_doc(pool) -> str:
    return await store.create_document(
        pool, knowledge_base_id="00000000-0000-0000-0000-000000000001",
        filename="f.txt", content_type="text/plain", size_bytes=1,
        content_hash="h" + uuid.uuid4().hex, object_key="k" + uuid.uuid4().hex,
    )


async def test_claim_graph_processing_atomic(pool):
    doc_id = await _mk_doc(pool)
    assert await store.claim_graph_processing(pool, doc_id) is True
    # 已 processing,再领取失败
    assert await store.claim_graph_processing(pool, doc_id) is False


async def test_set_graph_status_and_error(pool):
    doc_id = await _mk_doc(pool)
    await store.set_graph_status(pool, doc_id, "failed", error="boom")
    doc = await store.get_document(pool, doc_id)
    assert doc["graph_status"] == "failed"
    assert doc["graph_error"] == "boom"


async def test_get_chunks_for_graph_orders_and_extracts_title(pool):
    doc_id = await _mk_doc(pool)
    await store.store_chunks_and_complete(
        pool, doc_id, "00000000-0000-0000-0000-000000000001",
        [
            (1, "second", [0.0] * 1024, {"title": "第二章"}),
            (0, "first", [0.0] * 1024, {}),
        ],
    )
    rows = await store.get_chunks_for_graph(pool, doc_id)
    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert rows[0]["title"] is None
    assert rows[1]["title"] == "第二章"
