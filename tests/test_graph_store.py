import os

import pytest

pytestmark = pytest.mark.integration

neo4j = pytest.importorskip("neo4j")
from rag.graph.store import purge_document, write_graph  # noqa: E402


@pytest.fixture
async def driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "neo4j_pass")
    drv = neo4j.AsyncGraphDatabase.driver(uri, auth=(user, pw))
    async with drv.session() as s:
        await s.run("MATCH (n:Entity) DETACH DELETE n")
    yield drv
    async with drv.session() as s:
        await s.run("MATCH (n:Entity) DETACH DELETE n")
    await drv.close()


async def _counts(driver):
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity) WITH count(e) AS ec "
            "OPTIONAL MATCH ()-[r:RELATES]->() RETURN ec, count(r) AS rc"
        )
        rec = await r.single()
        return rec["ec"], rec["rc"]


async def test_write_then_rerun_is_idempotent(driver):
    ents = [
        {"name": "孙悟空", "type": "Person", "chunk_ids": ["d1:0"]},
        {"name": "唐僧", "type": "Person", "chunk_ids": ["d1:0"]},
    ]
    rels = [{"source": "孙悟空", "target": "唐僧", "keywords": ["师徒"]}]

    await purge_document(driver, "neo4j", "d1")
    await write_graph(driver, "neo4j", "d1", ents, rels)
    assert await _counts(driver) == (2, 1)

    # 重跑同一文档:先 purge 再写,数量不翻倍
    await purge_document(driver, "neo4j", "d1")
    await write_graph(driver, "neo4j", "d1", ents, rels)
    assert await _counts(driver) == (2, 1)


async def test_shared_entity_not_deleted_on_other_doc_purge(driver):
    await write_graph(
        driver, "neo4j", "d1",
        [{"name": "孙悟空", "type": "Person", "chunk_ids": ["d1:0"]}], [],
    )
    await write_graph(
        driver, "neo4j", "d2",
        [{"name": "孙悟空", "type": "Person", "chunk_ids": ["d2:0"]}], [],
    )
    # 孙悟空被 d1、d2 共享,purge d1 后节点仍在,只剩 d2 的 chunk_id
    await purge_document(driver, "neo4j", "d1")
    async with driver.session() as s:
        r = await s.run("MATCH (e:Entity {name:'孙悟空'}) RETURN e.chunk_ids AS c, e.doc_ids AS d")
        rec = await r.single()
    assert rec["c"] == ["d2:0"]
    assert rec["d"] == ["d2"]
