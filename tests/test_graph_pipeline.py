from types import SimpleNamespace

import rag.graph.pipeline as gp
from rag.document.entity_extraction import Entity, ExtractionResult, Relationship


def _ctx():
    return {
        "pg": None, "neo4j": object(),
        "settings": SimpleNamespace(NEO4J_DATABASE="neo4j"),
        "llm": object(),
    }


async def test_skips_when_not_claimed(monkeypatch):
    async def fake_claim(pool, doc_id):
        return False

    called = []
    async def fail(*a, **k):
        called.append(a)
        raise AssertionError("未领取不应继续")

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fail)

    await gp.extract_document_entities(_ctx(), "d1")
    assert called == []


async def test_happy_path_writes_graph_and_marks_done(monkeypatch):
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_chunks(pool, doc_id):
        return [
            {"chunk_index": 0, "text": "孙悟空拜唐僧为师", "title": "第一回"},
        ]

    async def fake_extract(llm, text, *, chapter_context=None, **kw):
        return ExtractionResult(
            entities=[Entity("孙悟空", "Person", ""), Entity("唐僧", "Person", "")],
            relationships=[Relationship("孙悟空", "唐僧", "师徒", "")],
        )

    async def fake_purge(driver, database, doc_id):
        events.append(("purge", database, doc_id))

    async def fake_write(driver, database, doc_id, entities, relations):
        events.append(("write", doc_id, [e["name"] for e in entities], len(relations)))

    async def fake_status(pool, doc_id, status, error=None):
        events.append(("status", status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fake_chunks)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)
    monkeypatch.setattr(gp, "extract_entities", fake_extract)
    monkeypatch.setattr(gp, "purge_document", fake_purge)
    monkeypatch.setattr(gp, "write_graph", fake_write)

    await gp.extract_document_entities(_ctx(), "d1")

    assert ("purge", "neo4j", "d1") in events
    write_evt = [e for e in events if e[0] == "write"][0]
    assert sorted(write_evt[2]) == ["唐僧", "孙悟空"]
    assert write_evt[3] == 1
    assert ("status", "done", None) in events


async def test_failure_marks_failed_and_not_reraise(monkeypatch):
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def boom(pool, doc_id):
        raise RuntimeError("db down")

    async def fake_status(pool, doc_id, status, error=None):
        events.append((status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", boom)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)

    # 阶段一:不 re-raise
    await gp.extract_document_entities(_ctx(), "d1")
    assert events and events[0][0] == "failed" and "db down" in events[0][1]
