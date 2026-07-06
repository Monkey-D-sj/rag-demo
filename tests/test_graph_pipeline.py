import asyncio
from types import SimpleNamespace

import pytest

import rag.graph.pipeline as gp
from rag.document.entity_extraction import Entity, ExtractionResult, Relationship


async def test_cancellation_marks_graph_failed_and_reraises(monkeypatch):
    """arq 超时取消(CancelledError)必须置 graph_status=failed 后 re-raise,
    否则卡在 processing 且 retry 白名单救不回。"""
    statuses = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_chunks(pool, doc_id):
        return [{"chunk_index": 0, "text": "文本", "title": None}]

    async def fake_extract(llm, text, *, chapter_context=None, **kw):
        raise asyncio.CancelledError

    async def fake_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fake_chunks)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)
    monkeypatch.setattr(gp, "extract_entities", fake_extract)

    with pytest.raises(asyncio.CancelledError):
        await gp.extract_document_entities(_ctx(), "d1")
    assert len(statuses) == 1
    assert statuses[0][0] == "failed"


def _ctx():
    return {
        "pg": None, "neo4j": object(),
        "settings": SimpleNamespace(NEO4J_DATABASE="neo4j", GRAPH_EXTRACT_CONCURRENCY=4),
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
            entities=[
                Entity(name="孙悟空", type="Person"),
                Entity(name="唐僧", type="Person"),
            ],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
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


async def test_partial_chunk_failure_skips_and_marks_done(monkeypatch):
    """部分 chunk 抽取失败:跳过失败项,用其余成功结果写入,整篇仍标 done。"""
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_chunks(pool, doc_id):
        return [
            {"chunk_index": 0, "text": "孙悟空拜唐僧为师", "title": "第一回"},
            {"chunk_index": 1, "text": "BOOM 触发失败", "title": "第二回"},
        ]

    async def fake_extract(llm, text, *, chapter_context=None, **kw):
        if "BOOM" in text:
            raise RuntimeError("llm down")
        return ExtractionResult(
            entities=[
                Entity(name="孙悟空", type="Person"),
                Entity(name="唐僧", type="Person"),
            ],
            relationships=[
                Relationship(source="孙悟空", target="唐僧", keywords="师徒"),
            ],
        )

    async def fake_purge(driver, database, doc_id):
        events.append(("purge", doc_id))

    async def fake_write(driver, database, doc_id, entities, relations):
        events.append(("write", sorted(e["name"] for e in entities)))

    async def fake_status(pool, doc_id, status, error=None):
        events.append(("status", status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fake_chunks)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)
    monkeypatch.setattr(gp, "extract_entities", fake_extract)
    monkeypatch.setattr(gp, "purge_document", fake_purge)
    monkeypatch.setattr(gp, "write_graph", fake_write)

    await gp.extract_document_entities(_ctx(), "d1")

    # 失败 chunk 被跳过,成功 chunk 的实体照常写入,整篇 done。
    assert ("write", ["唐僧", "孙悟空"]) in events
    assert ("status", "done", None) in events


async def test_all_chunks_fail_marks_failed_without_purge(monkeypatch):
    """全部 chunk 失败:不 purge 旧图,标 failed 交由 cron 重试自愈。"""
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_chunks(pool, doc_id):
        return [{"chunk_index": 0, "text": "x", "title": None}]

    async def boom_extract(llm, text, *, chapter_context=None, **kw):
        raise RuntimeError("llm down")

    async def fail_if_called(*a, **k):
        raise AssertionError("全失败不应 purge/write")

    async def fake_status(pool, doc_id, status, error=None):
        events.append((status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fake_chunks)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)
    monkeypatch.setattr(gp, "extract_entities", boom_extract)
    monkeypatch.setattr(gp, "purge_document", fail_if_called)
    monkeypatch.setattr(gp, "write_graph", fail_if_called)

    await gp.extract_document_entities(_ctx(), "d1")

    assert events and events[-1][0] == "failed"


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
