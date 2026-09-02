"""agent 工具模块纯函数测试：handler 行形、去空、降级与工具注册表。"""

import pytest

from rag.agent.nodes.agent import tools as agent_tools
from rag.agent.type import ContextSchema


class _StubRetriever:
    def __init__(self, search_rows=None, contents=None):
        self.search_rows = search_rows or []
        self.contents = contents or {}
        self.search_calls = []

    async def search(self, query, knowledge_base_ids=None, top_k=5):
        self.search_calls.append((query, top_k))
        return self.search_rows

    async def fetch_parent_contents(self, document_ids):
        return {d: self.contents.get(d, "") for d in document_ids}


class _StubMemory:
    def __init__(self, rows=None):
        self.rows = rows or []

    async def search(self, session_id, query, top_k=5, filters=None):
        return self.rows


def _ctx(retriever=None, memory=None):
    return ContextSchema(llm=None, memory_manager=memory, retriever=retriever)


async def test_build_tools_returns_three_named_tools():
    ctx = _ctx(retriever=_StubRetriever())
    names = sorted(t.name for t in agent_tools.build_tools(ctx, "s1"))
    assert names == ["fetch_document", "retrieve_kb", "search_memory"]


async def test_kb_search_normalizes_rows_and_drops_empty_text():
    retriever = _StubRetriever(search_rows=[
        {"text": " 有内容 ", "filename": "《办法》", "metadata": {"article": "第十二条"}, "document_id": "d1", "chunk_index": 3},
        {"text": "   ", "document_id": "d2"},
        {"text": "第二段", "document_id": "d2", "chunk_index": 4},
    ])
    ctx = _ctx(retriever=retriever)
    rows = await agent_tools._kb_search(ctx, "s1", query="问题", top_k=5)
    assert len(rows) == 2
    assert rows[0] == {
        "text": "有内容",
        "filename": "《办法》",
        "metadata": {"article": "第十二条"},
        "document_id": "d1",
        "chunk_index": 3,
    }
    assert retriever.search_calls == [("问题", 5)]


async def test_kb_search_handles_retriever_none_and_raises():
    assert await agent_tools._kb_search(_ctx(), "s1", query="q") == []

    class _Boom:
        async def search(self, query, knowledge_base_ids=None, top_k=5):
            raise RuntimeError("db down")

    ctx = _ctx(retriever=_Boom())
    assert await agent_tools._kb_search(ctx, "s1", query="q") == []


async def test_kb_search_clamps_top_k_to_range():
    retriever = _StubRetriever()
    ctx = _ctx(retriever=retriever)
    await agent_tools._kb_search(ctx, "s1", query="q", top_k=999)
    await agent_tools._kb_search(ctx, "s1", query="q", top_k="x")
    assert [c[1] for c in retriever.search_calls] == [10, 5]


async def test_doc_fetch_returns_full_text_row():
    ctx = _ctx(retriever=_StubRetriever(contents={"d1": "全文内容"}))
    rows = await agent_tools._doc_fetch(ctx, "s1", document_id="d1")
    assert rows == [{
        "text": "全文内容",
        "filename": "d1",
        "metadata": {},
        "document_id": "d1",
        "chunk_index": None,
    }]


async def test_doc_fetch_empty_when_missing_or_no_retriever():
    ctx = _ctx(retriever=_StubRetriever(contents={}))
    assert await agent_tools._doc_fetch(ctx, "s1", document_id="nope") == []
    assert await agent_tools._doc_fetch(_ctx(), "s1", document_id="d1") == []


async def test_mem_search_empty_without_memory_manager():
    ctx = _ctx()
    assert await agent_tools._mem_search(ctx, "s1", query="q") == []


async def test_mem_search_wraps_rows_as_session_memory():
    ctx = _ctx(memory=_StubMemory(rows=[
        {"text": "上轮说过资质门槛 500 万", "metadata": {"role": "assistant"}},
        {"text": "  ", "metadata": {}},
    ]))
    rows = await agent_tools._mem_search(ctx, "s1", query="门槛")
    assert len(rows) == 1
    assert rows[0]["filename"] == "会话记忆"
    assert rows[0]["text"] == "上轮说过资质门槛 500 万"


def test_handlers_registry_covers_build_tools():
    from langchain_core.tools import BaseTool

    ctx = _ctx(retriever=_StubRetriever())
    for t in agent_tools.build_tools(ctx, "s1"):
        assert isinstance(t, BaseTool)
        assert t.name in agent_tools.HANDLERS


def test_render_empty_and_content():
    assert agent_tools._render([]) == "未检索到相关内容。"
    text = agent_tools._render([{"text": "x", "filename": "《A》"}])
    assert "《A》" in text and "x" in text
