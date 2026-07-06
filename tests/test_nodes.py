from types import SimpleNamespace

import rag.agent.nodes.generate.generate as generate_mod
import rag.agent.nodes.query.query as query_mod
import rag.agent.nodes.recall.recall as kb_recall_mod
import rag.agent.nodes.recall_memory.memory as recall_mod
from rag.agent.type import ContextSchema, StreamEventType, stream_event


class _FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return "rewritten"


class _FakeMM:
    def __init__(self):
        self.search_calls = []
        self.recent_calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.search_calls.append((session_id, query))
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        self.recent_calls.append(session_id)
        return [{"text": "S1"}]


async def test_recall_memory_awaits_and_composes_context(monkeypatch):
    monkeypatch.setattr(recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    mm = _FakeMM()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=mm))
    state = {"session_id": "s1", "raw_query": "q1"}

    out = await recall_mod.recall_memory(state, runtime)

    assert mm.search_calls == [("s1", "q1")]
    assert mm.recent_calls == ["s1"]
    assert out["context"] == "S1\nL1"


async def test_recall_memory_dedup_short_priority(monkeypatch):
    """短片优先，长片中与短片重复的内容应被去重。"""
    monkeypatch.setattr(recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _DupMM:
        async def search(self, session_id, query, top_k=5, filters=None):
            return [{"text": "dup"}, {"text": "L1"}]

        async def get_recent_messages(self, session_id, n=10):
            return [{"text": "dup"}, {"text": "S1"}]

    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=_DupMM())
    )
    out = await recall_mod.recall_memory(
        {"session_id": "s1", "raw_query": "q1"}, runtime
    )

    # dup 只出现一次（短片中的），L1 保留
    assert out["context"] == "dup\nS1\nL1"


async def test_handle_query_awaits_ainvoke(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q1", "context": "ctx"}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "rewritten"
    assert len(llm.calls) == 1


async def test_recall_searches_kb_with_rewrite_query(monkeypatch):
    monkeypatch.setattr(
        kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _FakeRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_id, top_k=5):
            self.calls.append((query, knowledge_base_id))
            return [{"text": "KB1"}]

    retriever = _FakeRetriever()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, retriever=retriever)
    )
    state = {"session_id": "s1", "raw_query": "q1", "rewrite_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["recall_vec_results"] == [{"text": "KB1"}]
    assert retriever.calls[0][0] == "rw"


async def test_recall_no_retriever_yields_empty(monkeypatch):
    monkeypatch.setattr(
        kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, retriever=None)
    )
    state = {"session_id": "s1", "raw_query": "q1", "rewrite_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["recall_vec_results"] == []


async def test_generate_writes_back_memory(monkeypatch):
    """生成结束后本轮问答须写回短期+长期记忆。"""
    monkeypatch.setattr(
        generate_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _WritableMM:
        def __init__(self):
            self.added = []

        async def add_message(self, session_id, text, metadata=None):
            self.added.append((session_id, text, metadata))

        async def add(self, session_id, text, metadata=None):
            self.added.append((session_id, text, metadata))

    class _StreamLLM:
        async def astream(self, messages):
            for tok in ("答", "案"):
                yield tok

    mm = _WritableMM()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=_StreamLLM(), memory_manager=mm)
    )
    state = {"session_id": "s1", "raw_query": "问题", "rewrite_query": "q", "context": ""}

    await generate_mod.generate(state, runtime)

    assert len(mm.added) == 4  # 2 短期 + 2 长期
    sid, user_text, user_meta = mm.added[0]
    assert sid == "s1" and "问题" in user_text and user_meta == {"role": "user"}
    sid, asst_text, asst_meta = mm.added[1]
    assert sid == "s1" and "答案" in asst_text and asst_meta == {"role": "assistant"}


async def test_generate_memory_write_failure_does_not_fail_request(monkeypatch):
    monkeypatch.setattr(
        generate_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _BrokenMM:
        async def add_message(self, session_id, text, metadata=None):
            raise RuntimeError("redis down")

        async def add(self, session_id, text, metadata=None):
            raise RuntimeError("pg down")

    class _StreamLLM:
        async def astream(self, messages):
            yield "答案"

    runtime = SimpleNamespace(
        context=ContextSchema(llm=_StreamLLM(), memory_manager=_BrokenMM())
    )
    state = {"session_id": "s1", "raw_query": "问题", "rewrite_query": "q", "context": ""}

    out = await generate_mod.generate(state, runtime)

    assert out["generated"] == "答案"


async def test_generate_streams_tokens_and_accumulates(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        generate_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev))
    )

    class _StreamLLM:
        async def astream(self, messages):
            for tok in ("答", "案"):
                yield tok

    runtime = SimpleNamespace(
        context=ContextSchema(llm=_StreamLLM(), memory_manager=None)
    )
    state = {"session_id": "s1", "raw_query": "问题", "rewrite_query": "q", "context": "ctx"}

    out = await generate_mod.generate(state, runtime)

    assert out["generated"] == "答案"
    assert stream_event(StreamEventType.MESSAGE, "答") in emitted
    assert stream_event(StreamEventType.MESSAGE, "案") in emitted
