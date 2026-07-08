from types import SimpleNamespace

import rag.agent.nodes.add_memory.memory as add_memory_mod
import rag.agent.nodes.generate.direct_answer as direct_answer_mod
import rag.agent.nodes.generate.generate as generate_mod
import rag.agent.nodes.generate.no_results as no_results_mod
import rag.agent.nodes.query.query as query_mod
import rag.agent.nodes.recall.recall as kb_recall_mod
import rag.agent.nodes.recall_memory.memory as recall_mod
import rag.agent.nodes.rerank.rerank as rerank_mod
from rag.agent.type import ContextSchema, StreamEventType, stream_event


class _FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke_structured(self, messages, schema):
        from rag.agent.nodes.query.query import QueryRewriteOutput

        self.calls.append(messages)
        return QueryRewriteOutput(rewrite_query="rewritten", is_out_of_scope=False)


class _FakeMM:
    def __init__(self):
        self.search_calls = []
        self.recent_calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.search_calls.append((session_id, query))
        return [{"text": "L1", "metadata": {"role": "user"}}]

    async def get_recent_messages(self, session_id, n=10):
        self.recent_calls.append(session_id)
        return [{"text": "S1", "metadata": {"role": "assistant"}}]


async def test_recall_memory_awaits_and_composes_context(monkeypatch):
    monkeypatch.setattr(recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    mm = _FakeMM()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=mm))
    state = {"session_id": "s1", "raw_query": "q1"}

    out = await recall_mod.recall_memory(state, runtime)

    assert mm.search_calls == [("s1", "q1")]
    assert mm.recent_calls == ["s1"]
    assert out["context"] == "AI：S1\n用户：L1"


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


async def test_handle_query_structured_output_in_scope(monkeypatch):
    """查询在知识库范围内时，应返回改写后的查询且 is_out_of_scope=False。"""
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q1", "context": "ctx"}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "rewritten"
    assert out["is_out_of_scope"] is False
    assert len(llm.calls) == 1


async def test_handle_query_detects_out_of_scope(monkeypatch):
    """闲聊/无关查询应标记 is_out_of_scope=True。"""
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _OutOfScopeLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke_structured(self, messages, schema):
            self.calls.append(messages)
            return QueryRewriteOutput(rewrite_query="你好啊", is_out_of_scope=True)

    llm = _OutOfScopeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "你好啊", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "你好啊"
    assert out["is_out_of_scope"] is True
    assert len(llm.calls) == 1


async def test_direct_answer_uses_direct_prompt(monkeypatch):
    """direct_answer 节点应使用直接回答 prompt，不依赖知识库内容。"""
    captured_messages = []
    monkeypatch.setattr(
        direct_answer_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _CaptureLLM:
        async def astream(self, messages):
            captured_messages.extend(messages)
            yield "直接回答"

    llm = _CaptureLLM()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=llm, memory_manager=None)
    )
    state = {
        "session_id": "s1",
        "raw_query": "你好啊",
    }

    out = await direct_answer_mod.direct_answer(state, runtime)

    assert out["generated"] == "直接回答"
    # 确认使用了 direct_system_prompt 而非 kb_system_prompt
    system_msg = captured_messages[0]
    system_content = (
        system_msg.content if hasattr(system_msg, "content") else str(system_msg)
    )
    assert "知识库内容无关" in system_content or "直接基于你的知识" in system_content


async def test_direct_answer_does_not_persist(monkeypatch):
    """direct_answer 不写回记忆（闲聊/无关问题无上下文价值）。"""
    monkeypatch.setattr(
        direct_answer_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _SpyMM:
        def __init__(self):
            self.added = []

        async def add_message(self, session_id, text, metadata=None):
            self.added.append(("add_message", session_id, text, metadata))

        async def add(self, session_id, text, metadata=None):
            self.added.append(("add", session_id, text, metadata))

    class _StreamLLM:
        async def astream(self, messages):
            yield "兜底回答"

    mm = _SpyMM()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=_StreamLLM(), memory_manager=mm)
    )
    state = {"session_id": "s1", "raw_query": "你好"}

    await direct_answer_mod.direct_answer(state, runtime)

    assert mm.added == []  # 不写记忆


async def test_route_after_query_in_scope():
    """知识库范围内查询路由到 recall。"""
    import rag.agent.workflow as wf

    assert wf._route_after_query({"is_out_of_scope": False}) == "recall"
    assert wf._route_after_query({}) == "recall"  # 缺失时默认走检索


async def test_route_after_query_out_of_scope():
    """知识库范围外查询路由到 direct_answer。"""
    import rag.agent.workflow as wf

    assert wf._route_after_query({"is_out_of_scope": True}) == "direct_answer"


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


async def test_add_memory_calls_persist_turn():
    """add_memory 节点应将本轮问答通过 persist_turn 写回记忆。"""

    class _SpyMM:
        def __init__(self):
            self.persisted = []

        async def persist_turn(self, session_id, query, answer):
            self.persisted.append((session_id, query, answer))

    mm = _SpyMM()
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=mm)
    )
    state = {
        "session_id": "s1",
        "raw_query": "问题",
        "generated": "答案",
    }

    await add_memory_mod.add_memory(state, runtime)

    assert mm.persisted == [("s1", "问题", "答案")]


async def test_add_memory_none_manager_skips():
    """memory_manager 为 None 时 add_memory 应静默跳过。"""
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None)
    )
    state = {"session_id": "s1", "raw_query": "问题", "generated": "答案"}

    # 不应抛异常
    await add_memory_mod.add_memory(state, runtime)


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


async def test_no_results_emits_canned_message(monkeypatch):
    """召回为空时 no_results 节点应发送兜底话术，不调 LLM。"""
    emitted = []
    monkeypatch.setattr(
        no_results_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev))
    )
    state = {"session_id": "s1", "raw_query": "查无此问"}

    out = await no_results_mod.no_results(state)

    assert out["generated"] == no_results_mod._NO_RESULT_MSG
    assert "未在当前知识库中找到" in out["generated"]
    assert stream_event(StreamEventType.STATUS, "未找到相关内容") in emitted
    assert stream_event(StreamEventType.MESSAGE, no_results_mod._NO_RESULT_MSG) in emitted


async def test_route_after_recall_empty():
    """召回为空时路由到 no_results。"""
    import rag.agent.workflow as wf

    assert wf._route_after_recall({"recall_vec_results": []}) == "no_results"
    assert wf._route_after_recall({}) == "no_results"


async def test_route_after_recall_has_results():
    """召回有结果时路由到 generate。"""
    import rag.agent.workflow as wf

    assert wf._route_after_recall({"recall_vec_results": [{"text": "KB1"}]}) == "generate"


async def test_rerank_passthrough_when_no_reranker(monkeypatch):
    """未注入 reranker 时节点应透传原始结果。"""
    monkeypatch.setattr(rerank_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, reranker=None)
    )
    chunks = [{"id": "a", "text": "A"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await rerank_mod.rerank(state, runtime)

    assert out["recall_vec_results"] is chunks  # 引用不变，未重排


async def test_rerank_reorders_chunks(monkeypatch):
    """有 reranker 时 chunk 应按 LLM 打分重排。"""
    monkeypatch.setattr(rerank_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _FakeReranker:
        async def rerank(self, query, chunks, top_k=None):
            # 模拟：把 id="b" 的排到最前
            reordered = sorted(chunks, key=lambda c: c["id"], reverse=True)
            for c in reordered:
                c["rerank_score"] = 9.0 if c["id"] == "b" else 1.0
            return reordered

    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, reranker=_FakeReranker())
    )
    chunks = [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await rerank_mod.rerank(state, runtime)

    assert out["recall_vec_results"][0]["id"] == "b"
    assert out["recall_vec_results"][0]["rerank_score"] == 9.0


async def test_rerank_empty_chunks_skips(monkeypatch):
    """空 recall 结果直接跳过，不调 reranker。"""
    monkeypatch.setattr(rerank_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _NoCallReranker:
        async def rerank(self, query, chunks, top_k=None):
            raise RuntimeError("should not be called")

    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, reranker=_NoCallReranker())
    )
    state = {"recall_vec_results": [], "raw_query": "q"}

    out = await rerank_mod.rerank(state, runtime)  # 不应抛异常
    assert out["recall_vec_results"] == []
