from types import SimpleNamespace

import rag.agent.nodes.add_memory.memory as add_memory_mod
import rag.agent.nodes.generate.direct_answer as direct_answer_mod
import rag.agent.nodes.generate.generate as generate_mod
import rag.agent.nodes.generate.no_results as no_results_mod
import rag.agent.nodes.query.query as query_mod
import rag.agent.nodes.recall.recall as kb_recall_mod
import rag.agent.nodes.recall_memory.memory as recall_mod
import rag.agent.nodes.rerank.rerank as rerank_mod
import rag.agent.nodes.dynamic_topk.topk as topk_mod
import rag.agent.nodes.neighbor_expand.expand as neighbor_expand_mod
import rag.agent.nodes.parent_expand.expand as parent_expand_mod
import rag.agent.nodes.cache_lookup.lookup as cache_lookup_mod
import rag.agent.nodes.cache_store.store as cache_store_mod
from rag.agent.type import ContextSchema, StreamEventType, stream_event
from rag.document import REGULATION_KB_ID


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

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append((query, knowledge_base_ids))
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


async def test_rerank_passthrough_when_no_reranker():
    """未注入 reranker 时节点应透传原始结果。"""
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


async def test_rerank_empty_chunks_skips():
    """空 recall 结果直接跳过，不调 reranker。"""

    class _NoCallReranker:
        async def rerank(self, query, chunks, top_k=None):
            raise RuntimeError("should not be called")

    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, reranker=_NoCallReranker())
    )
    state = {"recall_vec_results": [], "raw_query": "q"}

    out = await rerank_mod.rerank(state, runtime)  # 不应抛异常
    assert out["recall_vec_results"] == []


# ── dynamic_truncate pure function tests ──


async def test_dynamic_truncate_empty():
    """空列表应直接返回空列表。"""
    assert topk_mod._dynamic_truncate([], 5, 0.7) == []


async def test_dynamic_truncate_single():
    """只有一条结果时直接返回，不做比较。"""
    chunk = {"rerank_score": 9.0, "text": "A"}
    result = topk_mod._dynamic_truncate([chunk], 5, 0.7)
    assert result == [chunk]


async def test_dynamic_truncate_no_gap_returns_default_topk():
    """所有相邻分差都小，应返回 default_top_k 条。"""
    chunks = [
        {"rerank_score": 9.0},
        {"rerank_score": 8.5},
        {"rerank_score": 8.0},
        {"rerank_score": 7.5},
        {"rerank_score": 7.0},
        {"rerank_score": 6.5},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5
    assert result[0]["rerank_score"] == 9.0
    assert result[4]["rerank_score"] == 7.0


async def test_dynamic_truncate_gap_truncates():
    """第 2-3 名之间 gap 大（1.0/8.5=0.118 < 0.7），且 post-gap 内部也有 gap
    （0.3/1.0=0.3 < 0.7），非高原 → 应在第 2 条后截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 8.5, "text": "B"},
        {"rerank_score": 1.0, "text": "C"},
        {"rerank_score": 0.3, "text": "D"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 2
    assert result[0]["text"] == "A"
    assert result[1]["text"] == "B"


async def test_dynamic_truncate_plateau_skips_gap():
    """gap 之后的分数形成高原（内部比值均 >= threshold），跳过此 gap 继续扫描。
    模拟场景：前 2 条高分 → 一条低分 → 但后续多条分数紧密（第二梯队），不应截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 8.5, "text": "B"},
        {"rerank_score": 1.0, "text": "C"},
        {"rerank_score": 0.98, "text": "D"},
        {"rerank_score": 0.96, "text": "E"},
        {"rerank_score": 0.95, "text": "F"},
    ]
    # 1.0/8.5=0.118 < 0.7 → gap 候选
    # post=[1.0, 0.98, 0.96, 0.95] 内部: 0.98, 0.98, 0.99 均 >= 0.7 → plateau → 跳过
    # 后续无 gap → 取 default_top_k=5
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5
    assert [r["text"] for r in result] == ["A", "B", "C", "D", "E"]


async def test_dynamic_truncate_gap_after_first_returns_one():
    """第一名本身分数极低且与第二名 gap 大，保留第1条。"""
    chunks = [
        {"rerank_score": 1.0, "text": "A"},
        {"rerank_score": 0.1, "text": "B"},
    ]
    # score[1]/score[0] = 0.1/1.0 = 0.1 < 0.7 → 截断保留前 1 条
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 1


async def test_dynamic_truncate_score_zero_infinite_gap():
    """score 为 0 时视为无限大 gap，在 0 分处截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 0.0, "text": "B"},
        {"rerank_score": 0.0, "text": "C"},
    ]
    # score[1]=0.0, score[0]=9.0 → 0/9=0 < threshold, 截断到前 1 条
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 1
    assert result[0]["text"] == "A"


async def test_dynamic_truncate_first_score_zero_empty():
    """第一名 score 就是 0，直接在首位截断，返回空。"""
    chunks = [
        {"rerank_score": 0.0, "text": "A"},
        {"rerank_score": 9.0, "text": "B"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert result == []


async def test_dynamic_truncate_missing_rerank_score_hard_truncate():
    """无 rerank_score 字段时（rerank 未启用），硬截断 default_top_k。"""
    chunks = [
        {"text": "A"}, {"text": "B"}, {"text": "C"},
        {"text": "D"}, {"text": "E"}, {"text": "F"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


async def test_dynamic_truncate_all_same_score():
    """所有分数完全相同时，无 gap，取 default_top_k。"""
    chunks = [
        {"rerank_score": 5.0} for _ in range(10)
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


async def test_dynamic_truncate_negative_score_treated_as_zero():
    """负分数视为 0 处理，触发 gap 截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": -1.0, "text": "B"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    # -1 → 0, gap 无限大，截断到前 1 条
    assert len(result) == 1
    assert result[0]["text"] == "A"


async def test_dynamic_truncate_extreme_threshold():
    """ratio_threshold 为 0.01 时几乎不截断；0.99 时几乎总是截断。"""
    chunks = [
        {"rerank_score": 9.0},
        {"rerank_score": 8.0},
        {"rerank_score": 7.0},
    ]
    # threshold 很低 → 极小的比值才会触发截断
    r1 = topk_mod._dynamic_truncate(chunks, 5, 0.01)
    # 8/9 ≈ 0.89 > 0.01, 7/8=0.875 > 0.01, no gap → all 3
    assert len(r1) == 3

    # threshold 很高 → 几乎任何相邻变化都截断
    r2 = topk_mod._dynamic_truncate(chunks, 5, 0.99)
    # 8/9 ≈ 0.89 < 0.99 → 截断到前 1 条
    assert len(r2) == 1


# ── dynamic_topk node integration tests ──


async def test_dynamic_topk_enabled_filters(monkeypatch):
    """启用时节点应调用 _dynamic_truncate 并写回结果。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    # 确保启用
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 0.5, "text": "B"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    # gap 9→0.5 (ratio 0.056 < 0.7) → 截断到前 1 条
    assert len(out["recall_vec_results"]) == 1
    assert out["recall_vec_results"][0]["text"] == "A"


async def test_dynamic_topk_disabled_passthrough(monkeypatch):
    """禁用时节点应透传原始结果不做任何修改。"""
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": False}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0}, {"rerank_score": 8.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] is chunks  # 引用不变


async def test_dynamic_topk_empty_passthrough(monkeypatch):
    """空结果直接透传，不调 _dynamic_truncate。"""
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    state = {"recall_vec_results": [], "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == []


async def test_dynamic_topk_non_list_passthrough(monkeypatch):
    """recall_vec_results 不是 list 时透传，不抛异常。"""
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    state = {"recall_vec_results": "not a list", "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == "not a list"


async def test_dynamic_topk_all_truncated_to_zero_emits_status(monkeypatch):
    """截断到 0 条时应发送 '未找到足够相关内容' 状态。"""
    emitted = []
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    # 第一个 chunk rerank_score=0.0 → scores[0]=0 → 在 i=0 处截断返回 []
    chunks = [{"rerank_score": 0.0}, {"rerank_score": 9.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == []
    assert stream_event(StreamEventType.STATUS, "未找到足够相关内容") in emitted


async def test_dynamic_topk_exception_passthrough(monkeypatch):
    """节点内部异常时应捕获并透传原始结果，不抛异常。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    # 让 _dynamic_truncate 抛异常
    monkeypatch.setattr(topk_mod, "_dynamic_truncate", lambda c, d, r: (_ for _ in ()).throw(ValueError("boom")))

    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    # 透传原始结果，不抛异常
    assert out["recall_vec_results"] is chunks


# ── parent_expand node integration tests ──


class _FakeParentRetriever:
    """支持 fetch_parent_contents 的假检索器：记录调用并返回预设全文。"""

    def __init__(self, contents=None, exc=None):
        self.contents = contents or {}
        self.exc = exc
        self.calls = []

    async def fetch_parent_contents(self, document_ids):
        self.calls.append(sorted(document_ids))
        if self.exc is not None:
            raise self.exc
        return self.contents


def _parent_runtime(retriever=None):
    return SimpleNamespace(context=SimpleNamespace(retriever=retriever))


async def test_parent_expand_enabled_expands(monkeypatch):
    """法规 KB chunk 应去重，并通过 retriever 按 doc_id 补查全文替换 text。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever({"doc-1": "全文内容A", "doc-2": "全文内容C"})
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-1", "rerank_score": 8.0, "text": "chunk B",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-2", "rerank_score": 7.0, "text": "chunk C",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 2  # doc-1 去重，doc-2 独立
    # 补查只发起一次，doc_id 已去重
    assert retriever.calls == [["doc-1", "doc-2"]]
    # doc-1 取最高 rerank_score 的 chunk，text 替换为全文
    doc1 = next(r for r in results if r["document_id"] == "doc-1")
    assert doc1["text"] == "全文内容A"
    assert doc1["rerank_score"] == 9.0
    doc2 = next(r for r in results if r["document_id"] == "doc-2")
    assert doc2["text"] == "全文内容C"


async def test_parent_expand_disabled_passthrough(monkeypatch):
    """禁用时节点应透传原始结果不做任何修改，也不发起补查。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": False}
    )())
    retriever = _FakeParentRetriever({"doc-1": "全文A"})
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "text": "chunk A"},
        {"document_id": "doc-1", "text": "chunk B"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    assert out["recall_vec_results"] is chunks  # 引用不变
    assert retriever.calls == []


async def test_parent_expand_no_content_keeps_text(monkeypatch):
    """法规 KB 但补查无全文（如旧文档 content 为空）时按文档去重但保留原始 text。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever({})
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-1", "rerank_score": 8.0, "text": "chunk B",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1  # 按文档去重
    assert results[0]["text"] == "chunk A"  # 保留原始 text（无全文可展开）


async def test_parent_expand_fetch_failure_keeps_text(monkeypatch):
    """补查失败时降级：按文档去重但保留原始 text，不向上抛。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever(exc=RuntimeError("db down"))
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1
    assert results[0]["text"] == "chunk A"


async def test_parent_expand_no_retriever_keeps_text(monkeypatch):
    """runtime 未注入 retriever 时降级为仅去重。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1
    assert results[0]["text"] == "chunk A"


async def test_parent_expand_different_docs_kept(monkeypatch):
    """不同法规文档各自保留，去重仅在文档内生效。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever(
        {"doc-1": "全文A", "doc-2": "全文B", "doc-3": "全文C"}
    )
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-2", "rerank_score": 8.0, "text": "B",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-3", "rerank_score": 7.0, "text": "C",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3
    doc_ids = {r["document_id"] for r in results}
    assert doc_ids == {"doc-1", "doc-2", "doc-3"}
    assert [r["text"] for r in results] == ["全文A", "全文B", "全文C"]


async def test_parent_expand_non_regulation_passthrough(monkeypatch):
    """非法规 KB 的 chunk 不展开也不补查，仅按文档去重保留原始 text。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever({"doc-1": "全文A"})
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": "00000000-0000-0000-0000-000000000002"},
        {"document_id": "doc-1", "rerank_score": 8.0, "text": "chunk B",
         "knowledge_base_id": "00000000-0000-0000-0000-000000000002"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1  # 按文档去重
    assert results[0]["text"] == "chunk A"  # text 是原始 chunk 文本，未展开为全文
    assert retriever.calls == []  # 非法规 KB 不触发补查


async def test_parent_expand_orphan_chunks_kept(monkeypatch):
    """无 document_id 的孤立 chunk 应原样保留在末尾。"""
    monkeypatch.setattr(parent_expand_mod, "get_settings", lambda: type(
        "S", (), {"PARENT_CHILD_ENABLED": True}
    )())
    retriever = _FakeParentRetriever({"doc-1": "全文A"})
    runtime = _parent_runtime(retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"rerank_score": 5.0, "text": "orphan"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await parent_expand_mod.parent_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 2
    assert results[0]["text"] == "全文A"  # 法规 chunk 展开为补查到的全文
    assert results[-1]["text"] == "orphan"  # orphan 在末尾
    assert "document_id" not in results[-1] or not results[-1]["document_id"]


# ── neighbor_expand node integration tests ──


async def test_neighbor_expand_enabled_adds_context(monkeypatch):
    """启用时邻居 chunk 应被拉取并追加到结果中。"""
    settings = type("S", (), {
        "SENTENCE_WINDOW_ENABLED": True,
        "SENTENCE_WINDOW_SIZE": 2,
        "SENTENCE_WINDOW_MAX_MULTIPLIER": 3,
    })()
    monkeypatch.setattr(neighbor_expand_mod, "get_settings", lambda: settings)

    async def fake_neighbors(pool, doc_id, center, window):
        if doc_id == "doc-1" and center == 5:
            return [
                {"chunk_index": 3, "text": "neighbor before", "metadata": {}},
                {"chunk_index": 6, "text": "neighbor after", "metadata": {}},
            ]
        return []

    monkeypatch.setattr(neighbor_expand_mod.store, "get_neighbor_chunks", fake_neighbors)

    pool = SimpleNamespace()
    runtime = SimpleNamespace(context=SimpleNamespace(pool=pool))
    chunks = [
        {"document_id": "doc-1", "chunk_index": 5, "rerank_score": 0.9, "text": "center",
         "filename": "test.txt", "knowledge_base_id": "kb-1", "sources": ["vec"]},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await neighbor_expand_mod.neighbor_expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3  # 原始 1 + 邻居 2
    texts = {r["text"] for r in results}
    assert "center" in texts
    assert "neighbor before" in texts
    assert "neighbor after" in texts


async def test_neighbor_expand_disabled_passthrough(monkeypatch):
    """禁用时节点应透传原始结果。"""
    settings = type("S", (), {
        "SENTENCE_WINDOW_ENABLED": False,
    })()
    monkeypatch.setattr(neighbor_expand_mod, "get_settings", lambda: settings)

    runtime = SimpleNamespace(context=SimpleNamespace(pool=SimpleNamespace()))
    chunks = [{"document_id": "doc-1", "chunk_index": 1, "text": "A"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await neighbor_expand_mod.neighbor_expand(state, runtime)

    assert out["recall_vec_results"] is chunks


async def test_neighbor_expand_no_pool_skips(monkeypatch):
    """pool 未注入时静默跳过，不报错。"""
    settings = type("S", (), {
        "SENTENCE_WINDOW_ENABLED": True,
    })()
    monkeypatch.setattr(neighbor_expand_mod, "get_settings", lambda: settings)

    runtime = SimpleNamespace(context=SimpleNamespace(pool=None))
    chunks = [{"document_id": "doc-1", "chunk_index": 1, "text": "A"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await neighbor_expand_mod.neighbor_expand(state, runtime)

    assert out["recall_vec_results"] is chunks


async def test_neighbor_expand_dedup(monkeypatch):
    """已存在的 chunk 不应被邻居重复添加。"""
    settings = type("S", (), {
        "SENTENCE_WINDOW_ENABLED": True,
        "SENTENCE_WINDOW_SIZE": 2,
        "SENTENCE_WINDOW_MAX_MULTIPLIER": 3,
    })()
    monkeypatch.setattr(neighbor_expand_mod, "get_settings", lambda: settings)

    async def fake_neighbors(pool, doc_id, center, window):
        # 返回邻居中包含中心 chunk 自身（应被去重）
        return [
            {"chunk_index": 4, "text": "already there", "metadata": {}},
            {"chunk_index": 6, "text": "new neighbor", "metadata": {}},
        ]

    monkeypatch.setattr(neighbor_expand_mod.store, "get_neighbor_chunks", fake_neighbors)

    pool = SimpleNamespace()
    runtime = SimpleNamespace(context=SimpleNamespace(pool=pool))
    chunks = [
        {"document_id": "doc-1", "chunk_index": 4, "text": "already there",
         "filename": "t.txt", "sources": ["vec"]},
        {"document_id": "doc-1", "chunk_index": 5, "text": "center",
         "filename": "t.txt", "sources": ["vec"]},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await neighbor_expand_mod.neighbor_expand(state, runtime)

    results = out["recall_vec_results"]
    # doc-1 idx=4 已存在，不应重复；idx=6 是新邻居
    assert len(results) == 3


# ── updated route tests ──


async def test_route_after_topk_empty():
    """动态截断后为空时路由到 no_results。"""
    import rag.agent.workflow as wf

    assert wf._route_after_topk({"recall_vec_results": []}) == "no_results"
    assert wf._route_after_topk({}) == "no_results"


async def test_route_after_topk_has_results():
    """动态截断后有结果时路由到 generate。"""
    import rag.agent.workflow as wf

    assert wf._route_after_topk({"recall_vec_results": [{"text": "KB1"}]}) == "generate"


class _FakeCache:
    def __init__(self, hit=None):
        self.hit = hit
        self.lookup_calls = []
        self.store_calls = []

    async def lookup(self, query, session_id=None):
        self.lookup_calls.append((query, session_id))
        return self.hit

    async def store(self, query, answer, citations):
        self.store_calls.append((query, answer, citations))


def _cache_settings(monkeypatch, mod, enabled=True):
    from types import SimpleNamespace
    monkeypatch.setattr(
        mod, "get_settings",
        lambda: SimpleNamespace(SEMANTIC_CACHE_ENABLED=enabled),
    )


async def test_cache_lookup_disabled_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=False)
    cache = _FakeCache(hit={"answer": "A", "citations": []})
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "rewrite_query": "q2"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert cache.lookup_calls == []
    assert "cache_hit" not in out


async def test_cache_lookup_no_cache_injected_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert "cache_hit" not in out


async def test_cache_lookup_hit_sets_state_and_streams(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    events = []
    monkeypatch.setattr(
        cache_lookup_mod, "get_stream_writer", lambda: events.append
    )
    cache = _FakeCache(hit={"answer": "缓存答案", "citations": [{"index": 1}]})
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "rewrite_query": "改写q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert cache.lookup_calls == [("改写q", "s1")]
    assert out["cache_hit"] is True
    assert out["generated"] == "缓存答案"
    assert out["citations"] == [{"index": 1}]
    types = [e.get("type") for e in events]
    assert types == ["status", "message", "citations"]


async def test_cache_lookup_miss_leaves_state(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    monkeypatch.setattr(
        cache_lookup_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )
    cache = _FakeCache(hit=None)
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert "cache_hit" not in out
    assert "generated" not in out


async def test_cache_store_writes_generated(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=True)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {
        "session_id": "s1", "raw_query": "q", "rewrite_query": "改写q",
        "generated": "新答案", "citations": [{"index": 2}],
    }

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == [("改写q", "新答案", [{"index": 2}])]


async def test_cache_store_skips_empty_answer(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=True)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "generated": ""}

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == []


async def test_cache_store_disabled_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=False)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "generated": "a"}

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == []
