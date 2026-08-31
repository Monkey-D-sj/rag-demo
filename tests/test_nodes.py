from types import SimpleNamespace

import rag.agent.nodes.add_memory.memory as add_memory_mod
import rag.agent.nodes.generate.generate as generate_mod
import rag.agent.nodes.generate.no_results as no_results_mod
import rag.agent.nodes.query.query as query_mod
import rag.agent.nodes.recall.recall as kb_recall_mod
import rag.agent.nodes.recall_memory.memory as recall_mod
import rag.agent.nodes.rerank.rerank as rerank_mod
import rag.agent.nodes.dynamic_topk.topk as topk_mod
import rag.agent.nodes.expand.expand as expand_mod
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
        return QueryRewriteOutput(
            rewrite_query="rewritten", is_out_of_scope=False
        )


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
    assert "当前日期:" in llm.calls[0][1].content


def test_query_rewrite_prompt_normalizes_time_and_retrieval_vocabulary():
    """查询改写层负责时间与检索词归一化，retriever 不承载领域同义词规则。"""
    from rag.prompts.query import system_prompt

    assert "时间归一化" in system_prompt
    assert "检索词归一化" in system_prompt
    assert "2026年贷款贴息、贷款补贴、贷款扶持政策" in system_prompt
    assert "不得臆造" in system_prompt


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

        async def astream(self, messages):
            yield "你好呀"

    llm = _OutOfScopeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "你好啊", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "你好啊"
    assert out["is_out_of_scope"] is True
    assert len(llm.calls) == 1


async def test_handle_query_out_of_scope_clamps_skipped_fields(monkeypatch):
    """短路命中(is_out_of_scope=true)时,即使 LLM 违反契约输出改写/拆解/会话语,节点也应钳为固定值。"""
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _NonCompliantLLM:
        async def ainvoke_structured(self, messages, schema):
            # 违反任务优先级:短路后仍输出任务二/三/四的内容
            return QueryRewriteOutput(
                rewrite_query="帮我写一个快速排序的完整实现",
                is_out_of_scope=True,
                answer_from_context=True,
                sub_queries=["排序算法怎么写"],
            )

        async def astream(self, messages):
            yield "直接回答"

    runtime = SimpleNamespace(context=ContextSchema(llm=_NonCompliantLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "帮我写个快速排序", "context": ""}, runtime
    )

    assert out["is_out_of_scope"] is True
    assert out["rewrite_query"] == "帮我写个快速排序"  # 任务三短路,回退原文
    assert out["answer_from_context"] is False  # 任务二短路
    assert out["sub_queries"] == []  # 任务四短路


async def test_handle_query_detects_context_only_question(monkeypatch):
    """会话历史问题应标记为仅依据上下文回答。"""
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _ContextLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="我第一次讲了啥",
                is_out_of_scope=False,
                answer_from_context=True,
            )

    runtime = SimpleNamespace(context=ContextSchema(llm=_ContextLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s1", "raw_query": "我第一次讲了啥", "context": "用户：我叫小明"},
        runtime,
    )

    assert out["answer_from_context"] is True
    assert out["is_out_of_scope"] is False


async def test_handle_query_failure_degrades_to_raw_query(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _BoomLLM:
        async def ainvoke_structured(self, messages, schema):
            raise RuntimeError("llm down")

    runtime = SimpleNamespace(context=ContextSchema(llm=_BoomLLM(), memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "q"
    assert out["is_out_of_scope"] is False
    assert out["sub_queries"] == []


async def test_handle_query_sub_queries_clamped_to_three(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="rw", is_out_of_scope=False,
                sub_queries=["s1", "s2", "s3", "s4"],
            )

    runtime = SimpleNamespace(context=ContextSchema(llm=_DecomposeLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    # 代码层 clamp 到 3,防 LLM 超量输出
    assert out["sub_queries"] == ["s1", "s2", "s3"]


async def test_handle_query_decompose_emits_status(monkeypatch):
    emitted: list[dict] = []
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="rw", is_out_of_scope=False, sub_queries=["s1", "s2"],
            )

    runtime = SimpleNamespace(context=ContextSchema(llm=_DecomposeLLM(), memory_manager=None))
    await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert {"type": "status", "data": "已拆解为 2 个子问题"} in emitted


async def test_handle_query_no_decompose_by_default(monkeypatch):
    """简单问题 LLM 不拆解(默认空数组),不发拆解状态事件。"""
    emitted: list[dict] = []
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))

    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert out["sub_queries"] == []
    assert not any("拆解" in ev.get("data", "") for ev in emitted if ev.get("type") == "status")


async def test_handle_query_failure_degrades_sub_queries_empty(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _BoomLLM:
        async def ainvoke_structured(self, messages, schema):
            raise RuntimeError("llm down")

    runtime = SimpleNamespace(context=ContextSchema(llm=_BoomLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert out["sub_queries"] == []


async def test_handle_query_streams_out_of_scope_answer(monkeypatch):
    """范围外查询在 handle_query 内直接流式作答,使用直接回答 prompt。"""
    from rag.agent.nodes.query.query import QueryRewriteOutput

    captured_messages = []
    monkeypatch.setattr(
        query_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _CaptureLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(rewrite_query="你好啊", is_out_of_scope=True)

        async def astream(self, messages):
            captured_messages.extend(messages)
            yield "直接回答"

    llm = _CaptureLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "你好啊", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["generated"] == "直接回答"
    # 确认使用了 direct_system_prompt 而非 kb_system_prompt
    system_msg = captured_messages[0]
    system_content = (
        system_msg.content if hasattr(system_msg, "content") else str(system_msg)
    )
    assert "知识库内容无关" in system_content or "直接基于你的知识" in system_content


async def test_context_answer_uses_conversation_context(monkeypatch):
    """上下文回答节点应把会话上下文传给 LLM。"""
    import rag.agent.nodes.generate.context_answer as context_answer_mod

    captured_messages = []
    monkeypatch.setattr(
        context_answer_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )

    class _CaptureLLM:
        async def astream(self, messages):
            captured_messages.extend(messages)
            yield "你第一次说的是：我叫小明。"

    runtime = SimpleNamespace(
        context=ContextSchema(llm=_CaptureLLM(), memory_manager=None)
    )
    out = await context_answer_mod.context_answer(
        {
            "session_id": "s1",
            "raw_query": "我第一次讲了啥",
            "context": "用户：我叫小明\nAI：你好，小明。",
        },
        runtime,
    )

    assert out["generated"] == "你第一次说的是：我叫小明。"
    system_content = captured_messages[0].content
    assert "我叫小明" in system_content


async def test_route_after_query_in_scope():
    """知识库范围内查询路由到 recall。"""
    import rag.agent.workflow as wf

    assert wf._route_after_query({"is_out_of_scope": False}) == "recall"
    assert wf._route_after_query({}) == "recall"  # 缺失时默认走检索


def test_route_after_query_context_only_skips_recall():
    import rag.agent.workflow as wf

    assert wf._route_after_query({"answer_from_context": True}) == "context_answer"


async def test_route_after_query_out_of_scope():
    """知识库范围外查询由 handle_query 直接作答,路由直达 END。"""
    import rag.agent.workflow as wf

    assert wf._route_after_query({"is_out_of_scope": True}) == "end"


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
    state = {"sub_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["sub_recall_results"] == [[{"text": "KB1"}]]
    assert retriever.calls[0][0] == "rw"


async def test_recall_no_retriever_yields_empty(monkeypatch):
    monkeypatch.setattr(
        kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )
    runtime = SimpleNamespace(
        context=ContextSchema(llm=None, memory_manager=None, retriever=None)
    )
    state = {"sub_query": "rw"}

    out = await kb_recall_mod.recall(state, runtime)

    assert out["sub_recall_results"] == [[]]


async def test_recall_send_payload_contract(monkeypatch):
    """recall 接收 Send 负载,返回 sub_recall_results 单元素列表交由 reducer 拼接。"""
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _R:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append(query)
            return [{"id": 1, "text": "KB1"}]

    r = _R()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=r))

    out = await kb_recall_mod.recall({"sub_query": "sq"}, runtime)

    assert r.calls == ["sq"]
    assert out == {"sub_recall_results": [[{"id": 1, "text": "KB1"}]]}


async def test_recall_branch_failure_degrades_empty(monkeypatch):
    """单分支 retriever 异常必须兜住:Send 分支抛异常会 fail 整个 run。"""
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _Boom:
        async def search(self, query, knowledge_base_ids=None, top_k=5):
            raise RuntimeError("pg down")

    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=_Boom()))
    out = await kb_recall_mod.recall({"sub_query": "sq"}, runtime)

    assert out == {"sub_recall_results": [[]]}


async def test_recall_no_retriever_degrades_empty(monkeypatch):
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=None))

    out = await kb_recall_mod.recall({"sub_query": "sq"}, runtime)

    assert out == {"sub_recall_results": [[]]}


async def test_recall_fuse_writes_recall_vec_results(monkeypatch):
    from rag.agent.nodes.recall_fuse import fuse as fuse_mod

    monkeypatch.setattr(fuse_mod, "get_settings", lambda: SimpleNamespace(RETRIEVER_RRF_K=60))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))
    state = {"sub_recall_results": [
        [{"id": 1, "text": "c1", "sources": ["vec"]}],
        [{"id": 1, "text": "c1", "sources": ["bm25"]}, {"id": 2, "text": "c2", "sources": ["vec"]}],
    ]}

    out = await fuse_mod.recall_fuse(state, runtime)

    ids = [r["id"] for r in out["recall_vec_results"]]
    assert ids == [1, 2]  # 双命中去重且排前
    assert out["recall_vec_results"][0]["sources"] == ["vec", "bm25"]


async def test_recall_fuse_empty_input(monkeypatch):
    from rag.agent.nodes.recall_fuse import fuse as fuse_mod

    monkeypatch.setattr(fuse_mod, "get_settings", lambda: SimpleNamespace(RETRIEVER_RRF_K=60))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))

    out = await fuse_mod.recall_fuse({"sub_recall_results": [[], []]}, runtime)

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
    assert topk_mod._dynamic_truncate([], 5, 0.7) == []


async def test_dynamic_truncate_single():
    chunk = {"rerank_score": 9.0, "text": "A"}
    result = topk_mod._dynamic_truncate([chunk], 5, 0.7)
    assert result == [chunk]


async def test_dynamic_truncate_no_gap_returns_default_topk():
    chunks = [
        {"rerank_score": 9.0}, {"rerank_score": 8.5},
        {"rerank_score": 8.0}, {"rerank_score": 7.5},
        {"rerank_score": 7.0}, {"rerank_score": 6.5},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


async def test_dynamic_truncate_gap_truncates():
    """相邻分数比值低于阈值时截断，含零分处理。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 8.5, "text": "B"},
        {"rerank_score": 1.0, "text": "C"},
        {"rerank_score": 0.0, "text": "D"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 2
    assert result[0]["text"] == "A"

    # 零分处无限大 gap → 截断
    chunks2 = [{"rerank_score": 9.0, "text": "A"}, {"rerank_score": 0.0, "text": "B"}]
    assert len(topk_mod._dynamic_truncate(chunks2, 5, 0.7)) == 1


async def test_dynamic_truncate_plateau_skips_gap():
    """gap 之后分数形成高原（内部比值均 >= threshold），跳过该 gap 继续扫描。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 8.5, "text": "B"},
        {"rerank_score": 1.0, "text": "C"},
        {"rerank_score": 0.98, "text": "D"},
        {"rerank_score": 0.96, "text": "E"},
        {"rerank_score": 0.95, "text": "F"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5
    assert [r["text"] for r in result] == ["A", "B", "C", "D", "E"]


async def test_dynamic_truncate_missing_rerank_score_hard_truncate():
    """无 rerank_score 时（rerank 未启用），硬截断 default_top_k。"""
    chunks = [{"text": c} for c in ["A", "B", "C", "D", "E", "F"]]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


# ── dynamic_topk node integration tests ──


async def test_dynamic_topk_enabled_filters(monkeypatch):
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": True, "RERANK_DYNAMIC_TOPK_DEFAULT": 5, "RERANK_DYNAMIC_TOPK_RATIO": 0.7}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0, "text": "A"}, {"rerank_score": 0.5, "text": "B"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}
    out = await topk_mod.dynamic_topk(state, runtime)
    assert len(out["recall_vec_results"]) == 1
    assert out["recall_vec_results"][0]["text"] == "A"


async def test_dynamic_topk_disabled_passthrough(monkeypatch):
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": False}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0}, {"rerank_score": 8.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}
    out = await topk_mod.dynamic_topk(state, runtime)
    assert out["recall_vec_results"] is chunks


async def test_dynamic_topk_empty_passthrough(monkeypatch):
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": True, "RERANK_DYNAMIC_TOPK_DEFAULT": 5, "RERANK_DYNAMIC_TOPK_RATIO": 0.7}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    out = await topk_mod.dynamic_topk({"recall_vec_results": [], "raw_query": "q"}, runtime)
    assert out["recall_vec_results"] == []


async def test_dynamic_topk_truncated_to_zero_emits_status(monkeypatch):
    """截断到 0 条时发送用户可见的状态事件。"""
    emitted = []
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": True, "RERANK_DYNAMIC_TOPK_DEFAULT": 5, "RERANK_DYNAMIC_TOPK_RATIO": 0.7}
    )())
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 0.0}, {"rerank_score": 9.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}
    out = await topk_mod.dynamic_topk(state, runtime)
    assert out["recall_vec_results"] == []
    assert stream_event(StreamEventType.STATUS, "未找到足够相关内容") in emitted


# ── expand node integration tests ──


class _FakeParentRetriever:
    """支持 fetch_parent_contents 的假检索器。"""

    def __init__(self, contents=None, exc=None):
        self.contents = contents or {}
        self.exc = exc
        self.calls = []

    async def fetch_parent_contents(self, document_ids):
        self.calls.append(sorted(document_ids))
        if self.exc is not None:
            raise self.exc
        return self.contents


# ── helpers ──

def _settings(**kw):
    return type("S", (), kw)()


def _runtime(retriever=None, pool=None):
    return SimpleNamespace(context=SimpleNamespace(retriever=retriever, pool=pool))


# ── 法规 KB：parent-child ──


async def test_expand_regulation_dedup_and_fetch(monkeypatch):
    """法规 KB chunk 去重 + 补查全文替换 text。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever({"doc-1": "全文A", "doc-2": "全文C"})
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-1", "rerank_score": 8.0, "text": "chunk B",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-2", "rerank_score": 7.0, "text": "chunk C",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 2
    assert retriever.calls == [["doc-1", "doc-2"]]
    doc1 = next(r for r in results if r["document_id"] == "doc-1")
    assert doc1["text"] == "全文A"
    assert doc1["rerank_score"] == 9.0
    doc2 = next(r for r in results if r["document_id"] == "doc-2")
    assert doc2["text"] == "全文C"


async def test_expand_regulation_disabled_passthrough(monkeypatch):
    """禁用时法规 chunk 不做任何修改。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=False, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever({"doc-1": "全文A"})
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-1", "text": "chunk B",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    assert out["recall_vec_results"] == chunks
    assert retriever.calls == []


async def test_expand_regulation_no_content_keeps_text(monkeypatch):
    """补查无全文时按文档去重但保留原始 text。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever({})
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-1", "rerank_score": 8.0, "text": "chunk B",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1
    assert results[0]["text"] == "chunk A"


async def test_expand_regulation_fetch_failure_keeps_text(monkeypatch):
    """补查失败时降级去重，不向上抛。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever(exc=RuntimeError("db down"))
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1
    assert results[0]["text"] == "chunk A"


async def test_expand_regulation_no_retriever_keeps_text(monkeypatch):
    """runtime 未注入 retriever 时降级为仅去重。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "chunk A",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 1
    assert results[0]["text"] == "chunk A"


async def test_expand_regulation_different_docs_kept(monkeypatch):
    """不同法规文档各自保留。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever(
        {"doc-1": "全文A", "doc-2": "全文B", "doc-3": "全文C"}
    )
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-2", "rerank_score": 8.0, "text": "B",
         "knowledge_base_id": REGULATION_KB_ID},
        {"document_id": "doc-3", "rerank_score": 7.0, "text": "C",
         "knowledge_base_id": REGULATION_KB_ID},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3
    assert {r["document_id"] for r in results} == {"doc-1", "doc-2", "doc-3"}
    assert [r["text"] for r in results] == ["全文A", "全文B", "全文C"]


# ── 普通 KB：sentence window ──


async def test_expand_general_adds_neighbors(monkeypatch):
    """普通 KB chunk 应拉取 ±N 邻居追加到结果中。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        SENTENCE_WINDOW_ENABLED=True, SENTENCE_WINDOW_SIZE=2,
        SENTENCE_WINDOW_MAX_MULTIPLIER=3, PARENT_CHILD_ENABLED=False,
    ))

    async def fake_neighbors(pool, doc_id, center, window):
        if doc_id == "doc-1" and center == 5:
            return [
                {"chunk_index": 3, "text": "neighbor before", "metadata": {}},
                {"chunk_index": 6, "text": "neighbor after", "metadata": {}},
            ]
        return []

    monkeypatch.setattr(expand_mod.store, "get_neighbor_chunks", fake_neighbors)

    pool = SimpleNamespace()
    runtime = _runtime(pool=pool)
    chunks = [
        {"document_id": "doc-1", "chunk_index": 5, "rerank_score": 0.9, "text": "center",
         "filename": "test.txt", "knowledge_base_id": "kb-1", "sources": ["vec"]},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3
    texts = {r["text"] for r in results}
    assert "center" in texts
    assert "neighbor before" in texts
    assert "neighbor after" in texts


async def test_expand_general_disabled_passthrough(monkeypatch):
    """禁用时普通 KB chunk 透传。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        SENTENCE_WINDOW_ENABLED=False, PARENT_CHILD_ENABLED=False,
    ))
    pool = SimpleNamespace()
    runtime = _runtime(pool=pool)
    chunks = [{"document_id": "doc-1", "chunk_index": 1, "text": "A",
               "knowledge_base_id": "kb-1"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    assert out["recall_vec_results"] == chunks


async def test_expand_general_no_pool_skips(monkeypatch):
    """pool 未注入时静默跳过。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        SENTENCE_WINDOW_ENABLED=True, PARENT_CHILD_ENABLED=False,
    ))
    runtime = _runtime(pool=None)
    chunks = [{"document_id": "doc-1", "chunk_index": 1, "text": "A",
               "knowledge_base_id": "kb-1"}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    assert out["recall_vec_results"] == chunks


async def test_expand_general_dedup(monkeypatch):
    """已存在的 chunk 不应被邻居重复添加。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        SENTENCE_WINDOW_ENABLED=True, SENTENCE_WINDOW_SIZE=2,
        SENTENCE_WINDOW_MAX_MULTIPLIER=3, PARENT_CHILD_ENABLED=False,
    ))

    async def fake_neighbors(pool, doc_id, center, window):
        return [
            {"chunk_index": 4, "text": "already there", "metadata": {}},
            {"chunk_index": 6, "text": "new neighbor", "metadata": {}},
        ]

    monkeypatch.setattr(expand_mod.store, "get_neighbor_chunks", fake_neighbors)

    pool = SimpleNamespace()
    runtime = _runtime(pool=pool)
    chunks = [
        {"document_id": "doc-1", "chunk_index": 4, "text": "already there",
         "filename": "t.txt", "sources": ["vec"], "knowledge_base_id": "kb-1"},
        {"document_id": "doc-1", "chunk_index": 5, "text": "center",
         "filename": "t.txt", "sources": ["vec"], "knowledge_base_id": "kb-1"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3


# ── 混合 KB ──


async def test_expand_mixed_kb(monkeypatch):
    """法规 + 普通混合结果：法规走 parent-child，普通走 sentence window。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True,
        SENTENCE_WINDOW_ENABLED=True, SENTENCE_WINDOW_SIZE=1,
        SENTENCE_WINDOW_MAX_MULTIPLIER=3,
    ))

    retriever = _FakeParentRetriever({"doc-reg": "法规全文"})
    pool = SimpleNamespace()

    async def fake_neighbors(pool, doc_id, center, window):
        if doc_id == "doc-gen" and center == 2:
            return [{"chunk_index": 1, "text": "neighbor gen", "metadata": {}}]
        return []

    monkeypatch.setattr(expand_mod.store, "get_neighbor_chunks", fake_neighbors)

    runtime = _runtime(retriever=retriever, pool=pool)
    chunks = [
        # 法规
        {"document_id": "doc-reg", "rerank_score": 9.0, "text": "reg chunk",
         "knowledge_base_id": REGULATION_KB_ID},
        # 普通
        {"document_id": "doc-gen", "chunk_index": 2, "rerank_score": 7.0,
         "text": "gen chunk", "knowledge_base_id": "kb-1", "filename": "f.txt",
         "sources": ["vec"]},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 3  # 1 法规全文 + 1 普通 chunk + 1 邻居
    assert retriever.calls == [["doc-reg"]]
    reg = next(r for r in results if r.get("knowledge_base_id") == REGULATION_KB_ID)
    assert reg["text"] == "法规全文"
    gen_texts = {r["text"] for r in results if r.get("knowledge_base_id") != REGULATION_KB_ID}
    assert "gen chunk" in gen_texts
    assert "neighbor gen" in gen_texts


async def test_expand_empty_chunks(monkeypatch):
    """空结果透传。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=True,
    ))
    runtime = _runtime()
    state: dict = {"recall_vec_results": [], "raw_query": "q"}
    out = await expand_mod.expand(state, runtime)
    assert out["recall_vec_results"] == []


async def test_expand_orphan_chunks_kept(monkeypatch):
    """无 document_id 的孤立 chunk 原样保留。"""
    monkeypatch.setattr(expand_mod, "get_settings", lambda: _settings(
        PARENT_CHILD_ENABLED=True, SENTENCE_WINDOW_ENABLED=False,
    ))
    retriever = _FakeParentRetriever({"doc-1": "全文A"})
    runtime = _runtime(retriever=retriever)
    chunks = [
        {"document_id": "doc-1", "rerank_score": 9.0, "text": "A",
         "knowledge_base_id": REGULATION_KB_ID},
        {"rerank_score": 5.0, "text": "orphan"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await expand_mod.expand(state, runtime)

    results = out["recall_vec_results"]
    assert len(results) == 2
    assert results[0]["text"] == "全文A"
    assert "orphan" in {r["text"] for r in results}


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

    async def lookup(self, query, *, session_id):
        self.lookup_calls.append((query, session_id))
        return self.hit

    async def store(self, query, answer, citations, *, session_id):
        self.store_calls.append((query, answer, citations, session_id))


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


async def test_cache_lookup_missing_session_is_fail_closed(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    cache = _FakeCache(hit={"answer": "不应返回", "citations": []})
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))

    for session_id in (None, ""):
        state = {"raw_query": "q", "session_id": session_id}
        out = await cache_lookup_mod.cache_lookup(state, runtime)
        assert cache.lookup_calls == []
        assert "cache_hit" not in out


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

    assert cache.store_calls == [("改写q", "新答案", [{"index": 2}], "s1")]


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


async def test_cache_store_missing_session_is_fail_closed(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=True)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))

    for session_id in (None, ""):
        state = {
            "session_id": session_id, "raw_query": "q", "generated": "答案",
        }
        await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == []
