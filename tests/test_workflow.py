from types import SimpleNamespace

import rag.agent.workflow as wf
from rag.agent.type import ContextSchema


class _FakeMM:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S1"}]

    async def persist_turn(self, session_id, query, answer):
        pass


class _FakeLLM:
    async def ainvoke_structured(self, messages, schema):
        from rag.agent.nodes.query.query import QueryRewriteOutput

        return QueryRewriteOutput(rewrite_query="rw", is_out_of_scope=False)

    async def astream(self, messages):
        for tok in ("你好", "世界"):
            yield tok


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    async def search(self, query, knowledge_base_ids=None, top_k=5):
        self.calls.append((query, knowledge_base_ids))
        return [{"text": "KB1"}]


async def test_invoke_runs_full_graph_with_context():
    retriever = _FakeRetriever()
    ctx = ContextSchema(
        llm=_FakeLLM(), memory_manager=_FakeMM(), retriever=retriever
    )

    statuses: list[str] = []
    messages: list[str] = []
    async for event in wf.invoke("s1", "q1", ctx):
        if event["type"] == "status":
            statuses.append(event["data"])
        elif event["type"] == "message":
            messages.append(event["data"])

    # recall 节点用改写后的查询检索知识库
    assert retriever.calls == [("rw", None)]

    # 节点经由 custom 通道发出的进度事件
    assert "检索记忆中..." in statuses
    assert "检索知识库中..." in statuses
    assert "生成回答中" in statuses

    # 最终生成逐 token 流式
    assert messages == ["你好", "世界"]


async def test_invoke_passes_config_to_astream(monkeypatch):
    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover - 使函数成为异步生成器
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None, config={"callbacks": []}):
        pass
    assert captured["config"] == {"callbacks": []}


async def test_invoke_out_of_scope_skips_recall(monkeypatch):
    """is_out_of_scope=True 时 graph 应跳过 recall,由 handle_query 直接作答。"""
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _OutOfScopeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(rewrite_query="你好", is_out_of_scope=True)

        async def astream(self, messages):
            yield "你好呀！"

    class _NoCallRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append(query)
            return [{"text": "SHOULD_NOT_APPEAR"}]

    retriever = _NoCallRetriever()
    ctx = ContextSchema(
        llm=_OutOfScopeLLM(), memory_manager=_FakeMM(), retriever=retriever
    )

    messages: list[str] = []
    statuses: list[str] = []
    async for event in wf.invoke("s1", "你好", ctx):
        if event["type"] == "message":
            messages.append(event["data"])
        elif event["type"] == "status":
            statuses.append(event["data"])

    # recall 完全未被调用
    assert retriever.calls == []
    # 没有"检索知识库中..."的状态
    assert "检索知识库中..." not in statuses
    # handle_query 内直接作答拿到了回复
    assert messages == ["你好呀！"]


async def test_invoke_context_question_skips_kb_recall():
    """会话历史问题只使用上下文回答，知识库检索器不应被调用。"""
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _ContextLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="我第一次讲了啥",
                is_out_of_scope=False,
                answer_from_context=True,
            )

        async def astream(self, messages):
            yield "你第一次说的是：我叫小明。"

    class _NoCallRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append(query)
            return [{"text": "SHOULD_NOT_APPEAR"}]

    class _ContextMemory(_FakeMM):
        async def get_recent_messages(self, session_id, n=10):
            return [{"text": "我叫小明", "metadata": {"role": "user"}}]

    retriever = _NoCallRetriever()
    ctx = ContextSchema(
        llm=_ContextLLM(), memory_manager=_ContextMemory(), retriever=retriever
    )

    messages: list[str] = []
    statuses: list[str] = []
    async for event in wf.invoke("s1", "我第一次讲了啥", ctx):
        if event["type"] == "message":
            messages.append(event["data"])
        elif event["type"] == "status":
            statuses.append(event["data"])

    assert retriever.calls == []
    assert "检索知识库中..." not in statuses
    assert "根据会话上下文回答中" in statuses
    assert messages == ["你第一次说的是：我叫小明。"]


async def test_invoke_decomposition_fans_out(monkeypatch):
    """开关开启时:主查询+2 子查询共 3 个 Send 分支并行检索,融合后走完生成链路。"""
    import rag.agent.workflow as wf_mod
    from rag.agent.nodes.query.query import QueryRewriteOutput

    monkeypatch.setattr(
        wf_mod, "get_settings",
        lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=True, AGENT_MODE_ENABLED=False),
    )

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="孙悟空和猪八戒的兵器",
                is_out_of_scope=False,
                sub_queries=["孙悟空的兵器", "猪八戒的兵器"],
            )

        async def astream(self, messages):
            yield "答"

    class _PerQueryRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append(query)
            return [{"id": f"c-{query}", "text": f"KB-{query}", "sources": ["vec"]}]

    retriever = _PerQueryRetriever()
    ctx = ContextSchema(llm=_DecomposeLLM(), memory_manager=_FakeMM(), retriever=retriever)

    messages: list[str] = []
    async for event in wf.invoke("s1", "q", ctx):
        if event["type"] == "message":
            messages.append(event["data"])

    # 三个分支各检索一次(Send 并行,顺序不保证,用 set 比较)
    assert set(retriever.calls) == {"孙悟空和猪八戒的兵器", "孙悟空的兵器", "猪八戒的兵器"}
    assert len(retriever.calls) == 3
    # 融合结果非空,正常走到 generate
    assert messages == ["答"]


async def test_invoke_config_defaults_none(monkeypatch):
    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None):
        pass
    assert captured["config"] is None


def test_route_after_cache_hit_goes_to_add_memory():
    from rag.agent.workflow import _route_after_cache

    assert _route_after_cache({"cache_hit": True}) == "add_memory"


def test_route_after_cache_miss_single_branch_when_disabled(monkeypatch):
    """开关默认关闭:仅主查询单分支,sub_queries 被忽略。"""
    import rag.agent.workflow as wf_mod
    from langgraph.types import Send

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=False)
    )
    out = wf_mod._route_after_cache(
        {"rewrite_query": "rw", "sub_queries": ["s1"]}
    )

    assert out == [Send("recall", {"sub_query": "rw"})]


def test_route_after_cache_fans_out_when_enabled(monkeypatch):
    """开关开启:按 [主查询]+子查询 扇出并行召回。"""
    import rag.agent.workflow as wf_mod
    from langgraph.types import Send

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=True)
    )
    out = wf_mod._route_after_cache(
        {"rewrite_query": "rw", "sub_queries": ["s1", "s2"]}
    )

    assert out == [
        Send("recall", {"sub_query": "rw"}),
        Send("recall", {"sub_query": "s1"}),
        Send("recall", {"sub_query": "s2"}),
    ]


def test_route_after_cache_falls_back_to_raw_query(monkeypatch):
    import rag.agent.workflow as wf_mod

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=False)
    )
    out = wf_mod._route_after_cache({"raw_query": "raw"})

    assert out[0].arg == {"sub_query": "raw"}


def test_build_initial_state_defaults_agent_fields():
    from rag.agent.workflow import build_initial_state

    s = build_initial_state("sid", "q")
    assert s["needs_agent"] is False
    assert s["agent_succeeded"] is False
    assert s["agent_skip_cache"] is False


# ── agent 分支路由 ──────────────────────────────────────────

def test_route_after_query_agent_enabled(monkeypatch):
    import rag.agent.workflow as wf_mod

    monkeypatch.setattr(wf_mod, "get_settings", lambda: SimpleNamespace(AGENT_MODE_ENABLED=True))
    assert wf_mod._route_after_query({"needs_agent": True}) == "agent"


def test_route_after_query_agent_disabled_falls_to_recall(monkeypatch):
    import rag.agent.workflow as wf_mod

    monkeypatch.setattr(wf_mod, "get_settings", lambda: SimpleNamespace(AGENT_MODE_ENABLED=False))
    assert wf_mod._route_after_query({"needs_agent": True}) == "recall"


def test_route_after_query_context_and_scope_take_precedence(monkeypatch):
    import rag.agent.workflow as wf_mod

    monkeypatch.setattr(wf_mod, "get_settings", lambda: SimpleNamespace(AGENT_MODE_ENABLED=True))
    assert wf_mod._route_after_query({"answer_from_context": True, "needs_agent": True}) == "context_answer"
    assert wf_mod._route_after_query({"is_out_of_scope": True, "needs_agent": True}) == "end"


def test_route_after_agent_routes_to_generate_or_cache_lookup():
    from rag.agent.workflow import _route_after_agent

    assert _route_after_agent({
        "agent_succeeded": True, "recall_vec_results": [{"text": "x"}],
    }) == "generate"
    assert _route_after_agent({
        "agent_succeeded": False, "recall_vec_results": [{"text": "partial"}],
    }) == "cache_lookup"
    assert _route_after_agent({"recall_vec_results": []}) == "cache_lookup"
    assert _route_after_agent({}) == "cache_lookup"


def test_graph_contains_agent_execute():
    from rag.agent.workflow import graph

    assert "agent_execute" in set(graph.get_graph().nodes)


async def test_invoke_agent_path_degrades_to_mainline_when_no_tools(monkeypatch):
    """agent 开关开启且 needs_agent 时进入 agent_execute;空结果降级回主链完成生成。"""
    import rag.agent.workflow as wf_mod
    from langchain_core.messages import AIMessage
    from rag.agent.nodes.query.query import QueryRewriteOutput

    monkeypatch.setattr(
        wf_mod, "get_settings",
        lambda: SimpleNamespace(AGENT_MODE_ENABLED=True, QUERY_DECOMPOSITION_ENABLED=False),
    )

    class _AgentLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(rewrite_query="rw", is_out_of_scope=False, needs_agent=True)

        async def ainvoke_with_tools(self, messages, tools):
            return AIMessage(content="证据已齐备")  # 不调工具 → 空结果 → 降级主链

        async def astream(self, messages):
            for tok in ("甲", "乙"):
                yield tok

    retriever = _FakeRetriever()
    ctx = ContextSchema(llm=_AgentLLM(), memory_manager=_FakeMM(), retriever=retriever)

    statuses: list[str] = []
    messages: list[str] = []
    async for event in wf_mod.invoke("s1", "q", ctx):
        if event["type"] == "status":
            statuses.append(event["data"])
        elif event["type"] == "message":
            messages.append(event["data"])

    assert "深度检索中…" in statuses  # agent 分支确实进入
    assert messages == ["甲", "乙"]  # 降级主链正常生成


async def test_invoke_agent_two_hop_path_collects_both_sources(monkeypatch):
    """确定性验证完整图的两跳工具链，不能靠主链降级蒙混通过。"""
    import rag.agent.nodes.agent.agent as agent_mod
    import rag.agent.workflow as wf_mod
    from langchain_core.messages import AIMessage, ToolMessage
    from rag.agent.nodes.query.query import QueryRewriteOutput

    settings = SimpleNamespace(
        AGENT_MODE_ENABLED=True,
        QUERY_DECOMPOSITION_ENABLED=False,
        AGENT_MAX_STEPS=3,
        AGENT_MAX_TOOL_CALLS_PER_STEP=4,
        AGENT_MAX_EVIDENCE_CHARS=24_000,
        AGENT_TOTAL_TIMEOUT_SECONDS=30,
    )
    monkeypatch.setattr(wf_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(agent_mod, "get_settings", lambda: settings)

    class _TwoHopLLM:
        def __init__(self):
            self.tool_rounds = 0
            self.tool_messages = []

        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="A办法引用的B条例实际门槛",
                is_out_of_scope=False,
                needs_agent=True,
            )

        async def ainvoke_with_tools(self, messages, tools):
            self.tool_messages.append(list(messages))
            self.tool_rounds += 1
            if self.tool_rounds == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "retrieve_kb", "args": {"query": "A办法资质要求"},
                    "id": "c1", "type": "tool_call",
                }])
            if self.tool_rounds == 2:
                assert any(
                    isinstance(m, ToolMessage) and "B条例" in m.content
                    for m in messages
                )
                return AIMessage(content="", tool_calls=[{
                    "name": "retrieve_kb", "args": {"query": "B条例注册资本门槛"},
                    "id": "c2", "type": "tool_call",
                }])
            return AIMessage(content="", tool_calls=[{
                "name": "finish_evidence_collection", "args": {},
                "id": "finish", "type": "tool_call",
            }])

        async def astream(self, messages):
            yield "最终答案"

    class _TwoHopRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5):
            self.calls.append(query)
            if query == "A办法资质要求":
                return [{
                    "text": "A办法规定资质要求参照B条例执行。", "filename": "A办法",
                    "document_id": "A", "chunk_index": 1, "metadata": {},
                }]
            if query == "B条例注册资本门槛":
                return [{
                    "text": "B条例规定注册资本不得低于500万元。", "filename": "B条例",
                    "document_id": "B", "chunk_index": 1, "metadata": {},
                }]
            raise AssertionError(f"意外回退主链查询: {query}")

        async def fetch_parent_contents(self, document_ids):
            return {}

    llm = _TwoHopLLM()
    retriever = _TwoHopRetriever()
    ctx = ContextSchema(llm=llm, memory_manager=_FakeMM(), retriever=retriever)
    messages, citations = [], []
    async for event in wf_mod.invoke("s1", "那实际门槛呢", ctx):
        if event["type"] == "message":
            messages.append(event["data"])
        elif event["type"] == "citations":
            citations.extend(event["data"])

    assert retriever.calls == ["A办法资质要求", "B条例注册资本门槛"]
    assert llm.tool_rounds == 3
    assert messages == ["最终答案"]
    assert [c["document_title"] for c in citations] == ["A办法", "B条例"]


async def test_cache_store_skips_agent_answers(monkeypatch):
    import rag.agent.nodes.cache_store.store as cs_mod

    class _Cache:
        def __init__(self):
            self.stored = []

        async def store(self, query, answer, citations, *, session_id):
            self.stored.append((query, answer))

    cache = _Cache()
    monkeypatch.setattr(cs_mod, "get_settings", lambda: SimpleNamespace(SEMANTIC_CACHE_ENABLED=True))
    ctx = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s", "generated": "x", "rewrite_query": "q", "citations": [], "agent_skip_cache": True}
    await cs_mod.cache_store(state, ctx)
    assert cache.stored == []  # agent 答案不写缓存

    state.pop("agent_skip_cache")
    await cs_mod.cache_store(state, ctx)
    assert cache.stored == [("q", "x")]


def test_graph_contains_recall_fuse():
    from rag.agent.workflow import graph

    assert "recall_fuse" in set(graph.get_graph().nodes)


def test_graph_edges_recall_via_fuse():
    from rag.agent.workflow import graph

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("recall", "recall_fuse") in edges
    assert ("recall_fuse", "rerank") in edges
    assert ("rerank", "dynamic_topk") in edges
    assert ("dynamic_topk", "expand") in edges


def test_graph_contains_cache_nodes():
    from rag.agent.workflow import graph

    nodes = set(graph.get_graph().nodes)
    assert "cache_lookup" in nodes
    assert "cache_store" in nodes


def test_graph_edges_generate_via_cache_store():
    from rag.agent.workflow import graph

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("generate", "cache_store") in edges
    assert ("cache_store", "add_memory") in edges
    # 旧的 generate -> add_memory 直连必须移除
    assert ("generate", "add_memory") not in edges
