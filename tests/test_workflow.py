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
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=True)
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
