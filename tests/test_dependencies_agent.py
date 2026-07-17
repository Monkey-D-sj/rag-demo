from types import SimpleNamespace

from rag.api.dependencies.agent import get_memory_manager, get_llm


def test_get_semantic_cache_returns_state_attr():
    from rag.api.dependencies.agent import get_semantic_cache

    sentinel = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(semantic_cache=sentinel))
    )
    assert get_semantic_cache(request) is sentinel


def test_get_semantic_cache_missing_attr_returns_none():
    from rag.api.dependencies.agent import get_semantic_cache

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert get_semantic_cache(request) is None


def test_get_memory_manager_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(memory_manager=sentinel))
    )
    assert get_memory_manager(request) is sentinel


def test_get_llm_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(llm=sentinel)))
    assert get_llm(request) is sentinel


def test_graph_retriever_wiring_flags():
    """开关组合决定 KnowledgeRetriever 是否携带图召回。纯构造测试,不起 app。"""
    from unittest.mock import MagicMock

    from rag.document.retriever import KnowledgeRetriever
    from rag.graph.retriever import GraphRetriever
    from rag.config import Settings

    s = Settings()
    r_plain = KnowledgeRetriever(MagicMock(), MagicMock(), s)
    assert r_plain.has_graph is False

    gr = GraphRetriever(MagicMock(), "neo4j", MagicMock())
    r_graph = KnowledgeRetriever(MagicMock(), MagicMock(), s, graph_retriever=gr)
    assert r_graph.has_graph is True


def test_lifespan_wires_graph_retriever_conditionally():
    import inspect

    import rag.api.main as api_main

    src = inspect.getsource(api_main)
    assert "GRAPH_RECALL_ENABLED" in src
    assert "GraphRetriever(" in src
    # neo4j 初始化必须在 retriever 构造之前
    assert src.index("create_neo4j_driver") < src.index("KnowledgeRetriever(")
