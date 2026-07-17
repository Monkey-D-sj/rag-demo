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
