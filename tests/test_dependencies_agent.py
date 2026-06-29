from types import SimpleNamespace

from rag.api.dependencies.agent import get_memory_manager, get_llm


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
