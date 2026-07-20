import sys
import types

import pytest

import rag.observability.langfuse as ob


@pytest.mark.parametrize("enabled, has_keys", [
    ("false", True),
    ("true", False),
])
def test_noop_when_disabled_or_missing_keys(monkeypatch, enabled, has_keys):
    """可观测性禁用或缺 key 时，handler/observe/span 全部降级为 noop。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", enabled)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test" if has_keys else "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test" if has_keys else "")

    # handler → None
    assert ob.get_callback_handler() is None

    # observe → 原函数直通
    async def fn(x):
        return x + 1
    assert ob.observe_if_enabled("t")(fn) is fn

    # span_scope → yield None
    with ob.span_scope("bm25_recall") as span:
        assert span is None


def test_span_scope_degrades_to_nullcontext_on_sdk_failure(monkeypatch):
    """可观测性 SDK 加载失败只能丢 span，不能抛给业务调用方（曾致整个检索请求 500）。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            pass

    def _boom_get_client():
        raise AttributeError("no such api")

    fake_module = types.ModuleType("langfuse")
    fake_module.Langfuse = _FakeLangfuse
    fake_module.get_client = _boom_get_client
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    with ob.span_scope("vector_recall") as span:
        assert span is None


def test_get_callback_handler_inits_client_once(monkeypatch):
    """Langfuse 全局客户端只初始化一次，CallbackHandler 每次新建。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    client_calls: list[dict] = []

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    class _FakeCallbackHandler:
        pass

    fake_langfuse_module = types.ModuleType("langfuse")
    fake_langfuse_module.Langfuse = _FakeLangfuse
    fake_langchain_submodule = types.ModuleType("langfuse.langchain")
    fake_langchain_submodule.CallbackHandler = _FakeCallbackHandler
    fake_langfuse_module.langchain = fake_langchain_submodule

    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langchain_submodule)

    handler1 = ob.get_callback_handler()
    handler2 = ob.get_callback_handler()

    assert len(client_calls) == 1
    assert isinstance(handler1, _FakeCallbackHandler)
    assert isinstance(handler2, _FakeCallbackHandler)
    assert handler1 is not handler2
