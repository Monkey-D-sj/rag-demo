import sys
import types

import pytest

import rag.observability.langfuse as ob


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    ob.get_langfuse_settings.cache_clear()
    ob._init_client.cache_clear()
    yield
    ob.get_langfuse_settings.cache_clear()
    ob._init_client.cache_clear()


def test_handler_none_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert ob.get_callback_handler() is None


def test_handler_none_when_enabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    assert ob.get_callback_handler() is None


def test_observe_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    async def fn(x):
        return x + 1

    assert ob.observe_if_enabled("t")(fn) is fn


def test_observe_passthrough_when_enabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    ob.get_langfuse_settings.cache_clear()

    async def fn(x):
        return x

    assert ob.observe_if_enabled("t")(fn) is fn


def test_get_callback_handler_inits_client_once(monkeypatch):
    """连续两次 get_callback_handler() 应只构造一次全局 Langfuse 客户端,
    CallbackHandler 则每次都新建(不是同一个对象)。"""
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

    assert len(client_calls) == 1  # 客户端只初始化一次
    assert isinstance(handler1, _FakeCallbackHandler)
    assert isinstance(handler2, _FakeCallbackHandler)
    assert handler1 is not handler2  # handler 每次请求新建
