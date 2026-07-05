import pytest

import rag.observability.langfuse as ob


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    ob.get_langfuse_settings.cache_clear()
    yield
    ob.get_langfuse_settings.cache_clear()


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
