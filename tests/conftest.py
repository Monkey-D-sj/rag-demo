import pytest

from rag.common.platform import setup_windows_loop

setup_windows_loop()


@pytest.fixture(autouse=True)
def _langfuse_disabled(monkeypatch):
    """测试密闭性：.env 里 LANGFUSE_ENABLED 可能为 true，单测一律强制关闭。"""
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    import rag.observability.langfuse as ob

    ob.get_langfuse_settings.cache_clear()
    ob._init_client.cache_clear()
    yield
    ob.get_langfuse_settings.cache_clear()
    ob._init_client.cache_clear()
