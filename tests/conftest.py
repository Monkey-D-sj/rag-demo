import asyncio
import sys

import pytest

from rag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


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


# psycopg3 (AsyncConnectionPool) 在 Windows 上不兼容默认的 ProactorEventLoop，
# 需要切换为 SelectorEventLoop。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
