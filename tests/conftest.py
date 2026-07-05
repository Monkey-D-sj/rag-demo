import asyncio
import sys

import pytest

from rag.config import Settings
from rag.common.logging import setup_logging


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session", autouse=True)
def _setup_logging() -> None:
    """初始化日志系统,确保测试可以捕获日志。"""
    setup_logging()


# psycopg3 (AsyncConnectionPool) 在 Windows 上不兼容默认的 ProactorEventLoop，
# 需要切换为 SelectorEventLoop。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
