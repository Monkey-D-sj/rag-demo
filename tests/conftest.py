import asyncio
import sys

import pytest

from rag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


# psycopg3 (AsyncConnectionPool) 在 Windows 上不兼容默认的 ProactorEventLoop，
# 需要切换为 SelectorEventLoop。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
