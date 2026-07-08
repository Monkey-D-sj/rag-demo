"""Windows 兼容性：asyncio 事件循环策略统一设置。

psycopg3 async 依赖 SelectorEventLoop，但 Windows 默认使用 ProactorEventLoop，
因此需要在所有入口点（API / Worker / Eval / 测试）调用此函数。
"""

import sys


def setup_windows_loop() -> None:
    """Windows 下将事件循环策略设为 WindowsSelectorEventLoopPolicy。"""
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
