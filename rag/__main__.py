"""``python -m rag`` 入口 —— 启动 FastAPI 服务（开发模式热重载）。"""

import os
import sys

import uvicorn


def _env_flag(name: str) -> bool:
    """读取布尔型环境变量,1/true/yes/on(忽略大小写)视为 True。"""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from rag.common.logging import _BANNER, get_logger, setup_logging
from rag.config import get_settings

logger = get_logger()


def main(argv: list[str] | None = None) -> None:
    """FastAPI 启动入口。

    用法:
        rag-api                          # 读取项目默认配置
        rag-api --host 0.0.0.0 --port 8080
        python -m rag                    # 等价于 rag-api
    """
    setup_logging()
    sys.stderr.write(_BANNER)
    sys.stderr.flush()
    settings = get_settings()

    host = "0.0.0.0"
    port = 8000

    # 允许命令行覆盖 host / port（轻量解析，不引入 argparse）
    args = argv or sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1

    level = settings.LOG_LEVEL.upper()
    # 开发用 RAG_RELOAD=1 开启热重载;生产保持关闭(默认),交由进程管理器/容器重启
    reload = _env_flag("RAG_RELOAD")
    logger.info("FastAPI 启动 → %s:%s (reload=%s)", host, port, reload)
    uvicorn.run(
        "rag.api.main:start_app",
        host=host,
        port=port,
        reload=reload,
        # Windows: uvicorn 默认在单进程下强制 ProactorEventLoop,而 psycopg
        # 异步模式只支持 SelectorEventLoop。loop="none" 让 uvicorn 不指定 loop
        # factory,改用上面 set_event_loop_policy 设定的 SelectorEventLoop。
        loop="none",
        log_level=level,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "colored": {
                    "()": "rag.common.logging.ColorTextFormatter",
                    "use_color": True,
                },
            },
            "filters": {
                "namer": {
                    "()": "rag.common.logging._NameRewriter",
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "colored",
                    "filters": ["namer"],
                },
            },
            "root": {
                "level": level,
                "handlers": ["default"],
            },
            "loggers": {
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
                "uvicorn.error": {"handlers": [], "propagate": True},
                "watchfiles.main": {"handlers": [], "propagate": True},
            },
        },
    )


if __name__ == "__main__":
    main()
