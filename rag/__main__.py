"""``python -m rag`` 入口 —— 启动 FastAPI 服务（开发模式热重载）。"""

import sys

import uvicorn

from rag.common.logging import get_logger, setup_logging
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

    logger.info("FastAPI 启动 → %s:%s", host, port)
    uvicorn.run(
        "rag.api.main:start_app",
        host=host,
        port=port,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
