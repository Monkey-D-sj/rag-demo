import datetime
import inspect
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag.config import Settings, get_settings

_logging_initialized = False

# logging 模块自身不能走 get_logger()（栈帧会追溯到调用者），用标准写法
_logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """获取调用者模块的 logger —— ``logging.getLogger(__name__)`` 的便利封装。

    用法（模块顶部一行）::

        from rag.common.logging import get_logger
        logger = get_logger()
    """
    frame = inspect.currentframe()
    try:
        # 模块顶层调用时栈深度不足（只有 2 层），回退到 f_back
        f = frame.f_back.f_back or frame.f_back  # type: ignore[union-attr]
        module_name = f.f_globals.get("__name__", "__unknown__")
    finally:
        del frame
    return logging.getLogger(module_name)


class JsonFormatter(logging.Formatter):
    """每行一个 JSON 的日志格式器（纯标准库）。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created).isoformat()
        data: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",     # cyan
    "INFO": "\x1b[32m",      # green
    "WARNING": "\x1b[33m",   # yellow
    "ERROR": "\x1b[31m",     # red
    "CRITICAL": "\x1b[1;31m",
}
_TIME_COLOR = "\x1b[90m"     # dim gray
_RESET = "\x1b[0m"

_BANNER = """\x1b[1;36m
   ╔══════════════════════════════════════════╗
   ║                                          ║
   ║\x1b[0m\x1b[1;36m     ██████╗  \x1b[1;35m █████╗  \x1b[1;36m ██████╗  \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;36m     ██╔══██╗\x1b[1;35m ██╔══██╗\x1b[1;36m ██╔════╝  \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;36m     ██████╔╝\x1b[1;35m ███████║\x1b[1;36m ██║  ███╗ \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;36m     ██╔══██╗\x1b[1;35m ██╔══██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;36m     ██║  ██║\x1b[1;35m ██║  ██║\x1b[1;36m ╚██████╔╝ \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;36m     ╚═╝  ╚═╝\x1b[1;35m ╚═╝  ╚═╝\x1b[1;36m  ╚═════╝  \x1b[0m\x1b[1;36m     ║
   ║                                          ║
   ║\x1b[0m\x1b[1;35m     ██████╗  \x1b[1;36m ███████╗\x1b[1;35m ███╗   ███╗\x1b[1;36m  ██████╗  \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;35m     ██╔══██╗\x1b[1;36m ██╔════╝\x1b[1;35m ████╗ ████║\x1b[1;36m ██╔═══██╗ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;35m     ██║  ██║\x1b[1;36m █████╗  \x1b[1;35m ██╔████╔██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;35m     ██║  ██║\x1b[1;36m ██╔══╝  \x1b[1;35m ██║╚██╔╝██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;35m     ██████╔╝\x1b[1;36m ███████╗\x1b[1;35m ██║ ╚═╝ ██║\x1b[1;36m ╚██████╔╝ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║\x1b[0m\x1b[1;35m     ╚═════╝ \x1b[1;36m ╚══════╝\x1b[1;35m ╚═╝     ╚═╝\x1b[1;36m  ╚═════╝  \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m     ║
   ║                                          ║
   ║  \x1b[1;33m⚡\x1b[1;37m RAG 知识库 \x1b[0;90m│\x1b[1;33m 本地开发 \x1b[0;90m│\x1b[1;32m 就绪 \x1b[1;33m⚡\x1b[0m\x1b[1;36m    ║
   ║                                          ║
   ╚══════════════════════════════════════════╝\x1b[0m
"""

_TEXT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class ColorTextFormatter(logging.Formatter):
    """可读文本格式器；use_color=True 时给时间/level 上 ANSI 颜色。"""

    def __init__(self, use_color: bool = True):
        super().__init__(fmt=_TEXT_FMT, datefmt=_DATE_FMT)
        self._use_color = use_color

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        s = super().formatTime(record, datefmt)
        if self._use_color:
            s = f"{_TIME_COLOR}{s}{_RESET}"
        return s

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            if color:
                # 复制以免污染原始 record（影响其他 handler）
                record = logging.makeLogRecord(record.__dict__)
                visible = record.levelname
                pad = max(0, 8 - len(visible))
                record.levelname = f"{color}{visible}{_RESET}{' ' * pad}"
        return super().format(record)


_LOGGER_ALIASES = {
    "uvicorn.error": "uvicorn",
}
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class _NameRewriter(logging.Filter):
    """重命名形如 ``uvicorn.error`` 的 logger 名称。"""

    def filter(self, record: logging.LogRecord) -> bool:
        alias = _LOGGER_ALIASES.get(record.name)
        if alias:
            record.name = alias
        return True


def _build_formatter(log_format: str, *, use_color: bool) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter()
    return ColorTextFormatter(use_color=use_color)


def setup_logging(settings: Settings | None = None) -> None:
    """幂等配置 root logger。重复调用安全，只有首次生效。"""
    global _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    settings = settings or get_settings()

    root = logging.getLogger()
    root.handlers.clear()

    # 解析 level（非法则回退 INFO）
    level_name = settings.LOG_LEVEL.upper()
    invalid_level = level_name not in _VALID_LEVELS
    level = logging.INFO if invalid_level else getattr(logging, level_name)
    root.setLevel(level)

    # 控制台 handler（默认 stderr）
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(_NameRewriter())
    use_color = bool(getattr(stream_handler.stream, "isatty", lambda: False)())
    stream_handler.setFormatter(
        _build_formatter(settings.LOG_FORMAT, use_color=use_color)
    )
    root.addHandler(stream_handler)

    # 可选文件 handler（始终非彩色）
    if settings.LOG_FILE:
        path = Path(settings.LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            _build_formatter(settings.LOG_FORMAT, use_color=False)
        )
        root.addHandler(file_handler)

    # 收编 uvicorn / watchfiles，统一走 root
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "watchfiles.main"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    if invalid_level:
        _logger.warning(
            "未知 LOG_LEVEL %r，已回退为 INFO", settings.LOG_LEVEL
        )
