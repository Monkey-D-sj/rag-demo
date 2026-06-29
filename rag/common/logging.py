import datetime
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rag.config import Settings, get_settings

_logging_initialized = False


def get_logger() -> logging.Logger:
    """获取调用者模块的 logger —— ``logging.getLogger(__name__)`` 的便利封装。

    用法（模块顶部一行）::

        from rag.common.logging import get_logger
        logger = get_logger()
    """
    frame = inspect.currentframe()
    try:
        caller_globals = frame.f_back.f_back.f_globals  # type: ignore[union-attr]
        module_name = caller_globals.get("__name__", "__unknown__")
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
_RESET = "\x1b[0m"

_TEXT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class ColorTextFormatter(logging.Formatter):
    """可读文本格式器；use_color=True 时给 level 上 ANSI 颜色。"""

    def __init__(self, use_color: bool = True):
        super().__init__(fmt=_TEXT_FMT, datefmt=_DATE_FMT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            if color:
                # 复制以免污染原始 record（影响其他 handler）
                record = logging.makeLogRecord(record.__dict__)
                record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


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
    level_name = settings.log_level.upper()
    invalid_level = level_name not in _VALID_LEVELS
    level = logging.INFO if invalid_level else getattr(logging, level_name)
    root.setLevel(level)

    # 控制台 handler（默认 stderr）
    stream_handler = logging.StreamHandler()
    use_color = bool(getattr(stream_handler.stream, "isatty", lambda: False)())
    stream_handler.setFormatter(
        _build_formatter(settings.log_format, use_color=use_color)
    )
    root.addHandler(stream_handler)

    # 可选文件 handler（始终非彩色）
    if settings.log_file:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(path),
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            _build_formatter(settings.log_format, use_color=False)
        )
        root.addHandler(file_handler)

    # 收编 uvicorn，统一走 root
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    if invalid_level:
        logging.getLogger(__name__).warning(
            "未知 log_level %r，已回退为 INFO", settings.log_level
        )
