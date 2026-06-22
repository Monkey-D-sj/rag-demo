import datetime
import json
import logging


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
