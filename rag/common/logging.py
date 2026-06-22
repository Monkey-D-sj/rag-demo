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
