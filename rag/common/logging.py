import atexit
import contextvars
import datetime
import inspect
import json
import logging
import queue
import threading
import urllib.request
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
        # f_back 即调用方帧(模块顶层或函数内均可),其 __name__ 就是调用方模块名。
        # 注意不能再往上走一层:模块顶层调用时 f_back.f_back 是 importlib 的帧。
        f = frame.f_back  # type: ignore[union-attr]
        module_name = f.f_globals.get("__name__", "__unknown__")
    finally:
        del frame
    return logging.getLogger(module_name)


# ── session 关联:contextvars 贯穿单次请求内的所有日志 ──

_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "log_session_id", default=None
)


def bind_session(session_id: str) -> contextvars.Token:
    """绑定当前上下文的 session_id,返回 token 供 reset_session 恢复。"""
    return _session_id_var.set(session_id)


def reset_session(token: contextvars.Token) -> None:
    _session_id_var.reset(token)


def current_session_id() -> str | None:
    """读取当前异步上下文绑定的 session_id(无则 None)。供治理层统计记录使用。"""
    return _session_id_var.get()


class _SessionContextFilter(logging.Filter):
    """把 contextvar 中的 session_id 注入日志记录;显式 extra 优先。"""

    def filter(self, record: logging.LogRecord) -> bool:
        sid = _session_id_var.get()
        if sid and not hasattr(record, "session_id"):
            record.session_id = sid
        return True


# LogRecord 标准属性集合;record.__dict__ 中此外的键视为 extra 结构化字段。
# taskName 是 3.12 asyncio 加的,message/asctime 由 Formatter 动态注入。
_STD_RECORD_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """每行一个 JSON 的日志格式器（纯标准库）;extra 字段自动并入输出。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created).isoformat()
        data: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_KEYS and key not in data:
                data[key] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        # default=str 兜底不可序列化值(datetime/UUID 等),日志不因序列化炸掉
        return json.dumps(data, ensure_ascii=False, default=str)


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
   ╔════════════════════════════════════════════════╗
   ║                                                ║
   ║\x1b[0m\x1b[1;36m     ██████╗  \x1b[1;35m █████╗  \x1b[1;36m ██████╗  \x1b[0m\x1b[1;36m               ║
   ║\x1b[0m\x1b[1;36m     ██╔══██╗\x1b[1;35m ██╔══██╗\x1b[1;36m ██╔════╝  \x1b[0m\x1b[1;36m               ║
   ║\x1b[0m\x1b[1;36m     ██████╔╝\x1b[1;35m ███████║\x1b[1;36m ██║  ███╗ \x1b[0m\x1b[1;36m               ║
   ║\x1b[0m\x1b[1;36m     ██╔══██╗\x1b[1;35m ██╔══██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;36m               ║
   ║\x1b[0m\x1b[1;36m     ██║  ██║\x1b[1;35m ██║  ██║\x1b[1;36m ╚██████╔╝ \x1b[0m\x1b[1;36m               ║
   ║\x1b[0m\x1b[1;36m     ╚═╝  ╚═╝\x1b[1;35m ╚═╝  ╚═╝\x1b[1;36m  ╚═════╝  \x1b[0m\x1b[1;36m               ║
   ║                                                ║
   ║\x1b[0m\x1b[1;35m     ██████╗  \x1b[1;36m ███████╗\x1b[1;35m ███╗   ███╗\x1b[1;36m  ██████╗  \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m ║
   ║\x1b[0m\x1b[1;35m     ██╔══██╗\x1b[1;36m ██╔════╝\x1b[1;35m ████╗ ████║\x1b[1;36m ██╔═══██╗ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m  ║
   ║\x1b[0m\x1b[1;35m     ██║  ██║\x1b[1;36m █████╗  \x1b[1;35m ██╔████╔██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m  ║
   ║\x1b[0m\x1b[1;35m     ██║  ██║\x1b[1;36m ██╔══╝  \x1b[1;35m ██║╚██╔╝██║\x1b[1;36m ██║   ██║ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m  ║
   ║\x1b[0m\x1b[1;35m     ██████╔╝\x1b[1;36m ███████╗\x1b[1;35m ██║ ╚═╝ ██║\x1b[1;36m ╚██████╔╝ \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m  ║
   ║\x1b[0m\x1b[1;35m     ╚═════╝ \x1b[1;36m ╚══════╝\x1b[1;35m ╚═╝     ╚═╝\x1b[1;36m  ╚═════╝  \x1b[0m\x1b[1;35m \x1b[0m\x1b[1;36m  ║
   ║                                                ║
   ║\x1b[1;33m⚡\x1b[1;37m RAG 知识库 \x1b[0;90m│\x1b[1;33m 本地开发 \x1b[0;90m│\x1b[1;32m 就绪 \x1b[1;33m⚡\x1b[0m\x1b[1;36m                ║
   ║                                                ║
   ╚════════════════════════════════════════════════╝\x1b[0m
"""

_TEXT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# 文本模式下 extra 值最大字符数,超长截断(避免 vec_hits 等列表炸终端)
_MAX_EXTRA_LEN = 120


def _fmt_extra(value: object) -> str:
    """text 模式格式化 extra 值:列表/字典显示摘要,标量原样,超长截断。"""
    if isinstance(value, list):
        return f"[{len(value)}条]"
    if isinstance(value, dict):
        return f"{{{len(value)}键}}"
    raw = str(value)
    if len(raw) > _MAX_EXTRA_LEN:
        raw = raw[:_MAX_EXTRA_LEN] + "…"
    return raw
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
        line = super().format(record)
        # extra 字段以紧凑 key=value 形式追加(排除标准属性和内部键)
        extras: list[str] = []
        for key in sorted(record.__dict__):
            if key in _STD_RECORD_KEYS or key.startswith("_"):
                continue
            extras.append(f"{key}={_fmt_extra(getattr(record, key, None))}")
        if extras:
            line = f"{line} | {' '.join(extras)}"
        return line


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


_LOKI_BATCH_SIZE = 100
_LOKI_FLUSH_INTERVAL = 2.0  # 秒
_LOKI_TIMEOUT = 3.0


class _LokiHandler(logging.Handler):
    """后台线程批量推送 JSON 日志到 Grafana Loki。

    每条日志以 ``JsonFormatter`` 序列化,按 ``_LOKI_BATCH_SIZE`` 攒批,
    ``_LOKI_FLUSH_INTERVAL`` 定时刷盘,避免同步 HTTP 阻塞调用线程。
    Loki 不可达时静默丢日志(调高自身 logger 级别可查看丢弃计数)。
    """

    def __init__(self, url: str, labels: dict[str, str]):
        super().__init__()
        self._url = url.rstrip("/") + "/loki/api/v1/push"
        self._labels = labels
        self._queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def emit(self, record: logging.LogRecord) -> None:
        if not self._running:
            return
        try:
            ts_ns = str(int(record.created * 1e9))
            line = self.format(record)
            self._queue.put_nowait((record.levelno, f"{ts_ns} {line}"))
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._running = False
        self._queue.put_nowait(None)  # 通知后台线程退出
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        super().close()

    def _run(self) -> None:
        """后台线程:攒批 → 推送。"""
        batch: list[tuple[int, str]] = []
        last_flush = datetime.datetime.now()

        def _flush() -> None:
            nonlocal batch, last_flush
            if not batch:
                return
            payload = _build_loki_payload(batch, self._labels)
            batch.clear()
            last_flush = datetime.datetime.now()
            try:
                req = urllib.request.Request(
                    self._url,
                    data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=_LOKI_TIMEOUT)
            except Exception:
                pass  # Loki 不可达,静默丢弃(避免日志写日志的死循环)

        while self._running:
            try:
                item = self._queue.get(timeout=_LOKI_FLUSH_INTERVAL)
                if item is None:
                    _flush()
                    return
                batch.append(item)
                if len(batch) >= _LOKI_BATCH_SIZE:
                    _flush()
            except queue.Empty:
                _flush()  # 超时,刷盘当前批次
        _flush()


def _build_loki_payload(
    batch: list[tuple[int, str]], base_labels: dict[str, str]
) -> dict:
    """将批次合并为 Loki push API 的 streams 结构。
    按 (level, logger) 分组,同组共享 stream labels,减少 HTTP body。
    """
    from collections import defaultdict

    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for levelno, line in batch:
        try:
            ts, body = line.split(" ", 1)
        except ValueError:
            ts, body = str(int(datetime.datetime.now().timestamp() * 1e9)), line
        level = logging.getLevelName(levelno)
        # 从 JSON line 中提取 logger(省得再传一遍)
        try:
            entry = json.loads(body)
            logger_name = entry.get("logger", "unknown")
        except Exception:
            logger_name = "unknown"
        groups[(level, logger_name)].append((ts, body))

    streams = []
    for (level, logger_name), values in groups.items():
        streams.append({
            "stream": {**base_labels, "level": level, "logger": logger_name},
            "values": [[ts, body] for ts, body in values],
        })
    return {"streams": streams}


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
    stream_handler.addFilter(_SessionContextFilter())
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
        file_handler.addFilter(_SessionContextFilter())
        file_handler.setFormatter(
            _build_formatter(settings.LOG_FORMAT, use_color=False)
        )
        root.addHandler(file_handler)

    # 可选 Loki handler（JSON 格式,后台线程批量推送）
    if settings.LOKI_ENABLED:
        loki = _LokiHandler(
            settings.LOKI_URL,
            {"app": settings.LOKI_APP_LABEL},
        )
        loki.addFilter(_SessionContextFilter())
        loki.setFormatter(JsonFormatter())
        root.addHandler(loki)

    # 收编 uvicorn / watchfiles，统一走 root
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "watchfiles.main"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    if invalid_level:
        _logger.warning(
            "未知 LOG_LEVEL %r，已回退为 INFO", settings.LOG_LEVEL
        )
