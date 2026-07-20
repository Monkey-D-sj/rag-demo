import datetime
import json
import logging

import pytest

from rag.common.logging import (
    ColorTextFormatter,
    JsonFormatter,
    _SessionContextFilter,
    bind_session,
    reset_session,
    setup_logging,
)
from rag.config import Settings


def _record(level=logging.INFO, msg="hello", exc_info=None, **extra):
    record = logging.LogRecord(
        name="rag.test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


# ── Formatter ──

def test_json_formatter_basic_fields():
    out = JsonFormatter().format(_record(msg="hello world"))
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["logger"] == "rag.test"
    assert data["message"] == "hello world"
    assert "timestamp" in data


def test_json_formatter_includes_exc_info_and_extras():
    """exc_info 被序列化；自定义 extra 字段出现在输出中；std attrs 不泄漏。"""
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record(
            level=logging.ERROR, msg="failed", exc_info=sys.exc_info(),
            kb_id="kb1", when=datetime.datetime(2026, 7, 5, 12, 0, 0),
        )
    data = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in data["exc_info"]
    assert data["kb_id"] == "kb1"
    assert "2026-07-05" in data["when"]
    assert "args" not in data        # std attrs 不泄漏
    assert "levelno" not in data


def test_text_formatter_color_toggle():
    """use_color=False 无 ANSI 转义；use_color=True 含 ANSI。"""
    plain = ColorTextFormatter(use_color=False).format(_record(msg="hi"))
    assert "\x1b[" not in plain
    assert "INFO" in plain

    colored = ColorTextFormatter(use_color=True).format(_record(msg="hi"))
    assert "\x1b[" in colored


# ── setup_logging ──

@pytest.fixture(autouse=True)
def _clean_root_handlers(monkeypatch):
    import rag.common.logging as logging_mod

    root = logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    monkeypatch.setattr(logging_mod, "_logging_initialized", False)
    yield
    for h in root.handlers:
        h.close()
    root.handlers.clear()
    root.handlers.extend(saved)
    root.setLevel(saved_level)


def test_setup_logging_adds_stream_handler():
    setup_logging(Settings(LOG_FORMAT="text"))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_is_idempotent():
    s = Settings(LOG_FORMAT="text")
    setup_logging(s)
    setup_logging(s)
    assert len(logging.getLogger().handlers) == 1


def test_setup_logging_respects_level_and_falls_back_on_invalid():
    setup_logging(Settings(LOG_LEVEL="WARNING"))
    assert logging.getLogger().level == logging.WARNING

    # 重置后测非法 level 回退
    import rag.common.logging as logging_mod
    for h in logging.getLogger().handlers:
        h.close()
    logging.getLogger().handlers.clear()
    logging_mod._logging_initialized = False

    setup_logging(Settings(LOG_LEVEL="NOTALEVEL"))
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_creates_file_and_parent_dir(tmp_path):
    log_path = tmp_path / "logs" / "app.log"
    setup_logging(Settings(LOG_FILE=str(log_path)))
    from logging.handlers import RotatingFileHandler
    root = logging.getLogger()
    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    logging.getLogger("rag.test").error("written")
    for h in root.handlers:
        h.flush()
    assert log_path.exists()
    assert "written" in log_path.read_text(encoding="utf-8")


def test_setup_logging_tames_uvicorn_loggers():
    uv = logging.getLogger("uvicorn")
    uv.addHandler(logging.NullHandler())
    uv.propagate = False

    setup_logging(Settings(LOG_FORMAT="text"))

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        assert lg.handlers == []
        assert lg.propagate is True


# ── Session context ──

def test_session_filter_and_text_suffix():
    """session_id 注入 LogRecord 并在 text 输出末尾追加。"""
    f = _SessionContextFilter()
    token = bind_session("s-1")
    try:
        record = _record()
        assert f.filter(record) is True
        assert record.session_id == "s-1"
    finally:
        reset_session(token)

    record2 = _record()
    f.filter(record2)
    assert not hasattr(record2, "session_id")

    # text formatter 追加 session 后缀
    line = ColorTextFormatter(use_color=False).format(_record(session_id="s-9"))
    assert line.endswith("| session_id=s-9")

    # 无 session 时不追加
    line_no = ColorTextFormatter(use_color=False).format(_record())
    assert "session_id=" not in line_no
