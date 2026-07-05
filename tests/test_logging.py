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


def _record(level=logging.INFO, msg="hello", exc_info=None):
    return logging.LogRecord(
        name="rag.test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )


def test_json_formatter_basic_fields():
    out = JsonFormatter().format(_record(msg="hello world"))
    data = json.loads(out)
    assert data["level"] == "INFO"
    assert data["logger"] == "rag.test"
    assert data["message"] == "hello world"
    assert "timestamp" in data


def test_json_formatter_includes_exc_info():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record(level=logging.ERROR, msg="failed", exc_info=sys.exc_info())
    data = json.loads(JsonFormatter().format(rec))
    assert "exc_info" in data
    assert "ValueError: boom" in data["exc_info"]


def test_text_formatter_plain_has_no_ansi():
    out = ColorTextFormatter(use_color=False).format(_record(msg="hi"))
    assert "\x1b[" not in out
    assert "INFO" in out
    assert "rag.test" in out
    assert "hi" in out


def test_text_formatter_color_has_ansi():
    out = ColorTextFormatter(use_color=True).format(_record(msg="hi"))
    assert "\x1b[" in out


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    root = logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    root.handlers.extend(saved)
    root.setLevel(saved_level)


def test_setup_logging_text_adds_stream_handler():
    setup_logging(Settings(LOG_FORMAT="text"))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_is_idempotent():
    s = Settings(LOG_FORMAT="text")
    setup_logging(s)
    setup_logging(s)
    assert len(logging.getLogger().handlers) == 1


def test_setup_logging_respects_level():
    setup_logging(Settings(LOG_LEVEL="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_json_output_is_valid_json(capsys):
    setup_logging(Settings(LOG_FORMAT="json"))
    logging.getLogger("rag.test").error("boom")
    err = capsys.readouterr().err
    line = [ln for ln in err.splitlines() if ln.strip()][-1]
    data = json.loads(line)
    assert data["message"] == "boom"
    assert data["level"] == "ERROR"


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


def test_setup_logging_invalid_level_falls_back_to_info():
    setup_logging(Settings(LOG_LEVEL="NOTALEVEL"))
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_tames_uvicorn_loggers():
    # 预先给 uvicorn logger 装一个 handler、关掉 propagate，模拟 uvicorn 默认状态
    uv = logging.getLogger("uvicorn")
    uv.addHandler(logging.NullHandler())
    uv.propagate = False

    setup_logging(Settings(LOG_FORMAT="text"))

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        assert lg.handlers == []
        assert lg.propagate is True


def _make_record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        "rag.test", logging.INFO, __file__, 1, msg, (), None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_includes_extra_fields():
    out = json.loads(JsonFormatter().format(_make_record(kb_id="kb1", top_k=5)))
    assert out["kb_id"] == "kb1"
    assert out["top_k"] == 5


def test_json_formatter_serializes_non_json_values_via_str():
    rec = _make_record(when=datetime.datetime(2026, 7, 5, 12, 0, 0))
    out = json.loads(JsonFormatter().format(rec))
    assert "2026-07-05" in out["when"]


def test_json_formatter_does_not_leak_std_attrs():
    out = json.loads(JsonFormatter().format(_make_record()))
    assert "args" not in out
    assert "lineno" not in out
    assert "levelno" not in out


def test_session_filter_injects_and_resets():
    f = _SessionContextFilter()
    token = bind_session("s-1")
    try:
        record = _make_record()
        assert f.filter(record) is True
        assert record.session_id == "s-1"
    finally:
        reset_session(token)
    record2 = _make_record()
    f.filter(record2)
    assert not hasattr(record2, "session_id")


def test_session_filter_keeps_explicit_extra():
    f = _SessionContextFilter()
    token = bind_session("ctx-session")
    try:
        record = _make_record(session_id="explicit")
        f.filter(record)
        assert record.session_id == "explicit"
    finally:
        reset_session(token)


def test_text_formatter_appends_session_suffix():
    line = ColorTextFormatter(use_color=False).format(
        _make_record(session_id="s-9")
    )
    assert line.endswith("| session=s-9")


def test_text_formatter_no_suffix_without_session():
    line = ColorTextFormatter(use_color=False).format(_make_record())
    assert "session=" not in line


def test_get_logger_returns_caller_module_name():
    # 存量 bug 回归:模块顶层 get_logger() 曾错误追溯到 importlib 帧
    import rag.document.retriever as m

    assert m.logger.name == "rag.document.retriever"
