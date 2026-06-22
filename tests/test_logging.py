import json
import logging

import pytest

from rag.common.logging import JsonFormatter, ColorTextFormatter, setup_logging
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


import json as _json


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
    setup_logging(Settings(log_format="text"))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


def test_setup_logging_is_idempotent():
    s = Settings(log_format="text")
    setup_logging(s)
    setup_logging(s)
    assert len(logging.getLogger().handlers) == 1


def test_setup_logging_respects_level():
    setup_logging(Settings(log_level="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_json_output_is_valid_json(capsys):
    setup_logging(Settings(log_format="json"))
    logging.getLogger("rag.test").error("boom")
    err = capsys.readouterr().err
    line = [ln for ln in err.splitlines() if ln.strip()][-1]
    data = _json.loads(line)
    assert data["message"] == "boom"
    assert data["level"] == "ERROR"


def test_setup_logging_creates_file_and_parent_dir(tmp_path):
    log_path = tmp_path / "logs" / "app.log"
    setup_logging(Settings(log_file=str(log_path)))
    from logging.handlers import RotatingFileHandler
    root = logging.getLogger()
    assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
    logging.getLogger("rag.test").error("written")
    for h in root.handlers:
        h.flush()
    assert log_path.exists()
    assert "written" in log_path.read_text(encoding="utf-8")


def test_setup_logging_invalid_level_falls_back_to_info():
    setup_logging(Settings(log_level="NOTALEVEL"))
    assert logging.getLogger().level == logging.INFO
