import json
import logging

from rag.common.logging import JsonFormatter, ColorTextFormatter


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
