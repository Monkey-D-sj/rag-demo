import pytest

import rag.document.parser as parser_mod


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, stream):
        self.pages = [_FakePage("第一页"), _FakePage(""), _FakePage("第三页")]


class _EmptyReader:
    def __init__(self, stream):
        self.pages = [_FakePage(""), _FakePage("   ")]


def test_parse_txt_decodes_utf8():
    assert parser_mod.parse("你好".encode("utf-8"), "txt") == "你好"


def test_parse_md_decodes_utf8():
    assert parser_mod.parse(b"# title", "md") == "# title"


def test_parse_unknown_type_raises():
    with pytest.raises(ValueError):
        parser_mod.parse(b"x", "exe")


def test_parse_pdf_joins_nonempty_pages(monkeypatch):
    monkeypatch.setattr(parser_mod, "PdfReader", _FakeReader)
    out = parser_mod.parse(b"%PDF-fake", "pdf")
    assert "第一页" in out
    assert "第三页" in out


def test_parse_pdf_empty_raises(monkeypatch):
    monkeypatch.setattr(parser_mod, "PdfReader", _EmptyReader)
    with pytest.raises(ValueError):
        parser_mod.parse(b"%PDF-empty", "pdf")
