from rag.document.chunker import chunk


def test_chunk_blank_returns_empty():
    assert chunk("   \n  ") == []


def test_chunk_short_text_single_chunk():
    assert chunk("hello world", 800, 100) == ["hello world"]


def test_chunk_long_text_splits_into_multiple():
    text = "段落。" * 1000
    out = chunk(text, 100, 20)
    assert len(out) > 1
