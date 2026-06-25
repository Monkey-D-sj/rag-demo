from rag.document.chunker import chunk


async def test_chunk_blank_returns_empty():
    assert await chunk("   \n  ") == []


async def test_chunk_short_text_single_chunk():
    assert await chunk("hello world", 800, 100) == ["hello world"]


async def test_chunk_long_text_splits_into_multiple():
    text = "段落。" * 1000
    out = await chunk(text, 100, 20)
    assert len(out) > 1
