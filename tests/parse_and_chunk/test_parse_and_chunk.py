from pathlib import Path

from rag.document.chunker import SplitStrategy
from rag.document.pipeline import _parse_and_chunk

_HERE = Path(__file__).parent


def test_parse_and_chunk_docx():
    """解析 test.docx 并用三种策略切分，验证归一化输出结构。"""
    data = (_HERE / "test.docx").read_bytes()

    full_text, chunks = _parse_and_chunk(data, "docx", SplitStrategy.paragraph_semantic, 800, 100)

    assert isinstance(full_text, str)
    assert len(full_text) > 0
    assert isinstance(chunks, list)
    for text, meta in chunks:
        assert isinstance(text, str)
        assert isinstance(meta, dict)
        print(text[:80], meta)
