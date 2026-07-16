from pathlib import Path

from rag.document.chunker import SplitStrategy
from rag.document.pipeline import _parse_and_chunk

_HERE = Path(__file__).parent


def test_parse_and_chunk_docx():
    """解析 test.docx 并用三种策略切分，验证归一化输出结构。"""
    data = (_HERE / "test.docx").read_bytes()

    result = _parse_and_chunk(data, "docx", SplitStrategy.paragraph_semantic, 800, 100)

    for chunk in result:
        print(chunk)
