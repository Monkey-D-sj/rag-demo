from rag.config import SplitStrategy
from rag.document.chunker import chunk


def test_chunk_blank_returns_empty():
    assert chunk(SplitStrategy.recursive_character, "   \n  ", 800, 100) == []


def test_chunk_short_text_single_chunk():
    assert chunk(
        SplitStrategy.recursive_character, "hello world", 800, 100
    ) == ["hello world"]


def test_chunk_long_text_splits_into_multiple():
    text = "段落。" * 1000
    out = chunk(SplitStrategy.recursive_character, text, 100, 20)
    assert len(out) > 1


def test_chunk_fixed_size():
    out = chunk(SplitStrategy.fixed_size, "a" * 250, 100, 0)
    assert len(out) == 3
    assert all(len(c) <= 100 for c in out)


def test_paragraph_semantic_splits_by_chapter():
    text = "第一回 灵根育孕\n内容甲。\n第二回 悟彻菩提\n内容乙。"
    out = chunk(SplitStrategy.paragraph_semantic, text, 800, 100)
    assert [c["title"] for c in out] == ["第一回 灵根育孕", "第二回 悟彻菩提"]
    assert out[0]["content"] == "内容甲。"


def test_paragraph_semantic_no_heading_returns_whole_text():
    out = chunk(SplitStrategy.paragraph_semantic, "没有章节标题的正文。", 800, 100)
    assert out == [{"title": "", "content": "没有章节标题的正文。"}]
