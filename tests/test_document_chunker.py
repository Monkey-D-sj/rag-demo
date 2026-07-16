from rag.document.chunker import SplitStrategy
from rag.document.chunker import chunk


def test_chunk_blank_returns_empty():
    assert chunk(SplitStrategy.recursive_character, "   \n  ", 800, 100) == []


def test_chunk_short_text_single_chunk():
    result = chunk(SplitStrategy.recursive_character, "hello world", 800, 100)
    assert len(result) == 1
    assert result[0][0] == "hello world"
    assert result[0][1] == {}


def test_chunk_long_text_splits_into_multiple():
    text = "段落。" * 1000
    out = chunk(SplitStrategy.recursive_character, text, 100, 20)
    assert len(out) > 1
    for text_piece, meta in out:
        assert isinstance(text_piece, str)
        assert isinstance(meta, dict)


def test_chunk_fixed_size():
    out = chunk(SplitStrategy.fixed_size, "a" * 250, 100, 0)
    assert len(out) == 3
    assert all(len(text) <= 100 for text, _ in out)


def test_paragraph_semantic_splits_by_chapter():
    text = "第一回 灵根育孕\n内容甲。\n第二回 悟彻菩提\n内容乙。"
    out = chunk(SplitStrategy.paragraph_semantic, text, 800, 100)
    assert len(out) == 2
    assert out[0][0] == "第一回 灵根育孕\n内容甲。"
    assert out[1][0] == "第二回 悟彻菩提\n内容乙。"
    # metadata should contain chapter title
    assert out[0][1]["chapter"] == "第一回 灵根育孕"
    assert out[1][1]["chapter"] == "第二回 悟彻菩提"


def test_paragraph_semantic_no_heading_returns_whole_text():
    out = chunk(SplitStrategy.paragraph_semantic, "没有章节标题的正文。", 800, 100)
    assert len(out) == 1
    assert out[0][0] == "没有章节标题的正文。"
    assert out[0][1] == {}


def test_paragraph_semantic_no_heading_long_text_respects_max_size():
    """无章节标题的长文本必须回退到尺寸切分,不能整篇返回单个巨型 chunk。"""
    text = "这是一段没有任何章节标题的普通正文，用来验证回退切分。" * 200
    out = chunk(SplitStrategy.paragraph_semantic, text, 800, 100)
    assert len(out) > 1
    assert all(len(text) <= 800 for text, _ in out)


def test_paragraph_semantic_article_metadata():
    """条款标题应写入 metadata 的 article 字段。"""
    text = "第一章 总则\n第一条 立法目的\n法律条文内容。\n第二条 适用范围\n其他内容。"
    out = chunk(SplitStrategy.paragraph_semantic, text, 800, 100)
    assert len(out) == 2
    assert out[0][1]["chapter"] == "第一章 总则"
    assert out[0][1]["article"] == "第一条 立法目的"
    assert out[1][1]["chapter"] == "第一章 总则"
    assert out[1][1]["article"] == "第二条 适用范围"
