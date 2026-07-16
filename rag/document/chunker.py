import re
from enum import Enum
from typing import assert_never

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class SplitStrategy(Enum):
    fixed_size = "fixed_size"
    recursive_character = "recursive_character"
    paragraph_semantic = "paragraph_semantic"


def chunk_by_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按固定字符数切分，不感知语义边界。"""
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# 中文分隔符:在默认 "\n\n" / "\n" / " " / "" 之前插入中文标点，
# 使得句子级的切分优先于空格和字符级，避免在句子中间切断。
_SEPARATORS = [
    "\n\n", "\n",
    "。", "！", "？", "；", "，",
    " ", "",
]

def chunk_by_recursive_character(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按递归字符策略切分：依次用章节标题、段落、句子、字符等分隔符切分。"""
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        is_separator_regex=True,
        keep_separator=False,
    )
    return splitter.split_text(text)

def chunk(
    strategy: SplitStrategy, text: str, chunk_size: int, overlap: int,
) -> list[str]:
    """统一入口：按策略切分文本。"""
    text = text.replace("\x00", "")
    if not text.strip():
        return []

    match strategy:
        case SplitStrategy.fixed_size:
            return chunk_by_fixed_size(text, chunk_size, overlap)
        case SplitStrategy.recursive_character:
            return chunk_by_recursive_character(text, chunk_size, overlap)
        case SplitStrategy.paragraph_semantic:
            return chunk_by_paragraph_semantic(text, chunk_size, overlap)
        case _:
            assert_never(strategy)


# 章内二次切分后，低于此阈值的碎片会被合并到相邻 chunk
_MIN_SUB_CHUNK_SIZE = 200

# 匹配章节标题行
_CHAPTER_HEADING_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千万\d]+[章节回卷篇])\s*[^\n]*",
    re.MULTILINE,
)

# 匹配条款标题行（整行，含后续内容直至换行）
_ARTICLE_HEADING_RE = re.compile(
    r"^(第[零一二三四五六七八九十百\d]+条[^\n]*\n)",
    re.MULTILINE,
)


def _split_by_headings(
    text: str,
    heading_re: re.Pattern,
) -> list[tuple[str, str]]:
    """按标题行切分文本，返回 [(标题, 正文), ...]。无匹配则整篇作为一个分段。

    第一个标题之前的前导内容会作为空标题段保留，不会丢失。
    """
    matches = list(heading_re.finditer(text))
    if not matches:
        return [("", text)]

    segments: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        # 第一个标题之前的前导内容作为前置段
        if i == 0:
            preamble = text[: m.start()].strip()
            if preamble:
                segments.append(("", preamble))

        if content:
            segments.append((title, content))
    return segments


def chunk_by_paragraph_semantic(
    text: str, max_chunk_size: int, chunk_overlap: int,
) -> list[str]:
    """逐级切分：章 → 条 → 尺寸回退。标题拼回正文，返回纯字符串列表。"""
    results: list[str] = []

    chapters = _split_by_headings(text, _CHAPTER_HEADING_RE) or [("", text)]
    for ch_title, ch_text in chapters:
        articles = _split_by_headings(ch_text, _ARTICLE_HEADING_RE)

        if not articles:
            _emit(results, ch_title, ch_text, max_chunk_size, chunk_overlap)
            continue

        for art_title, art_content in articles:
            prefix = f"{ch_title}\n{art_title}".strip() if ch_title else art_title
            _emit(results, prefix, art_content, max_chunk_size, chunk_overlap)

    return results


def _emit(
    results: list[str],
    prefix: str,
    content: str,
    max_chunk_size: int,
    chunk_overlap: int,
) -> None:
    """将标题拼回正文，填入结果。超过大小限制时切分。"""
    full = f"{prefix}\n{content}".strip() if prefix else content
    if len(full) <= max_chunk_size:
        results.append(full)
    else:
        results.extend(_merge_small_chunks(
            chunk_by_recursive_character(full, max_chunk_size, chunk_overlap),
            min_size=min(_MIN_SUB_CHUNK_SIZE, max_chunk_size // 2),
        ))


def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    """合并过短的 chunk：将不满足 min_size 的 chunk 向前并入相邻 chunk。

    递归消除：首轮合并后可能产生新的短 chunk（原短 chunk 并入后仍不足），
    故循环直到不再变化或只剩一个 chunk。
    """
    if len(chunks) <= 1:
        return list(chunks)

    while True:
        changed = False
        merged: list[str] = []
        pending: str = ""  # 待并入下一个 chunk 的短内容

        for c in chunks:
            if pending:
                c = pending + "\n\n" + c
                pending = ""
                changed = True
            if len(c) < min_size:
                pending = c
            else:
                merged.append(c)

        # 末尾残留的短内容：并入前一个 chunk
        if pending:
            if merged:
                merged[-1] = merged[-1] + "\n\n" + pending
            else:
                merged.append(pending)
            changed = True

        chunks = merged
        if not changed or len(chunks) <= 1:
            break

    return chunks
