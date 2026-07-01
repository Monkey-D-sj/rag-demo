import re

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from rag.config import SplitStrategy


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
    r"第[零一二三四五六七八九十百千万0-9]+章",
    r"第[零一二三四五六七八九十百千万0-9]+节",
    r"第[零一二三四五六七八九十百千万0-9]+回",
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
    )
    return splitter.split_text(text)

def chunk(
    strategy: SplitStrategy, text: str, chunk_size: int, overlap: int,
) -> list[str] | list[dict[str, str]]:
    """统一入口：按策略切分文本。

    fixed_size / recursive_character → list[str]
    paragraph_semantic → list[dict[str, str]]（含章节标题元数据）
    """
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


# 章内二次切分后，低于此阈值的碎片会被合并到相邻 chunk
_MIN_SUB_CHUNK_SIZE = 200

# 匹配章节标题行，group(1) 是 "第一回" 这类编号+量词
_CHAPTER_HEADING_RE = re.compile(
    r"^(第[零一二三四五六七八九十百千万\d]+[章节回卷篇])\s*[^\n]*",
    re.MULTILINE,
)
def chunk_by_paragraph_semantic(
    text: str, max_chunk_size: int, chunk_overlap: int,
) -> list[dict[str, str]]:
    """按章节切分文本，返回 [{"title": str, "content": str}, ...]。

    章节由 "第X回/第X章/第X节" 等标题行识别。
    若单章过长则用 RecursiveCharacterTextSplitter 进一步切分，
    title 追加 "(1/3)" 式分段号。
    """
    matches = list(_CHAPTER_HEADING_RE.finditer(text))

    if not matches:
        return [{"title": "", "content": text.strip()}]

    results: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        # 章节标题行的下一行开始
        start = m.end()
        if start < len(text) and text[start] == "\n":
            start += 1
        # 到下一个章节标题行之前为止
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if not content:
            continue

        if len(content) > max_chunk_size:
            sub_chunks = chunk_by_recursive_character(content, max_chunk_size, chunk_overlap)
            # 合并因段落边界产生的孤立短 chunk（如章首诗歌、对话片段）
            sub_chunks = _merge_small_chunks(
                sub_chunks,
                min_size=min(_MIN_SUB_CHUNK_SIZE, max_chunk_size // 2),
            )
            for j, sub in enumerate(sub_chunks):
                results.append({
                    "title": f"{title}({j + 1}/{len(sub_chunks)})",
                    "content": sub,
                })
        else:
            results.append({"title": title, "content": content})

    return results


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
