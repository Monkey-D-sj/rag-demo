import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文分隔符:在默认 "\n\n" / "\n" / " " / "" 之前插入中文标点，
# 使得句子级的切分优先于空格和字符级，避免在句子中间切断。
_SEPARATORS = [
    "\n\n", "\n",
    "。", "！", "？", "；", "，",
    " ", "",
]


async def chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """把文本切成块。空白文本返回空列表。CPU 操作走线程池避免阻塞 event loop。"""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return await asyncio.to_thread(splitter.split_text, text)
