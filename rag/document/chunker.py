from langchain_text_splitters import RecursiveCharacterTextSplitter

# 中文分隔符:在默认 "\n\n" / "\n" / " " / "" 之前插入中文标点，
# 使得句子级的切分优先于空格和字符级，避免在句子中间切断。
_SEPARATORS = [
    "\n\n", "\n",
    "。", "！", "？", "；", "，",
    " ", "",
]


def chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """把文本切成块。空白文本返回空列表。

    纯 CPU 同步函数;调用方(pipeline)负责把 parse+chunk 合并到一次线程池调用,
    避免在 worker event loop 上阻塞其他并发 job。
    """
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
