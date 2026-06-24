from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """把文本切成块。空白文本返回空列表。"""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
