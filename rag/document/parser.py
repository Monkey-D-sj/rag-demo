import io

from pypdf import PdfReader


def parse(data: bytes, content_type: str) -> str:
    """按类型把文件字节解析为纯文本。"""
    if content_type in ("txt", "md"):
        return data.decode("utf-8", errors="replace")

    if content_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        text = "\n".join(parts)
        if not text.strip():
            raise ValueError("PDF 无可提取文本(可能是扫描件)")
        return text

    raise ValueError(f"不支持的文件类型: {content_type}")
