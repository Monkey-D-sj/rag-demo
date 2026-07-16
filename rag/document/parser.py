import io

from docx import Document
from pypdf import PdfReader


def parse(data: bytes, content_type: str) -> str:
    """按类型把文件字节解析为纯文本。"""
    if content_type in ("txt", "md"):
        text = data.decode("utf-8", errors="replace")

    elif content_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        text = "\n".join(parts)
        if not text.strip():
            raise ValueError("PDF 无可提取文本(可能是扫描件)")

    elif content_type == "docx":
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(parts)
        if not text.strip():
            raise ValueError("DOCX 无可提取文本(可能是空文档或仅含图片)")

    else:
        raise ValueError(f"不支持的文件类型: {content_type}")

    # 统一清理：non-breaking space → 普通空格，去掉其他不可见控制字符
    return text.replace("\xa0", " ")
