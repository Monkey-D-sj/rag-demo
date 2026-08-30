"""PDF / DOCX 表格提取,输出 列名+值 拼接文本 + 结构化元数据。

PDF 使用 pdfplumber（处理跨页表格、合并单元格）；
DOCX 使用 python-docx（已有依赖，无需额外安装）。
"""
from __future__ import annotations

import io
import re

from rag.common.logging import get_logger

logger = get_logger()

# ── 表格数据模型 ──


class TableBlock:
    """从文档提取的一张表格的结构化表示。"""

    __slots__ = ("caption", "headers", "rows", "text", "page")

    def __init__(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        caption: str = "",
        page: int | None = None,
    ) -> None:
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.page = page
        self.text = _flatten_table(headers, rows, caption)

    @property
    def meta(self) -> dict:
        """返回供 chunk metadata JSONB 列存储的元数据。"""
        return {
            "is_table": True,
            "table_headers": self.headers,
            "table_rows": len(self.rows),
            "table_cols": len(self.headers),
            "table_caption": self.caption,
            "table_summary": self.text,
        }

    @property
    def embed_text(self) -> str:
        """供 embedding 使用的文本:全表 列名+值 拼接。"""
        return self.text


# ── 列名+值 拼接(不依赖 LLM) ──


def _flatten_table(
    headers: list[str],
    rows: list[list[str]],
    caption: str = "",
) -> str:
    """全表按 列名+值 拼接为单段文本,不做截断,供 embedding 检索。

    每行把非空单元格展开为"列名:值",行内用逗号连接,行间用句号连接,
    表名作前缀。让"查找 Q4 营收"这类查询能命中纯数字表格。
    """
    parts: list[str] = []

    if caption.strip():
        parts.append(f"表格：{caption.strip()}")

    for row in rows:
        cells = [f"{h}：{c}" for h, c in zip(headers, row) if h and c]
        if cells:
            parts.append("，".join(cells))

    return "。".join(parts) if parts else "表格数据"


# ── 公共入口 ──


def extract_tables(
    data: str | bytes,
    content_type: str,
) -> list[TableBlock]:
    """从文件字节或路径中提取所有表格，统一为 TableBlock 列表。

    支持 PDF 和 DOCX；TXT/MD 无表格返回空列表。
    单个表格解析失败时跳过（记 warning），不中断整体提取。
    PDF 传路径利用 pdfplumber 延迟加载,大文件不占内存。
    """
    if content_type == "pdf":
        return _extract_pdf_tables(data)
    elif content_type == "docx":
        return _extract_docx_tables(data)
    else:
        # txt / md 无表格结构
        return []


# ── PDF 表格提取（pdfplumber） ──


def _extract_pdf_tables(data: str | bytes) -> list[TableBlock]:
    """用 pdfplumber 逐页提取 PDF 表格。传路径延迟加载,传 bytes 兼容旧调用。"""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber 未安装，跳过 PDF 表格提取")
        return []

    tables: list[TableBlock] = []
    try:
        src = data if isinstance(data, str) else io.BytesIO(data)
        with pdfplumber.open(src) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = _extract_page_tables(page, page_num)
                tables.extend(page_tables)
    except Exception:
        logger.warning("PDF 表格提取失败（可能是扫描件或无表格层）", exc_info=True)

    if tables:
        logger.info("PDF 提取到 %d 个表格（共 %d 页）", len(tables), len(tables))
    return tables


def _extract_page_tables(page, page_num: int) -> list[TableBlock]:
    """从单页提取表格列表。

    pdfplumber 对扫描件页返回空列表（不抛异常）；单个表格解析失败跳过。
    """
    tables: list[TableBlock] = []
    try:
        raw_tables = page.extract_tables()
    except Exception:
        logger.warning("第 %d 页表格提取异常，跳过", page_num, exc_info=True)
        return tables

    for table_idx, raw in enumerate(raw_tables):
        try:
            block = _pdfplumber_table_to_block(raw, page_num)
            if block is not None:
                tables.append(block)
        except Exception:
            logger.warning(
                "第 %d 页第 %d 个表格解析失败，跳过",
                page_num, table_idx + 1, exc_info=True,
            )
    return tables


def _pdfplumber_table_to_block(
    raw: list[list[str | None]], page_num: int
) -> TableBlock | None:
    """pdfplumber 原始输出 → TableBlock。

    自动识别首行为表头（全有值且与后续行结构一致）。
    过滤全空行；合并单元格的 None 填充为空字符串。
    """
    if not raw:
        return None

    # None → ""
    cleaned_rows = [[c if c is not None else "" for c in row] for row in raw]

    # 去全空行
    cleaned_rows = [r for r in cleaned_rows if any(c.strip() for c in r)]
    if not cleaned_rows:
        return None

    # 列数统一（取最多列）
    max_cols = max(len(r) for r in cleaned_rows)

    # 判断首行是否为表头：首行都有值 且 至少有一列在后续行中出现数字（表头通常不含纯数字）
    first = cleaned_rows[0]
    all_nonempty = all(c.strip() for c in first)
    if all_nonempty and len(cleaned_rows) > 1:
        headers = first
        body = cleaned_rows[1:]
    else:
        # 无表头：自动补列序号
        headers = [f"列{i + 1}" for i in range(max_cols)]
        body = cleaned_rows

    return TableBlock(headers=headers, rows=body, page=page_num)


# ── DOCX 表格提取（python-docx） ──


def _extract_docx_tables(data: str | bytes) -> list[TableBlock]:
    """用 python-docx 提取 DOCX 中的所有表格。"""
    from docx import Document

    tables: list[TableBlock] = []
    doc = Document(data) if isinstance(data, str) else Document(io.BytesIO(data))
    if not doc.tables:
        return tables

    for ti, table in enumerate(doc.tables):
        try:
            block = _docx_table_to_block(table)
            if block is not None:
                tables.append(block)
        except Exception:
            logger.warning("DOCX 第 %d 个表格解析失败，跳过", ti + 1, exc_info=True)

    if tables:
        logger.info("DOCX 提取到 %d 个表格", len(tables))
    return tables


def _docx_table_to_block(table) -> TableBlock | None:
    """python-docx Table 对象 → TableBlock。

    尝试从表格上方段落提取表名（以"表"开头的短文本）。
    合并单元格通过 grid_span / vmerge 检测（简化处理）。
    """
    cells_2d: list[list[str]] = []
    for row in table.rows:
        cells_2d.append([cell.text for cell in row.cells])

    if not cells_2d:
        return None

    # 去全空行
    cells_2d = [r for r in cells_2d if any(c.strip() for c in r)]
    if not cells_2d:
        return None

    # 尝试提取表名（表格前一个段落的短文本）
    caption = _extract_docx_caption(table)

    # 判断表头
    first = cells_2d[0]
    all_nonempty = all(c.strip() for c in first)
    if all_nonempty and len(cells_2d) > 1:
        headers = first
        body = cells_2d[1:]
    else:
        max_cols = max(len(r) for r in cells_2d)
        headers = [f"列{i + 1}" for i in range(max_cols)]
        body = cells_2d

    return TableBlock(headers=headers, rows=body, caption=caption)


_CAPTION_RE = re.compile(r"^(表|Table)\s*\d*\s*[：:.\-—]?\s*")
def _extract_docx_caption(table) -> str:
    """从 python-docx 表格对象的上方段落推断表名。"""
    try:
        # _tbl 是 lxml 元素，preceding-sibling 找前面的 <w:p>
        prev = table._tbl.getprevious()
        while prev is not None:
            if prev.tag.endswith("}p"):  # w:p paragraph
                text = "".join(prev.itertext()).strip()
                if text and len(text) <= 80 and _CAPTION_RE.match(text):
                    return text
                elif text:
                    # 非表名段落（停止向上搜索，避免取到正文）
                    break
            prev = prev.getprevious()
    except Exception:
        pass
    return ""
