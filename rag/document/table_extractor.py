"""PDF / DOCX 表格提取，输出 Markdown pipe table + 结构化元数据。

PDF 使用 pdfplumber（处理跨页表格、合并单元格）；
DOCX 使用 python-docx（已有依赖，无需额外安装）。
"""
from __future__ import annotations

import io
import re

from rag.common.logging import get_logger

logger = get_logger()

# ── 表格数据模型 ──

# 超大表格截断：保留表头 + 前 N 行，其余行折叠提示
_MAX_TABLE_ROWS = 40
# 嵌入摘要的首行采样数
_SUMMARY_SAMPLE_ROWS = 3


class TableBlock:
    """从文档提取的一张表格的结构化表示。"""

    __slots__ = ("caption", "headers", "rows", "markdown", "page")

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
        self.markdown = _to_markdown(headers, rows)

    def truncated(self) -> bool:
        """该表格是否因行数过多被截断。"""
        return len(self.rows) > _MAX_TABLE_ROWS

    @property
    def meta(self) -> dict:
        """返回供 chunk metadata JSONB 列存储的元数据。"""
        return {
            "is_table": True,
            "table_headers": self.headers,
            "table_rows": len(self.rows),
            "table_cols": len(self.headers),
            "table_caption": self.caption,
            "table_summary": _build_summary(
                self.headers, self.rows, self.caption
            ),
        }

    @property
    def embed_text(self) -> str:
        """供 embedding 使用的文本：表名/摘要 + Markdown 正文。

        摘要在前，利用 embedding 模型对前置信息的偏好提升检索准确度。
        """
        summary = self.meta["table_summary"]
        return f"{summary}\n\n{self.markdown}"


# ── Markdown 转换 ──


def _escape_pipe(text: str) -> str:
    """转义 Markdown 表格中的 | 字符，避免破坏列结构。"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """将表头和行转为 Markdown pipe table 字符串。

    超大表格截断为前 _MAX_TABLE_ROWS 行 + 折叠提示。
    """
    n_cols = len(headers)

    # 补齐列数不一致的行
    def _pad(row: list[str], width: int) -> list[str]:
        padded = list(row)
        while len(padded) < width:
            padded.append("")
        return padded[:width]

    h = _pad(headers, n_cols)
    lines: list[str] = []
    # 表头行
    lines.append("| " + " | ".join(_escape_pipe(c) for c in h) + " |")
    # 分隔行
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")

    truncated = len(rows) > _MAX_TABLE_ROWS
    display_rows = rows[:_MAX_TABLE_ROWS] if truncated else rows
    for row in display_rows:
        r = _pad(row, n_cols)
        lines.append("| " + " | ".join(_escape_pipe(c) for c in r) + " |")

    if truncated:
        lines.append(f"\n*（表格共 {len(rows)} 行，此处仅展示前 {_MAX_TABLE_ROWS} 行）*")

    return "\n".join(lines)


# ── 自然语言摘要（不依赖 LLM） ──


def _build_summary(
    headers: list[str],
    rows: list[list[str]],
    caption: str = "",
) -> str:
    """不依赖 LLM 的规则摘要：表名 + 列名 + 前几行数据转自然语言。

    用于 embedding 检索，让"查找 Q4 营收"这类查询能命中纯数字表格。
    """
    parts: list[str] = []

    if caption.strip():
        parts.append(f"表格：{caption.strip()}")

    if headers:
        parts.append("包含列：" + "、".join(h for h in headers if h))

    if rows:
        sample = rows[:_SUMMARY_SAMPLE_ROWS]
        prose_lines: list[str] = []
        for i, row in enumerate(sample):
            pairs = [
                f"{h}为{cell}"
                for h, cell in zip(headers, row)
                if h and cell
            ]
            if pairs:
                prose_lines.append("第{}行：{}。".format(i + 1, "，".join(pairs)))
        if prose_lines:
            parts.append("示例数据：" + " ".join(prose_lines))

    return "；".join(parts) if parts else "表格数据"


# ── LLM 摘要（可选，best-effort） ──


async def generate_table_summary(
    llm,  # ChatModel, 惰性类型避免循环导入
    markdown: str,
    headers: list[str],
    caption: str = "",
) -> str | None:
    """用 LLM 生成表格自然语言摘要，提升 embedding 检索语义覆盖。

    失败时返回 None，调用方降级使用规则摘要（`_build_summary`）。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    header_hint = ""
    if headers:
        header_hint = f"列名：{'、'.join(headers)}。"
    caption_hint = f"表名：{caption}。" if caption.strip() else ""

    system = (
        "你是一个数据分析助手。请用 1-2 句流畅的中文概述给定表格的核心内容，"
        "包含：表格主题、关键字段、数据范围或趋势（如有）。"
        "只输出摘要文字，不要加前缀和引号。"
    )
    human = (
        f"{caption_hint}{header_hint}\n"
        f"表格内容（Markdown 格式）：\n{markdown}"
    )

    try:
        result = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=human),
        ])
        summary = str(result).strip() if result else ""
        if summary:
            return summary
    except Exception:
        logger.warning("表格摘要 LLM 调用失败，降级为规则摘要", exc_info=True)

    return None


# ── 公共入口 ──


def extract_tables(
    data: bytes,
    content_type: str,
) -> list[TableBlock]:
    """从文件字节中提取所有表格，统一为 TableBlock 列表。

    支持 PDF 和 DOCX；TXT/MD 无表格返回空列表。
    单个表格解析失败时跳过（记 warning），不中断整体提取。
    """
    if content_type == "pdf":
        return _extract_pdf_tables(data)
    elif content_type == "docx":
        return _extract_docx_tables(data)
    else:
        # txt / md 无表格结构
        return []


# ── PDF 表格提取（pdfplumber） ──


def _extract_pdf_tables(data: bytes) -> list[TableBlock]:
    """用 pdfplumber 逐页提取 PDF 表格。"""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber 未安装，跳过 PDF 表格提取")
        return []

    tables: list[TableBlock] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
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


def _extract_docx_tables(data: bytes) -> list[TableBlock]:
    """用 python-docx 提取 DOCX 中的所有表格。"""
    from docx import Document

    tables: list[TableBlock] = []
    doc = Document(io.BytesIO(data))
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
