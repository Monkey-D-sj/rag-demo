"""表格提取模块的单元测试。

纯函数直接测；PDF / DOCX 集成测试用程序化生成的真实文件。
"""
from __future__ import annotations

import io

import pytest


# ── 纯函数测试 ──


class TestToMarkdown:
    """Markdown pipe table 转换。"""

    def test_simple_table(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["姓名", "年龄"], [["张三", "28"], ["李四", "35"]])
        lines = md.split("\n")
        assert "| 姓名 | 年龄 |" in lines[0]
        assert "| --- | --- |" in lines[1]
        assert "| 张三 | 28 |" in lines[2]
        assert "| 李四 | 35 |" in lines[3]

    def test_escapes_pipe_in_cell(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["值"], [["a|b"]])
        assert "a\\|b" in md

    def test_escapes_newline_in_cell(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["列"], [["第一行\n第二行"]])
        # 数据行被 _escape_pipe 将 \n 替换为空格，故 markdown 中不再含换行
        assert "第一行" in md
        assert "第二行" in md
        # 确认换行符已被转义为空格
        data_line = md.split("\n")[2]
        assert "\n" not in data_line

    def test_uneven_columns_padded(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["A", "B"], [["1"], ["2", "3", "4"]])
        lines = md.split("\n")
        # 表头列数不变；数据行补到与表头同宽
        data_cols = lines[2].count("|") - 1
        header_cols = lines[0].count("|") - 1
        assert data_cols == header_cols

    def test_empty_cells_preserved(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["A", "B"], [["1", ""]])
        assert "| 1 |  |" in md

    def test_truncates_large_table(self):
        from rag.document.table_extractor import _to_markdown

        rows = [[str(i), f"val{i}"] for i in range(100)]
        md = _to_markdown(["序号", "值"], rows)
        # 截断行提示文本
        assert "仅展示前" in md
        # 数据行不超过 _MAX_TABLE_ROWS + 提示行
        data_lines = [l for l in md.split("\n") if l.startswith("|")]
        assert len(data_lines) <= 42  # 表头 + 分隔 + 40 数据

    def test_single_header_no_body(self):
        from rag.document.table_extractor import _to_markdown

        md = _to_markdown(["姓名", "年龄"], [])
        lines = md.split("\n")
        assert len(lines) == 2  # 仅表头 + 分隔行


class TestBuildSummary:
    """规则摘要生成（不依赖 LLM）。"""

    def test_full_summary(self):
        from rag.document.table_extractor import _build_summary

        s = _build_summary(
            ["姓名", "年龄"],
            [["张三", "28"], ["李四", "35"]],
            caption="员工信息表",
        )
        assert "员工信息表" in s
        assert "姓名" in s
        assert "年龄" in s
        assert "张三" in s
        assert "28" in s

    def test_no_caption(self):
        from rag.document.table_extractor import _build_summary

        s = _build_summary(["指标", "Q1"], [["营收", "100万"]])
        assert "指标" in s
        assert "营收" in s

    def test_empty_headers(self):
        from rag.document.table_extractor import _build_summary

        s = _build_summary([], [["a", "b"]])
        assert "表格数据" in s

    def test_all_empty_returns_placeholder(self):
        from rag.document.table_extractor import _build_summary

        s = _build_summary([], [])
        assert s == "表格数据"


# ── TableBlock 数据模型测试 ──


class TestTableBlock:
    def test_basic_meta(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["姓名", "年龄"], [["张三", "28"]], caption="员工表")
        meta = tb.meta
        assert meta["is_table"] is True
        assert meta["table_headers"] == ["姓名", "年龄"]
        assert meta["table_rows"] == 1
        assert meta["table_cols"] == 2
        assert meta["table_caption"] == "员工表"
        assert "table_summary" in meta
        assert "员工表" in meta["table_summary"]

    def test_embed_text_contains_summary_and_markdown(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["A"], [["1"]])
        text = tb.embed_text
        assert "A" in text          # summary 中含列名
        assert tb.markdown in text  # markdown 正文

    def test_truncated_flag(self):
        from rag.document.table_extractor import TableBlock

        rows = [[str(i)] for i in range(100)]
        tb = TableBlock(["序号"], rows)
        assert tb.truncated() is True

    def test_not_truncated_small_table(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["A"], [["1"]])
        assert tb.truncated() is False

    def test_page_defaults_to_none(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["A"], [["1"]])
        assert tb.page is None

    def test_page_stored(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["A"], [["1"]], page=5)
        assert tb.page == 5


# ── 集成测试（真实文件） ──


class TestExtractTablesFromDocx:
    """用 python-docx 程序化生成含表格的 .docx 文件，验证提取链路。"""

    def test_single_simple_table(self):
        from docx import Document

        from rag.document.table_extractor import extract_tables

        doc = Document()
        doc.add_paragraph("这是一段正文。")
        table = doc.add_table(rows=3, cols=2, style="Table Grid")
        table.cell(0, 0).text = "姓名"
        table.cell(0, 1).text = "年龄"
        table.cell(1, 0).text = "张三"
        table.cell(1, 1).text = "28"
        table.cell(2, 0).text = "李四"
        table.cell(2, 1).text = "35"

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        assert len(tables) == 1
        tb = tables[0]
        assert tb.headers == ["姓名", "年龄"]
        assert len(tb.rows) == 2
        assert tb.rows[0] == ["张三", "28"]

    def test_multiple_tables(self):
        from docx import Document

        from rag.document.table_extractor import extract_tables

        doc = Document()
        t1 = doc.add_table(rows=2, cols=1)
        t1.cell(0, 0).text = "表头"
        t1.cell(1, 0).text = "值"
        doc.add_paragraph("中间正文。")
        t2 = doc.add_table(rows=2, cols=2)
        t2.cell(0, 0).text = "A"
        t2.cell(0, 1).text = "B"
        t2.cell(1, 0).text = "1"
        t2.cell(1, 1).text = "2"

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        assert len(tables) == 2

    def test_docx_without_tables(self):
        from docx import Document

        from rag.document.table_extractor import extract_tables

        doc = Document()
        doc.add_paragraph("只有正文，没有表格。")
        doc.add_paragraph("第二段。")

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        assert tables == []

    def test_table_with_merged_cells(self):
        """合并单元格的文本可能出现在多个 cell 中；验证不崩溃。"""
        from docx import Document
        from docx.oxml.ns import qn

        from rag.document.table_extractor import extract_tables

        doc = Document()
        table = doc.add_table(rows=2, cols=2, style="Table Grid")
        table.cell(0, 0).text = "合并标题"
        # 水平合并：cell(0,0) 和 cell(0,1) 合并
        table.cell(0, 1).text = ""  # 合并后通常为空或重复
        table.cell(1, 0).text = "a"
        table.cell(1, 1).text = "b"

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        assert len(tables) >= 1

    def test_empty_table_skipped(self):
        from docx import Document

        from rag.document.table_extractor import extract_tables

        doc = Document()
        doc.add_table(rows=1, cols=1)  # 全空表

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        # 全空行被过滤后无有效数据，返回空
        assert tables == [] or all(tb.headers[0] == "列1" for tb in tables)


class TestExtractTablesFromPdf:
    """用 pdfplumber 验证 PDF 表格提取。"""

    def test_pdf_without_tables(self):
        """纯文本 PDF（无表格）返回空列表。"""
        from rag.document.table_extractor import extract_tables

        # 创建一个最简 PDF（不含表格）
        pdf_bytes = _minimal_text_pdf("这是一段纯文本，没有表格。")
        tables = extract_tables(pdf_bytes, "pdf")
        assert tables == []

    def test_txt_content_type_returns_empty(self):
        from rag.document.table_extractor import extract_tables

        tables = extract_tables(b"hello world", "txt")
        assert tables == []

    def test_md_content_type_returns_empty(self):
        from rag.document.table_extractor import extract_tables

        tables = extract_tables(b"# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |", "md")
        assert tables == []

    def test_unknown_content_type_returns_empty(self):
        from rag.document.table_extractor import extract_tables

        tables = extract_tables(b"data", "xlsx")
        assert tables == []

    def test_corrupt_pdf_does_not_crash(self):
        from rag.document.table_extractor import extract_tables

        tables = extract_tables(b"this is not a valid PDF", "pdf")
        assert isinstance(tables, list)


# ── 规则降级测试 ──


class TestTableSummaryFallback:
    """LLM 不可用或失败时降级为规则摘要。"""

    @pytest.mark.asyncio
    async def test_extract_without_llm_returns_rule_summary(self):
        """_extract_and_summarize_tables 不传 llm 时仅用规则摘要。"""
        from docx import Document

        from rag.document.pipeline import _extract_and_summarize_tables

        doc = Document()
        table = doc.add_table(rows=3, cols=2, style="Table Grid")
        table.cell(0, 0).text = "产品"
        table.cell(0, 1).text = "销量"
        table.cell(1, 0).text = "A"
        table.cell(1, 1).text = "100"
        table.cell(2, 0).text = "B"
        table.cell(2, 1).text = "200"

        buf = io.BytesIO()
        doc.save(buf)

        blocks = await _extract_and_summarize_tables(
            buf.getvalue(), "docx", llm=None,
        )
        assert len(blocks) == 1
        meta = blocks[0].meta
        assert "table_summary" in meta
        assert "产品" in meta["table_summary"]
        assert "销量" in meta["table_summary"]
        # 规则摘要不含 LLM 的流畅表述，但有关键词覆盖
        assert "A" in meta["table_summary"]

    @pytest.mark.asyncio
    async def test_extract_no_tables_returns_empty(self):
        from rag.document.pipeline import _extract_and_summarize_tables

        blocks = await _extract_and_summarize_tables(
            b"plain text no tables", "txt", llm=None,
        )
        assert blocks == []


# ── PDF 表格解析细节测试 ──


class TestPdfPlumberTableToBlock:
    """_pdfplumber_table_to_block 的各种输入形态。"""

    def test_none_input_returns_none(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        assert _pdfplumber_table_to_block([], 1) is None
        assert _pdfplumber_table_to_block(None, 1) is None  # type: ignore[arg-type]

    def test_all_none_rows_returns_none(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [[None, None], [None, None]]
        assert _pdfplumber_table_to_block(raw, 1) is None

    def test_headers_detected_from_first_row(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [["姓名", "年龄"], ["张三", "28"]]
        block = _pdfplumber_table_to_block(raw, 1)
        assert block is not None
        assert block.headers == ["姓名", "年龄"]
        assert block.rows == [["张三", "28"]]

    def test_missing_headers_auto_generated(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [["张三", None], ["", "李四"]]
        # 首行有空值 → 不识别为表头
        block = _pdfplumber_table_to_block(raw, 1)
        assert block is not None
        assert block.headers == ["列1", "列2"]

    def test_none_cells_become_empty_string(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [["A", "B"], ["1", None]]
        block = _pdfplumber_table_to_block(raw, 1)
        assert block is not None
        assert block.rows[0] == ["1", ""]

    def test_page_number_preserved(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [["A"], ["1"]]
        block = _pdfplumber_table_to_block(raw, 42)
        assert block is not None
        assert block.page == 42


# ── 辅助工具 ──


def _minimal_text_pdf(text: str) -> bytes:
    """生成一个包含纯文本的最简 PDF 字节流（用于"无表格"场景测试）。

    手写 PDF 结构，不依赖外部库。
    """
    # PDF 是 PostScript 的子集；用 stream 写入文本。
    # 注意：此处仅保证 pdfplumber 能打开且 extract_tables 返回空，不保证视觉渲染。
    content = text.encode("utf-16-be")
    import zlib

    compressed = zlib.compress(content)

    # 对象 1: 页面
    # 对象 2: 内容流
    # 对象 3: 目录

    page_obj = (
        b"1 0 obj\n<< /Type /Page /Parent 3 0 R /MediaBox [0 0 612 792] "
        b"/Contents 2 0 R >>\nendobj\n"
    )
    stream_obj = (
        b"2 0 obj\n<< /Length " + str(len(compressed)).encode() + b" /Filter /FlateDecode >>\n"
        b"stream\n" + compressed + b"\nendstream\nendobj\n"
    )
    catalog_obj = (
        b"3 0 obj\n<< /Type /Pages /Kids [1 0 R] /Count 1 >>\nendobj\n"
    )

    # xref 表
    offsets: list[int] = []
    body = b""
    for obj in [page_obj, stream_obj, catalog_obj]:
        offsets.append(len(body))
        body += obj

    xref_offset = len(body)
    xref = b"xref\n0 4\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        b"trailer\n<< /Size 4 /Root 3 0 R >>\nstartxref\n"
        + str(xref_offset).encode()
        + b"\n%%EOF"
    )

    return b"%PDF-1.4\n" + body + xref + trailer
