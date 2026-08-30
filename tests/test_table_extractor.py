"""表格提取模块的单元测试。

纯函数直接测；PDF / DOCX 集成测试用程序化生成的真实文件。
"""
from __future__ import annotations

import io


# ── 列名+值 拼接 ──


class TestFlattenTable:
    """列名+值 拼接生成（不依赖 LLM）。"""

    def test_full_table(self):
        from rag.document.table_extractor import _flatten_table

        s = _flatten_table(
            ["姓名", "年龄"],
            [["张三", "28"], ["李四", "35"]],
            caption="员工信息表",
        )
        assert "员工信息表" in s
        assert "姓名" in s
        assert "年龄" in s
        assert "张三" in s

    def test_all_rows_included_no_truncation(self):
        from rag.document.table_extractor import _flatten_table

        rows = [[str(i), f"val{i}"] for i in range(100)]
        s = _flatten_table(["序号", "值"], rows)
        # 末行保留,无折叠提示
        assert "值：val99" in s
        assert "仅展示前" not in s

    def test_all_empty_returns_placeholder(self):
        from rag.document.table_extractor import _flatten_table

        s = _flatten_table([], [])
        assert s == "表格数据"


# ── TableBlock 数据模型 ──


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
        assert "员工表" in meta["table_summary"]

    def test_all_rows_kept_no_truncation(self):
        from rag.document.table_extractor import TableBlock

        rows = [[str(i)] for i in range(100)]
        tb = TableBlock(["序号"], rows)
        # 不做行数截断,末行保留在嵌入文本
        assert "序号：99" in tb.embed_text

    def test_page_stored(self):
        from rag.document.table_extractor import TableBlock

        tb = TableBlock(["A"], [["1"]], page=5)
        assert tb.page == 5
        # 默认不传 page
        assert TableBlock(["A"], [["1"]]).page is None


# ── DOCX 表格提取 ──


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

        buf = io.BytesIO()
        doc.save(buf)
        data = buf.getvalue()

        tables = extract_tables(data, "docx")
        assert tables == []


# ── PDF / 非表格类型 ──


class TestExtractTablesFromPdf:
    """PDF 表格提取及非表格类型降级。"""

    def test_pdf_without_tables(self):
        from rag.document.table_extractor import extract_tables

        pdf_bytes = _minimal_text_pdf("这是一段纯文本，没有表格。")
        tables = extract_tables(pdf_bytes, "pdf")
        assert tables == []

    def test_non_table_types_return_empty(self):
        from rag.document.table_extractor import extract_tables

        assert extract_tables(b"hello world", "txt") == []
        assert extract_tables(b"# Title\n\n| a | b |", "md") == []
        assert extract_tables(b"data", "xlsx") == []

    def test_corrupt_pdf_does_not_crash(self):
        from rag.document.table_extractor import extract_tables

        tables = extract_tables(b"this is not a valid PDF", "pdf")
        assert isinstance(tables, list)


# ── 管线层表格提取（降级） ──


class TestTableExtract:
    """pipeline 表格提取返回 TableBlock,meta 携带 列名+值 拼接文本。"""

    async def test_extract_returns_flattened_summary(self):
        from docx import Document

        from rag.document.pipeline import _extract_tables

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

        blocks = await _extract_tables(buf.getvalue(), "docx")
        assert len(blocks) == 1
        meta = blocks[0].meta
        assert "产品" in meta["table_summary"]
        assert "销量" in meta["table_summary"]
        assert "A" in meta["table_summary"]

    async def test_extract_no_tables_returns_empty(self):
        from rag.document.pipeline import _extract_tables

        blocks = await _extract_tables(b"plain text no tables", "txt")
        assert blocks == []


# ── PDF plumber 解析细节 ──


class TestPdfPlumberTableToBlock:
    """_pdfplumber_table_to_block 的各种输入形态。"""

    def test_empty_or_none_returns_none(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        assert _pdfplumber_table_to_block([], 1) is None
        assert _pdfplumber_table_to_block([[None, None], [None, None]], 1) is None

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
        block = _pdfplumber_table_to_block(raw, 1)
        assert block is not None
        assert block.headers == ["列1", "列2"]

    def test_none_cells_become_empty_string(self):
        from rag.document.table_extractor import _pdfplumber_table_to_block

        raw = [["A", "B"], ["1", None]]
        block = _pdfplumber_table_to_block(raw, 1)
        assert block is not None
        assert block.rows[0] == ["1", ""]


# ── 辅助工具 ──


def _minimal_text_pdf(text: str) -> bytes:
    """生成一个包含纯文本的最简 PDF 字节流（用于"无表格"场景测试）。"""
    content = text.encode("utf-16-be")
    import zlib
    compressed = zlib.compress(content)

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
