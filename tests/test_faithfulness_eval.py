"""Faithfulness 评测单元测试。"""

import json
import tempfile
from pathlib import Path

import pytest

from rag.eval.answer_harness import AnswerGoldenItem, load_answer_golden


class TestLoadAnswerGolden:
    def test_loads_valid_file(self):
        """正常 jsonl 文件应正确解析。"""
        data = [
            {
                "id": "q001",
                "query": "孙悟空出生在哪里？",
                "answer": "花果山山顶仙石。",
                "contexts": ["那座山正当顶上有一块仙石"],
            },
            {
                "id": "q002",
                "query": "金箍棒多重？",
                "answer": "一万三千五百斤。",
                "contexts": ["如意金箍棒重一万三千五百斤"],
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            tmp = f.name

        try:
            items = load_answer_golden(tmp)
            assert len(items) == 2
            assert items[0].id == "q001"
            assert items[0].query == "孙悟空出生在哪里？"
            assert items[0].answer == "花果山山顶仙石。"
            assert items[0].contexts == ["那座山正当顶上有一块仙石"]
            assert isinstance(items[0], AnswerGoldenItem)
        finally:
            Path(tmp).unlink()

    def test_raises_on_missing_query(self):
        """缺少 query 字段应抛 ValueError。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"answer": "x", "contexts": ["y"]}, ensure_ascii=False) + "\n")
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="缺少 query"):
                load_answer_golden(tmp)
        finally:
            Path(tmp).unlink()

    def test_raises_on_empty_contexts(self):
        """contexts 为空列表应抛 ValueError。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    {"id": "q001", "query": "x", "answer": "y", "contexts": []},
                    ensure_ascii=False,
                )
                + "\n"
            )
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="contexts"):
                load_answer_golden(tmp)
        finally:
            Path(tmp).unlink()

    def test_skips_empty_lines(self):
        """空白行应被跳过。"""
        data = [
            {"id": "q001", "query": "x", "answer": "y", "contexts": ["z"]},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("\n")
            f.write(json.dumps(data[0], ensure_ascii=False) + "\n")
            f.write("\n")
            tmp = f.name
        try:
            items = load_answer_golden(tmp)
            assert len(items) == 1
        finally:
            Path(tmp).unlink()
