import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError
from tenacity import wait_none

import rag.models.normal as normal
from rag.common.exception import LLMAuthenticationError
from rag.config import Settings
from rag.models.normal import NormalModel


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    async def ainvoke(self, messages):
        return _FakeMsg("answer")


async def test_ainvoke_returns_content():
    m = NormalModel(Settings())
    m._model = _FakeModel()
    assert await m.ainvoke(["hi"]) == "answer"


class _Out(BaseModel):
    x: int


def _validation_error() -> ValidationError:
    try:
        _Out(x="不是数字")
    except ValidationError as e:
        return e
    raise AssertionError("unreachable")


class _FakeStructured:
    """按序返回预设结果;Exception 项则抛出,None 项模拟模型未调用工具。"""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        item = self._results.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeChatModel:
    def __init__(self, structured):
        self._structured = structured

    def with_structured_output(self, schema, method):
        assert method == "json_mode"
        return self._structured


def _model_with(results):
    m = NormalModel(Settings())
    fake = _FakeStructured(results)
    m._model = _FakeChatModel(fake)
    return m, fake


async def test_ainvoke_structured_returns_schema_instance():
    m, fake = _model_with([_Out(x=1)])
    result = await m.ainvoke_structured(["hi"], _Out)
    assert result == _Out(x=1)
    assert fake.calls == 1


async def test_validation_error_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    m, fake = _model_with([_validation_error(), _Out(x=2)])
    result = await m.ainvoke_structured(["hi"], _Out)
    assert result == _Out(x=2)
    assert fake.calls == 2


async def test_none_result_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    m, fake = _model_with([None, None, None])
    with pytest.raises(OutputParserException):
        await m.ainvoke_structured(["hi"], _Out)
    assert fake.calls == 3


async def test_non_retryable_llm_error_propagates_immediately():
    m, fake = _model_with([LLMAuthenticationError("bad key", status_code=401)])
    with pytest.raises(LLMAuthenticationError):
        await m.ainvoke_structured(["hi"], _Out)
    assert fake.calls == 1


async def test_schema_instruction_merged_into_leading_system_message():
    m, fake = _model_with([_Out(x=1)])
    await m.ainvoke_structured(
        [SystemMessage(content="sys"), HumanMessage(content="hi")], _Out
    )
    msgs = fake.last_messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content.startswith("sys")
    assert "JSON Schema" in msgs[0].content
    assert '"x"' in msgs[0].content  # schema 字段名确实传给了模型
    assert msgs[1].content == "hi"


async def test_schema_instruction_prepended_when_no_system_message():
    m, fake = _model_with([_Out(x=1)])
    await m.ainvoke_structured([HumanMessage(content="hi")], _Out)
    msgs = fake.last_messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage)
    assert "JSON Schema" in msgs[0].content
    assert msgs[1].content == "hi"
