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


# ── 治理层织入 ──────────────────────────────────────────

from contextlib import asynccontextmanager

from rag.common.exception import LLMServerError
from rag.governance.guard import CallTracker


class _FakeGuard:
    """记录 acquire/track 调用的假 guard。"""

    def __init__(self):
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        yield

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


class _FakeMsgWithUsage:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 100, "output_tokens": 50}


class _FakeModelWithUsage:
    async def ainvoke(self, messages):
        return _FakeMsgWithUsage("answer")


async def test_guard_weave_records_quota_and_tokens():
    guard = _FakeGuard()
    m = NormalModel(Settings(), guard=guard)
    m._model = _FakeModelWithUsage()
    assert await m.ainvoke(["hi"]) == "answer"
    assert guard.acquired == ["chat"]
    t = guard.trackers[0]
    assert (t.call_type, t.attempts) == ("chat", 1)
    assert (t.input_tokens, t.output_tokens) == (100, 50)


async def test_guard_weave_retry_acquires_per_attempt(monkeypatch):
    monkeypatch.setattr(normal, "wait_exponential_jitter", lambda **kw: wait_none())
    guard = _FakeGuard()

    class _FailOnceModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise LLMServerError("500", status_code=500)
            return _FakeMsgWithUsage("ok")

    m = NormalModel(Settings(), guard=guard)
    m._model = _FailOnceModel()
    assert await m.ainvoke(["hi"]) == "ok"
    assert guard.acquired == ["chat", "chat"]  # 每次尝试各过一次 guard
    assert guard.trackers[0].attempts == 2


async def test_astream_weave_uses_chat_stream_call_type():
    guard = _FakeGuard()

    class _StreamOnly:
        def astream(self, messages):
            async def _gen():
                yield _FakeMsgWithUsage("tok")
            return _gen()

    m = NormalModel(Settings(), guard=guard)
    m._model = _StreamOnly()
    tokens = [c.content async for c in m.astream(["hi"])]
    assert tokens == ["tok"]
    t = guard.trackers[0]
    assert t.call_type == "chat_stream"
    assert (t.input_tokens, t.output_tokens) == (100, 50)
