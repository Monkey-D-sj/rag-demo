from contextlib import asynccontextmanager

import httpx

from rag.common.exception import CircuitOpenError, LLMServerError
from rag.config import Settings
from rag.governance.guard import CallTracker
from rag.models.rerank import QwenReranker


class _GuardSpy:
    def __init__(self, open_=False):
        self.open_ = open_
        self.acquired: list[str] = []
        self.trackers: list[CallTracker] = []
        self.seen_exceptions: list[Exception] = []

    @asynccontextmanager
    async def acquire(self, quota):
        self.acquired.append(quota)
        if self.open_:
            raise CircuitOpenError("熔断")
        try:
            yield
        except Exception as e:
            # 记录穿过 guard 作用域的异常,验证"翻译发生在 guard 内"(真 guard 靠它计熔断)
            self.seen_exceptions.append(e)
            raise

    @asynccontextmanager
    async def track(self, call_type, model):
        t = CallTracker(call_type=call_type, model=model)
        self.trackers.append(t)
        yield t


def _reranker(guard=None) -> QwenReranker:
    settings = Settings(RERANK_BASE_URL="http://mock-rerank.local", RERANK_KEY="k")
    return QwenReranker(settings, guard=guard)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


async def test_rerank_guard_weave(monkeypatch):
    guard = _GuardSpy()
    r = _reranker(guard=guard)

    async def _fake_post(url, json=None):
        return _FakeResponse({"results": [{"index": 0, "relevance_score": 0.9}]})

    monkeypatch.setattr(r._client, "post", _fake_post)
    chunks = [{"text": "a"}]
    result = await r.rerank("q", chunks)
    assert result[0]["rerank_score"] == 0.9
    assert guard.acquired == ["rerank"]
    assert guard.trackers[0].call_type == "rerank"


async def test_rerank_circuit_open_degrades_to_original_order():
    guard = _GuardSpy(open_=True)
    r = _reranker(guard=guard)
    chunks = [{"text": "a"}, {"text": "b"}]
    assert await r.rerank("q", chunks) == chunks  # 熔断拒绝也降级,不打断 chat


async def test_rerank_5xx_translated_inside_guard(monkeypatch):
    """5xx 在 guard 作用域内被翻译成 LLM 异常(可计入熔断),对外仍降级。"""
    guard = _GuardSpy()
    r = _reranker(guard=guard)

    class _ErrResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500", request=httpx.Request("POST", "http://x"),
                response=httpx.Response(500, request=httpx.Request("POST", "http://x")),
            )

    async def _fake_post(url, json=None):
        return _ErrResponse()

    monkeypatch.setattr(r._client, "post", _fake_post)
    chunks = [{"text": "a"}]
    assert await r.rerank("q", chunks) == chunks  # 降级
    # 穿过 guard 的必须是翻译后的 LLMServerError,而非裸 httpx 异常,否则熔断记不到账
    assert len(guard.seen_exceptions) == 1
    assert isinstance(guard.seen_exceptions[0], LLMServerError)
