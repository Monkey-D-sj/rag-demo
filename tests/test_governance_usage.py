import asyncio
from contextlib import asynccontextmanager

import rag.governance.usage as usage_mod
from rag.governance.usage import CallRecord, UsageRecorder


class _FakeCursor:
    def __init__(self):
        self.executed = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))


def _recorder_with_fake_cursor(monkeypatch, pricing=None):
    cur = _FakeCursor()

    @asynccontextmanager
    async def _fake_get_cursor(pool):
        yield cur

    monkeypatch.setattr(usage_mod, "get_cursor", _fake_get_cursor)
    return UsageRecorder(pool=object(), source="api", pricing=pricing or {}), cur


async def test_record_writes_row_with_cost(monkeypatch):
    rec_obj, cur = _recorder_with_fake_cursor(
        monkeypatch, pricing={"m": {"input": 2.0, "output": 8.0}}
    )
    rec_obj.record(CallRecord(
        call_type="chat", model="m", status="success",
        attempts=1, latency_ms=42, input_tokens=1000, output_tokens=500,
    ))
    await asyncio.sleep(0)  # 让 fire-and-forget task 跑完
    await asyncio.sleep(0)
    assert len(cur.executed) == 1
    _, params = cur.executed[0]
    assert params["source"] == "api"
    assert params["cost"] == 0.006
    assert params["status"] == "success"


async def test_write_failure_never_propagates(monkeypatch):
    @asynccontextmanager
    async def _broken_get_cursor(pool):
        raise ConnectionError("pg down")
        yield  # pragma: no cover

    monkeypatch.setattr(usage_mod, "get_cursor", _broken_get_cursor)
    recorder = UsageRecorder(pool=object(), source="api", pricing={})
    recorder.record(CallRecord(
        call_type="chat", model="m", status="failed", attempts=3, latency_ms=1,
    ))
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # 不抛即通过(fail-open)
