from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

import rag.api.modules.stats.service as stats_service


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))

    async def fetchall(self):
        return self._rows


def _patch_cursor(monkeypatch, rows):
    cur = _FakeCursor(rows)

    @asynccontextmanager
    async def _fake_get_cursor(pool):
        yield cur

    monkeypatch.setattr(stats_service, "get_cursor", _fake_get_cursor)
    return cur


async def test_summary_group_by_day_builds_bucket_sql(monkeypatch):
    cur = _patch_cursor(monkeypatch, [{"bucket": "2026-07-17", "calls": 3}])
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 18, tzinfo=timezone.utc)
    result = await stats_service.llm_summary(object(), start, end, "day")
    sql, params = cur.executed[0]
    assert "date_trunc('day', created_at)" in sql
    assert params == {"start": start, "end": end}
    assert result == {"group_by": "day", "rows": [{"bucket": "2026-07-17", "calls": 3}]}


async def test_summary_group_by_model(monkeypatch):
    cur = _patch_cursor(monkeypatch, [])
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    end = datetime(2026, 7, 18, tzinfo=timezone.utc)
    await stats_service.llm_summary(object(), start, end, "model")
    sql, _ = cur.executed[0]
    assert "GROUP BY bucket" in sql
    assert "model AS bucket" in sql


async def test_summary_rejects_unknown_group_by(monkeypatch):
    _patch_cursor(monkeypatch, [])
    with pytest.raises(ValueError):
        await stats_service.llm_summary(
            object(),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
            "user; DROP TABLE llm_call_log",  # 注入尝试必须被白名单挡下
        )


async def test_recent_clamps_limit(monkeypatch):
    cur = _patch_cursor(monkeypatch, [])
    await stats_service.llm_recent(object(), limit=9999)
    _, params = cur.executed[0]
    assert params == {"limit": 200}  # 上限 200
