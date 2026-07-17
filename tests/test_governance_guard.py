import pytest

from rag.common.exception import (
    CircuitOpenError,
    LLMRateLimitError,
    LLMServerError,
    RateLimitExceededError,
)
from rag.governance.config import GovernanceSettings
from rag.governance.guard import CallTracker, LLMGuard
from rag.governance.usage import CallRecord


class _FakeLimiter:
    def __init__(self, reject=False):
        self.reject = reject
        self.released = []
        self.cooldowns = []

    async def acquire(self, quota):
        if self.reject:
            raise RateLimitExceededError("限流")
        return "member-1"

    async def release(self, quota, member):
        self.released.append((quota, member))

    async def start_cooldown(self, quota, seconds=None):
        self.cooldowns.append((quota, seconds))


class _FakeBreaker:
    def __init__(self, probe=None, open_=False):
        self.probe = probe
        self.open_ = open_
        self.successes = []
        self.failures = []
        self.probe_released = []

    async def check(self, quota):
        if self.open_:
            raise CircuitOpenError("熔断")
        return self.probe

    async def record_success(self, quota, probe):
        self.successes.append((quota, probe))

    async def record_failure(self, quota, probe):
        self.failures.append((quota, probe))

    async def release_probe(self, quota):
        self.probe_released.append(quota)


class _FakeRecorder:
    def __init__(self):
        self.records: list[CallRecord] = []

    def record(self, rec):
        self.records.append(rec)


def _guard(limiter=None, breaker=None, recorder=None):
    return LLMGuard(
        limiter=limiter or _FakeLimiter(),
        breaker=breaker or _FakeBreaker(),
        recorder=recorder or _FakeRecorder(),
        settings=GovernanceSettings(_env_file=None),
    )


async def test_acquire_success_reports_breaker_and_releases():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    async with _guard(limiter, breaker).acquire("chat"):
        pass
    assert breaker.successes == [("chat", None)]
    assert limiter.released == [("chat", "member-1")]


async def test_acquire_retryable_failure_counts_to_breaker():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    with pytest.raises(LLMServerError):
        async with _guard(limiter, breaker).acquire("chat"):
            raise LLMServerError("500", status_code=500)
    assert breaker.failures == [("chat", None)]
    assert limiter.released == [("chat", "member-1")]


async def test_provider_429_starts_cooldown_not_breaker():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    with pytest.raises(LLMRateLimitError):
        async with _guard(limiter, breaker).acquire("chat"):
            raise LLMRateLimitError("429", status_code=429)
    assert len(limiter.cooldowns) == 1
    assert breaker.failures == []  # 429 不计入熔断


async def test_limiter_rejection_releases_probe():
    limiter, breaker = _FakeLimiter(reject=True), _FakeBreaker(probe="probe")
    with pytest.raises(RateLimitExceededError):
        async with _guard(limiter, breaker).acquire("chat"):
            pass  # pragma: no cover
    assert breaker.probe_released == ["chat"]


async def test_track_records_success_with_tokens():
    recorder = _FakeRecorder()
    async with _guard(recorder=recorder).track("chat", "m") as t:
        t.attempts += 1
        t.set_tokens(100, 50)
    rec = recorder.records[0]
    assert (rec.status, rec.attempts, rec.input_tokens, rec.output_tokens) == (
        "success", 1, 100, 50,
    )
    assert rec.latency_ms >= 0


async def test_track_records_rejected_status():
    recorder = _FakeRecorder()
    with pytest.raises(CircuitOpenError):
        async with _guard(recorder=recorder).track("chat", "m") as t:
            t.attempts += 1
            raise CircuitOpenError("熔断")
    rec = recorder.records[0]
    assert rec.status == "rejected"
    assert rec.error_type == "CircuitOpenError"


async def test_create_guard_disabled_returns_none(monkeypatch):
    from rag.governance import create_guard
    from rag.governance.config import get_governance_settings

    # 环境变量优先级高于 .env 文件:保证本机 .env 开了 GOVERNANCE_ENABLED 也不影响本测试
    monkeypatch.setenv("GOVERNANCE_ENABLED", "false")
    get_governance_settings.cache_clear()
    try:
        assert create_guard(object(), object(), source="api") is None
    finally:
        get_governance_settings.cache_clear()


async def test_acquire_non_retryable_non_429_only_releases():
    limiter, breaker = _FakeLimiter(), _FakeBreaker()
    with pytest.raises(ValueError):
        async with _guard(limiter, breaker).acquire("chat"):
            raise ValueError("plain error")
    assert breaker.failures == []       # 不可重试异常不计熔断
    assert limiter.cooldowns == []      # 非 429 不触发冷却
    assert limiter.released == [("chat", "member-1")]  # 坑位仍释放


async def test_track_records_failed_status_on_generic_error():
    recorder = _FakeRecorder()
    with pytest.raises(RuntimeError):
        async with _guard(recorder=recorder).track("chat", "m") as t:
            t.attempts += 1
            raise RuntimeError("boom")
    rec = recorder.records[0]
    assert (rec.status, rec.error_type) == ("failed", "RuntimeError")
