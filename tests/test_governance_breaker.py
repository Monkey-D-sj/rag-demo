import pytest
from fakeredis.aioredis import FakeRedis

from rag.common.exception import CircuitOpenError
from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


def _breaker(redis, **kw) -> RedisCircuitBreaker:
    settings = GovernanceSettings(
        _env_file=None, BREAKER_FAILURE_THRESHOLD=3, BREAKER_COOLDOWN_SECONDS=30, **kw
    )
    return RedisCircuitBreaker(redis, settings)


async def test_closed_allows(redis):
    b = _breaker(redis)
    assert await b.check("chat") is None


async def test_opens_after_consecutive_failures(redis):
    b = _breaker(redis)
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    with pytest.raises(CircuitOpenError):
        await b.check("chat")


async def test_success_resets_failure_streak(redis):
    b = _breaker(redis)
    await b.record_failure("chat", probe=None)
    await b.record_failure("chat", probe=None)
    await b.record_success("chat", probe=None)  # 连续中断,计数清零
    await b.record_failure("chat", probe=None)
    assert await b.check("chat") is None  # 未达阈值,仍闭合


async def test_half_open_single_probe_then_close(redis):
    b = _breaker(redis)
    t = 1000.0
    b._now = lambda: t
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    t += 31  # 冷却期满 → half-open
    assert await b.check("chat") == "probe"  # 第一个请求拿到探针
    with pytest.raises(CircuitOpenError):
        await b.check("chat")  # 第二个请求被拒(探针已被占)
    await b.record_success("chat", probe="probe")  # 探针成功 → 闭合
    assert await b.check("chat") is None


async def test_probe_failure_reopens(redis):
    b = _breaker(redis)
    t = 1000.0
    b._now = lambda: t
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    t += 31
    assert await b.check("chat") == "probe"
    await b.record_failure("chat", probe="probe")  # 探针失败 → 重新 open
    t += 10  # 未满新一轮冷却
    with pytest.raises(CircuitOpenError):
        await b.check("chat")


async def test_release_probe_allows_next_probe(redis):
    b = _breaker(redis)
    t = 1000.0
    b._now = lambda: t
    for _ in range(3):
        await b.record_failure("chat", probe=None)
    t += 31
    assert await b.check("chat") == "probe"
    await b.release_probe("chat")  # 归还探针
    assert await b.check("chat") == "probe"  # 下一个请求可再探


async def test_redis_failure_fails_open():
    class _BrokenRedis:
        def __getattr__(self, name):
            async def _fail(*a, **kw):
                raise ConnectionError("redis down")
            return _fail

    b = _breaker(_BrokenRedis())
    assert await b.check("chat") is None  # fail-open
    await b.record_failure("chat", probe=None)  # 不抛
    await b.record_success("chat", probe=None)  # 不抛
