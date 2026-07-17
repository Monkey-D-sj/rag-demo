import pytest
from fakeredis.aioredis import FakeRedis

from rag.common.exception import RateLimitExceededError
from rag.governance.config import GovernanceSettings
from rag.governance.limiter import RedisRateLimiter


def _limiter(redis, **kw) -> RedisRateLimiter:
    settings = GovernanceSettings(
        _env_file=None, ACQUIRE_MAX_WAIT_SECONDS=0, **kw
    )  # 等待窗口 0:超限立即拒绝,测试不耗时
    return RedisRateLimiter(redis, settings)


@pytest.fixture
def redis():
    return FakeRedis(decode_responses=True)


async def test_acquire_release_roundtrip(redis):
    lim = _limiter(redis)
    member = await lim.acquire("chat")
    assert member is not None
    assert await redis.zcard("gov:conc:chat") == 1
    await lim.release("chat", member)
    assert await redis.zcard("gov:conc:chat") == 0


async def test_concurrency_limit_rejects(redis):
    lim = _limiter(redis, CHAT_MAX_CONCURRENCY=1)
    await lim.acquire("chat")
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_stale_concurrency_slots_are_evicted(redis):
    lim = _limiter(redis, CHAT_MAX_CONCURRENCY=1)
    t = 1000.0
    lim._now = lambda: t
    await lim.acquire("chat")  # 占坑但不释放(模拟实例崩溃)
    t += 121  # 超过 120s 僵尸阈值
    member = await lim.acquire("chat")  # 僵尸被清,坑位可复用
    assert member is not None


async def test_rpm_limit_rejects(redis):
    lim = _limiter(redis, CHAT_RPM_LIMIT=2)
    lim._now = lambda: 1000.0  # 固定时钟:避免跨分钟桶导致的窗口加权波动
    await lim.release("chat", await lim.acquire("chat"))
    await lim.release("chat", await lim.acquire("chat"))
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_rpm_rejection_does_not_leak_concurrency_slot(redis):
    lim = _limiter(redis, CHAT_RPM_LIMIT=1)
    lim._now = lambda: 1000.0
    await lim.release("chat", await lim.acquire("chat"))
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")
    assert await redis.zcard("gov:conc:chat") == 0


async def test_cooldown_blocks_acquire(redis):
    lim = _limiter(redis)
    await lim.start_cooldown("chat")
    with pytest.raises(RateLimitExceededError):
        await lim.acquire("chat")


async def test_redis_failure_fails_open():
    class _BrokenRedis:
        def __getattr__(self, name):
            async def _fail(*a, **kw):
                raise ConnectionError("redis down")
            return _fail

    lim = _limiter(_BrokenRedis())
    assert await lim.acquire("chat") is None  # fail-open 放行
    await lim.release("chat", None)  # 不抛
