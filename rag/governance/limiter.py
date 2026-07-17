import time
import uuid

from rag.common.exception import RateLimitExceededError
from rag.common.logging import get_logger
from rag.governance.config import GovernanceSettings

logger = get_logger()

# 相邻两分钟桶加权近似滑动窗口。
# KEYS[1]=当前分钟桶 KEYS[2]=上一分钟桶
# ARGV[1]=limit ARGV[2]=当前分钟内已经过秒数(0-59.999)
# 返回 1=放行(已计数) 0=超限(未计数)
_RPM_LUA = """
local cur = tonumber(redis.call('GET', KEYS[1]) or '0')
local prev = tonumber(redis.call('GET', KEYS[2]) or '0')
local weight = 1 - (tonumber(ARGV[2]) / 60)
if cur + prev * weight >= tonumber(ARGV[1]) then
    return 0
end
local n = redis.call('INCR', KEYS[1])
if n == 1 then
    redis.call('EXPIRE', KEYS[1], 120)
end
return 1
"""

# 并发坑位僵尸阈值下限:超过此秒数视为僵尸(实例崩溃未释放)。
# 实际阈值每个配额桶动态推导为 max(120, quota_timeout * 3)。
# 流式逐 chunk 读超时 60s 意味着活跃流的静默上限约为 timeout,3 倍余量足够。
_STALE_SLOT_SECONDS = 120


class RedisRateLimiter:
    """RPM 滑动窗口 + 并发 ZSET 信号量 + 429 冷却,状态全在 Redis,多实例共享。

    fail-open:Redis 故障时记 warning 放行(acquire 返回 None,release 跳过)。
    """

    def __init__(self, redis, settings: GovernanceSettings) -> None:
        self._redis = redis
        self._settings = settings
        self._now = time.time  # 可注入假时钟,供测试控制窗口滚动

    async def acquire(self, quota: str) -> str | None:
        """获取配额,返回并发坑位 member(release 时传回)。

        超限直接拒绝,由上层重试循环(指数退避)等待。
        """
        rpm_limit, max_conc = self._settings.limits_for(quota)
        try:
            ok, detail = await self._try_acquire(quota, rpm_limit, max_conc)
        except Exception:  # noqa: BLE001 - fail-open:限流器故障不阻断业务
            logger.warning("限流器 Redis 异常,fail-open 放行: %s", quota, exc_info=True)
            return None
        if ok:
            return detail
        raise RateLimitExceededError(f"本地限流({quota}): {detail}", status_code=429)

    async def _try_acquire(
        self, quota: str, rpm_limit: int, max_conc: int
    ) -> tuple[bool, str]:
        """单次尝试。返回 (True, member) 或 (False, 拒因)。"""
        # 1. 429 冷却检查
        if await self._redis.exists(f"gov:cooldown:{quota}"):
            return False, "429 冷却中"
        # 2. 并发坑位:先清僵尸,ZADD 占坑后查总数,超了回滚自己
        now = self._now()
        conc_key = f"gov:conc:{quota}"
        # 按配额超时动态推导僵尸阈值(至少 120s,避免短超时导致活跃流被误清)
        stale_after = max(_STALE_SLOT_SECONDS, self._settings.timeout_for(quota) * 3)
        member = uuid.uuid4().hex
        await self._redis.zremrangebyscore(conc_key, 0, now - stale_after)
        await self._redis.zadd(conc_key, {member: now})
        if await self._redis.zcard(conc_key) > max_conc:
            await self._redis.zrem(conc_key, member)
            return False, "并发已满"
        # 3. RPM 滑动窗口(Lua 原子判定+计数)
        minute = int(now // 60)
        allowed = await self._redis.eval(
            _RPM_LUA,
            2,
            f"gov:rpm:{quota}:{minute}",
            f"gov:rpm:{quota}:{minute - 1}",
            rpm_limit,
            now % 60,
        )
        if not int(allowed):
            await self._redis.zrem(conc_key, member)
            return False, "RPM 超限"
        return True, member

    async def release(self, quota: str, member: str | None) -> None:
        """释放并发坑位。member=None(fail-open 时)跳过。"""
        if member is None:
            return
        try:
            await self._redis.zrem(f"gov:conc:{quota}", member)
        except Exception:  # noqa: BLE001 - 释放失败由僵尸清理兜底
            logger.warning("释放并发坑位失败(僵尸清理会兜底): %s", quota, exc_info=True)

    async def start_cooldown(self, quota: str, seconds: float | None = None) -> None:
        """供应商 429 后开启冷却:所有实例在 TTL 内拒绝发起该桶的调用。"""
        ttl = seconds if seconds and seconds > 0 else self._settings.COOLDOWN_DEFAULT_SECONDS
        try:
            await self._redis.set(f"gov:cooldown:{quota}", "1", ex=max(1, int(ttl)))
        except Exception:  # noqa: BLE001
            logger.warning("写入 429 冷却键失败: %s", quota, exc_info=True)
