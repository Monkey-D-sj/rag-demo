import time

from rag.common.exception import CircuitOpenError
from rag.common.logging import get_logger
from rag.governance.config import GovernanceSettings

logger = get_logger()

# 探针键 TTL:探针方崩溃时最多阻塞 half-open 这么多秒,之后其他实例可再探
_PROBE_TTL_SECONDS = 60


class RedisCircuitBreaker:
    """三态熔断器,状态存 Redis hash(gov:cb:{quota}),多实例共享。

    closed --连续失败≥阈值--> open --冷却期满--> half-open(SET NX 抢探针)
    half-open 探针成功 → closed(清零);探针失败 → open(重计冷却)。
    fail-open:Redis 故障时放行,状态操作失败只记日志。
    """

    def __init__(self, redis, settings: GovernanceSettings) -> None:
        self._redis = redis
        self._settings = settings
        self._now = time.time  # 可注入假时钟

    def _key(self, quota: str) -> str:
        return f"gov:cb:{quota}"

    async def check(self, quota: str) -> str | None:
        """入口检查。None=放行;"probe"=half-open 探针放行;打开则抛 CircuitOpenError。"""
        try:
            data = await self._redis.hgetall(self._key(quota))
        except Exception:  # noqa: BLE001 - fail-open
            logger.warning("熔断器 Redis 异常,fail-open 放行: %s", quota, exc_info=True)
            return None
        if not data or data.get("state") != "open":
            return None
        opened_at = float(data.get("opened_at", 0))
        if self._now() - opened_at < self._settings.BREAKER_COOLDOWN_SECONDS:
            raise CircuitOpenError(f"熔断打开({quota}),快速失败", status_code=503)
        # 冷却期满 → half-open:全集群只放行一个探针
        try:
            got = await self._redis.set(
                f"{self._key(quota)}:probe", "1", nx=True, ex=_PROBE_TTL_SECONDS
            )
        except Exception:  # noqa: BLE001
            logger.warning("抢占熔断探针失败,fail-open 放行: %s", quota, exc_info=True)
            return None
        if got:
            return "probe"
        raise CircuitOpenError(f"熔断半开({quota}),探针已被占用", status_code=503)

    async def record_success(self, quota: str, probe: str | None) -> None:
        """记录成功调用。

        probe=None 时仅清零连续失败计数;若熔断已 open,不改变 state/opened_at,
        熔断仍按冷却期恢复。open 期间的普通成功只可能来自 fail-open 放行的调用,
        清零计数是可接受的。
        """
        try:
            if probe:
                # 探针成功 → 闭合:删状态与探针键
                await self._redis.delete(self._key(quota), f"{self._key(quota)}:probe")
            else:
                # 普通成功 → 连续失败清零
                await self._redis.hset(self._key(quota), "failures", 0)
        except Exception:  # noqa: BLE001
            logger.warning("熔断器记录成功异常: %s", quota, exc_info=True)

    async def record_failure(self, quota: str, probe: str | None) -> None:
        try:
            key = self._key(quota)
            if probe:
                # 探针失败 → 重新 open,重计冷却
                await self._redis.hset(
                    key, mapping={"state": "open", "opened_at": self._now(), "failures": 0}
                )
                await self._redis.delete(f"{key}:probe")
                return
            failures = await self._redis.hincrby(key, "failures", 1)
            if failures >= self._settings.BREAKER_FAILURE_THRESHOLD:
                await self._redis.hset(
                    key, mapping={"state": "open", "opened_at": self._now()}
                )
                logger.warning("熔断器打开: %s(连续失败 %d 次)", quota, failures)
        except Exception:  # noqa: BLE001
            logger.warning("熔断器记录失败异常: %s", quota, exc_info=True)

    async def release_probe(self, quota: str) -> None:
        """探针请求未真正发出(如被限流拒绝)时归还探针,让其他请求可探。"""
        try:
            await self._redis.delete(f"{self._key(quota)}:probe")
        except Exception:  # noqa: BLE001
            logger.warning("归还熔断探针失败: %s", quota, exc_info=True)
