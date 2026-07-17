from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings, get_governance_settings
from rag.governance.guard import LLMGuard
from rag.governance.limiter import RedisRateLimiter
from rag.governance.usage import UsageRecorder

__all__ = [
    "GovernanceSettings",
    "LLMGuard",
    "create_guard",
    "get_governance_settings",
]


def create_guard(redis, pool, source: str) -> LLMGuard | None:
    """按配置组装 LLMGuard。GOVERNANCE_ENABLED=false 返回 None(模型层零开销直通)。

    source: api | worker | eval,写入 llm_call_log.source 列。
    """
    settings = get_governance_settings()
    if not settings.GOVERNANCE_ENABLED:
        return None
    return LLMGuard(
        limiter=RedisRateLimiter(redis, settings),
        breaker=RedisCircuitBreaker(redis, settings),
        recorder=UsageRecorder(pool, source, settings.pricing()),
        settings=settings,
    )
