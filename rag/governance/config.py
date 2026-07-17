import json
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.common.logging import get_logger

logger = get_logger()


class GovernanceSettings(BaseSettings):
    """LLM 调用治理配置。

    不并入 rag.config.Settings —— 治理关注点独立成模块(同 LangfuseSettings 模式)。
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GOVERNANCE_ENABLED: bool = False

    # ── 限流(按配额桶) ──
    CHAT_RPM_LIMIT: int = 60
    CHAT_MAX_CONCURRENCY: int = 8
    EMBEDDING_RPM_LIMIT: int = 500
    EMBEDDING_MAX_CONCURRENCY: int = 10
    RERANK_RPM_LIMIT: int = 120
    RERANK_MAX_CONCURRENCY: int = 8
    ACQUIRE_MAX_WAIT_SECONDS: float = 10.0
    COOLDOWN_DEFAULT_SECONDS: float = 10.0

    # ── 熔断 ──
    BREAKER_FAILURE_THRESHOLD: int = 5
    BREAKER_COOLDOWN_SECONDS: float = 30.0

    # ── 超时 ──
    LLM_TIMEOUT_SECONDS: float = 60.0
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0
    RERANK_TIMEOUT_SECONDS: float = 30.0

    # ── 价格表:JSON,{"模型名": {"input": 每百万token价, "output": ...}} ──
    LLM_PRICING: str = "{}"

    def limits_for(self, quota: str) -> tuple[int, int]:
        """配额桶 → (rpm_limit, max_concurrency)。"""
        return {
            "chat": (self.CHAT_RPM_LIMIT, self.CHAT_MAX_CONCURRENCY),
            "embedding": (self.EMBEDDING_RPM_LIMIT, self.EMBEDDING_MAX_CONCURRENCY),
            "rerank": (self.RERANK_RPM_LIMIT, self.RERANK_MAX_CONCURRENCY),
        }[quota]

    def pricing(self) -> dict[str, dict[str, float]]:
        """解析价格表 JSON;解析失败降级为空表(成本记 NULL),不阻断启动。"""
        try:
            parsed = json.loads(self.LLM_PRICING)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        logger.warning("LLM_PRICING 不是合法 JSON,成本折算将全部记 NULL")
        return {}


@lru_cache
def get_governance_settings() -> GovernanceSettings:
    return GovernanceSettings()
