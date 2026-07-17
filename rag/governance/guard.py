import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag.common.exception import (
    GovernanceRejectedError,
    LLMRateLimitError,
    is_retryable,
)
from rag.common.logging import current_session_id, get_logger
from rag.governance.breaker import RedisCircuitBreaker
from rag.governance.config import GovernanceSettings
from rag.governance.limiter import RedisRateLimiter
from rag.governance.usage import CallRecord, UsageRecorder

logger = get_logger()


def _is_provider_429(exc: BaseException) -> bool:
    """供应商侧 429(触发全局冷却)。本地限流拒绝不算(那是自己拒的)。"""
    if isinstance(exc, GovernanceRejectedError):
        return False
    if isinstance(exc, LLMRateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def _retry_after_seconds(exc: BaseException) -> float | None:
    """从异常链提取 Retry-After 头(秒);拿不到返回 None(用默认冷却)。"""
    for e in (exc, exc.__cause__):
        rsp = getattr(e, "response", None)
        headers = getattr(rsp, "headers", None)
        if headers is not None:
            ra = headers.get("retry-after")
            if ra is not None and str(ra).isdigit():
                return float(ra)
    return None


@dataclass
class CallTracker:
    """一次逻辑调用的可变跟踪器:重试循环内累加 attempts,拿到响应后填 tokens。"""

    call_type: str
    model: str
    attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def set_tokens(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMGuard:
    """LLM 调用治理:限流 + 熔断(acquire,每次尝试一次)与统计(track,每次逻辑调用一次)。"""

    def __init__(
        self,
        limiter: RedisRateLimiter,
        breaker: RedisCircuitBreaker,
        recorder: UsageRecorder,
        settings: GovernanceSettings,
    ) -> None:
        self._limiter = limiter
        self._breaker = breaker
        self._recorder = recorder
        self._settings = settings

    @asynccontextmanager
    async def acquire(self, quota: str):
        """包住单次供应商调用:熔断检查 → 限流获取 → 执行 → 结果上报。"""
        probe = await self._breaker.check(quota)  # CircuitOpenError 直接外抛
        try:
            member = await self._limiter.acquire(quota)  # RateLimitExceededError 外抛
        except BaseException:
            if probe:
                # 探针请求没能发出去,归还探针让别人探
                await self._breaker.release_probe(quota)
            raise
        try:
            yield
        except BaseException as e:
            if is_retryable(e):
                # 可重试集合(5xx/网络/超时)与熔断计数集合一致
                try:
                    await self._breaker.record_failure(quota, probe)
                except Exception:  # noqa: BLE001 - 上报失败不阻断原始异常
                    logger.warning("记录熔断失败异常", exc_info=True)
            if _is_provider_429(e):
                try:
                    await self._limiter.start_cooldown(quota, _retry_after_seconds(e))
                except Exception:  # noqa: BLE001 - 上报失败不阻断原始异常
                    logger.warning("记录 429 冷却异常", exc_info=True)
            raise
        else:
            try:
                await self._breaker.record_success(quota, probe)
            except Exception:  # noqa: BLE001 - 上报失败不阻断正常返回
                logger.warning("记录熔断成功异常", exc_info=True)
        finally:
            await self._limiter.release(quota, member)

    @asynccontextmanager
    async def track(self, call_type: str, model: str):
        """包住整个逻辑调用(重试循环之外):计时并生成一行 CallRecord。"""
        tracker = CallTracker(call_type=call_type, model=model)
        status = "success"
        error_type: str | None = None
        t0 = time.perf_counter()
        try:
            yield tracker
        except BaseException as e:
            status = "rejected" if isinstance(e, GovernanceRejectedError) else "failed"
            error_type = type(e).__name__
            raise
        finally:
            try:
                self._recorder.record(CallRecord(
                    call_type=call_type,
                    model=model,
                    status=status,
                    attempts=tracker.attempts,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    session_id=current_session_id(),
                    error_type=error_type,
                    input_tokens=tracker.input_tokens,
                    output_tokens=tracker.output_tokens,
                ))
            except Exception:  # noqa: BLE001 - 统计写失败不阻断调用
                logger.warning("llm_call_log 记录异常", exc_info=True)
