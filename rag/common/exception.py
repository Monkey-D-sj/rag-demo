class AppError(Exception):
    """应用层领域异常基类:由 API 层 exception handler 统一映射为 HTTP 响应。

    service 只抛此类异常,不依赖 fastapi;status_code 决定最终响应码。
    """

    status_code: int = 500

    def __init__(self, detail: str = "internal error"):
        self.detail = detail
        super().__init__(detail)


class LLMException(Exception):
    """LLM 异常基类"""

    def __init__(self, message: str, status_code: int = 0, model: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.model = model


# ── 可重试异常（5xx / 网络） ──────────────────────────────

class LLMRetryableError(LLMException):
    """可重试的 LLM 异常"""
    pass


class LLMServerError(LLMRetryableError):
    """500 — 服务器内部故障"""
    pass


class LLMServiceUnavailableError(LLMRetryableError):
    """503 — 服务器繁忙"""
    pass


# ── 不可重试异常（4xx） ───────────────────────────────────

class LLMNonRetryableError(LLMException):
    """不可重试的 LLM 异常（客户端错误，重试无意义）"""
    pass


class LLMBadRequestError(LLMNonRetryableError):
    """400 — 请求体格式错误"""
    pass


class LLMAuthenticationError(LLMNonRetryableError):
    """401 — API key 错误或失效"""
    pass


class LLMInsufficientBalanceError(LLMNonRetryableError):
    """402 — 账户余额不足"""
    pass


class LLMInvalidParameterError(LLMNonRetryableError):
    """422 — 请求体参数错误"""
    pass


class LLMRateLimitError(LLMNonRetryableError):
    """429 — 请求速率达到上限，应由调用方限速而非无脑重试"""
    pass


class LLMTimeoutError(LLMRetryableError):
    """请求超时 — 可重试,计入熔断"""
    pass


# ── 治理层拒绝（本地产生，非供应商响应） ──────────────────

class GovernanceRejectedError(LLMNonRetryableError):
    """治理层拒绝基类:限流/熔断在本地拦截,未打到供应商"""
    pass


class RateLimitExceededError(GovernanceRejectedError):
    """本地限流:RPM/并发等待超限或 429 冷却中"""
    pass


class CircuitOpenError(GovernanceRejectedError):
    """熔断打开,快速失败"""
    pass


# ── 用户可见的友好文案（chat SSE error 事件用） ────────────

_FRIENDLY_MESSAGES: list[tuple[type[BaseException], str]] = [
    (RateLimitExceededError, "当前咨询人数较多,请稍后重试"),
    (CircuitOpenError, "AI 服务暂时不可用,请稍后重试"),
    (LLMTimeoutError, "回答生成超时,请重试"),
]

_DEFAULT_FRIENDLY = "服务开小差了,请稍后重试"


def friendly_message(exc: BaseException) -> str:
    """异常 → 用户可见文案。原始异常信息绝不外泄,只进日志。"""
    for exc_type, msg in _FRIENDLY_MESSAGES:
        if isinstance(exc, exc_type):
            return msg
    return _DEFAULT_FRIENDLY


# ── 错误码映射 ──────────────────────────────────────────────

# 可重试：服务端故障
_RETRYABLE_STATUS: dict[int, type[LLMRetryableError]] = {
    500: LLMServerError,
    503: LLMServiceUnavailableError,
}

# 不可重试：客户端错误 + 限流
_NON_RETRYABLE_STATUS: dict[int, type[LLMNonRetryableError]] = {
    400: LLMBadRequestError,
    401: LLMAuthenticationError,
    402: LLMInsufficientBalanceError,
    422: LLMInvalidParameterError,
    429: LLMRateLimitError,
}

# 所有映射的错误码
_STATUS_MAP: dict[int, type[LLMException]] = {
    **_RETRYABLE_STATUS,
    **_NON_RETRYABLE_STATUS,
}


def from_http_error(
    status_code: int,
    message: str,
    model: str = "",
) -> LLMException:
    """根据 HTTP 状态码构造对应的异常"""
    exc_cls = _STATUS_MAP.get(status_code, LLMException)
    return exc_cls(message, status_code=status_code, model=model)


def is_retryable(exc: BaseException) -> bool:
    """判断异常是否可重试（供 tenacity retry 谓词使用）"""
    if isinstance(exc, LLMRetryableError):
        return True
    if isinstance(exc, LLMNonRetryableError):
        return False

    # OpenAI SDK / httpx 级别的网络异常也应当重试
    exc_name = type(exc).__qualname__
    if exc_name in ("APITimeoutError", "APIConnectionError", "InternalServerError"):
        return True
    if exc_name in ("ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError"):
        return True

    return False
