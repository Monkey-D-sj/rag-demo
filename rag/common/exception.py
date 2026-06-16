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
