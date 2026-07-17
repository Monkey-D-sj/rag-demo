from rag.common.exception import (
    CircuitOpenError,
    LLMTimeoutError,
    RateLimitExceededError,
    friendly_message,
    is_retryable,
)


def test_timeout_is_retryable():
    assert is_retryable(LLMTimeoutError("超时")) is True


def test_governance_rejections_are_non_retryable():
    assert is_retryable(RateLimitExceededError("限流")) is False
    assert is_retryable(CircuitOpenError("熔断")) is False


def test_friendly_message_mapping():
    assert friendly_message(RateLimitExceededError("x")) == "当前咨询人数较多,请稍后重试"
    assert friendly_message(CircuitOpenError("x")) == "AI 服务暂时不可用,请稍后重试"
    assert friendly_message(LLMTimeoutError("x")) == "回答生成超时,请重试"


def test_friendly_message_fallback_hides_raw_error():
    msg = friendly_message(RuntimeError("Traceback: secret internal detail"))
    assert msg == "服务开小差了,请稍后重试"
    assert "secret" not in msg
