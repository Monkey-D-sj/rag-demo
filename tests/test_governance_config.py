from rag.governance.config import GovernanceSettings
from rag.governance.pricing import compute_cost


def _settings(**kw) -> GovernanceSettings:
    # _env_file=None:隔离本机 .env,只用默认值+显式覆盖
    return GovernanceSettings(_env_file=None, **kw)


def test_defaults_disabled_and_limits():
    s = _settings()
    assert s.GOVERNANCE_ENABLED is False
    assert s.limits_for("chat") == (60, 8)
    assert s.limits_for("embedding") == (500, 10)
    assert s.limits_for("rerank") == (120, 8)


def test_pricing_parses_json():
    s = _settings(LLM_PRICING='{"deepseek-chat": {"input": 2.0, "output": 8.0}}')
    assert s.pricing() == {"deepseek-chat": {"input": 2.0, "output": 8.0}}


def test_pricing_invalid_json_falls_back_to_empty():
    s = _settings(LLM_PRICING="not-json")
    assert s.pricing() == {}


def test_compute_cost_per_million_tokens():
    pricing = {"m": {"input": 2.0, "output": 8.0}}
    # 1000 输入 + 500 输出 → 2*1000/1e6 + 8*500/1e6 = 0.006
    assert compute_cost(pricing, "m", 1000, 500) == 0.006


def test_compute_cost_unknown_model_or_no_tokens_is_none():
    assert compute_cost({}, "unknown", 100, 100) is None
    assert compute_cost({"m": {"input": 1, "output": 1}}, "m", None, None) is None
