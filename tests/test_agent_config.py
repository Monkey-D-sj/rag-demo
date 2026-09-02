"""Agent 分支配置默认值与必填集合隔离测试。"""

from rag.config import Settings


def test_agent_config_defaults():
    s = Settings()
    assert s.AGENT_MODE_ENABLED is False
    assert s.AGENT_MAX_STEPS == 3
    assert s.AGENT_MAX_TOOL_CALLS_PER_STEP == 4
    assert s.AGENT_MAX_EVIDENCE_CHARS == 24_000
    assert s.AGENT_TOTAL_TIMEOUT_SECONDS == 30


def test_agent_config_not_in_required_fields():
    """agent 开关缺省不阻断 API 启动:不进入 _REQUIRED_FIELDS。"""
    required = set(Settings()._REQUIRED_FIELDS)
    assert "AGENT_MODE_ENABLED" not in required
    assert "AGENT_MAX_STEPS" not in required
    assert "AGENT_MAX_TOOL_CALLS_PER_STEP" not in required
    assert "AGENT_MAX_EVIDENCE_CHARS" not in required
    assert "AGENT_TOTAL_TIMEOUT_SECONDS" not in required
