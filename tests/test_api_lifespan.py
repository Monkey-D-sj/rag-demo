import pytest
from fastapi.testclient import TestClient

from rag.api.main import app


@pytest.mark.integration
def test_lifespan_populates_app_state():
    # TestClient 上下文进入即触发 lifespan 启动,退出触发关闭
    with TestClient(app):
        assert app.state.pg is not None
        assert app.state.redis is not None
        assert app.state.memory_manager is not None
        assert app.state.llm is not None
