"""``rag.__main__`` 的 uvicorn 接线回归测试。

关注点:uvicorn.run() 收到的 log_config 必须是 None,否则 uvicorn 会用它
自己的默认配置执行 logging.config.dictConfig(),整个覆盖掉 setup_logging()
已经装好的、挂了 _SessionContextFilter 的 root handler —— 生产环境日志会
丢失 `| session=<id>` 关联后缀(见 .superpowers/sdd/obs-task-5-report.md
Gap 2 的根因分析)。这里不起真实 uvicorn 服务,只断言接线参数。
"""

import rag.__main__ as main_mod


def test_main_passes_log_config_none_to_uvicorn(monkeypatch):
    calls: dict = {}

    def _fake_run(app, **kwargs):
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr(main_mod.uvicorn, "run", _fake_run)
    monkeypatch.setattr(main_mod, "setup_logging", lambda: None)
    monkeypatch.setattr(main_mod.sys.stderr, "write", lambda *a, **k: None)

    main_mod.main(["--host", "127.0.0.1", "--port", "9999"])

    assert calls["log_config"] is None
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9999
    assert calls["app"] == "rag.api.main:start_app"


def test_main_calls_setup_logging_before_uvicorn_run(monkeypatch):
    """setup_logging() 必须先于 uvicorn.run() 执行 —— root handler(含
    _SessionContextFilter)要在 uvicorn 启动前就位,顺序颠倒的话即便
    log_config=None,也可能出现 uvicorn 内部先输出日志时 handler 尚未挂载。"""
    order: list[str] = []

    def _fake_setup_logging():
        order.append("setup_logging")

    def _fake_run(app, **kwargs):
        order.append("uvicorn.run")

    monkeypatch.setattr(main_mod, "setup_logging", _fake_setup_logging)
    monkeypatch.setattr(main_mod.uvicorn, "run", _fake_run)
    monkeypatch.setattr(main_mod.sys.stderr, "write", lambda *a, **k: None)

    main_mod.main([])

    assert order == ["setup_logging", "uvicorn.run"]
