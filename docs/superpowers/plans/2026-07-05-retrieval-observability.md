# 检索可观测性（结构化日志 + Langfuse）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检索链路获得结构化日志（extra 字段 + session_id 关联）与 Langfuse 自托管追踪（LangGraph 回调 + 检索 span），评测留待后续。

**Architecture:** Layer 1 增强现有 `rag/common/logging.py`（JsonFormatter 输出 extra、contextvars 注入 session_id）并在 retriever/recall/chat 入口埋点；Layer 2 新增独立 `rag/observability/langfuse.py`（自带 Settings + handler 工厂 + 条件装饰器，绕开并行会话正在改的 `config.py`），docker-compose 添加 langfuse 自托管栈（headless init 预置 key），最后在 `service.py`/`workflow.py` 接线。

**Tech Stack:** Python 3.12+ / stdlib logging + contextvars / pydantic-settings / langfuse v3 SDK / docker compose / pytest (asyncio_mode=auto) / uv

**Spec:** `docs/superpowers/specs/2026-07-05-retrieval-observability-design.md`

## Global Constraints

- 所有命令用 `uv run` 前缀执行（uv 管理的项目）。
- pytest `asyncio_mode = "auto"`：async 测试不需要装饰器。
- 除 Task 4/5 的真实验证外，测试不触网、不依赖 docker/真实 key。
- 中文注释风格与现有代码一致（半角逗号，注释解释「为什么」）。
- 分支基线预先存在 18 个测试失败（test_config / test_logging / test_document_controller / test_document_pipeline / test_document_store / test_migration），「全量通过」= 不引入新失败。
- 并行会话脏文件**不得 add/改/stash**：`rag/config.py`、`rag/agent/workflow.py`、`rag/agent/nodes/generate/generate.py`、`rag/document/store.py`、`rag/graph/pipeline.py`、`rag/memory/adapters/long_term_pgsql.py`（执行时以 `git status` 实况为准；Task 5 对 workflow.py 有专门规程）。
- 每次提交只 `git add` 本任务明确列出的文件。
- Commit message 末尾加：`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: logging.py 增强 —— JsonFormatter extra 输出 + session_id 上下文关联

**Files:**
- Modify: `rag/common/logging.py`
- Test: `tests/test_logging.py`（追加；该文件已有的失败是分支基线问题，新增测试不依赖全局 setup_logging 状态）

**Interfaces:**
- Consumes: 无。
- Produces: `bind_session(session_id: str) -> contextvars.Token`、`reset_session(token) -> None`（Task 2 的 service.py 使用）；JsonFormatter 自动输出 `logger.info(..., extra={...})` 的自定义字段（Task 2 埋点依赖）；`_SessionContextFilter` 挂在 setup_logging 的两个 handler 上。

- [ ] **Step 1: 写失败测试**

在 `tests/test_logging.py` 顶部补充缺失的 import（与现有 import 合并，勿重复）：

```python
import datetime
import json
import logging

from rag.common.logging import (
    ColorTextFormatter,
    JsonFormatter,
    _SessionContextFilter,
    bind_session,
    reset_session,
)
```

文件末尾追加：

```python
def _make_record(msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        "rag.test", logging.INFO, __file__, 1, msg, (), None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_includes_extra_fields():
    out = json.loads(JsonFormatter().format(_make_record(kb_id="kb1", top_k=5)))
    assert out["kb_id"] == "kb1"
    assert out["top_k"] == 5


def test_json_formatter_serializes_non_json_values_via_str():
    rec = _make_record(when=datetime.datetime(2026, 7, 5, 12, 0, 0))
    out = json.loads(JsonFormatter().format(rec))
    assert "2026-07-05" in out["when"]


def test_json_formatter_does_not_leak_std_attrs():
    out = json.loads(JsonFormatter().format(_make_record()))
    assert "args" not in out
    assert "lineno" not in out
    assert "levelno" not in out


def test_session_filter_injects_and_resets():
    f = _SessionContextFilter()
    token = bind_session("s-1")
    try:
        record = _make_record()
        assert f.filter(record) is True
        assert record.session_id == "s-1"
    finally:
        reset_session(token)
    record2 = _make_record()
    f.filter(record2)
    assert not hasattr(record2, "session_id")


def test_session_filter_keeps_explicit_extra():
    f = _SessionContextFilter()
    token = bind_session("ctx-session")
    try:
        record = _make_record(session_id="explicit")
        f.filter(record)
        assert record.session_id == "explicit"
    finally:
        reset_session(token)


def test_text_formatter_appends_session_suffix():
    line = ColorTextFormatter(use_color=False).format(
        _make_record(session_id="s-9")
    )
    assert line.endswith("| session=s-9")


def test_text_formatter_no_suffix_without_session():
    line = ColorTextFormatter(use_color=False).format(_make_record())
    assert "session=" not in line
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_logging.py -v -k "extra or session or suffix"`
Expected: FAIL —— import 报 `cannot import name '_SessionContextFilter'`（及 `bind_session`）。

- [ ] **Step 3: 实现**

`rag/common/logging.py`：

1. 顶部 import 增加 `import contextvars`。
2. `get_logger` 之后新增（模块级）：

```python
# ── session 关联:contextvars 贯穿单次请求内的所有日志 ──

_session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "log_session_id", default=None
)


def bind_session(session_id: str) -> contextvars.Token:
    """绑定当前上下文的 session_id,返回 token 供 reset_session 恢复。"""
    return _session_id_var.set(session_id)


def reset_session(token: contextvars.Token) -> None:
    _session_id_var.reset(token)


class _SessionContextFilter(logging.Filter):
    """把 contextvar 中的 session_id 注入日志记录;显式 extra 优先。"""

    def filter(self, record: logging.LogRecord) -> bool:
        sid = _session_id_var.get()
        if sid and not hasattr(record, "session_id"):
            record.session_id = sid
        return True
```

3. `JsonFormatter` 前新增标准属性集合，并替换 `format` 方法：

```python
# LogRecord 标准属性集合;record.__dict__ 中此外的键视为 extra 结构化字段。
# taskName 是 3.12 asyncio 加的,message/asctime 由 Formatter 动态注入。
_STD_RECORD_KEYS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """每行一个 JSON 的日志格式器（纯标准库）;extra 字段自动并入输出。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created).isoformat()
        data: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_KEYS and key not in data:
                data[key] = value
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        # default=str 兜底不可序列化值(datetime/UUID 等),日志不因序列化炸掉
        return json.dumps(data, ensure_ascii=False, default=str)
```

4. `ColorTextFormatter.format` 末尾追加 session 后缀——整个方法替换为：

```python
    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            if color:
                # 复制以免污染原始 record（影响其他 handler）
                record = logging.makeLogRecord(record.__dict__)
                visible = record.levelname
                pad = max(0, 8 - len(visible))
                record.levelname = f"{color}{visible}{_RESET}{' ' * pad}"
        line = super().format(record)
        # 文本格式不输出全量 extra(避免终端刷屏),仅追加关联 ID
        sid = getattr(record, "session_id", None)
        if sid:
            line = f"{line} | session={sid}"
        return line
```

5. `setup_logging` 中给两个 handler 挂 Filter：`stream_handler.addFilter(_NameRewriter())` 之后加一行 `stream_handler.addFilter(_SessionContextFilter())`；文件 handler 的 `file_handler.setFormatter(...)` 之前加 `file_handler.addFilter(_SessionContextFilter())`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_logging.py -v -k "extra or session or suffix"`
Expected: 7 个新测试全部 PASS。再跑 `uv run pytest tests/test_logging.py -v` 确认该文件失败数量不多于基线（基线 5 个失败）。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 失败集合与基线一致（18 个），无新增。

```bash
git add rag/common/logging.py tests/test_logging.py
git commit -m "feat: 日志支持 extra 结构化字段与 session_id 上下文关联"
```

---

### Task 2: 检索链路埋点 —— retriever 计时与命中日志、recall 查询来源、chat 入口绑定

**Files:**
- Modify: `rag/document/retriever.py`
- Modify: `rag/agent/nodes/recall/recall.py`
- Modify: `rag/api/modules/chat/service.py`
- Test: `tests/test_retriever.py`（新建）

**Interfaces:**
- Consumes: Task 1 的 `bind_session` / `reset_session`；`store.search_chunks` 返回列 `id, document_id, chunk_index, text, similarity`。
- Produces: `KnowledgeRetriever.search` 签名与返回值不变（Task 5 在其上加装饰器）。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_retriever.py`：

```python
import logging

from rag.document.retriever import KnowledgeRetriever


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


def _retriever(monkeypatch, rows):
    import rag.document.retriever as mod

    async def fake_search_chunks(pool, embedding, kb_id, top_k):
        return rows

    monkeypatch.setattr(mod.store, "search_chunks", fake_search_chunks)
    return KnowledgeRetriever(None, _FakeEmbedding())


async def test_search_logs_hits_with_structured_fields(monkeypatch, caplog):
    rows = [
        {
            "id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "text": "花果山",
            "similarity": 0.87654,
        }
    ]
    r = _retriever(monkeypatch, rows)
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("孙悟空是谁", "kb-1")
    assert result == rows
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert rec.levelno == logging.INFO
    assert rec.kb_id == "kb-1"
    assert rec.query == "孙悟空是谁"
    assert rec.top_k == 5
    assert rec.hits == [
        {"chunk_id": "c1", "chunk_index": 0, "similarity": 0.8765}
    ]
    assert rec.embed_ms >= 0
    assert rec.search_ms >= 0


async def test_search_empty_result_logs_warning(monkeypatch, caplog):
    r = _retriever(monkeypatch, [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("无关问题", "kb-1")
    assert result == []
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert rec.levelno == logging.WARNING
    assert rec.hits == []


async def test_search_truncates_long_query_in_log(monkeypatch, caplog):
    r = _retriever(monkeypatch, [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("长" * 300, "kb-1")
    rec = next(x for x in caplog.records if "向量召回" in x.getMessage())
    assert len(rec.query) == 200
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL —— caplog 中找不到「向量召回」记录（`StopIteration`），当前 search 无日志。

- [ ] **Step 3: 实现**

`rag/document/retriever.py` 全文替换为：

```python
import time

from rag.common.logging import get_logger
from rag.document import store
from rag.models.embedding import EmbeddingModel

logger = get_logger()


class KnowledgeRetriever:
    """知识库检索:embed 查询 → pgvector 相似度召回 document_chunks。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
        t0 = time.perf_counter()
        query_embedding = (await self._embedding.embed([query]))[0]
        embed_ms = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        rows = await store.search_chunks(
            self._pool, query_embedding, knowledge_base_id, top_k
        )
        search_ms = round((time.perf_counter() - t1) * 1000, 1)

        fields = {
            "kb_id": knowledge_base_id,
            "query": query[:200],  # 截断,长文本进日志没有意义
            "top_k": top_k,
            "hits": [
                {
                    "chunk_id": str(r["id"]),
                    "chunk_index": r["chunk_index"],
                    "similarity": round(r["similarity"], 4),
                }
                for r in rows
            ],
            "embed_ms": embed_ms,
            "search_ms": search_ms,
        }
        if rows:
            logger.info("向量召回完成: %d 条", len(rows), extra=fields)
        else:
            # 空结果是检索质量最直接的信号,升级 WARNING
            logger.warning("向量召回为空", extra=fields)
        return rows
```

`rag/agent/nodes/recall/recall.py` 全文替换为：

```python
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.common.logging import get_logger
from rag.document import DEFAULT_KB_ID
from rag.agent.type import ContextSchema, MyState

logger = get_logger()


async def recall(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从知识库召回相关 chunk(向量检索),写入 recall_vec_results。"""
    writer = get_stream_writer()
    writer({"type": "status", "data": "检索知识库中..."})

    retriever = runtime.context.retriever
    if retriever is None:
        logger.warning("retriever 未注入,跳过知识库召回")
        state["recall_vec_results"] = []
        return state

    # 改写后的查询更适合检索,缺失时回退原始查询
    query = state.get("rewrite_query") or state["raw_query"]
    # rewrite 节点当前被禁用,此日志持续暴露「改写从未生效」的事实
    logger.info(
        "知识库召回使用%s查询",
        "改写后" if state.get("rewrite_query") else "原始",
        extra={"query": query[:200]},
    )
    state["recall_vec_results"] = await retriever.search(query, DEFAULT_KB_ID)
    return state
```

`rag/api/modules/chat/service.py`：import 增加 `from rag.common.logging import bind_session, get_logger, reset_session`（替换原有 get_logger import 行），`stream_chat` 函数体首行绑定、末尾重置——把现有函数体包进 try/finally：

```python
    token = bind_session(session_id)
    try:
        context = ContextSchema(
            llm=llm, memory_manager=memory_manager, retriever=retriever
        )
        stream = ChatStream()

        async def _produce() -> None:
            try:
                async for event in invoke(session_id, query, context):
                    await stream.send_event(event)
            except Exception as e:  # noqa: BLE001
                logger.exception("chat stream failed")
                await stream.error(str(e))
            finally:
                stream.close()

        task = asyncio.create_task(_produce())

        async for sse_line in stream:
            yield sse_line

        await task  # 确保生产者异常不被静默吞掉
    finally:
        reset_session(token)
```

注意：`_produce` 在独立 task 中运行，`asyncio.create_task` 会拷贝当前 context，session_id 会正确传播到后台协程。若执行时 `service.py` 内容与上述基线不一致（并行会话可能又改过），保持其现有逻辑不变，只做「首行 bind + finally reset」的包裹。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retriever.py tests/test_nodes.py tests/test_chat_controller.py -v`
Expected: test_retriever 3 个 PASS；test_nodes / test_chat_controller 不劣于基线。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 失败集合与基线一致。

```bash
git add rag/document/retriever.py rag/agent/nodes/recall/recall.py rag/api/modules/chat/service.py tests/test_retriever.py
git commit -m "feat: 检索链路结构化日志埋点(召回命中/耗时/查询来源/session 关联)"
```

---

### Task 3: `rag/observability/langfuse.py` —— 独立配置 + handler 工厂 + 条件装饰器

**Files:**
- Create: `rag/observability/__init__.py`（空文件）
- Create: `rag/observability/langfuse.py`
- Modify: `pyproject.toml`（dependencies 增加 `"langfuse>=3.0",`）
- Test: `tests/test_observability.py`（新建）

**Interfaces:**
- Consumes: 无（配置独立，明确不进 `rag/config.py`）。
- Produces: `get_callback_handler() -> Any | None`、`observe_if_enabled(name: str) -> Callable`（Task 5 使用）；env 变量 `LANGFUSE_ENABLED`（默认 false）、`LANGFUSE_HOST`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`。

- [ ] **Step 1: 加依赖**

`pyproject.toml` 的 `dependencies` 列表末尾（`"neo4j>=5.28",` 之后）加 `"langfuse>=3.0",`，然后：

Run: `uv sync`
Expected: langfuse 及其依赖安装成功。若安装的主版本不是 3.x，停下报告（后续 API 按 v3 写）。

- [ ] **Step 2: 写失败测试**

新建 `tests/test_observability.py`：

```python
import pytest

import rag.observability.langfuse as ob


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    ob.get_langfuse_settings.cache_clear()
    yield
    ob.get_langfuse_settings.cache_clear()


def test_handler_none_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert ob.get_callback_handler() is None


def test_handler_none_when_enabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    assert ob.get_callback_handler() is None


def test_observe_passthrough_when_disabled(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    async def fn(x):
        return x + 1

    assert ob.observe_if_enabled("t")(fn) is fn
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_observability.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'rag.observability'`。

- [ ] **Step 4: 实现**

新建空的 `rag/observability/__init__.py`。新建 `rag/observability/langfuse.py`：

```python
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.common.logging import get_logger

logger = get_logger()


class LangfuseSettings(BaseSettings):
    """Langfuse 独立配置。

    不并入 rag.config.Settings:可观测性关注点独立成模块,也避免与并行改动的
    config.py 冲突(见设计文档)。变量名与 langfuse SDK 原生环境变量同名。
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LANGFUSE_ENABLED: bool = False
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""


@lru_cache
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()


def get_callback_handler() -> Any | None:
    """开启时返回 LangChain CallbackHandler,关闭时返回 None(零开销)。

    凭据可能只写在 .env 而未导出为进程环境变量,SDK 读不到,
    所以这里显式初始化全局客户端而不是依赖 SDK 自读环境。
    """
    settings = get_langfuse_settings()
    if not settings.LANGFUSE_ENABLED:
        return None
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning("LANGFUSE_ENABLED=true 但缺少 key,追踪已跳过")
        return None
    # 延迟导入:关闭时不加载 SDK
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
    return CallbackHandler()


def observe_if_enabled(name: str) -> Callable:
    """开启时用 langfuse.observe 包装函数,关闭时原样返回(零包装开销)。

    装饰器在 import 时求值,LANGFUSE_ENABLED 需在进程启动前设定。
    """
    settings = get_langfuse_settings()
    if not settings.LANGFUSE_ENABLED:
        return lambda fn: fn
    from langfuse import observe

    return observe(name=name)
```

若安装的 langfuse v3 中 `CallbackHandler` 的 import 路径或 `Langfuse(...)` 初始化方式与上述不符，以 `uv run python -c "import langfuse; help(...)"` 与官方文档为准调整，**保持两个公开函数的签名与关闭时行为不变**，并在报告中说明差异。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_observability.py -v`
Expected: 3 个 PASS。

- [ ] **Step 6: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 失败集合与基线一致。

```bash
git add rag/observability/__init__.py rag/observability/langfuse.py pyproject.toml uv.lock tests/test_observability.py
git commit -m "feat: Langfuse 可观测性模块(独立配置+handler 工厂+条件装饰器)"
```

---

### Task 4: docker-compose 添加 Langfuse 自托管栈并验证启动

**Files:**
- Modify: `docker-compose.yaml`
- Modify: `.env.example`（追加 LANGFUSE_* 示例）
- Modify: `.env`（本地追加实际值，**不提交**）

**Interfaces:**
- Consumes: 现有 compose 服务 `redis`、`minio`（网络内主机名即服务名）。
- Produces: `http://localhost:3000` 的 Langfuse 实例，API key 由 headless init 预置为 `.env` 中的 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`（Task 5 smoke 依赖）。

**这是一个需要现场判断的任务**：镜像 tag、env 变量名可能随 Langfuse 版本演进，以下 YAML 是按 v3 官方 self-host compose 改写的基线；启动报错时对照官方文档（https://langfuse.com/self-hosting/docker-compose）调整，验收标准不变。

- [ ] **Step 1: 前置检查**

Run: `docker info` —— Docker 不可用则直接报告 BLOCKED。
Run: `git status --short docker-compose.yaml` —— 若该文件有未提交改动（并行会话），报告 BLOCKED。

- [ ] **Step 2: 写 compose 服务**

`docker-compose.yaml` 的 `services:` 下追加（`neo4j` 之后）：

```yaml
  langfuse-postgres:
    image: postgres:16-alpine
    container_name: rag-langfuse-postgres
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse123
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_pg_data:/var/lib/postgresql/data
    restart: unless-stopped

  clickhouse:
    image: clickhouse/clickhouse-server:24.12-alpine
    container_name: rag-clickhouse
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse123
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    restart: unless-stopped

  minio-init-langfuse:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "mc alias set local http://minio:9000 minioadmin minioadmin
      && mc mb --ignore-existing local/langfuse"
    restart: "no"

  langfuse-worker:
    image: langfuse/langfuse-worker:3
    container_name: rag-langfuse-worker
    depends_on:
      - langfuse-postgres
      - clickhouse
      - redis
      - minio
    environment: &langfuse_env
      DATABASE_URL: postgresql://langfuse:langfuse123@langfuse-postgres:5432/langfuse
      SALT: rag-demo-salt-not-for-prod
      ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000"
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse123
      REDIS_CONNECTION_STRING: redis://redis:6379/1
      LANGFUSE_S3_EVENT_UPLOAD_ENABLED: "true"
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_REGION: auto
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: minioadmin
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: minioadmin
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
    restart: unless-stopped

  langfuse-web:
    image: langfuse/langfuse:3
    container_name: rag-langfuse-web
    depends_on:
      - langfuse-postgres
      - clickhouse
      - redis
      - minio
    ports:
      - "3000:3000"
    environment:
      <<: *langfuse_env
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: rag-demo-nextauth-secret
      # headless init:预置组织/项目/key,全程无需人工点 UI
      LANGFUSE_INIT_ORG_ID: rag-demo
      LANGFUSE_INIT_ORG_NAME: RAG Demo
      LANGFUSE_INIT_PROJECT_ID: rag-demo-project
      LANGFUSE_INIT_PROJECT_NAME: rag-demo
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: pk-lf-rag-demo-local
      LANGFUSE_INIT_PROJECT_SECRET_KEY: sk-lf-rag-demo-local
      LANGFUSE_INIT_USER_EMAIL: admin@example.com
      LANGFUSE_INIT_USER_NAME: admin
      LANGFUSE_INIT_USER_PASSWORD: admin12345
    restart: unless-stopped
```

`volumes:` 下追加：

```yaml
  langfuse_pg_data:
  clickhouse_data:
```

- [ ] **Step 3: 配置文件**

`.env.example` 末尾追加：

```
# ── Langfuse(可观测性,默认关闭) ──
LANGFUSE_ENABLED=false
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-rag-demo-local
LANGFUSE_SECRET_KEY=sk-lf-rag-demo-local
```

`.env` 末尾追加同样 4 行但 `LANGFUSE_ENABLED=true`（本地开启；`.env` 已被 gitignore，确认 `git status` 不出现它）。

- [ ] **Step 4: 启动并验收**

```bash
docker compose up -d langfuse-postgres clickhouse minio-init-langfuse langfuse-worker langfuse-web
```

验收（全部满足才算过）：
1. `docker compose ps` —— langfuse-web / langfuse-worker / clickhouse / langfuse-postgres 均 running（web 首次迁移可能需 1-2 分钟，可轮询）。
2. `curl -s http://localhost:3000/api/public/health` 返回 JSON 且无 error。
3. `curl -s -u pk-lf-rag-demo-local:sk-lf-rag-demo-local http://localhost:3000/api/public/projects` 返回含 `rag-demo` 的项目列表（验证 headless init 生效）。
4. `docker logs rag-langfuse-worker --tail 20` 无 crash-loop。

失败时看 `docker logs` 对照官方 compose 调整 env/镜像，重试；连续无法解决则报告 BLOCKED 附完整错误。

- [ ] **Step 5: 提交**

Run: `uv run pytest -q`（确认无代码改动引入失败——本任务只动 compose 与配置）

```bash
git add docker-compose.yaml .env.example
git commit -m "feat: docker-compose 添加 Langfuse 自托管栈(headless init 预置 key)"
```

---

### Task 5: 接线与端到端验证 —— workflow 回调透传、service 组装、retriever span

**Files:**
- Modify: `rag/agent/workflow.py`（⚠️ 冲突敏感，见 Step 0）
- Modify: `rag/api/modules/chat/service.py`
- Modify: `rag/document/retriever.py`（加装饰器一行 + import）
- Test: `tests/test_workflow.py`（追加）

**Interfaces:**
- Consumes: Task 3 的 `get_callback_handler` / `observe_if_enabled`；Task 4 的运行中 Langfuse 实例与 `.env` key；Task 2 的 service.py 现状。
- Produces: `invoke(session_id, query, context, config: dict | None = None)`。

- [ ] **Step 0: 冲突规程（必须先做）**

Run: `git status --short rag/agent/workflow.py rag/api/modules/chat/service.py rag/document/retriever.py`

- 三个文件都干净 → 正常执行全部 Step 并提交。
- `workflow.py` 有未提交改动（并行会话）→ **本任务所有代码改动照做、照测，但一个文件都不提交**（避免提交后 HEAD 上 service 调新签名而 workflow 还是旧签名），报告 DONE_WITH_CONCERNS 说明「改动留在工作区待并行会话落地后统一提交」。
- 其他文件脏 → 同上原则：要么全提交要么全不提交，不允许部分提交造成 HEAD 不自洽。

- [ ] **Step 1: 写失败测试**

`tests/test_workflow.py` 追加：

```python
async def test_invoke_passes_config_to_astream(monkeypatch):
    import rag.agent.workflow as wf

    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover - 使函数成为异步生成器
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None, config={"callbacks": []}):
        pass
    assert captured["config"] == {"callbacks": []}


async def test_invoke_config_defaults_none(monkeypatch):
    import rag.agent.workflow as wf

    captured = {}

    async def fake_astream(input, *, context, stream_mode, config=None):
        captured["config"] = config
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(wf.graph, "astream", fake_astream)
    async for _ in wf.invoke("s-1", "q", context=None):
        pass
    assert captured["config"] is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_workflow.py -v -k config`
Expected: FAIL —— `invoke() got an unexpected keyword argument 'config'`。

- [ ] **Step 3: 实现**

`rag/agent/workflow.py::invoke` 改为（若并行会话已改动此函数，保持其逻辑，只加 `config` 参数并透传）：

```python
async def invoke(
    session_id: str,
    query: str,
    context: ContextSchema,
    config: dict | None = None,
):
    """归一化事件流:仅保留 custom 通道事件(status/message/error),
    updates 通道(state 增量)不再下发。config 用于透传 LangChain 回调(如 Langfuse)。
    """
    async for mode, chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
```

`rag/api/modules/chat/service.py`：import 增加 `from rag.observability.langfuse import get_callback_handler`；`stream_chat` 中构建 context 之后、创建 ChatStream 之前加：

```python
        handler = get_callback_handler()
        # metadata 里的 langfuse_session_id 让 trace 与业务 session 关联
        config = (
            {"callbacks": [handler], "metadata": {"langfuse_session_id": session_id}}
            if handler
            else None
        )
```

`_produce` 中 `invoke(session_id, query, context)` 改为 `invoke(session_id, query, context, config=config)`。

`rag/document/retriever.py`：import 增加 `from rag.observability.langfuse import observe_if_enabled`，`search` 方法上加装饰器：

```python
    @observe_if_enabled(name="knowledge_retrieve")
    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
```

- [ ] **Step 4: 单测回归**

Run: `uv run pytest tests/test_workflow.py tests/test_retriever.py tests/test_observability.py tests/test_chat_controller.py -v`
Expected: 全部 PASS（test_chat_controller 不劣于基线）。注意 `tests/test_retriever.py` 在 `LANGFUSE_ENABLED=false` 下必须原样通过（装饰器直通）——若 .env 里是 true 导致测试行为变化，测试进程用 `LANGFUSE_ENABLED=false uv run pytest ...` 方式隔离并在报告中说明。

- [ ] **Step 5: 端到端 smoke（真实链路，Task 4 的 Langfuse 必须在跑）**

1. 确认基础设施：`docker compose ps`（pg/redis/minio/neo4j/langfuse 全部 up）。
2. 启动 API：`uv run rag-api`（后台，等待就绪日志）。
3. 发一次真实 chat 请求（端点与请求体以 `rag/api/modules/chat/controller.py` 实况为准，SSE 流读到结束）。
4. 验收：
   - API 日志出现「向量召回完成/为空」且行尾带 `session=<id>`（文本格式）；
   - `curl -s -u pk-lf-rag-demo-local:sk-lf-rag-demo-local "http://localhost:3000/api/public/traces?limit=5"` 返回的最新 trace 含本次会话（metadata 的 langfuse_session_id 匹配），且 observations 中同时存在 LLM generation 与 `knowledge_retrieve` span；
   - 数据库无文档时空召回也算通过（WARNING 日志 + trace 存在即可），但在报告中注明。
5. 停掉 API 进程。

- [ ] **Step 6: 全量测试 + 提交（受 Step 0 规程约束）**

Run: `uv run pytest -q`
Expected: 失败集合与基线一致。

```bash
git add rag/agent/workflow.py rag/api/modules/chat/service.py rag/document/retriever.py tests/test_workflow.py
git commit -m "feat: Langfuse 追踪接线(workflow 回调透传+chat 组装+检索 span)"
```
