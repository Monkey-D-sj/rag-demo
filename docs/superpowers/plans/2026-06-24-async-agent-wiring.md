# Async Agent 接线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 `db → memory → agent` 的依赖注入链,让编译后的 langgraph graph 能真正异步运行,并修复两个 dead-on-arrival 的图节点。

**Architecture:** db 资源(pool/redis)及依赖它们的单例(embedding/memory_manager/llm)在 FastAPI `lifespan` 启动时构造一次、挂到 `app.state`;薄 Depends 函数从 `app.state` 读取;controller 每请求组装 `ContextSchema` 传给异步 `workflow.invoke`;图节点只通过 `runtime.context` 访问依赖,不直接碰 pool/redis。

**Tech Stack:** Python 3.12、FastAPI、langgraph、psycopg3(AsyncConnectionPool)、redis.asyncio、langchain-openai、pytest + pytest-asyncio。

## Global Constraints

- Python `>=3.12`。
- pytest `asyncio_mode = "auto"`:`async def test_...` 无需 `@pytest.mark.asyncio`。
- 需真实 pg/redis 的测试必须标 `@pytest.mark.integration`。
- 默认跑非 integration 用例:`uv run pytest -m "not integration"`(若不用 uv,去掉 `uv run` 前缀)。
- Windows 事件循环策略已在 `tests/conftest.py` 处理,勿改。
- 薄 Depends 风格须对齐现有 `rag/api/dependence/db.py`(一资源一读取函数)。
- 频繁提交:每个 Task 末尾一次提交。

---

### Task 1: 修复 `recall_memory` 节点(async + 正确调用)

**Files:**
- Modify: `rag/nodes/recall_memory/memory.py`
- Test: `tests/test_nodes.py`(新建)

**Interfaces:**
- Consumes: `MemoryManager.search(session_id, query, top_k=5, filters=None)`(async)、`MemoryManager.get_recent_messages(session_id, n=10)`(async);`ContextSchema(llm, memory_manager)`。
- Produces: `async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState`,写入 `state["context"]`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_nodes.py`:

```python
from types import SimpleNamespace

import rag.nodes.recall_memory.memory as recall_mod
from rag.type import ContextSchema


class _FakeMM:
    def __init__(self):
        self.search_calls = []
        self.recent_calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.search_calls.append((session_id, query))
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        self.recent_calls.append(session_id)
        return [{"text": "S1"}]


async def test_recall_memory_awaits_and_composes_context(monkeypatch):
    monkeypatch.setattr(recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    mm = _FakeMM()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=mm))
    state = {"session_id": "s1", "raw_query": "q1"}

    out = await recall_mod.recall_memory(state, runtime)

    assert mm.search_calls == [("s1", "q1")]
    assert mm.recent_calls == ["s1"]
    assert out["context"] == "S1\nL1"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_nodes.py::test_recall_memory_awaits_and_composes_context -v`
Expected: FAIL(当前 `recall_memory` 是同步 `def`、未 await,且 `search` 参数为 `(raw_query, filters=...)`;`await` 协程对象会报错或断言不符)

- [ ] **Step 3: 写实现**

将 `rag/nodes/recall_memory/memory.py` 整体替换为:

```python
from langgraph.runtime import Runtime
from langgraph.config import get_stream_writer

from rag.type import MyState, ContextSchema


async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从长期和短期记忆中召回相关内容"""

    writer = get_stream_writer()
    writer("检索记忆中...")

    try:
        memory_manager = runtime.context.memory_manager

        # 长期记忆:向量检索(适配器内部已按 session_id 过滤)
        long_results = await memory_manager.search(
            state["session_id"], state["raw_query"]
        )

        # 短期记忆:最近会话消息
        short_results = await memory_manager.get_recent_messages(state["session_id"])

        # 合并为上下文文本
        parts: list[str] = []
        for r in short_results:
            parts.append(r.get("text", ""))
        for r in long_results:
            parts.append(r.get("text", ""))

        state["context"] = "\n".join(parts)
        return state
    except Exception as e:
        writer(f"记忆检索失败: {e}")
        raise
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_nodes.py::test_recall_memory_awaits_and_composes_context -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/nodes/recall_memory/memory.py tests/test_nodes.py
git commit -m "fix: recall_memory 改 async 并修正 search 调用/await"
```

---

### Task 2: 修复 `handle_query` 节点(async + ainvoke)

**Files:**
- Modify: `rag/nodes/query/query.py`
- Test: `tests/test_nodes.py`(追加)

**Interfaces:**
- Consumes: `ChatModel.ainvoke(messages) -> str`(async);`ContextSchema(llm, memory_manager)`。
- Produces: `async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState`,写入 `state["rewrite_query"]`。

- [ ] **Step 1: 写失败测试**

向 `tests/test_nodes.py` 追加:

```python
import rag.nodes.query.query as query_mod


class _FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return "rewritten"


async def test_handle_query_awaits_ainvoke(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q1", "context": "ctx"}

    out = await query_mod.handle_query(state, runtime)

    assert out["rewrite_query"] == "rewritten"
    assert len(llm.calls) == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_nodes.py::test_handle_query_awaits_ainvoke -v`
Expected: FAIL(当前是同步 `def` 且调用不存在的 `llm.invoke`)

- [ ] **Step 3: 写实现**

将 `rag/nodes/query/query.py` 整体替换为:

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.type import MyState, ContextSchema

system_prompt = """
你是一个专业的关务助手, 你的任务是根据用户的查询, 提供专业的关务信息.
有不确定的地方，例如：他/那么。
从上下文获取信息，改写消息返回
"""


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """处理查询"""
    # ----------- 输出 -----------
    writer = get_stream_writer()
    writer("深度思考中")

    llm = runtime.context.llm

    state["rewrite_query"] = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
用户查询: {state["raw_query"]}
上下文: {state["context"]}
"""),
    ])
    return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_nodes.py::test_handle_query_awaits_ainvoke -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/nodes/query/query.py tests/test_nodes.py
git commit -m "fix: handle_query 改 async 并改用 ainvoke"
```

---

### Task 3: `workflow.invoke` 改异步并删除模块级单例

**Files:**
- Modify: `rag/workflow.py`
- Test: `tests/test_workflow.py`(新建)

**Interfaces:**
- Consumes: `recall_memory` / `handle_query`(Task 1/2);`graph.astream(state, context=...)`。
- Produces: `async def invoke(session_id: str, query: str, context: ContextSchema)` —— 异步生成器,逐块 yield。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_workflow.py`:

```python
import rag.workflow as wf
from rag.type import ContextSchema


class _FakeMM:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L1"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S1"}]


class _FakeLLM:
    async def ainvoke(self, messages):
        return "rw"


async def test_invoke_runs_full_graph_with_context():
    ctx = ContextSchema(llm=_FakeLLM(), memory_manager=_FakeMM())

    merged = {}
    async for chunk in wf.invoke("s1", "q1", ctx):
        for _node, payload in chunk.items():
            merged.update(payload)

    assert merged["context"] == "S1\nL1"
    assert merged["rewrite_query"] == "rw"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: FAIL(当前 `workflow.py` 在 import 期执行 `get_memory_manager()`——该函数不存在,模块导入即 ImportError)

- [ ] **Step 3: 写实现**

将 `rag/workflow.py` 整体替换为:

```python
from langgraph.graph import END, START, StateGraph

from rag.nodes.query.query import handle_query
from rag.nodes.recall_memory.memory import recall_memory
from rag.type import MyState, ContextSchema

# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

builder.add_node("handle_query", handle_query)
builder.add_node("recall_memory", recall_memory)

builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", END)

graph = builder.compile()


async def invoke(session_id: str, query: str, context: ContextSchema):
    async for chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
    ):
        yield chunk
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/workflow.py tests/test_workflow.py
git commit -m "refactor: workflow.invoke 改异步接收 context,删除模块级单例"
```

---

### Task 4: 新增 `dependence/agent.py` 薄 Depends 读取

**Files:**
- Create: `rag/api/dependence/agent.py`
- Test: `tests/test_dependence_agent.py`(新建)

**Interfaces:**
- Consumes: `request.app.state.memory_manager` / `request.app.state.llm`(Task 5 写入)。
- Produces: `get_memory_manager(request) -> MemoryManager`、`get_llm(request) -> ChatModel`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_dependence_agent.py`:

```python
from types import SimpleNamespace

from rag.api.dependence.agent import get_memory_manager, get_llm


def test_get_memory_manager_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(memory_manager=sentinel))
    )
    assert get_memory_manager(request) is sentinel


def test_get_llm_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(llm=sentinel)))
    assert get_llm(request) is sentinel
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_dependence_agent.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.api.dependence.agent`)

- [ ] **Step 3: 写实现**

新建 `rag/api/dependence/agent.py`:

```python
from fastapi import Request

from rag.memory import MemoryManager
from rag.models.base import ChatModel


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


def get_llm(request: Request) -> ChatModel:
    return request.app.state.llm
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_dependence_agent.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/api/dependence/agent.py tests/test_dependence_agent.py
git commit -m "feat: 新增 get_memory_manager/get_llm 薄 Depends 读取"
```

---

### Task 5: lifespan 构造单例并挂 `app.state`

**Files:**
- Modify: `rag/api/main.py`
- Test: `tests/test_api_lifespan.py`(新建,integration)

**Interfaces:**
- Consumes: `create_pg_pool`、`create_redis_client`、`EmbeddingModel(settings)`、`PgVectorLongTermMemory(pool, embedding)`、`RedisShortTermMemory(redis)`、`MemoryManager(long_term, short_term)`、`NormalModel(settings)`。
- Produces: 启动后 `app.state` 上具备 `pg` / `redis` / `memory_manager` / `llm`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_api_lifespan.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_api_lifespan.py -v -m integration`
Expected: FAIL(`AttributeError: ... 'memory_manager'`——当前 lifespan 只挂了 pg/redis)

- [ ] **Step 3: 写实现**

将 `rag/api/main.py` 整体替换为:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.modules import register_modules
from rag.common.logging import setup_logging
from rag.config import get_settings
from rag.db import create_pg_pool, create_redis_client
from rag.memory import MemoryManager
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()

    # ------ 初始化 pg -------
    pool = await create_pg_pool(settings)
    app.state.pg = pool

    # ------ 初始化 redis -------
    app.state.redis = create_redis_client(settings)

    # ------ 初始化 agent 依赖单例 -------
    embedding = EmbeddingModel(settings)
    app.state.memory_manager = MemoryManager(
        long_term=PgVectorLongTermMemory(pool, embedding),
        short_term=RedisShortTermMemory(app.state.redis),
    )
    app.state.llm = NormalModel(settings)

    yield

    # ------ 关闭资源 -------
    await pool.close()
    await app.state.redis.aclose()


app = FastAPI(lifespan=lifespan)
register_modules(app)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_api_lifespan.py -v -m integration`
Expected: PASS(需 docker compose 起的 pg/redis)

- [ ] **Step 5: 提交**

```bash
git add rag/api/main.py tests/test_api_lifespan.py
git commit -m "feat: lifespan 构造 embedding/memory_manager/llm 单例挂 app.state"
```

---

### Task 6: controller 最小可跑实现(打通整条链)

**Files:**
- Modify: `rag/api/modules/chat/controller.py`
- Test: `tests/test_chat_controller.py`(新建)

**Interfaces:**
- Consumes: `get_memory_manager` / `get_llm`(Task 4)、`workflow.invoke`(Task 3)、`ContextSchema`。
- Produces: `POST /chat/` 接收 `{session_id, query}`,返回 `{"chunks": [...]}`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_chat_controller.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.chat.controller as controller_mod
from rag.api.dependence.agent import get_llm, get_memory_manager


async def _fake_invoke(session_id, query, context):
    yield {"recall_memory": {"context": "c"}}
    yield {"handle_query": {"rewrite_query": "rw"}}


def test_chat_controller_runs_chain(monkeypatch):
    monkeypatch.setattr(controller_mod, "invoke", _fake_invoke)

    app = FastAPI()
    app.include_router(controller_mod.chat_router)
    app.dependency_overrides[get_memory_manager] = lambda: object()
    app.dependency_overrides[get_llm] = lambda: object()

    client = TestClient(app)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    assert resp.json()["chunks"][-1] == {"handle_query": {"rewrite_query": "rw"}}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_chat_controller.py -v`
Expected: FAIL(当前 controller 是返回 `{"message": "chat"}` 的桩,且无 `invoke` 可 monkeypatch)

- [ ] **Step 3: 写实现**

将 `rag/api/modules/chat/controller.py` 整体替换为:

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.api.dependence.agent import get_llm, get_memory_manager
from rag.memory import MemoryManager
from rag.models.base import ChatModel
from rag.type import ContextSchema
from rag.workflow import invoke

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    session_id: str
    query: str


@chat_router.post("/")
async def chat(
    body: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    llm: ChatModel = Depends(get_llm),
):
    context = ContextSchema(llm=llm, memory_manager=memory_manager)
    chunks = [chunk async for chunk in invoke(body.session_id, body.query, context)]
    return {"chunks": chunks}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_chat_controller.py -v`
Expected: PASS

- [ ] **Step 5: 全量非 integration 回归 + 提交**

Run: `uv run pytest -m "not integration" -v`
Expected: 全绿

```bash
git add rag/api/modules/chat/controller.py tests/test_chat_controller.py
git commit -m "feat: chat controller 打通 Depends→context→invoke 整条链"
```

---

## Self-Review 备注

- **Spec 覆盖**:lifespan 单例(Task 5)、薄 Depends(Task 4)、invoke 改异步删全局(Task 3)、recall_memory 修复(Task 1)、handle_query 修复(Task 2)、controller 最小实现(Task 6)、测试三类(节点单测 Task1/2、lifespan integration Task5、invoke 冒烟 Task3、controller Task6)——逐条对应。
- **范围外**未触碰:SSE 流式、全局异常映射、embedding/llm 显式 close。
- **类型一致**:`invoke(session_id, query, context)` 签名在 Task 3 定义,Task 6 一致调用;`get_memory_manager`/`get_llm` 在 Task 4 定义,Task 6 一致引用;`ContextSchema(llm, memory_manager)` 全程一致。
