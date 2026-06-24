# 设计:把 db→memory→agent 接线打通

- 日期:2026-06-24
- 分支:`feat/async-foundation`
- 状态:已批准,待实现

## 背景与问题

db 资源(`pool` / `redis`)的生命周期已绑定到 FastAPI `lifespan`,启动后挂在
`app.state.pg` / `app.state.redis` 上,**只在运行期存在**。

但 `rag/workflow.py` 仍在 **import 期**构造模块级单例:

```python
memory_manager = get_memory_manager()   # 该函数在 manager.py 中不存在
llm = NormalModel()                       # 现签名是 NormalModel(settings),缺参
context = ContextSchema(memory_manager=memory_manager, llm=llm)
graph = builder.compile()
```

这与"db 在 lifespan 初始化"直接冲突:import 期还没有 pool/redis,且
`get_memory_manager` 根本不存在、`NormalModel()`/`EmbeddingModel()` 现都要求
`settings`。此外两个图节点是 dead-on-arrival:

- `recall_memory`:同步 `def` 却调用 `async` 方法且未 `await`;`search` 参数与
  `MemoryManager.search(session_id, query, top_k, filters)` 签名不符。
- `handle_query`:同步 `def`,且调用 `llm.invoke(...)` —— `ChatModel` 只有
  `ainvoke`,无同步 `invoke`。

## 目标与范围

**目标**:打通 `db → memory → agent` 的依赖注入链,让编译后的 graph 能真正运行,
并把分支留在一个自洽、可跑通的状态。

**范围内**:
- lifespan 中构造依赖 db 的单例并挂 `app.state`。
- 新增薄 Depends 读取函数。
- `workflow.invoke` 改为接收 `context`,删除模块级单例。
- 修复 `recall_memory` 与 `handle_query` 两个节点的 async/参数/方法名 bug。
- `controller` 给最小但能跑通整条链的实现。

**范围外(后续 PR)**:
- 完整 `/chat` 接口(请求体 schema 校验、SSE 流式返回、错误映射)。
- 全局异常处理中间件。
- `embedding`/`llm` 的显式关闭(进程退出回收即可)。

## 关键决策(已敲定)

1. **对象生命周期 = lifespan 单例 + 薄 Depends。**
   `EmbeddingModel`/`NormalModel` 内部分别包着 `AsyncOpenAI`/`ChatOpenAI`,底层是
   httpx 连接池。若每请求新建会使 keep-alive 失效、每次重做 TLS 握手、高并发下连接/fd
   飙升。async 客户端设计为跨并发请求共享,故启动建一次、全程复用。
   "用 Depends" ≠ "在 Depends 里 new":单例在 lifespan 建好挂 `app.state`,Depends
   只做薄读取(`return request.app.state.xxx`),既复用又保留可测试/可 override 的接缝。

2. **接线形状 = 方案 A(按资源分别挂 `app.state`,controller 组 context)。**
   与现有 `dependence/db.py` 的 `get_pg`/`get_redis`(一资源一薄读取)完全同构;
   测试可单独 override `llm` 或 `memory_manager`;不引入新抽象。

## 架构与数据流

```
HTTP 请求
  └─ chat controller
       ├─ Depends(get_memory_manager) ─┐  (薄读取 app.state)
       └─ Depends(get_llm) ────────────┤
                                        ▼
                         ContextSchema(memory_manager, llm)   ← 每请求组装
                                        ▼
                  workflow.invoke(session_id, query, context)
                                        ▼
                        graph.astream(state, context=context)
                          ├─ recall_memory → runtime.context.memory_manager → adapters → pool/redis
                          └─ handle_query  → runtime.context.llm.ainvoke(...)
```

- **单例(启动建一次,挂 `app.state`)**:`pool`、`redis`、`embedding`、
  `memory_manager`、`llm`。
- **每请求新建(零成本)**:`ContextSchema`。
- **模块级保留**:编译后的 `graph`(无状态)。

节点永远不直接触碰 `pool`/`redis`,只通过 `runtime.context.memory_manager` /
`runtime.context.llm` 访问。

## 各文件改动

### `rag/api/main.py`(lifespan)
在已有 pool/redis 之后追加:

```python
embedding = EmbeddingModel(settings)
app.state.memory_manager = MemoryManager(
    long_term=PgVectorLongTermMemory(pool, embedding),
    short_term=RedisShortTermMemory(app.state.redis),
)
app.state.llm = NormalModel(settings)
```

teardown 维持现状(`await pool.close()` / `await app.state.redis.aclose()`)。
`embedding`/`llm` 不显式 close(范围外)。

### `rag/api/dependence/agent.py`(新增)
与 `db.py` 同构的薄读取:

```python
from fastapi import Request
from rag.memory import MemoryManager
from rag.models.base import ChatModel


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


def get_llm(request: Request) -> ChatModel:
    return request.app.state.llm
```

### `rag/workflow.py`
- 删除模块级单例(`memory_manager` / `llm` / `context`)及不存在的
  `get_memory_manager` import。
- 保留 `graph = builder.compile()`。
- `invoke` 改为异步、接收 context:

```python
async def invoke(session_id: str, query: str, context: ContextSchema):
    async for chunk in graph.astream(
        {"session_id": session_id, "raw_query": query}, context=context
    ):
        yield chunk
```

### `rag/api/modules/chat/controller.py`
最小可跑实现:Depends 取出 mm/llm → 组 `ContextSchema` → 调 `invoke`,
`async for` 消费收集后返回。流式/SSE 留后续。

### `rag/nodes/recall_memory/memory.py`
- 改 `async def`。
- `await` 两个 manager 调用。
- `search` 改为 `await mm.search(state["session_id"], state["raw_query"])`(去掉冗余
  `filters`,长期适配器内部已按 `session_id` 过滤)。

### `rag/nodes/query/query.py`
- 改 `async def`。
- `state["rewrite_query"] = await llm.ainvoke([...])`。

## 错误处理

- 节点内现有 `try/except` + `writer` 模式保留。
- controller 层暂不加全局异常映射(出范围)。

## 测试

- **节点单测**:`recall_memory` / `handle_query` 各加一个单测,用假 mm / 假 llm
  注入 `ContextSchema`,断言节点 `await` 了正确方法、参数顺序正确(对齐
  `tests/test_memory.py` 中 `_FakeLong`/`_FakeShort` 的假对象写法)。
- **lifespan 接线**:用 ASGI 起 app,断言启动后 `app.state.memory_manager` /
  `app.state.llm` 存在;需真实 db 的标 `@pytest.mark.integration`(沿用现有约定)。
- **invoke 签名**:轻量冒烟测,验证 `invoke` 接收 context;若无真实 LLM 不便跑全图,
  降级为只测 controller→invoke 的参数传递。

## 验收标准

- App 启动后 `app.state` 上五个单例齐备。
- `rag/workflow.py` 无模块级 db 依赖单例,import 不再报错。
- 两个节点为 `async def` 且正确 `await`、调用签名正确。
- controller 能走通 Depends → context → invoke 整条链。
- 新增/修改的非 integration 测试全绿。
