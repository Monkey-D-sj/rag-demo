# 异步地基（Async Foundation）设计文档

- 日期：2026-06-18
- 子项目：A —— 异步地基（生产化第一阶段）
- 状态：待用户复核

## 1. 背景与目标

当前 `rag-demo` 是一套同步的 RAG 骨架，要推向生产。生产化整体拆为四个子项目：

| | 子项目 | 现状 |
|---|---|---|
| **A** | 异步地基（本文档） | 同步，且部分不安全 |
| B | RAG 核心检索（documents 表 / 文档入库 / 混合检索召回） | 完全缺失 |
| C | 流式 API 层（FastAPI + SSE） | 无入口 |
| D | 可观测性 + 测试 + CI | 几乎没有 |

B/C/D 都建在 A 的异步地基上，故 A 优先。

**部署前提（已与用户确认）：** 单机 / 小规模、对外提供流式 HTTP API、技术栈重构为异步。

### 本子项目目标

把基础设施与核心节点改造为 async，建立干净的资源生命周期，并修复一批确定性 bug，使一条最小请求能端到端异步跑通——为 B/C 提供不返工的地基。

### 已确认的关键决策

1. **PostgreSQL 异步驱动：psycopg3 async**（保留现有 `%(name)s` 命名参数与 dict 行，SQL 几乎不改；自带 `AsyncConnectionPool`）。
2. **Schema 管理：alembic**（版本化迁移，启动只校验不改结构）。
3. **完成边界：基础设施 + 节点异步化 + 最小端到端跑通**。
4. **对象生命周期：方案 A —— 显式资源容器 + async lifespan**，彻底消除 import 时连库。

### 不在本子项目范围内（明确排除）

- FastAPI / SSE 流式端点、健康探针、鉴权、限流 → C
- 知识库 documents 表、文档加载/切分/入库、混合检索召回节点 → B
- 结构化日志 / trace / metrics / request-id / CI → D
- `bm25_search` / `hybrid_search`（当前无人调用的知识库检索函数）→ 随 B 实现，本轮移走

## 2. 架构与生命周期

核心思想：用「显式资源容器 + async lifespan」替换「全局单例 + import 时连库」，所有 I/O 走 async。

```
启动顺序（由 lifespan 驱动，而非 import）：
  Settings（pydantic-settings 读 env）
        │
        ▼
  AppResources.startup()                 ← async
    ├── AsyncConnectionPool  (psycopg3)   ← 池在这里建，不在 import
    ├── redis.asyncio client
    ├── NormalModel          (ChatOpenAI.ainvoke / .astream)
    └── MemoryManager
          ├── PgVectorLongTermMemory(pool)
          └── RedisShortTermMemory(redis)
        │
        ▼
  ContextSchema(llm, memory_manager)  ──注入──▶  graph.astream(...)
        │
        ▼
  AppResources.aclose()  ← 关池、关 redis（优雅关闭）
```

**关键不变量：** 模块 import 期只允许纯 CPU 工作（定义类、编译 graph）。任何连接、建池、跑 SQL 一律推迟到 `startup()`。这是消除当前「import rag.workflow 就连库」的根本手段。

### 文件清单

| 文件 | 角色 |
|---|---|
| `rag/config.py` 🆕 | `Settings`（pydantic-settings），统一所有 env，消灭散落的 `os.getenv`/`load_dotenv` |
| `rag/resources.py` 🆕 | `AppResources` 容器 + `startup()/aclose()` + `lifespan()` 异步上下文管理器 |
| `rag/db/postgres.py` ✏️ | `AsyncConnectionPool` + async `get_cursor`；移除所有 DDL（交给 alembic）；移走未使用的 bm25/hybrid 检索函数 |
| `rag/db/redis.py` ✏️ | `redis.asyncio` 客户端；池由 resources 持有 |
| `rag/models/base.py` ✏️ | `ChatModel` 抽象方法异步化（`ainvoke` / `astream`） |
| `rag/models/normal.py` ✏️ | `ChatOpenAI.ainvoke/.astream`；tenacity async 重试；`_translate` 异步化 |
| `rag/models/embedding.py` ✏️ | `AsyncOpenAI` + async；补重试/超时/异常映射；客户端由 resources 持有 |
| `rag/memory/adapters/base.py` ✏️ | 方法异步化；统一 `search` 签名 |
| `rag/memory/adapters/long_term_pgsql.py` ✏️ | async；`__init__` 只存 pool；修 `search` 签名 |
| `rag/memory/adapters/short_term_redis.py` ✏️ | async pipeline |
| `rag/memory/manager.py` ✏️ | async；修 `search`；新增 `add_message`；去掉全局单例 |
| `rag/nodes/recall_memory/memory.py` ✏️ | async；`Runtime[ContextSchema]`；降级 |
| `rag/nodes/query/query.py` ✏️ | async；带重试 `ainvoke`；结果写 `generated` |
| `rag/nodes/persist_memory/` 🆕 | async 写回短期记忆节点 |
| `rag/type.py` ✏️ | `MyState` 字段对齐 |
| `rag/workflow.py` ✏️ | graph import 期编译（无 I/O）；`invoke` 异步 + `graph.astream`；资源不在 import 期创建 |
| `main.py` ✏️ | 重写为最小 async 入口，跑样例 query 验证端到端 |
| `alembic/`、`alembic.ini` 🆕 | 版本化迁移 |
| `pyproject.toml` ✏️ | 依赖调整（见 §6） |
| `tests/test_memory.py` ✏️ | 修残缺语法 + 异步测试 |
| `.gitignore` / `.env.example` ✏️🆕 | `.env` 移出 git |

## 3. db · models · memory 改造

### 3.1 db 层

**`db/postgres.py`**

```python
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

@asynccontextmanager
async def get_cursor(pool):              # pool 由调用方（adapter）持有，显式传入
    async with pool.connection() as conn:        # 自动归还
        async with conn.cursor(row_factory=dict_row) as cur:
            yield cur
        # psycopg3：connection() 块正常结束自动 commit；异常自动 rollback
```

- `%(name)s` 命名参数、dict 行 → 现有 SQL 原样保留。
- 删除 `ensure_pgvector_extension` / `ensure_pgbm25_extension`（移到 alembic）。
- 删除/移走 `bm25_search` / `hybrid_search`（无人调用，属 B）。

**`db/redis.py`** → `redis.asyncio`，客户端由 `AppResources` 持有并在 `aclose()` 里关闭。

### 3.2 models 层

**`models/base.py`**

```python
class ChatModel(ABC):
    @abstractmethod
    async def ainvoke(self, messages) -> str: ...
    @abstractmethod
    def astream(self, messages): ...        # async generator
```

**`models/normal.py`** — `ChatOpenAI.ainvoke / .astream`；tenacity 原生 async（`AsyncRetrying`），现有重试/异常映射体系照搬到 async 版；`_translate` 改 async 上下文管理器。

**`models/embedding.py`** — `openai.AsyncOpenAI`，`generate_embeddings_batch` 改 async；加 tenacity 重试 + 超时 + 异常映射（复用 `common/exception.py`）；客户端由 resources 持有。

### 3.3 memory 层（异步化 + 修 bug）

**`adapters/base.py`** — 抽象方法全 async；**统一 `search` 签名为 `search(self, session_id, query, top_k, filters)`**（基类与实现从此一致）。

**`adapters/long_term_pgsql.py`** — 所有方法 async，`async with get_cursor(self._pool)`；`__init__` 只存 pool，不再建表；`search` 参数顺序按统一签名修正。

**`adapters/short_term_redis.py`** — async pipeline（`await pipe.execute()`）；保留 TTL/ltrim 逻辑。

**`memory/manager.py`**
- `search` 调用方/签名对齐（统一顺序后）；
- 新增 `async def add_message(session_id, text, metadata)` 调 `_short_term.add(...)`（短期记忆此前从不被写入，这是它一直为空的根因）；
- 去掉全局单例 `_memory_manager` / `get_memory_manager`，由 `AppResources` 构建注入。

## 4. 节点 · 数据流 · 写回 · 降级

本轮 graph 仍是两个真实节点（知识库 `recall` 属 B，不接），新增一个写回节点闭合记忆环：

```
START → recall_memory → handle_query → persist_memory → END
         （读短/长期）     （LLM 生成）      （写回短期）
```

### 4.1 节点改造

- **`recall_memory`** — async，`Runtime[ContextSchema]`，`await memory_manager.search(...)` 按统一签名调用。
- **`handle_query`** — async，改用带重试的 `await llm.ainvoke(...)`；结果写 `state["generated"]`（不再错写 `rewrite_query`）。
- **`persist_memory` 🆕** — async，把本轮 `raw_query` 与 `generated` 写进短期记忆（`manager.add_message`）。本轮只写短期；长期向量库写入（记忆策展）留给 B/API。

### 4.2 MyState 字段对齐

```python
class MyState(TypedDict):
    session_id: str
    raw_query: str        # 输入
    context: str          # recall_memory 产出
    generated: str        # handle_query 产出（最终答案）
    # 移除：rewrite_query / recall_bm25_results / recall_vec_results
    #   —— 属 B 的检索链路，避免“声明了没人填”的死字段
```

### 4.3 降级策略

| 节点 | 失败处理 | 理由 |
|---|---|---|
| `recall_memory` | 吞异常 + 记日志，`context=""` 继续 | 记忆是增强项，挂了仍要能答 |
| `handle_query` | 重试 3 次仍失败 → 抛 `LLMException` | 核心能力，无答案=请求失败，交由入口/未来 API 边界处理 |
| `persist_memory` | 吞异常 + 记日志，不影响已生成答案 | 写回失败不该让用户拿不到答案 |

> 当前 `recall_memory` 的 `except: raise e` 与此相反（记忆挂了整条死），本轮改为降级。

### 4.4 数据流

`invoke(session_id, query)` → `graph.astream` 逐节点产出 chunk（状态增量 + `get_stream_writer` 进度文案）→ 调用方拿到流。最小入口 `main.py` 消费这个流打印，证明端到端跑通。

## 5. alembic 迁移

```
alembic/
├── env.py            # 同步驱动跑迁移（psycopg3 sync url）；应用运行时才是 async
├── script.py.mako
└── versions/
    └── 0001_initial.py
alembic.ini
```

**迁移用同步驱动：** alembic 同步执行 DDL 最简单稳妥；应用运行时用 async 池，两者不同连接、互不影响，是社区标准做法。

**`0001_initial.py` 内容**（收敛现散在 adapter 的 DDL 为一份版本化迁移）：
- `CREATE EXTENSION IF NOT EXISTS vector` / `pg_bm25`
- `CREATE TABLE long_term_memories(...)`（原样）
- `CREATE INDEX ... hnsw (embedding vector_cosine_ops)`
- `CALL paradedb.create_bm25(...)` —— 错误不再 `except: pass` 吞掉，失败显式报错

**权限：** `CREATE EXTENSION` 需较高权限 → 迁移由部署时特权角色执行；应用运行时角色无需建库权限（消除「运行时要超管权限」隐患）。

> 本轮迁移只含 `long_term_memories`。documents 知识库表属 B，届时新增 `0002_*`。

## 6. 依赖调整（pyproject.toml）

```diff
- "psycopg2-binary>=2.9"          # 删（同步、与下面冲突）
- "psycopg2>=2.9.12"              # 删（冗余）
+ "psycopg[binary,pool]>=3.2"     # psycopg3 async + AsyncConnectionPool
+ "pydantic-settings>=2.0"        # 配置
+ "alembic>=1.13"                 # 迁移
+ "tenacity>=8.0"                 # normal.py 已 import 但未声明（隐性 bug）
+ "openai>=1.0"                   # embedding.py 已 import 但未声明（隐性 bug）
+ "pytest>=8.0"  "pytest-asyncio>=0.23"   # 测试
  # redis>=5.0 保留（自带 redis.asyncio）；langchain / langgraph 保留
```

**`requires-python`：已确认从 `>=3.14` 降到 `>=3.12`**（3.14 太新、部分轮子未必齐，生产风险）。`.python-version` 同步改为 3.12，需在 3.12 环境重建 `.venv` 并刷新 `uv.lock`。

## 7. 安全 / 配置

- `.env` 当前被 git 跟踪且含 API key → `git rm --cached .env`，加入 `.gitignore`，提供 `.env.example`（占位、无真实密钥）。
- 默认口令 `rag123` 等敏感默认值仅用于本地 docker-compose；生产值经 `Settings` 从环境注入，不进仓库。

## 8. 测试方案

- **修 `tests/test_memory.py`**：当前残缺语法（`def test_memory_add(self):` 无函数体）导致整个 pytest 收集崩溃，必先修。
- **pytest-asyncio 异步测试，分两层：**
  - 数据层往返（需 docker-compose 起 pg/redis）：长期 `add → search` 命中；短期 `add_message → get_recent` 命中；`search` 签名修正后参数不再错位。
  - 地基冒烟：`async with lifespan()` 起资源 → 跑一条 `invoke` → 断言 `generated` 非空、`persist_memory` 写回成功；LLM/embedding 用 fake/mock，不打真实外部 API。
- **降级测试**：令 `recall_memory` 内部抛异常，断言请求仍返回答案。
- CI 属 D，不在本轮；测试按「能进 CI」组织。

## 9. 完成定义（Definition of Done）

- [ ] `import rag.workflow` 不触发任何数据库连接
- [ ] 全链路 async：db / redis / embedding / llm / memory / 三个节点 / `invoke`
- [ ] `AppResources` 经 `lifespan()` 构建与释放；无全局单例残留
- [ ] alembic `0001_initial` 可在干净库上 `upgrade head` 成功；adapter 不再跑 DDL
- [ ] 4 个 bug 修复：`search` 三处签名统一；短期记忆经 `add_message` 真正写入；`handle_query` 写 `generated`；`recall_memory` 改为降级
- [ ] 2 个缺失依赖（tenacity、openai）补入 pyproject；psycopg2 系列移除
- [ ] `.env` 移出 git，`.env.example` 就位
- [ ] `pytest` 收集不再因语法崩溃；数据层往返 + 地基冒烟 + 降级测试通过
- [ ] `main.py` 跑样例 query，端到端打印出流式结果

## 10. 风险与备注

- **psycopg3 + ParadeDB**：本轮迁移含 `paradedb.create_bm25`，但运行时 BM25 检索（`@@@ paradedb.parse`）属 B，本轮不在热路径调用；迁移阶段需确认 ParadeDB 镜像下 `create_bm25` 正常。
- **Python 版本**：已定 3.12（见 §6）。改 `requires-python` + `.python-version` 后需在 3.12 重建 `.venv`、刷新 `uv.lock`，并验证 langgraph/langchain/psycopg3 等在 3.12 下轮子齐备。
- **alembic 与运行时双 URL**：需在 `Settings` 暴露同步/异步两种连接串（或由异步串推导同步串），env.py 用同步串。
