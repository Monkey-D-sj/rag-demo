# 异步地基 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 rag-demo 的基础设施与核心节点从同步改造为 async，建立「资源容器 + lifespan」的生命周期，消除 import 时连库，修复一批确定性 bug，使一条最小请求能端到端异步跑通。

**Architecture:** 显式 `AppResources` 容器经 async `lifespan()` 构建/释放所有 I/O 资源（psycopg3 异步池、redis.asyncio、LLM、embedding、MemoryManager），注入 LangGraph 的 `ContextSchema`。import 期只编译 graph，不做任何 I/O。Schema 由 alembic 版本化管理。

**Tech Stack:** Python 3.12、psycopg3（async + AsyncConnectionPool）、pgvector、redis.asyncio、langchain-openai（ainvoke/astream）、openai AsyncOpenAI、tenacity、alembic、pydantic-settings、pytest + pytest-asyncio。

## Global Constraints

- **Python：** `requires-python>=3.12`，`.python-version` = `3.12`（不得用 3.14 特性）。
- **import 不连库：** 任何模块 import 期只允许纯 CPU 工作；连接/建池/SQL 一律推迟到 `AppResources.startup()`。
- **全链路 async：** db / redis / embedding / llm / memory / 节点 / `invoke` 全部 `async`，不得保留同步 I/O。
- **无全局单例：** 不得保留 `_pool` / `_memory_manager` 等模块级懒加载单例；资源一律由 `AppResources` 持有并注入。
- **search 统一签名：** `search(self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None)`，基类与实现一致。
- **DDL 只在 alembic：** adapter / db 模块不得含 `CREATE EXTENSION/TABLE/INDEX`；错误不得 `except: pass` 静默吞掉。
- **降级：** `recall_memory` / `persist_memory` 失败吞异常+记日志并继续；`handle_query` 失败向上抛。
- **密钥：** `.env` 不进 git；敏感值经 `Settings` 从环境注入。
- **commit message 结尾：** `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## 文件结构

| 文件 | 责任 | Task |
|---|---|---|
| `pyproject.toml` | 依赖与 Python 版本基线 | 1 |
| `.python-version` / `.gitignore` / `.env.example` | 版本、密钥治理 | 1 |
| `rag/config.py` 🆕 | `Settings`（pydantic-settings）+ DSN 派生 | 1 |
| `tests/conftest.py` 🆕 | pytest 共享 fixture | 1, 3 |
| `alembic.ini` / `alembic/env.py` / `alembic/versions/0001_initial.py` 🆕 | 版本化迁移 | 2 |
| `rag/db/postgres.py` ✏️ | `create_pg_pool` + async `get_cursor`（无 DDL） | 3 |
| `rag/db/redis.py` ✏️ | `create_redis_client`（redis.asyncio） | 3 |
| `rag/models/embedding.py` ✏️ | `EmbeddingModel`（AsyncOpenAI + 韧性） | 4 |
| `rag/models/base.py` ✏️ | `ChatModel` async 抽象 | 5 |
| `rag/models/normal.py` ✏️ | `NormalModel` async + 重试 | 5 |
| `rag/memory/adapters/base.py` ✏️ | async 抽象 + 统一 search 签名 | 6 |
| `rag/memory/adapters/long_term_pgsql.py` ✏️ | async；注入 pool + embedding | 6 |
| `rag/memory/adapters/short_term_redis.py` ✏️ | async pipeline | 6 |
| `rag/memory/manager.py` ✏️ | async；`add_message`；去单例；修 search | 7 |
| `rag/memory/__init__.py` ✏️ | 去掉 `get_memory_manager` 导出 | 7 |
| `rag/resources.py` 🆕 | `AppResources` + `lifespan` | 8 |
| `rag/type.py` ✏️ | `MyState` 字段对齐 | 9 |
| `rag/nodes/recall_memory/memory.py` ✏️ | async + 降级 | 9 |
| `rag/nodes/query/query.py` ✏️ | async + 重试 + 写 `generated` | 9 |
| `rag/nodes/persist_memory/memory.py` 🆕 | async 写回短期 | 9 |
| `rag/workflow.py` ✏️ | async `invoke` + `graph.astream` | 10 |
| `main.py` ✏️ | 最小 async 入口 | 10 |
| `tests/test_memory.py` ✏️ | 修语法 + 异步往返测试 | 6, 7 |
| `tests/test_foundation.py` 🆕 | 地基冒烟 + 降级测试 | 10 |

> **测试分层：** 标 `@pytest.mark.integration` 的测试需 `docker compose up -d`（pg/redis）。地基冒烟 / 降级 / 配置测试用 fake，不打真实外部 API。

---

### Task 1: 配置层 + 依赖与版本基线

**Files:**
- Modify: `pyproject.toml`
- Modify: `.python-version`
- Modify: `.gitignore`
- Create: `.env.example`
- Create: `rag/config.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `rag.config.Settings` — pydantic-settings `BaseSettings`，字段见下。
  - `rag.config.get_settings() -> Settings` — `lru_cache` 单例工厂。
  - `Settings.pg_async_dsn -> str`（`postgresql://user:pw@host:port/db`，给 psycopg3 async）
  - `Settings.pg_sync_url -> str`（`postgresql+psycopg://user:pw@host:port/db`，给 alembic/SQLAlchemy）

- [ ] **Step 1: 写失败测试**

`tests/test_config.py`:
```python
from rag.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("PG_HOST", "db.internal")
    monkeypatch.setenv("PG_PORT", "6000")
    monkeypatch.setenv("PG_PASSWORD", "secret")
    s = Settings()
    assert s.pg_host == "db.internal"
    assert s.pg_port == 6000
    assert s.pg_async_dsn == "postgresql://rag:secret@db.internal:6000/rag_memory"
    assert s.pg_sync_url == "postgresql+psycopg://rag:secret@db.internal:6000/rag_memory"


def test_settings_defaults(monkeypatch):
    for k in ("PG_HOST", "PG_PORT", "REDIS_HOST"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.pg_host == "localhost"
    assert s.redis_port == 6379
    assert s.embedding_dim == 1024
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: rag.config`）

- [ ] **Step 3: 写配置实现**

`rag/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── PostgreSQL ──
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "rag_memory"
    pg_user: str = "rag"
    pg_password: str = "rag123"
    pg_pool_min: int = 2
    pg_pool_max: int = 10

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_max_connections: int = 10

    # ── LLM ──
    model_key: str = ""
    model_name: str = ""
    model_url: str = ""

    # ── Embedding ──
    embedding_key: str = ""
    embedding_url: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024

    @property
    def pg_async_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def pg_sync_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_config.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 改 pyproject 依赖与 Python 版本**

`pyproject.toml` 替换为：
```toml
[project]
name = "rag-demo"
version = "0.1.0"
description = "生产级 RAG 服务"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "python-dotenv>=1.0",
    "pydantic-settings>=2.0",
    "langchain>=1.3.7",
    "langchain-openai>=1.3.0",
    "langgraph>=1.2.4",
    "openai>=1.0",
    "psycopg[binary,pool]>=3.2",
    "pgvector>=0.3",
    "redis>=5.0",
    "tenacity>=8.0",
    "alembic>=1.13",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: 需要 docker compose 起的 pg/redis（默认不跑）",
]
```

- [ ] **Step 6: 改版本/密钥治理文件**

`.python-version`:
```
3.12
```

`.gitignore` 末尾追加（`.venv`/`.idea` 已有，补 `.env` 与缓存）：
```
.env
.pytest_cache
.traces
```

`.env.example`（占位，无真实密钥）:
```
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=rag_memory
PG_USER=rag
PG_PASSWORD=changeme
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
MODEL_KEY=sk-xxx
MODEL_NAME=qwen-plus
MODEL_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_KEY=sk-xxx
EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`tests/conftest.py`（占位最小内容，Task 3 补 db fixture）:
```python
import pytest

from rag.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: 把 .env 移出 git 跟踪并重建环境**

Run:
```bash
git rm --cached .env
uv lock
uv sync
```
Expected: `.env` 不再被跟踪；`uv.lock` 在 3.12 下刷新；`uv sync` 成功安装新依赖（验证 langgraph/langchain/psycopg3 在 3.12 轮子齐备）。

- [ ] **Step 8: 全量跑测试**

Run: `pytest -v`
Expected: `tests/test_config.py` 2 passed；旧 `tests/test_memory.py` 仍是语法错误（Task 6 修），此时可临时 `pytest tests/test_config.py -v` 单独验证。

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore .env.example rag/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: 配置层 + 依赖与 3.12 基线 + 密钥治理"
```

---

### Task 2: alembic 初始迁移

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_initial.py`
- Test: `tests/test_migration.py`

**Interfaces:**
- Consumes: `rag.config.get_settings().pg_sync_url`
- Produces: 数据库表 `long_term_memories`、扩展 `vector`/`pg_bm25`、索引 `idx_ltm_embedding`/`idx_ltm_bm25`。

- [ ] **Step 1: 初始化 alembic 骨架**

Run: `uv run alembic init alembic`
Expected: 生成 `alembic.ini` / `alembic/env.py` / `alembic/script.py.mako` / `alembic/versions/`。

- [ ] **Step 2: 配置 env.py 用 Settings 的同步 URL**

`alembic/env.py` 关键改动（替换 `run_migrations_online` 上方的 url 获取，并删掉对 `alembic.ini` 中 `sqlalchemy.url` 的依赖）:
```python
# 在 import 段后加：
from rag.config import get_settings

# 在 run_migrations_offline / run_migrations_online 里，用：
def _get_url() -> str:
    return get_settings().pg_sync_url

# run_migrations_offline:
#   url = _get_url()
# run_migrations_online: 用 engine_from_config 前覆盖 url：
#   configuration = config.get_section(config.config_ini_section, {})
#   configuration["sqlalchemy.url"] = _get_url()
#   connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
```
`alembic.ini` 把 `sqlalchemy.url =` 一行留空（由 env.py 注入）。

- [ ] **Step 3: 写初始迁移**

`alembic/versions/0001_initial.py`:
```python
"""initial schema: long_term_memories + extensions + indexes

Revision ID: 0001
Revises:
Create Date: 2026-06-18
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_bm25")
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ltm_embedding
            ON long_term_memories
            USING hnsw (embedding vector_cosine_ops)
        """
    )
    # BM25 索引：错误不再静默吞掉；已存在用 IF NOT EXISTS 语义由 ParadeDB 处理
    op.execute(
        """
        CALL paradedb.create_bm25(
            table_name => 'long_term_memories',
            index_name => 'idx_ltm_bm25',
            key_field  => 'id',
            text_fields => '{text}',
            json_fields => '{metadata}'
        )
        """
    )


def downgrade() -> None:
    op.execute("CALL paradedb.drop_bm25('idx_ltm_bm25')")
    op.execute("DROP TABLE IF EXISTS long_term_memories")
```

- [ ] **Step 4: 起库并跑迁移（手动验证）**

Run:
```bash
docker compose up -d
uv run alembic upgrade head
```
Expected: 输出 `Running upgrade -> 0001`，无报错。若 `paradedb.create_bm25` 报「已存在」，说明库非干净——先 `alembic downgrade base` 或重建容器卷。

- [ ] **Step 5: 写迁移结果测试**

`tests/test_migration.py`:
```python
import psycopg
import pytest

from rag.config import get_settings


@pytest.mark.integration
def test_table_and_extensions_exist():
    s = get_settings()
    with psycopg.connect(s.pg_async_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass('public.long_term_memories')"
            )
            assert cur.fetchone()[0] == "long_term_memories"
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None
```

- [ ] **Step 6: 运行集成测试**

Run: `pytest tests/test_migration.py -v -m integration`
Expected: PASS（需 Step 4 已 upgrade）。

- [ ] **Step 7: Commit**

```bash
git add alembic.ini alembic/ tests/test_migration.py
git commit -m "feat: alembic 初始迁移（扩展+表+索引，错误不静默）"
```

---

### Task 3: db 异步层（postgres + redis）

**Files:**
- Modify: `rag/db/postgres.py`（整体重写）
- Modify: `rag/db/redis.py`（整体重写）
- Modify: `tests/conftest.py`（补 db fixture）
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `rag.config.Settings`
- Produces:
  - `rag.db.postgres.create_pg_pool(settings) -> AsyncConnectionPool`（已 `open`，每连接注册 pgvector）
  - `rag.db.postgres.get_cursor(pool)` — `@asynccontextmanager`，yield dict-row async cursor，块结束 commit / 异常 rollback。
  - `rag.db.redis.create_redis_client(settings) -> redis.asyncio.Redis`

- [ ] **Step 1: 写失败测试**

`tests/test_db.py`:
```python
import pytest

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.db.redis import create_redis_client


@pytest.mark.integration
async def test_get_cursor_returns_dict_row():
    pool = await create_pg_pool(get_settings())
    try:
        async with get_cursor(pool) as cur:
            await cur.execute("SELECT 1 AS one")
            row = await cur.fetchone()
            assert row["one"] == 1
    finally:
        await pool.close()


@pytest.mark.integration
async def test_redis_ping():
    client = create_redis_client(get_settings())
    try:
        assert await client.ping() is True
    finally:
        await client.aclose()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_db.py -v -m integration`
Expected: FAIL（`ImportError: create_pg_pool`）

- [ ] **Step 3: 写 postgres 异步实现**

`rag/db/postgres.py`（整体替换）:
```python
from contextlib import asynccontextmanager

from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from rag.config import Settings


async def _configure(conn) -> None:
    """每条连接注册 pgvector，使 Python list 可作为 vector 绑定。"""
    await register_vector_async(conn)


async def create_pg_pool(settings: Settings) -> AsyncConnectionPool:
    """创建并打开异步连接池（不在 import 期调用）。"""
    pool = AsyncConnectionPool(
        conninfo=settings.pg_async_dsn,
        min_size=settings.pg_pool_min,
        max_size=settings.pg_pool_max,
        open=False,
        configure=_configure,
    )
    await pool.open()
    return pool


@asynccontextmanager
async def get_cursor(pool: AsyncConnectionPool):
    """获取 dict-row 异步游标；块正常结束自动 commit，异常 rollback。"""
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            yield cur
```

> psycopg3 的 `pool.connection()` 块正常退出自动 commit、异常自动 rollback，无需手写。

- [ ] **Step 4: 写 redis 异步实现**

`rag/db/redis.py`（整体替换）:
```python
import redis.asyncio as redis

from rag.config import Settings


def create_redis_client(settings: Settings) -> redis.Redis:
    """创建异步 Redis 客户端（连接池复用，解码为 str）。"""
    pool = redis.ConnectionPool(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password or None,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    return redis.Redis(connection_pool=pool)
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_db.py -v -m integration`
Expected: PASS（需 Task 2 已 upgrade、docker 在跑）。

- [ ] **Step 6: Commit**

```bash
git add rag/db/postgres.py rag/db/redis.py tests/test_db.py
git commit -m "feat: db 异步层（psycopg3 AsyncConnectionPool + redis.asyncio）"
```

---

### Task 4: embedding 异步化 + 韧性

**Files:**
- Modify: `rag/models/embedding.py`（整体重写）
- Test: `tests/test_embedding.py`

**Interfaces:**
- Consumes: `rag.config.Settings`、`rag.common.exception`（`from_http_error`/`is_retryable`）
- Produces:
  - `rag.models.embedding.EmbeddingModel(settings)`
  - `EmbeddingModel.embed(texts: list[str]) -> list[list[float]]`（async，顺序与输入一致，带重试）

- [ ] **Step 1: 写失败测试**（用 fake，不打真实 API）

`tests/test_embedding.py`:
```python
import pytest

from rag.config import Settings
from rag.models.embedding import EmbeddingModel


class _FakeItem:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeEmbeddings:
    def __init__(self, calls):
        self._calls = calls

    async def create(self, **kwargs):
        self._calls.append(kwargs)
        # 故意乱序返回，验证按 index 排序
        return _FakeResp([_FakeItem(1, [0.2]), _FakeItem(0, [0.1])])


async def test_embed_orders_by_index():
    m = EmbeddingModel(Settings())
    calls = []
    m._client.embeddings = _FakeEmbeddings(calls)  # 注入 fake
    out = await m.embed(["a", "b"])
    assert out == [[0.1], [0.2]]
    assert calls[0]["input"] == ["a", "b"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_embedding.py -v`
Expected: FAIL（`ImportError: EmbeddingModel`）

- [ ] **Step 3: 写实现**

`rag/models/embedding.py`（整体替换）:
```python
import logging

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.common.exception import is_retryable
from rag.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """异步 embedding 模型，带重试与超时。"""

    def __init__(self, settings: Settings):
        self._client = AsyncOpenAI(
            api_key=settings.embedding_key,
            base_url=settings.embedding_url,
            timeout=30,
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                rsp = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dim,
                    encoding_format="float",
                )
        ordered = sorted(rsp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_embedding.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rag/models/embedding.py tests/test_embedding.py
git commit -m "feat: embedding 异步化 + 重试/超时"
```

---

### Task 5: models base + normal 异步化

**Files:**
- Modify: `rag/models/base.py`
- Modify: `rag/models/normal.py`
- Test: `tests/test_normal.py`

**Interfaces:**
- Consumes: `rag.config.Settings`、`rag.common.exception`
- Produces:
  - `rag.models.base.ChatModel`（抽象：`async def ainvoke(messages) -> str`、`def astream(messages)` async-gen）
  - `rag.models.normal.NormalModel(settings)`，方法 `async def ainvoke(messages) -> str`、`async def astream(messages)`、保留错误分发 `dispatch_error`。

- [ ] **Step 1: 写失败测试**（fake 底层 model，不打真实 API）

`tests/test_normal.py`:
```python
from rag.config import Settings
from rag.models.normal import NormalModel


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    async def ainvoke(self, messages):
        return _FakeMsg("answer")


async def test_ainvoke_returns_content():
    m = NormalModel(Settings())
    m._model = _FakeModel()
    assert await m.ainvoke(["hi"]) == "answer"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_normal.py -v`
Expected: FAIL（`NormalModel.__init__` 仍读 env / 无 ainvoke）

- [ ] **Step 3: 写 base 抽象**

`rag/models/base.py`（整体替换）:
```python
from abc import ABC, abstractmethod
from typing import Any


class ChatModel(ABC):
    @abstractmethod
    async def ainvoke(self, messages: list[Any]) -> str: ...

    @abstractmethod
    def astream(self, messages: list[Any]):
        """async generator of chunks"""
        ...
```

- [ ] **Step 4: 写 normal 实现**

`rag/models/normal.py`（整体替换）:
```python
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from rag.common.exception import LLMException, from_http_error, is_retryable
from rag.config import Settings
from rag.models.base import ChatModel

logger = logging.getLogger(__name__)

ErrorHandler = Callable[[LLMException], None]
ErrorHandlers = dict[int | str, ErrorHandler]


def dispatch_error(exc: LLMException, handlers: ErrorHandlers | None) -> None:
    """按 status_code 分发，支持 '*' 通配。"""
    if not handlers:
        return
    handler = handlers.get(exc.status_code) or handlers.get("*")
    if handler:
        try:
            handler(exc)
        except Exception:
            logger.exception("error handler failed")


def _extract_status_code(exc: BaseException) -> int:
    if hasattr(exc, "status_code"):
        return exc.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code
    return 0


class NormalModel(ChatModel):
    def __init__(self, settings: Settings):
        self._model = ChatOpenAI(
            api_key=settings.model_key,
            model=settings.model_name,
            base_url=settings.model_url,
            temperature=0,
            seed=42,
        )
        self._model_name = settings.model_name

    def bind_tools(self, tools: list[BaseTool]) -> None:
        self._model = self._model.bind_tools(tools)

    @asynccontextmanager
    async def _translate(self):
        try:
            yield
        except LLMException:
            raise
        except Exception as e:
            code = _extract_status_code(e)
            if code:
                raise from_http_error(code, str(e), model=self._model_name) from e
            raise

    async def ainvoke(self, messages: list[BaseMessage | str]) -> str:
        """带重试的异步调用。"""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
            retry=retry_if_exception(is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                async with self._translate():
                    rsp = await self._model.ainvoke(messages)
                    return rsp.content
        raise AssertionError("unreachable")

    async def astream(self, messages: list[BaseMessage | str]):
        """单次异步流式（不重试，流式中途重试语义复杂，留待 C）。"""
        async with self._translate():
            async for chunk in self._model.astream(messages):
                yield chunk
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_normal.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add rag/models/base.py rag/models/normal.py tests/test_normal.py
git commit -m "feat: ChatModel/NormalModel 异步化 + 重试"
```

---

### Task 6: memory adapters 异步化 + 统一 search 签名

**Files:**
- Modify: `rag/memory/adapters/base.py`
- Modify: `rag/memory/adapters/long_term_pgsql.py`
- Modify: `rag/memory/adapters/short_term_redis.py`
- Modify: `tests/test_memory.py`（修残缺语法 + 短期往返测试）

**Interfaces:**
- Consumes: `get_cursor`、`EmbeddingModel.embed`、`redis.asyncio.Redis`
- Produces:
  - `LongTermMemoryAdapter` / `ShortTermMemoryAdapter`（全 async；search 签名见 Global Constraints）
  - `PgVectorLongTermMemory(pool, embedding)`
  - `RedisShortTermMemory(client, max_messages=50, ttl_seconds=86400)`，方法 `add/get_recent/clear` 全 async

- [ ] **Step 1: 修 tests/test_memory.py 残缺语法 + 写短期往返测试**

`tests/test_memory.py`（整体替换；长期测试在 Task 7 加）:
```python
import pytest

from rag.config import get_settings
from rag.db.redis import create_redis_client
from rag.memory.adapters.short_term_redis import RedisShortTermMemory


@pytest.mark.integration
async def test_short_term_roundtrip():
    client = create_redis_client(get_settings())
    adapter = RedisShortTermMemory(client)
    sid = "test-session-stm"
    try:
        await adapter.clear(sid)
        await adapter.add(sid, "hello")
        await adapter.add(sid, "world")
        recent = await adapter.get_recent(sid, n=10)
        assert [r["text"] for r in recent] == ["hello", "world"]
    finally:
        await adapter.clear(sid)
        await client.aclose()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_memory.py -v -m integration`
Expected: FAIL（`RedisShortTermMemory` 还是同步 / 构造签名不符）

- [ ] **Step 3: 写 base 异步抽象**

`rag/memory/adapters/base.py`（整体替换）:
```python
from abc import ABC, abstractmethod


class ShortTermMemoryAdapter(ABC):
    """短期记忆适配器 — 会话级，轻量快速，自动过期。"""

    @abstractmethod
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> None: ...

    @abstractmethod
    async def get_recent(self, session_id: str, n: int = 10) -> list[dict]: ...

    @abstractmethod
    async def clear(self, session_id: str) -> None: ...


class LongTermMemoryAdapter(ABC):
    """长期记忆适配器 — 持久化，向量 + BM25 检索。"""

    @abstractmethod
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str: ...

    @abstractmethod
    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None: ...

    @abstractmethod
    async def delete(self, memory_id: str) -> None: ...
```

> 注：`bm25_search` / `get_by_time` 属知识库检索（B），本轮从抽象中移除，避免空实现。

- [ ] **Step 4: 写 long_term 异步实现**

`rag/memory/adapters/long_term_pgsql.py`（整体替换）:
```python
import uuid

from psycopg.types.json import Json

from rag.db.postgres import get_cursor
from rag.memory.adapters.base import LongTermMemoryAdapter
from rag.models.embedding import EmbeddingModel


class PgVectorLongTermMemory(LongTermMemoryAdapter):
    """基于 PostgreSQL + pgvector 的长期记忆（建表/索引由 alembic 负责）。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        memory_id = str(uuid.uuid4())
        embedding = (await self._embedding.embed([text]))[0]
        merged = {"session_id": session_id, **(metadata or {})}
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                """
                INSERT INTO long_term_memories (id, text, embedding, metadata)
                VALUES (%(id)s, %(text)s, %(embedding)s, %(metadata)s)
                """,
                {"id": memory_id, "text": text, "embedding": embedding, "metadata": Json(merged)},
            )
        return memory_id

    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        query_embedding = (await self._embedding.embed([query]))[0]
        sql = """
            SELECT id, text, metadata, created_at,
                   1 - (embedding <=> %(embedding)s) AS similarity
            FROM long_term_memories
            WHERE metadata->>'session_id' = %(session_id)s
        """
        params: dict = {
            "embedding": query_embedding,
            "session_id": session_id,
            "top_k": top_k,
        }
        if filters:
            for i, (key, value) in enumerate(filters.items()):
                sql += f" AND metadata->>%(key_{i})s = %(filter_{i})s"
                params[f"key_{i}"] = key
                params[f"filter_{i}"] = str(value)
        sql += " ORDER BY embedding <=> %(embedding)s LIMIT %(top_k)s"
        async with get_cursor(self._pool) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()

    async def update(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        embedding = (await self._embedding.embed([text]))[0]
        payload = Json(metadata) if metadata is not None else None
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                """
                UPDATE long_term_memories
                SET text = %(text)s,
                    embedding = %(embedding)s,
                    metadata = CASE
                        WHEN %(metadata)s IS NULL THEN long_term_memories.metadata
                        ELSE COALESCE(long_term_memories.metadata, '{}'::jsonb) || %(metadata)s::jsonb
                    END,
                    updated_at = now()
                WHERE id = %(id)s
                """,
                {"id": memory_id, "text": text, "embedding": embedding, "metadata": payload},
            )

    async def delete(self, memory_id: str) -> None:
        async with get_cursor(self._pool) as cur:
            await cur.execute(
                "DELETE FROM long_term_memories WHERE id = %(id)s", {"id": memory_id}
            )
```

> `search` 现在统一以 `session_id` 作为主过滤（直接进 SQL），`filters` 作为附加 metadata 条件，修掉了原先「调用方把 session_id 塞进 filters」的混乱。

- [ ] **Step 5: 写 short_term 异步实现**

`rag/memory/adapters/short_term_redis.py`（整体替换）:
```python
import json

from rag.memory.adapters.base import ShortTermMemoryAdapter


class RedisShortTermMemory(ShortTermMemoryAdapter):
    """基于 Redis List 的短期记忆，按 session 隔离，自动 TTL。"""

    def __init__(self, client, max_messages: int = 50, ttl_seconds: int = 24 * 60 * 60):
        self._redis = client
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        key = self._key(session_id)
        payload = json.dumps({"text": text, "metadata": metadata or {}}, ensure_ascii=False)
        pipe = self._redis.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        await pipe.execute()

    async def get_recent(self, session_id: str, n: int = 10) -> list[dict]:
        if n <= 0:
            return []
        raw = await self._redis.lrange(self._key(session_id), -n, -1)
        return [json.loads(item) for item in raw]

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_memory.py -v -m integration`
Expected: PASS（短期往返）

- [ ] **Step 7: Commit**

```bash
git add rag/memory/adapters/ tests/test_memory.py
git commit -m "feat: memory adapters 异步化 + 统一 search 签名"
```

---

### Task 7: memory manager 异步化（add_message / 去单例 / 修 search）

**Files:**
- Modify: `rag/memory/manager.py`（整体重写）
- Modify: `rag/memory/__init__.py`
- Modify: `tests/test_memory.py`（补长期往返 + manager 测试）

**Interfaces:**
- Consumes: `LongTermMemoryAdapter`、`ShortTermMemoryAdapter`
- Produces:
  - `MemoryManager(long_term, short_term=None)`
  - `async def add(session_id, text, metadata=None) -> str`（写长期）
  - `async def add_message(session_id, text, metadata=None) -> None`（写短期）
  - `async def search(session_id, query, top_k=5, filters=None) -> list[dict]`
  - `async def get_recent_messages(session_id, n=10) -> list[dict]`
  - `async def clear_session(session_id) -> None`
  - **不再导出** `get_memory_manager`

- [ ] **Step 1: 写失败测试（manager 用 fake adapter，纯单元）**

`tests/test_memory.py` 追加：
```python
from rag.memory.manager import MemoryManager


class _FakeLong:
    def __init__(self):
        self.calls = []

    async def search(self, session_id, query, top_k=5, filters=None):
        self.calls.append((session_id, query, top_k, filters))
        return [{"text": "L"}]


class _FakeShort:
    def __init__(self):
        self.added = []

    async def add(self, session_id, text, metadata=None):
        self.added.append((session_id, text))

    async def get_recent(self, session_id, n=10):
        return [{"text": "S"}]


async def test_manager_search_passes_args_in_order():
    long = _FakeLong()
    mgr = MemoryManager(long_term=long, short_term=_FakeShort())
    out = await mgr.search("sid1", "q1")
    assert out == [{"text": "L"}]
    assert long.calls == [("sid1", "q1", 5, None)]


async def test_manager_add_message_writes_short_term():
    short = _FakeShort()
    mgr = MemoryManager(long_term=_FakeLong(), short_term=short)
    await mgr.add_message("sid1", "hi")
    assert short.added == [("sid1", "hi")]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_memory.py -v` （不带 -m，跑单元）
Expected: FAIL（`MemoryManager.search` 旧签名/无 `add_message`）

- [ ] **Step 3: 写 manager 实现**

`rag/memory/manager.py`（整体替换）:
```python
from rag.memory.adapters.base import LongTermMemoryAdapter, ShortTermMemoryAdapter


class MemoryManager:
    """组合短期/长期记忆适配器，统一管理记忆存取。"""

    def __init__(
        self,
        long_term: LongTermMemoryAdapter,
        short_term: ShortTermMemoryAdapter | None = None,
    ):
        self._long_term = long_term
        self._short_term = short_term

    # ── 长期记忆 ──
    async def add(self, session_id: str, text: str, metadata: dict | None = None) -> str:
        return await self._long_term.add(session_id, text, metadata)

    async def search(
        self, session_id: str, query: str, top_k: int = 5, filters: dict | None = None
    ) -> list[dict]:
        return await self._long_term.search(session_id, query, top_k, filters)

    # ── 短期记忆 ──
    async def add_message(
        self, session_id: str, text: str, metadata: dict | None = None
    ) -> None:
        if self._short_term:
            await self._short_term.add(session_id, text, metadata)

    async def get_recent_messages(self, session_id: str, n: int = 10) -> list[dict]:
        if self._short_term:
            return await self._short_term.get_recent(session_id, n)
        return []

    async def clear_session(self, session_id: str) -> None:
        if self._short_term:
            await self._short_term.clear(session_id)
```

- [ ] **Step 4: 改 memory/__init__.py（去掉单例导出）**

`rag/memory/__init__.py`（整体替换）:
```python
from rag.memory.adapters.base import LongTermMemoryAdapter, ShortTermMemoryAdapter
from rag.memory.manager import MemoryManager

__all__ = [
    "MemoryManager",
    "LongTermMemoryAdapter",
    "ShortTermMemoryAdapter",
]
```

- [ ] **Step 5: 补长期往返集成测试**

`tests/test_memory.py` 追加：
```python
from rag.db.postgres import create_pg_pool
from rag.models.embedding import EmbeddingModel
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory


class _StubEmbedding(EmbeddingModel):
    def __init__(self):
        pass  # 跳过真实 client

    async def embed(self, texts):
        return [[0.01] * 1024 for _ in texts]


@pytest.mark.integration
async def test_long_term_add_then_search():
    pool = await create_pg_pool(get_settings())
    adapter = PgVectorLongTermMemory(pool, _StubEmbedding())
    sid = "test-session-ltm"
    try:
        mid = await adapter.add(sid, "关税申报流程")
        assert mid
        results = await adapter.search(sid, "关税")
        assert any(r["text"] == "关税申报流程" for r in results)
    finally:
        async with get_cursor_cleanup(pool, sid):
            pass
        await pool.close()
```
并在文件顶部补清理辅助：
```python
from contextlib import asynccontextmanager
from rag.db.postgres import get_cursor


@asynccontextmanager
async def get_cursor_cleanup(pool, session_id):
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM long_term_memories WHERE metadata->>'session_id' = %(sid)s",
            {"sid": session_id},
        )
    yield
```

- [ ] **Step 6: 运行确认通过**

Run: `pytest tests/test_memory.py -v`（单元）然后 `pytest tests/test_memory.py -v -m integration`（集成）
Expected: 全 PASS

- [ ] **Step 7: Commit**

```bash
git add rag/memory/manager.py rag/memory/__init__.py tests/test_memory.py
git commit -m "feat: MemoryManager 异步化 + add_message + 去单例 + 修 search"
```

---

### Task 8: AppResources + lifespan

**Files:**
- Create: `rag/resources.py`
- Test: `tests/test_resources.py`

**Interfaces:**
- Consumes: `create_pg_pool`、`create_redis_client`、`NormalModel`、`EmbeddingModel`、`PgVectorLongTermMemory`、`RedisShortTermMemory`、`MemoryManager`、`get_settings`
- Produces:
  - `rag.resources.AppResources`（dataclass：`pool, redis, llm, embedding, memory_manager`）
  - `classmethod async AppResources.startup(settings) -> AppResources`
  - `async AppResources.aclose() -> None`
  - `rag.resources.lifespan(settings=None)` — `@asynccontextmanager`，yield `AppResources`

- [ ] **Step 1: 写失败测试（集成：真起资源再关）**

`tests/test_resources.py`:
```python
import pytest

from rag.memory.manager import MemoryManager
from rag.resources import AppResources, lifespan


@pytest.mark.integration
async def test_lifespan_builds_and_closes():
    async with lifespan() as res:
        assert isinstance(res, AppResources)
        assert isinstance(res.memory_manager, MemoryManager)
        # 池可用
        from rag.db.postgres import get_cursor
        async with get_cursor(res.pool) as cur:
            await cur.execute("SELECT 1 AS one")
            assert (await cur.fetchone())["one"] == 1
    # 退出后池应已关闭
    assert res.pool.closed
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_resources.py -v -m integration`
Expected: FAIL（`ImportError: rag.resources`）

- [ ] **Step 3: 写实现**

`rag/resources.py`:
```python
from contextlib import asynccontextmanager
from dataclasses import dataclass

from rag.config import Settings, get_settings
from rag.db.postgres import create_pg_pool
from rag.db.redis import create_redis_client
from rag.memory.adapters.long_term_pgsql import PgVectorLongTermMemory
from rag.memory.adapters.short_term_redis import RedisShortTermMemory
from rag.memory.manager import MemoryManager
from rag.models.embedding import EmbeddingModel
from rag.models.normal import NormalModel


@dataclass
class AppResources:
    """持有全部 I/O 资源，由 lifespan 构建/释放。"""

    pool: object
    redis: object
    llm: NormalModel
    embedding: EmbeddingModel
    memory_manager: MemoryManager

    @classmethod
    async def startup(cls, settings: Settings) -> "AppResources":
        pool = await create_pg_pool(settings)
        redis = create_redis_client(settings)
        embedding = EmbeddingModel(settings)
        llm = NormalModel(settings)
        memory_manager = MemoryManager(
            long_term=PgVectorLongTermMemory(pool, embedding),
            short_term=RedisShortTermMemory(redis),
        )
        return cls(
            pool=pool,
            redis=redis,
            llm=llm,
            embedding=embedding,
            memory_manager=memory_manager,
        )

    async def aclose(self) -> None:
        await self.pool.close()
        await self.redis.aclose()


@asynccontextmanager
async def lifespan(settings: Settings | None = None):
    settings = settings or get_settings()
    resources = await AppResources.startup(settings)
    try:
        yield resources
    finally:
        await resources.aclose()
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_resources.py -v -m integration`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rag/resources.py tests/test_resources.py
git commit -m "feat: AppResources + lifespan（消除 import 连库）"
```

---

### Task 9: type.py 字段对齐 + 三节点异步化 + 降级

**Files:**
- Modify: `rag/type.py`
- Modify: `rag/nodes/recall_memory/memory.py`
- Modify: `rag/nodes/query/query.py`
- Create: `rag/nodes/persist_memory/__init__.py`
- Create: `rag/nodes/persist_memory/memory.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `MemoryManager`、`NormalModel.ainvoke`、`Runtime[ContextSchema]`
- Produces:
  - `rag.type.MyState`（精简字段）、`rag.type.ContextSchema`（不变）
  - `async def recall_memory(state, runtime) -> MyState`
  - `async def handle_query(state, runtime) -> MyState`
  - `async def persist_memory(state, runtime) -> MyState`

- [ ] **Step 1: 写失败测试（fake runtime/manager/llm，纯单元）**

`tests/test_nodes.py`:
```python
from dataclasses import dataclass

from rag.nodes.recall_memory.memory import recall_memory
from rag.nodes.query.query import handle_query
from rag.nodes.persist_memory.memory import persist_memory


@dataclass
class _Ctx:
    llm: object
    memory_manager: object


class _Runtime:
    def __init__(self, ctx):
        self.context = ctx


class _OkManager:
    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "long-ctx"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "short-ctx"}]

    def __init__(self):
        self.saved = []

    async def add_message(self, session_id, text, metadata=None):
        self.saved.append((session_id, text))


class _BoomManager(_OkManager):
    async def search(self, session_id, query, top_k=5, filters=None):
        raise RuntimeError("db down")


class _OkLLM:
    async def ainvoke(self, messages):
        return "generated-answer"


async def test_recall_memory_builds_context():
    rt = _Runtime(_Ctx(llm=_OkLLM(), memory_manager=_OkManager()))
    state = {"session_id": "s", "raw_query": "q"}
    out = await recall_memory(state, rt)
    assert "short-ctx" in out["context"] and "long-ctx" in out["context"]


async def test_recall_memory_degrades_on_failure():
    rt = _Runtime(_Ctx(llm=_OkLLM(), memory_manager=_BoomManager()))
    state = {"session_id": "s", "raw_query": "q"}
    out = await recall_memory(state, rt)
    assert out["context"] == ""  # 降级：不抛，空上下文继续


async def test_handle_query_writes_generated():
    rt = _Runtime(_Ctx(llm=_OkLLM(), memory_manager=_OkManager()))
    state = {"session_id": "s", "raw_query": "q", "context": "c"}
    out = await handle_query(state, rt)
    assert out["generated"] == "generated-answer"


async def test_persist_memory_writes_back():
    mgr = _OkManager()
    rt = _Runtime(_Ctx(llm=_OkLLM(), memory_manager=mgr))
    state = {"session_id": "s", "raw_query": "q", "generated": "a"}
    out = await persist_memory(state, rt)
    assert ("s", "q") in mgr.saved and ("s", "a") in mgr.saved
```

> 注：`get_stream_writer()` 在非 graph 上下文调用会报错。实现里对 writer 调用做容错（见 Step 3 的 `_safe_writer`），使节点可被单元测试直接调用。

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_nodes.py -v`
Expected: FAIL（节点仍同步 / 无 persist_memory / handle_query 写错字段）

- [ ] **Step 3: 写 type.py + 三节点**

`rag/type.py`（整体替换）:
```python
from dataclasses import dataclass
from typing import TypedDict

from rag.memory import MemoryManager
from rag.models.base import ChatModel


class MyState(TypedDict, total=False):
    session_id: str
    raw_query: str    # 输入
    context: str      # recall_memory 产出
    generated: str    # handle_query 产出（最终答案）


@dataclass
class ContextSchema:
    llm: ChatModel
    memory_manager: MemoryManager
```

新增 `rag/nodes/_writer.py`（容错 writer，DRY 给三节点用）:
```python
from langgraph.config import get_stream_writer


def safe_writer():
    """在 graph 外（如单元测试）调用时返回 no-op，避免报错。"""
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_args, **_kwargs: None
```

`rag/nodes/recall_memory/memory.py`（整体替换）:
```python
import logging

from langgraph.runtime import Runtime

from rag.nodes._writer import safe_writer
from rag.type import ContextSchema, MyState

logger = logging.getLogger(__name__)


async def recall_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从长期和短期记忆召回上下文；失败则降级为空上下文。"""
    writer = safe_writer()
    writer("检索记忆中...")
    try:
        mgr = runtime.context.memory_manager
        long_results = await mgr.search(state["session_id"], state["raw_query"])
        short_results = await mgr.get_recent_messages(state["session_id"])
        parts = [r.get("text", "") for r in short_results]
        parts += [r.get("text", "") for r in long_results]
        state["context"] = "\n".join(p for p in parts if p)
    except Exception:
        logger.exception("记忆召回失败，降级为空上下文")
        state["context"] = ""
    return state
```

`rag/nodes/query/query.py`（整体替换）:
```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from rag.nodes._writer import safe_writer
from rag.type import ContextSchema, MyState

system_prompt = """
你是一个专业的关务助手, 你的任务是根据用户的查询, 提供专业的关务信息.
有不确定的地方，例如：他/那么。
从上下文获取信息，改写消息返回
"""


async def handle_query(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """根据上下文生成回答（带重试）。"""
    writer = safe_writer()
    writer("深度思考中")
    llm = runtime.context.llm
    state["generated"] = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
用户查询: {state["raw_query"]}
上下文: {state.get("context", "")}
"""),
    ])
    return state
```

`rag/nodes/persist_memory/__init__.py`:
```python
```

`rag/nodes/persist_memory/memory.py`:
```python
import logging

from langgraph.runtime import Runtime

from rag.type import ContextSchema, MyState

logger = logging.getLogger(__name__)


async def persist_memory(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """把本轮问答写回短期记忆；失败不影响已生成答案。"""
    try:
        mgr = runtime.context.memory_manager
        await mgr.add_message(state["session_id"], state["raw_query"], {"role": "user"})
        if state.get("generated"):
            await mgr.add_message(state["session_id"], state["generated"], {"role": "assistant"})
    except Exception:
        logger.exception("记忆写回失败，忽略")
    return state
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_nodes.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: Commit**

```bash
git add rag/type.py rag/nodes/
git commit -m "feat: 节点异步化 + 降级 + 记忆写回 + MyState 对齐"
```

---

### Task 10: workflow.invoke 异步 + main.py 入口 + 地基冒烟

**Files:**
- Modify: `rag/workflow.py`（整体重写）
- Modify: `main.py`（整体重写）
- Test: `tests/test_foundation.py`

**Interfaces:**
- Consumes: `AppResources`、`lifespan`、`ContextSchema`、三节点
- Produces:
  - `rag.workflow.graph`（import 期编译，无 I/O）
  - `async def rag.workflow.invoke(resources, session_id, query)` — async generator，yield graph chunk

- [ ] **Step 1: 写失败测试（地基冒烟 + 降级，用 fake resources）**

`tests/test_foundation.py`:
```python
from dataclasses import dataclass

from rag.workflow import invoke


class _FakeLLM:
    async def ainvoke(self, messages):
        return "最终答案"


class _FakeManager:
    def __init__(self):
        self.saved = []

    async def search(self, session_id, query, top_k=5, filters=None):
        return [{"text": "L"}]

    async def get_recent_messages(self, session_id, n=10):
        return [{"text": "S"}]

    async def add_message(self, session_id, text, metadata=None):
        self.saved.append((session_id, text))


@dataclass
class _FakeResources:
    llm: object
    memory_manager: object


async def test_invoke_runs_end_to_end():
    mgr = _FakeManager()
    res = _FakeResources(llm=_FakeLLM(), memory_manager=mgr)
    final = {}
    async for chunk in invoke(res, "sess", "我要报关"):
        for node_state in chunk.values():
            if isinstance(node_state, dict):
                final.update(node_state)
    assert final.get("generated") == "最终答案"
    # 写回闭环
    assert ("sess", "最终答案") in mgr.saved
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_foundation.py -v`
Expected: FAIL（`invoke` 仍同步 / 签名不符）

- [ ] **Step 3: 写 workflow 实现**

`rag/workflow.py`（整体替换）:
```python
from langgraph.graph import END, START, StateGraph

from rag.nodes.persist_memory.memory import persist_memory
from rag.nodes.query.query import handle_query
from rag.nodes.recall_memory.memory import recall_memory
from rag.type import ContextSchema, MyState

# ── 构建状态图（import 期纯 CPU，无 I/O）──
builder = StateGraph(MyState, context_schema=ContextSchema)
builder.add_node("recall_memory", recall_memory)
builder.add_node("handle_query", handle_query)
builder.add_node("persist_memory", persist_memory)
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", "persist_memory")
builder.add_edge("persist_memory", END)
graph = builder.compile()


async def invoke(resources, session_id: str, query: str):
    """运行一次工作流，流式产出各节点 chunk。"""
    context = ContextSchema(
        llm=resources.llm,
        memory_manager=resources.memory_manager,
    )
    async for chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
    ):
        yield chunk
```

- [ ] **Step 4: 写 main.py 入口**

`main.py`（整体替换）:
```python
import asyncio

from rag.resources import lifespan
from rag.workflow import invoke


async def main() -> None:
    async with lifespan() as resources:
        async for chunk in invoke(resources, "demo-session", "进口货物如何报关？"):
            print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_foundation.py -v`
Expected: PASS

- [ ] **Step 6: 全量测试 + 端到端手动验证**

Run:
```bash
pytest -v                      # 单元全绿
pytest -v -m integration       # 集成全绿（需 docker + 真实 LLM/embedding 配置）
python main.py                 # 打印流式 chunk，最后含 generated 答案
```
Expected: 单元/集成全 PASS；`python main.py` 端到端跑通打印结果。

- [ ] **Step 7: 验证 import 不连库**

Run: `python -c "import rag.workflow; print('ok no db')"`（停掉 docker 也应成功）
Expected: 打印 `ok no db`，无连接报错（证明 import 期不连库）。

- [ ] **Step 8: Commit**

```bash
git add rag/workflow.py main.py tests/test_foundation.py
git commit -m "feat: workflow.invoke 异步 + main 入口 + 地基冒烟测试"
```

---

## Self-Review

**1. Spec coverage（逐节核对）：**
- §2 架构/生命周期 → Task 8（AppResources/lifespan）、Task 10（import 不连库验证 Step 7）✓
- §2 文件清单 → Task 1-10 全覆盖 ✓
- §3.1 db 层（psycopg3 / 移除 DDL / 移走 bm25/hybrid）→ Task 3 ✓
- §3.2 models（normal async / embedding 韧性）→ Task 4、5 ✓
- §3.3 memory（统一 search / add_message / 去单例）→ Task 6、7 ✓
- §4 节点 + 数据流 + 写回 + 降级 → Task 9 ✓
- §5 alembic → Task 2 ✓
- §6 依赖（含 tenacity/openai 缺失 + 删 psycopg2 + 3.12）→ Task 1 ✓
- §7 安全（.env 出 git / .env.example）→ Task 1 ✓
- §8 测试（修 test_memory / 异步往返 / 冒烟 / 降级）→ Task 6、7、9、10 ✓
- §9 DoD 10 项 → 分布在各 Task 的验证步骤 ✓
- §10 风险（pgvector 注册）→ Task 3 `_configure` 补 `register_vector_async`，并在 Task 1 加 `pgvector` 依赖 ✓

**2. Placeholder scan：** 无 TBD/TODO；每个 code step 均给出完整代码；alembic env.py 改动以「具体位置 + 代码」描述（`alembic init` 生成的模板因版本不同，故以增量说明而非整文件覆盖，属合理）。

**3. Type consistency：**
- `search(session_id, query, top_k, filters)` 在 base / long_term / manager / 节点调用四处一致 ✓
- `EmbeddingModel.embed`、`NormalModel.ainvoke`、`MemoryManager.add_message`、`AppResources` 字段名跨 Task 一致 ✓
- `lifespan` yield `AppResources`，`invoke(resources, ...)` 取 `resources.llm/.memory_manager` 一致 ✓
- `get_cursor(pool)` 签名在 db / 各 adapter / 测试一致 ✓

> 一处刻意取舍：`tests/test_memory.py` Task 7 Step 5 的清理辅助 `get_cursor_cleanup` 与正式 `get_cursor` 同名前缀但用途不同（仅测试清理），已用不同函数名避免冲突。
