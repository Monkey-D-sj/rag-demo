# 语义缓存 (Semantic Cache) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对语义相近的重复提问复用历史答案,跳过检索链与 LLM 生成;命中率与节省成本可从 `llm_call_log` 对账。

**Architecture:** 新增 `SemanticCache` 管理器(pgvector 表 + embedding 相似度查询),LangGraph 图中在 `handle_query` 之后插入 `cache_lookup` 节点(命中直达 `add_memory`),`generate` 之后插入 `cache_store` 节点回写。全局作用域(聊天链路为全库检索,无 KB 维度);任何文档入库成功即整表失效 + TTL 兜底。

**Tech Stack:** Python 3.12 / psycopg3 async + pgvector / LangGraph / Alembic / pytest

**Spec:** `docs/superpowers/specs/2026-07-17-semantic-cache-design.md`

## Global Constraints

- 中文注释一律用半角标点(项目约定,见 memory `halfwidth-punctuation-convention`)
- 禁止重复声明已有的变量(CLAUDE.md 代码规范)
- 日志一律 `from rag.common.logging import get_logger`,不用 `logging.getLogger`
- 新配置默认 `SEMANTIC_CACHE_ENABLED=false`(opt-in 惯例)
- 缓存路径上的一切故障必须降级(lookup 失败=未命中,store 失败=仅 warning),绝不阻断主链路
- 单测命令:`uv run pytest tests/<file> -v`(pyproject addopts 已跳过 integration/eval)
- 注意:`tests/test_document_pipeline.py`、`tests/test_worker.py` 存在 18 个**既有**失败(分支遗留,与本计划无关)。本计划新增的测试必须全部通过;既有失败不需要修,也不得被新改动扩大

---

### Task 1: 配置项 + 数据库迁移

**Files:**
- Modify: `rag/config.py`(在 `# ── Loki ──` 块之前插入)
- Create: `alembic/versions/0011_semantic_cache.py`
- Test: `tests/test_migration.py`(追加 integration 用例)

**Interfaces:**
- Produces: `Settings.SEMANTIC_CACHE_ENABLED: bool = False`、`Settings.SEMANTIC_CACHE_SIM_THRESHOLD: float = 0.95`、`Settings.SEMANTIC_CACHE_TTL_HOURS: int = 168`;表 `semantic_cache(id, question, answer, citations, embedding, hit_count, created_at)`

- [ ] **Step 1: 写 migration**

创建 `alembic/versions/0011_semantic_cache.py`:

```python
"""semantic_cache - 语义缓存(答案级,全局作用域)

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-17
"""
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id         BIGSERIAL PRIMARY KEY,
            question   text NOT NULL,
            answer     text NOT NULL,
            citations  jsonb NOT NULL DEFAULT '[]'::jsonb,
            embedding  vector(1024) NOT NULL,
            hit_count  int NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding"
        " ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS semantic_cache")
```

- [ ] **Step 2: 加配置项**

`rag/config.py`,在 `# ── Loki ──` 块之前插入:

```python
    # ── 语义缓存(handle_query 之后,答案级,全局作用域)──
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_SIM_THRESHOLD: float = 0.95  # 余弦相似度命中阈值
    SEMANTIC_CACHE_TTL_HOURS: int = 168         # 缓存有效期(7 天)
```

- [ ] **Step 3: 追加迁移 integration 测试**

`tests/test_migration.py` 末尾追加:

```python
@pytest.mark.integration
def test_semantic_cache_table_exists():
    s = get_settings()
    with psycopg.connect(s.pg_async_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.semantic_cache')")
            assert cur.fetchone()[0] == "semantic_cache"
            cur.execute(
                "SELECT indexname FROM pg_indexes"
                " WHERE tablename = 'semantic_cache'"
                " AND indexname = 'idx_semantic_cache_embedding'"
            )
            assert cur.fetchone() is not None
```

- [ ] **Step 4: 验证配置默认值(单测方式跑通即可)**

Run: `uv run python -c "from rag.config import Settings; s=Settings(); print(s.SEMANTIC_CACHE_ENABLED, s.SEMANTIC_CACHE_SIM_THRESHOLD, s.SEMANTIC_CACHE_TTL_HOURS)"`
Expected: `False 0.95 168`

(integration 迁移测试需 docker pg,本地有环境则跑 `uv run alembic upgrade head && uv run pytest tests/test_migration.py -v -m integration`;无环境跳过,CI eval 工作流会覆盖)

- [ ] **Step 5: Commit**

```bash
git add rag/config.py alembic/versions/0011_semantic_cache.py tests/test_migration.py
git commit -m "feat(cache): 语义缓存配置项与 semantic_cache 表迁移"
```

---

### Task 2: SemanticCache 管理器 + 协议 + 状态字段

**Files:**
- Create: `rag/agent/cache/__init__.py`
- Create: `rag/agent/cache/manager.py`
- Modify: `rag/agent/type.py`(加 `SemanticCacheProtocol`、`ContextSchema.semantic_cache`、`MyState.cache_hit`)
- Test: `tests/test_semantic_cache.py`(新建)

**Interfaces:**
- Consumes: Task 1 的表结构与配置项;`rag.models.embedding.EmbeddingModel.embed(texts: list[str]) -> list[list[float]]`;`rag.db.postgres.get_cursor(pool)`;`rag.governance.usage.UsageRecorder.record(CallRecord)`
- Produces:
  - `SemanticCache(pool, embedding, *, threshold: float, ttl_hours: int, recorder=None)`
  - `async SemanticCache.lookup(query: str, session_id: str | None = None) -> dict | None`(返回 `{"answer": str, "citations": list}`,内部吞异常)
  - `async SemanticCache.store(query: str, answer: str, citations: list) -> None`(内部吞异常)
  - 模块级 `async clear_semantic_cache(pool) -> None`、`async purge_expired(pool, ttl_hours: int) -> None`(会抛异常,由调用方决定降级)
  - `SemanticCacheProtocol`(runtime_checkable,含 lookup/store)
  - `ContextSchema.semantic_cache: SemanticCacheProtocol | None = None`
  - `MyState.cache_hit: bool`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_semantic_cache.py`:

```python
from unittest.mock import MagicMock

from rag.agent.cache.manager import (
    SemanticCache,
    clear_semantic_cache,
    purge_expired,
)


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


class _FakeCursor:
    """模拟 psycopg cursor,记录全部 execute 调用,fetchone 按序出队。"""

    def __init__(self, rows=None):
        self.calls = []          # [(sql, params), ...]
        self._rows = list(rows or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, sql, params=None):
        self.calls.append((sql, params or {}))

    async def fetchone(self):
        return self._rows.pop(0) if self._rows else None


def _make_cache(cur, monkeypatch, recorder=None):
    monkeypatch.setattr(
        "rag.agent.cache.manager.get_cursor", lambda p: cur
    )
    return SemanticCache(
        MagicMock(), _FakeEmbedding(),
        threshold=0.95, ttl_hours=168, recorder=recorder,
    )


async def test_lookup_hit_above_threshold(monkeypatch):
    row = {"id": 1, "answer": "答案A", "citations": [{"index": 1}], "similarity": 0.97}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    hit = await cache.lookup("孙悟空是谁")

    assert hit == {"answer": "答案A", "citations": [{"index": 1}]}
    # 第二条 SQL 是 hit_count 自增
    assert any("hit_count" in sql for sql, _ in cur.calls)


async def test_lookup_miss_below_threshold(monkeypatch):
    row = {"id": 1, "answer": "答案A", "citations": [], "similarity": 0.90}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    assert await cache.lookup("孙悟空是谁") is None
    # 未命中不得自增 hit_count
    assert not any("hit_count" in sql for sql, _ in cur.calls)


async def test_lookup_empty_table_returns_none(monkeypatch):
    cur = _FakeCursor(rows=[])
    cache = _make_cache(cur, monkeypatch)
    assert await cache.lookup("孙悟空是谁") is None


async def test_lookup_db_error_degrades_to_none(monkeypatch):
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache = _make_cache(_BoomCursor(), monkeypatch)
    assert await cache.lookup("孙悟空是谁") is None  # 不抛异常


async def test_lookup_hit_records_usage(monkeypatch):
    row = {"id": 1, "answer": "A", "citations": [], "similarity": 0.99}
    recorder = MagicMock()
    cache = _make_cache(_FakeCursor(rows=[row]), monkeypatch, recorder=recorder)

    await cache.lookup("q", session_id="s1")

    assert recorder.record.call_count == 1
    rec = recorder.record.call_args[0][0]
    assert rec.call_type == "semantic_cache"
    assert rec.status == "cache_hit"
    assert rec.session_id == "s1"


async def test_store_inserts_when_no_near_duplicate(monkeypatch):
    cur = _FakeCursor(rows=[None])  # 查重无结果
    cache = _make_cache(cur, monkeypatch)

    await cache.store("q", "answer", [{"index": 1}])

    assert any("INSERT INTO semantic_cache" in sql for sql, _ in cur.calls)


async def test_store_skips_near_duplicate(monkeypatch):
    row = {"id": 1, "answer": "old", "citations": [], "similarity": 0.98}
    cur = _FakeCursor(rows=[row])
    cache = _make_cache(cur, monkeypatch)

    await cache.store("q", "answer", [])

    assert not any("INSERT INTO" in sql for sql, _ in cur.calls)


async def test_store_db_error_swallowed(monkeypatch):
    class _BoomCursor(_FakeCursor):
        async def execute(self, sql, params=None):
            raise RuntimeError("db down")

    cache = _make_cache(_BoomCursor(), monkeypatch)
    await cache.store("q", "a", [])  # 不抛异常即通过


async def test_clear_and_purge_sql(monkeypatch):
    cur = _FakeCursor()
    monkeypatch.setattr("rag.agent.cache.manager.get_cursor", lambda p: cur)

    await clear_semantic_cache(MagicMock())
    await purge_expired(MagicMock(), ttl_hours=168)

    assert any("DELETE FROM semantic_cache" in sql and "created_at" not in sql
               for sql, _ in cur.calls)
    assert any("DELETE FROM semantic_cache" in sql and "created_at" in sql
               for sql, _ in cur.calls)


def test_protocol_conformance():
    from rag.agent.type import SemanticCacheProtocol

    cache = SemanticCache(MagicMock(), _FakeEmbedding(), threshold=0.95, ttl_hours=1)
    assert isinstance(cache, SemanticCacheProtocol)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_semantic_cache.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'rag.agent.cache'`

- [ ] **Step 3: 实现 manager**

创建 `rag/agent/cache/manager.py`:

```python
import time

from pgvector import Vector
from psycopg.types.json import Json

from rag.common.logging import get_logger
from rag.db.postgres import get_cursor
from rag.governance.usage import CallRecord

logger = get_logger()

_LOOKUP_SQL = """
SELECT id, answer, citations,
       1 - (embedding <=> %(vec)s) AS similarity
FROM semantic_cache
WHERE created_at > now() - make_interval(hours => %(ttl)s)
ORDER BY embedding <=> %(vec)s
LIMIT 1
"""

_INSERT_SQL = """
INSERT INTO semantic_cache (question, answer, citations, embedding)
VALUES (%(question)s, %(answer)s, %(citations)s, %(vec)s)
"""


async def clear_semantic_cache(pool) -> None:
    """整表清空。文档入库成功后调用:答案可能引用任意 KB,局部失效不正确。"""
    async with get_cursor(pool) as cur:
        await cur.execute("DELETE FROM semantic_cache")


async def purge_expired(pool, ttl_hours: int) -> None:
    """物理删除过期行,worker cron 定期调用。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM semantic_cache"
            " WHERE created_at <= now() - make_interval(hours => %(ttl)s)",
            {"ttl": ttl_hours},
        )


class SemanticCache:
    """答案级语义缓存:以 rewrite_query 的 embedding 为 key,pgvector 相似度命中。

    lookup/store 内部吞掉一切异常(缓存故障绝不阻断主链路):
    lookup 失败返回 None(降级为未命中),store 失败仅记 warning。
    """

    def __init__(
        self,
        pool,
        embedding,
        *,
        threshold: float = 0.95,
        ttl_hours: int = 168,
        recorder=None,
    ) -> None:
        self._pool = pool
        self._embedding = embedding
        self._threshold = threshold
        self._ttl_hours = ttl_hours
        self._recorder = recorder

    async def _nearest(self, cur, query: str) -> dict | None:
        vec = Vector((await self._embedding.embed([query]))[0])
        await cur.execute(_LOOKUP_SQL, {"vec": vec, "ttl": self._ttl_hours})
        return await cur.fetchone()

    async def lookup(self, query: str, session_id: str | None = None) -> dict | None:
        start = time.monotonic()
        try:
            async with get_cursor(self._pool) as cur:
                row = await self._nearest(cur, query)
                if row is None or row["similarity"] < self._threshold:
                    return None
                await cur.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1"
                    " WHERE id = %(id)s",
                    {"id": row["id"]},
                )
        except Exception:  # noqa: BLE001 - 缓存故障降级为未命中
            logger.warning("语义缓存查询失败,降级为未命中", exc_info=True)
            return None

        if self._recorder is not None:
            self._recorder.record(CallRecord(
                call_type="semantic_cache",
                model="semantic-cache",
                status="cache_hit",
                attempts=1,
                latency_ms=int((time.monotonic() - start) * 1000),
                session_id=session_id,
            ))
        logger.info("语义缓存命中: similarity=%.4f", row["similarity"])
        return {"answer": row["answer"], "citations": row["citations"]}

    async def store(self, query: str, answer: str, citations: list) -> None:
        try:
            async with get_cursor(self._pool) as cur:
                row = await self._nearest(cur, query)
                if row is not None and row["similarity"] >= self._threshold:
                    return  # 已有近重复条目,跳过插入防膨胀
                vec = Vector((await self._embedding.embed([query]))[0])
                await cur.execute(_INSERT_SQL, {
                    "question": query,
                    "answer": answer,
                    "citations": Json(citations),
                    "vec": vec,
                })
        except Exception:  # noqa: BLE001 - 回写失败不拖垮回答
            logger.warning("语义缓存回写失败", exc_info=True)
```

注意:`store` 里 `_nearest` 与 INSERT 各 embed 一次(共 2 次 embedding 调用)。
如实现时想省一次,可把 `_nearest` 拆成先 embed 再查询的两步复用向量——
允许这样优化,但保持测试语义不变。

创建 `rag/agent/cache/__init__.py`:

```python
from rag.agent.cache.manager import (
    SemanticCache,
    clear_semantic_cache,
    purge_expired,
)

__all__ = ["SemanticCache", "clear_semantic_cache", "purge_expired"]
```

- [ ] **Step 4: 加协议与状态字段**

`rag/agent/type.py` 三处修改。

在 `RerankerProtocol` 定义之后(注意该文件缩进混用 tab/空格,新增代码跟随所在块的既有风格)追加:

```python
@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """agent 层所需的语义缓存接口。具体实现(如 SemanticCache)只需满足此协议即可。"""

    async def lookup(
        self, query: str, session_id: str | None = None
    ) -> dict | None: ...

    async def store(
        self, query: str, answer: str, citations: list
    ) -> None: ...
```

`MyState` 的 `# ----------- 生成 -----------` 块追加一行:

```python
	cache_hit: bool  # cache_lookup 命中时置 True,路由直达 add_memory
```

`ContextSchema` 追加字段(放在 `pool` 之后):

```python
	semantic_cache: "SemanticCacheProtocol | None" = None
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_semantic_cache.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add rag/agent/cache/ rag/agent/type.py tests/test_semantic_cache.py
git commit -m "feat(cache): SemanticCache 管理器与协议定义"
```

---

### Task 3: cache_lookup / cache_store 节点

**Files:**
- Create: `rag/agent/nodes/cache_lookup/lookup.py`(含空 `__init__.py` 同目录惯例:参照 `nodes/dynamic_topk/`,若该目录无 `__init__.py` 则同样不建)
- Create: `rag/agent/nodes/cache_store/store.py`
- Test: `tests/test_nodes.py`(追加)

**Interfaces:**
- Consumes: Task 2 的 `ContextSchema.semantic_cache`、`SemanticCache.lookup/store`、`MyState.cache_hit`;`rag.config.get_settings().SEMANTIC_CACHE_ENABLED`
- Produces: `async cache_lookup(state, runtime) -> MyState`(命中时置 `generated`/`citations`/`cache_hit=True` 并下发 SSE 事件);`async cache_store(state, runtime) -> MyState`

- [ ] **Step 1: 写失败测试**

`tests/test_nodes.py` 头部 import 区追加:

```python
import rag.agent.nodes.cache_lookup.lookup as cache_lookup_mod
import rag.agent.nodes.cache_store.store as cache_store_mod
```

文件末尾追加(fake settings 用 monkeypatch 改 `get_settings` 返回对象的属性,
参照文件内既有对 `SENTENCE_WINDOW_ENABLED` 的处理方式;若无先例则如下直接 patch):

```python
class _FakeCache:
    def __init__(self, hit=None):
        self.hit = hit
        self.lookup_calls = []
        self.store_calls = []

    async def lookup(self, query, session_id=None):
        self.lookup_calls.append((query, session_id))
        return self.hit

    async def store(self, query, answer, citations):
        self.store_calls.append((query, answer, citations))


def _cache_settings(monkeypatch, mod, enabled=True):
    from types import SimpleNamespace
    monkeypatch.setattr(
        mod, "get_settings",
        lambda: SimpleNamespace(SEMANTIC_CACHE_ENABLED=enabled),
    )


async def test_cache_lookup_disabled_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=False)
    cache = _FakeCache(hit={"answer": "A", "citations": []})
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "rewrite_query": "q2"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert cache.lookup_calls == []
    assert "cache_hit" not in out


async def test_cache_lookup_no_cache_injected_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert "cache_hit" not in out


async def test_cache_lookup_hit_sets_state_and_streams(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    events = []
    monkeypatch.setattr(
        cache_lookup_mod, "get_stream_writer", lambda: events.append
    )
    cache = _FakeCache(hit={"answer": "缓存答案", "citations": [{"index": 1}]})
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "rewrite_query": "改写q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert cache.lookup_calls == [("改写q", "s1")]
    assert out["cache_hit"] is True
    assert out["generated"] == "缓存答案"
    assert out["citations"] == [{"index": 1}]
    types = [e.get("type") for e in events]
    assert types == ["status", "message", "citations"]


async def test_cache_lookup_miss_leaves_state(monkeypatch):
    _cache_settings(monkeypatch, cache_lookup_mod, enabled=True)
    monkeypatch.setattr(
        cache_lookup_mod, "get_stream_writer", lambda: (lambda *a, **k: None)
    )
    cache = _FakeCache(hit=None)
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q"}

    out = await cache_lookup_mod.cache_lookup(state, runtime)

    assert "cache_hit" not in out
    assert "generated" not in out


async def test_cache_store_writes_generated(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=True)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {
        "session_id": "s1", "raw_query": "q", "rewrite_query": "改写q",
        "generated": "新答案", "citations": [{"index": 2}],
    }

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == [("改写q", "新答案", [{"index": 2}])]


async def test_cache_store_skips_empty_answer(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=True)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "generated": ""}

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == []


async def test_cache_store_disabled_passthrough(monkeypatch):
    _cache_settings(monkeypatch, cache_store_mod, enabled=False)
    cache = _FakeCache()
    runtime = SimpleNamespace(context=ContextSchema(
        llm=None, memory_manager=None, semantic_cache=cache))
    state = {"session_id": "s1", "raw_query": "q", "generated": "a"}

    await cache_store_mod.cache_store(state, runtime)

    assert cache.store_calls == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_nodes.py -v -k cache`
Expected: FAIL,`ModuleNotFoundError: No module named 'rag.agent.nodes.cache_lookup'`

- [ ] **Step 3: 实现两个节点**

创建 `rag/agent/nodes/cache_lookup/lookup.py`:

```python
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


async def cache_lookup(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """语义缓存查询:命中则直接下发缓存答案,路由跳过检索链与生成。

    开关关闭或未注入时透传;lookup 内部已吞异常,失败即未命中。
    """
    settings = get_settings()
    cache = runtime.context.semantic_cache
    if not settings.SEMANTIC_CACHE_ENABLED or cache is None:
        return state

    query = state.get("rewrite_query") or state["raw_query"]
    hit = await cache.lookup(query, session_id=state.get("session_id"))
    if hit is None:
        return state

    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "缓存命中"))
    writer(stream_event(StreamEventType.MESSAGE, hit["answer"]))
    # 复放原次回答的引用元数据,与 generate 节点的下发格式一致
    writer({"type": "citations", "data": hit["citations"]})

    state["generated"] = hit["answer"]
    state["citations"] = hit["citations"]
    state["cache_hit"] = True
    return state
```

创建 `rag/agent/nodes/cache_store/store.py`:

```python
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.config import get_settings


async def cache_store(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """把本轮生成的答案回写语义缓存。store 内部已吞异常,best-effort。"""
    settings = get_settings()
    cache = runtime.context.semantic_cache
    if not settings.SEMANTIC_CACHE_ENABLED or cache is None:
        return state

    answer = state.get("generated")
    if not answer:
        return state

    query = state.get("rewrite_query") or state["raw_query"]
    await cache.store(query, answer, state.get("citations") or [])
    return state
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_nodes.py -v -k cache`
Expected: 7 个新用例全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/agent/nodes/cache_lookup/ rag/agent/nodes/cache_store/ tests/test_nodes.py
git commit -m "feat(cache): cache_lookup 与 cache_store 节点"
```

---

### Task 4: workflow 接线(11 → 13 节点)

**Files:**
- Modify: `rag/agent/workflow.py`
- Test: `tests/test_workflow.py`(追加)

**Interfaces:**
- Consumes: Task 3 的 `cache_lookup`/`cache_store` 节点函数
- Produces: 图结构 `handle_query → cache_lookup ┬(命中)→ add_memory / └(未命中)→ recall`;`generate → cache_store → add_memory`;路由函数 `_route_after_cache(state) -> str`

- [ ] **Step 1: 写失败测试**

`tests/test_workflow.py` 追加(import 区加 `from rag.agent.workflow import _route_after_cache, graph`,与文件既有 import 合并,不重复声明):

```python
def test_route_after_cache_hit_goes_to_add_memory():
    assert _route_after_cache({"cache_hit": True}) == "add_memory"


def test_route_after_cache_miss_goes_to_recall():
    assert _route_after_cache({}) == "recall"


def test_graph_contains_cache_nodes():
    nodes = set(graph.get_graph().nodes)
    assert "cache_lookup" in nodes
    assert "cache_store" in nodes


def test_graph_edges_generate_via_cache_store():
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("generate", "cache_store") in edges
    assert ("cache_store", "add_memory") in edges
    # 旧的 generate → add_memory 直连必须移除
    assert ("generate", "add_memory") not in edges
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_workflow.py -v -k cache`
Expected: FAIL,`ImportError: cannot import name '_route_after_cache'`

- [ ] **Step 3: 修改 workflow.py**

import 区追加:

```python
from rag.agent.nodes.cache_lookup.lookup import cache_lookup
from rag.agent.nodes.cache_store.store import cache_store
```

`_route_after_query` 之后追加路由函数:

```python
def _route_after_cache(state: MyState) -> str:
    """条件边:缓存命中直达记忆写入(答案已流式下发),未命中走召回。"""
    if state.get("cache_hit"):
        return "add_memory"
    return "recall"
```

节点注册区,`handle_query` 之后追加:

```python
# 语义缓存查询(handle_query 之后,命中跳过检索链与生成)
builder.add_node("cache_lookup", cache_lookup)
# 语义缓存回写(generate 之后,best-effort)
builder.add_node("cache_store", cache_store)
```

边定义区改动(共 3 处):

原:

```python
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {"recall": "recall", "direct_answer": "direct_answer"},
)
```

改为(in-scope 先进缓存查询):

```python
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {"recall": "cache_lookup", "direct_answer": "direct_answer"},
)
builder.add_conditional_edges(
    "cache_lookup",
    _route_after_cache,
    {"add_memory": "add_memory", "recall": "recall"},
)
```

原 `builder.add_edge("generate", "add_memory")` 改为:

```python
builder.add_edge("generate", "cache_store")
builder.add_edge("cache_store", "add_memory")
```

同步更新文件内 ASCII 图注释为:

```python
#                                       ┌─ out-of-scope -> direct_answer ─────────────────────────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                                           END
#                                       └─ in-scope -> cache_lookup ┬─ 命中 -> add_memory ────────────────────────────┘
#                                                                   └─ 未命中 -> recall -> neighbor_expand -> rerank
#                                                                      -> dynamic_topk -> parent_expand ┬─ generate -> cache_store -> add_memory
#                                                                                                       └─ no_results -> END
```

- [ ] **Step 4: 跑测试确认通过(含既有 workflow 测试无回归)**

Run: `uv run pytest tests/test_workflow.py tests/test_nodes.py -v`
Expected: 全部 PASS(不含 18 个既有失败文件)

- [ ] **Step 5: Commit**

```bash
git add rag/agent/workflow.py tests/test_workflow.py
git commit -m "feat(cache): workflow 接入 cache_lookup/cache_store 条件路由"
```

---

### Task 5: DI 装配(lifespan + controller + service)

**Files:**
- Modify: `rag/api/main.py`(lifespan,重排序器初始化之后)
- Modify: `rag/api/dependencies/agent.py`
- Modify: `rag/api/modules/chat/controller.py`
- Modify: `rag/api/modules/chat/service.py`
- Test: `tests/test_dependencies_agent.py`(追加)

**Interfaces:**
- Consumes: Task 2 的 `SemanticCache`;`rag.governance.usage.UsageRecorder`
- Produces: `app.state.semantic_cache: SemanticCache | None`;`get_semantic_cache(request) -> SemanticCache | None`;`stream_chat(..., semantic_cache=None)` 透传进 `ContextSchema`

- [ ] **Step 1: 写失败测试**

`tests/test_dependencies_agent.py` 追加(参照文件内既有用例的 fake request 写法):

```python
def test_get_semantic_cache_returns_state_attr():
    from rag.api.dependencies.agent import get_semantic_cache

    sentinel = object()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(semantic_cache=sentinel))
    )
    assert get_semantic_cache(request) is sentinel


def test_get_semantic_cache_missing_attr_returns_none():
    from rag.api.dependencies.agent import get_semantic_cache

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert get_semantic_cache(request) is None
```

(若该文件用真实 `Request` 对象构造,跟随其既有写法改写上述两用例,断言语义不变)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_dependencies_agent.py -v -k semantic`
Expected: FAIL,`ImportError: cannot import name 'get_semantic_cache'`

- [ ] **Step 3: 实现装配**

`rag/api/dependencies/agent.py` 追加:

```python
def get_semantic_cache(request: Request):
    return getattr(request.app.state, "semantic_cache", None)
```

`rag/api/main.py` lifespan,重排序器初始化块之后追加
(`UsageRecorder` 从 `rag.governance.usage` import;独立于 GOVERNANCE_ENABLED,
缓存命中统计始终可用;pricing 传空表,cost 记 None):

```python
        logger.info("初始化语义缓存")
        if settings.SEMANTIC_CACHE_ENABLED:
            from rag.agent.cache import SemanticCache
            from rag.governance.usage import UsageRecorder

            app.state.semantic_cache = SemanticCache(
                pool,
                embedding,
                threshold=settings.SEMANTIC_CACHE_SIM_THRESHOLD,
                ttl_hours=settings.SEMANTIC_CACHE_TTL_HOURS,
                recorder=UsageRecorder(pool, "api", {}),
            )
            logger.info("语义缓存初始化完成")
        else:
            app.state.semantic_cache = None
            logger.info("语义缓存未启用,跳过")
```

`rag/api/modules/chat/service.py`:`stream_chat` 签名 `pool` 参数后加
`semantic_cache=None`,`ContextSchema(...)` 构造处加 `semantic_cache=semantic_cache`。

`rag/api/modules/chat/controller.py`:import 区加
`from rag.api.dependencies.agent import get_semantic_cache`(合并进既有 import 行),
`chat` 端点参数加 `semantic_cache=Depends(get_semantic_cache)`,
`service.stream_chat(...)` 调用处透传 `semantic_cache=semantic_cache`。

- [ ] **Step 4: 跑测试确认通过(含 lifespan/controller 既有测试无回归)**

Run: `uv run pytest tests/test_dependencies_agent.py tests/test_api_lifespan.py tests/test_chat_controller.py tests/test_chat_service_errors.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/api/main.py rag/api/dependencies/agent.py rag/api/modules/chat/controller.py rag/api/modules/chat/service.py tests/test_dependencies_agent.py
git commit -m "feat(cache): lifespan 装配语义缓存并注入 chat 链路"
```

---

### Task 6: 失效钩子(worker 入库)+ TTL 清理 cron

**Files:**
- Modify: `rag/document/pipeline.py`(`store_chunks_and_complete` 成功之后)
- Modify: `rag/worker/main.py`(新 cron)
- Test: `tests/test_document_pipeline.py`、`tests/test_worker.py`(各追加;注意这两个文件有既有失败,新用例必须独立可过)

**Interfaces:**
- Consumes: Task 2 的模块级 `clear_semantic_cache(pool)`、`purge_expired(pool, ttl_hours)`
- Produces: 入库成功即清空缓存(best-effort);每小时 cron `purge_semantic_cache(ctx)`

- [ ] **Step 1: 写失败测试**

`tests/test_document_pipeline.py` 追加:

```python
async def test_ingest_clears_semantic_cache_on_success(monkeypatch):
    """入库成功后必须调用 clear_semantic_cache(best-effort)。"""
    import rag.document.pipeline as pipeline_mod

    calls = []

    async def _fake_clear(pool):
        calls.append(pool)

    monkeypatch.setattr(pipeline_mod, "clear_semantic_cache", _fake_clear)
    # 复用本文件 happy path 用例的 ctx/monkeypatch 搭建方式跑一次 ingest,
    # 然后:
    # assert len(calls) == 1
```

说明:该文件的 happy path 用例(`test_ingest_happy_path_batches_and_completes`)
当前是既有失败——搭建方式以**当前代码实际接口**为准重新写最小 ctx,不照抄坏用例。
若搭建成本过高,退化为纯单元断言:直接调用 `pipeline_mod._ingest` 太重时,
改为校验 `clear_semantic_cache` 在 `pipeline.py` 中于 `store_chunks_and_complete`
之后被调用的集成点(mock `store.store_chunks_and_complete` 等全部外部依赖)。

`tests/test_worker.py` 追加:

```python
def test_worker_registers_semantic_cache_cron():
    from rag.worker.main import run  # noqa: F401
    import rag.worker.main as worker_main

    src = None
    import inspect
    src = inspect.getsource(worker_main)
    assert "purge_semantic_cache" in src


async def test_purge_semantic_cache_calls_purge_expired(monkeypatch):
    import rag.worker.main as worker_main

    calls = []

    async def _fake_purge(pool, ttl_hours):
        calls.append((pool, ttl_hours))

    monkeypatch.setattr(worker_main, "purge_expired", _fake_purge)
    ctx = {"pg": object()}
    await worker_main.purge_semantic_cache(ctx)

    assert len(calls) == 1
    assert calls[0][1] == 168
```

(第一个用例用源码断言而非 WorkerSettings 内省,是因为该文件既有的
`test_worker_registers_extract_function` 内省方式当前已坏;若实现时发现
cron_jobs 列表可直接断言,优先改为结构断言。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_worker.py -v -k semantic`
Expected: FAIL,`AttributeError: ... no attribute 'purge_semantic_cache'`

- [ ] **Step 3: 实现**

`rag/document/pipeline.py`:

import 区追加 `from rag.agent.cache import clear_semantic_cache`。

`_ingest` 内,`await store.store_chunks_and_complete(...)` 之后、实体抽取投递之前插入:

```python
        # 入库成功,历史缓存答案可能已过时,整表失效(best-effort)
        try:
            await clear_semantic_cache(pool)
        except Exception:  # noqa: BLE001 - 缓存失效失败不影响入库结果
            logger.warning("清空语义缓存失败", exc_info=True)
```

`rag/worker/main.py`:

import 区追加 `from rag.agent.cache import purge_expired`。

`retry_failed_documents` 之后追加:

```python
async def purge_semantic_cache(ctx: dict) -> None:
    """每小时物理删除过期语义缓存行(lookup 已按 TTL 过滤,此处仅回收空间)。"""
    settings = get_settings()
    try:
        await purge_expired(ctx["pg"], settings.SEMANTIC_CACHE_TTL_HOURS)
    except Exception:  # noqa: BLE001 - 清理失败等下一轮
        logger.warning("语义缓存过期清理失败", exc_info=True)
```

`cron_jobs` 列表追加:

```python
        cron(purge_semantic_cache, minute={0}),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_worker.py -v -k semantic && uv run pytest tests/test_document_pipeline.py -v -k semantic`
Expected: 新用例全部 PASS(该两文件的既有失败用例不在本任务范围)

- [ ] **Step 5: Commit**

```bash
git add rag/document/pipeline.py rag/worker/main.py tests/test_document_pipeline.py tests/test_worker.py
git commit -m "feat(cache): 入库失效钩子与过期清理 cron"
```

---

### Task 7: 全量回归 + 文档同步

**Files:**
- Modify: `README.md`(架构图、技术栈表、项目结构、配置表)
- Modify: `CLAUDE.md`(workflow 章节)
- Modify: `.env.example`(若存在,追加三个配置项)

**Interfaces:**
- Consumes: Tasks 1-6 全部产出

- [ ] **Step 1: 全量单测回归**

Run: `uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py`
Expected: 除分支既有的 18 个失败外,无新增失败;所有本计划新增用例 PASS。
对比基线:既有失败清单见 git 提交前的 CI/本地记录(test_logging 6、test_retriever 5、
test_document_pipeline 4、test_worker 2、parse_and_chunk 1)。

- [ ] **Step 2: 文档同步**

README.md:
- 架构图 `handle_query` 行后插入 `├─ cache_lookup    → 语义缓存查询(命中跳过检索与生成)`,`generate` 行后对应位置插入 `├─ cache_store     → 答案回写语义缓存`;"11 节点" 改 "13 节点"(两处:架构图标题行、技术栈表、项目结构 workflow.py 注释)
- 技术栈表加一行:`| **语义缓存** | pgvector 相似度命中 + 入库失效 + TTL(SEMANTIC_CACHE_*) |`
- 项目结构树 nodes/ 下加 `cache_lookup/` 与 `cache_store/` 两行
- 配置说明表(若有)加三个 `SEMANTIC_CACHE_*` 项

CLAUDE.md workflow 章节:
- "11-node" 改 "13-node",ASCII 图与节点表加 `cache_lookup`/`cache_store` 行(role 描述与 spec 一致),Conditional routing 加 `_route_after_cache` 条目
- State type 节 `MyState` 字段列表加 `cache_hit`

`.env.example`(存在则):

```bash
# 语义缓存
SEMANTIC_CACHE_ENABLED=false
SEMANTIC_CACHE_SIM_THRESHOLD=0.95
SEMANTIC_CACHE_TTL_HOURS=168
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md .env.example
git commit -m "docs(cache): 同步语义缓存节点与配置说明"
```

---

## Self-Review 记录

- **Spec 覆盖**:配置/迁移(T1)、管理器+协议(T2)、双节点(T3)、图接线(T4)、DI(T5)、失效+清理(T6)、文档(T7);spec 的"压测 S5"明确列为非本期,无对应任务 ✓
- **类型一致性**:`lookup(query, session_id=None) -> dict|None`、`store(query, answer, citations)`、`clear_semantic_cache(pool)`、`purge_expired(pool, ttl_hours)`、`_route_after_cache`、`MyState.cache_hit` 在 T2-T6 间引用一致 ✓
- **占位符**:T6 Step 1 的 pipeline 测试给了两条落地路径(复用搭建/最小 mock),因该文件既有用例已坏,无法照抄——这是有意的实现自由度,不是 TBD ✓
