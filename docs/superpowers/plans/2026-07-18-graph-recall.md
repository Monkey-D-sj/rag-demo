# GraphRAG 查询侧闭环 (Graph Recall) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 查询实体经 Neo4j 1 跳扩展成第三路召回,与向量/BM25 三路 RRF 融合,eval 第 8 轨量化增益。

**Architecture:** `handle_query` 结构化输出顺带抽查询实体 → 新建 `GraphRetriever`(Cypher 1 跳 + 回 pg 取正文) → `KnowledgeRetriever` 内部三路 `asyncio.gather` + `_merge_dedup` 第三个 RRF 循环 → 复用 rerank/dynamic_topk 质量门。默认关(`GRAPH_RECALL_ENABLED=false`)。

**Tech Stack:** Python 3.12 / neo4j async driver / psycopg3 / LangGraph / pytest

**Spec:** `docs/superpowers/specs/2026-07-18-graph-recall-design.md`

## Global Constraints

- 中文注释一律用半角标点(逗号,冒号:分号;括号());句号。「」可用。新增行逐字节自检
- 禁止重复声明已有的变量
- 日志一律 `from rag.common.logging import get_logger`
- 图路降级与 BM25 同等待遇:实体为空/未注入 → 不起协程;异常 → warning + 空列表;向量仍是唯一关键路径
- 图路返回行结构必须与 `store.search_chunks` 对齐(id/document_id/chunk_index/text/filename/knowledge_base_id/metadata),下游 rerank/generate 零改动
- `GRAPH_RECALL_ENABLED: bool = False` 默认关;生效需同时 `NEO4J_ENABLED=true`
- 工作区可能有用户在飞改动(当前:暂存的 alembic/0007 删除、未跟踪 .codegraph/)。提交前 `git restore --staged alembic/versions/0007_partition_chunks_by_kb.py`,显式 add 本任务文件,提交后 `git add alembic/versions/0007_partition_chunks_by_kb.py` 复原;`git diff --cached --stat` 必须恰为本任务文件
- Commit message 末尾:空行 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- 单测命令 `uv run pytest tests/<file> -v`;全量 `uv run pytest tests/ -q` 预期 0 失败(基线 305 passed)

---

### Task 1: 查询实体抽取(prompt + 结构化输出 + 状态)

**Files:**
- Modify: `rag/prompts/query.py`
- Modify: `rag/agent/nodes/query/query.py`
- Modify: `rag/agent/type.py`(MyState 加一行)
- Test: `tests/test_nodes.py`(追加)

**Interfaces:**
- Produces: `QueryRewriteOutput.entities: list[str]`(default_factory=list);`MyState.query_entities: list[str]`;`handle_query` 成功时写 `state["query_entities"] = result.entities`,失败降级写 `[]`

- [ ] **Step 1: 写失败测试**

`tests/test_nodes.py` 追加(`_FakeLLM.ainvoke_structured` 在文件头部已存在,返回 `QueryRewriteOutput(rewrite_query="rewritten", is_out_of_scope=False)` — 该构造处同步加 `entities=["孙悟空"]`):

```python
async def test_handle_query_writes_entities(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))
    state = {"session_id": "s1", "raw_query": "他为什么大闹天宫", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["query_entities"] == ["孙悟空"]


async def test_handle_query_failure_degrades_entities_empty(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _BoomLLM:
        async def ainvoke_structured(self, messages, schema):
            raise RuntimeError("llm down")

    runtime = SimpleNamespace(context=ContextSchema(llm=_BoomLLM(), memory_manager=None))
    state = {"session_id": "s1", "raw_query": "q", "context": ""}

    out = await query_mod.handle_query(state, runtime)

    assert out["query_entities"] == []
    assert out["rewrite_query"] == "q"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_nodes.py -v -k entities`
Expected: FAIL(`_FakeLLM` 构造 `QueryRewriteOutput` 时无 entities 字段 → 第一个用例 KeyError/ValidationError 路线均算 RED)

- [ ] **Step 3: 实现**

`rag/agent/nodes/query/query.py` 的 `QueryRewriteOutput` 追加字段:

```python
    entities: list[str] = Field(
        default_factory=list,
        description="查询中出现的专有名词实体(人名/地名/物名等);is_out_of_scope 为 true 或查询无实体时返回空数组",
    )
```

`handle_query` 的 try 块内 `state["is_out_of_scope"] = result.is_out_of_scope` 之后加:

```python
        state["query_entities"] = result.entities
```

except 块内 `state["is_out_of_scope"] = False` 之后加:

```python
        state["query_entities"] = []
```

`rag/agent/type.py` 的 `MyState` `# ----------- 检索 -----------` 块追加一行(注意该块用 tab 缩进):

```python
	query_entities: list[str]  # handle_query 抽取的查询实体,图召回入口
```

`rag/prompts/query.py`:
- "同时负责两项任务" 改为 "同时负责三项任务:**范围判断**、**查询改写**和**实体抽取**"
- 任务二之后插入:

```
## 任务三:实体抽取(entities)

从用户查询(含指代消解后的实体)中抽取专有名词实体:人名、地名、物名、组织名等。
- 只抽查询中明确出现或经指代消解得出的实体,不推测、不扩展
- 不抽泛义词(如"武器"、"师父"这类普通名词)
- is_out_of_scope 为 true 或查询无实体时返回空数组
```

- 输出格式段改为 "包含 rewrite_query(字符串)、is_out_of_scope(布尔值)和 entities(字符串数组)三个字段"
- 示例全部加 entities 键:

```
上下文：用户刚才在问刘备的结拜兄弟有哪些。
用户查询：他三弟是谁
→ {"rewrite_query": "刘备的三弟是谁", "is_out_of_scope": false, "entities": ["刘备"]}

上下文：空或无关。
用户查询：孙悟空为什么被压在五指山下
→ {"rewrite_query": "孙悟空为什么被压在五指山下", "is_out_of_scope": false, "entities": ["孙悟空", "五指山"]}

上下文：空。
用户查询：你好啊
→ {"rewrite_query": "你好啊", "is_out_of_scope": true, "entities": []}

上下文：空。
用户查询：帮我用 Python 写一个快速排序
→ {"rewrite_query": "帮我用 Python 写一个快速排序", "is_out_of_scope": true, "entities": []}

上下文：空。
用户查询：今天天气真不错
→ {"rewrite_query": "今天天气真不错", "is_out_of_scope": true, "entities": []}
```

同时把 `tests/test_nodes.py` 头部 `_FakeLLM.ainvoke_structured` 的返回改为
`QueryRewriteOutput(rewrite_query="rewritten", is_out_of_scope=False, entities=["孙悟空"])`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_nodes.py tests/test_workflow.py -v`
Expected: 全部 PASS(含既有用例;`entities` 有默认值,旧构造不破坏)

- [ ] **Step 5: Commit**

```bash
git add rag/prompts/query.py rag/agent/nodes/query/query.py rag/agent/type.py tests/test_nodes.py
git commit -m "feat(graph): handle_query 结构化输出顺带抽取查询实体"
```

---

### Task 2: store.get_chunks_by_uids

**Files:**
- Modify: `rag/document/store.py`(在 `get_neighbor_chunks` 之后追加)
- Test: `tests/test_document_store.py`(追加)

**Interfaces:**
- Produces: `async get_chunks_by_uids(pool, uids: list[tuple[str, int]]) -> list[dict]` — 行结构与 `search_chunks` 对齐(无 similarity 列);空入参返回 `[]` 不发 SQL;SQL 返回顺序不保证(调用方自行排序)

- [ ] **Step 1: 写失败测试**

`tests/test_document_store.py` 追加(参照文件内既有 fake cursor 写法;若既有风格不同,保持断言语义改写):

```python
async def test_get_chunks_by_uids_empty_short_circuit(monkeypatch):
    called = []
    monkeypatch.setattr(
        "rag.document.store.get_cursor",
        lambda p: (_ for _ in ()).throw(AssertionError("不应建立游标")),
    )
    from rag.document.store import get_chunks_by_uids

    assert await get_chunks_by_uids(object(), []) == []
    assert called == []


async def test_get_chunks_by_uids_builds_unnest_join(monkeypatch):
    class _Cur:
        def __init__(self):
            self.sql = ""
            self.params = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def execute(self, sql, params=None):
            self.sql = sql
            self.params = params or {}

        async def fetchall(self):
            return [{"id": "c1", "document_id": "d1", "chunk_index": 3, "text": "t"}]

    cur = _Cur()
    monkeypatch.setattr("rag.document.store.get_cursor", lambda p: cur)
    from rag.document.store import get_chunks_by_uids

    rows = await get_chunks_by_uids(object(), [("d1", 3), ("d2", 0)])

    assert rows[0]["id"] == "c1"
    assert "unnest" in cur.sql
    assert cur.params["docs"] == ["d1", "d2"]
    assert cur.params["idxs"] == [3, 0]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_store.py -v -k get_chunks_by_uids`
Expected: FAIL,`ImportError: cannot import name 'get_chunks_by_uids'`

- [ ] **Step 3: 实现**

`rag/document/store.py` 在 `get_neighbor_chunks` 之后追加:

```python
async def get_chunks_by_uids(
    pool: AsyncConnectionPool,
    uids: list[tuple[str, int]],
) -> list[dict]:
    """按 (document_id, chunk_index) 批量取 chunk,行结构与 search_chunks 对齐(无 similarity)。

    图召回回 pg 取正文用;返回顺序不保证,调用方按自身评分排序。
    """
    if not uids:
        return []
    doc_ids = [d for d, _ in uids]
    idxs = [i for _, i in uids]
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.text,
                   d.filename,
                   d.knowledge_base_id,
                   dc.metadata
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            JOIN unnest(%(docs)s::uuid[], %(idxs)s::int[]) AS u(doc_id, idx)
              ON dc.document_id = u.doc_id AND dc.chunk_index = u.idx
            """,
            {"docs": doc_ids, "idxs": idxs},
        )
        return await cur.fetchall()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/document/store.py tests/test_document_store.py
git commit -m "feat(graph): store 按 uid 批量取 chunk,供图召回回表"
```

---

### Task 3: GraphRetriever

**Files:**
- Create: `rag/graph/retriever.py`
- Test: `tests/test_graph_retriever.py`(新建)

**Interfaces:**
- Consumes: Task 2 的 `store.get_chunks_by_uids`;Neo4j 图模型 `Entity{name, chunk_ids}` + `RELATES` 边;chunk_uid 格式 `"{document_id}:{chunk_index}"`
- Produces: `GraphRetriever(driver, database: str, pool)`,`async search(entities: list[str], limit: int) -> list[dict]`(按图相关性降序;异常向上抛,由调用方降级)

- [ ] **Step 1: 写失败测试**

创建 `tests/test_graph_retriever.py`:

```python
from unittest.mock import MagicMock

import pytest

from rag.graph.retriever import GraphRetriever, _score_chunks


class _FakeResult:
    def __init__(self, records):
        self._records = records

    async def data(self):
        return self._records


class _FakeSession:
    def __init__(self, records):
        self._records = records
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def run(self, query, **params):
        self.queries.append((query, params))
        return _FakeResult(self._records)


class _FakeDriver:
    def __init__(self, records):
        self.records = records
        self.sessions = []

    def session(self, database=None):
        s = _FakeSession(self.records)
        self.sessions.append(s)
        return s


def test_score_chunks_seed_outranks_neighbor():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:1", "d1:2"],
         "neighbor_chunk_lists": [["d1:3"], ["d2:0"]]},
    ]
    ranked = _score_chunks(records)
    # 种子实体 chunk(权重2)排在仅邻居命中的 chunk(权重1)之前
    assert ranked.index(("d1", 1)) < ranked.index(("d1", 3))
    assert ranked.index(("d1", 2)) < ranked.index(("d2", 0))


def test_score_chunks_multi_entity_hit_outranks_single():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:1"], "neighbor_chunk_lists": []},
        {"seed": "唐僧", "seed_chunks": ["d1:1", "d1:9"], "neighbor_chunk_lists": []},
    ]
    ranked = _score_chunks(records)
    # d1:1 被两个种子实体命中,应排最前
    assert ranked[0] == ("d1", 1)


def test_score_chunks_skips_malformed_uid():
    records = [
        {"seed": "孙悟空", "seed_chunks": ["badformat", "d1:2"],
         "neighbor_chunk_lists": [None]},
    ]
    ranked = _score_chunks(records)
    assert ranked == [("d1", 2)]


async def test_search_fetches_rows_in_score_order(monkeypatch):
    records = [
        {"seed": "孙悟空", "seed_chunks": ["d1:2"], "neighbor_chunk_lists": [["d2:5"]]},
    ]
    driver = _FakeDriver(records)

    async def _fake_get(pool, uids):
        # 模拟 SQL 乱序返回
        return [
            {"id": "b", "document_id": "d2", "chunk_index": 5, "text": "nb"},
            {"id": "a", "document_id": "d1", "chunk_index": 2, "text": "seed"},
        ]

    monkeypatch.setattr("rag.graph.retriever.store.get_chunks_by_uids", _fake_get)
    gr = GraphRetriever(driver, "neo4j", MagicMock())

    rows = await gr.search(["孙悟空"], limit=10)

    assert [r["id"] for r in rows] == ["a", "b"]  # 按图评分序,非 SQL 返回序


async def test_search_respects_limit(monkeypatch):
    records = [
        {"seed": "孙悟空", "seed_chunks": [f"d1:{i}" for i in range(20)],
         "neighbor_chunk_lists": []},
    ]
    driver = _FakeDriver(records)
    captured = {}

    async def _fake_get(pool, uids):
        captured["uids"] = uids
        return []

    monkeypatch.setattr("rag.graph.retriever.store.get_chunks_by_uids", _fake_get)
    gr = GraphRetriever(driver, "neo4j", MagicMock())

    await gr.search(["孙悟空"], limit=5)

    assert len(captured["uids"]) == 5


async def test_search_empty_entities_returns_empty():
    gr = GraphRetriever(_FakeDriver([]), "neo4j", MagicMock())
    assert await gr.search([], limit=5) == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_graph_retriever.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'rag.graph.retriever'`

- [ ] **Step 3: 实现**

创建 `rag/graph/retriever.py`:

```python
from __future__ import annotations

from collections import defaultdict

from rag.common.logging import get_logger
from rag.document import store

logger = get_logger()

# 精确匹配种子实体,沿 RELATES 扩 1 跳收集邻居实体的 chunk_ids
_EXPAND_QUERY = """
UNWIND $names AS name
MATCH (e:Entity {name: name})
OPTIONAL MATCH (e)-[:RELATES]-(nb:Entity)
RETURN e.name AS seed,
       e.chunk_ids AS seed_chunks,
       collect(nb.chunk_ids) AS neighbor_chunk_lists
"""

_SEED_WEIGHT = 2
_NEIGHBOR_WEIGHT = 1


def _parse_uid(uid: str) -> tuple[str, int] | None:
    """chunk_uid 格式 "{document_id}:{chunk_index}";异常格式返回 None 跳过。"""
    doc_id, sep, idx = uid.rpartition(":")
    if not sep or not doc_id or not idx.isdigit():
        return None
    return doc_id, int(idx)


def _score_chunks(records: list[dict]) -> list[tuple[str, int]]:
    """图命中 chunk 加权评分:种子实体 chunk 权重 2,邻居权重 1,多实体命中累加。

    返回按分数降序(同分保持首次出现序)的 (document_id, chunk_index) 列表。
    """
    scores: dict[tuple[str, int], int] = defaultdict(int)
    order: dict[tuple[str, int], int] = {}

    def _add(uid: str, weight: int) -> None:
        parsed = _parse_uid(uid)
        if parsed is None:
            return
        scores[parsed] += weight
        order.setdefault(parsed, len(order))

    for rec in records:
        for uid in rec.get("seed_chunks") or []:
            _add(uid, _SEED_WEIGHT)
        for chunk_list in rec.get("neighbor_chunk_lists") or []:
            for uid in chunk_list or []:
                _add(uid, _NEIGHBOR_WEIGHT)

    return sorted(scores, key=lambda k: (-scores[k], order[k]))


class GraphRetriever:
    """图召回:查询实体 1 跳扩展 → chunk_uid 评分排序 → 回 pg 取正文。

    异常向上抛,由 KnowledgeRetriever 的图路协程统一降级。
    """

    def __init__(self, driver, database: str, pool) -> None:
        self._driver = driver
        self._database = database
        self._pool = pool

    async def search(self, entities: list[str], limit: int) -> list[dict]:
        if not entities:
            return []

        async with self._driver.session(database=self._database) as session:
            result = await session.run(_EXPAND_QUERY, names=entities)
            records = await result.data()

        ranked = _score_chunks(records)[:limit]
        if not ranked:
            return []

        rows = await store.get_chunks_by_uids(self._pool, ranked)
        # SQL 返回顺序不保证,按图评分序重排
        pos = {uid: i for i, uid in enumerate(ranked)}
        rows.sort(key=lambda r: pos.get((str(r["document_id"]), r["chunk_index"]), len(pos)))
        return rows
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_graph_retriever.py -v`
Expected: 7 个用例全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/graph/retriever.py tests/test_graph_retriever.py
git commit -m "feat(graph): GraphRetriever 实体 1 跳扩展与 chunk 加权排序"
```

---

### Task 4: 三路融合(KnowledgeRetriever + recall 节点 + 协议 + 配置)

**Files:**
- Modify: `rag/document/retriever.py`
- Modify: `rag/agent/nodes/recall/recall.py`
- Modify: `rag/agent/type.py`(RetrieverProtocol.search 签名)
- Modify: `rag/config.py`(Graph 块加 `GRAPH_RECALL_ENABLED`)
- Test: `tests/test_retriever.py`、`tests/test_nodes.py`(各追加)

**Interfaces:**
- Consumes: Task 3 的 `GraphRetriever.search(entities, limit)`;Task 1 的 `MyState.query_entities`
- Produces: `KnowledgeRetriever(pool, embedding, settings, graph_retriever=None)`;`search(..., entities: list[str] | None = None)`;`_merge_dedup(vec_rows, bm25_rows, top_k, rrf_k=60, graph_rows=None)`;`KnowledgeRetriever.has_graph: bool` property(eval 用);`Settings.GRAPH_RECALL_ENABLED: bool = False`

- [ ] **Step 1: 写失败测试**

`tests/test_retriever.py` 追加(复用文件内既有的 fake store/embedding monkeypatch 风格;下面按语义给出,实施时对齐现有夹具写法):

```python
def test_merge_dedup_three_way_rrf():
    from rag.document.retriever import _merge_dedup

    vec = [{"id": "a", "chunk_index": 0, "text": "A", "similarity": 0.9}]
    bm25 = [{"id": "b", "chunk_index": 1, "text": "B", "score": 5.0}]
    graph = [
        {"id": "a", "chunk_index": 0, "text": "A"},
        {"id": "c", "chunk_index": 2, "text": "C"},
    ]
    out = _merge_dedup(vec, bm25, top_k=10, rrf_k=60, graph_rows=graph)

    by_id = {r["id"]: r for r in out}
    assert by_id["a"]["sources"] == ["vec", "graph"]
    assert by_id["c"]["sources"] == ["graph"]
    # a 双路命中,RRF 分 = 1/61 + 1/61,必高于单路的 b/c
    assert out[0]["id"] == "a"


def test_merge_dedup_two_way_backward_compat():
    from rag.document.retriever import _merge_dedup

    vec = [{"id": "a", "chunk_index": 0, "text": "A", "similarity": 0.9}]
    out = _merge_dedup(vec, [], top_k=5)
    assert [r["id"] for r in out] == ["a"]


def test_merge_dedup_graph_only():
    from rag.document.retriever import _merge_dedup

    graph = [{"id": "g", "chunk_index": 0, "text": "G"}]
    out = _merge_dedup([], [], top_k=5, graph_rows=graph)
    assert [r["id"] for r in out] == ["g"]
    assert out[0]["sources"] == ["graph"]


async def test_search_runs_graph_leg_when_entities(monkeypatch):
    # 沿用本文件既有 search 测试的 store/embedding mock 搭建;
    # 额外注入 fake graph_retriever,断言:
    # 1) entities 传入时 graph_retriever.search 被调用(参数 entities + candidates)
    # 2) 图路结果并入 sources 含 "graph"
    ...


async def test_search_graph_leg_failure_degrades(monkeypatch):
    # fake graph_retriever.search 抛 RuntimeError;
    # 断言 search 不抛,返回值等同两路结果
    ...


async def test_search_no_entities_skips_graph(monkeypatch):
    # graph_retriever 注入但 entities=None;
    # 断言 graph_retriever.search 未被调用
    ...
```

(后三个用例的搭建照抄本文件 `test_search_fuses_and_logs_both_legs` 的 monkeypatch 方式,
把 fake graph_retriever 做成带 `calls` 列表的类;断言语义如注释所述,必须完整实现,
不得留 `...`。)

`tests/test_nodes.py` 追加:

```python
async def test_recall_passes_entities_to_retriever(monkeypatch):
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _Ret:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5, entities=None):
            self.calls.append((query, entities))
            return []

    ret = _Ret()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=ret))
    state = {"session_id": "s1", "raw_query": "q", "rewrite_query": "rq",
             "query_entities": ["孙悟空"]}

    await kb_recall_mod.recall(state, runtime)

    assert ret.calls == [("rq", ["孙悟空"])]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retriever.py tests/test_nodes.py -v -k "three_way or graph or entities_to_retriever"`
Expected: FAIL(`_merge_dedup` 无 graph_rows 参数 → TypeError;recall 未传 entities → 断言失败)

- [ ] **Step 3: 实现**

`rag/config.py` Graph 块(`GRAPH_EXTRACT_CONCURRENCY` 之后)追加:

```python
    GRAPH_RECALL_ENABLED: bool = False  # 图召回第三路,需配合 NEO4J_ENABLED=true
```

`rag/document/retriever.py`:

1. `_merge_dedup` 签名改为
   `def _merge_dedup(vec_rows, bm25_rows, top_k, rrf_k=60, graph_rows=None):`,
   空判改 `if not vec_rows and not bm25_rows and not graph_rows:`,
   bm25 循环之后追加:

```python
    # graph 路：rank 1 = 图评分最高
    for rank, row in enumerate(graph_rows or [], 1):
        rid = str(row["id"])
        rrf = 1.0 / (rrf_k + rank)
        if rid in chunk_map:
            chunk_map[rid]["sources"].append("graph")
            chunk_map[rid]["rrf_score"] = chunk_map[rid]["rrf_score"] + rrf
        else:
            r = dict(row)
            r["sources"] = ["graph"]
            r["rrf_score"] = rrf
            r["similarity"] = None
            r["score"] = None
            chunk_map[rid] = r
```

2. `KnowledgeRetriever.__init__` 签名加 `graph_retriever=None`,
   体内加 `self._graph_retriever = graph_retriever`。加 property:

```python
    @property
    def has_graph(self) -> bool:
        """图召回是否可用(eval 分轨判断用)。"""
        return self._graph_retriever is not None
```

3. `search` 签名加 `entities: list[str] | None = None`;`_bm25_leg` 之后加:

```python
        async def _graph_leg() -> list[dict]:
            if self._graph_retriever is None or not entities:
                return []
            with span_scope("graph_recall", input={**span_input, "entities": entities}) as span:
                t0 = time.perf_counter()
                try:
                    rows = await self._graph_retriever.search(entities, candidates)
                except Exception:  # noqa: BLE001 - 图路失败降级,与 BM25 路对等容错
                    logger.warning("图召回失败,降级两路", exc_info=True)
                    rows = []
                timings["graph_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                if span is not None:
                    span.update(output=[
                        {"chunk_id": str(r["id"]), "chunk_index": r["chunk_index"], "text": r["text"]}
                        for r in rows
                    ])
                return rows
```

   gather 行改为:

```python
        vec_rows, bm25_rows, graph_rows = await asyncio.gather(
            _vec_leg(), _bm25_leg(), _graph_leg()
        )
        results = _merge_dedup(vec_rows, bm25_rows, top_k, self._rrf_k, graph_rows=graph_rows)
```

`rag/agent/nodes/recall/recall.py` 检索行改为:

```python
    state["recall_vec_results"] = await retriever.search(
        query, entities=state.get("query_entities") or None
    )
```

`rag/agent/type.py` 的 `RetrieverProtocol.search` 签名改为:

```python
	async def search(
		self, query: str, knowledge_base_ids: list[str] | None = None, top_k: int = 5,
		entities: list[str] | None = None,
	) -> list[dict]: ...
```

(保持该块的既有缩进风格。)

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retriever.py tests/test_nodes.py tests/test_workflow.py -v`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add rag/config.py rag/document/retriever.py rag/agent/nodes/recall/recall.py rag/agent/type.py tests/test_retriever.py tests/test_nodes.py
git commit -m "feat(graph): 三路 RRF 融合,recall 透传查询实体"
```

---

### Task 5: lifespan 接线(Neo4j 前移 + GraphRetriever 注入)

**Files:**
- Modify: `rag/api/main.py`
- Test: `tests/test_api_lifespan.py` 或 `tests/test_dependencies_agent.py`(视既有测试风格追加)

**Interfaces:**
- Consumes: Task 3 `GraphRetriever`、Task 4 `KnowledgeRetriever(graph_retriever=...)` 与 `Settings.GRAPH_RECALL_ENABLED`
- Produces: lifespan 中 neo4j 初始化块移到 retriever 之前;`NEO4J_ENABLED and GRAPH_RECALL_ENABLED` 时构造 `GraphRetriever(app.state.neo4j, settings.NEO4J_DATABASE, pool)` 并传入 `KnowledgeRetriever`

- [ ] **Step 1: 理解现状**

`rag/api/main.py` lifespan 当前顺序:embedding → memory_manager → llm → retriever → reranker → semantic_cache → minio → **neo4j** → arq。
Neo4j 在 retriever 之后,必须把 neo4j 初始化块(含 `stack.push_async_callback` 关闭回调)整体移到 retriever 构造之前(建议放 llm 之后)。AsyncExitStack 是 LIFO,neo4j 前移意味着关闭顺序后移,无副作用。

- [ ] **Step 2: 写失败测试**

在 `tests/test_api_lifespan.py`(若该文件用例均为 integration 标记则改放 `tests/test_dependencies_agent.py`,断言语义不变)追加:

```python
def test_graph_retriever_wiring_flags():
    """开关组合决定 KnowledgeRetriever 是否携带图召回。纯构造测试,不起 app。"""
    from unittest.mock import MagicMock

    from rag.document.retriever import KnowledgeRetriever
    from rag.graph.retriever import GraphRetriever
    from rag.config import Settings

    s = Settings()
    r_plain = KnowledgeRetriever(MagicMock(), MagicMock(), s)
    assert r_plain.has_graph is False

    gr = GraphRetriever(MagicMock(), "neo4j", MagicMock())
    r_graph = KnowledgeRetriever(MagicMock(), MagicMock(), s, graph_retriever=gr)
    assert r_graph.has_graph is True
```

并追加对 main.py 接线的源码级断言(lifespan 难以单测起真依赖,退而校验接线存在):

```python
def test_lifespan_wires_graph_retriever_conditionally():
    import inspect

    import rag.api.main as api_main

    src = inspect.getsource(api_main)
    assert "GRAPH_RECALL_ENABLED" in src
    assert "GraphRetriever(" in src
    # neo4j 初始化必须在 retriever 构造之前
    assert src.index("create_neo4j_driver") < src.index("KnowledgeRetriever(")
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_api_lifespan.py tests/test_dependencies_agent.py -v -k graph`
Expected: 第一个用例 PASS(Task 4 已实现 has_graph),第二个 FAIL(main.py 尚无接线)

- [ ] **Step 4: 实现**

`rag/api/main.py`:
1. 把 neo4j 初始化 if/else 块(含关闭回调注册)整体移到 `app.state.llm = ...` 之后、retriever 构造之前;保持日志语句原样。
2. retriever 构造处改为:

```python
        logger.info("初始化知识检索器")
        graph_retriever = None
        if app.state.neo4j is not None and settings.GRAPH_RECALL_ENABLED:
            from rag.graph.retriever import GraphRetriever

            graph_retriever = GraphRetriever(
                app.state.neo4j, settings.NEO4J_DATABASE, pool
            )
            logger.info("图召回已启用(第三路)")
        app.state.retriever = KnowledgeRetriever(
            pool, embedding, settings, graph_retriever=graph_retriever
        )
        logger.info("知识检索器初始化完成")
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_api_lifespan.py tests/test_dependencies_agent.py tests/test_main.py -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add rag/api/main.py tests/<实际追加的测试文件>
git commit -m "feat(graph): lifespan 前移 neo4j 并按开关注入图召回"
```

---

### Task 6: eval 第 8 轨 graph_fused

**Files:**
- Modify: `rag/eval/harness.py`(GoldenItem + load_golden + build_retriever + run_eval)
- Modify: `rag/eval/run.py`(_LEG_LABEL/_LEG_ORDER)
- Modify: `rag/eval/datasets/retrieval_golden.jsonl`(部分条目手标 entities)
- Test: `tests/test_eval_harness.py`(追加)

**Interfaces:**
- Consumes: Task 4 的 `retriever.search(..., entities=...)` 与 `has_graph`;Task 5 同款 GraphRetriever 构造方式
- Produces: `GoldenItem.entities: list[str]`(默认 [])、`build_retriever` 按开关注入图召回、`result["graph_fused"]`(仅当 retriever.has_graph 且存在带 entities 的条目)

- [ ] **Step 1: 写失败测试**

`tests/test_eval_harness.py` 追加(对齐文件内既有夹具风格):

```python
def test_golden_item_entities_optional(tmp_path):
    from rag.eval.harness import load_golden

    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id": "1", "query": "q1", "gold_snippets": ["s"], "entities": ["孙悟空"]}\n'
        '{"id": "2", "query": "q2", "gold_snippets": ["s"]}\n',
        encoding="utf-8",
    )
    items = load_golden(p)
    assert items[0].entities == ["孙悟空"]
    assert items[1].entities == []


async def test_run_eval_graph_leg_gated_by_has_graph(monkeypatch):
    # 用文件内既有的 fake retriever/embedding 搭建跑 run_eval:
    # 1) retriever.has_graph=False 时 result 不含 "graph_fused"
    # 2) has_graph=True 且条目带 entities 时 result 含 "graph_fused",
    #    且该轨调用 search 时传了 entities
    # (完整实现,照抄既有 run_eval 测试的最小夹具)
    ...
```

(第二个用例同样必须完整实现;`...` 仅为本计划的排版占位,提交的测试不得含 `...`。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_harness.py -v -k "entities or graph_leg"`
Expected: FAIL,`GoldenItem` 无 entities 属性

- [ ] **Step 3: 实现**

`rag/eval/harness.py`:

1. `GoldenItem` 加字段 `entities: list[str] = field(default_factory=list)`;
   `load_golden` 构造处加 `entities=list(obj.get("entities", []))`,docstring 的可选字段说明同步。
2. `build_retriever` 改为:

```python
async def build_retriever(settings: Settings) -> tuple[AsyncConnectionPool, KnowledgeRetriever]:
    """构造 pool + 复用生产 KnowledgeRetriever;调用方负责 pool.close()。

    NEO4J_ENABLED 且 GRAPH_RECALL_ENABLED 时携带图召回(driver 随进程退出释放)。
    """
    pool = await create_pg_pool(settings)
    embedding = EmbeddingModel(settings)
    graph_retriever = None
    if settings.NEO4J_ENABLED and settings.GRAPH_RECALL_ENABLED:
        from rag.db.neo4j import create_neo4j_driver
        from rag.graph.retriever import GraphRetriever

        driver = create_neo4j_driver(settings)
        graph_retriever = GraphRetriever(driver, settings.NEO4J_DATABASE, pool)
    return pool, KnowledgeRetriever(pool, embedding, settings, graph_retriever=graph_retriever)
```

3. `run_eval` 的 fused 轨循环内(与 fused 同层)追加 graph 轨:

```python
    graph_pq: list[dict] = []
```

   条目循环里 fused 计算之后:

```python
        # ── graph_fused：三路(向量+BM25+图)融合,仅当图召回可用且条目带实体标注 ──
        if retriever.has_graph and item.entities:
            graph_rows = await retriever.search(rq, None, top_k, entities=item.entities)
            graph_pq.append({
                "id": item.id, "query": rq,
                **evaluate_query([r["text"] for r in graph_rows], item.gold_snippets, ks),
                "chunks": _chunk_preview(graph_rows),
            })
```

   (`rq` 为该循环中既有的改写后查询变量名,实施时以实际变量名为准;
   result 组装处加:)

```python
    if graph_pq:
        result["graph_fused"] = {"aggregate": aggregate(graph_pq), "per_query": graph_pq}
```

`rag/eval/run.py`:
- `_LEG_LABEL` 加 `"graph_fused": "图谱融合"`(对齐既有命名风格,如现有值为「混合改写」式四字标签则用「图谱融合」)
- `_LEG_ORDER` 在 `"fused_reranked"` 之后插入 `"graph_fused"`

`rag/eval/datasets/retrieval_golden.jsonl`:
- 为查询文本中含明确专有名词实体的条目加 `"entities": [...]`
  (实体必须逐字出现在 query 或 rewrite_query 中;拿不准的条目不加,跳过即可;
  预计可标 1/3 以上条目)。只加键,不改任何既有键值。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_harness.py tests/test_eval_metrics.py tests/test_eval_retrieval.py -v`
Expected: 全部 PASS(eval 标记用例照常 deselect)

- [ ] **Step 5: Commit**

```bash
git add rag/eval/harness.py rag/eval/run.py rag/eval/datasets/retrieval_golden.jsonl tests/test_eval_harness.py
git commit -m "feat(graph): eval 第 8 轨 graph_fused 与 golden 实体标注"
```

**运维备注(不在本任务实现):** 本地跑 graph 轨前,需 `NEO4J_ENABLED=true GRAPH_RECALL_ENABLED=true ENABLE_ENTITY_EXTRACTION=true` 且 eval 语料已跑过实体抽取入图(可对 eval KB 文档逐个调 `rag.graph.pipeline.extract_document_entities`);CI 无 Neo4j service,graph 轨自动缺席,7 轨门禁不受影响。

---

### Task 7: 全量回归 + 文档同步

**Files:**
- Modify: `README.md`、`CLAUDE.md`、`.env.example`

**Interfaces:**
- Consumes: Tasks 1-6 全部产出

- [ ] **Step 1: 全量回归**

Run: `uv run pytest tests/ -q`
Expected: 0 失败(基线 305 passed + 本 feature 新增用例)。任何失败先归因,禁止为绿改测试/源码。

- [ ] **Step 2: 文档同步**

README.md:
- 技术栈表 `| **图数据库** | Neo4j（实体关系抽取，可选） |` 改为
  `| **图数据库** | Neo4j GraphRAG（实体抽取入图 + 查询实体 1 跳扩展第三路召回,可选） |`
- 架构概览图 `recall` 行描述加 "/图谱" 或同风格三路说明
- 检索评测节 "7 路分轨" 改为 "8 路分轨",说明 `graph_fused` 为条件轨(需 Neo4j + 实体标注)
- 配置参考表加 `GRAPH_RECALL_ENABLED` 行

CLAUDE.md:
- Hybrid retrieval 章节:两路 → 三路(图路条件参与),`_merge_dedup` 描述同步,
  graph 路降级说明(与 BM25 同等待遇)
- Agent workflow 节点表 `recall` 行、`handle_query` 行(加实体抽取)同步
- eval 章节 7 legs → 8 legs(`graph_fused` 条件轨)
- State type 节加 `query_entities`

`.env.example` Graph 块加:

```bash
GRAPH_RECALL_ENABLED=false
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md .env.example
git commit -m "docs(graph): 同步三路召回与 8 轨评测说明"
```

---

## Self-Review 记录

- **Spec 覆盖**:实体抽取(T1)、回表(T2)、GraphRetriever(T3)、三路融合+协议+配置(T4)、lifespan(T5)、eval 8 轨+golden 标注(T6)、文档(T7);spec 非目标(多跳/LightRAG/模糊匹配)均无对应任务 ✓
- **类型一致性**:`GraphRetriever.search(entities, limit)`、`KnowledgeRetriever(pool, embedding, settings, graph_retriever=None)`、`search(..., entities=None)`、`has_graph`、`_merge_dedup(..., graph_rows=None)`、`MyState.query_entities`、`GoldenItem.entities` 全链一致 ✓
- **占位符**:T4/T6 各有一处测试以注释描述语义并显式要求"完整实现,不得含 `...`"——因需对齐既有夹具风格,属实现自由度,验收标准(断言语义)已给全 ✓
