# 查询分解 (Query Decomposition) 并行检索实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 扩展 `handle_query` 把多面复杂问题拆为子查询, 用 LangGraph Send API 扇出并行检索, `recall_fuse` 二级 RRF 融合后进入既有 rerank/dynamic_topk 质量门, 并加评测第 9 轨 `decomposed` 量化增益。

**Architecture:** `handle_query` 结构化输出加 `sub_queries` (零新增 LLM 调用); `_route_after_cache` 未命中时返回 `list[Send]`, 分支 = `[rewrite_query] + sub_queries` (最多 4 个, 仅主分支带 entities 走图召回); 各分支结果经 `MyState` 首个 Annotated reducer 字段 `sub_recall_results` fan-in 到新节点 `recall_fuse`, 用纯函数 `fuse_multi_query_results()` 融合写入 `recall_vec_results`, 下游零改动。eval 第 9 轨与 agent 共用该融合函数。

**Tech Stack:** LangGraph Send API (`langgraph.types.Send`, 已验证当前版本可导入且支持值相等), Pydantic 结构化输出, pytest。

**Spec:** `docs/superpowers/specs/2026-07-18-query-decomposition-design.md`

## Global Constraints

- 中文代码注释/docstring 一律用半角标点 (`,` `;` `:` `()`, 禁用 `，` `；` `：` `（）`)。每个含中文注释的提交前跑字节级自检:
  `python -c "s=open(FILE,encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"`
  只查本次新增的文件/行, 历史遗留全角不管。
- **git 隔离规程 (硬性)**: 用户暂存区常驻 WIP (当前有 `alembic/versions/0007_partition_chunks_by_kb.py` 的已暂存删除)。每次提交必须: `git add <白名单文件>` → `git commit <同一白名单文件> -m "..."` (pathspec 提交只带白名单) → `git status --short` 验证 0007 删除仍在暂存区。禁止裸 `git commit`、`git add -A`、`git add tests/`。
- 禁止重复声明已有的变量 (CLAUDE.md 规范)。
- `QUERY_DECOMPOSITION_ENABLED` 默认 `false`; 关闭时行为与现状完全一致。
- 子查询上限硬编码 3 (prompt 约束 + `handle_query` 代码 clamp), 不做配置。
- `rag/agent/type.py` 的 `MyState` 用 **tab 缩进**, 编辑时保持一致。
- 测试命令一律 `uv run pytest ...` (Windows, conftest 已设 SelectorEventLoop)。
- 提交信息末尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

### Task 1: 配置开关 + fuse_multi_query_results 纯函数

**Files:**
- Modify: `rag/config.py` (语义缓存段之后加开关)
- Modify: `rag/document/retriever.py` (`_merge_dedup` 之后加纯函数)
- Test: `tests/test_retriever.py` (文件末尾追加)

**Interfaces:**
- Consumes: 无 (独立任务)
- Produces: `fuse_multi_query_results(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]` (Task 3 的 `recall_fuse` 与 Task 4 的 eval 轨消费); `Settings.QUERY_DECOMPOSITION_ENABLED: bool` (Task 3 路由消费)

- [ ] **Step 1: 写失败测试**

在 `tests/test_retriever.py` 末尾追加:

```python
def test_fuse_multi_query_empty_and_single_passthrough():
    from rag.document.retriever import fuse_multi_query_results

    assert fuse_multi_query_results([]) == []
    assert fuse_multi_query_results([[], []]) == []

    single = [{"id": 1, "text": "a", "sources": ["vec"], "rrf_score": 0.9}]
    out = fuse_multi_query_results([single, []])
    # 单份非空列表原样返回(浅拷贝),保证单分支路径与现状行为一致
    assert out == single
    assert out is not single


def test_fuse_multi_query_rrf_sum_and_sources_union():
    from rag.document.retriever import fuse_multi_query_results

    a = [
        {"id": 1, "text": "c1", "sources": ["vec"]},
        {"id": 2, "text": "c2", "sources": ["bm25"]},
    ]
    b = [
        {"id": 2, "text": "c2", "sources": ["vec", "graph"]},
        {"id": 3, "text": "c3", "sources": ["vec"]},
    ]
    out = fuse_multi_query_results([a, b], rrf_k=60)
    by_id = {r["id"]: r for r in out}

    # chunk2 两列表均命中: rank2 于 a + rank1 于 b
    assert abs(by_id[2]["rrf_score"] - (1 / 62 + 1 / 61)) < 1e-9
    # sources 保序并集去重
    assert by_id[2]["sources"] == ["bm25", "vec", "graph"]
    # 双命中排最前
    assert out[0]["id"] == 2
    # 单命中: a 中 rank1
    assert abs(by_id[1]["rrf_score"] - 1 / 61) < 1e-9
    # 输入行未被原地修改(融合基于浅拷贝)
    assert "rrf_score" not in a[0]


def test_settings_has_query_decomposition_toggle():
    from rag.config import Settings

    assert Settings.model_fields["QUERY_DECOMPOSITION_ENABLED"].default is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retriever.py -v -k "fuse_multi_query or decomposition_toggle"`
Expected: FAIL, `ImportError: cannot import name 'fuse_multi_query_results'` 与 `KeyError: 'QUERY_DECOMPOSITION_ENABLED'`

- [ ] **Step 3: 实现**

`rag/config.py` 在语义缓存段 (`SEMANTIC_CACHE_TTL_HOURS` 行) 之后、Loki 段之前加:

```python
    # ── 查询分解(handle_query 拆子问题 → Send 扇出并行检索)──
    QUERY_DECOMPOSITION_ENABLED: bool = False  # 关闭时路由恒单分支,行为与现状一致
```

`rag/document/retriever.py` 在 `_merge_dedup` 函数之后加:

```python
def fuse_multi_query_results(
    result_lists: list[list[dict]], rrf_k: int = 60,
) -> list[dict]:
    """跨子查询二级 RRF 融合:每份列表按排名计 1/(k+rank),同 chunk 多列表命中分数相加。

    单份非空列表原样浅拷贝返回,保证单分支路径与现状行为一致。
    sources 取保序并集;不做 top_k 截断,交由下游 rerank/dynamic_topk 收敛。
    agent 的 recall_fuse 节点与 eval 第 9 轨共用此函数,保证评测即线上逻辑。
    """
    lists = [rows for rows in result_lists if rows]
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])

    chunk_map: dict[str, dict] = {}
    for rows in lists:
        for rank, row in enumerate(rows, 1):
            rid = str(row["id"])
            rrf = 1.0 / (rrf_k + rank)
            if rid in chunk_map:
                merged = chunk_map[rid]
                merged["rrf_score"] += rrf
                merged["sources"] = list(dict.fromkeys(
                    merged["sources"] + list(row.get("sources", []))
                ))
            else:
                merged = dict(row)
                merged["sources"] = list(row.get("sources", []))
                merged["rrf_score"] = rrf
                chunk_map[rid] = merged

    return sorted(chunk_map.values(), key=lambda r: r["rrf_score"], reverse=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: 全部 PASS (含既有测试, 确认无回归)

- [ ] **Step 5: 半角标点自检 + 提交**

```bash
python -c "s=open('rag/document/retriever.py',encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"
python -c "s=open('rag/config.py',encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"
```
(注意: 这两个文件有历史遗留全角, 只要求本次新增行是半角——用 `git diff` 目检新增行即可, 上面命令输出非 CLEAN 时对照 diff 确认全角不在新增行内。)

```bash
git add rag/config.py rag/document/retriever.py tests/test_retriever.py
git commit rag/config.py rag/document/retriever.py tests/test_retriever.py -m "feat(retriever): 跨子查询二级 RRF 融合纯函数与查询分解开关

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short
```
验证输出仍含 `D  alembic/versions/0007_partition_chunks_by_kb.py`。

---

### Task 2: handle_query 拆解输出 (sub_queries)

**Files:**
- Modify: `rag/agent/nodes/query/query.py`
- Modify: `rag/prompts/query.py`
- Modify: `rag/agent/type.py` (MyState 加 `sub_queries` 字段声明)
- Test: `tests/test_nodes.py` (handle_query 测试段之后追加)

**Interfaces:**
- Consumes: 无
- Produces: `QueryRewriteOutput.sub_queries: list[str]` (default_factory=list); `state["sub_queries"]: list[str]` (已 clamp 到 ≤3, Task 3 路由消费); 拆解时 STATUS 事件 `"已拆解为 N 个子问题"`

- [ ] **Step 1: 写失败测试**

在 `tests/test_nodes.py` 的 `test_handle_query_failure_degrades_entities_empty` 之后追加 (该文件顶部已有 `SimpleNamespace`/`ContextSchema`/`query_mod` 导入, 复用即可):

```python
async def test_handle_query_sub_queries_clamped_to_three(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="rw", is_out_of_scope=False,
                sub_queries=["s1", "s2", "s3", "s4"],
            )

    runtime = SimpleNamespace(context=ContextSchema(llm=_DecomposeLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    # 代码层 clamp 到 3,防 LLM 超量输出
    assert out["sub_queries"] == ["s1", "s2", "s3"]


async def test_handle_query_decompose_emits_status(monkeypatch):
    emitted: list[dict] = []
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    from rag.agent.nodes.query.query import QueryRewriteOutput

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="rw", is_out_of_scope=False, sub_queries=["s1", "s2"],
            )

    runtime = SimpleNamespace(context=ContextSchema(llm=_DecomposeLLM(), memory_manager=None))
    await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert {"type": "status", "data": "已拆解为 2 个子问题"} in emitted


async def test_handle_query_no_decompose_by_default(monkeypatch):
    """简单问题 LLM 不拆解(默认空数组),不发拆解状态事件。"""
    emitted: list[dict] = []
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    llm = _FakeLLM()
    runtime = SimpleNamespace(context=ContextSchema(llm=llm, memory_manager=None))

    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert out["sub_queries"] == []
    assert not any("拆解" in ev.get("data", "") for ev in emitted if ev.get("type") == "status")


async def test_handle_query_failure_degrades_sub_queries_empty(monkeypatch):
    monkeypatch.setattr(query_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _BoomLLM:
        async def ainvoke_structured(self, messages, schema):
            raise RuntimeError("llm down")

    runtime = SimpleNamespace(context=ContextSchema(llm=_BoomLLM(), memory_manager=None))
    out = await query_mod.handle_query(
        {"session_id": "s", "raw_query": "q", "context": ""}, runtime
    )

    assert out["sub_queries"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_nodes.py -v -k "sub_queries or decompose"`
Expected: FAIL, `QueryRewriteOutput` 无 `sub_queries` 字段 (ValidationError/TypeError) 或 `KeyError: 'sub_queries'`

- [ ] **Step 3: 实现节点与模型**

`rag/agent/nodes/query/query.py` 的 `QueryRewriteOutput` 在 `entities` 字段后加:

```python
    sub_queries: list[str] = Field(
        default_factory=list,
        description="复杂问题拆解出的独立子查询(每个自包含,实体显式,无指代);简单问题或 is_out_of_scope 为 true 时返回空数组",
    )
```

`handle_query` 成功分支在 `state["query_entities"] = result.entities` 之后加:

```python
        # 上限 3 个,防 LLM 超量输出;开关关闭时路由侧不消费,此处不做 gate
        state["sub_queries"] = result.sub_queries[:3]
        if state["sub_queries"] and not result.is_out_of_scope:
            writer(stream_event(
                StreamEventType.STATUS, f"已拆解为 {len(state['sub_queries'])} 个子问题"
            ))
```

降级 except 分支的 `state["query_entities"] = []` 之后加:

```python
        state["sub_queries"] = []
```

`rag/agent/type.py` 的 `MyState` 检索段 `query_entities` 行之后加 (tab 缩进):

```python
	sub_queries: list[str]  # handle_query 拆解的子查询,Send 扇出用
```

- [ ] **Step 4: 更新 prompt**

`rag/prompts/query.py`:

1. 首行任务列表改为: `你是一个查询分析助手,同时负责四项任务:**范围判断**、**查询改写**、**实体抽取**和**查询拆解**。`
2. 任务三之后加:

```
## 任务四:查询拆解(sub_queries)

当 is_out_of_scope = false 且查询包含**多个独立检索面**时,拆解为至多 3 个子查询:
- 适用:比较类("A 和 B 谁更强")、并列类("A 的 X 和 B 的 Y 分别是什么")、多实体多事实类
- 每个子查询必须自包含:实体显式写出,不留代词与省略
- 简单单面问题(单实体单事实)不拆,返回空数组
- is_out_of_scope 为 true 时返回空数组
```

3. 输出格式行改为: `仅输出一个 JSON 对象,包含 rewrite_query(字符串)、is_out_of_scope(布尔值)、entities(字符串数组)和 sub_queries(字符串数组)四个字段。`
4. 既有 5 个示例的 JSON 各追加 `, "sub_queries": []` (闭括号前)。
5. 示例末尾加一个拆解示例:

```
上下文：空。
用户查询：孙悟空和猪八戒的兵器分别是什么
→ {"rewrite_query": "孙悟空和猪八戒的兵器分别是什么", "is_out_of_scope": false, "entities": ["孙悟空", "猪八戒"], "sub_queries": ["孙悟空的兵器是什么", "猪八戒的兵器是什么"]}
```

(prompt 正文属既有全角标点风格的用户可见文案, 新增小节按上文原样写即可; 半角约定管的是代码注释, prompt 字符串内沿用该文件既有风格。)

- [ ] **Step 5: 跑测试确认通过 + 回归**

Run: `uv run pytest tests/test_nodes.py tests/test_workflow.py -v`
Expected: 全部 PASS (既有 handle_query 测试构造 `QueryRewriteOutput` 未传 `sub_queries`, default_factory 兜底, 不应失败)

- [ ] **Step 6: 半角标点自检 + 提交**

```bash
python -c "s=open('rag/agent/nodes/query/query.py',encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"
```
(query.py 新增行必须 CLEAN 于新增部分; prompts/query.py 与 type.py 对照 diff 目检新增行。)

```bash
git add rag/agent/nodes/query/query.py rag/prompts/query.py rag/agent/type.py tests/test_nodes.py
git commit rag/agent/nodes/query/query.py rag/prompts/query.py rag/agent/type.py tests/test_nodes.py -m "feat(agent): handle_query 结构化输出扩展查询拆解 sub_queries

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short
```
验证 0007 删除仍在暂存区。

---

### Task 3: Send 扇出 + recall 改造 + recall_fuse 汇聚

**Files:**
- Modify: `rag/agent/type.py` (MyState 加 reducer 字段)
- Modify: `rag/agent/nodes/recall/recall.py` (整体重写为 Send 负载模式)
- Create: `rag/agent/nodes/recall_fuse/__init__.py` (空文件)
- Create: `rag/agent/nodes/recall_fuse/fuse.py`
- Modify: `rag/agent/workflow.py` (路由 + 节点 + 边 + invoke 初始 state)
- Modify: `CLAUDE.md` (workflow 相关段落)
- Test: `tests/test_nodes.py`, `tests/test_workflow.py`

**Interfaces:**
- Consumes: `fuse_multi_query_results` (Task 1), `state["sub_queries"]` (Task 2), `Settings.QUERY_DECOMPOSITION_ENABLED` (Task 1)
- Produces: `recall` 节点新契约: 输入 Send 负载 `{"sub_query": str, "entities": list[str]}`, 返回 `{"sub_recall_results": [list[dict]]}`; `recall_fuse` 写 `state["recall_vec_results"]`; `MyState.sub_recall_results: Annotated[list[list[dict]], operator.add]`

- [ ] **Step 1: 写失败测试 (recall 节点新契约)**

`tests/test_nodes.py` 中现有 recall 节点测试 (`kb_recall_mod` 段, 约 252 行附近) 之后追加。注意: 既有的 recall 测试若断言 `state["recall_vec_results"]`, 本任务会使其失败, 在 Step 3 同步改写它们为新契约 (改断言 `out["sub_recall_results"]`, 输入加 `sub_query` 键):

```python
async def test_recall_send_payload_contract(monkeypatch):
    """recall 接收 Send 负载,返回 sub_recall_results 单元素列表交由 reducer 拼接。"""
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _R:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5, entities=None):
            self.calls.append((query, entities))
            return [{"id": 1, "text": "KB1"}]

    r = _R()
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=r))

    out = await kb_recall_mod.recall({"sub_query": "sq", "entities": ["e1"]}, runtime)

    assert r.calls == [("sq", ["e1"])]
    assert out == {"sub_recall_results": [[{"id": 1, "text": "KB1"}]]}


async def test_recall_branch_failure_degrades_empty(monkeypatch):
    """单分支 retriever 异常必须兜住:Send 分支抛异常会 fail 整个 run。"""
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))

    class _Boom:
        async def search(self, query, knowledge_base_ids=None, top_k=5, entities=None):
            raise RuntimeError("pg down")

    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=_Boom()))
    out = await kb_recall_mod.recall({"sub_query": "sq", "entities": []}, runtime)

    assert out == {"sub_recall_results": [[]]}


async def test_recall_no_retriever_degrades_empty(monkeypatch):
    monkeypatch.setattr(kb_recall_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None, retriever=None))

    out = await kb_recall_mod.recall({"sub_query": "sq", "entities": []}, runtime)

    assert out == {"sub_recall_results": [[]]}


async def test_recall_fuse_writes_recall_vec_results(monkeypatch):
    from rag.agent.nodes.recall_fuse import fuse as fuse_mod

    monkeypatch.setattr(fuse_mod, "get_settings", lambda: SimpleNamespace(RETRIEVER_RRF_K=60))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))
    state = {"sub_recall_results": [
        [{"id": 1, "text": "c1", "sources": ["vec"]}],
        [{"id": 1, "text": "c1", "sources": ["bm25"]}, {"id": 2, "text": "c2", "sources": ["vec"]}],
    ]}

    out = await fuse_mod.recall_fuse(state, runtime)

    ids = [r["id"] for r in out["recall_vec_results"]]
    assert ids == [1, 2]  # 双命中去重且排前
    assert out["recall_vec_results"][0]["sources"] == ["vec", "bm25"]


async def test_recall_fuse_empty_input(monkeypatch):
    from rag.agent.nodes.recall_fuse import fuse as fuse_mod

    monkeypatch.setattr(fuse_mod, "get_settings", lambda: SimpleNamespace(RETRIEVER_RRF_K=60))
    runtime = SimpleNamespace(context=ContextSchema(llm=None, memory_manager=None))

    out = await fuse_mod.recall_fuse({"sub_recall_results": [[], []]}, runtime)

    assert out["recall_vec_results"] == []
```

- [ ] **Step 2: 写失败测试 (workflow 路由与拓扑)**

`tests/test_workflow.py` 修改与追加。既有 `test_route_after_cache_miss_goes_to_recall` 改为断言 Send 列表; 文件顶部补 `from types import SimpleNamespace`:

```python
def test_route_after_cache_hit_goes_to_add_memory():
    from rag.agent.workflow import _route_after_cache

    assert _route_after_cache({"cache_hit": True}) == "add_memory"


def test_route_after_cache_miss_single_branch_when_disabled(monkeypatch):
    """开关默认关闭:仅主查询单分支,sub_queries 被忽略。"""
    import rag.agent.workflow as wf_mod
    from langgraph.types import Send

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=False)
    )
    out = wf_mod._route_after_cache(
        {"rewrite_query": "rw", "query_entities": ["e1"], "sub_queries": ["s1"]}
    )

    assert out == [Send("recall", {"sub_query": "rw", "entities": ["e1"]})]


def test_route_after_cache_fans_out_when_enabled(monkeypatch):
    """开关开启:主查询带 entities + 子查询不带 entities(避免重复图召回)。"""
    import rag.agent.workflow as wf_mod
    from langgraph.types import Send

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=True)
    )
    out = wf_mod._route_after_cache(
        {"rewrite_query": "rw", "query_entities": ["e1"], "sub_queries": ["s1", "s2"]}
    )

    assert out == [
        Send("recall", {"sub_query": "rw", "entities": ["e1"]}),
        Send("recall", {"sub_query": "s1", "entities": []}),
        Send("recall", {"sub_query": "s2", "entities": []}),
    ]


def test_route_after_cache_falls_back_to_raw_query(monkeypatch):
    import rag.agent.workflow as wf_mod

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=False)
    )
    out = wf_mod._route_after_cache({"raw_query": "raw"})

    assert out[0].arg == {"sub_query": "raw", "entities": []}


def test_graph_contains_recall_fuse():
    from rag.agent.workflow import graph

    assert "recall_fuse" in set(graph.get_graph().nodes)


def test_graph_edges_recall_via_fuse():
    from rag.agent.workflow import graph

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("recall", "recall_fuse") in edges
    assert ("recall_fuse", "neighbor_expand") in edges
    # 旧的 recall -> neighbor_expand 直连必须移除
    assert ("recall", "neighbor_expand") not in edges
```

再追加集成测试 (放在 `test_invoke_out_of_scope_skips_recall` 之后):

```python
async def test_invoke_decomposition_fans_out(monkeypatch):
    """开关开启时:主查询+2 子查询共 3 个 Send 分支并行检索,融合后走完生成链路。"""
    import rag.agent.workflow as wf_mod
    from rag.agent.nodes.query.query import QueryRewriteOutput

    monkeypatch.setattr(
        wf_mod, "get_settings", lambda: SimpleNamespace(QUERY_DECOMPOSITION_ENABLED=True)
    )

    class _DecomposeLLM:
        async def ainvoke_structured(self, messages, schema):
            return QueryRewriteOutput(
                rewrite_query="孙悟空和猪八戒的兵器",
                is_out_of_scope=False,
                sub_queries=["孙悟空的兵器", "猪八戒的兵器"],
            )

        async def astream(self, messages):
            yield "答"

    class _PerQueryRetriever:
        def __init__(self):
            self.calls = []

        async def search(self, query, knowledge_base_ids=None, top_k=5, entities=None):
            self.calls.append(query)
            return [{"id": f"c-{query}", "text": f"KB-{query}", "sources": ["vec"]}]

    retriever = _PerQueryRetriever()
    ctx = ContextSchema(llm=_DecomposeLLM(), memory_manager=_FakeMM(), retriever=retriever)

    messages: list[str] = []
    async for event in wf.invoke("s1", "q", ctx):
        if event["type"] == "message":
            messages.append(event["data"])

    # 三个分支各检索一次(Send 并行,顺序不保证,用 set 比较)
    assert set(retriever.calls) == {"孙悟空和猪八戒的兵器", "孙悟空的兵器", "猪八戒的兵器"}
    assert len(retriever.calls) == 3
    # 融合结果非空,正常走到 generate
    assert messages == ["答"]
```

- [ ] **Step 3: 实现**

`rag/agent/type.py` — `MyState` 召回段 (tab 缩进), 在 `recall_bm25_results` 之前加:

```python
	# Send 分支 fan-in:各 recall 分支 append 单元素列表,operator.add 拼接。
	# 本项目首个 reducer 字段,仅此一处,其余字段保持 last-write-wins。
	sub_recall_results: Annotated[list[list[dict]], operator.add]
```

文件顶部 import 区加 (注意不与既有导入重复):

```python
import operator
from typing import Annotated
```

(既有 `from typing import Protocol, TypedDict, runtime_checkable` 行直接并入 `Annotated`: `from typing import Annotated, Protocol, TypedDict, runtime_checkable`, 不要新开一行重复导入 typing。)

`rag/agent/nodes/recall/recall.py` 整体替换为:

```python
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.common.logging import get_logger
from rag.agent.type import ContextSchema, StreamEventType, stream_event

logger = get_logger()


async def recall(state: dict, runtime: Runtime[ContextSchema]) -> dict:
    """单分支知识库召回:接收 Send 负载 {sub_query, entities}。

    结果以单元素列表返回,经 sub_recall_results 的 operator.add reducer 与
    其他并行分支拼接,recall_fuse 统一融合。分支内任何异常必须兜住:
    Send 分支抛异常会 fail 整个 run。
    """
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "检索知识库中..."))

    retriever = runtime.context.retriever
    if retriever is None:
        logger.warning("retriever 未注入,跳过知识库召回")
        return {"sub_recall_results": [[]]}

    try:
        # kb_ids 不传 → 搜全部知识库;entities 仅主查询分支非空(图召回入口)
        rows = await retriever.search(
            state["sub_query"], entities=state.get("entities") or None
        )
    except Exception:  # noqa: BLE001 - 分支级容错,单分支失败不拖垮整图
        logger.warning("子查询召回失败,该分支降级为空", exc_info=True)
        rows = []
    return {"sub_recall_results": [rows]}
```

`rag/agent/nodes/recall_fuse/__init__.py`: 空文件。

`rag/agent/nodes/recall_fuse/fuse.py`:

```python
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState
from rag.config import get_settings
from rag.document.retriever import fuse_multi_query_results


async def recall_fuse(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """Send 分支 fan-in:跨子查询二级 RRF 融合去重,写入 recall_vec_results。

    单分支时融合函数原样透传,行为与拆解前完全一致;
    全空时产出空列表,交由 _route_after_topk 走 no_results 兜底。
    """
    result_lists = state.get("sub_recall_results") or []
    state["recall_vec_results"] = fuse_multi_query_results(
        result_lists, rrf_k=get_settings().RETRIEVER_RRF_K
    )
    return state
```

`rag/agent/workflow.py`:

1. import 区加:

```python
from langgraph.types import Send

from rag.agent.nodes.recall_fuse.fuse import recall_fuse
from rag.config import get_settings
```

2. `_route_after_cache` 替换为:

```python
def _route_after_cache(state: MyState):
    """条件边:缓存命中直达记忆写入(答案已流式下发);
    未命中按 [主查询]+sub_queries 扇出 Send 并行召回(开关关闭时恒单分支)。
    仅主查询分支携带 entities,避免多分支用同一实体集重复图召回。
    """
    if state.get("cache_hit"):
        return "add_memory"
    main_query = state.get("rewrite_query") or state["raw_query"]
    branches = [Send("recall", {
        "sub_query": main_query,
        "entities": state.get("query_entities") or [],
    })]
    if get_settings().QUERY_DECOMPOSITION_ENABLED:
        branches += [
            Send("recall", {"sub_query": sq, "entities": []})
            for sq in state.get("sub_queries") or []
        ]
    return branches
```

既有 `add_conditional_edges("cache_lookup", _route_after_cache, {...})` 的 path_map 保持不变: 返回 Send 列表时 LangGraph 直接按 Send 分发, path_map 仅服务字符串返回值与拓扑图渲染。

3. 节点注册区 `recall` 之后加:

```python
# Send 分支 fan-in:跨子查询融合去重
builder.add_node("recall_fuse", recall_fuse)
```

4. 边: `builder.add_edge("recall", "neighbor_expand")` 替换为:

```python
builder.add_edge("recall", "recall_fuse")
builder.add_edge("recall_fuse", "neighbor_expand")
```

5. ASCII 拓扑注释更新 (未命中行改为):

```
#                                                                    └─ 未命中 ─→ [Send x N] recall ─→ recall_fuse -> neighbor_expand -> rerank -> dynamic_topk -> parent_expand ┬─ generate -> cache_store -> add_memory ─┘
#                                                                                                                                                                              └─ no_results ────────────────────────────┘
```

6. `invoke()` 初始 state 加两个键:

```python
        {
            "session_id": session_id,
            "raw_query": query,
            "is_out_of_scope": False,
            "sub_queries": [],
            "sub_recall_results": [],
        },
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_nodes.py tests/test_workflow.py -v`
Expected: 全部 PASS。既有 `test_invoke_runs_full_graph_with_context` 应保持绿色 (默认开关关闭 → 单分支, `retriever.calls == [("rw", None)]` 不变)。若 tests/test_nodes.py 中旧 recall 测试断言旧契约, 按 Step 1 说明改写为新契约。

- [ ] **Step 5: 全量单测回归**

Run: `uv run pytest tests/ -v`
Expected: 全部 PASS (pyproject addopts 默认跳过 integration/eval)

- [ ] **Step 6: 更新 CLAUDE.md workflow 段**

1. "13-node pipeline" 两处改 "14-node pipeline"。
2. 拓扑图 miss 行改为:
   `└─ miss → [Send × N] recall → recall_fuse → neighbor_expand → rerank → dynamic_topk → parent_expand ┬─ generate → cache_store → add_memory ─┘`
3. 节点表 `recall` 行说明改为: "Hybrid retrieval per Send branch: vector + BM25 + graph (main branch only) concurrent recall; branches carry `{sub_query, entities}` payloads"; 其后插入一行:
   `| recall_fuse | nodes/recall_fuse/ | Fan-in of Send branches: second-level RRF fusion across sub-query result lists (shared fuse_multi_query_results()), writes recall_vec_results |`
4. handle_query 行说明追加: "+ query decomposition (`sub_queries`, ≤3, gated by `QUERY_DECOMPOSITION_ENABLED` at routing)"。
5. Conditional routing 的 `_route_after_cache` 条目改为: "`cache_hit=True` → `add_memory`; otherwise returns `list[Send]` fanning out to `recall` — branches are `[rewrite_query] + sub_queries` (sub-queries only when `QUERY_DECOMPOSITION_ENABLED`), entities ride only on the main branch"。
6. State type 段: 字段列表加 `sub_queries`、`sub_recall_results`; "no Annotated reducers" 表述改为 "`sub_recall_results` is the only Annotated reducer field (`operator.add`, Send fan-in); all other fields are plain last-write-wins"。

- [ ] **Step 7: 半角标点自检 + 提交**

```bash
python -c "s=open('rag/agent/nodes/recall/recall.py',encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"
python -c "s=open('rag/agent/nodes/recall_fuse/fuse.py',encoding='utf-8').read(); print([c for c in '：，；（）' if c in s] or 'CLEAN')"
```
(workflow.py/type.py 有历史全角, 对照 diff 目检新增行。)

```bash
git add rag/agent/type.py rag/agent/nodes/recall/recall.py rag/agent/nodes/recall_fuse/__init__.py rag/agent/nodes/recall_fuse/fuse.py rag/agent/workflow.py CLAUDE.md tests/test_nodes.py tests/test_workflow.py
git commit rag/agent/type.py rag/agent/nodes/recall/recall.py rag/agent/nodes/recall_fuse/__init__.py rag/agent/nodes/recall_fuse/fuse.py rag/agent/workflow.py CLAUDE.md tests/test_nodes.py tests/test_workflow.py -m "feat(agent): Send 扇出并行子查询检索与 recall_fuse 二级融合

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short
```
验证 0007 删除仍在暂存区。

---

### Task 4: 评测第 9 轨 decomposed

**Files:**
- Modify: `rag/eval/harness.py` (GoldenItem + load_golden + run_eval)
- Modify: `rag/eval/run.py` (leg 注册 + 拆解收益 + 门禁 legs)
- Modify: `CLAUDE.md` (eval 段 8 轨 → 9 轨)
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `fuse_multi_query_results` (Task 1)
- Produces: `GoldenItem.sub_queries: list[str]`; `run_eval` 结果字典可含 `"decomposed"` leg (aggregate + per_query, 结构与其他 leg 一致); run.py 的 `_LEG_LABEL["decomposed"] = "拆解融合"`

- [ ] **Step 1: 写失败测试**

`tests/test_eval_harness.py` 追加 (该文件既有 load_golden 测试, 沿用其临时文件写法; 若无现成 fixture, 用 `tmp_path`):

```python
def test_load_golden_parses_sub_queries(tmp_path):
    from rag.eval.harness import load_golden

    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id": "q1", "query": "A和B的兵器", "gold_snippets": ["s"], '
        '"sub_queries": ["A的兵器", "B的兵器"]}\n'
        '{"id": "q2", "query": "单面问题", "gold_snippets": ["s"]}\n',
        encoding="utf-8",
    )
    items = load_golden(p)

    assert items[0].sub_queries == ["A的兵器", "B的兵器"]
    assert items[1].sub_queries == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_harness.py -v -k sub_queries`
Expected: FAIL, `GoldenItem` 无 `sub_queries` 属性 (AttributeError)

- [ ] **Step 3: 实现 harness**

`rag/eval/harness.py`:

1. `GoldenItem` 加字段 (entities 之后): `sub_queries: list[str] = field(default_factory=list)`。
2. `load_golden` 的 `GoldenItem(...)` 构造加: `sub_queries=list(obj.get("sub_queries", [])),`; docstring 可选字段说明追加 `sub_queries(拆解轨用的子查询标注,默认空列表)`。
3. import 区: `from rag.document.retriever import KnowledgeRetriever, _lexical_query` 行加入 `fuse_multi_query_results`; 新增 `from rag.config import get_settings`。
4. `run_eval` docstring 追加一行: `当条目带 sub_queries 标注时追加 decomposed 拆解融合轨。`
5. embedding 预计算段, `unique_queries = list(dict.fromkeys(unique_queries))` 之前加:

```python
    sub_qs = [sq for it in in_scope for sq in it.sub_queries]
    unique_queries += sub_qs
```

6. 计数器区 `graph_pq: list[dict] = []` 之后加: `decomposed_pq: list[dict] = []`; 循环外(预计算段后)取一次 `rrf_k = get_settings().RETRIEVER_RRF_K`。
7. 循环内 graph_fused 块之后加:

```python
        # ── decomposed:主查询+子查询各自检索后二级 RRF 融合,仅当条目带 sub_queries 标注 ──
        if item.sub_queries:
            branch_rows = [rows]  # 主查询分支复用 fused 轨已检索的结果
            for sq in item.sub_queries:
                branch_rows.append(
                    await retriever.search(sq, None, top_k=top_k, query_emb=q_embs[sq])
                )
            dec_rows = fuse_multi_query_results(branch_rows, rrf_k=rrf_k)
            decomposed_pq.append({
                "id": item.id, "query": item.query,
                **evaluate_query([r["text"] for r in dec_rows], item.gold_snippets, ks),
                "chunks": _chunk_preview(dec_rows),
            })
```

8. 结果组装区 `if graph_pq:` 块之后加:

```python
    if decomposed_pq:
        result["decomposed"] = {"aggregate": aggregate(decomposed_pq), "per_query": decomposed_pq}
```

- [ ] **Step 4: 实现 run.py**

`rag/eval/run.py`:

1. `_LEG_LABEL` 加: `"decomposed": "拆解融合",` (graph_fused 行后)。
2. `_LEG_ORDER` 改: `("fused", "fused_reranked", "graph_fused", "decomposed", "raw", "vec_only", "raw_vec", "bm25_only", "raw_bm25")`。
3. 门禁 legs 元组改: `("fused", "fused_reranked", "graph_fused", "decomposed", "vec_only", "bm25_only")` (缺 baseline 时 `gate(cur, {})` 无 deltas 即视为跳过, 与 graph_fused 先例一致)。
4. import 行 `from rag.eval.metrics import ...` 加入 `aggregate`。
5. `_print_table` 的重排收益块之后加:

```python
    # ── 拆解收益:同子集(带 sub_queries 标注条目)内 decomposed vs fused ──
    if "decomposed" in result and "fused" in result:
        dec_ids = {pq["id"] for pq in result["decomposed"]["per_query"]}
        fused_subset = [pq for pq in result["fused"]["per_query"] if pq["id"] in dec_ids]
        if fused_subset:
            print(bold("  🧩 拆解收益(仅带标注条目)"))
            base_agg = aggregate(fused_subset)
            dec_agg = result["decomposed"]["aggregate"]
            deltas = []
            for key in sorted(base_agg):
                if key not in dec_agg:
                    continue
                deltas.append(f"{key}: {delta_str(dec_agg[key] - base_agg[key])}")
            print(f"    {'  '.join(deltas)}\n")
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_harness.py tests/test_eval_metrics.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 更新 CLAUDE.md eval 段**

1. 标题 "8-leg breakdown" → "9-leg breakdown"; "8 evaluation legs" → "9 evaluation legs"。
2. legs 列表加: `decomposed (拆解融合: 主查询+子查询二级 RRF)`; 条件说明追加 "`decomposed` only when the golden item carries `sub_queries`"。
3. Golden dataset 行: "items may also carry an optional `entities` annotation … " 追加 "and an optional `sub_queries` annotation consumed by the `decomposed` leg"。

- [ ] **Step 7: 半角标点自检 + 提交**

新增行对照 diff 目检 (harness.py/run.py 历史上中文注释混有全角, 只查本次新增)。

```bash
git add rag/eval/harness.py rag/eval/run.py CLAUDE.md tests/test_eval_harness.py
git commit rag/eval/harness.py rag/eval/run.py CLAUDE.md tests/test_eval_harness.py -m "feat(eval): 第 9 轨 decomposed 拆解融合与拆解收益汇总

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short
```
验证 0007 删除仍在暂存区。

---

### Task 5: golden 标注 sub_queries + 端到端验证

**Files:**
- Modify: `rag/eval/datasets/retrieval_golden.jsonl`
- (验证脚本临时放 scratchpad, 不入库)

**Interfaces:**
- Consumes: `load_golden` 的 `sub_queries` 解析 (Task 4)
- Produces: 带 `sub_queries` 标注的 golden 条目 (既有 multi_chunk/multi_doc 类适格条目 + 新增 5~10 条多面型条目, id 从既有最大编号顺延)

- [ ] **Step 1: 标注既有条目**

读取 `rag/eval/datasets/retrieval_golden.jsonl`, 逐条检查 `category` 为 `multi_chunk`/`multi_doc` 的条目: 若查询确含多个独立检索面 (比较/并列/多实体多事实), 加 `"sub_queries": [...]` 字段, 每个子查询自包含; 单面聚合类 (同一事实跨 chunk) 不标注。不确定的宁可不标。

- [ ] **Step 2: 新增 5~10 条多面型条目**

按既有行格式追加, id 顺延 (如 q101 起), category 用 `multi_doc` 或 `multi_chunk`, 全部带 `rewrite_query`、`entities`、`sub_queries`。题材: 西游记比较/并列问题 (语料为 `rag/eval/datasets/corpus.txt`)。示例形态 (gold_snippets 必须逐条从 corpus.txt 原文摘取, 保持 50 字以内的连续短句以避开 chunk 边界):

```json
{"id": "q101", "query": "孙悟空和猪八戒的兵器分别是什么？", "rewrite_query": "孙悟空 猪八戒 兵器", "gold_snippets": ["<从 corpus.txt 摘取的金箍棒描述短句>", "<从 corpus.txt 摘取的九齿钉钯描述短句>"], "out_of_scope": false, "category": "multi_doc", "entities": ["孙悟空", "猪八戒"], "sub_queries": ["孙悟空的兵器是什么", "猪八戒的兵器是什么"]}
```

(尖括号占位仅为本计划示意, 落盘条目必须是真实摘取的原文短句, 禁止占位符入库。)

- [ ] **Step 3: 校验标注格式与 snippet 真实性**

写临时脚本到 scratchpad 并运行:

```python
import json
from pathlib import Path

corpus = Path("rag/eval/datasets/corpus.txt").read_text(encoding="utf-8").replace("\n", "")
ok = True
n_annotated = 0
for lineno, line in enumerate(
    Path("rag/eval/datasets/retrieval_golden.jsonl").read_text(encoding="utf-8").splitlines(), 1
):
    if not line.strip():
        continue
    obj = json.loads(line)
    if obj.get("sub_queries"):
        n_annotated += 1
        if len(obj["sub_queries"]) < 2:
            ok = False
            print("BAD sub_queries(<2)", obj["id"])
    for sn in obj.get("gold_snippets", []):
        if sn.replace("\n", "") not in corpus:
            ok = False
            print("MISSING SNIPPET", obj["id"], sn[:40])
print(f"annotated={n_annotated}")
print("OK" if ok else "FAILED")
```

Run: `uv run python <scratchpad>/validate_golden.py`
Expected: `annotated=<既有标注数+5~10>`, `OK`。再跑 `uv run pytest tests/test_eval_harness.py -v` 确认 load_golden 无回归。

- [ ] **Step 4: 端到端 eval 验证 (需基础设施)**

前置: `docker compose up -d postgres` + eval KB 已 seed (`rag/eval/seed_corpus.py`) + `.env` 有 embedding key。若本机基础设施未就绪, 此步骤标注为待用户环境执行, 不阻塞提交。

Run: `uv run rag-eval`
Expected: 指标总览出现 "拆解融合" 列; "🧩 拆解收益(仅带标注条目)" 块输出各指标 delta; 门禁不因 decomposed 缺 baseline 而失败。

**不要在本任务执行 `--update-baseline`** — 是否把 decomposed 纳入基线由用户看完增益数据后决定。

- [ ] **Step 5: 提交**

```bash
git add rag/eval/datasets/retrieval_golden.jsonl
git commit rag/eval/datasets/retrieval_golden.jsonl -m "feat(eval): golden 集 sub_queries 标注与多面型条目

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git status --short
```
验证 0007 删除仍在暂存区。
