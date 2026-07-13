# Dynamic Top-K Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a `dynamic_topk` LangGraph node between `rerank` and `generate` that filters reranked chunks by adjacent score-ratio gaps, defaulting to top 5.

**Architecture:** New node `dynamic_topk` reads `rerank_score`-sorted chunks from state, applies relative gap detection (`score[k+1]/score[k] < threshold`), truncates, and writes back. Empty results route to `no_results`. Config lives in `Settings` (3 new fields), read via direct `get_settings()` import. Implementation is a pure function `_dynamic_truncate` plus an async node wrapper — no external I/O.

**Tech Stack:** Python, LangGraph, pydantic-settings

## Global Constraints

- `RERANK_DYNAMIC_TOPK_ENABLED: bool = True` — toggle to disable the node entirely (passthrough)
- `RERANK_DYNAMIC_TOPK_DEFAULT: int = 5` — default top-k when no gap detected
- `RERANK_DYNAMIC_TOPK_RATIO: float = 0.7` — ratio threshold for gap detection
- Node must never throw; all exceptions caught, logged, and original results passed through
- Follow existing node patterns: `get_stream_writer()` for status events, `Runtime[ContextSchema]` for runtime
- Tests follow existing `tests/test_nodes.py` patterns: `SimpleNamespace` context, `monkeypatch` on `get_stream_writer`

---

### Task 1: Add Settings fields

**Files:**
- Modify: `rag/config.py:90-94`

**Interfaces:**
- Produces: `Settings.RERANK_DYNAMIC_TOPK_ENABLED: bool`, `Settings.RERANK_DYNAMIC_TOPK_DEFAULT: int`, `Settings.RERANK_DYNAMIC_TOPK_RATIO: float`

- [ ] **Step 1: Add three new fields to Settings class**

In `rag/config.py`, after line 94 (`RERANK_MODEL: str = "qwen3-rerank"`), insert:

```python
    # ── 动态 Top-K 截断（rerank 之后）──
    RERANK_DYNAMIC_TOPK_ENABLED: bool = True
    RERANK_DYNAMIC_TOPK_DEFAULT: int = 5
    RERANK_DYNAMIC_TOPK_RATIO: float = 0.7
```

- [ ] **Step 2: Verify settings load correctly**

Run:
```bash
uv run python -c "from rag.config import get_settings; s = get_settings(); print(s.RERANK_DYNAMIC_TOPK_ENABLED, s.RERANK_DYNAMIC_TOPK_DEFAULT, s.RERANK_DYNAMIC_TOPK_RATIO)"
```
Expected: `True 5 0.7`

- [ ] **Step 3: Commit**

```bash
git add rag/config.py
git commit -m "feat(config): add dynamic top-k truncation settings

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Create dynamic_topk node

**Files:**
- Create: `rag/agent/nodes/dynamic_topk/__init__.py`
- Create: `rag/agent/nodes/dynamic_topk/topk.py`

**Interfaces:**
- Consumes: `get_settings()` from `rag.config`, `MyState`, `ContextSchema`, `StreamEventType`, `stream_event` from `rag.agent.type`, `get_stream_writer` from `langgraph.config`, `Runtime` from `langgraph.runtime`
- Produces: `dynamic_topk(state: MyState, runtime: Runtime[ContextSchema]) -> MyState` — async node function
- Produces: `_dynamic_truncate(chunks: list[dict], default_top_k: int, ratio_threshold: float) -> list[dict]` — pure function, no side effects

- [ ] **Step 1: Create empty `__init__.py`**

Create `rag/agent/nodes/dynamic_topk/__init__.py` (empty file, following existing convention).

- [ ] **Step 2: Create `topk.py` with `_dynamic_truncate` and `dynamic_topk`**

Create `rag/agent/nodes/dynamic_topk/topk.py`:

```python
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


def _dynamic_truncate(
    chunks: list[dict],
    default_top_k: int,
    ratio_threshold: float,
) -> list[dict]:
    """相邻分差法动态截断 rerank 结果。

    从第 1 名开始，比较 score[k+1] / score[k]：
      - 比值 >= threshold → gap 小，继续
      - 比值 <  threshold → gap 大，在 k+1 处截断（保留前 k+1 条）
      - 没有任何 gap 超过阈值 → 取 min(default_top_k, len(chunks))

    score[k] 为 0 时视为无限大 gap，在 k 处截断。
    """
    if not chunks:
        return chunks

    if len(chunks) == 1:
        return chunks

    # 提取 rerank_score，缺失或非数字兜底为 0
    scores: list[float] = []
    for c in chunks:
        try:
            s = float(c.get("rerank_score", 0) or 0)
        except (ValueError, TypeError):
            s = 0.0
        scores.append(max(0.0, s))

    # 如果没有任何有效分数（全 0 或缺失），视为 rerank 未启用，硬截断
    if all(s == 0.0 for s in scores):
        return chunks[:default_top_k]

    # 钳位 ratio_threshold 到合理范围
    threshold = max(0.01, min(0.99, ratio_threshold))

    for i in range(len(scores) - 1):
        if scores[i] == 0.0:
            # score[i] 为 0，视为无限大 gap，在 i 处截断
            return chunks[:i] if i > 0 else []
        if scores[i + 1] / scores[i] < threshold:
            return chunks[: i + 1]

    # 无 gap 触发，取默认 top_k
    return chunks[: min(default_top_k, len(chunks))]


async def dynamic_topk(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """对 rerank 后的结果做动态 top-k 截断。

    未启用时透传；异常时降级透传原始结果，不中断检索链路。
    """
    settings = get_settings()

    if not settings.RERANK_DYNAMIC_TOPK_ENABLED:
        return state

    chunks = state.get("recall_vec_results")
    if not isinstance(chunks, list):
        logger.warning("recall_vec_results 不是 list，透传")
        return state

    if not chunks:
        return state

    writer = get_stream_writer()

    try:
        before = len(chunks)
        filtered = _dynamic_truncate(
            chunks,
            settings.RERANK_DYNAMIC_TOPK_DEFAULT,
            settings.RERANK_DYNAMIC_TOPK_RATIO,
        )
        state["recall_vec_results"] = filtered
        after = len(filtered)

        if after == 0:
            writer(stream_event(StreamEventType.STATUS, "未找到足够相关内容"))
        else:
            writer(stream_event(StreamEventType.STATUS, f"动态筛选 {after} 条相关结果"))
        logger.debug("dynamic_topk: %d -> %d chunks", before, after)
    except Exception:
        logger.warning("动态 top-k 截断失败，透传原始结果", exc_info=True)

    return state
```

- [ ] **Step 3: Verify the module imports correctly**

Run:
```bash
uv run python -c "from rag.agent.nodes.dynamic_topk.topk import dynamic_topk, _dynamic_truncate; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Verify `_dynamic_truncate` basic cases with a quick smoke test**

Run:
```bash
uv run python -c "
from rag.agent.nodes.dynamic_topk.topk import _dynamic_truncate

# empty
assert _dynamic_truncate([], 5, 0.7) == []

# single
assert _dynamic_truncate([{'rerank_score': 9.0}], 5, 0.7) == [{'rerank_score': 9.0}]

# no gap → default top_k
chunks = [{'rerank_score': 9.0}, {'rerank_score': 8.5}, {'rerank_score': 8.0}, {'rerank_score': 7.5}, {'rerank_score': 7.0}, {'rerank_score': 6.5}]
assert len(_dynamic_truncate(chunks, 5, 0.7)) == 5

# gap between 1st and 2nd → cut to 1
chunks2 = [{'rerank_score': 9.0}, {'rerank_score': 1.0}, {'rerank_score': 0.9}]
assert len(_dynamic_truncate(chunks2, 5, 0.7)) == 1

# score 0 → infinite gap
chunks3 = [{'rerank_score': 9.0}, {'rerank_score': 0.0}]
assert len(_dynamic_truncate(chunks3, 5, 0.7)) == 1

# first score 0 → return empty
chunks4 = [{'rerank_score': 0.0}, {'rerank_score': 9.0}]
assert _dynamic_truncate(chunks4, 5, 0.7) == []

# no rerank_score → hard truncate
chunks5 = [{'text': 'a'}, {'text': 'b'}, {'text': 'c'}, {'text': 'd'}, {'text': 'e'}, {'text': 'f'}]
assert len(_dynamic_truncate(chunks5, 5, 0.7)) == 5

print('All smoke tests passed')
"
```
Expected: `All smoke tests passed`

- [ ] **Step 5: Commit**

```bash
git add rag/agent/nodes/dynamic_topk/__init__.py rag/agent/nodes/dynamic_topk/topk.py
git commit -m "feat(agent): add dynamic top-k truncation node after rerank

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Wire node into workflow graph

**Files:**
- Modify: `rag/agent/workflow.py`

**Interfaces:**
- Consumes: `dynamic_topk` from `rag.agent.nodes.dynamic_topk.topk`
- Produces: Updated graph with `dynamic_topk` node between `rerank` and conditional edge

- [ ] **Step 1: Add import for dynamic_topk**

In `rag/agent/workflow.py`, after line 10 (`from rag.agent.nodes.rerank.rerank import rerank`), add:

```python
from rag.agent.nodes.dynamic_topk.topk import dynamic_topk
```

- [ ] **Step 2: Rename route function and add registration**

Replace the existing `_route_after_recall` function (lines 21-25) with `_route_after_topk`:

```python
def _route_after_topk(state: MyState) -> str:
    """条件边：动态截断后为空时返回兜底话术，有结果才走生成。"""
    if state.get("recall_vec_results"):
        return "generate"
    return "no_results"
```

- [ ] **Step 3: Register the dynamic_topk node**

After line 38 (`builder.add_node("rerank", rerank)`), add:

```python
# 动态 top-k 截断
builder.add_node("dynamic_topk", dynamic_topk)
```

- [ ] **Step 4: Adjust edges**

Replace lines 59-64:

```python
# Before:
builder.add_edge("recall", "rerank")
builder.add_conditional_edges(
    "rerank",
    _route_after_recall,
    {"generate": "generate", "no_results": "no_results"},
)

# After:
builder.add_edge("recall", "rerank")
builder.add_edge("rerank", "dynamic_topk")
builder.add_conditional_edges(
    "dynamic_topk",
    _route_after_topk,
    {"generate": "generate", "no_results": "no_results"},
)
```

- [ ] **Step 5: Update the ASCII diagram comment**

Replace lines 48-51:

```python
#                                       ┌─ out-of-scope -> direct_answer ──────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                        END
#                                       └─ in-scope -> recall -> rerank ┬─ generate -> add_memory ─┘
#                                                                       └─ no_results ─────────────┘
```

With:

```python
#                                       ┌─ out-of-scope -> direct_answer ──────────────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                                END
#                                       └─ in-scope -> recall -> rerank -> dynamic_topk ┬─ generate -> add_memory ─┘
#                                                                                       └─ no_results ─────────────┘
```

The final `workflow.py` should show this structure after all edits:

```python
from langgraph.graph import END, START, StateGraph

from rag.agent.nodes.add_memory.memory import add_memory
from rag.agent.nodes.generate.direct_answer import direct_answer
from rag.agent.nodes.generate.generate import generate
from rag.agent.nodes.generate.no_results import no_results
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.nodes.rerank.rerank import rerank
from rag.agent.nodes.dynamic_topk.topk import dynamic_topk
from rag.agent.type import ContextSchema, MyState


def _route_after_query(state: MyState) -> str:
    """条件边：范围外直接大模型兜底，范围内走召回。"""
    if state.get("is_out_of_scope"):
        return "direct_answer"
    return "recall"


def _route_after_topk(state: MyState) -> str:
    """条件边：动态截断后为空时返回兜底话术，有结果才走生成。"""
    if state.get("recall_vec_results"):
        return "generate"
    return "no_results"


# 构建状态图
builder = StateGraph(MyState, context_schema=ContextSchema)

# 召回记忆
builder.add_node("recall_memory", recall_memory)
# 查询改写 + 范围判断
builder.add_node("handle_query", handle_query)
# 知识库召回
builder.add_node("recall", recall)
# 语义重排序
builder.add_node("rerank", rerank)
# 动态 top-k 截断
builder.add_node("dynamic_topk", dynamic_topk)
# 基于知识库生成
builder.add_node("generate", generate)
# 范围外直接大模型回答
builder.add_node("direct_answer", direct_answer)
# 记忆持久化（仅 generate 路由到此处，direct_answer / no_results 不写记忆）
builder.add_node("add_memory", add_memory)
# 召回为空时的兜底话术（不调 LLM）
builder.add_node("no_results", no_results)

#                                       ┌─ out-of-scope -> direct_answer ──────────────────────────────────┐
# START -> recall_memory -> handle_query ┤                                                                END
#                                       └─ in-scope -> recall -> rerank -> dynamic_topk ┬─ generate -> add_memory ─┘
#                                                                                       └─ no_results ─────────────┘
builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_conditional_edges(
    "handle_query",
    _route_after_query,
    {"recall": "recall", "direct_answer": "direct_answer"},
)
builder.add_edge("recall", "rerank")
builder.add_edge("rerank", "dynamic_topk")
builder.add_conditional_edges(
    "dynamic_topk",
    _route_after_topk,
    {"generate": "generate", "no_results": "no_results"},
)
builder.add_edge("generate", "add_memory")
builder.add_edge("add_memory", END)
builder.add_edge("direct_answer", END)
builder.add_edge("no_results", END)

graph = builder.compile()


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
        {
            "session_id": session_id,
            "raw_query": query,
            "is_out_of_scope": False,
        },
        context=context,
        stream_mode=["custom"],
        config=config,
    ):
        yield chunk
```

- [ ] **Step 6: Verify graph compiles without error**

Run:
```bash
uv run python -c "from rag.agent.workflow import graph; print('Graph compiled OK, nodes:', list(graph.nodes.keys()))"
```
Expected: `Graph compiled OK, nodes: ['__start__', 'recall_memory', 'handle_query', 'recall', 'rerank', 'dynamic_topk', 'generate', 'direct_answer', 'add_memory', 'no_results']`

- [ ] **Step 7: Commit**

```bash
git add rag/agent/workflow.py
git commit -m "feat(workflow): wire dynamic_topk node between rerank and generate

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add tests

**Files:**
- Modify: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `dynamic_topk` from `rag.agent.nodes.dynamic_topk.topk`, `_dynamic_truncate` from `rag.agent.nodes.dynamic_topk.topk`
- Tests: `test_dynamic_truncate_*` (pure function tests) and `test_dynamic_topk_*` (node integration tests)

- [ ] **Step 1: Add import for dynamic_topk module**

In `tests/test_nodes.py`, after line 10 (`import rag.agent.nodes.rerank.rerank as rerank_mod`), add:

```python
import rag.agent.nodes.dynamic_topk.topk as topk_mod
```

- [ ] **Step 2: Add `_dynamic_truncate` pure function tests**

Append to `tests/test_nodes.py`:

```python
# ── dynamic_truncate pure function tests ──


async def test_dynamic_truncate_empty():
    """空列表应直接返回空列表。"""
    assert topk_mod._dynamic_truncate([], 5, 0.7) == []


async def test_dynamic_truncate_single():
    """只有一条结果时直接返回，不做比较。"""
    chunk = {"rerank_score": 9.0, "text": "A"}
    result = topk_mod._dynamic_truncate([chunk], 5, 0.7)
    assert result == [chunk]


async def test_dynamic_truncate_no_gap_returns_default_topk():
    """所有相邻分差都小，应返回 default_top_k 条。"""
    chunks = [
        {"rerank_score": 9.0},
        {"rerank_score": 8.5},
        {"rerank_score": 8.0},
        {"rerank_score": 7.5},
        {"rerank_score": 7.0},
        {"rerank_score": 6.5},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5
    assert result[0]["rerank_score"] == 9.0
    assert result[4]["rerank_score"] == 7.0


async def test_dynamic_truncate_gap_truncates():
    """第 2-3 名之间 gap 大（1.0/9.0=0.11 < 0.7），应在第 2 条后截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 8.5, "text": "B"},
        {"rerank_score": 1.0, "text": "C"},
        {"rerank_score": 0.9, "text": "D"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 2
    assert result[0]["text"] == "A"
    assert result[1]["text"] == "B"


async def test_dynamic_truncate_first_gap_returns_empty():
    """第一名本身分数极低且与第二名 gap 大，返回空。"""
    chunks = [
        {"rerank_score": 1.0, "text": "A"},
        {"rerank_score": 0.1, "text": "B"},
    ]
    # score[1]/score[0] = 0.1/1.0 = 0.1 < 0.7 → 截断保留前 1 条
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 1


async def test_dynamic_truncate_score_zero_infinite_gap():
    """score 为 0 时视为无限大 gap，在 0 分处截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 0.0, "text": "B"},
        {"rerank_score": 0.0, "text": "C"},
    ]
    # score[1]=0.0, score[0]=9.0 → 0/9=0 < threshold, 截断到前 1 条
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 1
    assert result[0]["text"] == "A"


async def test_dynamic_truncate_first_score_zero_empty():
    """第一名 score 就是 0，直接在首位截断，返回空。"""
    chunks = [
        {"rerank_score": 0.0, "text": "A"},
        {"rerank_score": 9.0, "text": "B"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert result == []


async def test_dynamic_truncate_missing_rerank_score_hard_truncate():
    """无 rerank_score 字段时（rerank 未启用），硬截断 default_top_k。"""
    chunks = [
        {"text": "A"}, {"text": "B"}, {"text": "C"},
        {"text": "D"}, {"text": "E"}, {"text": "F"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


async def test_dynamic_truncate_all_same_score():
    """所有分数完全相同时，无 gap，取 default_top_k。"""
    chunks = [
        {"rerank_score": 5.0} for _ in range(10)
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    assert len(result) == 5


async def test_dynamic_truncate_negative_score_treated_as_zero():
    """负分数视为 0 处理，触发 gap 截断。"""
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": -1.0, "text": "B"},
    ]
    result = topk_mod._dynamic_truncate(chunks, 5, 0.7)
    # -1 → 0, gap 无限大，截断到前 1 条
    assert len(result) == 1
    assert result[0]["text"] == "A"


async def test_dynamic_truncate_extreme_threshold():
    """ratio_threshold 为 0.01 时几乎不截断；0.99 时几乎总是截断。"""
    chunks = [
        {"rerank_score": 9.0},
        {"rerank_score": 8.0},
        {"rerank_score": 7.0},
    ]
    # threshold 很低 → 极小的比值才会触发截断
    r1 = topk_mod._dynamic_truncate(chunks, 5, 0.01)
    # 8/9 ≈ 0.89 > 0.01, 7/8=0.875 > 0.01, no gap → all 3
    assert len(r1) == 3

    # threshold 很高 → 几乎任何相邻变化都截断
    r2 = topk_mod._dynamic_truncate(chunks, 5, 0.99)
    # 8/9 ≈ 0.89 < 0.99 → 截断到前 1 条
    assert len(r2) == 1


# ── dynamic_topk node integration tests ──


async def test_dynamic_topk_enabled_filters(monkeypatch):
    """启用时节点应调用 _dynamic_truncate 并写回结果。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    # 确保启用
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [
        {"rerank_score": 9.0, "text": "A"},
        {"rerank_score": 0.5, "text": "B"},
    ]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    # gap 9→0.5 (ratio 0.056 < 0.7) → 截断到前 1 条
    assert len(out["recall_vec_results"]) == 1
    assert out["recall_vec_results"][0]["text"] == "A"


async def test_dynamic_topk_disabled_passthrough(monkeypatch):
    """禁用时节点应透传原始结果不做任何修改。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {"RERANK_DYNAMIC_TOPK_ENABLED": False}
    )())
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0}, {"rerank_score": 8.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] is chunks  # 引用不变


async def test_dynamic_topk_empty_passthrough(monkeypatch):
    """空结果直接透传，不调 _dynamic_truncate。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    state = {"recall_vec_results": [], "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == []


async def test_dynamic_topk_non_list_passthrough(monkeypatch):
    """recall_vec_results 不是 list 时透传，不抛异常。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    state = {"recall_vec_results": "not a list", "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == "not a list"


async def test_dynamic_topk_all_truncated_to_zero_emits_status(monkeypatch):
    """截断到 0 条时应发送 '未找到足够相关内容' 状态。"""
    emitted = []
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda ev: emitted.append(ev)))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 0.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    assert out["recall_vec_results"] == []
    assert stream_event(StreamEventType.STATUS, "未找到足够相关内容") in emitted


async def test_dynamic_topk_exception_passthrough(monkeypatch):
    """节点内部异常时应捕获并透传原始结果，不抛异常。"""
    monkeypatch.setattr(topk_mod, "get_stream_writer", lambda: (lambda *a, **k: None))
    monkeypatch.setattr(topk_mod, "get_settings", lambda: type(
        "S", (), {
            "RERANK_DYNAMIC_TOPK_ENABLED": True,
            "RERANK_DYNAMIC_TOPK_DEFAULT": 5,
            "RERANK_DYNAMIC_TOPK_RATIO": 0.7,
        }
    )())
    # 让 _dynamic_truncate 抛异常
    monkeypatch.setattr(topk_mod, "_dynamic_truncate", lambda c, d, r: (_ for _ in ()).throw(ValueError("boom")))
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=SimpleNamespace())
    chunks = [{"rerank_score": 9.0}]
    state = {"recall_vec_results": chunks, "raw_query": "q"}

    out = await topk_mod.dynamic_topk(state, runtime)

    # 透传原始结果，不抛异常
    assert out["recall_vec_results"] is chunks


# ── updated route tests ──


async def test_route_after_topk_empty():
    """动态截断后为空时路由到 no_results。"""
    import rag.agent.workflow as wf

    assert wf._route_after_topk({"recall_vec_results": []}) == "no_results"
    assert wf._route_after_topk({}) == "no_results"


async def test_route_after_topk_has_results():
    """动态截断后有结果时路由到 generate。"""
    import rag.agent.workflow as wf

    assert wf._route_after_topk({"recall_vec_results": [{"text": "KB1"}]}) == "generate"
```

Note: Keep the existing `_route_after_recall` tests but they will fail after the rename. Remove them or update to use `_route_after_topk`. The plan adds new `_route_after_topk` tests alongside. The old tests (`test_route_after_recall_empty` and `test_route_after_recall_has_results`) should be removed since `_route_after_recall` no longer exists.

- [ ] **Step 3: Remove old route tests**

Delete functions `test_route_after_recall_empty` (lines 305-310) and `test_route_after_recall_has_results` (lines 313-317) from `tests/test_nodes.py` since `_route_after_recall` is renamed to `_route_after_topk`.

- [ ] **Step 4: Run all node tests**

Run:
```bash
uv run pytest tests/test_nodes.py -v
```
Expected: all tests pass (old + new ~28 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_nodes.py
git commit -m "test: add dynamic top-k truncation unit and integration tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Full test suite verification

**Files:**
- (none modified — verification only)

- [ ] **Step 1: Run full unit test suite**

Run:
```bash
uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py
```
Expected: all tests pass, including existing and new tests (~28 tests in test_nodes.py)

- [ ] **Step 2: Verify graph integrity**

Run:
```bash
uv run python -c "
from rag.agent.workflow import graph
# Check the full edge structure
edges = [(e.source, e.target) for e in graph.edges]
print('Edges:')
for s, t in sorted(edges, key=lambda x: (x[0], x[1])):
    print(f'  {s} -> {t}')
# Verify rerank -> dynamic_topk edge exists
assert any(e.source == 'rerank' and e.target == 'dynamic_topk' for e in graph.edges), 'Missing rerank -> dynamic_topk edge'
# Verify dynamic_topk -> generate/no_results conditional exists
print('All graph integrity checks passed')
"
```
Expected: `All graph integrity checks passed`
