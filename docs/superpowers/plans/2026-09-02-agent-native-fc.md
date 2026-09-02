# Agent 分支(原生 Function Calling)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:executing-plans / subagent-driven-development 逐任务实现。任务用 checkbox(`- [ ]`)跟踪。

**Goal:** 在确定性 RAG 主链路旁新增一条被 handle_query 路由的 Agent 子链,用**原生 function calling**(`bind_tools` + `tool_calls` + `ToolMessage`)做有界多步工具检索,把攒到的上下文喂给现有 generate,输出保持引用规范;覆盖单次检索无法解决的"交叉引用/多步"问题。

**Architecture:** 不改 13 节点确定性主链。`handle_query` 结构化输出新增 `needs_agent` 判定 → 命中且 `AGENT_MODE_ENABLED` 时路由进新节点 `agent_execute`(单节点内做有界工具循环,暂不拆子图)。循环用模型层新增的 `ainvoke_with_tools` 发原生工具调用,工具复用生产 `retriever`/`memory_manager`;结果行写入 `recall_vec_results`,复用 `generate → cache_store → add_memory` 尾链(agent 答案跳过 cache_store 防污染)。超步/异常/空结果降级回 `cache_lookup` 正常召回链。

**Tech Stack:** Python / LangGraph / LangChain `ChatOpenAI.bind_tools`(deepseek,`extra_body` 已全局关 thinking,原生 FC 已验证可用)/ PostgreSQL + pgvector。

**Spec:** 设计要点并入本计划「Global Constraints」,暂不另起 design doc;如需可后续拆分 `specs/2026-09-02-agent-native-fc-design.md`。

## Global Constraints(设计定案)

- **原生 FC 已验证**:`deepseek-v4-flash` + `extra_body={"thinking":{"type":"disabled"}}` 下第一轮返回合法 `tool_calls`、喂回 `ToolMessage` 第二轮收敛(`885260e` 已提交)。因此**不设独立 AGENT_MODEL 通道**,agent 步进与主链路同用 `NormalModel`。
- **insertion point 唯一**:只改 `handle_query` 之后的 `_route_after_query` 与新增节点;主链路其他边不动。
- **复用 generate**:agent 只负责"攒对上下文",最终答案必须由现有 `generate` 产出(读 `state["recall_vec_results"]` 行 `{text, filename, metadata}`,自建 citations,citations index 从 1 连续)。agent 不得自己写 `generated`。
- **工具是薄的**:直接调 `runtime.context.retriever`(协议 `search(query, kb_ids=None, top_k=5)` 与 `fetch_parent_contents(document_ids)`)与 `memory_manager.search(session_id, query, ...)`;`search` 不传 query_emb 时内部自算 embedding(与 `recall` 节点一致)。
- **有界 + 降级**:循环轮数 ≤ `AGENT_MAX_STEPS`,整节点包 `asyncio.timeout(AGENT_TOTAL_TIMEOUT_SECONDS)`。异常/超时/空结果一律不向用户抛错:空上下文 → 路由回 `cache_lookup` 走正常召回。
- **缓存隔离**:agent 分支不回读语义缓存(命中判定交给路由前的主链语义),成功路径置 `agent_skip_cache=True`,`cache_store` 据此跳过回写,避免 agent 答案污染全局缓存。
- **LLM 治理自动生效**:每轮工具调用走 `ainvoke_with_tools`,内部沿用 `_track`/`_acquire`/`_translate`/timeout 包裹(限流/熔断/成本按真实请求记账),无需新增 guard 接线。
- **streaming**:每步 `writer(STATUS "第 N 步:正在…")`;最终答案由 generate 流式下发,协议不变。
- 新增开关默认全关;路由字段默认 False;结构化输出失败/短路命中一律钳为 False(走主链)。
- 配置命名:`AGENT_MODE_ENABLED` / `AGENT_MAX_STEPS` / `AGENT_TOTAL_TIMEOUT_SECONDS`,追加到 Settings 尾部,不进 `_REQUIRED_FIELDS`。

---

### Task 1:模型层原生工具调用通道 `ainvoke_with_tools`

**Files:**
- Modify: `rag/models/base.py`(ChatModel 抽象)
- Modify: `rag/models/normal.py`(NormalModel 实现)
- Test: `tests/test_normal.py`

**Interfaces:**
- Produces:
  - `ChatModel.ainvoke_with_tools(messages: list[Any], tools: list[Any]) -> Any`(抽象)
  - `NormalModel.ainvoke_with_tools(messages, tools) -> BaseMessage`:返回**整条消息**(读取 `.tool_calls`),不是 `content` 字符串;工具调用轮复用 ainvoke 的重试/guard/超时包裹,`_track("chat_tool")` 计费。

- [ ] **Step 1:写失败测试**(`tests/test_normal.py`)

用 stub 替换 `NormalModel._model`(`ChatOpenAI` 的 duck-typed 替代,`ainvoke(messages)` 返回带 `tool_calls` 的 `AIMessage`),断言:
- 返回对象带 `.tool_calls`(断言 `tool_calls[0]["name"]` 透传);
- 传入的 `tools` 被 `bind_tools` 接收(断言 stub 记录到一次 bind,且是**每次调用独立 bind**,不污染后续)。
  测试通过 `monkeypatch` 注入,不触网。

- [ ] **Step 2:跑测试确认失败**(`uv run pytest tests/test_normal.py -v`,报 "attribute not found")

- [ ] **Step 3:实现**

`rag/models/base.py`:
```python
    @abstractmethod
    async def ainvoke_with_tools(self, messages: list[Any], tools: list[Any]) -> Any:
        """原生 function calling:返回含 tool_calls 的完整消息(实现返回 BaseMessage)。"""
```

`rag/models/normal.py`(结构照 `ainvoke`,注意**每轮 bind 新 runnable**,别改 `self._model`):
```python
    async def ainvoke_with_tools(
        self, messages: list[BaseMessage | str], tools: list[BaseTool],
    ) -> BaseMessage:
        """带重试的原生工具调用:返回完整消息供读取 .tool_calls。"""
        async with self._track("chat_tool") as tracker:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10, jitter=1),
                retry=retry_if_exception(is_retryable),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    if tracker is not None:
                        tracker.attempts += 1
                    async with self._acquire("chat"):
                        async with self._translate():
                            async with asyncio.timeout(self._timeout):
                                rsp = await self._model.bind_tools(tools).ainvoke(messages)
                            if tracker is not None:
                                tracker.set_tokens(*_usage_from(rsp))
                            return rsp
            raise AssertionError("unreachable")
```

- [ ] **Step 4:跑测试确认通过**

- [ ] **Step 5:Commit**
```bash
git add rag/models/base.py rag/models/normal.py tests/test_normal.py
git commit -m "feat(models): 新增原生 function calling 调用通道 ainvoke_with_tools"
```

---

### Task 2:配置开关

**Files:**
- Modify: `rag/config.py`(Settings 类尾部追加)
- Test: `tests/test_governance_config.py` 或现有 config 相关测试文件

- [ ] **Step 1:加字段**(仿 `QUERY_DECOMPOSITION_ENABLED` 的书写位置与风格):
```python
    AGENT_MODE_ENABLED: bool = False
    AGENT_MAX_STEPS: int = Field(default=3, ge=1)
    AGENT_TOTAL_TIMEOUT_SECONDS: int = Field(default=30, ge=1)
```

- [ ] **Step 2:测试默认值**(`get_settings()` 构造后断言默认 False/3/30;确认不在必填集合)

- [ ] **Step 3:Commit**
```bash
git add rag/config.py tests/<config 测试文件>
git commit -m "feat(config): agent 开关与有界步数/超时配置"
```

---

### Task 3:状态字段

**Files:**
- Modify: `rag/agent/type.py`(MyState)
- Modify: `rag/agent/workflow.py`(`build_initial_state`)
- Test: `tests/test_workflow.py`

- [ ] **Step 1:MyState 加字段**
```python
    needs_agent: bool  # handle_query 判定为需多步工具检索时 True,路由进 agent 分支
    agent_skip_cache: bool  # agent 成功产出后置 True,cache_store 据此跳过回写
```

- [ ] **Step 2:`build_initial_state` 返回里补**
```python
        "needs_agent": False,
        "agent_skip_cache": False,
```

- [ ] **Step 3:测试**:初始 state 含两字段且 False(`tests/test_workflow.py`)

- [ ] **Step 4:Commit**
```bash
git add rag/agent/type.py rag/agent/workflow.py tests/test_workflow.py
git commit -m "feat(agent): 状态增加 needs_agent 与 agent_skip_cache"
```

---

### Task 4:handle_query 判定 + prompt 任务五

**Files:**
- Modify: `rag/agent/nodes/query/query.py`
- Modify: `rag/prompts/query.py`
- Modify: `rag/agent/workflow.py`(`build_initial_state` 不动;此任务只动 query 节点)
- Test: `tests/test_nodes.py`

- [ ] **Step 1:QueryRewriteOutput 加字段**(query.py)
```python
    needs_agent: bool = Field(
        default=False,
        description=(
            "问题是否必须先用工具检索到某文档、再依其结果去查另一处(如法规'参照 X 执行'、"
            "先归类再定许可、跨版本对照),单次检索无法覆盖;is_out_of_scope 或 answer_from_context "
            "命中时必须为 false"
        ),
    )
```

- [ ] **Step 2:prompt 加"任务五:多步/交叉引用判定"**(`rag/prompts/query.py`),放在任务三/四之后,并改"任务优先级与短路"段:任务一或任务二命中时 `needs_agent=false`;两者均未命中时模型据问题给出布尔值。给 1~2 个正/反例(反例:普通单跳查询 needs_agent=false)。

- [ ] **Step 3:handle_query 落 state**(query.py,结构化成功后)
```python
        state["needs_agent"] = bool(
            not result.is_out_of_scope
            and not state.get("answer_from_context", False)
            and getattr(result, "needs_agent", False)
        )
```
异常降级分支补 `state["needs_agent"] = False`。

- [ ] **Step 4:测试**(`tests/test_nodes.py`):短路命中(out_of_scope / answer_from_context)时 `needs_agent=False`(即使 LLM 输出 true);普通结构化输出透传;异常降级为 False。

- [ ] **Step 5:Commit**
```bash
git add rag/agent/nodes/query/query.py rag/prompts/query.py tests/test_nodes.py
git commit -m "feat(query): 结构化输出增加 needs_agent 多步判定"
```

---

### Task 5:Agent 工具模块

**Files:**
- Create: `rag/agent/nodes/agent/__init__.py`
- Create: `rag/agent/nodes/agent/tools.py`
- Test: `tests/test_agent_tools.py`(新文件)

**Interfaces:**
- Produces:
  - `build_tools(context: ContextSchema, session_id: str) -> list[BaseTool]`:闭包绑定运行时上下文,供 bind_tools。
  - 每工具 handler 返回 `list[dict]`(行形如 `{text, filename, metadata, document_id}`),**同时**提供给人看的 observation 文本。
  - `tools_by_name(context, session_id) -> dict[str, Callable]`,供 executor 直接拿结构化行(不解析 ToolMessage)。

- [ ] **Step 1:写实现**(tools.py,核心逻辑)
```python
from langchain_core.tools import BaseTool, tool
from rag.agent.type import ContextSchema

def _render(rows: list[dict]) -> str:
    out = []
    for i, r in enumerate(rows, 1):
        title = r.get("filename", "未知文档")
        text = (r.get("text") or "")[:400]
        out.append(f"[{i}] 《{title}》\n{text}")
    return "\n\n".join(out) if out else "未检索到相关内容。"

def build_tools(context: ContextSchema, session_id: str) -> list[BaseTool]:
    retriever = context.retriever
    memory = context.memory_manager

    @tool
    async def retrieve_kb(query: str, top_k: int = 5) -> str:
        """在知识库中按语义检索与 query 最相关的片段。query 需自包含、无指代。"""
        if retriever is None:
            return "检索服务不可用"
        rows = await retriever.search(query, top_k=min(max(top_k, 1), 10))
        return _render(rows)

    @tool
    async def fetch_document(document_id: str) -> str:
        """按文档 ID 获取该文档全文(法规条款需看原文时用)。返回全文或为空。"""
        if retriever is None:
            return "检索服务不可用"
        contents = await retriever.fetch_parent_contents([document_id])
        return contents.get(document_id, "")

    @tool
    async def search_memory(query: str) -> str:
        """在当前会话记忆中检索与本轮对话相关的历史内容。"""
        if memory is None:
            return ""
        rows = await memory.search(session_id, query, top_k=3)
        return _render([{"text": r.get("text", ""), "filename": "会话记忆", "metadata": {}} for r in rows])

    return [retrieve_kb, fetch_document, search_memory]
```
> 说明:executor 需要结构化行以喂 `recall_vec_results`,此处工具把行**渲染成文本**只给模型读;executor 侧用**同一 handler 的原生调用**(不解析 ToolMessage),所以把每个工具的底层 handler 单独提为纯函数 `_kb_search/_doc_fetch/_mem_search(context, session_id, **args) -> list[dict]`,工具与 executor 都调它(避免两处逻辑漂移)。`_kb_search` 的行需补 `filename/metadata/document_id`,来源为 `search` 返回行自带字段(`text/document_id/chunk_index/metadata/filename/...`)。

- [ ] **Step 2:测试**(`tests/test_agent_tools.py`,用 stub retriever/memory,不触网):
  - `build_tools` 返回 3 个工具,name 正确;
  - `_kb_search` 行含 `text/filename/metadata`;
  - 参数校验非法 → 抛 `ValidationError`(供 executor 转 ToolMessage);
  - `search_memory` 在 memory=None 时返回空行。

- [ ] **Step 3:Commit**
```bash
git add rag/agent/nodes/agent/ tests/test_agent_tools.py
git commit -m "feat(agent): 检索/取全文/记忆三工具,闭包绑定运行时"
```

---

### Task 6:agent 执行节点(有界原生工具循环)

**Files:**
- Create: `rag/agent/nodes/agent/agent.py`
- Create: `rag/prompts/agent.py`(循环 system prompt)
- Test: `tests/test_agent_loop.py`(新文件,stub llm,不触网)

**Interfaces:**
- Produces: `agent_execute(state: MyState, runtime: Runtime[ContextSchema]) -> MyState`。返回的 state:`recall_vec_results` 已写(可能为空)、`agent_skip_cache=True`、`needs_agent` 保持。

- [ ] **Step 1:prompt**(`rag/prompts/agent.py`)
  说明:系统里列出可用工具;指导模型"逐步调用工具直到证据足够,证据不足就换检索词;不要臆造,不要编引用编号;信息够了就**不再调用工具**,直接说一句'可以作答了'"。强调工具返回的编号片段不是最终引用依据,最终回答由后续生成器统一组织。

- [ ] **Step 2:实现节点**(agent.py 骨架,异常/超步兜底)
```python
import asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.nodes.agent.tools import _kb_search, _doc_fetch, _mem_search, _render, build_tools
from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings
from rag.prompts.agent import system_prompt

logger = get_logger()
_IMPL = {"retrieve_kb": _kb_search, "fetch_document": _doc_fetch, "search_memory": _mem_search}


async def agent_execute(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """有界原生工具循环:把工具捞到的上下文攒进 recall_vec_results,交给 generate。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "深度检索中…"))
    settings = get_settings()
    llm = runtime.context.llm
    session_id = state["session_id"]
    max_steps = settings.AGENT_MAX_STEPS
    collected: list[dict] = []
    messages: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["raw_query"]),
    ]
    try:
        async with asyncio.timeout(settings.AGENT_TOTAL_TIMEOUT_SECONDS):
            for step in range(1, max_steps + 1):
                writer(stream_event(StreamEventType.STATUS, f"第 {step}/{max_steps} 步"))
                rsp = await llm.ainvoke_with_tools(messages, build_tools(runtime.context, session_id))
                calls = getattr(rsp, "tool_calls", None) or []
                if not calls:
                    break
                messages.append(AIMessage(content=rsp.content, tool_calls=rsp.tool_calls))
                for c in calls:
                    impl = _IMPL.get(c.get("name"))
                    try:
                        rows = await impl(runtime.context, session_id, **(c.get("args") or {}))
                    except Exception as e:  # noqa: BLE001 - 参数/执行失败作为 observation 回喂,让模型自纠
                        rows = []
                        obs = f"工具 {c.get('name')} 调用失败: {type(e).__name__}: {e}"
                    else:
                        collected.extend(rows)
                        obs = _render(rows)
                    messages.append(ToolMessage(content=obs, tool_call_id=c.get("id")))
                if step == max_steps:
                    logger.warning("agent 达到步数上限，使用已收集上下文", extra={"query": state["raw_query"][:200]})
    except Exception:  # noqa: BLE001 - 超时/异常一律降级,不向用户抛错
        logger.warning("agent 循环失败，降级使用已收集上下文", exc_info=True)

    state["recall_vec_results"] = _dedupe(collected)
    state["agent_skip_cache"] = True
    return state
```
`_dedupe(rows)`:按 `(document_id, chunk_index)` 去重(无 id 时按 `text`),保留先到;doc 全文类行 `chunk_index` 缺失则按 `document_id` 去重。行需保证 `text` 非空、`filename/metadata` 存在(缺省补默认)。**节点内不写 `generated`**。

- [ ] **Step 3:测试**(`tests/test_agent_loop.py`,stub `runtime.context.llm` 的 `ainvoke_with_tools` 依次返回「带 tool_call → 带 tool_call → 无 tool_call」):
  - 两轮工具后收敛,`collected` 行数正确;
  - 达到 `AGENT_MAX_STEPS` 仍有 tool_call → 不抛错,用已收集行;
  - `ainvoke_with_tools` 抛异常 → 不抛错,`recall_vec_results` 为空(或已有行);
  - 非法参数 → 收到"调用失败"observation,循环能继续;
  - 结束时 `agent_skip_cache=True`、`generated` 未被写入。

- [ ] **Step 4:Commit**
```bash
git add rag/agent/nodes/agent/agent.py rag/prompts/agent.py tests/test_agent_loop.py
git commit -m "feat(agent): 有界原生工具循环节点 agent_execute"
```

---

### Task 7:工作流接线 + 缓存守卫

**Files:**
- Modify: `rag/agent/workflow.py`
- Modify: `rag/agent/nodes/cache_store/store.py`
- Test: `tests/test_workflow.py`

- [ ] **Step 1:`_route_after_query` 增加 agent 分支**(workflow.py)
```python
def _route_after_query(state: MyState) -> str:
    if state.get("answer_from_context"):
        return "context_answer"
    if state.get("is_out_of_scope"):
        return "end"
    if get_settings().AGENT_MODE_ENABLED and state.get("needs_agent"):
        return "agent"
    return "recall"
```
注意 `get_settings()` 已在模块导入;条件边 mapping 增 `"agent": "agent_execute"`。

- [ ] **Step 2:agent_execute 之后的降级路由**(新增 `_route_after_agent`)
```python
def _route_after_agent(state: MyState) -> str:
    return "generate" if state.get("recall_vec_results") else "cache_lookup"
```
接线:`agent_execute` 条件边 → `{"generate": "generate", "cache_lookup": "cache_lookup"}`。cache_lookup 对 agent 空结果再查一次缓存(miss)→ 走现有 Send 召回链,**复用现成 degrade**,不改 cache_lookup。

- [ ] **Step 3:cache_store 守卫**(`rag/agent/nodes/cache_store/store.py`,在 `cache = ...` 之前)
```python
    if state.get("agent_skip_cache"):
        return state
```

- [ ] **Step 4:测试**(`tests/test_workflow.py`):AGENT_MODE_ENABLED=true 且 needs_agent → 走到 agent_execute;false → 仍走 cache_lookup;agent 有结果 → generate、空 → cache_lookup;agent_skip_cache=True 时 cache_store 透传不改 state。

- [ ] **Step 5:Commit**
```bash
git add rag/agent/workflow.py rag/agent/nodes/cache_store/store.py tests/test_workflow.py
git commit -m "feat(workflow): agent 分支路由与缓存回写守卫"
```

---

### Task 8:端到端冒烟(integration)

**Files:**
- Test: `tests/test_agent_e2e.py`(标 `@pytest.mark.integration`,默认跳过)

- [ ] **Step 1:写冒烟测试**:真实 `graph.astream`(走 `build_initial_state`),注入真实依赖(复用 `answer_harness.build_real_collection_context`,semantic_cache=None、memory 可 None),发 1~2 条"参照 X 执行"式问题,断言:`recall_vec_results` 非空、`generated` 非空、`citations` index 连续、custom 流里出现 STATUS 与 citations 事件。

- [ ] **Step 2:跑通**:`uv run pytest tests/test_agent_e2e.py -v -m integration`(需 pg + embedding + 已 seed 的含交叉引用文档的 eval KB)。

- [ ] **Step 3:人工冒烟**:`RAG_RELOAD=1 uv run rag-api`,问一条交叉引用问题,肉眼确认 STATUS 逐步推进 + 引用卡片正确。

- [ ] **Step 4:Commit**
```bash
git add tests/test_agent_e2e.py
git commit -m "test(agent): 端到端冒烟覆盖工具循环与引用连续性"
```

---

## 不做(scope 边界)

- 不做 LangGraph 子图(单节点内循环够 v1;将来要"图上可见的环"再拆)。
- 不做语义缓存读写(agent 分支整体绕开,理由见 Global Constraints)。
- 不新增独立 AGENT_MODEL 通道(原生 FC 已验证可用)。
- **不在此计划内造交叉引用评测集**:正式 golden 题(两份互相引用的 seed 文档 + 3~5 条 `category="cross_ref"` 题,做 AGENT on/off 的 X→Y 对照)依赖 seed 语料基建,属数据工作,单列后续计划。
- 不动 `handle_query` 的任务一~四语义与现有 classify/route-precedence 评测契约(仅新增第五任务布尔,短路契约保持)。

## 后续(单独计划)

1. 交叉引用评测集:seed 两份互相引用文档 → `retrieval_golden.jsonl` 加 3~5 条 cross_ref → 分别 `AGENT_MODE_ENABLED=true/false` 跑 `rag-eval-answer --ids …`,产出简历用的答对率 X→Y 与延迟增量。
2. (可选)thinking 档位化:快模型跑路由/单跳,推理档只给难题,用评测定档(见对话记录,不在本计划)。

## Self-Review 记录

- Spec 覆盖:需求全落在 Task1~8;评测集与 thinking 档位明确划出范围。
- 类型一致性:`ainvoke_with_tools -> BaseMessage`(读 `.tool_calls`)在 Task1 定义、Task6 消费;`agent_execute/_route_after_agent` 在 Task6/7 命名一致;`build_tools`/`_kb_search` 等在 Task5 定义、Task6 消费;`agent_skip_cache` Task3 定义、Task6 置位、Task7 消费。无悬空引用。
