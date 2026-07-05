# Chat 流式协议实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Chat 场景建立前后端共享的 SSE 流式协议 SDK，替代当前手写 SSE 字符串 + 事件类型漂移的现状

**Architecture:** 后端 `ChatStream`（async context manager + asyncio.Queue）封装 SSE 格式化和自动收尾；前端 `createChatStream`（AsyncIterable）封装 fetch + SSE 解析。事件模型（`status` / `message` / `error` / `done`）在 Pydantic 和 TypeScript 中同构定义

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, asyncio, TypeScript 5, React 19

## Global Constraints

- Python: 保留 `from __future__ import annotations` 和 `@override` 等条目的文件须维持其条目
- 提交粒度：每个任务独立提交，commit message 遵循 `feat:` / `refactor:` 前缀
- 测试：pytest + pytest-asyncio，测试文件放 `tests/`
- 前端：Vite + React + TypeScript，不使用 `any` 类型

---

## File Mapping

| 文件 | 职责 | 任务 |
|------|------|------|
| `rag/api/common/stream.py` | `ChatStream` 类 + Pydantic 事件模型 | 1 |
| `rag/api/modules/chat/service.py` | 消费 `ChatStream`，不再手写 SSE | 2 |
| `rag/agent/workflow.py` | 移除 `updates` stream mode | 2 |
| `rag/agent/nodes/generate/generate.py` | `token` → `message` 改名 | 2 |
| `frontend/src/types.ts` | `ChatEvent` 类型对齐 | 3 |
| `frontend/src/api/stream.ts` | 前端 `createChatStream` | 4 |
| `frontend/src/api/client.ts` | 移除 `streamChat` | 5 |
| `frontend/src/components/ChatBox.tsx` | 消费新 SDK + `status` 事件 | 5 |
| `tests/test_stream.py` | `ChatStream` 单元测试 | 1 |
| `tests/test_chat_controller.py` | 更新事件名匹配新协议 | 6 |

---

### Task 1: 后端 ChatStream SDK

**Files:**
- Create: `rag/api/common/stream.py`
- Create: `tests/test_stream.py`

**Interfaces:**
- Produces:
  - `ChatStatus(BaseModel)` — `type: Literal["status"]`, `data: str`
  - `ChatMessage(BaseModel)` — `type: Literal["message"]`, `data: str`
  - `ChatError(BaseModel)` — `type: Literal["error"]`, `data: str`
  - `ChatEvent = ChatStatus | ChatMessage | ChatError`
  - `class ChatStream` — constructor `()`, methods `status(text)`, `message(text)`, `error(text)`, `send_event(raw: dict)`, `__aenter__`, `__aexit__`, `__aiter__`

- [ ] **Step 1: 创建 `rag/api/common/stream.py`**

```python
"""Chat SSE 流式协议 SDK。

提供 ChatStream 类——async context manager + async iterable，
封装 SSE 格式化、事件校验、自动收尾。

生产者-消费者模式：生产者（业务代码 / LangGraph workflow）通过
status()/message()/error()/send_event() 入队；消费者（FastAPI
StreamingResponse）通过 __aiter__ 出队。调用 close() 发送 [DONE] 并
终止迭代。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


# ── 事件模型 ────────────────────────────────────────────


class ChatStatus(BaseModel):
    type: Literal["status"] = "status"
    data: str


class ChatMessage(BaseModel):
    type: Literal["message"] = "message"
    data: str


class ChatError(BaseModel):
    type: Literal["error"] = "error"
    data: str


ChatEvent = ChatStatus | ChatMessage | ChatError

# 事件类型 → Pydantic 模型映射
_EVENT_MODELS: dict[str, type[BaseModel]] = {
    "status": ChatStatus,
    "message": ChatMessage,
    "error": ChatError,
}


# ── JSON 序列化辅助 ──────────────────────────────────────


def _json_default(obj: object) -> str:
    """处理 json.dumps 无法序列化的常见类型（UUID、datetime 等）。"""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── ChatStream ──────────────────────────────────────────


class ChatStream:
    """Chat SSE 流式响应封装。

    直接使用（手动管理 + 后台生产者）:

        stream = ChatStream()

        async def _produce():
            try:
                async for event in invoke(...):
                    await stream.send_event(event)
            except Exception as e:
                await stream.error(str(e))
            finally:
                stream.close()

        asyncio.create_task(_produce())
        async for sse_line in stream:
            yield sse_line

    async with 语法（适合直接调用 status/message/error 的场景）:

        async with ChatStream() as stream:
            await stream.status("检索记忆中…")
            ...
        # __aexit__ 自动 close()
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False

    # ── 类型化发送方法 ─────────────────────────────────

    async def status(self, text: str) -> None:
        """发送阶段提示事件。"""
        event = ChatStatus(data=text)
        await self._enqueue(event)

    async def message(self, text: str) -> None:
        """发送逐字 token 事件。"""
        event = ChatMessage(data=text)
        await self._enqueue(event)

    async def error(self, text: str) -> None:
        """发送错误事件。"""
        event = ChatError(data=text)
        await self._enqueue(event)

    async def send_event(self, raw: dict) -> None:
        """适配层：将 LangGraph writer 产出的原始 dict 转为 SSE 事件入队。

        已知类型（status / message / error）通过 Pydantic 校验后序列化；
        未知类型静默跳过。
        """
        event_type = raw.get("type", "")
        model_cls = _EVENT_MODELS.get(event_type)
        if model_cls is None:
            return
        event = model_cls(**raw)
        await self._enqueue(event)

    # ── 流生命周期 ─────────────────────────────────────

    def close(self) -> None:
        """关闭流：入队 [DONE] + sentinel，终止 __aiter__。

        幂等：多次调用只生效一次。
        """
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait("data: [DONE]\n\n")
        self._queue.put_nowait(None)  # sentinel

    # ── 内部 ───────────────────────────────────────────

    async def _enqueue(self, event: ChatEvent) -> None:
        """将事件序列化为 SSE 行并入队。"""
        payload = json.dumps(
            event.model_dump(), ensure_ascii=False, default=_json_default
        )
        await self._queue.put(f"data: {payload}\n\n")

    # ── Async Context Manager ──────────────────────────

    async def __aenter__(self) -> "ChatStream":
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> None:
        if exc is not None:
            await self.error(str(exc))
        self.close()

    # ── Async Iterable（供 FastAPI StreamingResponse 消费）──

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item
```

- [ ] **Step 2: 创建 `tests/test_stream.py`**

```python
"""ChatStream 单元测试。"""

from __future__ import annotations

import json

import pytest

from rag.api.common.stream import ChatStream, ChatStatus, ChatMessage, ChatError


def _parse_sse(text: str) -> list[dict | str]:
    """将 SSE 文本解析为事件列表。"""
    result: list[dict | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            result.append("[DONE]")
        else:
            result.append(json.loads(payload))
    return result


# ── 事件模型校验 ─────────────────────────────────────────

def test_chat_status_model():
    event = ChatStatus(data="检索记忆中...")
    d = event.model_dump()
    assert d == {"type": "status", "data": "检索记忆中..."}


def test_chat_message_model():
    event = ChatMessage(data="关务")
    d = event.model_dump()
    assert d == {"type": "message", "data": "关务"}


def test_chat_error_model():
    event = ChatError(data="超时")
    d = event.model_dump()
    assert d == {"type": "error", "data": "超时"}


# ── ChatStream 正常流程 ─────────────────────────────────

@pytest.mark.asyncio
async def test_stream_status_and_message():
    async with ChatStream() as stream:
        await stream.status("检索中...")
        await stream.message("你好")

    # 消费 SSE 输出
    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == {"type": "status", "data": "检索中..."}
    assert events[1] == {"type": "message", "data": "你好"}
    assert events[2] == "[DONE]"


@pytest.mark.asyncio
async def test_stream_error_then_done():
    async with ChatStream() as stream:
        await stream.error("出错了")

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == {"type": "error", "data": "出错了"}
    assert events[1] == "[DONE]"


@pytest.mark.asyncio
async def test_stream_auto_error_on_exception():
    """__aexit__ 收到异常时自动发送 error + [DONE]。"""
    stream = ChatStream()
    try:
        async with stream:
            raise RuntimeError("llm down")
    except RuntimeError:
        pass  # 异常仍会传播

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == {"type": "error", "data": "llm down"}
    assert events[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_stream_empty():
    """空流只有 [DONE]。"""
    stream = ChatStream()
    async with stream:
        pass  # 什么都不发

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)
    assert events == ["[DONE]"]


# ── send_event 适配层 ────────────────────────────────────

@pytest.mark.asyncio
async def test_send_event_known_types():
    async with ChatStream() as stream:
        await stream.send_event({"type": "status", "data": "s1"})
        await stream.send_event({"type": "message", "data": "m1"})
        await stream.send_event({"type": "error", "data": "e1"})

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events[0] == {"type": "status", "data": "s1"}
    assert events[1] == {"type": "message", "data": "m1"}
    assert events[2] == {"type": "error", "data": "e1"}
    assert events[3] == "[DONE]"


@pytest.mark.asyncio
async def test_send_event_skips_unknown():
    """未知事件类型（如旧的 'update'）被静默跳过。"""
    async with ChatStream() as stream:
        await stream.send_event({"type": "update", "node": "x", "data": {}})
        await stream.send_event({"type": "status", "data": "ok"})

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert len(events) == 2
    assert events[0] == {"type": "status", "data": "ok"}
    assert events[1] == "[DONE]"


@pytest.mark.asyncio
async def test_send_event_missing_type():
    """无 type 字段的事件被跳过。"""
    async with ChatStream() as stream:
        await stream.send_event({"data": "no type"})
        await stream.send_event({"type": "status", "data": "ok"})

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)

    assert events == [{"type": "status", "data": "ok"}, "[DONE]"]


# ── done 不重复 ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_done_not_duplicated():
    """多次调用 error/status 后再正常退出，[DONE] 只发一次。"""
    async with ChatStream() as stream:
        await stream.status("s")
        await stream.status("s2")

    output = "".join([chunk async for chunk in stream])
    events = _parse_sse(output)
    done_count = sum(1 for e in events if e == "[DONE]")
    assert done_count == 1
```

- [ ] **Step 3: 运行测试验证失败（ChatStream 还不存在）**

```bash
pytest tests/test_stream.py -v
```

期望：部分测试因 import 失败，待下一步实现 ChatStream 后通过。

- [ ] **Step 4: 运行测试验证通过**

```bash
pytest tests/test_stream.py -v
```

期望：12 tests PASS

- [ ] **Step 5: 提交**

```bash
git add rag/api/common/stream.py tests/test_stream.py
git commit -m "feat: add ChatStream SDK with typed event models and SSE formatting"
```

---

### Task 2: 后端集成 — service.py / workflow.py / generate.py

**Files:**
- Modify: `rag/api/modules/chat/service.py`
- Modify: `rag/agent/workflow.py`
- Modify: `rag/agent/nodes/generate/generate.py`

**Interfaces:**
- Consumes: `ChatStream` from Task 1, `invoke` from `rag.agent.workflow`
- Produces: `stream_chat()` returns `AsyncIterator[str]` (unchanged signature)

- [ ] **Step 1: 修改 `rag/api/modules/chat/service.py`**

将现有的 `_sse()` / `_json_default()` / 手动 `try/except` / `[DONE]` 替换为 ChatStream 生产者-消费者模式。

```python
"""Chat 流式服务 —— 对接 agent workflow 与 ChatStream SDK。"""

import asyncio
from collections.abc import AsyncIterator

from rag.api.common.stream import ChatStream
from rag.agent.type import ContextSchema
from rag.agent.workflow import invoke
from rag.common.logging import get_logger
from rag.document.retriever import KnowledgeRetriever
from rag.memory import MemoryManager
from rag.models.base import ChatModel

logger = get_logger()


async def stream_chat(
    session_id: str,
    query: str,
    *,
    llm: ChatModel,
    memory_manager: MemoryManager,
    retriever: KnowledgeRetriever,
) -> AsyncIterator[str]:
    """把工作流事件通过 ChatStream 编码为 SSE 行下发。

    生产者-消费者模式：后台 task 将 LangGraph 事件喂入 ChatStream，
    主协程从 ChatStream 出队 SSE 行并 yield 给 StreamingResponse。
    """
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
```

- [ ] **Step 2: 修改 `rag/agent/workflow.py`**

从 `stream_mode` 移除 `"updates"`——协议不再包含 `update` 事件：

```python
from langgraph.graph import END, START, StateGraph

from rag.agent.nodes.generate.generate import generate
from rag.agent.nodes.query.query import handle_query
from rag.agent.nodes.recall.recall import recall
from rag.agent.nodes.recall_memory.memory import recall_memory
from rag.agent.type import MyState, ContextSchema

builder = StateGraph(MyState, context_schema=ContextSchema)

builder.add_node("recall_memory", recall_memory)
builder.add_node("handle_query", handle_query)
builder.add_node("recall", recall)
builder.add_node("generate", generate)

builder.add_edge(START, "recall_memory")
builder.add_edge("recall_memory", "handle_query")
builder.add_edge("handle_query", "recall")
builder.add_edge("recall", "generate")
builder.add_edge("generate", END)

graph = builder.compile()


async def invoke(session_id: str, query: str, context: ContextSchema):
    """归一化事件流:仅保留 custom 通道事件(status/message/error),
    updates 通道(state 增量)不再下发。
    """
    async for mode, chunk in graph.astream(
        {"session_id": session_id, "raw_query": query},
        context=context,
        stream_mode=["custom"],
    ):
        yield chunk
```

- [ ] **Step 3: 修改 `rag/agent/nodes/generate/generate.py`**

`{"type": "token"}` → `{"type": "message"}`：

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState

system_prompt = """
你是一个专业的关务助手。请基于上下文与改写后的查询,简洁回答用户问题。
"""


async def generate(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """最终生成:逐 token 流式输出,同时累积为完整 generated。"""
    writer = get_stream_writer()
    writer({"type": "status", "data": "生成回答中"})

    llm = runtime.context.llm
    chunks = state.get("recall_vec_results") or []
    knowledge = "\n".join(c.get("text", "") for c in chunks)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
查询: {state["rewrite_query"]}
记忆上下文: {state["context"]}
知识库内容: {knowledge}
"""
        ),
    ]

    parts: list[str] = []
    async for chunk in llm.astream(messages):
        token = getattr(chunk, "content", chunk)
        if not token:
            continue
        parts.append(token)
        writer({"type": "message", "data": token})

    state["generated"] = "".join(parts)
    return state
```

- [ ] **Step 4: 运行现有测试确认无回归**

```bash
pytest tests/test_chat_controller.py -v
```

期望：`test_chat_controller_streams_chain` 可能 FAIL（事件名变更），这是预期行为，Task 6 会修复。

```bash
pytest tests/test_workflow.py -v
```

期望：PASS（或按 workflow 测试实际情况判断）。

- [ ] **Step 5: 提交**

```bash
git add rag/api/modules/chat/service.py rag/agent/workflow.py rag/agent/nodes/generate/generate.py
git commit -m "refactor: integrate ChatStream SDK, remove updates channel, rename token to message"
```

---

### Task 3: 前端类型对齐

**Files:**
- Modify: `frontend/src/types.ts`

**Interfaces:**
- Produces: `ChatEvent` 联合类型（4 个变体）

- [ ] **Step 1: 更新 `frontend/src/types.ts` 中的 ChatEvent**

定位到 `// ── SSE 事件类型 ──────────────────────────────────────` 注释块，替换为：

```typescript
// ── SSE 事件类型 ──────────────────────────────────────

/** 后端 /chat SSE 下发的事件的联合类型 */
export type ChatEvent =
  | { type: "status";  data: string }
  | { type: "message"; data: string }
  | { type: "error";   data: string }
  | { type: "done";    data: null };
```

> 注意：旧类型中 `query` / `recall` / `generate` 全部移除，`done` 保留。

- [ ] **Step 2: 确认 TypeScript 编译无错误**

```bash
cd frontend && npx tsc --noEmit
```

期望：如果其他文件还在引用旧的 `ChatEvent` 成员（如 `ev.type === "generate"`），会有类型错误——这在 Task 5 中修复。

- [ ] **Step 3: 提交**

```bash
git add frontend/src/types.ts
git commit -m "refactor: align ChatEvent types with new streaming protocol"
```

---

### Task 4: 前端 ChatStream SDK

**Files:**
- Create: `frontend/src/api/stream.ts`

**Interfaces:**
- Produces: `function createChatStream(sessionId: string, query: string): ChatStream`
- Produces: `interface ChatStream { [Symbol.asyncIterator](): AsyncIterator<ChatEvent>; abort(): void }`

- [ ] **Step 1: 创建 `frontend/src/api/stream.ts`**

```typescript
import type { ChatEvent } from "@/types";

const BASE = "/api";

export interface ChatStream {
  [Symbol.asyncIterator](): AsyncIterator<ChatEvent>;
  abort(): void;
}

/**
 * 创建 Chat SSE 流——封装 fetch + ReadableStream 解析 + SSE 反序列化。
 *
 * 用法:
 *   for await (const ev of createChatStream(sessionId, query)) {
 *     switch (ev.type) {
 *       case "status":  /* 显示阶段提示 * /; break;
 *       case "message": /* 逐字追加 * /;     break;
 *       case "error":   /* 显示错误 * /;     break;
 *       case "done":    /* 标记结束 * /;     break;
 *     }
 *   }
 */
export function createChatStream(
  sessionId: string,
  query: string,
): ChatStream {
  const controller = new AbortController();

  const iterator = (async function* (): AsyncIterator<ChatEvent> {
    const res = await fetch(`${BASE}/chat/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, query }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? "chat request failed");
    }

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const payload = line.slice(6);

          if (payload === "[DONE]") {
            yield { type: "done", data: null };
            return;
          }

          try {
            yield JSON.parse(payload) as ChatEvent;
          } catch {
            // 解析失败的行静默跳过
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  })();

  return {
    [Symbol.asyncIterator]: () => iterator,
    abort: () => controller.abort(),
  };
}
```

- [ ] **Step 2: 确认 TypeScript 编译无错误**

```bash
cd frontend && npx tsc --noEmit
```

期望：仅有 `client.ts` 和 `ChatBox.tsx` 中引用旧 API 的类型错误（Task 5 修复），stream.ts 本身无错误。

- [ ] **Step 3: 提交**

```bash
git add frontend/src/api/stream.ts
git commit -m "feat: add frontend ChatStream SDK (createChatStream)"
```

---

### Task 5: 前端集成 — client.ts / ChatBox.tsx

**Files:**
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/ChatBox.tsx`

**Interfaces:**
- Consumes: `createChatStream` from `@/api/stream`
- Consumes: `ChatEvent` from `@/types`

- [ ] **Step 1: 修改 `frontend/src/api/client.ts` — 移除 `streamChat`**

删除 `streamChat` 函数（第 15-55 行）及其 import 中对 `ChatEvent` 的引用：

现有文件头部：
```typescript
import type {
  ChatEvent,         // ← 删除此行
  DocumentItem,
  DocumentListResponse,
  DocumentStatus,
  GraphRetryResult,
  RetryResult,
} from "@/types";
```

改为：
```typescript
import type {
  DocumentItem,
  DocumentListResponse,
  DocumentStatus,
  GraphRetryResult,
  RetryResult,
} from "@/types";
```

然后删除整个 `streamChat` 函数定义（第 14-55 行，含注释和空行）。

- [ ] **Step 2: 修改 `frontend/src/components/ChatBox.tsx`**

替换 import 和事件处理逻辑：

```typescript
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { createChatStream } from "@/api/stream";
import type { ChatEvent } from "@/types";

interface Message {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  status?: string;  // 当前阶段提示
}

export default function ChatBox({
  sessionId,
  className,
}: {
  sessionId: string;
  className?: string;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const q = input.trim();
    if (!q || sending) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setSending(true);

    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      for await (const ev of createChatStream(sessionId, q)) {
        setMessages((prev) => {
          const next = [...prev];
          const msg = next[assistantIdx];
          if (!msg) return prev;

          switch (ev.type) {
            case "status":
              next[assistantIdx] = { ...msg, status: ev.data };
              break;
            case "message":
              next[assistantIdx] = {
                ...msg,
                content: msg.content + ev.data,
              };
              break;
            case "error":
              next[assistantIdx] = {
                ...msg,
                content: msg.content + `\n\n> ⚠️ ${ev.data}`,
                isStreaming: false,
              };
              break;
            case "done":
              next[assistantIdx] = { ...msg, isStreaming: false };
              break;
          }
          return next;
        });

        if (ev.type === "done" || ev.type === "error") break;
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "unknown error";
      setMessages((prev) => {
        const next = [...prev];
        next[assistantIdx] = {
          role: "assistant",
          content: `> ❌ 请求失败: ${msg}`,
          isStreaming: false,
        };
        return next;
      });
    } finally {
      setSending(false);
    }
  };

  return (
    <div className={`flex flex-col h-full ${className ?? ""}`}>
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <p className="text-gray-500 text-sm">输入问题开始对话</p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-emerald-500/15 border border-emerald-500/20 text-gray-100"
                  : "bg-gray-800/80 text-gray-200 prose-chat"
              }`}
            >
              {m.role === "assistant" ? (
                <>
                  {m.status && (
                    <p className="text-xs text-gray-500 mb-1">{m.status}</p>
                  )}
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                  {m.isStreaming && (
                    <span className="inline-block w-2 h-4 bg-emerald-400 ml-0.5 animate-pulse align-text-bottom" />
                  )}
                </>
              ) : (
                <p>{m.content}</p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* 输入框 */}
      <div className="border-t border-gray-800 p-4">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            disabled={sending}
            placeholder="输入你的问题… (Enter 发送)"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 text-sm
                       placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50
                       disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={sending || !input.trim()}
            className="px-5 py-3 bg-emerald-500 text-gray-950 font-semibold text-sm rounded-xl
                       hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed
                       transition-colors"
          >
            {sending ? "…" : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 确认 TypeScript 编译无错误**

```bash
cd frontend && npx tsc --noEmit
```

期望：0 errors。

- [ ] **Step 4: 提交**

```bash
git add frontend/src/api/client.ts frontend/src/components/ChatBox.tsx
git commit -m "refactor: switch ChatBox to createChatStream, handle status events, remove old streamChat"
```

---

### Task 6: 更新后端测试

**Files:**
- Modify: `tests/test_chat_controller.py`

**Interfaces:**
- Consumes: 新的 `ChatEvent` 类型（`message` 替代 `token`，无 `update`）

- [ ] **Step 1: 修改 `tests/test_chat_controller.py`**

将 `_fake_invoke` 中的 `{"type": "token"}` 改为 `{"type": "message"}`，移除 `update` 事件，更新断言：

```python
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.chat.controller as controller_mod
import rag.api.modules.chat.service as service_mod
from rag.api.dependencies.agent import get_llm, get_memory_manager, get_retriever


async def _fake_invoke(session_id, query, context):
    yield {"type": "status", "data": "检索记忆中..."}
    yield {"type": "status", "data": "深度思考中"}
    yield {"type": "status", "data": "检索知识库中..."}
    yield {"type": "status", "data": "生成回答中"}
    yield {"type": "message", "data": "关务"}
    yield {"type": "message", "data": "信息"}


async def _boom_invoke(session_id, query, context):
    yield {"type": "status", "data": "检索记忆中..."}
    raise RuntimeError("llm down")


def _parse_sse(text: str) -> list[str]:
    return [
        line[len("data: "):]
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


def _client(monkeypatch, fake_invoke):
    monkeypatch.setattr(service_mod, "invoke", fake_invoke)
    app = FastAPI()
    app.include_router(controller_mod.chat_router)
    app.dependency_overrides[get_memory_manager] = lambda: object()
    app.dependency_overrides[get_llm] = lambda: object()
    app.dependency_overrides[get_retriever] = lambda: object()
    return TestClient(app)


def test_chat_controller_streams_chain(monkeypatch):
    client = _client(monkeypatch, _fake_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]

    # 不应包含 update 事件
    assert all(e["type"] in ("status", "message", "error") for e in events)

    # 验证事件顺序
    types = [e["type"] for e in events]
    assert types == ["status", "status", "status", "status", "message", "message"]

    assert events[4] == {"type": "message", "data": "关务"}
    assert events[5] == {"type": "message", "data": "信息"}


def test_chat_controller_emits_error_frame(monkeypatch):
    client = _client(monkeypatch, _boom_invoke)
    resp = client.post("/chat/", json={"session_id": "s1", "query": "q1"})

    assert resp.status_code == 200
    payloads = _parse_sse(resp.text)
    assert payloads[-1] == "[DONE]"
    events = [json.loads(p) for p in payloads if p != "[DONE]"]
    assert events[-1] == {"type": "error", "data": "llm down"}
```

- [ ] **Step 2: 运行测试验证**

```bash
pytest tests/test_chat_controller.py -v
```

期望：2 tests PASS。

- [ ] **Step 3: 运行全量测试确认无回归**

```bash
pytest tests/ -v
```

期望：所有已有测试 PASS（允许已有的 skip/xfail）。

- [ ] **Step 4: 提交**

```bash
git add tests/test_chat_controller.py
git commit -m "test: update chat controller tests for new event protocol (message, no update)"
```

---

## Task Order

```
Task 1 (后端 SDK) → Task 2 (后端集成)
                  → Task 6 (测试更新)

Task 1          → Task 3 (前端类型) → Task 4 (前端 SDK) → Task 5 (前端集成)
```

Task 1 是基础依赖。Task 2 和 Task 3 可并行（后端/前端各自独立）。Task 6 依赖 Task 2。
