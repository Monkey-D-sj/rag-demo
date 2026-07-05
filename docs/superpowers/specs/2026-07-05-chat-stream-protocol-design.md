# Chat 流式协议设计

**日期**：2026-07-05  
**范围**：Chat 场景前后端 SSE 流式交互  
**协议层**：SSE（text/event-stream）  
**约束级别**：SDK 级封装，前后端各一轻量 SDK，共享事件模型定义

---

## 1. 事件模型

### 1.1 事件类型

| 事件 | 语义 | 方向 |
|------|------|------|
| `status` | 阶段提示（"检索记忆中…""生成回答中"） | server → client |
| `message` | 逐字 Token，LLM 流式生成内容 | server → client |
| `error` | 错误信息，流内异常通知 | server → client |
| `done` | 流正常结束 | SDK 合成，不在线路传输 |

`done` 由 SDK 在线路层将 `[DONE]` 标记转换而来，业务代码不手动发送。

### 1.2 TypeScript 类型

```typescript
type ChatEvent =
  | { type: "status";  data: string }
  | { type: "message"; data: string }
  | { type: "error";   data: string }
  | { type: "done";    data: null }
```

### 1.3 Python Pydantic 模型

```python
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
```

---

## 2. 线路格式

标准 SSE，不变：

```
data: {"type":"status","data":"检索记忆中..."}\n\n
data: {"type":"message","data":"关务"}\n\n
data: [DONE]\n\n
```

- 每条事件一行 `data:` + JSON 对象
- `\n\n`（双换行）分隔事件
- `[DONE]` 为流结束标记，由 SDK 自动发送 / 解析

---

## 3. 后端 SDK

位置：`rag/api/common/stream.py`

### 3.1 接口

```python
class ChatStream:
    """Chat SSE 流式响应封装，async context manager + async iterable"""

    async def status(self, text: str) -> None: ...
    async def message(self, text: str) -> None: ...
    async def error(self, text: str) -> None: ...

    async def __aenter__(self) -> "ChatStream": ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...
    async def __aiter__(self) -> AsyncIterator[str]: ...
```

### 3.2 行为约定

1. **自动收尾**：`__aexit__` 总是发送 `[DONE]`，无论是否抛异常
2. **异常兜底**：若 `__aexit__` 收到未处理异常，先发 `error` 事件再 `[DONE]`
3. **内部校验**：`message()` / `status()` / `error()` 用 Pydantic 构造事件，json.dumps 序列化异常（UUID、datetime 等）统一处理
4. **异步安全**：基于 `asyncio.Queue`，生产者（业务代码）和消费者（FastAPI StreamingResponse）解耦

### 3.3 与现有代码的关系

- `rag/api/modules/chat/service.py`：不再手写 `_sse()` / `try/except` / `[DONE]`，改用 `async with ChatStream() as stream: ...`
- `rag/api/modules/chat/controller.py`：不变，`StreamingResponse` 包裹 `ChatStream` 的 `__aiter__`
- `rag/agent/nodes/generate/generate.py`：`{"type": "token"}` 改为 `{"type": "message"}`

---

## 4. 前端 SDK

位置：`frontend/src/api/stream.ts`

### 4.1 接口

```typescript
function createChatStream(sessionId: string, query: string): ChatStream

interface ChatStream {
  [Symbol.asyncIterator](): AsyncIterator<ChatEvent>;
  abort(): void;
}
```

### 4.2 行为约定

1. 内部封装 `fetch` + `ReadableStream` 解析 + SSE 行反序列化
2. 将 `[DONE]` 转为 `{ type: "done", data: null }`
3. JSON 解析失败的行静默跳过
4. HTTP 非 2xx 抛 `Error`，外层 `try/catch` 兜底
5. `abort()` 用于用户主动停止生成

### 4.3 与现有代码的关系

- `frontend/src/api/client.ts`：移除 `streamChat`
- `frontend/src/types.ts`：`ChatEvent` 对齐新协议
- `frontend/src/components/ChatBox.tsx`：改用 `createChatStream`，新增 `status` 事件处理

---

## 5. 改动清单

| 文件 | 动作 | 内容 |
|------|------|------|
| `docs/superpowers/specs/2026-07-05-chat-stream-protocol-design.md` | 新增 | 本文件 |
| `rag/api/common/stream.py` | 新增 | `ChatStream` 类 |
| `rag/api/modules/chat/service.py` | 修改 | 改用 `async with ChatStream` |
| `rag/api/modules/chat/controller.py` | 不动 | — |
| `rag/agent/nodes/generate/generate.py` | 小改 | `token` → `message` |
| `frontend/src/api/stream.ts` | 新增 | `createChatStream` |
| `frontend/src/api/client.ts` | 修改 | 移除 `streamChat` |
| `frontend/src/types.ts` | 修改 | `ChatEvent` 对齐 |
| `frontend/src/components/ChatBox.tsx` | 修改 | 消费新 SDK + 处理 `status` |

---

## 6. 架构图

```
           ┌──────────────────┐
           │   protocol.md    │ ← 事件模型唯一真源
           └────────┬─────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼                             ▼
┌─────────────┐           ┌─────────────┐
│ 后端 SDK     │           │ 前端 SDK     │
│ stream.py   │           │ stream.ts   │
│ ChatStream  │           │ createChat  │
│             │           │ Stream()    │
└──────┬──────┘           └──────┬──────┘
       │                         │
┌──────▼──────┐           ┌──────▼──────┐
│ service.py  │           │ ChatBox.tsx │
└─────────────┘           └─────────────┘
```

---

## 7. 非目标

- 不做心跳 / 重连 / 事件 ID / 断点续传（Chat 场景不需要）
- 不做 OpenAPI / codegen 自动生成
- 不覆盖 Chat 以外的流式场景（文档处理进度等）
