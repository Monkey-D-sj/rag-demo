# 设计：检索可观测性（结构化日志 + Langfuse 追踪）

日期：2026-07-05
状态：已批准（用户对两层方案整体说「做吧」；部署方式与冲突规避为执行者依最稳妥原则代决，用户可随时修正）
分支：feat/async-foundation

## 背景与目标

检索链路（LangGraph `START → recall → generate`，`KnowledgeRetriever.search` = embed → pgvector
top-k）目前零日志：查询内容、召回结果、分数、耗时全部不可见。日志基建
（`rag/common/logging.py`）已有 JSON/文本格式器与轮转文件，但不支持结构化 extra 字段，
也没有请求关联 ID。

目标分两层：
1. **结构化日志**——排障与事实记录，不依赖外部服务；
2. **Langfuse 自托管追踪**——每次问答的链路级 trace（节点、LLM 调用、token、延迟）。

评测（黄金集 / RAGAS / LLM-as-judge）明确为非目标，攒到真实 trace 后单独立项。

## 并行会话约束（本设计的硬边界）

同分支存在并行会话，未提交改动覆盖 `rag/config.py`、`rag/agent/workflow.py`、
`rag/agent/nodes/generate/generate.py`。规避策略：
- Langfuse 配置**不进 `config.py`**，独立成 `rag/observability/langfuse.py` 自带
  `LangfuseSettings(BaseSettings)`；
- `workflow.py` 只做一处 ~3 行改动（可选 `callbacks` 参数），**放在最后执行**；
  执行前检查该文件是否仍有外部未提交改动，有则该文件不提交、单独向用户报告；
- 其余文件（`common/logging.py`、`document/retriever.py`、`agent/nodes/recall/recall.py`、
  `api/modules/chat/service.py`、`docker-compose.yaml`、新模块）当前干净，正常提交。
- 每次提交只 `git add` 本任务文件，绝不携带并行会话的改动。

## Layer 1：结构化日志

### 1.1 `rag/common/logging.py`

- **`JsonFormatter` 输出 extra 字段**：遍历 `record.__dict__`，跳过 LogRecord 标准属性
  集合，其余键并入输出 JSON；`json.dumps(..., default=str)` 兜底不可序列化值。
  文本格式器不输出 extra（开发终端不刷屏；生产 JSON 才带全量字段）。
- **session_id 关联**：模块级 `contextvars.ContextVar[str | None]`，提供
  `bind_session(session_id)` / `reset_session(token)` 两个函数；新增 logging `Filter`
  把当前值注入 `record.session_id`（未绑定时不注入）。JSON 格式带 `session_id` 字段；
  文本格式行尾追加 ` | session=<id>`（未绑定不追加）。Filter 挂在 `setup_logging`
  创建的两个 handler 上。

### 1.2 埋点

- **`rag/document/retriever.py::search`**：INFO 一条，extra 含 `kb_id`、`query`（截断
  200 字符）、`top_k`、`hits`（每项 chunk 标识 + score，具体字段名以
  `store.search_chunks` 返回列为准）、`embed_ms`、`search_ms`（`time.perf_counter`）。
  命中为空时改用 WARNING（检索质量最直接的信号）。
- **`rag/agent/nodes/recall/recall.py`**：INFO 记录本次使用 `rewrite_query` 还是
  `raw_query`（rewrite 节点当前被禁用，此日志持续暴露该事实）。
- **`rag/api/modules/chat/service.py::stream_chat`**：入口 `bind_session(session_id)`，
  `finally` 中 reset，保证请求内所有日志自动携带 session_id。

## Layer 2：Langfuse 自托管追踪

### 2.1 依赖与部署

- `pyproject.toml` 新增 `langfuse`（v3 SDK）。
- `docker-compose.yaml` 新增 `langfuse-web`、`langfuse-worker`、`clickhouse`、
  `langfuse-postgres` 四个服务。postgres 不复用现有实例：现有的是 paradedb 镜像且
  数据卷已初始化（init 脚本不会再执行，建第二个 database 需手工步骤，且 langfuse
  迁移对 paradedb 的兼容性未验证），专用容器更确定。redis / minio 复用现有实例
  （独立 DB index / 独立 bucket）。
- 用 Langfuse v3 headless init 环境变量（`LANGFUSE_INIT_ORG_ID`、
  `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`/`SECRET_KEY` 等）预置组织/项目/API key，
  使部署与验证全程无需人工点 UI，key 直接进 `.env`。

### 2.2 `rag/observability/langfuse.py`（新模块）

- `LangfuseSettings(BaseSettings)`：`LANGFUSE_ENABLED: bool = False`、
  `LANGFUSE_HOST`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`（均从 env/.env 读取）。
- `get_callback_handler() -> CallbackHandler | None` 工厂：
  `LANGFUSE_ENABLED=false` 时返回 None（零开销，langfuse 容器未启动不影响服务）；
  开启时返回 LangChain `CallbackHandler`。
  **实施修订（langfuse SDK v4 适配）**：v4 的 handler 构造不再接收凭据/session 参数，
  改为工厂内显式初始化全局客户端（进程级 lru_cache），session 关联经
  `session_scope`（`propagate_attributes`）与 astream config 的 metadata 传播；
  另有 `observe_root` 让 producer 成为 trace 根 span，使 CallbackHandler 与
  retriever 的 `@observe` span 嵌进同一条 trace。三个入口统一 `_tracing_active`
  判据（开关 + 双 key 齐备）。

### 2.3 接线

- `rag/agent/workflow.py::invoke`：新增可选参数 `callbacks=None`，透传
  `graph.astream(..., config={"callbacks": callbacks} if callbacks else None)`。
  （冲突敏感点，最后执行，见「并行会话约束」。）
- `rag/api/modules/chat/service.py::stream_chat`：调用工厂拿 handler，非 None 则传入
  `invoke`。
- `rag/document/retriever.py::search` 用 langfuse `@observe` 包装（自定义检索器不在
  LangChain 回调覆盖范围内），使 embed+pgvector 步骤出现在 trace 中；
  `LANGFUSE_ENABLED=false` / 客户端未初始化时装饰器必须退化为直通（测试验证）。

## 测试

- `JsonFormatter`：extra 字段并入输出、不可序列化值 `default=str` 兜底、标准属性不泄漏。
- session 关联：绑定后日志记录带 `session_id`，reset 后不带；文本格式行尾行为。
- `retriever.search`：fake pool/embedding + caplog 验证字段与空结果 WARNING。
- `get_callback_handler`：关闭返回 None；开启返回 handler（不触网的构造级验证）。
- `@observe` 直通：关闭状态下 search 行为与返回值不变。
- 真实验证（无法离线验证的点）：compose 启动 langfuse → 一次真实 chat →
  Langfuse UI 中可见含 recall span 与 LLM 调用的完整 trace。

## 非目标

- 黄金集构建、RAGAS 离线评测、LLM-as-judge——单独立项。
- 检索算法本身的改进（rerank、混合检索）。
- 其余模块（图抽取、入库 pipeline）的 trace 接入——本次只做检索问答链路。
