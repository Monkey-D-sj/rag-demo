# 设计: 记忆元问题直达回答 + 彻底移除实体/图体系

日期: 2026-08-29
状态: 待审阅

## 1. 背景与目标

本次改动包含两个相互独立的变更，它们恰好都触碰 `query` 节点、`workflow.py`、`type.py` 与前端,故合并为一份设计:

**变更 A — 记忆元问题直达回答**: 当用户询问"关于之前对话本身"的问题(如"我第一次问了啥"、"我们之前聊了什么"、"总结一下我说的话"),当前会走完整的业务链路(query 改写 → 缓存 → RAG 检索 → 生成)。这些问题的答案只存在于对话记忆中,检索知识库毫无意义。目标: 在 query 改写阶段识别这类问题,命中后直接基于 `state["context"]`(由 `recall_memory` 节点填充)流式作答,不进入检索/缓存/生成链路。

**变更 B — 彻底移除实体/图体系**: 删除整个实体抽取与图谱召回子系统(查询侧实体抽取、文档侧实体抽取任务、Neo4j 存储/聚合/约束、图召回 leg、配置开关),保留数据库 `graph_status` / `graph_error` 列(用户已确认)但代码不再读写。

## 2. 变更 A: 记忆元问题直达回答

### 2.1 判定范围

`handle_query` 节点新增第三类判断 `is_memory_query`。判定为 True 的问题类型:

- 询问历史内容: "我第一次问了啥"、"我们刚才聊了什么"
- 询问自己说过的话: "我之前说过什么"、"我上次让你做什么了"
- 总结/回顾类: "总结一下我们聊了什么"、"帮我回顾一下对话"

明确不算(判定为 False):

- 询问记忆能力/身份: "你能记住我吗"、"你还记得我是谁吗" —— 这类没有可回答的历史事实,应走 out_of_scope → `direct_answer`
- 与知识库相关的正常提问

`is_memory_query` 与 `is_out_of_scope` 允许同时为 True(记忆元问题通常也与知识库无关)。路由时 `is_memory_query` 优先,互不矛盾。

### 2.2 数据流

```
START → recall_memory (填充 state["context"]) → handle_query (新增 is_memory_query)
    └─ is_memory_query=True → memory_answer ──────────────────────→ END
    └─ is_out_of_scope=True → direct_answer ─────────────────────→ END
    └─ 否则 → cache_lookup → ... 原业务链路
```

### 2.3 修改点

| 文件 | 改动 |
|------|------|
| `rag/prompts/query.py` | system_prompt 新增"任务五: 记忆元问题判断",输出 `is_memory_query`; 同步移除实体抽取任务(变更 B); 调整 out_of_scope 任务示例——原示例"总结一下我说的话"改为归入任务五,避免引导 LLM 将记忆元问题误标为纯 out_of_scope |
| `rag/agent/nodes/query/query.py` | `QueryRewriteOutput` 新增 `is_memory_query: bool` 字段; `handle_query` 写入 `state["is_memory_query"]`; 结构化输出失败时降级为 `False` |
| `rag/agent/type.py` | `MyState` 新增 `is_memory_query: bool`; 移除 `query_entities: list[str]`(变更 B) |
| `rag/prompts/generate.py` | 新增 `memory_system_prompt` 常量: 提示 LLM 基于对话历史作答,历史不包含所问内容时诚实说明未记录 |
| `rag/agent/nodes/generate/memory_answer.py` | 新增节点(见 2.4) |
| `rag/agent/workflow.py` | 注册 `memory_answer` 节点; `_route_after_query` 优先检查 `is_memory_query`; `invoke()` 初始化 `is_memory_query: False`; Send 分支移除 `entities`(变更 B) |

### 2.4 memory_answer 节点

- 读取 `state["context"]` 与 `state["raw_query"]`
- **context 为空(无任何记忆)**: 不调用 LLM,直接流式输出固定话术"我还没有你之前对话的记录。",写入 `state["generated"]` 并结束
- **context 非空**: 使用 `NormalModel` + `memory_system_prompt`,以 `[system + context, human(raw_query)]` 构造消息流式生成(与 `direct_answer` 同一套 SSE writer)
- **不写回记忆**(与 `direct_answer` 一致): 这是对既有历史的复述,无需再次入记忆
- 降级策略: LLM 流式失败沿用现有 generate 节点的异常路径(不引入新失败模式)

### 2.5 测试(变更 A)

- `tests/test_nodes.py` 或新增 `tests/test_memory_answer.py`: context 为空 → 固定话术且不调 LLM; context 非空 → 流式回答写入 `generated`; 不写回记忆
- `tests/test_workflow.py`: `is_memory_query=True` 路由到 `memory_answer`; 优先于 `is_out_of_scope`
- `tests/test_nodes.py` query node: `is_memory_query` 解析与结构化输出失败降级

## 3. 变更 B: 彻底移除实体/图体系

### 3.1 删除文件

```
rag/graph/__init__.py
rag/graph/aggregate.py
rag/graph/store.py
rag/graph/pipeline.py
rag/graph/retriever.py
rag/db/neo4j.py
rag/document/entity_extraction.py
rag/prompts/entity_extraction.py
tests/test_graph_retriever.py
tests/test_graph_pipeline.py
tests/test_entity_extraction.py
tests/test_graph_aggregate.py
tests/test_graph_store_pg.py
```

### 3.2 修改文件 — 后端

| 文件 | 改动 |
|------|------|
| `rag/config.py` | 移除 `NEO4J_ENABLED` / `ENABLE_ENTITY_EXTRACTION` / `GRAPH_EXTRACT_CONCURRENCY` / `GRAPH_RECALL_ENABLED` / `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` / `GRAPH_RECALL_TIMEOUT_SECONDS` 及其 validator |
| `rag/agent/type.py` | `MyState` 移除 `query_entities`; `RetrieverProtocol.search` 移除 `entities` 参数 |
| `rag/agent/nodes/query/query.py` | `QueryRewriteOutput` 移除 `entities` 字段; `handle_query` 不再写入 `query_entities` |
| `rag/agent/nodes/recall/recall.py` | `retriever.search` 调用移除 `entities=` 传参; 不再读取 `state["entities"]` |
| `rag/agent/workflow.py` | Send 分支 payload 移除 `entities`(只保留 `sub_query`) |
| `rag/document/retriever.py` | `KnowledgeRetriever` 移除 `graph_retriever` / `has_graph` / `_graph_leg` / `entities` 参数; `_merge_dedup` 由三路 RRF 降为向量+BM25 两路; 移除图召回 `asyncio.wait_for` 超时逻辑 |
| `rag/document/pipeline.py` | `_IngestDeps` 移除 `task_publisher`(仅用于实体抽取投递); 删除 `ENABLE_ENTITY_EXTRACTION` 分支与 `set_graph_status("skipped")` 调用 |
| `rag/document/store.py` | 删除 `get_chunks_for_graph` / `get_chunks_by_uids` / `claim_graph_processing` / `set_graph_status`; `list_documents` 的 SELECT 移除 `graph_status, graph_error` |
| `rag/api/main.py` | 移除 neo4j driver 与 `GraphRetriever` 的创建/注入, `app.state.neo4j` |
| `rag/api/modules/document/controller.py` | 移除 `/{document_id}/retry-graph` 路由 |
| `rag/api/modules/document/schemas.py` | 移除 `graph_status` / `graph_error` 字段与 `GraphRetryResponse` |
| `rag/api/modules/document/service.py` | 移除 `graph_status`/`graph_error` 字段映射与 `retry_graph()` |
| `rag/worker/main.py` | 移除 neo4j driver、`ensure_graph_constraints`、`_extract_wrapper`、`extract_document_entities` 任务分支、`_timeout_for` 的 900 秒分支、on_shutdown neo4j close |
| `rag/tasks.py` | `TaskName` / `_TASK_NAMES` 移除 `extract_document_entities` |
| `rag/prompts/__init__.py` | 移除 `entity_extract_prompt` / `human_extract_prompt` / `entity_types_guidance` 导出 |
| `rag/api/modules/health/health.py` | 移除 Neo4j 检查块, docstring 更新为"检查 pg / redis / minio / rabbitmq" |
| `rag/eval/harness.py` | `GoldenItem` 移除 `entities` 字段; `build_retriever` 移除 graph_retriever; 删除 `graph_fused` leg |
| `rag/eval/run.py` | 移除 `graph_fused` 的 leg 标签与 `_LEG_ORDER` / legs 过滤引用 |

### 3.3 修改文件 — 前端

| 文件 | 改动 |
|------|------|
| `frontend/src/api/client.ts` | 移除 `retryGraph` |
| `frontend/src/types.ts` | 移除 `graph_status` / `graph_error` 字段与 `GraphRetryResult` |
| `frontend/src/components/DocumentList.tsx` | 移除 `GRAPH_STATUS_MAP`、graph_error 显示、重试图谱按钮 |

### 3.4 部署配置

| 文件 | 改动 |
|------|------|
| `docker-compose.yaml` | 移除 `neo4j` 服务、`x-app-env` 中的 `NEO4J_*` 环境变量、api/worker 的 `depends_on: neo4j`、`neo4j_data` volume |

`.env` 经确认不含 NEO4J/GRAPH 配置,无需修改。

### 3.5 数据库

- `graph_status` / `graph_error` 列**保留**(用户已确认),数据不动
- 已有行的 `graph_status` 维持现值(如 `done`),不影响其他字段
- 新上传文档的 `graph_status` 将停在默认值 `pending`——代码与前端不再读取该列,无实际影响
- **不新增迁移**: 无需 ALTER,列留白即可

### 3.6 golden 数据集

`rag/eval/datasets/retrieval_golden.jsonl` 中部分条目携带的 `entities` 标注字段保留在文件里但不被读取(读取方 `GoldenItem` 已移除该字段)。不改动数据文件。

## 4. 修改/删除的测试

- 删除: `test_graph_retriever.py` / `test_graph_pipeline.py` / `test_entity_extraction.py` / `test_graph_aggregate.py`
- 修改:
  - `tests/test_worker.py`: 移除 `extract_document_entities` 相关用例(`test_timeout_for` 的 900 分支)
  - `tests/test_document_pipeline.py`: 移除 `set_graph_status` monkeypatch 与 `extract_document_entities` 投递断言
  - `tests/test_document_controller.py`: `get_document` 断言 fixture 移除 `graph_status` / `graph_error`
  - `tests/test_retriever.py`: 移除 entities 参数、`_FakeGraphRetriever` 与三路融合用例
  - `tests/test_eval_harness.py`: 移除 graph_fused leg 与 entities 标注断言
  - `tests/test_dependencies_agent.py`: 移除 graph_retriever 注入断言
  - `tests/test_chat_service_errors.py` / `tests/test_workflow.py` / `tests/test_nodes.py`: 伪 `search()` 签名移除 `entities` 参数(跟随协议)及 query node 的 entities 断言
- 新增(见 2.5)

## 5. 文档更新

- `CLAUDE.md`: 更新 13-node 流程图与节点表(新增 `memory_answer` 分支; 移除 `handle_query` 实体抽取、`recall` 图召回 leg、`extract_document_entities` 任务、hybrid retrieval 三路描述、eval graph_fused/decomposed leg 描述)
- 如 `README` / 其他文档提及图系统,一并清理

## 6. 迁移与兼容性

- 功能移除不涉及 API 契约破坏——`graph_status`/`graph_error` 从 API 响应中消失,前端同步移除引用;`retry-graph` 端点删除,前端不再调用
- 已入队的 `extract_document_entities` 消息: 单一工作队列 `rag.documents`(DLQ: `rag.documents.dlq`)。代码移除后,残留消息会被新 worker 消费、在 `decode_document_task` 因 `task_name not in _TASK_NAMES` 被拒绝,按既有重试策略最终进 DLQ(自清理)。如需立即清空,可用 RabbitMQ 管理面板 purge 或删除该队列;spec 中不自动执行
- `docker compose down` 不会删除 `neo4j_data` volume(命名卷,移除服务后成为孤儿卷),如用户希望清理需手动 `docker volume rm`——设计中不自动执行

## 7. 影响面汇总

- 删除文件: 13(后端 8 + 测试 5)
- 修改文件: 约 23(后端 15 + 测试 5 + 前端 3 + 部署 1)
- 新增文件: 2(`memory_answer.py` + 测试)
- 数据库: 无迁移
