# RAG Demo

生产级 RAG（Retrieval-Augmented Generation）服务 —— 文档上传 → 智能解析 → 混合检索 → 流式对话，附带完整的可观测性与自愈能力。

## 架构概览

```
用户上传文档 → MinIO 存储 → ARQ Worker 异步入库
                                   ├─ PDF/文本解析
                                   ├─ 段落语义切块
                                   ├─ Embedding → pgvector
                                   └─ BM25 索引 (jieba 分词)

用户提问 → LangGraph Agent 流水线
           ├─ recall_memory  → 短期(Redis) + 长期(pgvector)记忆召回
           ├─ handle_query   → 查询改写
           ├─ recall         → 混合检索(pgvector + BM25, RRF 融合)
           └─ generate       → LLM 流式生成(SSE)
```

## 技术栈

| 层 | 技术 |
|---|---|
| **API 框架** | FastAPI + Uvicorn |
| **Agent 编排** | LangGraph（自定义状态图） |
| **向量数据库** | PostgreSQL + pgvector + ParadeDB（BM25 + jieba 分词） |
| **图数据库** | Neo4j（实体关系抽取） |
| **缓存/短期记忆** | Redis |
| **对象存储** | MinIO |
| **任务队列** | ARQ（异步文档入库 + 自愈 cron） |
| **数据库迁移** | Alembic |
| **可观测性** | Langfuse（LLM 追踪）+ Loki + Grafana（日志聚合） |
| **前端** | React 18 + TypeScript + TailwindCSS + Vite |
| **容器化** | Docker Compose（9 个服务一体化部署） |

## 快速开始

### 前置条件

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- Docker + Docker Compose
- Node.js ≥ 18（前端开发）

### 1. 克隆并配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key：
#   MODEL_KEY=sk-xxx
#   MODEL_NAME=qwen-plus
#   MODEL_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#   EMBEDDING_KEY=sk-xxx
#   EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2. 启动基础设施

```bash
docker compose up -d postgres redis minio neo4j
```

等待所有服务 healthy：

```bash
docker compose ps
```

### 3. 安装依赖 & 运行迁移

```bash
uv sync
uv run alembic upgrade head
```

### 4. 启动服务

**API 服务（开发模式，热重载）：**

```bash
RAG_RELOAD=1 uv run rag-api
# 或: python -m rag
```

**Worker（文档异步入库）：**

```bash
uv run rag-worker
```

API 默认监听 `http://localhost:8000`，Swagger 文档在 `http://localhost:8000/docs`。

### 5. 启动前端（可选）

```bash
cd frontend
pnpm install
pnpm dev
```

### 一键启动（Docker Compose 全栈）

```bash
# 先创建 .env 文件（参考 .env.example），然后：
docker compose up -d --build
```

这会启动所有服务：API、Worker、PostgreSQL、Redis、MinIO、Neo4j、Langfuse、Loki、Grafana。

| 服务 | 地址 |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| MinIO Console | http://localhost:9001 |
| Neo4j Browser | http://localhost:7474 |
| Langfuse | http://localhost:3000 |
| Grafana | http://localhost:3101 |

## API 模块

| 模块 | 端点 | 说明 |
|---|---|---|
| **Health** | `GET /health` | 健康检查 + 各依赖连通性 |
| **Chat** | `POST /chat/stream` | 流式对话（SSE），核心 RAG Agent 入口 |
| **Document** | `POST /documents` | 上传文档进入入库流水线 |
| | `GET /documents` | 按知识库列出文档状态 |
| | `POST /documents/{id}/retry` | 手动重试失败文档 |
| | `DELETE /documents/{id}` | 级联删除文档及其 chunks |
| **Files** | `GET /files/{id}` | 下载原始文件 |

### 对话请求示例

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "query": "孙悟空为什么要大闹天宫？",
    "knowledge_base_id": "my-kb"
  }' \
  --no-buffer
```

响应为 SSE（Server-Sent Events）流，事件类型：

- `status` — 节点状态推送（"正在召回记忆…"）
- `message` — LLM 生成的 token 片段
- `error` — 错误信息

## Agent 流水线

LangGraph 状态图包含 4 个节点，按序执行：

1. **recall_memory** — 从短期记忆（Redis，最近 N 轮对话）和长期记忆（pgvector 语义搜索）中召回相关历史
2. **handle_query** — 结合上下文改写/补全用户问题
3. **recall** — 混合检索：向量（pgvector cosine）+ 关键词（BM25 + jieba），RRF 融合排序
4. **generate** — 拼接系统提示 + 记忆 + 检索结果，调用 LLM 流式生成

每轮对话结束后，`generate` 节点自动将本轮交互写入短期和长期记忆。

## 记忆系统

| 类型 | 存储 | 特点 |
|---|---|---|
| **短期记忆** | Redis List | 最近 10 轮，24h TTL，先进先出 |
| **长期记忆** | PostgreSQL + pgvector | 语义搜索，按 session 隔离，持久化 |

## 文档处理流水线

```
上传 → 校验(类型/大小) → 存入 MinIO → ARQ 入队
                                              ↓
                              Worker: 下载 → 解析 → 切块 → Embedding
                                              ↓
                              写入 pgvector(向量) + BM25 索引(关键词)
                                              ↓
                              更新文档状态为 completed
```

**切块策略**（可通过 `SPLIT_STRATEGY` 配置）：

- `paragraph_semantic`（默认）— 段落级语义切分
- `recursive_character` — 递归字符分割
- `fixed_size` — 固定大小切块

**自愈机制**：

- Worker 每 5 分钟扫描 failed/stalled 文档，按指数退避自动重试
- 超过 `MAX_RETRY_ROUNDS`（默认 10）次重试后标记为真·死信，需手动 API 重试

## 项目结构

```
rag-demo/
├── rag/                        # Python 主包
│   ├── __main__.py             # API 启动入口 (rag-api)
│   ├── config.py               # 配置管理 (pydantic-settings)
│   ├── agent/                  # LangGraph Agent
│   │   ├── workflow.py         # 状态图定义(4 节点流水线)
│   │   ├── type.py             # 状态类型 & 协议定义
│   │   ├── memory/             # MemoryManager: 短期+长期记忆
│   │   └── nodes/              # 4 个图节点
│   │       ├── recall_memory/  #   记忆召回
│   │       ├── query/          #   查询改写
│   │       ├── recall/         #   知识库检索
│   │       └── generate/       #   LLM 生成
│   ├── api/                    # FastAPI 层
│   │   ├── main.py             # 应用工厂 & lifespan
│   │   ├── dependencies/       # 依赖注入
│   │   ├── common/             # SSE 流、错误处理、公共 schema
│   │   └── modules/            # 路由模块
│   │       ├── chat/           #   /chat/stream
│   │       ├── document/       #   /documents CRUD
│   │       ├── files/          #   /files 下载
│   │       └── health/         #   /health
│   ├── worker/                 # ARQ Worker
│   │   └── main.py             # 任务函数 + cron 自愈
│   ├── document/               # 文档处理
│   │   ├── parser.py           #   PDF/文本解析
│   │   ├── chunker.py          #   多策略切块
│   │   ├── pipeline.py         #   入库流水线
│   │   ├── store.py            #   数据库 CRUD
│   │   ├── retriever.py        #   混合检索(RRF)
│   │   └── entity_extraction.py#   实体抽取
│   ├── graph/                  # 知识图谱
│   │   ├── pipeline.py         #   图构建流水线
│   │   ├── store.py            #   Neo4j 操作
│   │   └── aggregate.py        #   实体聚合
│   ├── models/                 # LLM 模型封装
│   │   ├── base.py             #   抽象基类
│   │   ├── normal.py           #   Chat 模型
│   │   └── embedding.py        #   Embedding 模型
│   ├── db/                     # 数据库连接
│   │   ├── postgres.py         #   pg 连接池
│   │   ├── redis.py            #   redis 客户端
│   │   └── neo4j.py            #   neo4j 驱动
│   ├── common/                 # 公共工具
│   │   ├── logging.py          #   日志系统(console/file/Loki)
│   │   ├── exception.py        #   异常体系
│   │   └── minio_client.py     #   MinIO 客户端
│   └── observability/          # 可观测性
│       └── langfuse.py         #   Langfuse 集成
├── frontend/                   # React 前端
│   └── src/                    #   组件、页面、hooks
├── alembic/                    # 数据库迁移
├── tests/                      # 测试
├── docker-compose.yaml         # 9 服务一体化部署
├── Dockerfile                  # 生产镜像(API + Worker 共用)
└── pyproject.toml
```

## 配置参考

所有配置通过 `.env` 文件或环境变量设置，详见 `rag/config.py`。

| 变量 | 说明 | 默认值 |
|---|---|---|
| `MODEL_KEY` | LLM API Key | **必填** |
| `MODEL_NAME` | 模型名称 | **必填** |
| `MODEL_URL` | LLM API 地址 | **必填** |
| `EMBEDDING_KEY` | Embedding API Key | **必填** |
| `EMBEDDING_URL` | Embedding API 地址 | **必填** |
| `EMBEDDING_MODEL` | Embedding 模型名 | `text-embedding-v4` |
| `EMBEDDING_DIM` | 向量维度 | `1024` |
| `CHUNK_SIZE` | 切块大小 | `800` |
| `CHUNK_OVERLAP` | 块重叠 | `100` |
| `SPLIT_STRATEGY` | 切分策略 | `paragraph_semantic` |
| `ENABLE_ENTITY_EXTRACTION` | 开启实体抽取 | `false` |
| `LANGFUSE_ENABLED` | 开启 Langfuse 追踪 | `false` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FORMAT` | 日志格式（text/json） | `text` |
| `LOKI_ENABLED` | 推送日志到 Loki | `false` |

## 运行测试

```bash
# 单元测试（无需外部服务）
uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py

# 集成测试（需要 Docker Compose 基础设施）
uv run pytest tests/ -v -m integration
```
