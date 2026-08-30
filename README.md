# RAG Demo

生产级 RAG（Retrieval-Augmented Generation）服务 —— 文档上传 → 智能解析 → 混合检索 → 流式对话，附带完整的评测体系与可观测性。

## 架构概览

```
用户提问 → LangGraph Agent 流水线 (13 节点 + 条件路由)
           ├─ recall_memory   → 短期(Redis) + 长期(pgvector)记忆召回
           ├─ handle_query    → 查询改写 + 范围判断(LLM 结构化输出)
           ├─ cache_lookup    → 语义缓存查询(命中跳过检索与生成)
           ├─ recall          → 向量(pgvector) + BM25(ParadeDB/jieba) + 图谱(Neo4j,可选) 三路并发检索,RRF 融合
           ├─ neighbor_expand → Sentence Window: 拉取同文档相邻 chunk 扩展上下文
           ├─ rerank          → Qwen3-Rerank 语义重排序
           ├─ dynamic_topk    → 相邻分差法动态 top-k 截断
           ├─ parent_expand   → Parent-Child: chunk 按 doc_id 展开为父文档全文
           ├─ generate        → LLM 流式生成(SSE)
           │   ├─ in-scope    → 基于检索上下文回答
           │   ├─ out-of-scope→ 直接大模型知识兜底
           │   └─ no_results  → 空召回兜底话术
           ├─ cache_store     → 答案回写语义缓存
           └─ add_memory      → 本轮问答持久化(短期+长期记忆)

文档入库：上传 → MinIO → ARQ Worker 异步
                          ├─ TXT/MD/PDF/DOCX 解析
                          ├─ 段落语义切块
                          ├─ Embedding → pgvector
                          └─ BM25 索引 (jieba 分词)
```

## 技术栈

| 层 | 技术 |
|---|---|
| **API 框架** | FastAPI + Uvicorn |
| **Agent 编排** | LangGraph（13 节点状态图 + 条件分支） |
| **向量 DB** | PostgreSQL + pgvector + ParadeDB（BM25 + jieba 分词） |
| **图数据库** | Neo4j GraphRAG（实体抽取入图 + 查询实体 1 跳扩展第三路召回,可选） |
| **重排序** | Qwen3-Rerank（DashScope API） |
| **缓存/短期记忆** | Redis |
| **对象存储** | MinIO |
| **任务队列** | ARQ（异步文档入库 + 自愈 cron） |
| **数据库迁移** | Alembic |
| **语义缓存** | 按 session 隔离的精确相似度命中 + 入库全局失效 + TTL(SEMANTIC_CACHE_*) |
| **LLM 治理** | 滑动窗口 RPM + 并发 ZSET 信号量 + Redis 三态熔断 + 成本统计 |
| **可观测性** | Langfuse（LLM 追踪）+ Loki + Grafana（日志聚合） |
| **前端** | React 18 + TypeScript + TailwindCSS + Vite |
| **容器化** | Docker Compose（9 个服务一体化部署） |
| **评测** | 自建 retrieval golden + 检索分轨 + 生成三指标 + 基线门禁 |

语义缓存只在同一 `session_id` 内做精确向量相似度匹配，默认保留 7 天；文档入库成功后会全局清空缓存，删除会话时由数据库外键级联删除该会话的缓存。缓存不可用时自动降级为正常检索与生成。

## 快速开始

### 前置条件

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/)（Python 包管理器）
- Docker + Docker Compose
- Node.js ≥ 18（前端开发）

### 1. 克隆并配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 LLM / Embedding / Rerank 的 API Key
```

### 2. 启动基础设施

```bash
docker compose up -d postgres redis minio neo4j
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
```

**Worker（文档异步入库）：**

```bash
uv run rag-worker
```

**评测（检索与真实生成质量回归检查）：**

```bash
# 首次运行需先建立基线
uv run rag-eval --update-baseline
# 日常回归检查
uv run rag-eval
# 含范围判断分类评测
uv run rag-eval --classify
# 使用同一 retrieval golden 的真实 LangGraph 生成评测（默认稳定抽样 15 条）
uv run rag-eval-answer
# 可恢复的两阶段运行
uv run rag-eval-answer --collect-only
uv run rag-eval-answer --resume <run-id>
# 检索 + serving contexts + faithfulness/relevance/citation 综合门禁
uv run rag-eval-suite
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
docker compose up -d --build
```

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
| **Document** | `POST /documents` | 上传文档（txt/md/pdf/docx） |
| | `GET /documents` | 按知识库列出文档状态 |
| | `POST /documents/{id}/retry` | 手动重试失败文档 |
| | `DELETE /documents/{id}` | 级联删除文档及其 chunks |
| **Files** | `GET /files/download/{id}` | 下载原始文件 |

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

- `status` — 节点状态推送（"检索知识库中…"）
- `message` — LLM 生成的 token 片段
- `error` — 错误信息
- `citations` — 引用元数据(生成完成或缓存命中后下发)

## Agent 流水线

```
                                     ┌─ context-only → context_answer → add_memory ────────────────────────────────────────────────────────────────────────────────────────────────┐
                                     ├─ out-of-scope → END(handle_query 内直接作答) ───────────────────────────────────────────────────────────────────────────────────────────────┤
START → recall_memory → handle_query ┤                                                                                                                                           END
                                     └─ in-scope → cache_lookup ┬─ 命中 → add_memory ───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                └─ 未命中 → recall → neighbor_expand → rerank → dynamic_topk → parent_expand ┬─ generate → cache_store → add_memory ─┘
                                                                                                                                          └─ no_results ──────────────────────────┘
```

| # | 节点 | 职责 |
|---|---|---|
| 1 | `recall_memory` | 短期记忆（Redis 最近 N 轮）+ 长期记忆（pgvector 语义搜索）并行召回 |
| 2 | `handle_query` | LLM 结构化输出：范围判断（闲聊/编程→直接兜底）+ 指代消解改写 + 会话上下文路由 |
| 3 | `cache_lookup` | 语义缓存查询(按 session 隔离的精确相似度匹配);命中直接下发缓存答案与引用,跳过检索与生成 |
| 4 | `recall` | 向量（pgvector cosine）+ BM25（ParadeDB jieba）+ 图谱(Neo4j 1 跳,条件启用) 三路并发检索,RRF 融合 |
| 5 | `neighbor_expand` | Sentence Window:拉取同文档相邻 chunk 扩展上下文 |
| 6 | `rerank` | Qwen3-Rerank 语义重排序（可选） |
| 7 | `dynamic_topk` | 相邻分差法动态 top-k 截断 |
| 8 | `parent_expand` | Parent-Child:chunk 按 doc_id 展开为父文档全文 |
| 9a | `generate` | 基于检索上下文 + LLM 流式生成，token 级 SSE 推送 |
| 9b | `context_answer` | 仅依据会话上下文回答 |
| 9c | `no_results` | 召回为空时返回兜底话术，不调用 LLM |
| 10 | `cache_store` | 生成完成后将答案与引用回写当前 session 的语义缓存,best-effort |
| 11 | `add_memory` | 本轮问答持久化写入短期（Redis）+ 长期（pgvector）记忆 |

## 记忆系统

| 类型 | 存储 | 特点 |
|---|---|---|
| **短期记忆** | Redis List | 最近 10 轮，24h TTL，FIFO |
| **长期记忆** | pgvector | 语义搜索，按 session 隔离，持久化 |

## 文档处理

**支持格式**：TXT / Markdown / PDF / DOCX

**切块策略**（通过 `SPLIT_STRATEGY` 配置）：

- `paragraph_semantic`（默认）— 段落级语义切分
- `recursive_character` — 递归字符分割
- `fixed_size` — 固定大小切块

**自愈机制**：Worker 每 5 分钟扫描 failed/stalled 文档，按指数退避自动重试；超 `MAX_RETRY_ROUNDS` 后标记为死信，可手动 API 重试。

## 检索评测

基于手工标注的 golden 数据集实现检索质量量化评测与回归门禁。

**评测指标**：hit@k / recall@k / NDCG@k / MRR

**8 路分轨**：混合检索/纯向量/纯 BM25 各取改写前与改写后（6 路），注入 reranker 时追加混合重排（fused_reranked）完整链路，一次跑完输出对比表格，可精确定位回归来源。图召回可用(Neo4j + GRAPH_RECALL_ENABLED)且条目带实体标注时,追加三路融合轨 graph_fused,为条件轨。

**门禁机制**：核心指标（recall@5、MRR）相对基线下降超过 3% 即阻断；改写质量独立门禁（改写不得降低检索质量）。

**CI 集成**：`.github/workflows/test.yml` 在 push/PR 时跑全量单测；`.github/workflows/eval.yml` 在检索相关代码变更时起 ParadeDB 容器、种子语料并执行评测门禁。

**范围判断评测**：`--classify` 启用 LLM 分类 vs golden 标注，计算 Precision/Recall/F1。

```bash
# 初始基线
uv run rag-eval --update-baseline

# 日常检查
uv run rag-eval

# 含范围判断评测
uv run rag-eval --classify
```

## 生成评测

`rag-eval-answer` 从 `retrieval_golden.jsonl` 稳定抽样，通过生产 LangGraph 完整流水线真实生成答案，再评估 `faithfulness`、`answer_relevance` 和 `citation_accuracy`。每次运行保存在 `rag/eval/history/<run-id>/`，包含 manifest、逐题答案/上下文、评分和报告；`--score-only` 或 `--resume` 可在采集与评分中断后继续。`rag-eval-suite` 额外对最终送入生成节点的 contexts 计算 serving retrieval，并按 ID 输出检索好但生成差的题目。

生成评测默认不写死 hard floor；首轮人工校准后使用显式 baseline 更新。模型、样本或 Judge prompt provenance 不一致时不会静默比较。

## 项目结构

```
rag-demo/
├── rag/
│   ├── __main__.py             # API 启动入口 (rag-api)
│   ├── config.py               # 配置管理 (pydantic-settings)
│   ├── agent/                  # LangGraph Agent
│   │   ├── workflow.py         # 状态图定义 (13 节点 + 条件路由)
│   │   ├── type.py             # 状态类型 & 协议定义
│   │   ├── memory/             # MemoryManager: 短期+长期记忆
│   │   └── nodes/
│   │       ├── recall_memory/  #   记忆召回
│   │       ├── query/          #   查询改写 + 范围判断
│   │       ├── cache_lookup/   #   语义缓存查询
│   │       ├── recall/         #   知识库检索
│   │       ├── neighbor_expand/#   Sentence Window 相邻 chunk 扩展
│   │       ├── rerank/         #   语义重排序
│   │       ├── dynamic_topk/   #   动态 top-k 截断
│   │       ├── parent_expand/  #   父文档全文展开
│   │       ├── generate/       #   LLM 生成 (3 节点: generate/direct/no_results)
│   │       ├── cache_store/    #   语义缓存回写
│   │       └── add_memory/     #   记忆持久化
│   ├── prompts/                # 统一提示词管理
│   │   ├── generate.py         #   RAG 生成 + 兜底回答
│   │   ├── query.py            #   查询分析
│   │   ├── entity_extraction.py#   实体抽取
│   │   ├── rerank.py           #   相关性打分
│   │   └── eval.py             #   评测数据生成
│   ├── api/                    # FastAPI 层
│   │   ├── main.py             #   应用工厂 & lifespan
│   │   ├── dependencies/       #   依赖注入
│   │   ├── common/             #   SSE 流、错误处理
│   │   └── modules/            #   chat / document / files / health
│   ├── worker/                 # ARQ Worker
│   │   └── main.py             #   任务函数 + cron 自愈
│   ├── document/               # 文档处理
│   │   ├── parser.py           #   TXT/MD/PDF/DOCX 解析
│   │   ├── chunker.py          #   多策略切块
│   │   ├── pipeline.py         #   入库流水线
│   │   ├── store.py            #   数据库 CRUD
│   │   ├── retriever.py        #   混合检索(向量+BM25+图→RRF 融合)
│   │   └── entity_extraction.py#   实体抽取
│   ├── eval/                   # 检索与生成评测
│   │   ├── run.py              #   CLI 入口 (rag-eval)
│   │   ├── harness.py          #   评测编排 (8 路分轨)
│   │   ├── answer_harness.py   #   真实 LangGraph 采集与三指标评分
│   │   ├── answer_metrics.py   #   答案预处理与引用指标
│   │   ├── artifacts.py        #   可恢复 history artifact
│   │   ├── baseline.py         #   v1/v2 baseline 与门禁
│   │   ├── suite_run.py        #   综合评测入口
│   │   ├── metrics.py          #   指标计算 + 门禁
│   │   ├── datasets/           #   golden 集
│   │   └── baseline.json       #   基线数据
│   ├── governance/             # LLM 调用治理(限流/熔断/超时/成本统计)
│   │   ├── config.py           #   治理配置
│   │   ├── guard.py            #   策略编排(熔断→限流→执行→上报)
│   │   ├── limiter.py          #   RPM 滑动窗口 + 并发 ZSET 信号量
│   │   ├── breaker.py          #   三态熔断器(closed/open/half-open)
│   │   └── usage.py            #   成本统计与用量记录
│   ├── graph/                  # 知识图谱 (可选)
│   ├── models/                 # LLM 模型封装
│   │   ├── base.py             #   抽象基类
│   │   ├── normal.py           #   Chat 模型 (tenacity 重试)
│   │   ├── embedding.py        #   Embedding 模型
│   │   └── rerank.py           #   Rerank 模型 (Qwen3-Rerank)
│   ├── db/                     # 数据库连接 (pg/redis/neo4j)
│   ├── common/                 # 公共工具
│   │   ├── logging.py          #   日志系统
│   │   ├── exception.py        #   异常体系
│   │   ├── platform.py         #   Windows asyncio 兼容
│   │   └── minio_client.py     #   MinIO 客户端
│   └── observability/          # 可观测性
│       └── langfuse.py         #   Langfuse 集成
├── frontend/                   # React 18 + TailwindCSS
├── alembic/                    # 数据库迁移
├── tests/                      # 测试
├── docker-compose.yaml         # 9 服务一体化部署
├── Dockerfile                  # 生产镜像
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
| `RETRIEVER_CANDIDATE_MULTIPLIER` | 每路候选倍数 | `2` |
| `RETRIEVER_VEC_SIMILARITY_THRESHOLD` | 向量路相似度阈值 | `0.5` |
| `RERANK_ENABLED` | 开启 Rerank | `true` |
| `RERANK_KEY` | Rerank API Key | — |
| `RERANK_BASE_URL` | Rerank API 地址 | — |
| `RERANK_MODEL` | Rerank 模型 | `qwen3-rerank` |
| `CHUNK_SIZE` | 切块大小 | `800` |
| `CHUNK_OVERLAP` | 块重叠 | `100` |
| `SPLIT_STRATEGY` | 切分策略 | `paragraph_semantic` |
| `ENABLE_ENTITY_EXTRACTION` | 开启实体抽取 | `false` |
| `NEO4J_ENABLED` | 开启 Neo4j | `false` |
| `GRAPH_RECALL_ENABLED` | 图召回第三路(需 Neo4j) | `false` |
| `LANGFUSE_ENABLED` | 开启 Langfuse 追踪 | `false` |
| `LOKI_ENABLED` | 推送日志到 Loki | `false` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `LOG_FORMAT` | 日志格式（text/json） | `text` |
| `GOVERNANCE_ENABLED` | 开启 LLM 调用治理（限流+熔断+统计） | `false` |
| `CHAT_RPM_LIMIT` | Chat RPM 上限 | `60` |
| `CHAT_MAX_CONCURRENCY` | Chat 最大并发数 | `8` |
| `BREAKER_FAILURE_THRESHOLD` | 熔断连续失败阈值 | `5` |
| `BREAKER_COOLDOWN_SECONDS` | 熔断冷却时长（秒） | `30` |
| `LLM_TIMEOUT_SECONDS` | LLM 调用超时（秒） | `60` |
| `LLM_PRICING` | 模型价格表（JSON, 每百万 token 元） | `{}` |
| `SEMANTIC_CACHE_ENABLED` | 开启语义缓存 | `false` |
| `SEMANTIC_CACHE_SIM_THRESHOLD` | 语义缓存命中相似度阈值 | `0.95` |
| `SEMANTIC_CACHE_TTL_HOURS` | 语义缓存有效期(小时) | `168` |

## 运行测试

```bash
# 单元测试（无需外部服务）
uv run pytest tests/ -v

# 集成测试（需要 Docker Compose 基础设施）
uv run pytest tests/ -v -m integration

# 检索评测回归（需要 pg + embedding + 已 seed 的评测 KB）
uv run pytest tests/ -v -m eval
```

## CLI 命令

| 命令 | 说明 |
|---|---|
| `rag-api` | 启动 API 服务 |
| `rag-worker` | 启动文档入库 Worker |
| `rag-eval` | 检索评测与门禁 |
| `rag-eval --update-baseline` | 刷新全部基线 |
| `rag-eval --classify` | 含范围判断 LLM 分类评测 |
