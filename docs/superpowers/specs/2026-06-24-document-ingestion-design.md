# 文档入库管线 Design Spec

> Status: Approved (brainstorming) · Date: 2026-06-24

## 目标

为 `rag-demo` 增加**文档入库管线**:用户上传文件 → 存原件到 minio → 异步解析提取文本 → 切块 → 向量化 → 存进可检索的向量库(`document_chunks`)。这是 RAG 的「数据来源」环节,为后续的知识库检索(recall 节点)提供可检索数据。

**本次不含**:知识库检索/召回(另一个功能)、文档删除/重新入库、原件的鉴权下载、孤儿记录兜底、内容去重、死信队列(见「已知取舍」)。

## 选型决策(brainstorming 结论)

| 维度 | 决策 | 理由 |
|------|------|------|
| 范围 | 全链路含 minio | 上传必须有落地的对象存储 |
| 文件类型 | `.txt` / `.md` / `.pdf` | 覆盖 RAG demo 绝大多数场景;docx 留待以后 |
| 处理方式 | 异步 + 任务队列 + 独立 worker | 生产级:web 不阻塞,worker 横向扩展 |
| 队列 | **arq**(async 原生,broker 用 redis) | 项目全异步,worker 直接复用 async embedding / pg / minio,零阻抗;比 Celery 少 async/sync 桥接摩擦 |
| 知识库组织 | 按 `knowledge_base_id` 分组(collection) | 比全局共享更接近产品;比按 user 隔离更贴合「尚无 auth」的现状 |

## 架构与数据流

两个进程,redis 同时兼任 arq broker(队列用 db1,记忆用 db0,互不污染):

```
┌─────────────┐   POST /documents (multipart)        ┌──────────────┐
│  FastAPI    │ ─1─ 存原件 ──────────────────────────▶│    minio     │
│  (web 进程) │ ─2─ INSERT documents(status=pending) ─▶│  PostgreSQL  │
│             │ ─3─ enqueue ingest_document(doc_id) ──▶│  redis(db1)  │
│             │ ◀── 202 {document_id, status}         └──────────────┘
└─────────────┘
       ▲                                                     │
       │ GET /documents/{id} 轮询状态                         │ arq 取任务
       │                                                     ▼
┌──────┴───────────────────────────────────────────────────────────┐
│  arq worker (独立进程)  ingest_document(doc_id):                    │
│   置 processing → minio 取原件 → 解析文本 → 切块 → 分批 embed       │
│   → (同一事务) 删旧 chunk + 批量插新 chunk + status=done            │
│   失败: rollback + status=failed + error                           │
└────────────────────────────────────────────────────────────────────┘
```

**关键边界**:web 进程只做「快进快出」(校验 + 存原件 + 建记录 + 投递任务);所有重活在 worker。`pipeline.ingest_document` 是 web 与 worker 的**唯一汇合点** —— web 只 enqueue 这个函数名 + doc_id,worker 真正执行,两边不重复编排逻辑。web 用 `lifespan`、worker 用 arq `on_startup`/`on_shutdown` **各自**构造同一套 async 资源(pg pool / minio / embedding)。

## 生产级要点(纳入本次)

1. **幂等重跑**:arq 任务可能因超时/崩溃重试。`ingest_document` 开头 `DELETE FROM document_chunks WHERE document_id=?` 再整批重插,使任务可安全重放。配 arq `max_tries=3`、`job_timeout`(如 300s)。
2. **原子提交**:删旧 chunk + 插新 chunk + 置 `status=done` 在**同一事务**;中途异常整体 rollback 并置 `status=failed`,杜绝「done 但 chunk 不全」的脏数据。
3. **embedding 分批**:按 `embedding_batch_size`(默认 16)分批调用,避免单次过大超时;复用 `EmbeddingModel` 自带的 tenacity 重试。
4. **入口校验**:文件大小上限(`max_upload_mb`)、扩展名/MIME 白名单、文件名安全处理(防路径穿越)。

## 已知取舍(本次不做,后续再议)

- **孤儿记录兜底**:web 在「存 minio」与「投递任务」之间崩溃,会留下永久 `pending` 记录。生产做法是事务性 outbox 或定时 sweeper 重投。本次不做。
- **内容去重**:`documents.content_hash`(sha256)上传时填入,但**不建唯一约束、不写去重分支**;为以后去重预留数据。
- **死信队列**、**原件鉴权下载**:推迟。

## 数据库 Schema(新增 alembic `0002`)

知识库 chunk 不进 `long_term_memories`(那是记忆),新建 3 张表,镜像现有「pgvector + bm25 双索引」写法,为后续 recall 双路检索铺路。

### `knowledge_bases`
```
id          UUID PK DEFAULT gen_random_uuid()
name        TEXT NOT NULL
created_at  TIMESTAMPTZ DEFAULT now()
```
migration 插入一条**默认 kb**(固定 UUID),demo 缺省往里塞。

### `documents`
```
id                 UUID PK DEFAULT gen_random_uuid()
knowledge_base_id  UUID NOT NULL REFERENCES knowledge_bases(id)
filename           TEXT NOT NULL       -- 原始文件名(已安全处理)
content_type       TEXT NOT NULL       -- txt / md / pdf
size_bytes         BIGINT NOT NULL
content_hash       TEXT                -- sha256;本次仅填入不去重
object_key         TEXT NOT NULL       -- minio 中的 key
status             TEXT NOT NULL DEFAULT 'pending'  -- pending → processing → done | failed
error              TEXT                -- 失败原因(截断)
chunk_count        INT NOT NULL DEFAULT 0
created_at         TIMESTAMPTZ DEFAULT now()
updated_at         TIMESTAMPTZ DEFAULT now()
```
**状态机**:web 建记录写 `pending` → worker 接到先置 `processing` → 结束置 `done`(带 `chunk_count`)或 `failed`(带 `error`)。

### `document_chunks`
```
id                 UUID PK DEFAULT gen_random_uuid()
document_id        UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE
knowledge_base_id  UUID NOT NULL       -- 冗余;检索时直接按 kb 过滤
chunk_index        INT NOT NULL        -- 块在文档内的序号
text               TEXT NOT NULL
embedding          vector(1024)
metadata           JSONB DEFAULT '{}'
created_at         TIMESTAMPTZ DEFAULT now()
```
索引(与 `long_term_memories` 一致,便于 recall 复用):
```
hnsw (embedding vector_cosine_ops)
bm25 (id, text, metadata) WITH (key_field='id')
```

## 模块结构与职责

```
rag/
├── common/
│   └── minio_client.py        新增:create_minio_client + async put/get(to_thread 包裹)
├── document/
│   ├── __init__.py            导出 ingest_document / parse / chunk
│   ├── parser.py              新增:(bytes, content_type) -> str(纯函数,无 IO)
│   ├── chunker.py             新增:text -> list[str](纯函数,无 IO)
│   ├── store.py               新增:async,文档/chunk 的 DB 读写(SQL 收口于此)
│   └── pipeline.py            新增:ingest_document(doc_id) 编排(不写 SQL)
├── worker/
│   ├── __init__.py
│   └── main.py                新增:arq WorkerSettings(functions / on_startup / on_shutdown)
├── api/
│   ├── dependence/
│   │   └── storage.py         新增:get_minio / get_arq_pool 薄 Depends(读 app.state)
│   └── modules/document/
│       ├── __init__.py
│       └── controller.py      新增:POST /documents、GET /documents/{id}
└── config.py                  改:新增 minio_* / arq_redis_db / chunk_* / embedding_batch_size / max_upload_mb
```

**职责切分**:
- `parser` / `chunker`:纯函数、无 IO,最易测。
- `store`:所有 SQL 收口;`pipeline` 只编排不写 SQL(测试可 mock store)。
- `pipeline.ingest_document`:web 与 worker 唯一汇合点。
- minio client / arq pool:lifespan(web)与 on_startup(worker)各自构造,薄 Depends 读取(对齐现有 `dependence/db.py`)。

## 解析与切块

**`parser.py`** — `(data: bytes, content_type: str) -> str`:
- `txt` / `md`:`decode("utf-8")`,失败回退 `errors="replace"`(md 不渲染,纯文本入库)。
- `pdf`:**pypdf** 逐页 `extract_text()` 拼接;空页跳过;整篇为空 → 抛错(扫描版无文本层是真实失败)。

**`chunker.py`** — `(text: str) -> list[str]`,纯函数:
- **langchain `RecursiveCharacterTextSplitter`**(`\n\n → \n → 空格` 逐级切)。
- 默认 `chunk_size=800`、`chunk_overlap=100`,两者进 `Settings` 可调。
- 空文本 → `[]`,pipeline 据此判定无内容。

> 选字符而非 token 切分:避免引 tiktoken、且与 embedding tokenizer 未必一致;字符切分简单可预测好测。接口稳定,以后换 token 策略只改内部。

## Worker / minio / 配置 / docker-compose

**`worker/main.py`**(arq `WorkerSettings`):
- `functions = [ingest_document]`,`redis_settings` 复用 redis 的 `arq_redis_db=1`。
- `on_startup`:建 pg pool + minio client + `EmbeddingModel`,塞进 arq `ctx`;`on_shutdown` 关闭。
- `max_tries=3`、`job_timeout`(如 300s)。

**`common/minio_client.py`**:minio 官方 SDK 为同步,封装时用 `asyncio.to_thread()` 转线程池避免阻塞 event loop:
```
create_minio_client(settings) -> Minio        # 建客户端 + 确保 bucket 存在
async put_object(client, key, data, length, content_type)
async get_object(client, key) -> bytes
```
> 备选 aioboto3(原生 async / S3 协议);对 minio 单一用途,官方 SDK + to_thread 更直接、依赖更轻。

**`config.py` 新增**:
```
minio_endpoint / minio_access_key / minio_secret_key / minio_bucket / minio_secure
arq_redis_db: int = 1
chunk_size: int = 800
chunk_overlap: int = 100
embedding_batch_size: int = 16
max_upload_mb: int = 20
```

**`docker-compose.yaml`**:新增 minio 服务(API + console 端口、持久化 volume、默认 bucket)。

**新增依赖**:`arq`、`minio`、`pypdf`、`langchain-text-splitters`(若 splitter 不在已装 langchain 内)。

## API 契约

### `POST /documents`(multipart/form-data)
- 入参:`file`(必填)、`knowledge_base_id`(选填,缺省用默认 kb)。
- 流程:校验大小/类型 → 算 sha256 → 存 minio → INSERT `documents(status=pending)` → enqueue `ingest_document(doc_id)` → 返回。
- 响应 `202 Accepted`:`{document_id, status: "pending"}`。
- 校验失败 → `400`(类型不白名单 / 超过 `max_upload_mb`),不落库、不进 minio。

### `GET /documents/{id}`
- 响应:`{document_id, filename, status, chunk_count, error}`。
- 不存在 → `404`。

## 错误处理

- 入口校验失败:`400`,不落库、不进 minio。
- worker 内任何异常:整事务 rollback,`status=failed` + `error=str(e)`(截断),写日志(复用现有 logging 模块)。
- arq 重试 3 次仍失败:文档停在 `failed`,用户可见 error。
- PDF 无文本层 / 切块为空:视为 `failed`(如实暴露,不假装成功)。

## 测试策略(对齐现有三类)

- `parser` / `chunker`:**纯单测**,无 IO。覆盖 txt/md/pdf(小样本 PDF 字节)、空文本、超长文本的切块边界与 overlap。
- `store`:**integration**(`@pytest.mark.integration`,真 pg),验证建记录/改状态/批量插 chunk/幂等删重插。
- `pipeline.ingest_document`:**单测 + mock**(mock store / minio / embedding),验证编排顺序、失败置 `failed`、分批 embed。
- `controller`:`TestClient` + `dependency_overrides`(对齐现有 `test_chat_controller`),mock arq enqueue 与 minio,验证校验 / 落库 / 202 / 404。
- minio 客户端、arq worker 端到端:**integration**(需 docker compose 起 minio)。
