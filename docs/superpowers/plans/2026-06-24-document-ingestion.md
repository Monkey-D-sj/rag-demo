# 文档入库管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现文档入库管线 —— 上传文件 → 存 minio → arq 异步解析(txt/md/pdf)→ 切块 → embed → 存进 `document_chunks` 向量库。

**Architecture:** web 进程(FastAPI)只做「快进快出」:校验 + 存原件到 minio + 建 `documents(pending)` 记录 + 投递 arq 任务;独立 arq worker 进程执行 `ingest_document`,完成解析/切块/分批 embed/原子入库。redis 兼任 arq broker(db1,与记忆 db0 隔离)。`pipeline.ingest_document` 是 web 与 worker 的唯一汇合点。

**Tech Stack:** Python 3.12、FastAPI、arq(async 任务队列)、minio(对象存储)、pypdf、langchain-text-splitters、psycopg3(AsyncConnectionPool + pgvector)、pytest + pytest-asyncio。

## Global Constraints

- Python `>=3.12`。
- pytest `asyncio_mode = "auto"`:`async def test_...` 无需 `@pytest.mark.asyncio`。
- 需真实 pg/redis/minio 的测试必须标 `@pytest.mark.integration`。
- 默认跑非 integration 用例:`uv run pytest -m "not integration"`。
- Windows 事件循环策略已在 `tests/conftest.py` 处理,勿改。
- 薄 Depends 风格须对齐现有 `rag/api/dependence/db.py`(一资源一读取函数)。
- embedding 维度固定 `1024`(`EMBEDDING_DIM`),与 `0001` 迁移一致。
- 默认知识库 UUID:`00000000-0000-0000-0000-000000000001`(迁移与代码两处必须一致)。
- chunk 入库 SQL 须沿用 `long_term_memories` 的 pgvector + bm25 双索引写法。
- 频繁提交:每个 Task 末尾一次提交。

---

### Task 1: 依赖、配置、docker-compose(基础设施)

**Files:**
- Modify: `pyproject.toml`(由 `uv add` 自动改)
- Modify: `rag/config.py`
- Modify: `docker-compose.yaml`
- Test: `tests/test_config.py`(追加)

**Interfaces:**
- Produces: `Settings` 新增字段 `minio_endpoint/minio_access_key/minio_secret_key/minio_bucket/minio_secure`、`arq_redis_db`、`chunk_size`、`chunk_overlap`、`embedding_batch_size`、`max_upload_mb`,供后续所有 Task 消费。

- [ ] **Step 1: 安装依赖**

Run: `uv add arq minio pypdf langchain-text-splitters`
Expected: `pyproject.toml` 的 `dependencies` 多出这 4 个包,`uv.lock` 更新。

- [ ] **Step 2: 写失败测试**

向 `tests/test_config.py` 追加:

```python
def test_settings_has_document_ingestion_defaults():
    s = Settings()
    assert s.minio_endpoint == "localhost:9000"
    assert s.minio_bucket == "rag-documents"
    assert s.minio_secure is False
    assert s.arq_redis_db == 1
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.embedding_batch_size == 16
    assert s.max_upload_mb == 20
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py::test_settings_has_document_ingestion_defaults -v`
Expected: FAIL(`AttributeError: 'Settings' object has no attribute 'minio_endpoint'`)

- [ ] **Step 4: 写实现**

在 `rag/config.py` 的 `Settings` 类中,`# ── Logging ──` 块之前插入:

```python
    # ── MinIO ──
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    # ── arq / 文档入库 ──
    arq_redis_db: int = 1
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_batch_size: int = 16
    max_upload_mb: int = 20

```

在 `docker-compose.yaml` 的 `services:` 下新增 minio 服务(放在 `postgres` 之后):

```yaml
  minio:
    image: minio/minio:latest
    container_name: rag-minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    restart: unless-stopped
```

并在 `volumes:` 块追加一行 `minio_data:`:

```yaml
volumes:
  redis_data:
  pg_data:
  minio_data:
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS(全部 config 用例通过)

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml uv.lock rag/config.py docker-compose.yaml tests/test_config.py
git commit -m "feat: 入库管线依赖/配置/minio 服务"
```

---

### Task 2: 切块器 `chunker.py`(纯函数)

**Files:**
- Create: `rag/document/chunker.py`
- Test: `tests/test_document_chunker.py`(新建)

**Interfaces:**
- Consumes: `langchain_text_splitters.RecursiveCharacterTextSplitter`。
- Produces: `chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_document_chunker.py`:

```python
from rag.document.chunker import chunk


def test_chunk_blank_returns_empty():
    assert chunk("   \n  ") == []


def test_chunk_short_text_single_chunk():
    assert chunk("hello world", 800, 100) == ["hello world"]


def test_chunk_long_text_splits_into_multiple():
    text = "段落。" * 1000
    out = chunk(text, 100, 20)
    assert len(out) > 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_chunker.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.document.chunker`)

- [ ] **Step 3: 写实现**

新建 `rag/document/chunker.py`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """把文本切成块。空白文本返回空列表。"""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_chunker.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/document/chunker.py tests/test_document_chunker.py
git commit -m "feat: 文档切块器 chunker(递归字符切分)"
```

---

### Task 3: 解析器 `parser.py`(纯函数)

**Files:**
- Create: `rag/document/parser.py`
- Test: `tests/test_document_parser.py`(新建)

**Interfaces:**
- Consumes: `pypdf.PdfReader`(模块级导入,便于测试 monkeypatch)。
- Produces: `parse(data: bytes, content_type: str) -> str`;不支持类型或 PDF 无文本 → 抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_document_parser.py`:

```python
import pytest

import rag.document.parser as parser_mod


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakeReader:
    def __init__(self, stream):
        self.pages = [_FakePage("第一页"), _FakePage(""), _FakePage("第三页")]


class _EmptyReader:
    def __init__(self, stream):
        self.pages = [_FakePage(""), _FakePage("   ")]


def test_parse_txt_decodes_utf8():
    assert parser_mod.parse("你好".encode("utf-8"), "txt") == "你好"


def test_parse_md_decodes_utf8():
    assert parser_mod.parse(b"# title", "md") == "# title"


def test_parse_unknown_type_raises():
    with pytest.raises(ValueError):
        parser_mod.parse(b"x", "exe")


def test_parse_pdf_joins_nonempty_pages(monkeypatch):
    monkeypatch.setattr(parser_mod, "PdfReader", _FakeReader)
    out = parser_mod.parse(b"%PDF-fake", "pdf")
    assert "第一页" in out
    assert "第三页" in out


def test_parse_pdf_empty_raises(monkeypatch):
    monkeypatch.setattr(parser_mod, "PdfReader", _EmptyReader)
    with pytest.raises(ValueError):
        parser_mod.parse(b"%PDF-empty", "pdf")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_parser.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.document.parser`)

- [ ] **Step 3: 写实现**

新建 `rag/document/parser.py`:

```python
import io

from pypdf import PdfReader


def parse(data: bytes, content_type: str) -> str:
    """按类型把文件字节解析为纯文本。"""
    if content_type in ("txt", "md"):
        return data.decode("utf-8", errors="replace")

    if content_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        text = "\n".join(parts)
        if not text.strip():
            raise ValueError("PDF 无可提取文本(可能是扫描件)")
        return text

    raise ValueError(f"不支持的文件类型: {content_type}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_parser.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/document/parser.py tests/test_document_parser.py
git commit -m "feat: 文档解析器 parser(txt/md/pdf)"
```

---

### Task 4: minio 客户端 `minio_client.py`

**Files:**
- Create: `rag/common/minio_client.py`
- Test: `tests/test_minio_client.py`(新建)

**Interfaces:**
- Consumes: `minio.Minio`(模块级导入,便于 monkeypatch)、`Settings`。
- Produces: `create_minio_client(settings) -> Minio`(确保 bucket 存在)、`async put_object(client, bucket, key, data: bytes, content_type: str) -> None`、`async get_object(client, bucket, key) -> bytes`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_minio_client.py`:

```python
import rag.common.minio_client as mc
from rag.config import Settings


class _FakeMinio:
    def __init__(self):
        self.exists = False
        self.made = []

    def bucket_exists(self, bucket):
        return self.exists

    def make_bucket(self, bucket):
        self.made.append(bucket)


def test_create_minio_makes_bucket_when_missing(monkeypatch):
    fake = _FakeMinio()
    monkeypatch.setattr(mc, "Minio", lambda *a, **k: fake)
    client = mc.create_minio_client(Settings())
    assert client is fake
    assert fake.made == ["rag-documents"]


def test_create_minio_skips_make_when_exists(monkeypatch):
    fake = _FakeMinio()
    fake.exists = True
    monkeypatch.setattr(mc, "Minio", lambda *a, **k: fake)
    mc.create_minio_client(Settings())
    assert fake.made == []


async def test_put_object_passes_bytes_and_length():
    calls = []

    class _C:
        def put_object(self, bucket, key, stream, length, content_type):
            calls.append((bucket, key, stream.read(), length, content_type))

    await mc.put_object(_C(), "bk", "k1", b"hello", "text/plain")
    assert calls == [("bk", "k1", b"hello", 5, "text/plain")]


async def test_get_object_reads_and_closes():
    closed = {"closed": False, "released": False}

    class _Resp:
        def read(self):
            return b"data"

        def close(self):
            closed["closed"] = True

        def release_conn(self):
            closed["released"] = True

    class _C:
        def get_object(self, bucket, key):
            return _Resp()

    out = await mc.get_object(_C(), "bk", "k1")
    assert out == b"data"
    assert closed == {"closed": True, "released": True}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_minio_client.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.common.minio_client`)

- [ ] **Step 3: 写实现**

新建 `rag/common/minio_client.py`:

```python
import asyncio
import io

from minio import Minio

from rag.config import Settings


def create_minio_client(settings: Settings) -> Minio:
    """创建 minio 客户端并确保 bucket 存在。"""
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
    return client


async def put_object(
    client: Minio, bucket: str, key: str, data: bytes, content_type: str
) -> None:
    """上传字节对象(同步 SDK 转线程池,避免阻塞 event loop)。"""
    await asyncio.to_thread(
        client.put_object, bucket, key, io.BytesIO(data), len(data), content_type
    )


async def get_object(client: Minio, bucket: str, key: str) -> bytes:
    """下载对象字节。"""

    def _get() -> bytes:
        resp = client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    return await asyncio.to_thread(_get)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_minio_client.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/common/minio_client.py tests/test_minio_client.py
git commit -m "feat: minio 客户端(to_thread 异步封装)"
```

---

### Task 5: 数据库迁移 `0002`(3 张表 + 默认 kb)

**Files:**
- Create: `alembic/versions/0002_document_ingestion.py`
- Modify: `rag/document/__init__.py`(新增 `DEFAULT_KB_ID` 常量)
- Test: `tests/test_migration.py`(追加,integration)

**Interfaces:**
- Produces: `knowledge_bases` / `documents` / `document_chunks` 三表 + 默认 kb 行;代码侧常量 `rag.document.DEFAULT_KB_ID`。

- [ ] **Step 1: 写失败测试**

向 `tests/test_migration.py` 追加:

```python
@pytest.mark.integration
def test_document_tables_exist_with_default_kb():
    s = get_settings()
    with psycopg.connect(s.pg_async_dsn) as conn:
        with conn.cursor() as cur:
            for table in ("knowledge_bases", "documents", "document_chunks"):
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                assert cur.fetchone()[0] == table
            cur.execute(
                "SELECT name FROM knowledge_bases "
                "WHERE id = '00000000-0000-0000-0000-000000000001'"
            )
            assert cur.fetchone()[0] == "default"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_migration.py::test_document_tables_exist_with_default_kb -v -m integration`
Expected: FAIL(`to_regclass` 返回 None —— 表尚不存在)

- [ ] **Step 3: 写实现**

新建 `alembic/versions/0002_document_ingestion.py`:

```python
"""document ingestion: knowledge_bases + documents + document_chunks

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-24
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        INSERT INTO knowledge_bases (id, name)
        VALUES ('{DEFAULT_KB_ID}', 'default')
        ON CONFLICT (id) DO NOTHING
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id),
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            content_hash TEXT,
            object_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT,
            chunk_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            knowledge_base_id UUID NOT NULL,
            chunk_index INT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, metadata)
            WITH (key_field='id')
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dchunks_doc ON document_chunks (document_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    op.execute("DROP INDEX IF EXISTS idx_dchunks_embedding")
    op.execute("DROP TABLE IF EXISTS document_chunks")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS knowledge_bases")
```

把 `rag/document/__init__.py` 内容设为(与迁移里的 UUID 必须一致):

```python
# 默认知识库 ID,须与 alembic 0002 迁移中的 DEFAULT_KB_ID 保持一致
DEFAULT_KB_ID = "00000000-0000-0000-0000-000000000001"
```

- [ ] **Step 4: 应用迁移并跑测试确认通过**

Run: `uv run alembic upgrade head`
Then: `uv run pytest tests/test_migration.py -v -m integration`
Expected: PASS(需 docker compose 起的 paradedb)

- [ ] **Step 5: 提交**

```bash
git add alembic/versions/0002_document_ingestion.py rag/document/__init__.py tests/test_migration.py
git commit -m "feat: 迁移 0002 新增 kb/documents/document_chunks 三表"
```

---

### Task 6: 文档存储层 `store.py`(async DB,SQL 收口)

**Files:**
- Create: `rag/document/store.py`
- Test: `tests/test_document_store.py`(新建,integration)

**Interfaces:**
- Consumes: `rag.db.postgres.get_cursor`、`AsyncConnectionPool`。
- Produces:
  - `async create_document(pool, *, knowledge_base_id, filename, content_type, size_bytes, content_hash, object_key) -> str`
  - `async get_document(pool, document_id) -> dict | None`
  - `async set_status(pool, document_id, status, *, error=None) -> None`
  - `async store_chunks_and_complete(pool, document_id, knowledge_base_id, embedded) -> None`,其中 `embedded: list[tuple[int, str, list[float]]]`,在**单事务**内删旧 chunk + 插新 chunk + 置 `status='done'` + 写 `chunk_count`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_document_store.py`:

```python
import pytest

from rag.config import get_settings
from rag.db import create_pg_pool
from rag.document import DEFAULT_KB_ID, store


@pytest.mark.integration
async def test_store_lifecycle_and_idempotent_replace():
    pool = await create_pg_pool(get_settings())
    try:
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=DEFAULT_KB_ID,
            filename="a.txt",
            content_type="txt",
            size_bytes=5,
            content_hash="hash1",
            object_key="k1",
        )

        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "pending"
        assert doc["filename"] == "a.txt"

        await store.set_status(pool, doc_id, "processing")
        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "processing"

        emb = [0.0] * get_settings().embedding_dim
        await store.store_chunks_and_complete(
            pool, doc_id, DEFAULT_KB_ID, [(0, "c0", emb), (1, "c1", emb)]
        )
        doc = await store.get_document(pool, doc_id)
        assert doc["status"] == "done"
        assert doc["chunk_count"] == 2

        # 幂等:重放只覆盖,不累加
        await store.store_chunks_and_complete(
            pool, doc_id, DEFAULT_KB_ID, [(0, "only", emb)]
        )
        doc = await store.get_document(pool, doc_id)
        assert doc["chunk_count"] == 1

        assert await store.get_document(pool, "00000000-0000-0000-0000-0000000000ff") is None
    finally:
        await pool.close()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_store.py -v -m integration`
Expected: FAIL(`ModuleNotFoundError: rag.document.store`)

- [ ] **Step 3: 写实现**

新建 `rag/document/store.py`:

```python
import uuid

from psycopg_pool import AsyncConnectionPool

from rag.db.postgres import get_cursor


async def create_document(
    pool: AsyncConnectionPool,
    *,
    knowledge_base_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    content_hash: str,
    object_key: str,
) -> str:
    document_id = str(uuid.uuid4())
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            INSERT INTO documents
                (id, knowledge_base_id, filename, content_type,
                 size_bytes, content_hash, object_key, status)
            VALUES
                (%(id)s, %(kb)s, %(fn)s, %(ct)s,
                 %(sz)s, %(hash)s, %(key)s, 'pending')
            """,
            {
                "id": document_id,
                "kb": knowledge_base_id,
                "fn": filename,
                "ct": content_type,
                "sz": size_bytes,
                "hash": content_hash,
                "key": object_key,
            },
        )
    return document_id


async def get_document(pool: AsyncConnectionPool, document_id: str) -> dict | None:
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT * FROM documents WHERE id = %(id)s", {"id": document_id}
        )
        return await cur.fetchone()


async def set_status(
    pool: AsyncConnectionPool, document_id: str, status: str, *, error: str | None = None
) -> None:
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET status = %(s)s, error = %(e)s, updated_at = now()
            WHERE id = %(id)s
            """,
            {"s": status, "e": error, "id": document_id},
        )


async def store_chunks_and_complete(
    pool: AsyncConnectionPool,
    document_id: str,
    knowledge_base_id: str,
    embedded: list[tuple[int, str, list[float]]],
) -> None:
    """单事务:删旧 chunk → 插新 chunk → 置 done + chunk_count。可安全重放。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM document_chunks WHERE document_id = %(id)s",
                {"id": document_id},
            )
            for chunk_index, text, embedding in embedded:
                await cur.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, knowledge_base_id, chunk_index, text, embedding)
                    VALUES
                        (%(doc)s, %(kb)s, %(idx)s, %(text)s, %(emb)s)
                    """,
                    {
                        "doc": document_id,
                        "kb": knowledge_base_id,
                        "idx": chunk_index,
                        "text": text,
                        "emb": embedding,
                    },
                )
            await cur.execute(
                """
                UPDATE documents
                SET status = 'done', chunk_count = %(n)s, updated_at = now()
                WHERE id = %(id)s
                """,
                {"n": len(embedded), "id": document_id},
            )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_store.py -v -m integration`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/document/store.py tests/test_document_store.py
git commit -m "feat: 文档存储层 store(幂等单事务入库)"
```

---

### Task 7: 编排管线 `pipeline.py`(web/worker 唯一汇合点)

**Files:**
- Create: `rag/document/pipeline.py`
- Test: `tests/test_document_pipeline.py`(新建)

**Interfaces:**
- Consumes: `parse`(Task 3)、`chunk`(Task 2)、`store`(Task 6)、`get_object`(Task 4);`ctx` 字典含 `pg`/`minio`/`bucket`/`embedding`/`settings`;`embedding.embed(list[str]) -> list[list[float]]`。
- Produces: `async ingest_document(ctx: dict, document_id: str) -> None`。失败时置 `status='failed'` 并 re-raise(供 arq 重试)。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_document_pipeline.py`:

```python
from types import SimpleNamespace

import pytest

import rag.document.pipeline as pipe


class _FakeEmbedding:
    def __init__(self):
        self.batches = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


def _settings(batch=2):
    return SimpleNamespace(chunk_size=800, chunk_overlap=100, embedding_batch_size=batch)


async def test_ingest_happy_path_batches_and_completes(monkeypatch):
    statuses = []
    completed = {}

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k1", "content_type": "txt", "knowledge_base_id": "kb1"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        completed["embedded"] = embedded
        completed["kb"] = kb

    async def fake_get_object(client, bucket, key):
        return b"ignored-by-fake-parse"

    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "full text")
    monkeypatch.setattr(pipe, "chunk", lambda text, size, overlap: ["a", "b", "c"])

    emb = _FakeEmbedding()
    ctx = {
        "pg": None, "minio": None, "bucket": "b",
        "embedding": emb, "settings": _settings(batch=2),
    }

    await pipe.ingest_document(ctx, "d1")

    assert statuses[0] == ("processing", None)
    assert emb.batches == [["a", "b"], ["c"]]
    assert completed["kb"] == "kb1"
    assert completed["embedded"] == [
        (0, "a", [0.0, 0.0, 0.0, 0.0]),
        (1, "b", [0.0, 0.0, 0.0, 0.0]),
        (2, "c", [0.0, 0.0, 0.0, 0.0]),
    ]


async def test_ingest_empty_chunks_marks_failed(monkeypatch):
    statuses = []

    async def fake_set_status(pool, doc_id, status, error=None):
        statuses.append((status, error))

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb"}

    monkeypatch.setattr(pipe.store, "set_status", fake_set_status)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe, "get_object", lambda *a, **k: _async_bytes())
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "")
    monkeypatch.setattr(pipe, "chunk", lambda text, size, overlap: [])

    ctx = {"pg": None, "minio": None, "bucket": "b",
           "embedding": _FakeEmbedding(), "settings": _settings()}

    with pytest.raises(ValueError):
        await pipe.ingest_document(ctx, "d1")
    assert ("processing", None) in statuses
    assert any(s == "failed" for s, _ in statuses)


async def _async_bytes():
    return b"data"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_pipeline.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.document.pipeline`)

- [ ] **Step 3: 写实现**

新建 `rag/document/pipeline.py`:

```python
import logging

from rag.common.minio_client import get_object
from rag.document import store
from rag.document.chunker import chunk
from rag.document.parser import parse

logger = logging.getLogger(__name__)


async def ingest_document(ctx: dict, document_id: str) -> None:
    """web 投递、worker 执行的入库编排。失败置 failed 并 re-raise 供重试。"""
    pool = ctx["pg"]
    minio = ctx["minio"]
    bucket = ctx["bucket"]
    embedding = ctx["embedding"]
    settings = ctx["settings"]

    await store.set_status(pool, document_id, "processing")
    try:
        doc = await store.get_document(pool, document_id)
        if doc is None:
            raise ValueError(f"document not found: {document_id}")

        data = await get_object(minio, bucket, doc["object_key"])
        text = parse(data, doc["content_type"])
        chunks = chunk(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("切块结果为空,无可入库内容")

        embedded: list[tuple[int, str, list[float]]] = []
        batch = settings.embedding_batch_size
        index = 0
        for i in range(0, len(chunks), batch):
            window = chunks[i : i + batch]
            vectors = await embedding.embed(window)
            for text_piece, vector in zip(window, vectors):
                embedded.append((index, text_piece, vector))
                index += 1

        await store.store_chunks_and_complete(
            pool, document_id, doc["knowledge_base_id"], embedded
        )
    except Exception as e:
        logger.exception("文档入库失败: %s", document_id)
        await store.set_status(pool, document_id, "failed", error=str(e)[:500])
        raise
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/document/pipeline.py tests/test_document_pipeline.py
git commit -m "feat: 入库编排 pipeline(分批 embed + 失败置 failed)"
```

---

### Task 8: arq worker `worker/main.py`

**Files:**
- Create: `rag/worker/__init__.py`(空)
- Create: `rag/worker/main.py`
- Test: `tests/test_worker.py`(新建)

**Interfaces:**
- Consumes: `ingest_document`(Task 7)、`create_pg_pool`、`create_minio_client`、`EmbeddingModel`、`Settings`、`arq.connections.RedisSettings`。
- Produces: `WorkerSettings`(`functions`/`redis_settings`/`on_startup`/`on_shutdown`/`max_tries`/`job_timeout`)、`async on_startup(ctx)`、`async on_shutdown(ctx)`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_worker.py`:

```python
from rag.document.pipeline import ingest_document
from rag.worker.main import WorkerSettings


def test_worker_registers_ingest_function():
    assert ingest_document in WorkerSettings.functions


def test_worker_retry_and_timeout_configured():
    assert WorkerSettings.max_tries == 3
    assert WorkerSettings.job_timeout == 300


def test_worker_uses_arq_redis_db():
    assert WorkerSettings.redis_settings.database == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_worker.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.worker.main`)

- [ ] **Step 3: 写实现**

新建空文件 `rag/worker/__init__.py`。

新建 `rag/worker/main.py`:

```python
from arq.connections import RedisSettings

from rag.common.logging import setup_logging
from rag.common.minio_client import create_minio_client
from rag.config import get_settings
from rag.db import create_pg_pool
from rag.document.pipeline import ingest_document
from rag.models.embedding import EmbeddingModel

_settings = get_settings()


async def on_startup(ctx: dict) -> None:
    setup_logging()
    settings = get_settings()
    ctx["settings"] = settings
    ctx["pg"] = await create_pg_pool(settings)
    ctx["minio"] = create_minio_client(settings)
    ctx["bucket"] = settings.minio_bucket
    ctx["embedding"] = EmbeddingModel(settings)


async def on_shutdown(ctx: dict) -> None:
    await ctx["pg"].close()


class WorkerSettings:
    functions = [ingest_document]
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_tries = 3
    job_timeout = 300
    redis_settings = RedisSettings(
        host=_settings.redis_host,
        port=_settings.redis_port,
        database=_settings.arq_redis_db,
        password=_settings.redis_password,
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_worker.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/worker/__init__.py rag/worker/main.py tests/test_worker.py
git commit -m "feat: arq worker(on_startup 建资源 + 重试/超时配置)"
```

---

### Task 9: 薄 Depends `dependence/storage.py`

**Files:**
- Create: `rag/api/dependence/storage.py`
- Test: `tests/test_dependence_storage.py`(新建)

**Interfaces:**
- Consumes: `request.app.state.minio` / `request.app.state.arq_pool`(Task 10 写入)。
- Produces: `get_minio(request) -> Minio`、`get_arq_pool(request)`。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_dependence_storage.py`:

```python
from types import SimpleNamespace

from rag.api.dependence.storage import get_arq_pool, get_minio


def test_get_minio_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(minio=sentinel)))
    assert get_minio(request) is sentinel


def test_get_arq_pool_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=sentinel)))
    assert get_arq_pool(request) is sentinel
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_dependence_storage.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.api.dependence.storage`)

- [ ] **Step 3: 写实现**

新建 `rag/api/dependence/storage.py`:

```python
from fastapi import Request
from minio import Minio


def get_minio(request: Request) -> Minio:
    return request.app.state.minio


def get_arq_pool(request: Request):
    return request.app.state.arq_pool
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_dependence_storage.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add rag/api/dependence/storage.py tests/test_dependence_storage.py
git commit -m "feat: 新增 get_minio/get_arq_pool 薄 Depends"
```

---

### Task 10: lifespan 构造 minio + arq pool 挂 `app.state`

**Files:**
- Modify: `rag/api/main.py`
- Test: `tests/test_api_lifespan.py`(追加,integration)

**Interfaces:**
- Consumes: `create_minio_client`(Task 4)、`arq.create_pool`、`RedisSettings`。
- Produces: 启动后 `app.state.minio` / `app.state.arq_pool` 就绪;关闭时 `arq_pool.aclose()`。

- [ ] **Step 1: 写失败测试**

向 `tests/test_api_lifespan.py` 追加:

```python
@pytest.mark.integration
def test_lifespan_populates_storage():
    from rag.api.main import app

    with TestClient(app):
        assert app.state.minio is not None
        assert app.state.arq_pool is not None
```

(若文件未导入 `pytest` / `TestClient`,沿用其已有顶部导入即可。)

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_api_lifespan.py::test_lifespan_populates_storage -v -m integration`
Expected: FAIL(`AttributeError: ... 'minio'`)

- [ ] **Step 3: 写实现**

修改 `rag/api/main.py`。在顶部 import 区追加:

```python
from arq import create_pool
from arq.connections import RedisSettings

from rag.common.minio_client import create_minio_client
```

在 `lifespan` 内、`app.state.llm = NormalModel(settings)` 之后追加:

```python
    # ------ 初始化对象存储与任务队列 -------
    app.state.minio = create_minio_client(settings)
    app.state.arq_pool = await create_pool(
        RedisSettings(
            host=settings.redis_host,
            port=settings.redis_port,
            database=settings.arq_redis_db,
            password=settings.redis_password,
        )
    )
```

在关闭段(`await app.state.redis.aclose()` 之后)追加:

```python
    await app.state.arq_pool.aclose()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_api_lifespan.py -v -m integration`
Expected: PASS(需 docker compose 起 pg/redis/minio)

- [ ] **Step 5: 提交**

```bash
git add rag/api/main.py tests/test_api_lifespan.py
git commit -m "feat: lifespan 构造 minio/arq_pool 挂 app.state"
```

---

### Task 11: 上传/查询 controller + 注册路由

**Files:**
- Create: `rag/api/modules/document/__init__.py`
- Create: `rag/api/modules/document/controller.py`
- Modify: `rag/api/modules/register.py`
- Test: `tests/test_document_controller.py`(新建)

**Interfaces:**
- Consumes: `get_pg`(`dependence/db.py`)、`get_minio`/`get_arq_pool`(Task 9)、`put_object`(Task 4)、`store`(Task 6)、`DEFAULT_KB_ID`(Task 5)、`get_settings`。
- Produces: `document_router`;`POST /documents/` → `202 {document_id, status}`;`GET /documents/{id}` → 状态 JSON 或 `404`;`register_modules` 挂载该路由。

- [ ] **Step 1: 写失败测试**

新建 `tests/test_document_controller.py`:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

import rag.api.modules.document.controller as ctrl
from rag.api.dependence.db import get_pg
from rag.api.dependence.storage import get_arq_pool, get_minio


class _FakeArq:
    def __init__(self):
        self.jobs = []

    async def enqueue_job(self, name, *args):
        self.jobs.append((name, args))


def _app(arq):
    app = FastAPI()
    app.include_router(ctrl.document_router)
    app.dependency_overrides[get_pg] = lambda: object()
    app.dependency_overrides[get_minio] = lambda: object()
    app.dependency_overrides[get_arq_pool] = lambda: arq
    return app


def test_upload_rejects_unknown_type():
    arq = _FakeArq()
    client = TestClient(_app(arq))
    resp = client.post(
        "/documents/", files={"file": ("a.exe", b"x", "application/octet-stream")}
    )
    assert resp.status_code == 400
    assert arq.jobs == []


def test_upload_happy_path_enqueues(monkeypatch):
    created = {}

    async def fake_put(client, bucket, key, data, content_type):
        created["key"] = key
        created["data"] = data

    async def fake_create_document(pool, **kw):
        created.update(kw)
        return "doc-1"

    monkeypatch.setattr(ctrl, "put_object", fake_put)
    monkeypatch.setattr(ctrl.store, "create_document", fake_create_document)

    arq = _FakeArq()
    client = TestClient(_app(arq))
    resp = client.post(
        "/documents/", files={"file": ("note.txt", b"hello", "text/plain")}
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body == {"document_id": "doc-1", "status": "pending"}
    assert arq.jobs == [("ingest_document", ("doc-1",))]
    assert created["content_type"] == "txt"
    assert created["data"] == b"hello"


def test_get_status_returns_404_when_missing(monkeypatch):
    async def fake_get(pool, doc_id):
        return None

    monkeypatch.setattr(ctrl.store, "get_document", fake_get)

    app = FastAPI()
    app.include_router(ctrl.document_router)
    app.dependency_overrides[get_pg] = lambda: object()
    client = TestClient(app)
    resp = client.get("/documents/missing-id")
    assert resp.status_code == 404


def test_get_status_returns_doc(monkeypatch):
    async def fake_get(pool, doc_id):
        return {
            "id": "doc-1", "filename": "a.txt", "status": "done",
            "chunk_count": 3, "error": None,
        }

    monkeypatch.setattr(ctrl.store, "get_document", fake_get)

    app = FastAPI()
    app.include_router(ctrl.document_router)
    app.dependency_overrides[get_pg] = lambda: object()
    client = TestClient(app)
    resp = client.get("/documents/doc-1")
    assert resp.status_code == 200
    assert resp.json() == {
        "document_id": "doc-1", "filename": "a.txt", "status": "done",
        "chunk_count": 3, "error": None,
    }
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_document_controller.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.api.modules.document.controller`)

- [ ] **Step 3: 写实现**

新建 `rag/api/modules/document/controller.py`:

```python
import hashlib

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from rag.api.dependence.db import get_pg
from rag.api.dependence.storage import get_arq_pool, get_minio
from rag.common.minio_client import put_object
from rag.config import get_settings
from rag.document import DEFAULT_KB_ID, store

document_router = APIRouter(prefix="/documents")

ALLOWED_TYPES = {"txt", "md", "pdf"}


def _safe_filename(name: str | None) -> str:
    base = (name or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return base or "upload"


@document_router.post("/", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(DEFAULT_KB_ID),
    pg=Depends(get_pg),
    minio=Depends(get_minio),
    arq_pool=Depends(get_arq_pool),
):
    settings = get_settings()
    filename = _safe_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件超过大小上限")

    content_hash = hashlib.sha256(data).hexdigest()
    object_key = f"{knowledge_base_id}/{content_hash[:16]}-{filename}"

    await put_object(
        minio,
        settings.minio_bucket,
        object_key,
        data,
        file.content_type or "application/octet-stream",
    )
    document_id = await store.create_document(
        pg,
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        content_type=ext,
        size_bytes=len(data),
        content_hash=content_hash,
        object_key=object_key,
    )
    await arq_pool.enqueue_job("ingest_document", document_id)

    return JSONResponse(
        status_code=202, content={"document_id": document_id, "status": "pending"}
    )


@document_router.get("/{document_id}")
async def get_document_status(document_id: str, pg=Depends(get_pg)):
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document_id": str(doc["id"]),
        "filename": doc["filename"],
        "status": doc["status"],
        "chunk_count": doc["chunk_count"],
        "error": doc["error"],
    }
```

新建 `rag/api/modules/document/__init__.py`:

```python
from rag.api.modules.document.controller import document_router

__all__ = ["document_router"]
```

修改 `rag/api/modules/register.py` 为:

```python
from fastapi import FastAPI

from rag.api.modules.chat import chat_router
from rag.api.modules.document import document_router


def register_modules(app: FastAPI):
    app.include_router(chat_router)
    app.include_router(document_router)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_document_controller.py -v`
Expected: PASS

- [ ] **Step 5: 全量非 integration 回归 + 提交**

Run: `uv run pytest -m "not integration" -v`
Expected: 全绿

```bash
git add rag/api/modules/document/ rag/api/modules/register.py tests/test_document_controller.py
git commit -m "feat: 文档上传/查询 controller 并注册路由"
```

---

## Self-Review 备注

- **Spec 覆盖**:配置/依赖/minio 服务(Task 1)、切块(Task 2)、解析(Task 3)、minio 客户端(Task 4)、三表迁移 + 默认 kb(Task 5)、存储层幂等单事务(Task 6)、编排分批 embed + 失败置 failed(Task 7)、arq worker + 重试/超时(Task 8)、薄 Depends(Task 9)、lifespan 挂载(Task 10)、上传/查询 controller + 注册(Task 11)—— 与 spec 各节逐条对应。
- **生产要点落地**:幂等重跑 = Task 6 `store_chunks_and_complete` 先删后插 + Task 8 `max_tries=3`;原子提交 = Task 6 单连接事务;分批 embed = Task 7;入口校验 = Task 11(类型白名单 + 大小上限 + 文件名安全)。
- **已知取舍**:孤儿记录兜底、内容去重(`content_hash` 仅填入)、死信队列 —— 按 spec 不实现。
- **类型一致**:`ingest_document(ctx, document_id)` 在 Task 7 定义,Task 8 注册、Task 11 enqueue 名称 `"ingest_document"` 一致;`store.*` 签名在 Task 6 定义,Task 7/11 一致引用;`DEFAULT_KB_ID` 迁移(Task 5)与代码常量一致;`put_object/get_object` 签名 Task 4 定义,Task 7/11 一致;`document_chunks` 列与 `store_chunks_and_complete` 的 INSERT 列一致。
```

