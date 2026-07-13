# Remove DEFAULT_KB_ID + Novel/Regulation KBs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded `DEFAULT_KB_ID` with two named KB constants (小说/法规); chat searches all KBs by default; document upload requires explicit KB selection.

**Architecture:** A new migration (0008) wipes the old default KB and seeds two KB rows with LIST partitions. `store.search_chunks` and `search_chunks_bm25` switch from `knowledge_base_id: str` to `knowledge_base_ids: list[str] | None`, using `ANY()` for multi-KB filtering. The recall node omits the arg entirely (None = all partitions). Upload API drops its default value and frontend adds a KB dropdown.

**Tech Stack:** Python 3.12, FastAPI, psycopg3, Alembic, React 18 + TypeScript, TailwindCSS

## Global Constraints

- No `DEFAULT_KB_ID` constant anywhere in code after this change
- `knowledge_base_ids: list[str] | None` — None means search all KBs
- New KB partitions are created at runtime by `ensure_kb_partition()`, not by DEFAULT partition
- `EVAL_KB_ID` remains unchanged and independent
- History migrations (0002, 0007) are not modified

---

### Task 1: Alembic Migration 0008 — Wipe old KB, seed novel + regulation

**Files:**
- Create: `alembic/versions/0008_remove_default_kb.py`

**Interfaces:**
- Produces: `knowledge_bases` rows for 小说 (`...0002`) and 法规 (`...0003`); partitioned `document_chunks` with `dchunks_novel` and `dchunks_regulation` partitions; no DEFAULT partition.

- [ ] **Step 1: Generate blank migration**

```bash
uv run alembic revision -m "remove default KB, seed novel + regulation KBs"
```

Expected: creates `alembic/versions/XXXX_remove_default_kb.py` with `upgrade()` / `downgrade()` stubs.

- [ ] **Step 2: Write migration**

Replace the generated stub content with:

```python
"""remove default KB, seed novel + regulation KBs

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-13
"""
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024
NOVEL_KB_ID = "00000000-0000-0000-0000-000000000002"
REGULATION_KB_ID = "00000000-0000-0000-0000-000000000003"


def upgrade() -> None:
    # 1. 清空旧数据
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS knowledge_bases CASCADE")

    # 2. 重建 knowledge_bases
    op.execute(
        """
        CREATE TABLE knowledge_bases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # 3. 种子两条 KB
    op.execute(
        f"""
        INSERT INTO knowledge_bases (id, name) VALUES
            ('{NOVEL_KB_ID}', '小说'),
            ('{REGULATION_KB_ID}', '法规')
        """
    )

    # 4. 重建 documents
    op.execute(
        """
        CREATE TABLE documents (
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
            graph_status TEXT NOT NULL DEFAULT 'pending',
            graph_error TEXT,
            retry_count INT NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # 5. 重建分区父表
    op.execute(
        f"""
        CREATE TABLE document_chunks (
            id UUID DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL,
            knowledge_base_id UUID NOT NULL,
            chunk_index INT NOT NULL,
            text TEXT NOT NULL,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (id, knowledge_base_id)
        ) PARTITION BY LIST (knowledge_base_id)
        """
    )

    # 6. 为两个 KB 创建分区（无 DEFAULT 分区 — 新 KB 由 ensure_kb_partition() runtime 建）
    op.execute(
        f"""
        CREATE TABLE dchunks_novel PARTITION OF document_chunks
            FOR VALUES IN ('{NOVEL_KB_ID}')
        """
    )
    op.execute(
        f"""
        CREATE TABLE dchunks_regulation PARTITION OF document_chunks
            FOR VALUES IN ('{REGULATION_KB_ID}')
        """
    )

    # 7. FK + 索引
    op.execute(
        """
        ALTER TABLE document_chunks
            ADD CONSTRAINT dchunks_doc_fk FOREIGN KEY (document_id)
            REFERENCES documents(id) ON DELETE CASCADE
        """
    )
    op.execute(
        f"""
        CREATE INDEX idx_dchunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        "CREATE INDEX idx_dchunks_doc ON document_chunks (document_id)"
    )

    # BM25: 尝试建在父表，失败则逐分区建
    try:
        op.execute(
            f"""
            CREATE INDEX idx_dchunks_bm25
                ON document_chunks USING bm25 (id, text, knowledge_base_id)
                WITH (
                    key_field='id',
                    text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                )
            """
        )
    except Exception:
        for partition in ["dchunks_novel", "dchunks_regulation"]:
            try:
                op.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_dchunks_bm25_{partition}
                        ON {partition} USING bm25 (id, text, knowledge_base_id)
                        WITH (
                            key_field='id',
                            text_fields='{{"text": {{"tokenizer": {{"type": "jieba"}}}}}}'
                        )
                    """
                )
            except Exception:
                pass


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS knowledge_bases CASCADE")
```

- [ ] **Step 3: Run migration**

```bash
uv run alembic upgrade head
```

Expected: `INFO  [alembic.runtime.migration] Running upgrade 0007 -> 0008, remove default KB, seed novel + regulation KBs`

- [ ] **Step 4: Verify DB state**

```bash
uv run python -c "
import asyncio
from rag.config import get_settings
from rag.db.postgres import create_pg_pool

async def check():
    settings = get_settings()
    pool = await create_pg_pool(settings)
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute('SELECT id, name FROM knowledge_bases ORDER BY name')
            rows = await cur.fetchall()
            for r in rows:
                print(r)
            # verify partitions
            await cur.execute(\"SELECT relname FROM pg_class WHERE relname LIKE 'dchunks_%' AND relkind = 'r' ORDER BY relname\")
            parts = await cur.fetchall()
            for p in parts:
                print(p)
    await pool.close()

asyncio.run(check())
"
```

Expected output shows: `('00000000-0000-0000-0000-000000000003', '法规')`, `('00000000-0000-0000-0000-000000000002', '小说')`, partitions `dchunks_novel`, `dchunks_regulation`.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/0008_remove_default_kb.py
git commit -m "feat(db): migration 0008 — replace default KB with 小说 + 法规 KBs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Replace DEFAULT_KB_ID with NOVEL_KB_ID + REGULATION_KB_ID constants

**Files:**
- Modify: `rag/document/__init__.py:1-2`

**Interfaces:**
- Deletes: `DEFAULT_KB_ID`
- Produces: `NOVEL_KB_ID = "00000000-0000-0000-0000-000000000002"`, `REGULATION_KB_ID = "00000000-0000-0000-0000-000000000003"`

- [ ] **Step 1: Replace constants**

Edit `rag/document/__init__.py`, replace lines 1-2:

```python
# 知识库 ID 常量，须与 alembic 0008 迁移中的种子数据保持一致
NOVEL_KB_ID      = "00000000-0000-0000-0000-000000000002"
REGULATION_KB_ID = "00000000-0000-0000-0000-000000000003"
```

- [ ] **Step 2: Verify no DEFAULT_KB_ID references remain in this file**

```bash
grep -n "DEFAULT_KB_ID" rag/document/__init__.py
```

Expected: no matches (exit code 1).

- [ ] **Step 3: Commit**

```bash
git add rag/document/__init__.py
git commit -m "refactor: replace DEFAULT_KB_ID with NOVEL_KB_ID + REGULATION_KB_ID

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Update store.py — search_chunks / search_chunks_bm25 to accept list[str] | None

**Files:**
- Modify: `rag/document/store.py:46-92`

**Interfaces:**
- Changes: `search_chunks(pool, embedding, knowledge_base_id: str, top_k)` → `search_chunks(pool, embedding, knowledge_base_ids: list[str] | None, top_k)`
- Changes: `search_chunks_bm25(pool, query_text, knowledge_base_id: str, top_k)` → `search_chunks_bm25(pool, query_text, knowledge_base_ids: list[str] | None, top_k)`

- [ ] **Step 1: Update search_chunks**

Edit `rag/document/store.py`, lines 46-67:

```python
async def search_chunks(
    pool: AsyncConnectionPool,
    embedding: list[float],
    knowledge_base_ids: list[str] | None,
    top_k: int = 5,
) -> list[dict]:
    """按余弦相似度从知识库召回最相关的 chunk。

    knowledge_base_ids 为 None 时搜全部 KB；传入列表时走分区裁剪。
    """
    kb_filter = ""
    params: dict = {"emb": Vector(embedding), "k": top_k}
    if knowledge_base_ids is not None:
        kb_filter = "AND dc.knowledge_base_id = ANY(%(kb_ids)s)"
        params["kb_ids"] = knowledge_base_ids

    async with get_cursor(pool) as cur:
        await cur.execute(
            f"""
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.text,
                   1 - (dc.embedding <=> %(emb)s) AS similarity,
                   d.filename
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE 1=1 {kb_filter}
            ORDER BY dc.embedding <=> %(emb)s
            LIMIT %(k)s
            """,
            params,
        )
        return await cur.fetchall()
```

- [ ] **Step 2: Update search_chunks_bm25**

Edit `rag/document/store.py`, lines 70-92:

```python
async def search_chunks_bm25(
    pool: AsyncConnectionPool,
    query_text: str,
    knowledge_base_ids: list[str] | None,
    top_k: int = 5,
) -> list[dict]:
    """BM25 词法召回:jieba 分词索引,paradedb.match 安全构造(特殊字符不炸解析器)。

    knowledge_base_ids 为 None 时搜全部 KB；传入列表时走分区裁剪。
    """
    kb_filter = ""
    params: dict = {"q": query_text, "k": top_k}
    if knowledge_base_ids is not None:
        kb_filter = "AND dc.knowledge_base_id = ANY(%(kb_ids)s)"
        params["kb_ids"] = knowledge_base_ids

    async with get_cursor(pool) as cur:
        await cur.execute(
            f"""
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.text,
                   paradedb.score(dc.id) AS score,
                   d.filename
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
            WHERE dc.text @@@ paradedb.match('text', %(q)s)
                  {kb_filter}
            ORDER BY score DESC
            LIMIT %(k)s
            """,
            params,
        )
        return await cur.fetchall()
```

- [ ] **Step 3: Commit**

```bash
git add rag/document/store.py
git commit -m "feat(store): search_chunks / search_chunks_bm25 accept list[str] | None for KB filter

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update retriever.py — KnowledgeRetriever.search() signature

**Files:**
- Modify: `rag/document/retriever.py:82-84`

**Interfaces:**
- Changes: `search(self, query, knowledge_base_id: str, top_k=5)` → `search(self, query, knowledge_base_ids: list[str] | None = None, top_k=5)`

- [ ] **Step 1: Update method signature and internal calls**

Edit `rag/document/retriever.py`, lines 82-84 and the two leg functions:

```python
@observe_if_enabled(name="knowledge_retrieve")
async def search(
    self, query: str, knowledge_base_ids: list[str] | None = None, top_k: int = 5
) -> list[dict]:
    candidates = top_k * self._candidate_multiplier
    timings: dict[str, float] = {}
    lexical_query = _lexical_query(query)

    span_input = {"query": query[:200], "kb_ids": knowledge_base_ids, "candidates": candidates}

    async def _vec_leg() -> list[dict]:
        with span_scope("vector_recall", input=span_input) as span:
            t0 = time.perf_counter()
            try:
                emb = (await self._embedding.embed([query]))[0]
                timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                t1 = time.perf_counter()
                rows = await store.search_chunks(
                    self._pool, emb, knowledge_base_ids, candidates
                )
                # ... rest unchanged
```

The `_bm25_leg` inner function similarly passes `knowledge_base_ids`:

```python
    async def _bm25_leg() -> list[dict]:
        with span_scope("bm25_recall", input=span_input) as span:
            t0 = time.perf_counter()
            if not lexical_query:
                rows: list[dict] = []
            else:
                try:
                    rows = await store.search_chunks_bm25(
                        self._pool, lexical_query, knowledge_base_ids, candidates
                    )
                except Exception:
                    logger.warning("BM25 召回失败，降级纯向量", exc_info=True)
                    rows = []
            # ... rest unchanged
```

- [ ] **Step 2: Verify signature grep — no old str param**

```bash
grep -n "knowledge_base_id" rag/document/retriever.py
```

Expected: the parameter is named `knowledge_base_ids` (plural).

- [ ] **Step 3: Commit**

```bash
git add rag/document/retriever.py
git commit -m "feat(retriever): search() accepts list[str] | None for multi-KB or all-KB recall

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Update eval harness — pass [EVAL_KB_ID] to updated APIs

**Files:**
- Modify: `rag/eval/harness.py:91,100,108,116,124,134`

**Interfaces:**
- Consumes: `retriever.search(query, knowledge_base_ids)` (list[str]|None), `store.search_chunks(pool, emb, knowledge_base_ids, top_k)`, `store.search_chunks_bm25(pool, lex, knowledge_base_ids, top_k)`

- [ ] **Step 1: Update all eval callsites**

The eval harness calls `retriever.search(rw, EVAL_KB_ID)`, `store.search_chunks(pool, emb, EVAL_KB_ID, top_k)`, and `store.search_chunks_bm25(pool, lex, EVAL_KB_ID, top_k)`. Change each to pass `[EVAL_KB_ID]`:

In `rag/eval/harness.py`:

- Line 91: `rows = await retriever.search(rw, [EVAL_KB_ID], top_k=top_k)`
- Line 100: `raw_rows = await retriever.search(item.query, [EVAL_KB_ID], top_k=top_k)`
- Line 108: `vec_rows = await store.search_chunks(pool, emb, [EVAL_KB_ID], top_k)`
- Line 116: `raw_vec_rows = await store.search_chunks(pool, raw_emb, [EVAL_KB_ID], top_k)`
- Line 124: `await store.search_chunks_bm25(pool, lex, [EVAL_KB_ID], top_k)`
- Line 135: `await store.search_chunks_bm25(pool, raw_lex, [EVAL_KB_ID], top_k)`

- [ ] **Step 2: Verify no bare EVAL_KB_ID string passed to these functions**

```bash
grep -n "search_chunks\|search_chunks_bm25\|retriever.search" rag/eval/harness.py
```

Expected: all calls pass `[EVAL_KB_ID]` (list), not `EVAL_KB_ID` (str).

- [ ] **Step 3: Commit**

```bash
git add rag/eval/harness.py
git commit -m "fix(eval): pass [EVAL_KB_ID] as list to updated search APIs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Update recall node — search all KBs by default

**Files:**
- Modify: `rag/agent/nodes/recall/recall.py:5,24`

**Interfaces:**
- Removes: import of `DEFAULT_KB_ID`
- Changes: `retriever.search(query, kb_id)` → `retriever.search(query)` (no kb_ids arg = search all)

- [ ] **Step 1: Edit recall.py**

Remove the `DEFAULT_KB_ID` import and the kb_id fallback logic:

```python
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.common.logging import get_logger
from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event

logger = get_logger()


async def recall(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """从知识库召回相关 chunk（向量+BM25 混合检索 RRF 融合），写入 recall_vec_results。"""
    writer = get_stream_writer()
    writer(stream_event(StreamEventType.STATUS, "检索知识库中..."))

    retriever = runtime.context.retriever
    if retriever is None:
        logger.warning("retriever 未注入,跳过知识库召回")
        state["recall_vec_results"] = []
        return state

    # 改写后的查询更适合检索,缺失时回退原始查询
    query = state.get("rewrite_query") or state["raw_query"]
    # kb_ids 不传 → 搜全部知识库
    state["recall_vec_results"] = await retriever.search(query)
    return state
```

- [ ] **Step 2: Commit**

```bash
git add rag/agent/nodes/recall/recall.py
git commit -m "feat(recall): search all KBs by default — no kb_ids filter

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Update workflow.py — remove knowledge_base_id from initial state

**Files:**
- Modify: `rag/agent/workflow.py:12,88`

**Interfaces:**
- Removes: `from rag.document import DEFAULT_KB_ID`
- Removes: `"knowledge_base_id": DEFAULT_KB_ID` from initial state dict

- [ ] **Step 1: Edit workflow.py**

Remove the import on line 12:

```python
# Delete this line:
# from rag.document import DEFAULT_KB_ID
```

Remove the `knowledge_base_id` entry from the initial state dict (line 88):

```python
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
```

- [ ] **Step 2: Verify no DEFAULT_KB_ID or knowledge_base_id references in agent/**

```bash
grep -rn "DEFAULT_KB_ID\|knowledge_base_id" rag/agent/
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add rag/agent/workflow.py
git commit -m "feat(workflow): remove knowledge_base_id from chat initial state

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Update document upload controller — make knowledge_base_id required

**Files:**
- Modify: `rag/api/modules/document/controller.py:21,37`

**Interfaces:**
- Removes: `from rag.document import DEFAULT_KB_ID`
- Changes: `knowledge_base_id: str = Form(DEFAULT_KB_ID)` → `knowledge_base_id: str = Form(...)`

- [ ] **Step 1: Edit controller.py**

Remove the `DEFAULT_KB_ID` import on line 21:

```python
# Delete this line:
# from rag.document import DEFAULT_KB_ID
```

Change the Form parameter on line 37:

```python
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(...),
    pg=Depends(get_pg),
    minio=Depends(get_minio),
    arq_pool=Depends(get_arq_pool),
) -> DocumentUploadResponse:
```

- [ ] **Step 2: Verify no DEFAULT_KB_ID in api/**

```bash
grep -rn "DEFAULT_KB_ID" rag/api/
```

Expected: no matches.

- [ ] **Step 3: Commit**

```bash
git add rag/api/modules/document/controller.py
git commit -m "feat(api): make knowledge_base_id required in document upload

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: Update frontend client.ts — make knowledgeBaseId required

**Files:**
- Modify: `frontend/src/api/client.ts:15,19`

**Interfaces:**
- Changes: `uploadDocument(file, knowledgeBaseId = "00000000-...")` → `uploadDocument(file, knowledgeBaseId)` (no default)

- [ ] **Step 1: Edit client.ts**

Change line 15 — remove the default value:

```typescript
export async function uploadDocument(
  file: File,
  knowledgeBaseId: string,
): Promise<{ document_id: string; status: string }> {
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api/client.ts
git commit -m "feat(frontend): make knowledgeBaseId required in uploadDocument

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Add KB selector dropdown to DocumentUpload component

**Files:**
- Modify: `frontend/src/components/DocumentUpload.tsx`

**Interfaces:**
- Consumes: `uploadDocument(file, knowledgeBaseId)` (required 2nd arg from Task 9)

- [ ] **Step 1: Add KB selector state and UI**

Edit `frontend/src/components/DocumentUpload.tsx`:

Add KB ID constants and state near the top of the component:

```typescript
const KB_OPTIONS: Record<string, string> = {
  "小说": "00000000-0000-0000-0000-000000000002",
  "法规": "00000000-0000-0000-0000-000000000003",
};
```

Add state inside the component (after `const [error, setError] = useState("")`):

```typescript
const [kbId, setKbId] = useState(Object.values(KB_OPTIONS)[0]);
```

Add the dropdown UI before the drop zone (inside the `space-y-3` div, before the drop zone div):

```tsx
{/* KB 选择器 */}
<div className="flex items-center gap-2">
  <label className="text-xs text-gray-400">知识库：</label>
  <select
    value={kbId}
    onChange={(e) => setKbId(e.target.value)}
    className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200
               focus:outline-none focus:border-emerald-500/50 transition-colors"
  >
    {Object.entries(KB_OPTIONS).map(([name, id]) => (
      <option key={id} value={id}>{name}</option>
    ))}
  </select>
</div>
```

Update `handleUpload` to pass `kbId`:

```typescript
const { document_id } = await uploadDocument(file, kbId);
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/DocumentUpload.tsx
git commit -m "feat(frontend): add KB selector dropdown to document upload

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: Update tests — replace DEFAULT_KB_ID with NOVEL_KB_ID

**Files:**
- Modify: `tests/test_document_store.py:5,14,32,40,56,79,93,115`

**Interfaces:**
- Consumes: `NOVEL_KB_ID` from `rag.document` (Task 2)
- Consumes: updated `store.search_chunks` and `store.store_chunks_and_complete` signatures

- [ ] **Step 1: Update imports and all references**

Edit `tests/test_document_store.py`:

Line 5 — change import:

```python
from rag.document import NOVEL_KB_ID, store
```

Replace all occurrences of `DEFAULT_KB_ID` with `NOVEL_KB_ID`:
- Line 14: `knowledge_base_id=NOVEL_KB_ID,`
- Line 32: `pool, doc_id, NOVEL_KB_ID, [(0, "c0", emb), (1, "c1", emb)]`
- Line 40: `pool, doc_id, NOVEL_KB_ID, [(0, "only", emb)]`
- Line 56: `knowledge_base_id=NOVEL_KB_ID,`
- Line 79: `await store.store_chunks_and_complete(pool, doc_id, NOVEL_KB_ID, [(0, "x", emb)])`
- Line 93: `knowledge_base_id=NOVEL_KB_ID,`
- Line 115: `pool, doc_id, NOVEL_KB_ID, [(0, "x", emb, {})]`

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_document_store.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_document_store.py
git commit -m "test: replace DEFAULT_KB_ID with NOVEL_KB_ID

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: Final verification — grep for any remaining DEFAULT_KB_ID references

**Files:**
- None (verification only)

- [ ] **Step 1: Grep entire codebase (excluding docs/migrations)**

```bash
grep -rn "DEFAULT_KB_ID" rag/ tests/ frontend/src/
```

Expected: **zero matches** across all source directories.

- [ ] **Step 2: Grep for old UUID in source**

```bash
grep -rn "00000000-0000-0000-0000-000000000001" rag/ tests/ frontend/src/
```

Expected: **zero matches** (the old default KB UUID should not appear in any source file).

- [ ] **Step 3: Quick smoke test — verify imports work**

```bash
uv run python -c "from rag.document import NOVEL_KB_ID, REGULATION_KB_ID; print(NOVEL_KB_ID, REGULATION_KB_ID)"
```

Expected: prints the two UUIDs without import errors.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v --ignore=tests/test_db.py --ignore=tests/test_db_neo4j.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit verification results (if any cleanup needed) or mark complete**
