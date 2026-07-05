# BM25 + 向量 RRF 混合检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检索从纯向量升级为「向量 + BM25(jieba 分词) 两路并发 → RRF 融合」，签名与图结构不变。

**Architecture:** 迁移 0005 用 jieba 分词器重建 `idx_dchunks_bm25`（含 kb 过滤字段）；`store.search_chunks_bm25` 提供与向量版对称的词法召回；`KnowledgeRetriever.search` 内部 `asyncio.gather` 并发两路、RRF(k=60) 融合，BM25 路失败降级纯向量；日志按路输出排名摘要。

**Tech Stack:** paradedb pg_search 0.24.1（jieba tokenizer、`@@@`、`paradedb.match`、`paradedb.score` 均已在容器实测）/ alembic / psycopg / pytest (asyncio_mode=auto) / uv

**Spec:** `docs/superpowers/specs/2026-07-05-hybrid-retrieval-design.md`

## Global Constraints

- 所有命令 `uv run` 前缀；pytest `asyncio_mode = "auto"`。
- 中文注释半角逗号，注释解释「为什么」。
- 动态基线：全量 suite 当前 **20 个失败**（18 分支基线 + 2 来自并行会话未提交 WIP）。「全量通过」= 失败集合不超出。
- 并行会话脏文件（`rag/agent/nodes/generate/generate.py`、`rag/agent/workflow.py`、`rag/config.py`、`rag/document/store.py`、`rag/graph/pipeline.py`、`rag/memory/adapters/long_term_pgsql.py`）——除 `store.py` 的本特性新增函数外一律不碰；`store.py` 提交必须 hunk 级暂存（见 Task 2 Step 5 的具体流程）。
- Task 1/4 需要 docker 的 rag-postgres 在跑（已在跑）；单测不触网不依赖 docker。
- Commit message 末尾加：`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 1: 迁移 0005 —— jieba 分词重建 BM25 索引

**Files:**
- Create: `alembic/versions/0005_bm25_jieba.py`

**Interfaces:**
- Consumes: 迁移 0004（`down_revision = "0004"`；执行前用 `ls alembic/versions/` 确认 0004 是最新）。
- Produces: `document_chunks` 上的 `idx_dchunks_bm25`（jieba 分词、含 `knowledge_base_id`），Task 2 的查询依赖它。

- [ ] **Step 1: 写迁移**

新建 `alembic/versions/0005_bm25_jieba.py`：

```python
"""rebuild document_chunks bm25 index: jieba tokenizer + kb filter field

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-05
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    # 默认分词器对中文切不出词,BM25 统计无意义;jieba 词典分词已在
    # pg_search 0.24.1 容器内实测。knowledge_base_id 进索引供查询按库过滤。
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, knowledge_base_id)
            WITH (
                key_field='id',
                text_fields='{"text": {"tokenizer": {"type": "jieba"}}}'
            )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dchunks_bm25")
    # 恢复 0002 的原始定义
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dchunks_bm25
            ON document_chunks USING bm25 (id, text, metadata)
            WITH (key_field='id')
        """
    )
```

- [ ] **Step 2: 真实执行迁移**

Run: `uv run alembic upgrade head`
Expected: 无报错，输出含 `Running upgrade 0004 -> 0005`。

- [ ] **Step 3: 真实验证索引生效**

```bash
docker exec rag-postgres psql -U rag -d rag_memory -c "SELECT COUNT(*) FROM document_chunks WHERE text @@@ paradedb.match('text', '悟空')"
```

Expected: 正常返回计数（库里有西游记语料应 >0；为 0 但无报错也算索引可用，报告中注明语料情况）。若报错，贴完整错误并对照 pg_search 0.24.1 文档调整语法后重试；migration 文件与实际执行的 SQL 必须一致（改了 SQL 就 `uv run alembic downgrade -1` 后重跑）。

- [ ] **Step 4: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 失败集合与 20 基线一致。

```bash
git add alembic/versions/0005_bm25_jieba.py
git commit -m "feat: BM25 索引重建为 jieba 分词并纳入 kb 过滤字段"
```

---

### Task 2: `store.search_chunks_bm25`

**Files:**
- Modify: `rag/document/store.py`（⚠️ 携带并行会话未提交改动——只新增一个函数，提交走 hunk 级暂存）

**Interfaces:**
- Consumes: Task 1 的索引；现有 `get_cursor`。
- Produces: `async def search_chunks_bm25(pool, query_text: str, knowledge_base_id: str, top_k: int = 5) -> list[dict]`，返回列 `id, document_id, chunk_index, text, score`（与 `search_chunks` 的 `similarity` 对称）。Task 3 调用。

- [ ] **Step 1: 实现**

在 `rag/document/store.py` 的 `search_chunks` 函数之后新增（保持与其相同的风格）：

```python
async def search_chunks_bm25(
    pool: AsyncConnectionPool,
    query_text: str,
    knowledge_base_id: str,
    top_k: int = 5,
) -> list[dict]:
    """BM25 词法召回:jieba 分词索引,paradedb.match 安全构造(特殊字符不炸解析器)。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT id, document_id, chunk_index, text,
                   paradedb.score(id) AS score
            FROM document_chunks
            WHERE text @@@ paradedb.match('text', %(q)s)
              AND knowledge_base_id = %(kb)s
            ORDER BY score DESC
            LIMIT %(k)s
            """,
            {"q": query_text, "kb": knowledge_base_id, "k": top_k},
        )
        return await cur.fetchall()
```

- [ ] **Step 2: 真实验证（SQL 层薄函数，验证走真实库）**

写临时脚本（scratchpad，不入库）连接真实 pg 调用该函数查「金箍棒」，Expected: 无异常，返回 list[dict] 且含 `score` 降序；库无语料时空列表也算过（报告注明）。

- [ ] **Step 3: 全量测试**

Run: `uv run pytest -q`
Expected: 失败集合与 20 基线一致。

- [ ] **Step 4: hunk 级暂存提交**

`store.py` 上有并行会话的其他改动，禁止整文件 add。流程：

```bash
git diff -U1 rag/document/store.py   # 确认新函数是独立 hunk
# 把仅含 search_chunks_bm25 新增 hunk 的补丁写到 scratchpad,然后:
git apply --cached <scratchpad>/store-bm25.patch
git diff --cached rag/document/store.py   # 复核:暂存区只有新函数
git diff rag/document/store.py            # 复核:并行会话改动仍留在工作区
git commit -m "feat: store 增加 search_chunks_bm25 词法召回"
```

若新函数与并行会话改动纠缠在同一 hunk 无法拆分，停下报告 BLOCKED。

---

### Task 3: `KnowledgeRetriever.search` 混合化（RRF 融合 + 分路日志）

**Files:**
- Modify: `rag/document/retriever.py`（当前干净）
- Test: `tests/test_retriever.py`（现有 4 个用例适配 + 新增 RRF 用例）

**Interfaces:**
- Consumes: Task 2 的 `store.search_chunks_bm25`；现有 `store.search_chunks`、`observe_if_enabled`。
- Produces: `search` 签名不变；返回行新增 `rrf_score`、`sources` 键（原有键保留，下游按 `text`/`id` 访问不受影响）。

- [ ] **Step 1: 写失败测试**

`tests/test_retriever.py` 全文替换为：

```python
import logging

from rag.document.retriever import KnowledgeRetriever, _rrf_fuse


class _FakeEmbedding:
    async def embed(self, texts):
        return [[0.1, 0.2]]


def _row(cid: str, text: str = "正文", **extra) -> dict:
    return {
        "id": cid,
        "document_id": "d1",
        "chunk_index": 0,
        "text": text,
        **extra,
    }


def _retriever(monkeypatch, vec_rows, bm25_rows=None, bm25_exc=None):
    import rag.document.retriever as mod

    async def fake_vec(pool, embedding, kb_id, top_k):
        return vec_rows

    async def fake_bm25(pool, query_text, kb_id, top_k):
        if bm25_exc is not None:
            raise bm25_exc
        return bm25_rows or []

    monkeypatch.setattr(mod.store, "search_chunks", fake_vec)
    monkeypatch.setattr(mod.store, "search_chunks_bm25", fake_bm25)
    return KnowledgeRetriever(None, _FakeEmbedding())


# ── _rrf_fuse 纯逻辑 ──

def test_rrf_overlap_ranks_shared_chunk_first():
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    bm25 = [_row("c", score=5.0), _row("a", score=4.0)]
    fused = _rrf_fuse(vec, bm25, top_k=3)
    assert [r["id"] for r in fused][0] == "a"  # 双路命中 RRF 最高
    assert fused[0]["sources"] == ["vec", "bm25"]


def test_rrf_disjoint_interleaves_by_rank():
    vec = [_row("a", similarity=0.9)]
    bm25 = [_row("b", score=5.0)]
    fused = _rrf_fuse(vec, bm25, top_k=5)
    assert {r["id"] for r in fused} == {"a", "b"}
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]  # 各自 rank 1,并列


def test_rrf_single_empty_leg_passthrough_order():
    vec = [_row("a", similarity=0.9), _row("b", similarity=0.8)]
    fused = _rrf_fuse(vec, [], top_k=5)
    assert [r["id"] for r in fused] == ["a", "b"]
    assert all(r["sources"] == ["vec"] for r in fused)


def test_rrf_truncates_to_top_k():
    vec = [_row(f"v{i}", similarity=1.0 - i * 0.1) for i in range(5)]
    fused = _rrf_fuse(vec, [], top_k=3)
    assert len(fused) == 3


# ── search 行为 ──

async def test_search_fuses_and_logs_both_legs(monkeypatch, caplog):
    vec = [_row("a", text="花果山", similarity=0.87654)]
    bm25 = [_row("b", text="金箍棒", score=4.2)]
    r = _retriever(monkeypatch, vec, bm25)
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("孙悟空的兵器", "kb-1")
    assert {x["id"] for x in result} == {"a", "b"}
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert rec.levelno == logging.INFO
    assert rec.kb_id == "kb-1"
    assert rec.vec_hits[0]["chunk_id"] == "a"
    assert rec.vec_hits[0]["similarity"] == 0.8765
    assert rec.bm25_hits[0]["chunk_id"] == "b"
    assert rec.bm25_hits[0]["score"] == 4.2
    assert rec.hits[0]["sources"] in (["vec"], ["bm25"])
    assert rec.hits[0]["text_preview"]
    assert rec.embed_ms >= 0 and rec.search_ms >= 0 and rec.bm25_ms >= 0


async def test_search_bm25_failure_degrades_to_vector_only(monkeypatch, caplog):
    vec = [_row("a", similarity=0.9)]
    r = _retriever(monkeypatch, vec, bm25_exc=RuntimeError("index missing"))
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("q", "kb-1")
    assert [x["id"] for x in result] == ["a"]
    assert any("BM25 召回失败" in x.getMessage() for x in caplog.records)


async def test_search_empty_both_legs_logs_warning(monkeypatch, caplog):
    r = _retriever(monkeypatch, [], [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        result = await r.search("无关问题", "kb-1")
    assert result == []
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert rec.levelno == logging.WARNING


async def test_search_truncates_long_query_in_log(monkeypatch, caplog):
    r = _retriever(monkeypatch, [], [])
    with caplog.at_level(logging.INFO, logger="rag.document.retriever"):
        await r.search("长" * 300, "kb-1")
    rec = next(x for x in caplog.records if "混合召回" in x.getMessage())
    assert len(rec.query) == 200
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: FAIL —— `ImportError: cannot import name '_rrf_fuse'`。

- [ ] **Step 3: 实现**

`rag/document/retriever.py` 全文替换为：

```python
import asyncio
import time

from rag.common.logging import get_logger
from rag.document import store
from rag.models.embedding import EmbeddingModel
from rag.observability.langfuse import observe_if_enabled

logger = get_logger()

# RRF 常数与每路候选倍数:检索内部细节,有评测数据前不进 Settings。
RRF_K = 60
CANDIDATE_MULTIPLIER = 2


def _rrf_fuse(
    vec_rows: list[dict], bm25_rows: list[dict], top_k: int
) -> list[dict]:
    """RRF 融合:score = Σ 1/(RRF_K + rank),按 id 去重,行数据取先见者。"""
    fused: dict[str, dict] = {}
    for source, rows in (("vec", vec_rows), ("bm25", bm25_rows)):
        for rank, row in enumerate(rows):
            entry = fused.setdefault(
                str(row["id"]), {"row": row, "rrf_score": 0.0, "sources": []}
            )
            entry["rrf_score"] += 1.0 / (RRF_K + rank + 1)
            entry["sources"].append(source)
    ranked = sorted(
        fused.values(), key=lambda e: e["rrf_score"], reverse=True
    )[:top_k]
    results = []
    for entry in ranked:
        row = dict(entry["row"])
        row["rrf_score"] = round(entry["rrf_score"], 6)
        row["sources"] = entry["sources"]
        results.append(row)
    return results


def _rank_summary(rows: list[dict], score_key: str) -> list[dict]:
    """单路排名摘要进日志:排查「为什么这条被召回」时看各路贡献。"""
    return [
        {
            "chunk_id": str(r["id"]),
            "chunk_index": r["chunk_index"],
            score_key: round(r[score_key], 4),
            "text_preview": r["text"][:80],
        }
        for r in rows
    ]


class KnowledgeRetriever:
    """知识库检索:向量(pgvector)+BM25(pg_search/jieba)并发召回,RRF 融合。"""

    def __init__(self, pool, embedding: EmbeddingModel) -> None:
        self._pool = pool
        self._embedding = embedding

    @observe_if_enabled(name="knowledge_retrieve")
    async def search(
        self, query: str, knowledge_base_id: str, top_k: int = 5
    ) -> list[dict]:
        candidates = top_k * CANDIDATE_MULTIPLIER
        timings: dict[str, float] = {}

        async def _vec_leg() -> list[dict]:
            t0 = time.perf_counter()
            emb = (await self._embedding.embed([query]))[0]
            timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            rows = await store.search_chunks(
                self._pool, emb, knowledge_base_id, candidates
            )
            timings["search_ms"] = round((time.perf_counter() - t1) * 1000, 1)
            return rows

        async def _bm25_leg() -> list[dict]:
            t0 = time.perf_counter()
            try:
                rows = await store.search_chunks_bm25(
                    self._pool, query, knowledge_base_id, candidates
                )
            except Exception:  # noqa: BLE001 - 词法路失败降级,不拖垮检索
                logger.warning("BM25 召回失败,降级纯向量", exc_info=True)
                rows = []
            timings["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return rows

        # 向量路异常照常传播(主路径语义不变);BM25 路已在内部兜底
        vec_rows, bm25_rows = await asyncio.gather(_vec_leg(), _bm25_leg())
        results = _rrf_fuse(vec_rows, bm25_rows, top_k)

        fields = {
            "kb_id": knowledge_base_id,
            "query": query[:200],  # 截断,长文本进日志没有意义
            "top_k": top_k,
            "vec_hits": _rank_summary(vec_rows, "similarity"),
            "bm25_hits": _rank_summary(bm25_rows, "score"),
            "hits": [
                {
                    "chunk_id": str(r["id"]),
                    "chunk_index": r["chunk_index"],
                    "rrf_score": r["rrf_score"],
                    "sources": r["sources"],
                    "text_preview": r["text"][:80],
                }
                for r in results
            ],
            **timings,
        }
        if results:
            logger.info(
                "混合召回完成: %d 条 (向量 %d + BM25 %d)",
                len(results), len(vec_rows), len(bm25_rows),
                extra=fields,
            )
        else:
            # 空结果是检索质量最直接的信号,升级 WARNING
            logger.warning("混合召回为空", extra=fields)
        return results
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retriever.py -v`
Expected: 8 个用例全部 PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `uv run pytest -q`
Expected: 失败集合与 20 基线一致（`retriever.py`、`test_retriever.py` 当前干净，可整文件提交）。

```bash
git add rag/document/retriever.py tests/test_retriever.py
git commit -m "feat: 检索改混合召回(向量+BM25 并发,RRF 融合,分路日志)"
```

---

### Task 4: 真实端到端验证

**Files:**
- Create: `<scratchpad>/hybrid_smoke.py`（临时脚本，不入库）

**Interfaces:**
- Consumes: Task 1-3 全部成果 + 真实 pg（语料）+ `.env` 的 embedding key。

**这是判断型任务**：验收标准固定，手段按现场情况调整。

- [ ] **Step 1: smoke 脚本**

脚本内容：建 `AsyncConnectionPool`（DSN 按 `rag/config.py` 的 PG_* 或项目现有连接工具），构造真实 `EmbeddingModel` 与 `KnowledgeRetriever`，跑两个查询并打印：
1. 关键词型：「金箍棒」——期望 `bm25_hits` 非空且预览文本确实含关键词；
2. 语义型：「孙悟空的师父是谁」——期望向量路正常，融合结果非空。

同时打印 `vec_hits` / `bm25_hits` / 融合 `hits`（含 sources），对比两路差异。

- [ ] **Step 2: 运行并验收**

Run: `uv run python <scratchpad>/hybrid_smoke.py`
验收：无 traceback；关键词查询的 BM25 路命中含关键词的 chunk；融合列表 `sources` 标记正确；`rrf_score` 降序。若库里没有语料，先用现有上传接口/脚本灌入西游记文本再跑（报告说明做法）。

- [ ] **Step 3: 全量回归 + 汇报**

Run: `uv run pytest -q`
Expected: 失败集合与 20 基线一致。smoke 输出摘录进报告，无代码提交。
