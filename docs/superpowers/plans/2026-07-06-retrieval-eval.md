# 检索层评测体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为检索层搭一套离线、确定性、可 CI 门禁的防回归评测：给定 golden（query → gold 原文片段），跑现有混合检索链路，算 hit@k/recall@k/mrr/ndcg@k，与 baseline 对比，跌破容差即 fail。

**Architecture:** 纯 Python 复用现有 `KnowledgeRetriever` / `store` / `chunker` / `EmbeddingModel` / `NormalModel`，不引第三方评测框架。评测语料冻结在 `rag/eval/datasets/corpus.txt`，灌入独立评测 KB。指标计算是与检索解耦的纯函数（可脱离 DB 单测）；跑检索的 runner 与门禁 CLI 依赖真实 DB + embedding，由 `@pytest.mark.eval` 用例守护。命中判定绑「归一化后子串包含 gold 片段」，跨 re-chunk 存活。

**Tech Stack:** Python ≥3.12、asyncio、psycopg3（AsyncConnectionPool，dict_row）、pgvector + paradedb/BM25、pytest + pytest-asyncio、langchain-openai（既有 LLM/embedding 封装）。

## Global Constraints

- Python `>=3.12`；全部 IO 走 async；DB 复用 `rag.db.postgres.create_pg_pool` / `get_cursor`（dict_row）。
- 不新增第三方依赖（本期）。检索执行必须复用生产链路 `rag.document.retriever.KnowledgeRetriever.search(query, kb_id, top_k)`。
- 评测 KB 常量 `EVAL_KB_ID = "00000000-0000-0000-0000-0000000000ee"`，与生产 `DEFAULT_KB_ID`（`...0001`）隔离。
- 所有数据文件 UTF-8；golden 为 jsonl（一行一 JSON 对象）。
- Windows 下 psycopg3 需 SelectorEventLoop —— `tests/conftest.py` 已全局设置，测试无需重复处理。
- 命中判定与归一化只允许在 `rag/eval/metrics.py` 内单一实现，golden 生成与评测共用，禁止复制。
- chunk 行结构（`KnowledgeRetriever.search` 返回）关键字段：`id`、`chunk_index`、`text`、`rrf_score`、`sources`、`similarity`/`score`。

---

### Task 1: 指标核心 —— 归一化 + 单条 query 指标

**Files:**
- Create: `rag/eval/__init__.py`
- Create: `rag/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Produces:
  - `EVAL_KB_ID: str`、`EVAL_DIR: Path`、`DATASETS_DIR: Path`（`rag/eval/__init__.py`）
  - `normalize(text: str) -> str`
  - `evaluate_query(retrieved_texts: list[str], gold_snippets: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]`
    返回键：`mrr`、`hit@{k}`、`recall@{k}`、`ndcg@{k}`（每个 k）。

- [ ] **Step 1: 建包与常量**

Create `rag/eval/__init__.py`:

```python
from pathlib import Path

# 评测专用知识库 id，与生产 DEFAULT_KB_ID(...0001) 隔离
EVAL_KB_ID = "00000000-0000-0000-0000-0000000000ee"

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"
```

- [ ] **Step 2: 写失败测试**

Create `tests/test_eval_metrics.py`:

```python
from rag.eval.metrics import evaluate_query, normalize


def test_normalize_strips_punct_ws_and_casefolds():
    assert normalize("如意金箍棒，重 一万三千五百斤！") == normalize("如意金箍棒重一万三千五百斤")
    assert normalize("ABC def") == "abcdef"


def test_evaluate_query_first_rank_hit():
    m = evaluate_query(
        retrieved_texts=["……定海神针……", "无关内容", "更多无关"],
        gold_snippets=["定海神针"],
        ks=(1, 3, 5),
    )
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["recall@1"] == 1.0
    assert m["ndcg@1"] == 1.0


def test_evaluate_query_hit_at_rank3_not_rank1():
    m = evaluate_query(
        retrieved_texts=["无关", "无关", "……定海神针……"],
        gold_snippets=["定海神针"],
        ks=(1, 3, 5),
    )
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["mrr"] == 1 / 3


def test_evaluate_query_partial_recall_multi_snippet():
    m = evaluate_query(
        retrieved_texts=["只含 定海神针 这一句", "无关"],
        gold_snippets=["定海神针", "重一万三千五百斤"],
        ks=(1, 3, 5),
    )
    assert m["recall@5"] == 0.5


def test_evaluate_query_no_hit_all_zero():
    m = evaluate_query(["无关一", "无关二"], ["定海神针"], ks=(1, 3, 5))
    assert m["hit@5"] == 0.0
    assert m["mrr"] == 0.0
    assert m["recall@5"] == 0.0
    assert m["ndcg@5"] == 0.0


def test_snippet_match_survives_rechunk_punctuation_noise():
    # gold 片段与检索文本标点/空白不同，仍应命中（跨 re-chunk 的核心诉求）
    m = evaluate_query(["如意金箍棒\n重一万三千五百斤"], ["如意金箍棒，重一万三千五百斤"], ks=(1,))
    assert m["hit@1"] == 1.0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_eval_metrics.py -q`
Expected: FAIL（`ModuleNotFoundError: rag.eval.metrics` 或函数未定义）

- [ ] **Step 4: 实现 `rag/eval/metrics.py`**

```python
import math
import re

# \W(默认 Unicode)保留中文/字母/数字为词字符，去掉标点与空白；补 _ 因其属词字符
_STRIP_RE = re.compile(r"[\W_]+")


def normalize(text: str) -> str:
    """归一化用于片段包含匹配：剔除所有标点与空白（中英文），大小写折叠。"""
    return _STRIP_RE.sub("", text).casefold()


def _hit_indices(chunk_text: str, gold_norm: list[str]) -> set[int]:
    """该 chunk 归一化后命中的 gold 片段下标集合（子串包含即命中）。"""
    norm_chunk = normalize(chunk_text)
    return {i for i, g in enumerate(gold_norm) if g and g in norm_chunk}


def evaluate_query(
    retrieved_texts: list[str],
    gold_snippets: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """对单条 query 的有序检索结果算指标（纯确定性）。

    - hit@k：top-k 内是否至少命中一个 gold 片段（0/1）
    - recall@k：top-k 内命中的不同 gold 片段数 / gold 片段总数
    - mrr：首个命中 chunk 位置的倒数（无命中=0）
    - ndcg@k：二值相关性 NDCG，IDCG 以 min(k, gold 数) 个理想命中位归一化
    """
    n_gold = len(gold_snippets)
    gold_norm = [normalize(g) for g in gold_snippets]
    per_rank = [_hit_indices(t, gold_norm) for t in retrieved_texts]

    result: dict[str, float] = {}

    first_hit = next((r for r, hits in enumerate(per_rank) if hits), None)
    result["mrr"] = 1.0 / (first_hit + 1) if first_hit is not None else 0.0

    for k in ks:
        topk = per_rank[:k]
        result[f"hit@{k}"] = 1.0 if any(topk) else 0.0

        covered: set[int] = set()
        for h in topk:
            covered |= h
        result[f"recall@{k}"] = len(covered) / n_gold if n_gold else 0.0

        dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(topk) if h)
        ideal_n = min(k, n_gold)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_n))
        result[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    return result
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_eval_metrics.py -q`
Expected: PASS（6 passed）

- [ ] **Step 6: 提交**

```bash
git add rag/eval/__init__.py rag/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): 检索指标核心 normalize + evaluate_query"
```

---

### Task 2: 指标聚合 + 门禁比较

**Files:**
- Modify: `rag/eval/metrics.py`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: 无（纯函数）
- Produces:
  - `aggregate(per_query: list[dict[str, float]]) -> dict[str, float]`
  - `gate(current: dict[str, float], baseline: dict[str, float], keys: tuple[str, ...] = ("recall@5", "mrr"), rel_tolerance: float = 0.03) -> tuple[bool, dict[str, dict[str, float]]]`

- [ ] **Step 1: 写失败测试**

Append to `tests/test_eval_metrics.py`:

```python
from rag.eval.metrics import aggregate, gate


def test_aggregate_means_by_key():
    agg = aggregate([{"hit@1": 1.0, "mrr": 1.0}, {"hit@1": 0.0, "mrr": 0.5}])
    assert agg["hit@1"] == 0.5
    assert agg["mrr"] == 0.75


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}


def test_gate_passes_within_tolerance():
    passed, deltas = gate({"recall@5": 0.79, "mrr": 0.80}, {"recall@5": 0.80, "mrr": 0.80})
    assert passed is True
    assert deltas["recall@5"]["rel_drop"] < 0.03


def test_gate_fails_on_regression_beyond_tolerance():
    passed, deltas = gate({"recall@5": 0.70, "mrr": 0.80}, {"recall@5": 0.80, "mrr": 0.80})
    assert passed is False
    assert deltas["recall@5"]["rel_drop"] > 0.03


def test_gate_skips_missing_baseline_keys():
    passed, deltas = gate({"recall@5": 0.5, "mrr": 0.5}, {})
    assert passed is True
    assert deltas == {}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_eval_metrics.py -k "aggregate or gate" -q`
Expected: FAIL（`aggregate` / `gate` 未定义）

- [ ] **Step 3: 追加实现到 `rag/eval/metrics.py`**

```python
def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """对多条 query 的同名指标取算术平均。空输入返回 {}。"""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {k: sum(q[k] for q in per_query) / n for k in keys}


def gate(
    current: dict[str, float],
    baseline: dict[str, float],
    keys: tuple[str, ...] = ("recall@5", "mrr"),
    rel_tolerance: float = 0.03,
) -> tuple[bool, dict[str, dict[str, float]]]:
    """门禁：对每个核心指标，相对基线下降超过 rel_tolerance 判 fail。

    返回 (passed, deltas)，deltas[key] = {"baseline", "current", "rel_drop"}。
    baseline 缺该 key 时跳过（无基线不阻断）。
    """
    passed = True
    deltas: dict[str, dict[str, float]] = {}
    for key in keys:
        if key not in baseline or key not in current:
            continue
        base = baseline[key]
        cur = current[key]
        rel_drop = (base - cur) / base if base > 0 else 0.0
        deltas[key] = {"baseline": base, "current": cur, "rel_drop": rel_drop}
        if rel_drop > rel_tolerance:
            passed = False
    return passed, deltas
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_eval_metrics.py -q`
Expected: PASS（11 passed）

- [ ] **Step 5: 提交**

```bash
git add rag/eval/metrics.py tests/test_eval_metrics.py
git commit -m "feat(eval): 指标聚合 aggregate + 门禁 gate"
```

---

### Task 3: harness —— golden 加载 + 检索 runner

**Files:**
- Create: `rag/eval/harness.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Consumes: `rag.eval.metrics.{evaluate_query, aggregate}`、`rag.eval.EVAL_KB_ID`、`rag.document.retriever.KnowledgeRetriever`、`rag.db.postgres.create_pg_pool`、`rag.models.embedding.EmbeddingModel`、`rag.config.Settings`
- Produces:
  - `GoldenItem`（dataclass：`id: str`、`query: str`、`gold_snippets: list[str]`）
  - `load_golden(path: str | Path) -> list[GoldenItem]`
  - `async build_retriever(settings: Settings) -> tuple[AsyncConnectionPool, KnowledgeRetriever]`
  - `async run_eval(items: list[GoldenItem], retriever: KnowledgeRetriever, ks: tuple[int, ...] = (1, 3, 5), top_k: int = 5) -> dict`
    返回 `{"aggregate": dict, "per_query": list[dict]}`；`per_query` 每项 `{"id", "query", **metrics}`。

- [ ] **Step 1: 写失败测试（纯逻辑：load_golden + run_eval 用 fake retriever）**

Create `tests/test_eval_harness.py`:

```python
from pathlib import Path

import pytest

from rag.eval.harness import GoldenItem, load_golden, run_eval


def test_load_golden_parses_and_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"id":"q1","query":"金箍棒多重?","gold_snippets":["一万三千五百斤"]}\n'
        "\n"
        '{"query":"谁是大师兄?","gold_snippets":["孙悟空"]}\n',
        encoding="utf-8",
    )
    items = load_golden(p)
    assert len(items) == 2
    assert items[0] == GoldenItem(id="q1", query="金箍棒多重?", gold_snippets=["一万三千五百斤"])
    assert items[1].id == "3"  # 无 id 时回退行号


def test_load_golden_rejects_missing_fields(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"query":"缺片段"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="gold_snippets"):
        load_golden(p)


class _FakeRetriever:
    def __init__(self, mapping):
        self._mapping = mapping  # query -> list[chunk text]

    async def search(self, query, knowledge_base_id, top_k=5):
        return [{"id": f"c{i}", "chunk_index": i, "text": t}
                for i, t in enumerate(self._mapping.get(query, [])[:top_k])]


async def test_run_eval_aggregates_over_items():
    items = [
        GoldenItem("q1", "金箍棒多重?", ["一万三千五百斤"]),
        GoldenItem("q2", "大师兄是谁?", ["孙悟空"]),
    ]
    retriever = _FakeRetriever({
        "金箍棒多重?": ["重一万三千五百斤", "无关"],
        "大师兄是谁?": ["无关", "无关"],  # 未命中
    })
    out = await run_eval(items, retriever, ks=(1, 3, 5), top_k=5)
    assert out["aggregate"]["hit@1"] == 0.5  # q1 命中、q2 未命中
    assert len(out["per_query"]) == 2
    assert out["per_query"][0]["id"] == "q1"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_eval_harness.py -q`
Expected: FAIL（`rag.eval.harness` 不存在）

- [ ] **Step 3: 实现 `rag/eval/harness.py`**

```python
import json
from dataclasses import dataclass
from pathlib import Path

from psycopg_pool import AsyncConnectionPool

from rag.config import Settings
from rag.db.postgres import create_pg_pool
from rag.document.retriever import KnowledgeRetriever
from rag.eval import EVAL_KB_ID
from rag.eval.metrics import aggregate, evaluate_query
from rag.models.embedding import EmbeddingModel


@dataclass
class GoldenItem:
    id: str
    query: str
    gold_snippets: list[str]


def load_golden(path) -> list[GoldenItem]:
    """读取 jsonl golden 集；跳过空行；校验必填字段，缺失即报错。"""
    items: list[GoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("query") or not obj.get("gold_snippets"):
                raise ValueError(f"{path}:{lineno} 缺少 query 或 gold_snippets")
            items.append(
                GoldenItem(
                    id=str(obj.get("id", lineno)),
                    query=obj["query"],
                    gold_snippets=list(obj["gold_snippets"]),
                )
            )
    return items


async def build_retriever(settings: Settings) -> tuple[AsyncConnectionPool, KnowledgeRetriever]:
    """构造 pool + 复用生产 KnowledgeRetriever；调用方负责 pool.close()。"""
    pool = await create_pg_pool(settings)
    embedding = EmbeddingModel(settings)
    return pool, KnowledgeRetriever(pool, embedding)


async def run_eval(
    items: list[GoldenItem],
    retriever: KnowledgeRetriever,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> dict:
    """对每条 golden 跑检索并算指标，返回 {"aggregate", "per_query"}。"""
    per_query: list[dict] = []
    for item in items:
        rows = await retriever.search(item.query, EVAL_KB_ID, top_k=top_k)
        texts = [r["text"] for r in rows]
        metrics = evaluate_query(texts, item.gold_snippets, ks)
        per_query.append({"id": item.id, "query": item.query, **metrics})

    metric_keys = [k for k in per_query[0] if k not in ("id", "query")] if per_query else []
    agg = aggregate([{k: q[k] for k in metric_keys} for q in per_query])
    return {"aggregate": agg, "per_query": per_query}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_eval_harness.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add rag/eval/harness.py tests/test_eval_harness.py
git commit -m "feat(eval): harness golden 加载 + 检索 runner"
```

---

### Task 4: 评测语料冻结 + 灌库脚本

**Files:**
- Create: `rag/eval/datasets/corpus.txt`（从未入库的 `rag/document/西游记.txt` 截取冻结选段）
- Create: `rag/eval/seed_corpus.py`

**Interfaces:**
- Consumes: `rag.document.chunker.chunk`、`rag.document.store.{create_document, store_chunks_and_complete}`、`rag.db.postgres.{create_pg_pool, get_cursor}`、`rag.models.embedding.EmbeddingModel`、`rag.config.get_settings`、`rag.eval.{EVAL_KB_ID, DATASETS_DIR}`
- Produces: `async seed() -> int`（入库 chunk 数）、`main()`（`python -m rag.eval.seed_corpus`）

**前置：** 需 `docker compose up -d`（pg）与 `.env` 内 embedding 配置（`EMBEDDING_KEY/URL`）就绪；DB 已跑完 alembic 迁移。

- [ ] **Step 1: 冻结语料**

从未入库的西游记原文截取约 6 万字（含前几回，保证 `第X回` 章节标题完整）写入 `corpus.txt`。编码自动兼容 utf-8 / gbk：

Run（Bash）:
```bash
python - <<'PY'
import pathlib
src = pathlib.Path("rag/document/西游记.txt")
raw = src.read_bytes()
for enc in ("utf-8", "gbk"):
    try:
        text = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
else:
    raise SystemExit("无法解码西游记.txt，请手动确认编码")
out = pathlib.Path("rag/eval/datasets/corpus.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text[:60000], encoding="utf-8")
print("corpus.txt 写入", len(text[:60000]), "字")
PY
```
Expected: 打印 `corpus.txt 写入 60000 字`，文件生成。

- [ ] **Step 2: 实现 `rag/eval/seed_corpus.py`**

```python
import asyncio

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.document import store
from rag.document.chunker import chunk
from rag.eval import DATASETS_DIR, EVAL_KB_ID
from rag.models.embedding import EmbeddingModel

CORPUS_PATH = DATASETS_DIR / "corpus.txt"


async def _reset_eval_kb(pool) -> None:
    """清空评测 KB 的既有文档与 chunk，保证 seed 幂等可重放。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM document_chunks WHERE knowledge_base_id = %(kb)s",
            {"kb": EVAL_KB_ID},
        )
        await cur.execute(
            "DELETE FROM documents WHERE knowledge_base_id = %(kb)s",
            {"kb": EVAL_KB_ID},
        )


def _normalize_pieces(pieces) -> list[tuple[str, dict]]:
    """chunk() 输出归一化为 (text, metadata)，与 pipeline 保持一致。"""
    out: list[tuple[str, dict]] = []
    for p in pieces:
        if isinstance(p, dict):
            out.append((p["content"], {"title": p["title"]} if p.get("title") else {}))
        else:
            out.append((p, {}))
    return out


async def seed() -> int:
    settings = get_settings()
    settings.check_required()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    pieces = chunk(settings.SPLIT_STRATEGY, text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    normalized = _normalize_pieces(pieces)
    if not normalized:
        raise SystemExit("corpus.txt 切块为空")

    pool = await create_pg_pool(settings)
    try:
        await _reset_eval_kb(pool)
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=EVAL_KB_ID,
            filename="corpus.txt",
            content_type="text/plain",
            size_bytes=len(text.encode("utf-8")),
            content_hash="eval-corpus-frozen",
            object_key="eval/corpus.txt",
        )

        embedding = EmbeddingModel(settings)
        embedded: list[tuple[int, str, list[float], dict]] = []
        batch = settings.EMBEDDING_BATCH_SIZE
        index = 0
        for i in range(0, len(normalized), batch):
            window = normalized[i : i + batch]
            vectors = await embedding.embed([t for t, _ in window])
            for (piece, meta), vec in zip(window, vectors):
                embedded.append((index, piece, vec, meta))
                index += 1

        await store.store_chunks_and_complete(pool, doc_id, EVAL_KB_ID, embedded)
        return len(embedded)
    finally:
        await pool.close()


def main() -> None:
    n = asyncio.run(seed())
    print(f"评测语料入库完成：{n} 个 chunk -> KB {EVAL_KB_ID}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 灌库并验证 chunk 数 > 0**

Run: `python -m rag.eval.seed_corpus`
Expected: 打印 `评测语料入库完成：N 个 chunk -> KB 00000000-0000-0000-0000-0000000000ee`，N > 0。

- [ ] **Step 4: 提交**

```bash
git add rag/eval/datasets/corpus.txt rag/eval/seed_corpus.py
git commit -m "feat(eval): 冻结评测语料 + seed_corpus 灌库脚本"
```

---

### Task 5: golden 半自动生成（LLM 造候选 → 人工筛）

**Files:**
- Create: `rag/eval/generate_golden.py`
- Create: `rag/eval/datasets/retrieval_golden.jsonl`（人工筛后的成品，提交入库）

**Interfaces:**
- Consumes: `rag.models.normal.NormalModel`、`rag.db.postgres.{create_pg_pool, get_cursor}`、`rag.config.get_settings`、`rag.eval.{EVAL_KB_ID, DATASETS_DIR}`
- Produces: `GoldenCandidate`（pydantic：`query: str`、`gold_snippets: list[str]`）、`async generate() -> int`、`main()`

**前置：** Task 4 已 seed 评测 KB；`.env` 内 `MODEL_KEY/MODEL_NAME/MODEL_URL` 就绪。

- [ ] **Step 1: 实现 `rag/eval/generate_golden.py`**

```python
import asyncio
import json

from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.eval import DATASETS_DIR, EVAL_KB_ID
from rag.models.normal import NormalModel

CANDIDATES_PATH = DATASETS_DIR / "retrieval_golden.candidates.jsonl"

_PROMPT = """你是检索评测数据的构造助手。下面给你一段原文，请基于它设计 1 个用户可能提出的问题，问题的答案必须能在这段原文中找到。同时抽取答案所依赖的 1~2 个原文短句（原样摘录，不要改写）作为 gold 片段。
要求：
- 问题具体、无歧义，不要出现“这段话”“本文”等指代词。
- gold 片段是原文中连续的短句，尽量短且在全书中唯一。

原文：
{chunk}
"""


class GoldenCandidate(BaseModel):
    query: str = Field(description="用户问题")
    gold_snippets: list[str] = Field(description="答案所依赖的原文短句，1~2 个")


async def _load_chunks(pool) -> list[dict]:
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT id, chunk_index, text FROM document_chunks "
            "WHERE knowledge_base_id = %(kb)s ORDER BY chunk_index",
            {"kb": EVAL_KB_ID},
        )
        return await cur.fetchall()


async def generate() -> int:
    settings = get_settings()
    settings.check_required()
    pool = await create_pg_pool(settings)
    llm = NormalModel(settings)
    try:
        chunks = await _load_chunks(pool)
        if not chunks:
            raise SystemExit("评测 KB 为空，请先运行 python -m rag.eval.seed_corpus")
        count = 0
        with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                try:
                    cand = await llm.ainvoke_structured(
                        [_PROMPT.format(chunk=ch["text"])], GoldenCandidate
                    )
                except Exception:  # 单条失败跳过，不中断整批
                    continue
                row = {
                    "id": f"q{i:03d}",
                    "query": cand.query,
                    "gold_snippets": cand.gold_snippets,
                    "source_chunk_index": ch["chunk_index"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count
    finally:
        await pool.close()


def main() -> None:
    n = asyncio.run(generate())
    print(f"生成候选 {n} 条 -> {CANDIDATES_PATH}")
    print("请人工筛选后另存为 retrieval_golden.jsonl（删去 source_chunk_index 字段可选）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成候选**

Run: `python -m rag.eval.generate_golden`
Expected: 打印 `生成候选 N 条 -> .../retrieval_golden.candidates.jsonl`，N>0。

- [ ] **Step 3: 人工筛（人工步骤，非自动）**

打开 `retrieval_golden.candidates.jsonl`，逐条筛：
- 删歧义题、答案跨多 chunk 的脏数据、太水/太泛的题。
- 校验每条 `gold_snippets` 确实是原文中的短句（可对照 `source_chunk_index`）。
- 保留 **30~50 条** 高质量条目，另存为 `rag/eval/datasets/retrieval_golden.jsonl`（`source_chunk_index` 字段可留可删，评测不读它）。
- `candidates.jsonl` 为中间产物，加入 `.gitignore` 或不提交。

- [ ] **Step 4: 提交成品**

```bash
git add rag/eval/generate_golden.py rag/eval/datasets/retrieval_golden.jsonl
git commit -m "feat(eval): golden 生成脚本 + 首批人工筛检索 golden 集"
```

---

### Task 6: 门禁 CLI + baseline + pytest 回归用例

**Files:**
- Create: `rag/eval/run.py`
- Create: `rag/eval/baseline.json`（由 `--update-baseline` 生成）
- Create: `tests/test_eval_retrieval.py`
- Modify: `pyproject.toml`（`[tool.pytest.ini_options].markers` 增加 `eval`）

**Interfaces:**
- Consumes: `rag.eval.harness.{build_retriever, load_golden, run_eval}`、`rag.eval.metrics.gate`、`rag.eval.{DATASETS_DIR, EVAL_DIR}`、`rag.config.get_settings`
- Produces: `GOLDEN_PATH`、`BASELINE_PATH`、`KS = (1, 3, 5)`、`async _run() -> dict`、`main()`

**前置：** Task 4 已 seed 评测 KB；Task 5 已产出 `retrieval_golden.jsonl`。

- [ ] **Step 1: 注册 pytest marker**

Modify `pyproject.toml` —— 在 `[tool.pytest.ini_options]` 的 `markers` 列表追加一行：

```toml
markers = [
    "integration: 需要 docker compose 起的 pg/redis（默认不跑）",
    "eval: 检索层评测，需 pg + embedding + 已 seed 的评测 KB（默认不跑）",
]
```

同时在 `addopts`（若无则新增）默认排除 eval，使裸 `pytest` 不跑评测：

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "-m 'not integration and not eval'"
markers = [
    "integration: 需要 docker compose 起的 pg/redis（默认不跑）",
    "eval: 检索层评测，需 pg + embedding + 已 seed 的评测 KB（默认不跑）",
]
```

> 注：若原本无 `addopts` 且现有 integration 用例依赖裸 `pytest` 收集，确认加 `addopts` 后 `pytest -m integration` 仍可显式运行（`-m` 覆盖 addopts 的 `-m`）。

- [ ] **Step 2: 实现 `rag/eval/run.py`**

```python
import argparse
import asyncio
import json

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"
KS = (1, 3, 5)


async def _run() -> dict:
    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    if not items:
        raise SystemExit("golden 集为空，请先构造 retrieval_golden.jsonl")
    pool, retriever = await build_retriever(settings)
    try:
        return await run_eval(items, retriever, ks=KS, top_k=max(KS))
    finally:
        await pool.close()


def _load_baseline() -> dict:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="检索层评测")
    parser.add_argument("--update-baseline", action="store_true", help="用本次结果刷新 baseline")
    args = parser.parse_args()

    result = asyncio.run(_run())
    agg = result["aggregate"]

    print("== 检索评测聚合指标 ==")
    for key in sorted(agg):
        print(f"  {key:10s} {agg[key]:.4f}")

    if args.update_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"已更新 baseline -> {BASELINE_PATH}")
        return

    baseline = _load_baseline()
    passed, deltas = gate(agg, baseline)
    print("== 与 baseline 对比 ==")
    for key, d in deltas.items():
        print(f"  {key}: base={d['baseline']:.4f} cur={d['current']:.4f} rel_drop={d['rel_drop']:+.2%}")
    if not passed:
        raise SystemExit("检索指标回归：核心指标跌破容差")
    print("门禁通过")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 首次跑出并冻结 baseline**

Run: `python -m rag.eval.run --update-baseline`
Expected: 打印聚合指标表 + `已更新 baseline -> .../baseline.json`，文件生成且含 `recall@5`、`mrr` 等键。

- [ ] **Step 4: 写 pytest 回归用例**

Create `tests/test_eval_retrieval.py`:

```python
import json

import pytest

from rag.config import get_settings
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate
from rag.eval.run import BASELINE_PATH, GOLDEN_PATH, KS


@pytest.mark.eval
async def test_retrieval_no_regression():
    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    assert items, "golden 集为空"

    pool, retriever = await build_retriever(settings)
    try:
        result = await run_eval(items, retriever, ks=KS, top_k=max(KS))
    finally:
        await pool.close()

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    passed, deltas = gate(result["aggregate"], baseline)
    assert passed, f"检索指标回归：{deltas}"
```

- [ ] **Step 5: 运行评测用例确认通过**

Run: `python -m pytest tests/test_eval_retrieval.py -m eval -q`
Expected: PASS（1 passed）。同时确认裸 `python -m pytest -q` **不** 收集该用例。

- [ ] **Step 6: 提交**

```bash
git add rag/eval/run.py rag/eval/baseline.json tests/test_eval_retrieval.py pyproject.toml
git commit -m "feat(eval): 门禁 CLI + baseline + 检索回归用例"
```

---

### Task 7（OPTIONAL）：`--breakdown` 分路诊断

回归 fail 时，定位是 vec 路还是 bm25 路引起。默认关闭，不影响门禁。

**Files:**
- Modify: `rag/eval/harness.py`（新增 `async run_breakdown(...)`，直接调 `store.search_chunks` / `store.search_chunks_bm25` 单路取回，各自算指标）
- Modify: `rag/eval/run.py`（新增 `--breakdown` 分支，打印 fused / vec-only / bm25-only 三张指标表，不参与 gate）

**Interfaces:**
- Consumes: `rag.document.store.{search_chunks, search_chunks_bm25}`、`rag.document.retriever._lexical_query`、`rag.models.embedding.EmbeddingModel`、`rag.eval.metrics.{evaluate_query, aggregate}`

- [ ] **Step 1: harness 增 `run_breakdown`**

```python
from rag.document import store
from rag.document.retriever import _lexical_query


async def run_breakdown(
    items, pool, embedding, ks=(1, 3, 5), top_k=5
) -> dict:
    """分别评测 vec-only 与 bm25-only 单路召回，用于回归归因。"""
    vec_pq, bm25_pq = [], []
    for item in items:
        emb = (await embedding.embed([item.query]))[0]
        vec_rows = await store.search_chunks(pool, emb, EVAL_KB_ID, top_k)
        vec_pq.append(evaluate_query([r["text"] for r in vec_rows], item.gold_snippets, ks))

        lex = _lexical_query(item.query)
        bm25_rows = await store.search_chunks_bm25(pool, lex, EVAL_KB_ID, top_k) if lex else []
        bm25_pq.append(evaluate_query([r["text"] for r in bm25_rows], item.gold_snippets, ks))
    return {"vec_only": aggregate(vec_pq), "bm25_only": aggregate(bm25_pq)}
```

- [ ] **Step 2: run.py 增 `--breakdown` 分支**

在 `main()` 里 `--update-baseline` 之前插入：解析 `--breakdown`；若开启，则 `build_retriever` 复用其 `pool` 与内部 `embedding`（或重建 `EmbeddingModel(settings)`），调 `run_breakdown`，打印 `vec_only` / `bm25_only` 两张表，与 fused 聚合表并列，然后 return（不做 gate）。

- [ ] **Step 3: 冒烟验证**

Run: `python -m rag.eval.run --breakdown`
Expected: 打印 fused / vec_only / bm25_only 三组指标，进程 0 退出。

- [ ] **Step 4: 提交**

```bash
git add rag/eval/harness.py rag/eval/run.py
git commit -m "feat(eval): --breakdown 分路诊断(可选)"
```

---

## Self-Review

**Spec coverage：**
- 目录结构 → Task 1/3/4/5/6 逐文件落地（`metrics.py`/`harness.py`/`seed_corpus.py`/`generate_golden.py`/`run.py`/`baseline.json`/`datasets/*`/`tests/test_eval_retrieval.py`）。✅
- 数据格式（`gold_snippets`、片段包含命中、归一化单一实现）→ Task 1 `normalize` + `_hit_indices`，Global Constraints 锁定单一实现。✅
- 指标 hit@k/recall@k/mrr/ndcg@k → Task 1 `evaluate_query`。✅
- 冻结独立语料 + 独立 KB → Task 4 `corpus.txt` + `EVAL_KB_ID`。✅
- golden 半自动（LLM 造 → 人工筛）→ Task 5。✅
- 门禁（run + baseline 对比 + `--update-baseline`，容差 3%，核心 recall@5/mrr）→ Task 2 `gate` + Task 6 `run.py`。✅
- pytest `@pytest.mark.eval` 默认不跑 → Task 6 marker + `addopts`。✅
- 可选 `--breakdown` → Task 7。✅
- 生成层预留（harness 指标无关）→ `run_eval` 接收 `evaluate_query` 结果、`gate` 按 key 通用，未来换 judge 不改骨架（设计层面满足，无需本期代码）。✅

**Placeholder scan：** 无 TBD/TODO；IO 类任务（seed/generate/run）给出完整代码 + 明确运行命令与预期输出；人工筛步骤显式标注为人工。✅

**Type consistency：** `GoldenItem(id,query,gold_snippets)` 在 harness 定义、test 与 run_eval 一致；`evaluate_query` 返回键 `hit@k/recall@k/mrr/ndcg@k` 与 `gate` 默认 keys `("recall@5","mrr")` 对齐（KS 含 5）；`EVAL_KB_ID`/`DATASETS_DIR`/`EVAL_DIR` 全程从 `rag.eval` 导入，无漂移。✅
```
