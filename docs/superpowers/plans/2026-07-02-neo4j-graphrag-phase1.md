# Neo4j GraphRAG 实体抽取(阶段一)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已有的单 chunk 实体抽取 `extract_entities` 接入入库链路,通过一个独立 arq 任务把整篇文档的实体/关系写入 Neo4j 知识图谱(GraphRAG 雏形),开关默认关闭、失败 best-effort 不拖垮向量入库。

**Architecture:** 文档向量入库(`ingest_document`)成功后,若开关开启则投递独立任务 `extract_document_entities`;该任务读回该文档所有 chunk → 逐 chunk 调 LLM 抽取 → Python 侧预聚合(同名合并、方向规范化)→ 幂等清理旧图数据(2 条批量 Cypher)→ 批量 `MERGE` 写入 Neo4j。向量入库状态 `status` 与图状态 `graph_status` 完全独立。

**Tech Stack:** Python ≥3.12、neo4j async driver、arq、psycopg3(async pool)、pydantic-settings、pytest + pytest-asyncio。

设计依据:`docs/superpowers/specs/2026-07-02-neo4j-graphrag-entity-extraction-design.md`(阶段一部分)。

## Global Constraints

- Python `>=3.12`;异步优先,neo4j 用原生 async driver(不套 `to_thread`)。
- **config 字段一律大写**(`Settings.NEO4J_URI` 等);代码访问用大写(`settings.NEO4J_URI`)。pydantic `case_sensitive=False` 只影响环境变量读取,不影响属性名。
- Neo4j 镜像 `neo4j:5-community`,**默认不含 APOC** → 列表去重用纯 Cypher 列表推导,禁止 `apoc.*`。
- 单知识库:去重键为 `name`(**无 `kb_id`**)。
- 溯源标识 `chunk_uid = f"{document_id}:{chunk_index}"`。
- 无向关系:预聚合阶段按 `name` 字典序规范化(`source <= target`),写入用无向 `MERGE (a)-[r:RELATES]-(b)`。
- 幂等清理:先边后节点,2 条批量 Cypher;边遍历用**有向** `()-[r:RELATES]->()` 避免重复匹配。
- best-effort:阶段一图抽取失败置 `graph_status='failed'` 记日志,**不 re-raise、不自动重试、不缓存**(阶段二再加)。
- 测试:`pytest-asyncio` 已配 `asyncio_mode=auto`(async 测试函数直接写,无需装饰器);触碰真实 PG/Neo4j 的测试标 `@pytest.mark.integration`(默认不跑)。
- 每个 commit 信息结尾加:`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## 文件结构

**新增**
- `rag/db/neo4j.py` —— `create_neo4j_driver` + `ensure_graph_constraints`
- `rag/graph/__init__.py` —— 空包标识
- `rag/graph/aggregate.py` —— 纯函数:多 chunk 抽取结果 → 待写入实体/关系(去重、方向规范化)
- `rag/graph/store.py` —— `write_graph` + `purge_document`(Cypher)
- `rag/graph/pipeline.py` —— `extract_document_entities` arq 任务
- `alembic/versions/0004_graph_status.py` —— documents 加 `graph_status` + `graph_error`
- 测试:`tests/test_graph_aggregate.py`、`tests/test_graph_store.py`、`tests/test_graph_pipeline.py`、`tests/test_db_neo4j.py`

**修改**
- `pyproject.toml` —— 加 `neo4j` 依赖
- `docker-compose.yaml` —— 加 `neo4j` 服务与卷
- `rag/config.py` —— `ENABLE_ENTITY_EXTRACTION` + `NEO4J_*` + 条件必填
- `rag/db/__init__.py` —— 导出 neo4j 封装
- `rag/document/store.py` —— `get_chunks_for_graph` + `claim_graph_processing` + `set_graph_status`
- `rag/document/pipeline.py` —— 入库成功后投递图任务
- `rag/worker/main.py` —— `WorkerCtx` 注入 `neo4j`+`llm`,注册任务,startup/shutdown

---

## Task 1: 加 neo4j 依赖与 docker-compose 服务

**Files:**
- Modify: `pyproject.toml`(dependencies 数组)
- Modify: `docker-compose.yaml`

**Interfaces:**
- Produces: 可 `import neo4j`;本地 `docker compose up neo4j` 起 Bolt(7687)/Browser(7474)。

- [ ] **Step 1: 加依赖到 pyproject.toml**

在 `dependencies` 数组末尾(`"langchain-core>=1.4.6",` 之后)加一行:

```toml
    "neo4j>=5.28",
```

- [ ] **Step 2: 安装依赖**

Run: `uv sync`
Expected: 成功安装 neo4j;`python -c "import neo4j; print(neo4j.__version__)"` 打印版本(≥5.28)。

- [ ] **Step 3: 加 neo4j 服务到 docker-compose.yaml**

在 `minio:` 服务块之后、`volumes:` 之前插入:

```yaml
  neo4j:
    image: neo4j:5-community
    container_name: rag-neo4j
    environment:
      NEO4J_AUTH: neo4j/neo4j_pass
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    restart: unless-stopped
```

并在 `volumes:` 段落加一行 `neo4j_data:`(与 `minio_data:` 同级):

```yaml
volumes:
  redis_data:
  pg_data:
  minio_data:
  neo4j_data:
```

- [ ] **Step 4: 校验 compose 文件合法**

Run: `docker compose config --quiet`
Expected: 无输出、退出码 0(YAML 合法)。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock docker-compose.yaml
git commit -m "feat: 加 neo4j 依赖与 docker-compose 服务

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: config 加开关与 NEO4J_* 配置(条件必填)

**Files:**
- Modify: `rag/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.ENABLE_ENTITY_EXTRACTION: bool`、`Settings.NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE: str`;`check_required()` 在开关开启且缺 NEO4J_* 时抛 `ValueError`。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_config.py`:

```python
def test_settings_has_graph_defaults():
    s = Settings()
    assert s.ENABLE_ENTITY_EXTRACTION is False
    assert s.NEO4J_URI == "bolt://localhost:7687"
    assert s.NEO4J_USER == "neo4j"
    assert s.NEO4J_DATABASE == "neo4j"


def test_check_required_ignores_neo4j_when_extraction_disabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("ENABLE_ENTITY_EXTRACTION", "false")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    Settings().check_required()  # 不抛


def test_check_required_needs_neo4j_when_extraction_enabled(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_URL", "u")
    monkeypatch.setenv("EMBEDDING_KEY", "ek")
    monkeypatch.setenv("EMBEDDING_URL", "eu")
    monkeypatch.setenv("ENABLE_ENTITY_EXTRACTION", "true")
    monkeypatch.setenv("NEO4J_PASSWORD", "")
    import pytest
    with pytest.raises(ValueError):
        Settings().check_required()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_config.py::test_settings_has_graph_defaults tests/test_config.py::test_check_required_needs_neo4j_when_extraction_enabled -v`
Expected: FAIL(`AttributeError: ... ENABLE_ENTITY_EXTRACTION` / 无条件校验)。

- [ ] **Step 3: 加配置字段**

在 `rag/config.py` 的 `# ── LLM ──` 块之后、`# ── Embedding ──` 之前插入:

```python
    # ── Graph / 实体抽取 ──
    ENABLE_ENTITY_EXTRACTION: bool = False
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_pass"
    NEO4J_DATABASE: str = "neo4j"
```

- [ ] **Step 4: 加条件必填校验**

把 `check_required` 方法体替换为(在原有 `_REQUIRED_FIELDS` 校验后追加 NEO4J 条件校验):

```python
    def check_required(self) -> None:
        """校验必填配置项已设置；未设置则抛 ValueError，启动即失败。"""
        missing = [f for f in self._REQUIRED_FIELDS if not getattr(self, f)]
        if self.ENABLE_ENTITY_EXTRACTION:
            missing += [
                f for f in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
                if not getattr(self, f)
            ]
        if missing:
            raise ValueError(
                f"缺少必要配置: {', '.join(missing)}，请检查 .env 文件"
            )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_config.py -v`
Expected: 新增 3 个测试 PASS(旧的小写用例若已失败与本任务无关,不修)。

- [ ] **Step 6: Commit**

```bash
git add rag/config.py tests/test_config.py
git commit -m "feat: config 加实体抽取开关与 NEO4J_* (条件必填)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Alembic 0004 —— documents 加 graph_status / graph_error

**Files:**
- Create: `alembic/versions/0004_graph_status.py`

**Interfaces:**
- Produces: `documents.graph_status TEXT DEFAULT 'pending'`、`documents.graph_error TEXT`;部分索引加速按 graph_status 扫描。

- [ ] **Step 1: 写迁移文件**

```python
"""add graph_status to documents for entity extraction

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-02
"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT 'pending'
        """
    )
    op.execute(
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS graph_error TEXT
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_graph_status
            ON documents (graph_status, updated_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_documents_graph_status")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS graph_error")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS graph_status")
```

- [ ] **Step 2: 校验迁移可加载(离线)**

Run: `python -c "import importlib.util as u; s=u.spec_from_file_location('m','alembic/versions/0004_graph_status.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(m.revision, m.down_revision)"`
Expected: 打印 `0004 0003`。

- [ ] **Step 3: 应用迁移(需 docker PG,integration)**

Run: `alembic upgrade head`
Expected: 迁移到 `0004` 无报错。若本机未起 PG,跳过此步,留待集成环境。

- [ ] **Step 4: Commit**

```bash
git add alembic/versions/0004_graph_status.py
git commit -m "feat: alembic 0004 — documents 加 graph_status/graph_error

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Neo4j 连接封装 rag/db/neo4j.py

**Files:**
- Create: `rag/db/neo4j.py`
- Modify: `rag/db/__init__.py`
- Test: `tests/test_db_neo4j.py`

**Interfaces:**
- Consumes: `Settings.NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD/NEO4J_DATABASE`。
- Produces:
  - `create_neo4j_driver(settings) -> neo4j.AsyncDriver`
  - `async ensure_graph_constraints(driver, database: str) -> None`(建 `entity_key` 唯一约束,幂等)

- [ ] **Step 1: 写失败测试**

`tests/test_db_neo4j.py`:

```python
from types import SimpleNamespace

import rag.db.neo4j as n


def test_create_driver_passes_uri_and_auth(monkeypatch):
    captured = {}

    class _FakeDriver:
        pass

    def fake_driver(uri, auth=None):
        captured["uri"] = uri
        captured["auth"] = auth
        return _FakeDriver()

    monkeypatch.setattr(n.AsyncGraphDatabase, "driver", staticmethod(fake_driver))
    settings = SimpleNamespace(
        NEO4J_URI="bolt://x:7687", NEO4J_USER="neo4j",
        NEO4J_PASSWORD="pw", NEO4J_DATABASE="neo4j",
    )
    drv = n.create_neo4j_driver(settings)
    assert isinstance(drv, _FakeDriver)
    assert captured["uri"] == "bolt://x:7687"
    assert captured["auth"] == ("neo4j", "pw")


def test_db_package_exports_neo4j_helpers():
    import rag.db as db
    assert hasattr(db, "create_neo4j_driver")
    assert hasattr(db, "ensure_graph_constraints")
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_db_neo4j.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.db.neo4j` 或 `AttributeError`)。

- [ ] **Step 3: 实现 rag/db/neo4j.py**

```python
from neo4j import AsyncDriver, AsyncGraphDatabase

from rag.config import Settings

# 单库去重键:实体名唯一。多库隔离需改为 (kb_id, name) 复合约束(见设计 §12)。
_ENTITY_CONSTRAINT = (
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
)


def create_neo4j_driver(settings: Settings) -> AsyncDriver:
    """创建 Neo4j 异步 driver（原生 async，无需 to_thread）。"""
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


async def ensure_graph_constraints(driver: AsyncDriver, database: str) -> None:
    """幂等建约束；worker 启动时调用一次。"""
    async with driver.session(database=database) as session:
        await session.run(_ENTITY_CONSTRAINT)
```

- [ ] **Step 4: 导出到 rag/db/__init__.py**

把 `rag/db/__init__.py` 改为:

```python
# db 层已迁移至异步实现（psycopg3 AsyncConnectionPool + redis.asyncio）
# 使用 create_pg_pool / get_cursor / create_redis_client 替代旧同步接口

from rag.db.neo4j import create_neo4j_driver, ensure_graph_constraints
from rag.db.postgres import create_pg_pool, get_cursor
from rag.db.redis import create_redis_client


__all__ = [
    "create_pg_pool",
    "get_cursor",
    "create_redis_client",
    "create_neo4j_driver",
    "ensure_graph_constraints",
]
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_db_neo4j.py -v`
Expected: 2 个测试 PASS。

- [ ] **Step 6: Commit**

```bash
git add rag/db/neo4j.py rag/db/__init__.py tests/test_db_neo4j.py
git commit -m "feat: Neo4j 异步 driver 封装与约束初始化

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: 预聚合纯函数 rag/graph/aggregate.py

**Files:**
- Create: `rag/graph/__init__.py`(空)
- Create: `rag/graph/aggregate.py`
- Test: `tests/test_graph_aggregate.py`

**Interfaces:**
- Consumes: `rag.document.entity_extraction.ExtractionResult`(含 `entities: list[Entity]`,`relationships: list[Relationship]`;`Entity(name, type, description)`;`Relationship(source, target, keywords, description)`)。
- Produces:
  - `AggEntity = {"name": str, "type": str, "chunk_ids": list[str]}`
  - `AggRelation = {"source": str, "target": str, "keywords": list[str]}`
  - `aggregate(chunks: list[tuple[str, ExtractionResult]]) -> tuple[list[AggEntity], list[AggRelation]]`
    入参每项为 `(chunk_uid, ExtractionResult)`。实体按 `name` 合并(chunk_ids 并集、type 众数平票取首现);关系按规范化方向 `(min(s,t), max(s,t))` 合并(keywords 并集);丢弃两端不在实体集合中的关系。

- [ ] **Step 1: 写失败测试**

`tests/test_graph_aggregate.py`:

```python
from rag.document.entity_extraction import Entity, ExtractionResult, Relationship
from rag.graph.aggregate import aggregate


def _res(entities, rels):
    return ExtractionResult(entities=entities, relationships=rels)


def test_merges_same_name_across_chunks_unions_chunk_ids():
    c0 = _res([Entity("孙悟空", "Person", "d")], [])
    c1 = _res([Entity("孙悟空", "Person", "d2")], [])
    ents, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(ents) == 1
    assert ents[0]["name"] == "孙悟空"
    assert sorted(ents[0]["chunk_ids"]) == ["doc:0", "doc:1"]


def test_type_conflict_uses_majority_then_first():
    c0 = _res([Entity("X", "Person", "d")], [])
    c1 = _res([Entity("X", "Creature", "d")], [])
    c2 = _res([Entity("X", "Person", "d")], [])
    ents, _ = aggregate([("doc:0", c0), ("doc:1", c1), ("doc:2", c2)])
    assert ents[0]["type"] == "Person"  # 众数


def test_relation_direction_normalized_and_keywords_unioned():
    c0 = _res(
        [Entity("唐僧", "Person", ""), Entity("孙悟空", "Person", "")],
        [Relationship("孙悟空", "唐僧", "师徒", "")],
    )
    c1 = _res(
        [Entity("唐僧", "Person", ""), Entity("孙悟空", "Person", "")],
        [Relationship("唐僧", "孙悟空", "取经", "")],
    )
    _, rels = aggregate([("doc:0", c0), ("doc:1", c1)])
    assert len(rels) == 1
    r = rels[0]
    assert (r["source"], r["target"]) == ("唐僧", "孙悟空")  # 字典序规范化
    assert sorted(r["keywords"]) == ["取经", "师徒"]


def test_drops_relation_with_unknown_endpoint():
    c0 = _res([Entity("A", "Person", "")], [Relationship("A", "B", "k", "")])
    ents, rels = aggregate([("doc:0", c0)])
    assert [e["name"] for e in ents] == ["A"]
    assert rels == []
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_graph_aggregate.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.graph.aggregate`)。

- [ ] **Step 3: 建空包文件**

创建 `rag/graph/__init__.py`,内容:

```python
```

(空文件即可。)

- [ ] **Step 4: 实现 aggregate**

`rag/graph/aggregate.py`:

```python
from __future__ import annotations

from collections import Counter

from rag.document.entity_extraction import ExtractionResult


def aggregate(
    chunks: list[tuple[str, ExtractionResult]],
) -> tuple[list[dict], list[dict]]:
    """把整篇文档各 chunk 的抽取结果预聚合为待写入 Neo4j 的实体/关系。

    - 实体按 name 合并：chunk_ids 取并集；type 取众数（平票取首次出现）。
    - 关系按规范化方向 (min, max) 合并：keywords 取并集。
    - 丢弃两端不在实体集合内的关系。
    每项入参为 (chunk_uid, ExtractionResult)。
    """
    ent_chunks: dict[str, set[str]] = {}
    ent_types: dict[str, list[str]] = {}
    for chunk_uid, result in chunks:
        for ent in result.entities:
            name = ent.name
            ent_chunks.setdefault(name, set()).add(chunk_uid)
            ent_types.setdefault(name, []).append(ent.type or "其他")

    entities: list[dict] = []
    for name, chunk_ids in ent_chunks.items():
        # Counter.most_common 对平票保留首次插入顺序 → 众数、平票取首现
        top_type = Counter(ent_types[name]).most_common(1)[0][0]
        entities.append(
            {"name": name, "type": top_type, "chunk_ids": sorted(chunk_ids)}
        )

    valid = set(ent_chunks)
    rel_keywords: dict[tuple[str, str], set[str]] = {}
    for _chunk_uid, result in chunks:
        for rel in result.relationships:
            s, t = rel.source, rel.target
            if s == t or s not in valid or t not in valid:
                continue
            key = (s, t) if s <= t else (t, s)
            kws = rel_keywords.setdefault(key, set())
            if rel.keywords:
                for kw in str(rel.keywords).split(","):
                    kw = kw.strip()
                    if kw:
                        kws.add(kw)

    relations = [
        {"source": s, "target": t, "keywords": sorted(kws)}
        for (s, t), kws in rel_keywords.items()
    ]
    return entities, relations
```

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_graph_aggregate.py -v`
Expected: 4 个测试 PASS。

- [ ] **Step 6: Commit**

```bash
git add rag/graph/__init__.py rag/graph/aggregate.py tests/test_graph_aggregate.py
git commit -m "feat: 实体/关系预聚合纯函数(同名合并+方向规范化)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 图写入与幂等清理 rag/graph/store.py

**Files:**
- Create: `rag/graph/store.py`
- Test: `tests/test_graph_store.py`(integration,需 docker neo4j)

**Interfaces:**
- Consumes: `neo4j.AsyncDriver`;Task 5 的 `AggEntity`/`AggRelation` 字典。
- Produces:
  - `async purge_document(driver, database: str, document_id: str) -> None`
  - `async write_graph(driver, database: str, document_id: str, entities: list[dict], relations: list[dict]) -> None`

- [ ] **Step 1: 写 integration 测试**

`tests/test_graph_store.py`:

```python
import os

import pytest

pytestmark = pytest.mark.integration

neo4j = pytest.importorskip("neo4j")
from rag.graph.store import purge_document, write_graph  # noqa: E402


@pytest.fixture
async def driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pw = os.getenv("NEO4J_PASSWORD", "neo4j_pass")
    drv = neo4j.AsyncGraphDatabase.driver(uri, auth=(user, pw))
    async with drv.session() as s:
        await s.run("MATCH (n:Entity) DETACH DELETE n")
    yield drv
    async with drv.session() as s:
        await s.run("MATCH (n:Entity) DETACH DELETE n")
    await drv.close()


async def _counts(driver):
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity) WITH count(e) AS ec "
            "OPTIONAL MATCH ()-[r:RELATES]->() RETURN ec, count(r) AS rc"
        )
        rec = await r.single()
        return rec["ec"], rec["rc"]


async def test_write_then_rerun_is_idempotent(driver):
    ents = [
        {"name": "孙悟空", "type": "Person", "chunk_ids": ["d1:0"]},
        {"name": "唐僧", "type": "Person", "chunk_ids": ["d1:0"]},
    ]
    rels = [{"source": "孙悟空", "target": "唐僧", "keywords": ["师徒"]}]

    await purge_document(driver, "neo4j", "d1")
    await write_graph(driver, "neo4j", "d1", ents, rels)
    assert await _counts(driver) == (2, 1)

    # 重跑同一文档:先 purge 再写,数量不翻倍
    await purge_document(driver, "neo4j", "d1")
    await write_graph(driver, "neo4j", "d1", ents, rels)
    assert await _counts(driver) == (2, 1)


async def test_shared_entity_not_deleted_on_other_doc_purge(driver):
    await write_graph(
        driver, "neo4j", "d1",
        [{"name": "孙悟空", "type": "Person", "chunk_ids": ["d1:0"]}], [],
    )
    await write_graph(
        driver, "neo4j", "d2",
        [{"name": "孙悟空", "type": "Person", "chunk_ids": ["d2:0"]}], [],
    )
    # 孙悟空被 d1、d2 共享,purge d1 后节点仍在,只剩 d2 的 chunk_id
    await purge_document(driver, "neo4j", "d1")
    async with driver.session() as s:
        r = await s.run("MATCH (e:Entity {name:'孙悟空'}) RETURN e.chunk_ids AS c, e.doc_ids AS d")
        rec = await r.single()
    assert rec["c"] == ["d2:0"]
    assert rec["d"] == ["d2"]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_graph_store.py -v -m integration`
Expected: FAIL(`ModuleNotFoundError: rag.graph.store`)。若无 neo4j 容器,则测试 error/skip —— 先确保能起容器:`docker compose up -d neo4j`。

- [ ] **Step 3: 实现 rag/graph/store.py**

```python
from __future__ import annotations

from neo4j import AsyncDriver

# 纯 Cypher 列表去重(neo4j:5-community 无 APOC):[x IN old WHERE NOT x IN new] + new
_WRITE_ENTITIES = """
UNWIND $entities AS ent
MERGE (e:Entity {name: ent.name})
ON CREATE SET e.type = ent.type
SET e.chunk_ids = [x IN coalesce(e.chunk_ids, []) WHERE NOT x IN ent.chunk_ids] + ent.chunk_ids,
    e.doc_ids   = [x IN coalesce(e.doc_ids, []) WHERE x <> $doc] + [$doc]
"""

_WRITE_RELATIONS = """
UNWIND $relations AS rel
MATCH (a:Entity {name: rel.source})
MATCH (b:Entity {name: rel.target})
MERGE (a)-[r:RELATES]-(b)
SET r.keywords = [x IN coalesce(r.keywords, []) WHERE NOT x IN rel.keywords] + rel.keywords,
    r.doc_ids  = [x IN coalesce(r.doc_ids, []) WHERE x <> $doc] + [$doc]
"""

_PURGE_EDGES = """
MATCH ()-[r:RELATES]->()
WHERE $doc IN r.doc_ids
SET r.doc_ids = [d IN r.doc_ids WHERE d <> $doc]
WITH r WHERE size(r.doc_ids) = 0
DELETE r
"""

_PURGE_NODES = """
MATCH (e:Entity)
WHERE $doc IN e.doc_ids
SET e.chunk_ids = [c IN e.chunk_ids WHERE NOT c STARTS WITH $prefix],
    e.doc_ids   = [d IN e.doc_ids WHERE d <> $doc]
WITH e WHERE size(e.doc_ids) = 0
DETACH DELETE e
"""


async def purge_document(driver: AsyncDriver, database: str, document_id: str) -> None:
    """重跑前清理该文档在图中的旧贡献(先边后节点,一个事务完成)。"""
    prefix = f"{document_id}:"

    async def _tx(tx):
        await tx.run(_PURGE_EDGES, doc=document_id)
        await tx.run(_PURGE_NODES, doc=document_id, prefix=prefix)

    async with driver.session(database=database) as session:
        await session.execute_write(_tx)


async def write_graph(
    driver: AsyncDriver,
    database: str,
    document_id: str,
    entities: list[dict],
    relations: list[dict],
) -> None:
    """批量 MERGE 写入实体与关系(实体先于关系;一个事务完成)。"""
    if not entities:
        return

    async def _tx(tx):
        await tx.run(_WRITE_ENTITIES, entities=entities, doc=document_id)
        if relations:
            await tx.run(_WRITE_RELATIONS, relations=relations, doc=document_id)

    async with driver.session(database=database) as session:
        await session.execute_write(_tx)
```

- [ ] **Step 4: 起容器并运行确认通过**

Run:
```bash
docker compose up -d neo4j
pytest tests/test_graph_store.py -v -m integration
```
Expected: 2 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add rag/graph/store.py tests/test_graph_store.py
git commit -m "feat: 图写入(批量 MERGE)与幂等清理(批量 Cypher)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: document/store.py 加 chunk 读取与 graph_status 状态函数

**Files:**
- Modify: `rag/document/store.py`
- Test: `tests/test_graph_store_pg.py`(integration,需 docker PG)

**Interfaces:**
- Consumes: `psycopg_pool.AsyncConnectionPool`、`get_cursor`(已 import 在 store.py)。
- Produces:
  - `async get_chunks_for_graph(pool, document_id) -> list[dict]`,每项 `{"chunk_index": int, "text": str, "title": str | None}`,按 chunk_index 升序。
  - `async claim_graph_processing(pool, document_id) -> bool`(graph_status pending/failed → processing,原子)
  - `async set_graph_status(pool, document_id, status, *, error=None) -> None`

- [ ] **Step 1: 写 integration 测试**

`tests/test_graph_store_pg.py`:

```python
import uuid

import pytest

pytestmark = pytest.mark.integration

from rag.config import get_settings  # noqa: E402
from rag.db import create_pg_pool  # noqa: E402
from rag.document import store  # noqa: E402


@pytest.fixture
async def pool():
    p = await create_pg_pool(get_settings())
    yield p
    await p.close()


async def _mk_doc(pool) -> str:
    return await store.create_document(
        pool, knowledge_base_id="00000000-0000-0000-0000-000000000001",
        filename="f.txt", content_type="text/plain", size_bytes=1,
        content_hash="h" + uuid.uuid4().hex, object_key="k" + uuid.uuid4().hex,
    )


async def test_claim_graph_processing_atomic(pool):
    doc_id = await _mk_doc(pool)
    assert await store.claim_graph_processing(pool, doc_id) is True
    # 已 processing,再领取失败
    assert await store.claim_graph_processing(pool, doc_id) is False


async def test_set_graph_status_and_error(pool):
    doc_id = await _mk_doc(pool)
    await store.set_graph_status(pool, doc_id, "failed", error="boom")
    doc = await store.get_document(pool, doc_id)
    assert doc["graph_status"] == "failed"
    assert doc["graph_error"] == "boom"


async def test_get_chunks_for_graph_orders_and_extracts_title(pool):
    doc_id = await _mk_doc(pool)
    await store.store_chunks_and_complete(
        pool, doc_id, "00000000-0000-0000-0000-000000000001",
        [
            (1, "second", [0.0] * 1024, {"title": "第二章"}),
            (0, "first", [0.0] * 1024, {}),
        ],
    )
    rows = await store.get_chunks_for_graph(pool, doc_id)
    assert [r["chunk_index"] for r in rows] == [0, 1]
    assert rows[0]["title"] is None
    assert rows[1]["title"] == "第二章"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_graph_store_pg.py -v -m integration`
Expected: FAIL(`AttributeError: module 'rag.document.store' has no attribute 'get_chunks_for_graph'`)。

- [ ] **Step 3: 实现三个函数**

追加到 `rag/document/store.py` 末尾:

```python
async def get_chunks_for_graph(
    pool: AsyncConnectionPool, document_id: str
) -> list[dict]:
    """读取文档所有 chunk 供实体抽取:chunk_index、text、章节标题(metadata.title)。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            SELECT chunk_index,
                   text,
                   metadata ->> 'title' AS title
            FROM document_chunks
            WHERE document_id = %(id)s
            ORDER BY chunk_index
            """,
            {"id": document_id},
        )
        return await cur.fetchall()


async def claim_graph_processing(
    pool: AsyncConnectionPool, document_id: str
) -> bool:
    """原子领取图抽取:graph_status pending/failed → processing,返回是否成功。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET graph_status = 'processing', updated_at = now()
            WHERE id = %(id)s AND graph_status IN ('pending', 'failed')
            RETURNING id
            """,
            {"id": document_id},
        )
        return await cur.fetchone() is not None


async def set_graph_status(
    pool: AsyncConnectionPool,
    document_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    """设置 graph_status(与向量入库 status 独立)。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            """
            UPDATE documents
            SET graph_status = %(s)s, graph_error = %(e)s, updated_at = now()
            WHERE id = %(id)s
            """,
            {"s": status, "e": error, "id": document_id},
        )
```

- [ ] **Step 4: 运行确认通过(需 PG + 已迁移到 0004)**

Run:
```bash
docker compose up -d postgres
alembic upgrade head
pytest tests/test_graph_store_pg.py -v -m integration
```
Expected: 3 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add rag/document/store.py tests/test_graph_store_pg.py
git commit -m "feat: store 加 get_chunks_for_graph 与 graph_status 领取/置位

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: 抽取任务编排 rag/graph/pipeline.py

**Files:**
- Create: `rag/graph/pipeline.py`
- Test: `tests/test_graph_pipeline.py`

**Interfaces:**
- Consumes: `WorkerCtx`(dict,含 `pg`、`neo4j`、`llm`、`settings`);`store.get_chunks_for_graph/claim_graph_processing/set_graph_status`;`extract_entities`;`aggregate`;`purge_document`/`write_graph`。
- Produces: `async extract_document_entities(ctx, document_id) -> None`(arq 任务)。

- [ ] **Step 1: 写失败测试**

`tests/test_graph_pipeline.py`:

```python
from types import SimpleNamespace

import rag.graph.pipeline as gp
from rag.document.entity_extraction import Entity, ExtractionResult, Relationship


def _ctx():
    return {
        "pg": None, "neo4j": object(),
        "settings": SimpleNamespace(NEO4J_DATABASE="neo4j"),
        "llm": object(),
    }


async def test_skips_when_not_claimed(monkeypatch):
    async def fake_claim(pool, doc_id):
        return False

    called = []
    async def fail(*a, **k):
        called.append(a)
        raise AssertionError("未领取不应继续")

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fail)

    await gp.extract_document_entities(_ctx(), "d1")
    assert called == []


async def test_happy_path_writes_graph_and_marks_done(monkeypatch):
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_chunks(pool, doc_id):
        return [
            {"chunk_index": 0, "text": "孙悟空拜唐僧为师", "title": "第一回"},
        ]

    async def fake_extract(llm, text, *, chapter_context=None, **kw):
        return ExtractionResult(
            entities=[Entity("孙悟空", "Person", ""), Entity("唐僧", "Person", "")],
            relationships=[Relationship("孙悟空", "唐僧", "师徒", "")],
        )

    async def fake_purge(driver, database, doc_id):
        events.append(("purge", database, doc_id))

    async def fake_write(driver, database, doc_id, entities, relations):
        events.append(("write", doc_id, [e["name"] for e in entities], len(relations)))

    async def fake_status(pool, doc_id, status, error=None):
        events.append(("status", status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", fake_chunks)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)
    monkeypatch.setattr(gp, "extract_entities", fake_extract)
    monkeypatch.setattr(gp, "purge_document", fake_purge)
    monkeypatch.setattr(gp, "write_graph", fake_write)

    await gp.extract_document_entities(_ctx(), "d1")

    assert ("purge", "neo4j", "d1") in events
    write_evt = [e for e in events if e[0] == "write"][0]
    assert sorted(write_evt[2]) == ["唐僧", "孙悟空"]
    assert write_evt[3] == 1
    assert ("status", "done", None) in events


async def test_failure_marks_failed_and_not_reraise(monkeypatch):
    events = []

    async def fake_claim(pool, doc_id):
        return True

    async def boom(pool, doc_id):
        raise RuntimeError("db down")

    async def fake_status(pool, doc_id, status, error=None):
        events.append((status, error))

    monkeypatch.setattr(gp.store, "claim_graph_processing", fake_claim)
    monkeypatch.setattr(gp.store, "get_chunks_for_graph", boom)
    monkeypatch.setattr(gp.store, "set_graph_status", fake_status)

    # 阶段一:不 re-raise
    await gp.extract_document_entities(_ctx(), "d1")
    assert events and events[0][0] == "failed" and "db down" in events[0][1]
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_graph_pipeline.py -v`
Expected: FAIL(`ModuleNotFoundError: rag.graph.pipeline`)。

- [ ] **Step 3: 实现 rag/graph/pipeline.py**

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from rag.common.logging import get_logger
from rag.document import store
from rag.document.entity_extraction import extract_entities
from rag.graph.aggregate import aggregate
from rag.graph.store import purge_document, write_graph

if TYPE_CHECKING:
    from rag.worker.main import WorkerCtx

logger = get_logger()

# demo 为中文语料(如西游记);显式指定中文输出,避免抽取默认语言。
_LANGUAGE = "中文"


async def extract_document_entities(ctx: "WorkerCtx", document_id: str) -> None:
    """独立 arq 任务:抽取整篇文档实体/关系并写入 Neo4j。

    best-effort(阶段一):失败置 graph_status=failed 记日志,不 re-raise、不重试。
    与向量入库 status 独立,不影响已 done 的检索能力。
    """
    pool = ctx["pg"]
    driver = ctx["neo4j"]
    llm = ctx["llm"]
    database = ctx["settings"].NEO4J_DATABASE

    if not await store.claim_graph_processing(pool, document_id):
        logger.info("图抽取非待处理状态或已被领取,跳过: %s", document_id)
        return

    try:
        chunks = await store.get_chunks_for_graph(pool, document_id)
        extracted: list[tuple[str, object]] = []
        for row in chunks:
            chunk_uid = f"{document_id}:{row['chunk_index']}"
            result = await extract_entities(
                llm, row["text"],
                chapter_context=row.get("title"),
                language=_LANGUAGE,
            )
            extracted.append((chunk_uid, result))

        entities, relations = aggregate(extracted)

        await purge_document(driver, database, document_id)
        await write_graph(driver, database, document_id, entities, relations)

        await store.set_graph_status(pool, document_id, "done")
        logger.info(
            "图抽取完成: %s 实体 %d 关系 %d",
            document_id, len(entities), len(relations),
        )
    except Exception as e:  # noqa: BLE001 - best-effort,阶段一不 re-raise
        logger.exception("图抽取失败: %s", document_id)
        await store.set_graph_status(pool, document_id, "failed", error=str(e)[:500])
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_graph_pipeline.py -v`
Expected: 3 个测试 PASS。

- [ ] **Step 5: Commit**

```bash
git add rag/graph/pipeline.py tests/test_graph_pipeline.py
git commit -m "feat: extract_document_entities 抽取任务编排(best-effort)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: worker 注入 neo4j+llm 并注册任务

**Files:**
- Modify: `rag/worker/main.py`
- Test: `tests/test_worker.py`

**Interfaces:**
- Consumes: `create_neo4j_driver`、`ensure_graph_constraints`、`NormalModel`、`extract_document_entities`。
- Produces: `WorkerCtx` 增加 `neo4j: AsyncDriver`、`llm: ChatModel`;`WorkerSettings.functions` 含 `extract_document_entities`。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_worker.py`:

```python
def test_worker_registers_extract_function():
    from rag.graph.pipeline import extract_document_entities
    from rag.worker.main import WorkerSettings
    assert extract_document_entities in WorkerSettings.functions
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_worker.py::test_worker_registers_extract_function -v`
Expected: FAIL(`extract_document_entities not in functions`)。

- [ ] **Step 3: 修改 worker/main.py**

在 import 区加(与现有 import 同区):

```python
from neo4j import AsyncDriver

from rag.db.neo4j import create_neo4j_driver, ensure_graph_constraints
from rag.graph.pipeline import extract_document_entities
from rag.models.base import ChatModel
from rag.models.normal import NormalModel
```

`WorkerCtx` 增加两个字段:

```python
class WorkerCtx(TypedDict):
    settings: Settings
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    neo4j: AsyncDriver
    llm: ChatModel
```

`on_startup` 末尾追加(建 driver/llm,并幂等建约束):

```python
    ctx["neo4j"] = create_neo4j_driver(settings)
    ctx["llm"] = NormalModel(settings)
    await ensure_graph_constraints(ctx["neo4j"], settings.NEO4J_DATABASE)
```

`on_shutdown` 增加关闭 driver:

```python
async def on_shutdown(ctx: dict) -> None:
    await ctx["pg"].close()
    await ctx["neo4j"].close()
```

`WorkerSettings.functions` 改为:

```python
    functions = [ingest_document, extract_document_entities]
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_worker.py -v`
Expected: `test_worker_registers_extract_function` PASS;既有 `test_worker_registers_ingest_function` 仍 PASS。

- [ ] **Step 5: Commit**

```bash
git add rag/worker/main.py tests/test_worker.py
git commit -m "feat: worker 注入 neo4j+llm 并注册抽取任务

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: pipeline 投递图抽取任务

**Files:**
- Modify: `rag/document/pipeline.py`
- Test: `tests/test_document_pipeline.py`

**Interfaces:**
- Consumes: arq worker `ctx["redis"]`(ArqRedis,运行时由 arq 注入)的 `enqueue_job`;`settings.ENABLE_ENTITY_EXTRACTION`;`store.set_graph_status`。
- Produces: `ingest_document` 在向量入库成功后,开关开则投递 `extract_document_entities`,开关关则置 `graph_status='skipped'`。

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_document_pipeline.py`(顶部已 `import rag.document.pipeline as pipe`):

```python
async def test_ingest_enqueues_graph_task_when_enabled(monkeypatch):
    enqueued = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        pass

    async def fake_get_object(*a, **k):
        return b"data"

    class _Redis:
        async def enqueue_job(self, name, *args):
            enqueued.append((name, args))

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: ["a"])

    class _FakeEmbedding:
        async def embed(self, texts):
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    from types import SimpleNamespace
    settings = SimpleNamespace(
        SPLIT_STRATEGY="fixed_size", CHUNK_SIZE=800, CHUNK_OVERLAP=100,
        EMBEDDING_BATCH_SIZE=8, ENABLE_ENTITY_EXTRACTION=True,
    )
    ctx = {"pg": None, "minio": None, "bucket": "b", "embedding": _FakeEmbedding(),
           "settings": settings, "redis": _Redis()}

    await pipe.ingest_document(ctx, "d1")
    assert ("extract_document_entities", ("d1",)) in enqueued


async def test_ingest_marks_graph_skipped_when_disabled(monkeypatch):
    skipped = []

    async def fake_claim(pool, doc_id):
        return True

    async def fake_get_document(pool, doc_id):
        return {"object_key": "k", "content_type": "txt", "knowledge_base_id": "kb"}

    async def fake_store_complete(pool, doc_id, kb, embedded):
        pass

    async def fake_get_object(*a, **k):
        return b"data"

    async def fake_set_graph_status(pool, doc_id, status, error=None):
        skipped.append((doc_id, status))

    monkeypatch.setattr(pipe.store, "claim_for_processing", fake_claim)
    monkeypatch.setattr(pipe.store, "get_document", fake_get_document)
    monkeypatch.setattr(pipe.store, "store_chunks_and_complete", fake_store_complete)
    monkeypatch.setattr(pipe.store, "set_graph_status", fake_set_graph_status)
    monkeypatch.setattr(pipe, "get_object", fake_get_object)
    monkeypatch.setattr(pipe, "parse", lambda data, ct: "text")
    monkeypatch.setattr(pipe, "chunk", lambda strategy, text, size, overlap: ["a"])

    class _FakeEmbedding:
        async def embed(self, texts):
            return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    from types import SimpleNamespace
    settings = SimpleNamespace(
        SPLIT_STRATEGY="fixed_size", CHUNK_SIZE=800, CHUNK_OVERLAP=100,
        EMBEDDING_BATCH_SIZE=8, ENABLE_ENTITY_EXTRACTION=False,
    )
    ctx = {"pg": None, "minio": None, "bucket": "b", "embedding": _FakeEmbedding(),
           "settings": settings, "redis": None}

    await pipe.ingest_document(ctx, "d1")
    assert ("d1", "skipped") in skipped
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_document_pipeline.py::test_ingest_enqueues_graph_task_when_enabled tests/test_document_pipeline.py::test_ingest_marks_graph_skipped_when_disabled -v`
Expected: FAIL(未投递 / 未置 skipped)。

- [ ] **Step 3: 修改 ingest_document**

在 `rag/document/pipeline.py` 的 `await store.store_chunks_and_complete(...)` 调用之后(仍在 `try` 块内、`except` 之前)追加:

```python
        # 向量入库已完成;实体图抽取为 best-effort 增强,投递独立任务(阶段一)。
        if settings.ENABLE_ENTITY_EXTRACTION:
            await ctx["redis"].enqueue_job("extract_document_entities", document_id)
        else:
            await store.set_graph_status(pool, document_id, "skipped")
```

说明:`ctx["redis"]` 是 arq 运行时注入的 ArqRedis 连接(worker ctx 默认含 `redis`),用于投递下游任务。

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_document_pipeline.py -v`
Expected: 2 个新测试 PASS。(注:该文件既有旧用例可能因 `chunk`/`store_chunks_and_complete` 签名早已过时而失败,与本任务无关;若既有用例本就通过则应保持通过。)

- [ ] **Step 5: 全量测试(非 integration)**

Run: `pytest -m "not integration" -q`
Expected: 本次新增的单元测试全部 PASS,无新增失败。

- [ ] **Step 6: Commit**

```bash
git add rag/document/pipeline.py tests/test_document_pipeline.py
git commit -m "feat: 入库成功后投递图抽取任务(开关控制)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## 集成冒烟(可选,需 docker 全家桶)

阶段一完成后,端到端手动验证一次:

1. `docker compose up -d`(postgres/redis/minio/neo4j)
2. `alembic upgrade head`
3. `.env` 设 `ENABLE_ENTITY_EXTRACTION=true` 与 `NEO4J_*`
4. 启动 `rag-api` 与 `rag-worker`,上传一个中文小文档
5. 待 `status=done` 后,查 `documents.graph_status` 应为 `done`
6. Neo4j Browser(`http://localhost:7474`)执行 `MATCH (e:Entity)-[r:RELATES]-(f) RETURN e,r,f LIMIT 50` 应看到实体与关系
7. 重新上传同一文档,确认图中节点/边不翻倍(幂等)

---

## Self-Review(计划自审记录)

- **Spec 覆盖**:依赖/docker(T1)、config 开关+NEO4J_*条件必填(T2)、graph_status 迁移(T3,不含 retry_count 属阶段二)、neo4j 连接+约束(T4)、预聚合与冲突合并§3.3(T5)、批量写入+幂等清理§4/§5.1(T6)、chunk 读取+状态领取(T7)、任务编排§5.1+best-effort异常(T8)、worker 注入§9(T9)、投递点+skipped§5.3(T10)。缓存§5.2、死信自愈§5.4、retry_count 均标阶段二,不在本计划——符合范围。
- **占位符**:无 TBD/TODO;每个代码步骤含完整代码与确切命令。
- **类型/命名一致**:`aggregate` 产出 `{"name","type","chunk_ids"}`/`{"source","target","keywords"}` 与 T6 `write_graph` 入参一致;`extract_entities(llm, text, *, chapter_context, language)` 与 entity_extraction.py 现签名一致;`ExtractionResult.entities/relationships`、`Entity(name,type,description)`、`Relationship(source,target,keywords,description)` 与现有 dataclass 一致;config 全大写访问一致。
