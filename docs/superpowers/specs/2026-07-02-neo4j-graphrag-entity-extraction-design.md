# Neo4j GraphRAG 实体抽取 · 设计文档

- 日期:2026-07-02
- 分支:feat/async-foundation
- 状态:已确认,待转实施计划

## 1. 背景与目标

现有入库流程(`rag/document/pipeline.py` 的 `ingest_document`)完成:解析 → 切块 →
embedding → 写入 Postgres `document_chunks`(向量检索)。`rag/document/entity_extraction.py`
已实现单 chunk 的实体/关系抽取函数 `extract_entities(llm, text, ...)`,但**尚未接入任何调用链路**,
抽取结果也无处存放。

本设计新增 **Neo4j 图存储**,把抽取出的实体与关系持久化为知识图谱,作为 **GraphRAG 雏形**:
检索时可从图中查实体邻居 / 多跳关系来增强召回。本设计只覆盖「抽取 + 写图 + 幂等 + 编排」,
**不含** retriever 侧的图查询改造(后续独立迭代)。

### 已确认的关键决策

| 维度 | 选择 |
|------|------|
| 用途 | GraphRAG 雏形(图查实体邻居 / 多跳增强召回) |
| 去重范围 | 按 `(知识库, 实体名)` 合并(`MERGE`) |
| 溯源 | 实体节点存 `chunk_ids`,检索时回 Postgres 取原文;PG 为原文单一真相源 |
| 触发 | 配置开关 `ENABLE_ENTITY_EXTRACTION`,默认关闭 |
| 失败语义 | best-effort:图抽取失败不影响向量入库(文档仍 `status=done`) |
| 编排形态 | 独立 arq 任务 `extract_document_entities`(彻底解耦,生产级) |
| 实体属性 | 精简版,**不存 descriptions**(溯源回 PG 取原文,描述冗余) |

## 2. 架构与数据流

```
[web 上传] → 投递 ingest_document(doc_id)
                    │
                    ▼
   ingest_document (现有任务, worker)
     解析 → 切块 → embedding → store_chunks_and_complete → 文档 status=done
                    │
                    │ 若 ENABLE_ENTITY_EXTRACTION=true:
                    │   入库成功后投递新任务(不阻塞、不影响 done 状态)
                    ▼
   extract_document_entities(doc_id)  ← 新增独立 arq 任务
     1. 领取 (graph_status: pending→processing)
     2. 从 PG 读该文档所有 chunk (id, text, metadata.title 作章节上下文)
     3. 幂等清理:删该 doc 在图里的旧贡献
     4. 逐 chunk 调 extract_entities(llm) → 实体/关系
     5. 写 Neo4j (MERGE 去重, chunk_ids/doc_ids 累积)
     6. graph_status → done
     失败 → graph_status=failed, re-raise 交 arq 重试(不影响向量入库)
```

向量入库(`status`)与图抽取(`graph_status`)**状态字段互相独立**,是 best-effort 语义的落点:
图抽取失败 / 重试不影响已 `done` 的向量检索能力。

## 3. Neo4j Schema

### 3.1 实体节点

```
(:Entity {
    kb_id:     知识库 ID (字符串)
    name:      实体名 (规范化后, 标题大小写)
    type:      实体类型 (Person / Location / ... / 其他)
    chunk_ids: [字符串数组]   ← 稳定复合标识 chunk_uid, 累积去重
    doc_ids:   [字符串数组]   ← 来源 document_id, 用于重跑时精准清理
})
```

- **唯一键 / 去重键**:`(kb_id, name)`。
  约束:`CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.kb_id, e.name) IS UNIQUE`
- `type` 取首次写入值;后续冲突保留原值并记 warning(不覆盖)。
- **溯源标识 `chunk_uid = f"{document_id}:{chunk_index}"`**:**不用** PG 主键 uuid。
  原因:图抽取任务在 `ingest` 之后运行,此时旧 chunk 已被「删旧插新」、uuid 全变,
  用 uuid 无法在重跑时对上旧数据、也无法精确清理。而 `chunk_index`(0..n)对同一文档重跑稳定,
  故复合标识稳定可对齐。回 Postgres 取原文时拆出 `document_id` + `chunk_index`,
  用 `WHERE document_id=? AND chunk_index=?` 查询(该组合天然唯一)。
- **不存 descriptions**:溯源方案是回 Postgres 取原文 chunk 喂 LLM,实体描述属冗余信息。
  抽取仍产出描述,但不落库;将来若需要,加字段回填即可,零损失。

### 3.2 关系(边)

```
(:Entity)-[:RELATES {
    kb_id, keywords:[...], doc_ids:[...]
}]->(:Entity)
```

- 关系视为**无向**。Neo4j 边有物理方向,写入前**按 `name` 字典序规范化**(小者作 source),
  避免同一对实体正反各存一条。查询时用无方向匹配 `(a)-[:RELATES]-(b)`。
- 同一对实体的边 `MERGE` 去重;`keywords` 累积去重。
- 关系上**不存 descriptions**(同实体,冗余;需要时回 chunk)。

### 3.3 去重与冲突合并语义

「同名合并」是设计目标(去重键 `(kb_id, name)`),但同一次抽取里多个 chunk 会产出重复项、
且字段可能冲突(如 `type` 不一致)。**在 Python 侧先做一次内存预聚合,再批量写 Neo4j**,
让冲突裁决集中、可单测,并减少写入行数(一个实体一行,而非每 chunk 一行):

- **实体**:按 `name` 分组 → `chunk_ids` 取并集;`type` 冲突用**众数**决定(平票取首次出现),
  并记 warning。
- **关系**:按「规范化方向后的 (源, 目标)」分组 → `keywords` 取并集。
- **跨任务累积**(节点/边已存在):`type` 用 `ON CREATE SET`(仅新建时写),已存节点不覆盖 ——
  即「最早写入者胜出」;`chunk_ids`/`doc_ids`/`keywords` 始终取并集累积。

**已知局限(本期不解决)**:该策略是「同名即同一实体」。**同名不同义**的实体会被错误合并
(如两个不同的「李明」,或「苹果」公司 vs 水果)。彻底解决需实体消歧(结合 type、上下文向量、
别名库),已在 §12 列为范围之外;雏形阶段接受此取舍。

## 4. 幂等(文档重跑)

文档重新入库时 PG 里 chunk 被删旧插新、`chunk_id` 变化;图侧必须清掉该文档旧贡献,
否则残留悬空 chunk_id。实体跨文档共享(`MERGE`),不能直接删节点。**重跑前按 `doc_id` 清理**。

Cypher 是集合式语言,列表属性用列表推导可整体重写,**禁止逐边 / 逐节点遍历往返**(N+1 反模式)。
purge 用 **2 条批量语句**在一个事务里完成,往返次数与文档 chunk 数无关,只和受影响子图大小有关:

```cypher
-- ① 边:整体重写 doc_ids,空了就删
MATCH (:Entity {kb_id:$kb})-[r:RELATES {kb_id:$kb}]-(:Entity)
WHERE $doc IN r.doc_ids
SET r.doc_ids = [d IN r.doc_ids WHERE d <> $doc]
WITH r WHERE size(r.doc_ids) = 0
DELETE r

-- ② 节点:批量剔除该 doc 的 chunk_uid 与 doc_id,doc_ids 空则删节点
MATCH (e:Entity {kb_id:$kb})
WHERE $doc IN e.doc_ids
SET e.chunk_ids = [c IN e.chunk_ids WHERE NOT c STARTS WITH $prefix],
    e.doc_ids   = [d IN e.doc_ids WHERE d <> $doc]
WITH e WHERE size(e.doc_ids) = 0
DETACH DELETE e
```

参数:`$kb`=知识库 ID,`$doc`=document_id,`$prefix = f"{doc}:"`。
**顺序先边后节点**:先收敛边 `doc_ids` 并删空边,再收敛节点并 `DETACH DELETE` 兜底残留边。

由于 `chunk_uid` 用稳定复合标识(见 3.1),按 `"{doc}:"` 前缀过滤即可**精确**移除该文档贡献,
无需额外的 chunk→doc 映射。又因不存 descriptions,清理完全干净、无残留取舍。清理先于写入执行。

## 5. 任务编排

### 5.1 新增任务 `rag/graph/pipeline.py :: extract_document_entities(ctx, document_id)`

1. `claim_graph_processing`:原子领取(`graph_status` pending/failed → processing),防并发重复。
   领取失败(非可处理态或已被领取)→ 记日志跳过。
2. `get_chunks_for_graph(doc_id)`:返回 `[{chunk_index, text, title}]`(title 取自 chunk metadata,
   作章节上下文)。抽取时用 `chunk_uid = f"{doc_id}:{chunk_index}"` 作为实体的溯源标识。
3. 幂等清理:`graph.store.purge_document(doc_id)`。
4. 逐 chunk 调 `extract_entities(llm, text, chapter_context=title)`,汇总实体与关系。
5. `graph.store.write_entities_and_relations(...)`:MERGE 写入,累积 chunk_ids/doc_ids。
   **批量写入**:用 `UNWIND $entities`/`UNWIND $relations` 把整篇文档的实体、关系各一条语句
   批量 `MERGE`(禁止逐实体一次往返)。累积用 `coalesce` + 列表去重,示例:
   ```cypher
   UNWIND $entities AS ent
   MERGE (e:Entity {kb_id:$kb, name:ent.name})
   ON CREATE SET e.type = ent.type
   SET e.chunk_ids = apoc.coll.toSet(coalesce(e.chunk_ids, []) + ent.chunk_ids),
       e.doc_ids   = apoc.coll.toSet(coalesce(e.doc_ids, []) + [$doc])
   ```
   若不引入 APOC,列表去重改用纯 Cypher 列表推导(`[x IN a WHERE NOT x IN b] + b`);
   实施计划阶段确定是否依赖 APOC(neo4j:5-community 默认不含,倾向纯 Cypher 去重)。
6. `graph_status → done`。
7. 异常:`graph_status=failed` 记 error,**re-raise** 交 arq 重试。

### 5.2 投递点(`rag/document/pipeline.py`)

`ingest_document` 在 `store_chunks_and_complete` 成功后,若 `settings.ENABLE_ENTITY_EXTRACTION`,
投递 `extract_document_entities(document_id)`;开关关闭时,把 `graph_status` 记为 `skipped`。
投递本身失败仅告警,不影响文档 `status=done`。

### 5.3 死信自愈

新增**独立平行 cron**(与向量的 `retry_failed_documents` 解耦):扫描
`graph_status='failed'` 且 `graph_retry_count < MAX_RETRY_ROUNDS` 的文档,按指数退避
(`RETRY_BACKOFF_BASE * 2^graph_retry_count`)重投 `extract_document_entities`,
每次递增 `graph_retry_count`。达上限视为真·死信,需人工介入。

## 6. 数据库变更(Alembic 0004)

documents 表新增:

```sql
graph_status      TEXT DEFAULT 'pending'   -- pending/processing/done/failed/skipped
graph_error       TEXT
graph_retry_count INT  NOT NULL DEFAULT 0
```

downgrade 删除以上三列。

## 7. 配置(`rag/config.py`)

```python
ENABLE_ENTITY_EXTRACTION: bool = False
NEO4J_URI:      str = "bolt://localhost:7687"
NEO4J_USER:     str = "neo4j"
NEO4J_PASSWORD: str = "neo4j_pass"
NEO4J_DATABASE: str = "neo4j"
```

`NEO4J_*` **仅在 `ENABLE_ENTITY_EXTRACTION=true` 时校验必填**(在 `check_required` 内条件校验),
避免不使用图的部署被卡启动。

## 8. 部署(docker-compose)

新增 `neo4j` 服务:

```yaml
neo4j:
  image: neo4j:5-community
  container_name: rag-neo4j
  environment:
    NEO4J_AUTH: neo4j/neo4j_pass
  ports:
    - "7474:7474"   # Browser
    - "7687:7687"   # Bolt
  volumes:
    - neo4j_data:/data
  restart: unless-stopped
```

新增卷 `neo4j_data`。

## 9. 连接管理(`rag/db/neo4j.py`)

- 用 `neo4j.AsyncGraphDatabase`(原生异步,无需 `to_thread`),仿 `rag/db/postgres.py` 风格。
- `create_neo4j_driver(settings) -> AsyncDriver`。
- 约束初始化:worker `on_startup` 执行一次 `CREATE CONSTRAINT IF NOT EXISTS`。
- worker `on_startup` 建 driver + `NormalModel`(注入 `WorkerCtx`);`on_shutdown` 关闭 driver。
- API 端只负责投递任务,不需要 driver。

### `WorkerCtx` 扩展

```python
class WorkerCtx(TypedDict):
    settings: Settings
    pg: AsyncConnectionPool
    minio: Minio
    bucket: str
    embedding: EmbeddingModel
    neo4j: AsyncDriver          # 新增
    llm: ChatModel             # 新增
```

## 10. 文件清单

**新增**
- `rag/db/neo4j.py` —— driver 封装 + 约束初始化
- `rag/graph/__init__.py`
- `rag/graph/store.py` —— MERGE 写实体/关系、`purge_document` 幂等清理
- `rag/graph/pipeline.py` —— `extract_document_entities` 任务 + 自愈 cron 函数
- `alembic/versions/0004_graph_status.py`

**修改**
- `rag/config.py` —— `ENABLE_ENTITY_EXTRACTION` + `NEO4J_*` + 条件必填校验
- `rag/worker/main.py` —— `WorkerCtx` 注入 `neo4j` + `llm`,注册新任务与自愈 cron,`on_startup`/`on_shutdown`
- `rag/document/pipeline.py` —— 入库成功后按开关投递图抽取任务
- `rag/document/store.py` —— `get_chunks_for_graph` + graph_status 领取/置位/自愈领取函数
- `docker-compose.yaml` —— `neo4j` 服务与卷
- `pyproject.toml` —— 加 `neo4j` 依赖

## 11. 测试策略

- **单元**:
  - `extract_entities` 的 JSON 解析(纯 JSON、```json 代码块、非法 JSON 兜底)——已可测
  - 无向关系 name 排序规范化
  - `purge_document` / MERGE 的 Cypher 语句构造
- **集成**(`@pytest.mark.integration`,需 docker 起 neo4j):
  - 抽取 → 写图 → 查询验证节点/边
  - 重跑同一文档验证幂等(无残留、无重复、共享实体不误删)
  - LLM 用 mock 返回固定 JSON,避免真实调用

## 12. 范围边界(本设计不含)

- retriever / agent 侧的图查询与 GraphRAG 召回改造(后续独立迭代)
- 实体消歧(同义不同名、别名归并)超出简单同名合并的部分
- 实体 / 关系描述的持久化(本期精简掉)
- 前端知识图谱可视化
