# GraphRAG 查询侧闭环 (Graph Recall) 设计

日期: 2026-07-18
状态: 已评审通过

## 目标

把已有的实体抽取图谱(Neo4j)接入查询链路:查询实体 1 跳扩展作为第三路召回,
与向量/BM25 三路 RRF 融合,并用评测第 8 轨量化图召回增益。
现状:实体抽取 pipeline 完整(Entity{name,type,chunk_ids,doc_ids} + RELATES{keywords,doc_ids}),
但查询侧零使用,属沉没成本。

## 已确认的决策

| 决策点 | 结论 |
|---|---|
| 查询实体识别 | `handle_query` 结构化输出顺带抽(`QueryRewriteOutput.entities`),零新增 LLM 调用 |
| 融合策略 | 图召回作为第三路进 RRF,与向量/BM25 统一走 rerank/dynamic_topk 质量门 |
| 融合位置 | `KnowledgeRetriever` 内部(构造注入 `graph_retriever`,`search()` 加可选 `entities` 参数),不动 `ContextSchema` |
| 评测 | 含第 8 轨 `graph_fused`(三路)对照 `fused`(两路);golden 手标可选 `entities` 字段;先记录不进 gate |
| 默认开关 | `GRAPH_RECALL_ENABLED=false`,需配合 `NEO4J_ENABLED=true` |
| 跳数 | 固定 1 跳,不做配置(YAGNI) |

## 架构

### 查询实体识别(handle_query 扩展)

- `QueryRewriteOutput` 加 `entities: list[str] = Field(default_factory=list, ...)`:
  抽取查询中的专有名词实体(人名/地名/物名),与图谱粒度一致,不抽泛义词。
- `rag/prompts/query.py` prompt 增加抽取指令与示例。
- `MyState` 加 `query_entities: list[str]`,`handle_query` 写入。
- 结构化输出失败沿用现有降级:raw query 透传 + `entities=[]`(图路自然跳过)。
- 语义缓存不受影响(缓存 key 仍为 rewrite_query 的 embedding)。

### GraphRetriever(rag/graph/retriever.py 新建)

`GraphRetriever(driver, database, pool)`,方法
`async search(entities: list[str], top_k: int) -> list[dict]`:

1. 单次 Cypher 往返:`UNWIND $names` 精确匹配 `Entity.name` →
   `OPTIONAL MATCH -[:RELATES]-` 扩 1 跳邻居 → 返回种子与邻居实体的 `chunk_ids`。
2. chunk 打分排序:命中种子实体的 chunk 优先于仅邻居命中;命中实体数多者优先;
   截断到 `top_k * RETRIEVER_CANDIDATE_MULTIPLIER`(与另两路候选量对齐)。
3. `chunk_uid`(`document_id:chunk_index`)解析后经新增
   `store.get_chunks_by_uids(pool, uids)` 回 pg 取正文。
   返回行结构与 `search_chunks` 完全一致
   (id/document_id/chunk_index/text/filename/metadata),
   rerank/generate 零改动兼容。

### 三路融合(KnowledgeRetriever 扩展)

- 构造函数加可选 `graph_retriever=None`;lifespan 在
  `NEO4J_ENABLED and GRAPH_RECALL_ENABLED` 时创建并注入。
- `search(query, knowledge_base_ids=None, top_k=5, entities=None)`:
  `graph_retriever` 与 `entities` 均存在时 `asyncio.gather` 三路并发。
- `_merge_dedup` 加第三个 RRF 循环:graph 路按排名计 `1/(rrf_k+rank)`,
  同 chunk 三路命中分数累加,`sources` 追加 `"graph"`。
  两路调用方式(不传 graph_rows)保持向后兼容。
- `recall` 节点:把 `state.get("query_entities")` 传给 `retriever.search()`,一行改动。
- `RetrieverProtocol.search` 签名同步加可选 `entities` kwarg。

## 降级链(与 BM25 同等待遇,向量仍是唯一关键路径)

- `entities` 为空 → 图路静默跳过(不起协程)。
- Neo4j 查询/回 pg 异常 → `logger.warning` + 空列表。
- `GRAPH_RECALL_ENABLED=false` 或驱动未注入 → 不起图路。
- 图路失败最坏情况 = 回到现有两路行为。

## 评测第 8 轨 graph_fused

- golden 条目加可选 `entities: list[str]` 字段(手工标注;西游记语料实体明确)。
  缺该字段的条目跳过此轨。不用 LLM 现场抽,保证可复现。
- `graph_fused` = 三路 RRF(改写 query + 标注 entities),对照 `fused`(两路)。
- 先记录不进 gate:新轨无基线;`--update-baseline` 后自然纳入 baseline.json,
  之后受常规门禁保护。
- 评测环境需先以 `ENABLE_ENTITY_EXTRACTION=true` 重灌 eval 语料入图。
- CI 无 Neo4j service:该轨检测不到可用驱动时自动跳过,现有 7 轨门禁不受影响。

## 配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `GRAPH_RECALL_ENABLED` | `false` | 图召回总开关,需配合 `NEO4J_ENABLED=true` |

## 测试

- `GraphRetriever` 单测:种子优先/多实体命中优先排序、uid 解析、候选截断、
  Neo4j 异常降级(mock driver + mock store)
- `store.get_chunks_by_uids` 单测:SQL 形状、空入参短路
- `_merge_dedup` 三路测试:graph 路 RRF 计分、三路同 chunk 累加、sources 标记、
  两路调用向后兼容
- `handle_query` 测试:entities 写入 state、结构化失败降级 []
- `recall` 节点测试:entities 透传
- eval harness 测试:graph 轨跳过逻辑(无 entities/无驱动)
- integration(`-m integration`):本地 Neo4j 真实图端到端

## 文档

README/CLAUDE.md:技术栈表 Neo4j 行升级为
"GraphRAG:实体抽取入图 + 查询实体 1 跳扩展第三路召回";
检索章节、评测章节(7 轨 → 8 轨)同步。

## 非目标

- 多跳/可配置跳数图遍历
- LightRAG 式关系文本直接进 prompt(候选下一迭代)
- 查询侧独立 LLM 实体抽取调用
- graph 轨进评测 gate(基线建立后自然纳入)
- 实体模糊匹配/别名归一(先精确匹配,命中率数据出来再决定)
