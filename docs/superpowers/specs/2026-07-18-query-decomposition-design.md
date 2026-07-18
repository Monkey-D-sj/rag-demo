# 查询分解 (Query Decomposition) 并行检索设计

日期: 2026-07-18
状态: 已评审通过

## 目标

复杂问题(比较类/并列类/多实体类)单条改写查询召回不全:各检索面在 embedding 中互相稀释,
BM25 关键词也被混杂。扩展 `handle_query` 把此类问题拆为独立子查询,
用 LangGraph Send API 扇出并行检索,融合后进入既有 rerank/dynamic_topk 质量门,
并用评测第 9 轨量化拆解增益。

范围界定:仅做**并行拆解**(一轮拆解,子查询相互独立);
迭代 multi-hop(子问题间有依赖,需中间答案驱动下一跳)不在本期,架构上不预留。

## 已确认的决策

| 决策点 | 结论 |
|---|---|
| 拆解类型 | 并行拆解(一轮 LLM 拆解,N 个独立子查询);迭代 multi-hop 不做 |
| 拆解位置 | 扩展 `handle_query` 现有结构化输出(`QueryRewriteOutput.sub_queries`),零新增 LLM 调用 |
| 并行机制 | LangGraph Send API 扇出 + reducer 字段 fan-in(本项目首个 Annotated reducer) |
| 分支构成 | `[rewrite_query] + sub_queries[:3]`,主查询始终保留为安全网分支,最多 4 分支 |
| entities 分配 | 仅主查询分支携带 `query_entities` 走图召回;子查询分支只走向量+BM25,避免重复打 Neo4j |
| 融合策略 | 跨子查询二级 RRF(同 `RETRIEVER_RRF_K`),纯函数 `fuse_multi_query_results()` 与 eval 共用 |
| 评测 | 第 9 轨 `decomposed` 对照 `fused`;golden 手标可选 `sub_queries`;新增 5~10 条多面型条目 |
| 默认开关 | `QUERY_DECOMPOSITION_ENABLED=false`,关闭时路由恒为单分支,行为与现状一致 |
| 子查询上限 | 硬编码 3(prompt 约束 + 代码 clamp),不做配置(YAGNI) |

## 架构

### 拆解输出 (handle_query 扩展)

- `QueryRewriteOutput` 加字段:
  `sub_queries: list[str] = Field(default_factory=list, description=...)`。
- `rag/prompts/query.py` 增加拆解指令与示例:仅当问题包含多个独立检索面
  (比较/并列/多实体)时拆解;每个子查询自包含(实体显式,无指代);
  简单问题与 out_of_scope 返回空数组。
- 代码层 clamp `sub_queries[:3]` 防超量输出。
- `MyState` 加 `sub_queries: list[str]`,`handle_query` 写入;
  结构化输出失败的现有降级分支同步置 `sub_queries = []`。
- 拆解发生时 `handle_query` 追加一条 STATUS 事件"已拆解为 N 个子问题"。
- 语义缓存不受影响(缓存 key 仍为 rewrite_query,cache_lookup 在扇出之前)。

### Send 扇出与图拓扑

```
cache_lookup ─ 命中 ─→ add_memory
             └ 未命中 ─→ [Send x N] → recall ─→ recall_fuse → neighbor_expand → rerank → ...(下游不变)
```

- `_route_after_cache` 改为:命中返回 `"add_memory"`;未命中构造分支列表
  `[rewrite_query] + sub_queries`(开关关闭或 `sub_queries` 为空时仅主查询一个分支),
  返回 `list[Send]`,每个 `Send("recall", payload)` 携带
  `{"sub_query": str, "entities": list[str]}`(entities 仅主分支非空)。
- `recall` 节点改造:输入从全量 state 改为 Send 负载,
  调 `retriever.search(sub_query, entities=entities or None)`,
  返回 `{"sub_recall_results": [results]}`(单元素列表,交由 reducer 拼接)。
- `MyState` 加 fan-in 字段:
  `sub_recall_results: Annotated[list[list[dict]], operator.add]`。
  这是项目首个 reducer 字段,CLAUDE.md 的"no Annotated reducers"表述同步修订。
- 新增 `recall_fuse` 节点(节点数 13 → 14):读取 N 份结果融合去重,
  写入 `recall_vec_results`;下游 `neighbor_expand`/`rerank`/`dynamic_topk`/
  `parent_expand` 与 `_route_after_topk` 零改动。
- LangGraph 同一 superstep 内跑完全部 Send 分支后才进 `recall_fuse`,天然 fan-in 屏障。

### 融合策略 (fuse_multi_query_results)

- 新建纯函数 `fuse_multi_query_results(result_lists: list[list[dict]]) -> list[dict]`
  (放在 `KnowledgeRetriever` 同模块):每份列表内按排名计 `1/(RETRIEVER_RRF_K+rank)`,
  同 chunk(以 id 去重)被多个子查询命中则分数相加,`sources` 取并集,
  行结构保持与 `search_chunks` 一致。
- 不设额外截断:4 分支 x top_k=5 去重后至多约 20 条,
  由下游 rerank(以 `rewrite_query` 完整问题为排序基准)+ dynamic_topk 收敛。
- agent 的 `recall_fuse` 节点与 eval 第 9 轨共用此函数,保证评测即线上逻辑。

## 降级与错误处理

| 故障 | 行为 |
|---|---|
| 结构化输出失败 | 现有降级路径顺带 `sub_queries=[]` → 单分支,等同现状 |
| 单分支 retriever 异常 | `recall` 节点内 try/except 返回空列表;Send 分支抛异常会 fail 整个 run,必须兜住 |
| 全部分支为空 | `recall_fuse` 产出空 `recall_vec_results` → 既有 `no_results` 路由 |
| 开关关闭 | 路由恒单分支;prompt 仍含拆解指令(输出 token 浪费可忽略,不做条件拼装) |

## 评测第 9 轨 decomposed

对齐 `graph_fused` 轨的既有条件轨模式:

- golden 条目加可选 `sub_queries: list[str]` 标注;第 9 轨仅在标注存在时运行。
- 新增 5~10 条多面型 golden 条目(比较类/并列类西游记问题),
  现有条目多为单事实型,拆解对其无意义。
- 轨逻辑:对 `[rewrite_query] + sub_queries` 各调 `retriever.search`
  (复用缓存 embedding 的既有做法),`fuse_multi_query_results()` 融合后
  算 hit@k/recall@k/ndcg@k/mrr,breakdown 表加一列。
- 与 `fused` 轨同条目对比即拆解增益,汇总行随 rewrite/rerank 增益一起输出。
- baseline 沿用条件轨先例:`--update-baseline` 时纳入,gate 对缺失基线的轨跳过。

## 测试

- 单测:
  - `sub_queries` 解析与 clamp(>3 截断);
  - `_route_after_cache` 返回形态(命中字符串/未命中 Send 列表/空拆解单分支/开关关闭);
  - `fuse_multi_query_results` 去重/RRF 加分/sources 并集/空输入;
  - 降级路径(结构化失败 → `sub_queries=[]`);
  - 分支异常隔离(retriever 抛错 → 该分支空结果,不 fail 整图)。
- 集成:编译后 graph 配 fake retriever 跑一条拆解查询,
  断言 fan-in 融合正确且正常走到 `generate`;单分支路径回归不变。
- eval:`-m eval` 下第 9 轨跑通并进 history/breakdown。

## 文档同步

- CLAUDE.md:节点表加 `recall_fuse`,拓扑图更新,
  "TypedDict with no Annotated reducers"表述修订为"仅 `sub_recall_results` 一个 reducer 字段",
  eval 段 8 轨 → 9 轨。
