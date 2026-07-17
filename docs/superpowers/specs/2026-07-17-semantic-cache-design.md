# 语义缓存 (Semantic Cache) 设计

日期: 2026-07-17
状态: 已评审通过

## 目标

对语义相近的重复提问复用历史答案,跳过检索链与 LLM 生成,降低成本与延迟。
收益必须可量化:命中率与节省成本可从 `llm_call_log` 直接对账。

## 已确认的决策

| 决策点 | 结论 |
|---|---|
| 共享范围 | 全局跨会话共享。实现时修正:聊天链路 `recall` 不带 KB 维度(全库检索),故缓存无 KB 作用域,直接全局 |
| 失效策略 | 任何文档入库成功即刻清空全部缓存 + TTL 兜底。实现时修正:项目当前无文档删除 API,失效钩子只挂 worker 入库完成处;答案可能引用多个 KB 的内容,按 KB 局部失效不正确,必须整表清空 |
| 缓存层级 | 答案级,挂在 `handle_query` 之后(方案 A) |
| 缓存 key | `rewrite_query` 的 embedding(指代消解后,跨会话命中安全) |
| 存储 | pgvector 新表,不引入新基础设施 |
| 默认开关 | `SEMANTIC_CACHE_ENABLED=false`,显式 opt-in |

## 架构

### 新组件 `SemanticCache`

位置 `rag/agent/cache/`(仿 `rag/agent/memory/` 的 MemoryManager 模式)。
持有 pg pool + embedding 模型实例(lifespan 复用已有实例注入)。

接口(`rag/agent/type.py` 定义 `SemanticCacheProtocol`,`@runtime_checkable`):

- `async lookup(query, session_id=None) -> dict | None` — embedding + 相似度查询,返回命中条目(answer + citations);内部吞掉一切异常,失败返回 None(降级为未命中)
- `async store(query, answer, citations) -> None` — 回写,写前查重防近重复堆积;内部吞掉一切异常
- 模块级 `async clear_semantic_cache(pool)` — 整表清空(worker 侧失效钩子直接调用,无需构造完整实例)
- 模块级 `async purge_expired(pool, ttl_hours)` — 物理删除过期行(worker cron 用)

`ContextSchema` 增加字段 `semantic_cache: SemanticCacheProtocol | None = None`。

### 图结构(11 节点 → 13 节点)

```
handle_query ─ in-scope → cache_lookup ┬─ 命中 → add_memory → END
                                        └─ 未命中 → recall → … → generate → cache_store → add_memory → END
```

- **`cache_lookup`**(新):对 `rewrite_query` embedding 后按 kb_id + 余弦相似度查缓存。
  命中时通过 `writer` 下发 status("缓存命中")+ 答案 message + citations 事件,
  完整复现原次回答的 SSE 行为;置 `state["generated"]` 与 `state["cache_hit"]=True`,
  条件路由直达 `add_memory`。**命中也写记忆**,保证会话历史连贯、后续指代消解不断链。
  未命中走正常检索链。
- **`cache_store`**(新,`generate` → `cache_store` → `add_memory`):
  回写 `rewrite_query` + 答案 + citations,best-effort,失败仅 warning。
- **不缓存**:`direct_answer`(范围外,与 KB 无关)、`no_results`(空答案)。
  两者本就不写记忆,行为一致。
- **开关关闭时**:两个节点透传,图结构不变。
- **降级**:lookup 阶段 DB/embedding 异常 → 视为未命中继续正常链路,仅 warning。

## 存储

### 新表 `semantic_cache`(Alembic 迁移 0011)

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | BIGSERIAL PK | 与 `llm_call_log` 风格一致 |
| `question` | text | rewrite_query 原文,调试/审计用 |
| `answer` | text | 缓存的完整回答 |
| `citations` | jsonb | 原次回答引用数据,命中时复放 |
| `embedding` | vector(1024) | rewrite_query 向量 |
| `hit_count` | int default 0 | 命中计数 |
| `created_at` | timestamptz default now() | TTL 判断依据 |

索引:`embedding` HNSW(`vector_cosine_ops`)。

### 查询与写入

- 命中查询:`WHERE created_at > now() - TTL ORDER BY embedding <=> ? LIMIT 1`,
  余弦相似度(1 - 距离)≥ 阈值才算命中;命中时 `hit_count` 自增。
- 防膨胀:`store` 前先做同样的相似度查询,已存在 ≥ 阈值条目则跳过插入。

## 失效

1. **入库即刻失效**:worker pipeline `store_chunks_and_complete` 成功后
   调用 `clear_semantic_cache(pool)` 整表清空(best-effort,失败仅 warning)。
2. **TTL 兜底**:lookup 按 `created_at` 过滤(默认 7 天)。
3. **物理清理**:worker 新增每小时 cron 调用 `purge_expired` 删除过期行。

## 可观测性

- 命中时向 `llm_call_log` 写一条记录(复用 `UsageRecorder`):
  `call_type='semantic_cache'`、`model='semantic-cache'`、`status='cache_hit'`。
  实现时修正:`source` 列是 UsageRecorder 构造参数(api/worker/eval),
  不能按记录设置,故用 `call_type` 区分缓存命中;cost 为 None(价格表无此模型,
  语义等同 0)。`hit_count` 同时自增。
- 命中率与节省成本可直接用现有对账 SQL 计算:
  节省成本 ≈ 命中次数 × generate 平均单次成本。

## 配置

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `SEMANTIC_CACHE_ENABLED` | `false` | 总开关,opt-in |
| `SEMANTIC_CACHE_SIM_THRESHOLD` | `0.95` | 余弦相似度命中阈值 |
| `SEMANTIC_CACHE_TTL_HOURS` | `168` | 缓存有效期(7 天) |

## 测试

- `SemanticCache` 单测:命中/未命中/阈值边界/TTL 过期/DB 异常降级(mock pool + embedding)
- 节点测试:`cache_lookup`/`cache_store` 透传(开关关闭/无注入)、命中路由、citations 复放
- workflow 路由测试:命中 → add_memory 不经过 recall;未命中走全链
- 迁移测试:仿 `test_migration.py`
- pipeline 失效钩子测试:入库 done / 删除文档触发 invalidate
- 压测验证(S5 重复 query 场景测命中率与延迟差)为后续可选项,不进本期范围

## 非目标

- 多租户维度的缓存隔离(项目尚无租户体系)
- Redis 热层 / 两级缓存
- direct_answer(范围外回答)的缓存
