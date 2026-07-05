# 设计：BM25 + 向量 RRF 混合检索

日期：2026-07-05
状态：已批准（融合策略/分词器/架构三个决策点均经用户确认）
分支：feat/async-foundation

## 背景与目标

当前检索是纯向量（embed → pgvector 余弦 top-k）。语料为中文（西游记类），
关键词型查询（专名、术语，如「金箍棒」）语义向量常吃亏。目标：增加 BM25 词法召回，
与向量召回做 RRF 融合，提升关键词与语义两类查询的综合命中。

基础已存在：paradedb 镜像的 `pg_search` 扩展已安装；`document_chunks` 上有
BM25 索引 `idx_dchunks_bm25`（迁移 0002），但从未被查询过，且**未配分词器**
（默认分词器对中文无效）；已实测容器内 pg_search 支持 `jieba` 分词器
（`paradedb.tokenize(paradedb.tokenizer('jieba'), ...)` 返回真实词切分）。

## 架构决策（已确认）

1. **融合在检索器内部**：两路检索与 RRF 全部在 `KnowledgeRetriever.search` 内完成
   （`asyncio.gather` 并发），签名与返回结构不变。图结构、`recall` 节点、`generate`、
   Langfuse span 零改动。state 预留的 `recall_bm25_results` 字段保持不动不使用。
2. **分词器用 `jieba`**（实测可用）：词典级切分，「悟空」「金箍棒」作为完整词进索引，
   BM25 词频/IDF 建立在真实词上；查询侧 pg_search 用同一配置切分，两侧自动一致。
3. **RRF**（k=60）：`score(chunk) = Σ_路 1/(60 + rank_路)`，按 chunk id 合并去重，
   取前 top_k。k 与每路候选数为 retriever 模块常量（不进 config.py——并行会话
   正在改该文件，且 YAGNI）。

## 1. 迁移 0005：重建 BM25 索引

- drop `idx_dchunks_bm25`，按当前 pg_search 语法重建：`text` 字段配 jieba 分词器，
  索引字段包含 `knowledge_base_id`（BM25 查询需按库过滤）。
- 具体 `WITH (...)` 语法以容器内实际 pg_search 版本现场验证为准（唯一无法离线
  确定的点，验收 = 真实中文查询跑通且走 BM25 索引）。
- `long_term_memories` 的 BM25 索引不动（本次只做文档检索）。
- downgrade 恢复 0002 的原索引定义。

## 2. 查询层：`store.search_chunks_bm25`

与 `search_chunks` 对称的新函数：

- 入参 `(pool, query_text, knowledge_base_id, top_k)`；
- `@@@` 匹配 + `paradedb.score(id)` 取分，按 `knowledge_base_id` 过滤，
  按分数降序 LIMIT；
- 返回列结构与向量版一致：`id, document_id, chunk_index, text, score`；
- 查询文本用 paradedb 的安全查询构造（避免特殊字符炸查询解析器），具体 API
  （如 `paradedb.match`）依版本现场确认。

**提交纪律**：`store.py` 当前携带并行会话未提交改动，提交时 hunk 级暂存
（同 workflow.py 先例）。

## 3. 融合层：`KnowledgeRetriever.search` 改混合

- 两路并发（`asyncio.gather`）：向量路（现有）与 BM25 路（新），每路取
  `top_k * 2` 候选（模块常量 `CANDIDATE_MULTIPLIER = 2`）；
- RRF 融合（模块常量 `RRF_K = 60`），按 `id` 去重合并，输出前 `top_k`，
  返回行结构不变（下游按 `id/text/...` 访问）；
- **BM25 路失败只降级不致命**：捕获异常记 WARNING，返回纯向量结果
  （覆盖迁移未跑/索引缺失的环境）；
- 向量路失败仍按现状抛出（主路径语义不变）。

## 4. 可观测性（沿用现有设施）

日志 extra 字段扩展：

- `vec_hits` / `bm25_hits`：各路 top 排名（chunk_id + 分数 + text_preview 80 字符）；
- `hits`（融合后）：增加 `rrf_score` 与 `sources`（`["vec"]` / `["bm25"]` /
  `["vec","bm25"]`）；
- `bm25_ms` 与现有 `embed_ms` / `search_ms` 并列；
- 空结果 WARNING 语义保持（两路都空才算空）。

`knowledge_retrieve` span 自动覆盖两路（都在 search 内部）。

## 5. 测试

- 单测（fake 两路查询函数）：RRF 合并（两路重叠 / 互斥 / 单路为空 / 排序正确）、
  BM25 失败降级为纯向量、向量失败仍抛、日志新字段、`sources` 标记正确。
- 现有 test_retriever 用例适配（hits 结构新增字段）。
- 真实验证：迁移跑通 → 关键词查询（如「金箍棒」）BM25 路真实返回且排序合理 →
  对比混合与纯向量的结果差异 → 日志/trace 字段齐全。

## 非目标

- rerank 模型、查询改写联动、memory 模块 BM25 查询、评测黄金集。
- 权重可配化（RRF k / 候选倍数进 Settings）——等有评测数据再调。
