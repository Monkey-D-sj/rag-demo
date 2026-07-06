# 检索层评测体系设计

- 日期：2026-07-06
- 范围：离线 · 检索层防回归评测
- 状态：设计已确认，待实现

## 背景与目标

当前 agent 是一条 LangGraph 链路（`rag/agent/workflow.py`）：
`recall_memory → handle_query（查询改写，当前禁用）→ recall（BM25+向量 RRF 混合检索）→ generate（带 [序号] 引用的流式生成）`。
已有 Langfuse 可观测性；测试以单元级为主，**缺少面向 agent 效果的评测体系**。

本设计只解决一件事：**改代码前后，对检索层做离线、可复现、可 CI 门禁的防回归评测。**

明确不做（本期）：
- 生成层评测（忠实度/引用正确性）—— 仅预留骨架接口，不实现。
- 记忆层、端到端任务成功率评测。
- 在线/生产流量打分（Langfuse experiment）—— 后续可把结果推一份到 Langfuse 做趋势，不属于本期。

## 核心设计决策

1. **只评检索层，确定性打分，零 LLM 参与评测时刻。** LLM 只在「造 golden」阶段用一次。评测本身纯确定性 → 快、便宜、可复现、可当 CI 硬门禁。
2. **golden 绑「来源文本片段」而非 `chunk_id`。** 判定命中 = 检索回的 chunk 的 `text` 归一化后**包含**某个 gold 片段。理由：chunk_id 依赖切分逻辑，而「改切分策略」正是最想回归的一类改动；绑片段可跨 re-chunk 存活。
3. **冻结独立语料 + 独立评测 KB。** 用固定 `corpus.txt`（西游记选段）+ 专用 `EVAL_KB_ID`，与生产库内容解耦，保证任何人都能重建同一评测库。
4. **harness 做成指标/分层无关。** 检索层今天插确定性 scorer；生成层以后插 RAGAS 或自定义 judge，复用同一 runner/baseline/门禁，不改骨架。

## 目录结构

```
rag/eval/
  __init__.py
  datasets/
    retrieval_golden.jsonl      # 人工筛过的 golden 集(提交入库)
    corpus.txt                  # 冻结的评测语料(西游记选段,后续可替换)
  harness.py                    # metric-agnostic 骨架:数据加载/runner/报告/门禁
  metrics.py                    # 检索指标:hit@k / recall@k / mrr / ndcg@k
  seed_corpus.py                # 把 corpus.txt 切分入库到评测专用 KB
  generate_golden.py            # LLM 造候选(人工筛前的原始产物)
  run.py                        # 主入口:跑检索 + 算指标 + 对比 baseline
  baseline.json                 # 冻结的基线指标(提交入库)
tests/
  test_eval_retrieval.py        # @pytest.mark.eval 门禁用例
```

## 数据格式

`rag/eval/datasets/retrieval_golden.jsonl`，一行一条：

```json
{
  "id": "q001",
  "query": "孙悟空的金箍棒原本是什么?",
  "gold_snippets": ["定海神针", "如意金箍棒重一万三千五百斤"],
  "note": "人工筛注释,可选"
}
```

- `gold_snippets`：该问题答案所依赖的原文片段列表（1~N 个）。
- 命中判定：某检索 chunk 命中 = 其 `text` 归一化后**包含任一** gold 片段。
- 归一化：去空白、去中英文标点、大小写折叠，避免中文标点/空格噪音误判。归一化函数在 `metrics.py` 内单一实现，golden 生成与评测共用。

## 指标（`metrics.py`）

纯确定性，零 LLM。对每条 query，按检索返回顺序判定每个位置是否命中 gold，计算：

- **Hit@k**（k=1,3,5）：top-k 内是否至少命中一个 gold 片段，二值。
- **Recall@k**：top-k 内命中的 gold 片段数 / gold 片段总数。
- **MRR**：第一个命中位置的倒数（无命中记 0）。
- **NDCG@k**：位置加权，惩罚「命中但排名靠后」。

报告：
- 聚合层输出各指标全局均值。
- 保留每条 query 明细（命中位置、缺失的 gold 片段），便于定位是哪几条掉了。
- 可选 `--breakdown`：按通道拆分（fused / 仅 vec / 仅 bm25），用于定位回归来自哪一路。默认关闭。

## 检索执行

复用生产检索链路 `rag/document/retriever.py::KnowledgeRetriever.search(query, kb_id, top_k=5)`，
返回 chunk dict 关键字段：`id`、`chunk_index`、`text`、`rrf_score`、`sources`、`similarity`/`score`。
评测对 `EVAL_KB_ID` 执行，`top_k` 取一个覆盖最大评估 k 的值（≥5）。
`--breakdown` 模式下分别取 vec 路、bm25 路的单路排名（复用 retriever 内部两路结果或单独调用 store 层）。

## 门禁逻辑

- `python -m rag.eval.run`
  - 跑全量 golden，打印指标表 + 与 `baseline.json` 的 delta。
  - 当任一核心指标跌破容差 → 退出码非零。
  - 核心门禁指标：`Recall@5`、`MRR`。默认容差：相对下降 > 3% 判 fail（小抖动放过）。容差可配。
- `python -m rag.eval.run --update-baseline`：确认为正向改动后刷新并提交 `baseline.json`。
- `tests/test_eval_retrieval.py` 标 `@pytest.mark.eval`：
  - 默认 `pytest` **不跑**（依赖 DB + embedding，慢）。
  - CI 单独 `pytest -m eval` 一步；本地改检索时手动跑。
  - `pyproject.toml` 注册 `eval` marker，避免 unknown-marker 告警。

## golden 生产流程（一次性 + 人工筛）

1. `python -m rag.eval.seed_corpus`：把 `corpus.txt` 经现有切分器入评测 KB（幂等：先清 `EVAL_KB_ID` 再灌）。
2. `python -m rag.eval.generate_golden`：逐 chunk 喂 LLM，「基于这段原文出 1 个问题，并抽出答案所依赖的原句作为 gold 片段」→ 产出 `retrieval_golden.candidates.jsonl`。
3. 人工快速筛：删歧义题 / 答案跨多 chunk 的脏数据 / 太水的题 → 另存 `retrieval_golden.jsonl` 提交。
4. 量级：首批 30~50 条守回归即可；后续 bad case 随手补入。

## 语料说明

`corpus.txt` 为西游记的一段冻结选段（取有界长度，控制 seed 速度与确定性）。
替换真实业务文档流程：替换 `corpus.txt` → 重跑 `seed_corpus` → 重跑 `generate_golden` + 人工筛 → 重建 `baseline`。

## 生成层预留（本期不实现）

`harness.py` 的 runner / 报告 / 门禁与具体指标解耦。以后加生成层评测时：
- 新增 `datasets/generation_golden.jsonl`（`query → 参考答案`）。
- 新增 `judges.py`（RAGAS 忠实度/相关性，或自定义 [序号] 引用正确性判分，很可能混用）。
- 复用同一 runner / baseline / 门禁骨架，不改现有结构。

注意：RAGAS 的检索指标是 LLM 判分、非确定性，不满足本期「确定性 CI 门禁」目标；即便未来上 RAGAS，检索层仍保留本设计的确定性指标。两者叠加，非二选一。

## 依赖与环境

- 需运行 PG（pgvector + BM25/jieba）+ embedding 服务，复用 `docker-compose.yaml` 现有栈。
- 评测 KB id 通过配置/常量提供（如 `EVAL_KB_ID = "__eval__"`），与生产 `DEFAULT_KB_ID` 隔离。
- 无新增第三方框架依赖（本期）。
```
