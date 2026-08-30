# P0-1 生成端评估体系 Implementation Plan

- 日期：2026-08-29
- 状态：设计已确认，待实现
- 范围：P0-1；P0-2 压测与成本基线不在本计划内
- 实施策略：在现有 eval 体系上增量补齐，保留现有检索评估能力

## 1. 目标

在不改变现有 LangGraph 节点架构的前提下，为真实 RAG 回答建立可复现、可留档、可门禁的生成端评估体系。

必须满足：

1. 每条答案由当前生产 LangGraph 完整流水线真实生成，不使用手工答案或简化版 `retriever + llm` 旁路。
2. 覆盖三个核心指标：
   - `faithfulness`：回答论断被最终检索上下文支撑的比例。
   - `answer_relevance`：回答是否回应原始用户问题。
   - `citation_accuracy`：正文中的引用编号是否真实支撑对应论断。
3. 继续读取 `rag/eval/datasets/retrieval_golden.jsonl`，不新增第二份 golden 数据集。
4. 默认固定种子抽取 15 条 eligible 记录；允许显式指定 ID、数量或全量。
5. 单题生成或 Judge 失败时继续整轮，失败原因必须落盘。
6. 结果进入 history，支持 baseline、硬下限、相对回归门禁和显式 baseline 更新。
7. 检索指标与答案指标同屏展示，并能定位“检索好但生成差”的题目。

## 2. 非目标

- 不实施 P0-2 的并发阶梯、TTFT、QPS、降级压测或成本报告。
- 不接入 CI；本计划只提供可供 CI 调用的稳定退出码和命令入口，CI 接入属于 P2-1。
- 不新增认证、ACL、告警或线上反馈闭环。
- 不创建新的 LangGraph 节点，不修改现有业务路由语义。
- 不以 15 条临时样本推导永久质量标准；硬下限须在首轮结果人工校准后显式设置。
- 不把 RAGAS 的非确定性指标替代现有确定性检索指标。

## 3. 当前实现与差距

当前已有：

- `rag/eval/run.py`：检索多路评估、history、baseline 和 3% 相对回归门禁。
- `rag/eval/answer_harness.py`：静态 `answer_golden.jsonl` 加载、RAGAS Faithfulness、简化答案生成。
- `rag/eval/answer_run.py`：独立 faithfulness CLI、history 和 baseline 雏形。
- `tests/test_faithfulness_eval.py`：仅覆盖静态数据加载。
- `ragas` 依赖及 `Faithfulness`、`ResponseRelevancy` 能力。

现有生成评估不能直接扩展后交付，原因：

1. `generate_answer_dataset()` 只调用 `retriever.search()` 和 `llm.ainvoke()`，绕过查询分析、拆解、二级融合、rerank、dynamic top-k、expand、缓存路由等生产逻辑。
2. 使用派生的静态 `answer_golden.jsonl`，答案不会随生产流水线变化而更新。
3. 只有 faithfulness，没有 answer relevance 和严格引用准确率。
4. 生成指标与检索指标分别留档，不能逐题关联。
5. Judge 失败只跳过，没有成功率门禁，可能出现“大量失败后剩余高分样本虚假通过”。
6. 现有 answer baseline 尚未生成，可以直接迁移，不需要兼容已提交的 answer 基线数据。

## 4. 总体架构

```text
retrieval_golden.jsonl
        │
        ├─ 过滤 eligible（默认 out_of_scope=false）
        ├─ 固定 seed 抽样 15 条 / --ids / --all
        ▼
collect
        │
        ├─ 真实 graph.astream(custom + values)
        ├─ 收集 answer / citations / final contexts / route
        └─ 每题即时写入 history/<run-id>/items/<id>.json
        ▼
score
        │
        ├─ RAGAS Faithfulness
        ├─ RAGAS ResponseRelevancy
        └─ 自建结构化 Citation Judge
        ▼
aggregate + serving retrieval
        │
        ├─ answer 三指标 macro average
        ├─ generation/judge success rate
        ├─ 对最终 generate contexts 计算检索指标
        └─ 与现有 retrieval legs 同屏关联
        ▼
history + report + baseline gate
```

生成和评分必须分成可恢复的两个阶段。Judge 失败后重跑评分不得重新生成答案。

## 5. 数据选择与可比性

### 5.1 Eligible 条目

三个核心 RAG 指标默认只评估：

```python
item.out_of_scope is False
```

原因：out-of-scope 路径没有检索上下文和引用，faithfulness/citation_accuracy 无定义。范围外数据继续由现有分类评估负责；本期不把不同口径混入同一 answer aggregate。

如未来 golden schema 变化，eligible 判断集中在一个函数中，不散落在 CLI 和 harness。

### 5.2 默认抽样

默认参数：

```text
limit = 15
seed = 42
```

算法：

1. 校验所有非空行 JSON 合法、ID 唯一。
2. 过滤 eligible 条目。
3. 按 ID 排序，消除文件重排对抽样的影响。
4. 使用局部 `random.Random(seed)` 抽取 `min(limit, eligible_count)` 条。
5. 将选中条目按 ID 排序后执行，保证输出顺序稳定。

不得调用全局 `random.seed()`，避免污染进程内其他随机行为。

### 5.3 CLI 覆盖

```bash
rag-eval-answer                         # 默认 15 条，seed=42
rag-eval-answer --limit 30 --seed 7
rag-eval-answer --all
rag-eval-answer --ids q001,q003,q017
```

`--ids`、`--all`、`--limit` 互斥。指定不存在、重复或 ineligible 的 ID 时直接报错，不静默替换。

### 5.4 指纹

manifest 同时记录：

- `dataset_fingerprint`：golden 文件原始字节的 SHA-256。
- `sample_fingerprint`：选中记录规范化 JSON 的 SHA-256。
- `selected_ids`、`limit`、`seed`。

门禁以 `sample_fingerprint + selected_ids` 判断题集可比性。完整数据集发生变化但选中样本完全不变时允许比较，同时在报告中提示 dataset 已变化。

## 6. 真实流水线采样

### 6.1 工作流入口

在 `rag/agent/workflow.py` 保留现有 `invoke()` 对生产调用方的行为，并新增内部可复用的初始 state 构造函数：

```python
def build_initial_state(session_id: str, query: str) -> MyState: ...
```

生产 `invoke()` 与评测 collector 必须共用它，禁止复制初始字段列表。

评测 collector 直接复用已编译的 `graph`：

```python
async for mode, payload in graph.astream(
    build_initial_state(session_id, query),
    context=context,
    stream_mode=["custom", "values"],
):
    ...
```

- `custom`：收集 status、message、citations 事件。
- `values`：保存最后一个完整 state 快照。
- 不修改生产 SSE 协议，不把 state 下发给 HTTP 客户端。

### 6.2 评测依赖

collector 使用真实：

- `NormalModel`
- `KnowledgeRetriever`
- `QwenReranker`（当生产设置启用）
- PostgreSQL pool
- EmbeddingModel

collector 显式禁用会污染结果或写用户数据的依赖：

- `memory_manager=None`：`recall_memory` 和 `add_memory` 走现有降级，不读写会话记忆。
- `semantic_cache=None`：每题必须进入真实检索和生成，不允许旧缓存代答，也不回写缓存。
- Langfuse callback 默认不注入；如以后需要 eval trace，另设显式开关。

`ContextSchema.memory_manager` 的类型调整为 `MemoryManagerProtocol | None`，与节点现有运行时判空行为保持一致。

### 6.3 Session ID

每题使用：

```text
eval-<run-id>-<safe-item-id>
```

仅用于日志关联，不写会话表和记忆表。item ID 进入 session ID 前只允许字母、数字、短横线和下划线，其余字符替换，原始 ID 仍完整保存在 artifact。

### 6.4 Collector 输出

```json
{
  "id": "q001",
  "query": "孙悟空从哪里出生？",
  "pipeline": {
    "route": "rag_single",
    "rewrite_query": "...",
    "sub_queries": [],
    "answer": "...",
    "contexts": [
      {
        "index": 1,
        "text": "...",
        "document_title": "西游记",
        "metadata": {}
      }
    ],
    "citations": [],
    "cache_hit": false,
    "latency_ms": 3812
  },
  "collection_error": null
}
```

最终 contexts 必须来自 graph 结束前最后一次 `state["recall_vec_results"]`，即 expand 后真正传给 generate 的内容。不得使用初始 recall 或离线 harness 的候选结果冒充。

### 6.5 一致性检查

每题 collect 成功至少满足：

- route 为 RAG 生成路径。
- `generated` 非空。
- 至少收到一个 message，或最终 state 中有非空 `generated`。
- 最终 state 的 `generated` 与拼接 message 的文本一致；不一致时记录诊断并以 state 为准。
- contexts 与 citations 的 index 连续，从 1 开始。
- cache_hit 必须为 false。

不满足时该题标记 collection failure，不进入 Judge，但整轮继续。

## 7. 回答预处理

新增纯函数预处理层，禁止三个 Judge 各自写一套清洗规则。

### 7.1 输出结构

```python
@dataclass
class NormalizedAnswer:
    raw_answer: str
    answer_body_with_citations: str
    answer_body_plain: str
    bibliography: list[dict]
```

### 7.2 规则

1. 识别答案尾部形如 `^\s*\[(\d+)\]:` 的 bibliography 行。
2. bibliography 之前为正文；正文中的 `[1]`、`[1][2]` 保留给 Citation Judge。
3. `answer_body_plain` 移除正文引用 token，但保留原句文字、段落和标点。
4. 不把 bibliography 中的文档标题当成答案论断。
5. 不修改 `raw_answer`，确保可审计。
6. 非法引用格式不静默修正，只记录 format diagnostics。

Faithfulness 与 Answer Relevance 使用 `answer_body_plain`；Citation Judge 使用 `answer_body_with_citations`。

## 8. 指标设计

### 8.1 Faithfulness

实现：RAGAS `Faithfulness`。

输入：

```python
SingleTurnSample(
    user_input=item.query,
    response=normalized.answer_body_plain,
    retrieved_contexts=[context.text for context in item.contexts],
)
```

对外指标名固定为 `faithfulness`。保留 RAGAS 原始分数，不二次缩放。

### 8.2 Answer Relevance

实现：RAGAS `ResponseRelevancy`。

关键约束：

- `user_input` 必须是原始 `query`。
- 不得使用 rewrite_query 或 sub_query。
- `response` 使用 `answer_body_plain`。
- 使用 eval Judge LLM 和现有 embedding 配置。
- 对外指标名固定为 `answer_relevance`，不暴露 RAGAS 历史命名差异。

### 8.3 Citation Accuracy

RAGAS 当前没有满足“论断—编号—chunk”要求的现成指标，使用独立结构化 Judge。

Judge 输入：

- 原始 query。
- 带正文引用的 answer body。
- `index -> chunk text/title/metadata` 映射。

Judge 输出 Pydantic schema：

```python
class CitationLinkVerdict(BaseModel):
    index: int
    valid: bool
    supported: bool
    reason: str


class CitationClaimVerdict(BaseModel):
    claim: str
    is_factual: bool
    citation_indices: list[int]
    links: list[CitationLinkVerdict]


class CitationJudgeOutput(BaseModel):
    claims: list[CitationClaimVerdict]
```

Judge 规则：

1. 将回答拆成最小可验证事实论断。
2. 找出紧邻或明确修饰该论断的正文引用编号。
3. 每个引用链接分别判断编号是否存在、对应 chunk 是否单独提供实质支撑。
4. 仅主题相关但不能推出论断，判 `supported=false`。
5. chunk 与另一 chunk 联合才支撑时，每个链接仍分别判断，并额外在 reason 中说明联合关系；核心指标不因“堆多个弱引用”加分。
6. 意见、格式标题和纯过渡句标记 `is_factual=false`，不进入分母。

每题指标：

```text
supported_valid_links
──────────────────────────────────────────────
all_citation_links + uncited_factual_claims
```

特殊情况：

- 引用不存在：进入分母，不进入分子。
- chunk 存在但不支撑：进入分母，不进入分子。
- 事实论断无引用：每个未引用事实论断给分母加 1。
- 无任何事实论断：`citation_accuracy = null`，记录为 N/A，不伪造 1.0。

同时保存诊断指标：

- `citation_validity`
- `citation_precision`
- `citation_coverage`
- `factual_claim_count`
- `uncited_factual_claim_count`

三个诊断指标不进入 P0 核心 baseline。

### 8.4 聚合

- 每题先计算单题分数，再对非 null 分数做 macro average。
- 每道题权重相同，避免长答案拥有更多引用链接而支配全局分数。
- 每个核心指标独立报告 `evaluated_count`、`skipped_count`、`success_rate`。
- collection failure 不作为 Judge skipped 混入；单独进入 `generation_success_rate`。

## 9. Judge 模型与调用治理

采用已确认的 J2：Judge 独立可配置，未配置时回退生成模型。

### 9.1 Settings

在 `rag/config.py` 增加全部可选字段：

```python
EVAL_JUDGE_MODEL_NAME: str = ""
EVAL_JUDGE_MODEL_URL: str = ""
EVAL_JUDGE_MODEL_KEY: str = ""
EVAL_JUDGE_CONCURRENCY: int = 2
EVAL_JUDGE_TIMEOUT_SECONDS: int = 60
EVAL_JUDGE_MAX_ATTEMPTS: int = 3
EVAL_RANDOM_SEED: int = 42
```

Judge 三个连接字段必须遵循“全空或全有”：

- 全空：回退 `MODEL_NAME/MODEL_URL/MODEL_KEY`。
- 部分填写：启动评测时报配置错误，禁止混用不同端点的字段。

这些字段不能加入 API 启动的 `_REQUIRED_FIELDS`，避免未使用 eval 时增加生产启动要求。

### 9.2 模型身份

manifest 必须记录：

- generator model/name/url host（URL 仅记录 scheme + host，不记录 query/userinfo）。
- judge model/name/url host。
- embedding model。
- reranker model 或 disabled。
- Judge prompt version/hash。
- RAGAS 版本。

API key 永不落盘。

### 9.3 稳定性

- Judge temperature 固定为 0。
- seed 默认 42；provider 不支持 seed 时记录 `seed_supported=false`。
- Judge 全局 semaphore 默认 2。
- 单项指标单独 timeout/retry；一个指标失败不得取消同题其他指标。
- 单题三个指标可并行，但必须受同一个全局 Judge semaphore 控制。
- 重试仅用于超时、5xx、解析/校验失败；认证、配额等明确不可恢复错误不盲目重试。

### 9.4 成本记录边界

本 P0-1 只在 artifact 记录可获得的 generator/Judge 输入输出 token，不计算或承诺成本数字。统一成本折算留给 P0-2。

## 10. 可恢复运行与 History

### 10.1 目录

```text
rag/eval/history/<run-id>/
├── manifest.json
├── items/
│   ├── q001.json
│   └── ...
├── result.json
└── report.md
```

`run-id` 格式：

```text
YYYYMMDD-HHMMSS-<short-sha>
```

同秒冲突时追加短随机后缀，不能覆盖既有目录。

### 10.2 Manifest 状态机

```text
collecting -> collected -> scoring -> completed
     |                         |
     └-------------------------└-> partial
```

manifest 记录：

- schema version、run ID、状态和各阶段时间。
- git commit、dirty flag。
- 数据和样本指纹。
- selected IDs、limit、seed。
- 模型、prompt、RAGAS 和关键开关身份。
- collection/metric 成功数与错误摘要。

### 10.3 原子写入

- manifest 和 item JSON 先写同目录临时文件，再 `os.replace()`。
- 每完成一题 collect 或一项 score 立即更新该 item。
- 不使用一个持续 append 的共享 JSONL，避免并发写损坏。
- `result.json` 和 `report.md` 只由最终 aggregate 阶段生成，可重复覆盖同一 run 的派生文件。

### 10.4 Resume

```bash
rag-eval-answer --resume <run-id>
```

行为：

- collecting 中断：只 collect 尚无成功 pipeline 数据的题目。
- scoring 中断：只运行缺失或失败且仍允许重试的指标。
- completed：默认拒绝重复执行；显式 `--rescore` 才生成新的 score attempt。
- resume 时发现当前模型、prompt 或样本指纹与 manifest 不同，拒绝继续，避免一个 run 混入不同实验条件。

## 11. Baseline Schema 与迁移

`rag/eval/baseline.json` 升级为：

```json
{
  "schema_version": 2,
  "provenance": {
    "sample_fingerprint": "...",
    "selected_ids": ["..."],
    "seed": 42,
    "generator_model": "...",
    "judge_model": "...",
    "embedding_model": "...",
    "judge_prompt_hash": "..."
  },
  "retrieval": {
    "legs": {},
    "serving": {}
  },
  "answer": {
    "metrics": {
      "faithfulness": 0.0,
      "answer_relevance": 0.0,
      "citation_accuracy": 0.0
    },
    "floors": {
      "faithfulness": null,
      "answer_relevance": null,
      "citation_accuracy": null
    }
  },
  "updated_at": "...",
  "updated_by_run": "...",
  "update_reason": "..."
}
```

### 11.1 兼容旧 retrieval baseline

shared baseline loader 识别当前 v1 形状（顶层为 leg 名）：

```text
v1 -> 内存中的 v2.retrieval.legs
```

读取兼容不立即改文件。第一次显式 `--update-baseline` 才写回 v2。现有 retrieval CLI 和测试改用 shared loader，确保迁移后不丢原门禁。

### 11.2 可比性

Answer baseline 比较要求以下字段一致：

- sample fingerprint 和 selected IDs。
- generator model。
- Judge model。
- embedding model。
- Judge prompt hash。

不一致时状态为 `incomparable`，生成报告但门禁非零退出，并列出差异。不得自动把不可比运行更新成 baseline。

## 12. 门禁

采用已确认的 G2：硬下限 + 锁定 baseline + 显式接受退化。

### 12.1 基本条件

正式通过必须满足：

1. `generation_success_rate == 1.0`。
2. 每项 Judge `success_rate >= 0.95`。
3. 三个核心指标均未低于各自硬下限（设置后）。
4. 三个核心指标相对 baseline 下降不超过 3%。
5. provenance 可比。

Judge 单条失败不会中断整轮，但过多失败不能让整轮虚假通过。

### 12.2 边界复评

对某一 answer 指标：

```text
relative_drop <= 3%       -> 正常判定
3% < relative_drop <= 5%  -> 对同一已生成样本再评分两次，取三次中位数
relative_drop > 5%        -> 直接失败
```

边界复评只重跑对应 Judge，不重新 collect。三次原始分数和最终中位数全部落盘。复评后仍超过 3% 则失败。

### 12.3 硬下限校准

首轮实现不写死阈值。流程：

1. 跑默认 15 条并生成报告。
2. 人工抽查高、中、低分样本及 Judge reason。
3. 显式设置三个 floor。
4. floor 进入版本控制。

floor 为 null 时：

- 相对 baseline 门禁仍执行。
- 报告明确显示 `hard-floor: calibration required`。
- 综合正式门禁状态不能标记为 fully active。

不得从本轮均值自动推导 floor，避免把已有低质量固化成标准。

### 12.4 Baseline 更新保护

初次建立：

```bash
rag-eval-suite --update-baseline
```

已有 baseline 时，普通 `--update-baseline`：

- 只允许保持或提高 answer 核心指标。
- 保留已有 floors，不自动修改。
- provenance 变化时要求显式确认实验重建。

接受有意退化：

```bash
rag-eval-suite \
  --update-baseline \
  --accept-regression \
  --reason "切换生成模型，以较小忠实度损失换取延迟下降"
```

`--accept-regression` 必须同时提供非空 reason。旧值、新值、原因、commit 和 run ID 写入 baseline 与 history。不能通过交互式默认回答绕过。

## 13. Serving Retrieval 与同屏报告

### 13.1 Serving Retrieval

对 collector 捕获的最终 contexts，复用 `rag.eval.metrics.evaluate_query()` 与 golden `gold_snippets` 计算：

- hit@1/3/5
- recall@1/3/5
- MRR
- NDCG@1/3/5

命名为 `serving_retrieval`，表示真实 generate 输入，不与现有离线 `fused`、`fused_reranked` 等 legs 混淆。

### 13.2 综合入口

保留：

```bash
rag-eval            # 现有检索评估
rag-eval-answer     # collect + score + answer 报告
```

新增：

```bash
rag-eval-suite      # 同一抽样上跑检索 + 答案评估 + 综合门禁
```

`rag-eval-suite` 是 P0-1 的正式综合门禁入口；两个原入口继续支持独立诊断。

### 13.3 总览表

```text
层次                 指标                 当前    Baseline    变化    门禁
Serving Retrieval    Recall@5           0.82       0.80     +2.5%     ✓
Serving Retrieval    NDCG@5             0.76       0.77     -1.3%     ✓
Answer               Faithfulness       0.89       0.91     -2.2%     ✓
Answer               Answer relevance   0.86       0.88     -2.3%     ✓
Answer               Citation accuracy  0.92       0.95     -3.2%     ✗
```

随后打印现有 retrieval legs 表及逐题对照：

```text
ID     Recall@5  Faithfulness  Relevance  Citation  诊断
q001      1.00       0.95         0.91       1.00    正常
q002      1.00       0.42         0.88       0.33    检索好、生成差
q003      0.00       0.20         0.31       0.00    检索失败向下传导
```

诊断阈值只用于展示：

- 检索好：`recall@5 >= 0.8`。
- 回答好：三个非 null answer 指标均 `>= 0.8`。

输出四象限数量，并可输出 Spearman 相关系数。15 条样本下相关系数仅作观察，不参与门禁或简历结论。

## 14. CLI

### 14.1 Answer CLI

```bash
rag-eval-answer
rag-eval-answer --limit 15 --seed 42
rag-eval-answer --ids q001,q003,q017
rag-eval-answer --all
rag-eval-answer --collect-only
rag-eval-answer --score-only --run-id <id>
rag-eval-answer --resume <id>
rag-eval-answer --rescore <id>
```

约束：

- `--score-only`、`--resume`、`--rescore` 必须指向已有 run。
- `--collect-only` 不做门禁，只产生 collected artifact。
- 新运行默认 collect 后 score。
- `--update-baseline` 只接受 completed run。

### 14.2 Suite CLI

```bash
rag-eval-suite
rag-eval-suite --limit 15 --seed 42
rag-eval-suite --update-baseline
rag-eval-suite --update-baseline --accept-regression --reason "..."
```

默认缺 baseline、不可比、门禁失败或 floor 尚未校准时，输出明确状态和非零退出码；`--update-baseline` 成功时退出 0。

## 15. 文件级实施清单

### 15.1 新增文件

| 文件 | 职责 |
|---|---|
| `rag/eval/answer_metrics.py` | 回答预处理、Citation Judge schema、引用指标纯函数、聚合 |
| `rag/eval/artifacts.py` | run ID、manifest/item 原子写、resume、result/report 路径 |
| `rag/eval/baseline.py` | v1/v2 baseline 加载、迁移、provenance、floor、更新保护 |
| `rag/eval/suite_run.py` | 检索 + 生成综合编排与同屏报告 |
| `rag/prompts/answer_eval.py` | Citation Judge prompt 与 prompt version |
| `tests/test_answer_metrics.py` | 预处理和引用公式纯单测 |
| `tests/test_eval_artifacts.py` | 原子写、状态机、resume 单测 |
| `tests/test_answer_gate.py` | 相对回归、floor、边界复评、baseline 保护 |
| `tests/test_eval_suite.py` | 同屏聚合和逐题 join 单测 |

### 15.2 修改文件

| 文件 | 修改 |
|---|---|
| `rag/agent/workflow.py` | 抽取共享 initial state；允许 collector 使用 graph 的 custom+values |
| `rag/agent/type.py` | memory_manager 类型与现有 None 降级行为对齐 |
| `rag/config.py` | 增加可选 EVAL_JUDGE 配置和校验 |
| `rag/eval/answer_harness.py` | 删除静态 answer golden 依赖；实现抽样、真实 collector、三指标评分 |
| `rag/eval/answer_run.py` | 改为 collect/score/resume CLI |
| `rag/eval/run.py` | 使用 shared sampler/baseline；允许 suite 复用选中 items |
| `rag/eval/metrics.py` | 如需要，增加对 serving retrieval 结果的复用辅助，不复制指标公式 |
| `rag/eval/baseline.json` | 首次显式更新时迁移到 schema v2 |
| `pyproject.toml` | 新增 `rag-eval-suite` console script；更新 eval marker 文案 |
| `tests/test_faithfulness_eval.py` | 替换静态 answer golden 测试为 collector/scorer 失败隔离测试 |
| `tests/test_eval_retrieval.py` | 通过 shared baseline loader 兼容 v2 |
| `CLAUDE.md` | 更新 eval 命令和生成评估说明 |
| `README.md` | 完成后补用户运行说明；不提前声称已有真实数字 |

现有 `answer_golden.jsonl` 常量和 `--generate` 静态数据集模式删除。仓库当前没有该数据文件和 answer baseline，删除不会丢失已提交评测资产。

## 16. 实施顺序

### Task 1：抽样与指纹

实现：

- shared eligible filter。
- 默认 15、seed 42 的稳定抽样。
- `--ids/--all/--limit` 参数互斥校验。
- dataset/sample fingerprint。

测试：

- 文件重排不改变抽样。
- 同 seed 结果一致，不同 seed 可变化。
- eligible 少于 limit 时全部选择。
- 重复/不存在/ineligible ID 报错。
- 不污染全局 random 状态。

### Task 2：真实 collector

实现：

- workflow initial state 单一来源。
- custom + values 捕获。
- memory/cache 隔离。
- pipeline artifact 和一致性检查。

测试使用 fake graph/fake dependencies，不调用外部服务：

- message 正确拼接。
- 最终 contexts 取最后 state。
- citations 保序。
- cache_hit 或空答案判 collection failure。
- 单题异常不会取消下一题。

### Task 3：回答预处理与 Citation 纯函数

先写纯测试：

- bibliography 不进入正文。
- `[1][2]` 正确保留/移除。
- 非法编号进入诊断。
- supported、unsupported、invalid、uncited 四类计分。
- 无事实论断返回 null。
- macro average 跳过 null 并报告计数。

### Task 4：J2 Judge 工厂

实现：

- 独立 Judge 配置及 fallback。
- RAGAS LLM/embedding adapter。
- Citation structured prompt/schema。
- concurrency、timeout、retry。
- 模型和 prompt provenance。

测试：

- 全空 eval 配置回退生产模型。
- 部分 eval 配置拒绝。
- 三指标失败相互隔离。
- 不可重试错误不重复调用。
- semaphore 不超过配置上限。

### Task 5：Artifacts 与 Resume

实现并测试：

- run 目录不覆盖。
- 原子写。
- 状态迁移合法性。
- collecting/scoring 中断恢复。
- completed 默认不可重复。
- provenance 改变拒绝 resume。

### Task 6：Answer CLI

替换现有静态 answer CLI：

- collect、score、resume、rescore。
- 终端聚合表和低分案例。
- JSON/Markdown 最终报告。
- 正确退出码。

### Task 7：Baseline v2 与门禁

实现并测试：

- v1 retrieval baseline 内存迁移。
- provenance 对比。
- 3% 相对门禁。
- 3%～5% 边界三次中位数。
- floor 检查。
- Judge/生成成功率检查。
- 普通 baseline 不允许向下更新。
- `--accept-regression` 强制 reason 并留档。

### Task 8：Serving Retrieval 与 Suite

实现：

- 对最终 contexts 复用现有检索指标。
- 同一选样调用 retrieval harness 与 answer harness。
- 逐题 join、四象限和同屏表。
- 新增 `rag-eval-suite`。

测试：

- join 只按 ID，不依赖列表位置。
- 某项 N/A 时表格不崩。
- retrieval/answer 任一失败时综合门禁失败。
- 报告含三项 answer 和 serving retrieval。

### Task 9：真实 eval 冒烟

标记为 `@pytest.mark.eval`，默认单元测试不运行。

最低真实验证：

```bash
uv run rag-eval-answer --ids <一个有效ID>
uv run rag-eval-suite --limit 3 --seed 42
```

确认：

- 实际调用完整工作流。
- history 可恢复。
- 三个指标均有结果或清晰失败原因。
- 单题 Judge 失败不终止整轮。
- suite 同屏展示。

### Task 10：文档与校准

1. 更新 README/CLAUDE 命令和架构说明。
2. 默认 15 条跑首轮 completed run。
3. 人工抽查 Judge 输出。
4. 与用户确认三个 hard floor。
5. 显式写入 baseline 并提交。

## 17. 单元测试矩阵

| 区域 | 必测边界 |
|---|---|
| Golden | 空文件、坏 JSON、重复 ID、eligible 不足 |
| Sampling | 稳定 seed、文件重排、ids/all/limit 互斥 |
| Collector | 空流、部分 message、state/stream 不一致、cache hit、节点异常 |
| Normalize | bibliography、连续引用、非法引用、中英文换行 |
| Citation | 无引用、坏编号、弱支撑、多引用、无事实论断 |
| Judge | 单项失败、全部失败、timeout、解析失败、不可重试错误 |
| Aggregate | null、部分成功、macro 权重、success rate |
| Artifact | 原子写、断点、重复 run、provenance 漂移 |
| Baseline | v1 兼容、缺 baseline、floor、3%/5% 边界、降级更新保护 |
| Suite | ID join、serving retrieval、同屏报告、退出码 |

## 18. 验收标准

代码完成不等于 P0-1 完成。最终验收必须同时满足：

1. 默认命令从现有 retrieval golden 稳定抽取 15 条。
2. 每条成功样本来自真实完整 LangGraph，artifact 中能看到 rewrite、subqueries、最终 contexts、answer 和 citations。
3. 输出 faithfulness、answer relevance、citation accuracy 三项核心指标。
4. 任一单题或单指标 Judge 失败不拖垮整轮，失败明细可查。
5. history 包含可复现 provenance，进程中断后可 resume。
6. baseline 缺失、不可比或指标退化时返回非零退出码。
7. baseline 不能通过普通更新悄悄下降；有意接受退化必须带 reason。
8. `rag-eval-suite` 同屏展示 serving retrieval、现有 retrieval legs 和三项 answer 指标。
9. 逐题表能识别“检索好、生成差”。
10. 单元测试全部通过；真实三题冒烟通过。
11. 首轮 15 条完成后人工校准 hard floors；在此之前报告明确标记门禁未 fully active。

## 19. 风险与应对

### Judge 与生成使用同一模型

当前允许 fallback，但 manifest 明确记录。后续只需配置 `EVAL_JUDGE_*` 即可换独立 Judge；模型变化使旧 baseline 自动不可比。

### 15 条样本波动大

使用固定样本和 seed；3%～5% 边界只重评 Judge 并取中位数。报告不声称小样本相关性具有统计显著性。

### RAGAS API 版本变化

所有 RAGAS 调用收口在一个 adapter；测试对 adapter 注入 fake scorer，不在业务代码到处直接 import 指标类。manifest 记录 RAGAS 版本。

### 上下文或答案过长

本计划评估生产实际输入，不在 eval 层私自截断。Judge provider 超限按单项失败记录；上下文硬上限属于 P1-2，不能在评测工具内掩盖生产问题。

### 现有脏工作区

实施时只修改本清单文件。开始每个 Task 前检查 `git diff -- <target files>`；与用户已有修改重叠时先合并意图，不覆盖或回滚无关改动。

## 20. 最终交付物

- 可恢复的真实流水线生成评估 CLI。
- 三项核心生成指标及逐题 Judge 证据。
- 版本化 history 和 baseline v2。
- 双层质量门禁及受保护的 baseline 更新。
- serving retrieval 与 answer 同屏综合报告。
- 完整单元测试与少量真实 eval 冒烟用例。
