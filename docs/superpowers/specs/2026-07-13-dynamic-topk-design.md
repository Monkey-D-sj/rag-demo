# 设计：Rerank 后动态 Top-K 截断

日期：2026-07-13
状态：已批准（算法/架构/配置三个决策点均经用户确认）
分支：feat/async-foundation

## 背景与目标

当前 rerank 节点对召回结果仅做排序，不截断——全部 chunk 流入 generate 节点。
低相关性 chunk 参与上下文拼接会稀释 LLM 注意力，增加 token 消耗，且可能引入噪声
降低回答质量。

目标：在 rerank 之后增加动态 top-k 节点，默认返回 top 5；当 rerank_score 相邻
分差过大时提前截断（允许归零走 no_results 兜底），减少无意义的上下文注入。

## 架构决策（已确认）

1. **独立 LangGraph 节点**：在 `rerank` 和条件边之间插入新节点 `dynamic_topk`，
   rerank 只负责排序、dynamic_topk 只负责截断，职责单一，独立可测，开关可控。
2. **相邻相对比分差法**：排序后比较相邻 chunk 的 `rerank_score` 比值——
   当 `score[k] / score[k-1] < ratio_threshold` 时截断，保留前 k 条；
   无触发时取 `default_top_k`。相较于绝对差值法，比值法对分数尺度不敏感。
3. **配置走 Settings + 直接 import**：不通过 ContextSchema 注入，直接用
   `get_settings()` 读取——动态 top-k 是纯配置读取，不是外部依赖。

## 1. 图结构调整

```
# 当前
recall → rerank ─┬─ generate → add_memory → END
                 └─ no_results → END

# 调整后
recall → rerank → dynamic_topk ─┬─ generate → add_memory → END
                                └─ no_results → END
```

- 图边 `rerank → dynamic_topk` 替代原有 `rerank → 条件边` 的直接连接
- 条件路由函数从 `_route_after_recall` 改为 `_route_after_topk`（语义更准确）
- 空结果路由到 `no_results`，非空路由到 `generate`

## 2. 算法：`_dynamic_truncate`

```python
def _dynamic_truncate(
    chunks: list[dict],
    default_top_k: int,
    ratio_threshold: float,
) -> list[dict]:
```

### 执行流程

1. 从 `state["recall_vec_results"]` 取 rerank 后的 chunks（已按 `rerank_score` 降序排列）
2. 空列表 → 直接返回空
3. 只有 1 条 → 直接返回
4. 无 `rerank_score` 字段（rerank 未启用/降级） → 按 `default_top_k` 硬截断
5. 遍历 i = 0..len-2：
   - `score[i]` 为 0 时 → 视为无限大 gap，在 i 处截断（保留前 i 条）
   - `score[i+1] / score[i] < ratio_threshold` → 在 i+1 处截断（保留前 i+1 条），break
   - 否则继续
6. 未触发截断 → `min(default_top_k, len(chunks))`
7. 截断后写回 `state["recall_vec_results"]`
8. 发送 stream event：`"动态筛选 N 条相关结果"`（归零时 `"未找到足够相关内容"`）

### 边界处理

| 场景 | 行为 |
|---|---|
| chunks 为空 | 透传空列表 |
| 只有 1 条 | 直接返回 |
| 无 `rerank_score` | 硬截断 default_top_k |
| score 为 0 | 视为 gap 无限大，截断 |
| score 为负数 | 视为 0 |
| 所有相邻比值 ≥ 阈值 | 取 default_top_k |
| 截断后为空 | 返回空，路由到 no_results |

## 3. 配置

`rag/config.py` 新增 3 个字段：

```python
# 动态 top-k 截断（rerank 之后）
RERANK_DYNAMIC_TOPK_ENABLED: bool = True     # 关闭时透传全部 rerank 结果
RERANK_DYNAMIC_TOPK_DEFAULT: int = 5         # 默认保留条数
RERANK_DYNAMIC_TOPK_RATIO: float = 0.7       # 相邻分差比值阈值
```

节点通过 `from rag.config import get_settings` 直接读取。

环境变量：

```bash
RERANK_DYNAMIC_TOPK_ENABLED=false   # 一键回退
RERANK_DYNAMIC_TOPK_DEFAULT=5
RERANK_DYNAMIC_TOPK_RATIO=0.7
```

## 4. 错误处理

节点不调外部服务，纯内存计算。唯一需防御的是数据异常：

| 异常 | 处理 |
|---|---|
| `recall_vec_results` 不是 list | `isinstance` 检查，不是则置空，透传 |
| chunk 内 `rerank_score` 缺失或非数字 | `float(c.get("rerank_score", 0) or 0)` 兜底为 0 |
| chunk 内 `rerank_score` 为负数 | 视为 0 |
| `ratio_threshold` 配置不合法（≤0 或 >1） | 钳位到 `[0.01, 0.99]` |
| 任何未预期异常 | catch 后记 WARNING，透传原始结果 |

**节点永不抛异常**——任何异常兜底透传原始结果，保证检索链路不中断。
与 rerank 节点 API 失败降级透传的哲学一致。

## 5. 文件变更清单

| 文件 | 变更 |
|---|---|
| `rag/config.py` | 新增 3 个 `Settings` 字段 |
| `rag/agent/nodes/dynamic_topk/__init__.py` | 新建 |
| `rag/agent/nodes/dynamic_topk/topk.py` | 新建 — 节点 + `_dynamic_truncate` 纯函数 |
| `rag/agent/workflow.py` | 注册 `dynamic_topk` 节点，调整条件边 |
| `tests/test_nodes.py` | 新增 `test_dynamic_topk` |

## 6. 测试策略

`_dynamic_truncate` 是纯函数，零外部依赖。`tests/test_nodes.py` 新增：

```python
# 核心逻辑
- 所有相邻分差都小 → 返回 default_top_k (5)
- 第 2-3 名之间分差大 → 截断返回前 2 条
- 第 1 名分数极低 + 首 gap 就很大 → 返回空
- 只有 1 条结果 → 直接返回

# 边界
- 空列表 → 空列表
- 无 rerank_score 字段 → 硬截断 default_top_k
- score 为 0 → 视为无限大 gap
- 所有分数完全相同 → 无 gap，取 default_top_k

# 降级
- 输入不是 list → 透传空
- ratio_threshold 极端值 0.01 / 0.99
```

## 7. 风险评估

| 风险 | 缓解 |
|---|---|
| 阈值选择不当时频繁空结果 | 默认值偏保守(0.7)，且可通过 `RERANK_DYNAMIC_TOPK_ENABLED=false` 一键回退 |
| rerank 分数分布未知，阈值需要调优 | 上线后观察 Langfuse trace 里的 rerank_score 分布再迭代 |
| 对现有检索质量有回退风险 | 开关隔离 + 独立节点，关闭即等效于当前行为 |
