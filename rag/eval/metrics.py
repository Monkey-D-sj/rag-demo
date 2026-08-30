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


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """对多条 query 的同名数值指标取算术平均。空输入返回 {}。

    自动跳过非数值键（如 "id"、"query"），只聚合指标列。
    """
    if not per_query:
        return {}
    numeric_keys = [k for k, v in per_query[0].items() if isinstance(v, (int, float))]
    n = len(per_query)
    return {k: sum(q[k] for q in per_query) / n for k in numeric_keys}


def aggregate_by_category(
    per_query: list[dict],
    id_to_category: dict[str, str],
) -> dict[str, dict[str, float]]:
    """按类别分组聚合：用 per_query 中每条记录的 "id" 查找类别，分别 aggregate。

    返回 {category: {metric: avg}}。无 id 或类别未知的条目归入 "unknown"。
    """
    groups: dict[str, list[dict]] = {}
    for pq in per_query:
        cat = id_to_category.get(pq.get("id", ""), "unknown")
        groups.setdefault(cat, []).append(pq)
    return {cat: aggregate(items) for cat, items in groups.items()}


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
    if not deltas:
        # 没有任何 key 能比对——baseline 实质为空，门禁无法生效
        passed = False
    return passed, deltas


def short_circuit_violations(result, raw_query: str) -> list[str]:
    """检查 query 结构化输出是否满足短路契约,返回违反的规则描述列表。

    契约(见 rag/prompts/query.py「任务优先级与短路」):
    - is_out_of_scope=true:任务二/三/四被跳过 → answer_from_context=false、
      sub_queries=[]、rewrite_query≈原文(归一化比较,容忍标点/空白差异)
    - answer_from_context=true:任务三/四被跳过 → sub_queries=[]、rewrite_query≈原文

    result 需具备 rewrite_query/is_out_of_scope/answer_from_context/sub_queries 四个属性。
    """
    violations: list[str] = []
    if result.is_out_of_scope:
        if result.answer_from_context:
            violations.append("is_out_of_scope=true 但 answer_from_context=true,任务二应被短路")
        if result.sub_queries:
            violations.append(f"is_out_of_scope=true 但 sub_queries={result.sub_queries},任务四应被短路")
        if normalize(result.rewrite_query) != normalize(raw_query):
            violations.append("is_out_of_scope=true 但 rewrite_query≠原文,任务三应被短路")
    if result.answer_from_context:
        if result.sub_queries:
            violations.append(f"answer_from_context=true 但 sub_queries={result.sub_queries},任务四应被短路")
        if normalize(result.rewrite_query) != normalize(raw_query):
            violations.append("answer_from_context=true 但 rewrite_query≠原文,任务三应被短路")
    return violations


def short_circuit_compliance(rows: list[tuple[object, str]]) -> dict:
    """聚合多条 (result, raw_query) 的短路合规率。

    返回 {"checked", "compliant", "compliant_rate", "rule_violations", "non_compliant"},
    non_compliant 每条含 {"raw_query", "violations"}。
    """
    checked = len(rows)
    per_rule: dict[str, int] = {}
    non_compliant: list[dict] = []
    for result, raw_query in rows:
        violations = short_circuit_violations(result, raw_query)
        if violations:
            non_compliant.append({"raw_query": raw_query, "violations": violations})
            for v in violations:
                per_rule[v] = per_rule.get(v, 0) + 1
    compliant = checked - len(non_compliant)
    return {
        "checked": checked,
        "compliant": compliant,
        "compliant_rate": compliant / checked if checked else 0.0,
        "rule_violations": per_rule,
        "non_compliant": non_compliant,
    }


def route_violations(expected: str, result) -> list[str]:
    """按预期路由校验 query 结构化输出的路由字段,返回违反描述。

    expected ∈ {"context_answer", "out_of_scope", "retrieval"}。
    context_answer 走会话上下文、out_of_scope 走直答、retrieval 走知识库检索。
    """
    violations: list[str] = []
    if expected == "context_answer":
        if result.is_out_of_scope:
            violations.append("会话题被误判为 out_of_scope(任务一不应抢占任务二)")
        if not result.answer_from_context:
            violations.append("会话题未被识别为 answer_from_context")
    elif expected == "out_of_scope":
        if not result.is_out_of_scope:
            violations.append("范围外查询未被识别为 out_of_scope")
    elif expected == "retrieval":
        if result.is_out_of_scope:
            violations.append("知识库问题被误判为 out_of_scope")
        if result.answer_from_context:
            violations.append("知识库问题被误判为 answer_from_context")
    return violations


def classify(
    predicted: list[bool],
    actual: list[bool],
) -> dict[str, float]:
    """二分类指标：precision / recall / F1。

    predicted=True 对应「判定为 out_of_scope」。
    """
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def rewrite_gate(
    raw_agg: dict[str, float],
    fused_agg: dict[str, float],
    keys: tuple[str, ...] = ("recall@5", "mrr"),
    rel_tolerance: float = 0.03,
) -> tuple[bool, dict[str, dict[str, float]]]:
    """改写质量门禁：改写后（fused）不得显著差于原文检索（raw）。

    改写使指标下降超过 rel_tolerance 则判 fail，表示改写方向有误。
    """
    passed = True
    deltas: dict[str, dict[str, float]] = {}
    for key in keys:
        if key not in raw_agg or key not in fused_agg:
            continue
        raw = raw_agg[key]
        fused = fused_agg[key]
        rel_drop = (raw - fused) / raw if raw > 0 else 0.0
        deltas[key] = {"raw": raw, "rewritten": fused, "rel_drop": rel_drop}
        if rel_drop > rel_tolerance:
            passed = False
    if not deltas:
        # 没有任何 key 能比对——baseline 实质为空，门禁无法生效
        passed = False
    return passed, deltas
