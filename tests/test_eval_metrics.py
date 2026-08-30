from types import SimpleNamespace

from rag.eval.metrics import (
    aggregate,
    evaluate_query,
    gate,
    normalize,
    route_violations,
    short_circuit_compliance,
    short_circuit_violations,
)


def _r(**kw):
    """构造 result-like 对象,短路字段缺省全为 false。"""
    return SimpleNamespace(
        rewrite_query=kw.get("rewrite_query", ""),
        is_out_of_scope=kw.get("is_out_of_scope", False),
        answer_from_context=kw.get("answer_from_context", False),
        sub_queries=kw.get("sub_queries", []),
    )


def test_normalize_strips_punct_ws_and_casefolds():
    assert normalize("如意金箍棒，重 一万三千五百斤！") == normalize("如意金箍棒重一万三千五百斤")
    assert normalize("ABC def") == "abcdef"


def test_evaluate_query_first_rank_hit():
    m = evaluate_query(
        retrieved_texts=["……定海神针……", "无关内容", "更多无关"],
        gold_snippets=["定海神针"],
        ks=(1, 3, 5),
    )
    assert m["hit@1"] == 1.0
    assert m["mrr"] == 1.0
    assert m["recall@1"] == 1.0
    assert m["ndcg@1"] == 1.0


def test_evaluate_query_hit_at_rank3_not_rank1():
    m = evaluate_query(
        retrieved_texts=["无关", "无关", "……定海神针……"],
        gold_snippets=["定海神针"],
        ks=(1, 3, 5),
    )
    assert m["hit@1"] == 0.0
    assert m["hit@3"] == 1.0
    assert m["mrr"] == 1 / 3


def test_evaluate_query_partial_recall_multi_snippet():
    m = evaluate_query(
        retrieved_texts=["只含 定海神针 这一句", "无关"],
        gold_snippets=["定海神针", "重一万三千五百斤"],
        ks=(1, 3, 5),
    )
    assert m["recall@5"] == 0.5


def test_evaluate_query_no_hit_all_zero():
    m = evaluate_query(["无关一", "无关二"], ["定海神针"], ks=(1, 3, 5))
    assert m["hit@5"] == 0.0
    assert m["mrr"] == 0.0
    assert m["recall@5"] == 0.0
    assert m["ndcg@5"] == 0.0


def test_snippet_match_survives_rechunk_punctuation_noise():
    # gold 片段与检索文本标点/空白不同，仍应命中（跨 re-chunk 的核心诉求）
    m = evaluate_query(["如意金箍棒\n重一万三千五百斤"], ["如意金箍棒，重一万三千五百斤"], ks=(1,))
    assert m["hit@1"] == 1.0


def test_aggregate_means_by_key():
    agg = aggregate([{"hit@1": 1.0, "mrr": 1.0}, {"hit@1": 0.0, "mrr": 0.5}])
    assert agg["hit@1"] == 0.5
    assert agg["mrr"] == 0.75


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}


def test_gate_passes_within_tolerance():
    passed, deltas = gate({"recall@5": 0.79, "mrr": 0.80}, {"recall@5": 0.80, "mrr": 0.80})
    assert passed is True
    assert deltas["recall@5"]["rel_drop"] < 0.03


def test_gate_fails_on_regression_beyond_tolerance():
    passed, deltas = gate({"recall@5": 0.70, "mrr": 0.80}, {"recall@5": 0.80, "mrr": 0.80})
    assert passed is False
    assert deltas["recall@5"]["rel_drop"] > 0.03


def test_gate_fails_when_baseline_empty():
    """baseline 为空或缺少全部要检查的 key 时，门禁必须 fail——不应静默放行。"""
    passed, deltas = gate({"recall@5": 0.5, "mrr": 0.5}, {})
    assert passed is False
    assert deltas == {}


def test_gate_skips_only_missing_keys_still_checks_present():
    """baseline 部分 key 缺失：只跳过缺失的，现有 key 正常比对。"""
    passed, deltas = gate(
        {"recall@5": 0.80, "mrr": 0.50},
        {"recall@5": 0.78},  # mrr 缺失，只比 recall@5
    )
    assert passed is True  # 0.78→0.80 没有下降
    assert "recall@5" in deltas
    assert "mrr" not in deltas


# ── 短路契约合规(short_circuit_violations / short_circuit_compliance)──


def test_short_circuit_compliant_out_of_scope_no_violations():
    r = _r(rewrite_query="你好啊", is_out_of_scope=True)
    assert short_circuit_violations(r, "你好啊") == []


def test_short_circuit_in_scope_allows_rewrite_and_decompose():
    # in-scope 时任务三/四正常执行,改写与拆解都不算违反
    r = _r(rewrite_query="重写后", sub_queries=["s1"])
    assert short_circuit_violations(r, "原文") == []


def test_short_circuit_out_of_scope_rewrite_not_original():
    r = _r(rewrite_query="帮我把快速排序写出来", is_out_of_scope=True)
    violations = short_circuit_violations(r, "帮我写个快速排序")
    assert any("rewrite_query" in v for v in violations)


def test_short_circuit_rewrite_match_tolerates_punctuation():
    # 原文无问号、改写回带问号,仍应视为合规
    r = _r(rewrite_query="今天北京天气怎么样？", is_out_of_scope=True)
    assert short_circuit_violations(r, "今天北京天气怎么样") == []


def test_short_circuit_out_of_scope_leaks_sub_queries():
    r = _r(rewrite_query="你好啊", is_out_of_scope=True, sub_queries=["排序算法"])
    violations = short_circuit_violations(r, "你好啊")
    assert any("sub_queries" in v for v in violations)


def test_short_circuit_out_of_scope_leaks_context_answer():
    r = _r(rewrite_query="你好啊", is_out_of_scope=True, answer_from_context=True)
    violations = short_circuit_violations(r, "你好啊")
    assert any("answer_from_context" in v for v in violations)


def test_short_circuit_context_answer_leaks_sub_queries():
    r = _r(rewrite_query="我第一次讲了啥", answer_from_context=True, sub_queries=["第一次讲的内容"])
    violations = short_circuit_violations(r, "我第一次讲了啥")
    assert any("sub_queries" in v for v in violations)


def test_short_circuit_compliance_rate():
    rows = [
        (_r(rewrite_query="你好啊", is_out_of_scope=True), "你好啊"),
        (_r(rewrite_query="改了", is_out_of_scope=True), "你好啊"),  # 违反 rewrite
    ]
    agg = short_circuit_compliance(rows)
    assert agg["checked"] == 2
    assert agg["compliant"] == 1
    assert agg["compliant_rate"] == 0.5


# ── 路由优先级校验(route_violations,任务一 > 任务二)──


def test_route_violations_context_answer_correct():
    r = _r(rewrite_query="我第一次讲了啥", answer_from_context=True)
    assert route_violations("context_answer", r) == []


def test_route_violations_context_answer_misrouted_out_of_scope():
    # 会话题被误判为 out_of_scope = 任务一抢占任务二
    r = _r(rewrite_query="我第一次讲了啥", is_out_of_scope=True, answer_from_context=False)
    violations = route_violations("context_answer", r)
    assert any("out_of_scope" in x for x in violations)


def test_route_violations_out_of_scope_missed():
    r = _r(rewrite_query="你好啊", is_out_of_scope=False)
    violations = route_violations("out_of_scope", r)
    assert violations  # 未识别为范围外
