"""检索与生成评测的综合编排、逐题 join 和报告。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from rag.eval.answer_metrics import aggregate_answer_metrics
from rag.eval.baseline import comparable_provenance, gate_answer_metrics, load_baseline, update_baseline
from rag.eval.harness import GoldenItem, evaluate_query
from rag.eval.metrics import gate as retrieval_gate


def serving_retrieval(items: list[GoldenItem], artifacts: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    by_id = {x.get("id"): x for x in artifacts}
    rows = []
    for item in items:
        artifact = by_id.get(item.id, {})
        contexts = (artifact.get("pipeline") or {}).get("contexts") or []
        scores = evaluate_query([x.get("text", "") for x in contexts], item.gold_snippets, ks)
        rows.append({"id": item.id, "query": item.query, **scores})
    aggregate: dict[str, float] = {}
    if rows:
        for key in rows[0]:
            if key not in {"id", "query"}:
                aggregate[key] = sum(float(row[key]) for row in rows) / len(rows)
    return {"aggregate": aggregate, "per_query": rows}


def join_suite_results(items: list[GoldenItem], answer_artifacts: list[dict[str, Any]], answer_result: dict[str, Any],
                       retrieval_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """按 ID join，不依赖两个结果列表的顺序。"""
    answer_rows = {row.get("id"): row for row in answer_result.get("per_query", [])}
    artifact_rows = {row.get("id"): row for row in answer_artifacts}
    retrieval_rows = {row.get("id"): row for row in (retrieval_result or {}).get("per_query", [])}
    rows = []
    for item in items:
        answer = answer_rows.get(item.id, {})
        retrieval = retrieval_rows.get(item.id, {})
        contexts = (artifact_rows.get(item.id, {}).get("pipeline") or {}).get("contexts") or []
        recall = retrieval.get("recall@5")
        answer_values = [answer.get(name) for name in ("faithfulness", "answer_relevance", "citation_accuracy")]
        good_retrieval = recall is not None and recall >= .8
        good_answer = all(value is not None and value >= .8 for value in answer_values)
        rows.append({"id": item.id, "query": item.query, "recall@5": recall,
                     "faithfulness": answer.get("faithfulness"), "answer_relevance": answer.get("answer_relevance"),
                     "citation_accuracy": answer.get("citation_accuracy"),
                     "diagnosis": "检索好、生成差" if good_retrieval and not good_answer else ("正常" if good_retrieval and good_answer else "检索失败向下传导"),
                     "collection_error": artifact_rows.get(item.id, {}).get("collection_error")})
    retrieval_aggregate = (retrieval_result or {}).get("aggregate", {})
    return {"serving_retrieval": {"aggregate": retrieval_aggregate, "per_query": retrieval_rows},
            "answer": answer_result.get("metrics", answer_result.get("aggregate", {})), "per_query": rows,
            "answer_counts": answer_result.get("counts", {}),
            "generation_success_rate": answer_result.get("generation_success_rate", 0.0),
            "judge_success_rate": answer_result.get("judge_success_rate", {})}


def suite_gate(suite_result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    answer_gate = gate_answer_metrics(suite_result.get("answer", {}), baseline.get("answer", {}).get("metrics", {}), baseline.get("answer", {}).get("floors", {}))
    judge_ok = all(value >= .95 for value in suite_result.get("judge_success_rate", {}).values())
    generation_ok = suite_result.get("generation_success_rate") == 1.0
    comparable, differences = comparable_provenance(suite_result.get("provenance", {}), baseline.get("provenance", {})) if baseline.get("provenance") else (True, {})
    retrieval_checks = {}
    retrieval_ok = True
    for leg, current in suite_result.get("retrieval_legs", {}).items():
        base = baseline.get("retrieval", {}).get("legs", {}).get(leg, {})
        if base:
            passed, deltas = retrieval_gate(current, base)
            retrieval_checks[leg] = {"passed": passed, "deltas": deltas}
            retrieval_ok = retrieval_ok and passed
    return {"passed": answer_gate["passed"] and judge_ok and generation_ok and comparable and retrieval_ok,
            "answer": answer_gate, "generation_success": generation_ok, "judge_success": judge_ok,
            "retrieval": retrieval_checks, "provenance_comparable": comparable, "provenance_differences": differences}


def render_suite_report(result: dict[str, Any], gate: dict[str, Any] | None = None) -> str:
    lines = ["# RAG 评测综合报告", "", "| 层次 | 指标 | 当前 |", "|---|---|---:|"]
    for key, value in result.get("serving_retrieval", {}).get("aggregate", {}).items():
        lines.append(f"| Serving Retrieval | {key} | {value:.4f} |" if value is not None else f"| Serving Retrieval | {key} | N/A |")
    for key, value in result.get("answer", {}).items():
        lines.append(f"| Answer | {key} | {value:.4f} |" if value is not None else f"| Answer | {key} | N/A |")
    lines += ["", "## 逐题诊断", "", "| ID | Recall@5 | Faithfulness | Relevance | Citation | 诊断 |", "|---|---:|---:|---:|---:|---|"]
    for row in result.get("per_query", []):
        fmt = lambda x: "N/A" if x is None else f"{x:.4f}"
        lines.append(f"| {row['id']} | {fmt(row.get('recall@5'))} | {fmt(row.get('faithfulness'))} | {fmt(row.get('answer_relevance'))} | {fmt(row.get('citation_accuracy'))} | {row['diagnosis']} |")
    if gate is not None:
        lines += ["", f"**门禁：{'通过' if gate.get('passed') else '不通过'}**"]
    return "\n".join(lines) + "\n"


async def _run_suite(args) -> int:
    from rag.config import get_settings
    from rag.eval.answer_run import _collect_run, _score_run
    from rag.eval.artifacts import list_items, load_manifest
    from rag.eval.answer_harness import select_answer_items
    from rag.eval.harness import build_retriever, load_golden, run_eval
    from rag.models.embedding import EmbeddingModel

    settings = get_settings()
    if args.seed is None:
        args.seed = settings.EVAL_RANDOM_SEED
    # 先创建答案 run，保证两层使用同一份选样 manifest。
    run_args = argparse.Namespace(limit=args.limit, seed=args.seed, ids=args.ids, all_items=args.all_items, all=args.all_items, resume=None, collect_only=False)
    run_id = await _collect_run(run_args, settings)
    answer_result = await _score_run(run_id, settings)
    artifacts = list_items(Path(__file__).resolve().parent / "history", run_id)
    items = select_answer_items(load_golden(Path(__file__).resolve().parent / "datasets" / "retrieval_golden.jsonl"), limit=args.limit, seed=args.seed, ids=args.ids.split(",") if args.ids else None, all_items=args.all_items)
    pool, retriever = await build_retriever(settings)
    try:
        retrieval_result = await run_eval(items, pool, EmbeddingModel(settings), retriever, ks=(1, 3, 5), top_k=5)
    finally:
        await pool.close()
    serving = serving_retrieval(items, artifacts)
    result = join_suite_results(items, artifacts, answer_result, serving)
    result["retrieval_legs"] = {key: value.get("aggregate", {}) for key, value in retrieval_result.items() if isinstance(value, dict) and "aggregate" in value}
    manifest = load_manifest(Path(__file__).resolve().parent / "history", run_id)
    result["provenance"] = {key: manifest.get(key) for key in ("sample_fingerprint", "selected_ids", "seed", "generator_model", "judge_model", "embedding_model", "judge_prompt_hash")}
    baseline = load_baseline(Path(__file__).resolve().parent / "baseline.json")
    gate = suite_gate(result, baseline)
    report = render_suite_report(result, gate)
    Path(__file__).resolve().parent.joinpath("history", run_id, "suite-report.md").write_text(report, encoding="utf-8")
    print(report)
    if args.update_baseline:
        provenance = {key: manifest.get(key) for key in ("sample_fingerprint", "selected_ids", "seed", "generator_model", "judge_model", "embedding_model", "judge_prompt_hash")}
        update_baseline(Path(__file__).resolve().parent / "baseline.json", provenance=provenance, retrieval={"legs": result["retrieval_legs"], "serving": result["serving_retrieval"]["aggregate"]}, answer_metrics=result["answer"], run_id=run_id, reason=args.reason or "", accept_regression=args.accept_regression)
        return 0
    return 0 if gate["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="检索 + 生成综合评测门禁")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ids")
    parser.add_argument("--all", action="store_true", dest="all_items")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--accept-regression", action="store_true")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.accept_regression and not (args.reason or "").strip():
        raise SystemExit("--accept-regression 必须同时提供 --reason")
    if args.accept_regression and not args.update_baseline:
        raise SystemExit("--accept-regression 只能与 --update-baseline 一起使用")
    raise SystemExit(asyncio.run(_run_suite(args)))


if __name__ == "__main__":
    main()
