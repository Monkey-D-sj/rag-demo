"""检索/生成共享 baseline loader 及受保护更新。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.eval.answer_metrics import relative_drop


def load_baseline(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"schema_version": 2, "provenance": {}, "retrieval": {"legs": {}, "serving": {}}, "answer": {"metrics": {}, "floors": {}}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") == 2:
        data.setdefault("retrieval", {}).setdefault("legs", {})
        data.setdefault("retrieval", {}).setdefault("serving", {})
        data.setdefault("answer", {}).setdefault("metrics", {})
        data.setdefault("answer", {}).setdefault("floors", {})
        return data
    # v1 retrieval baseline 是顶层 leg -> aggregate。
    legs = {key: value for key, value in data.items() if isinstance(value, dict) and key not in {"provenance", "answer"}}
    return {"schema_version": 2, "provenance": {}, "retrieval": {"legs": legs, "serving": {}}, "answer": {"metrics": {}, "floors": {}}}


def comparable_provenance(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, dict[str, tuple[Any, Any]]]:
    keys = ("sample_fingerprint", "selected_ids", "generator_model", "judge_model", "embedding_model", "judge_prompt_hash")
    differences = {key: (baseline.get(key), current.get(key)) for key in keys if baseline.get(key) != current.get(key)}
    return not differences, differences


def gate_answer_metrics(current: dict[str, float | None], baseline: dict[str, float], floors: dict[str, float | None] | None = None,
                        tolerance: float = 0.03) -> dict[str, Any]:
    floors = floors or {}
    rows: dict[str, Any] = {}
    passed = True
    for key in ("faithfulness", "answer_relevance", "citation_accuracy"):
        value, base, floor = current.get(key), baseline.get(key), floors.get(key)
        row = {"current": value, "baseline": base, "floor": floor, "rel_drop": None, "passed": True}
        if value is None:
            row["passed"] = False
        if floor is not None and (value is None or value < floor):
            row["passed"] = False
        if base is not None and value is not None:
            row["rel_drop"] = relative_drop(float(value), float(base))
            if row["rel_drop"] > tolerance:
                row["passed"] = False
        elif base is not None:
            row["passed"] = False
        rows[key] = row
        passed = passed and row["passed"]
    return {"passed": passed, "metrics": rows, "floors_calibrated": all(floors.get(k) is not None for k in ("faithfulness", "answer_relevance", "citation_accuracy"))}


def gate_with_boundary_rereview(current: dict[str, float | None], baseline: dict[str, float],
                                floors: dict[str, float | None] | None = None,
                                rereview: Any | None = None) -> dict[str, Any]:
    """3%~5% 的指标只重评分，不重新采集；rereview 返回两次额外原始分数。"""
    first = gate_answer_metrics(current, baseline, floors)
    for key, row in first["metrics"].items():
        drop = row.get("rel_drop")
        if drop is None or not .03 < drop <= .05 + 1e-9:
            continue
        if rereview is None:
            row["passed"] = False
            first["passed"] = False
            row["rereview_error"] = "需要提供同一已生成样本的二次评分"
            continue
        scores = [float(current[key]), *[float(x) for x in rereview(key)]]
        final = sorted(scores)[1]
        row["rereview_scores"] = scores
        row["rereview_median"] = final
        row["current"] = final
        row["rel_drop"] = relative_drop(final, float(baseline[key]))
        row["passed"] = row["rel_drop"] <= .03 and (row["floor"] is None or final >= row["floor"])
        first["passed"] = all(x.get("passed") for x in first["metrics"].values())
    return first


def update_baseline(path: str | Path, *, provenance: dict[str, Any], retrieval: dict[str, Any], answer_metrics: dict[str, float | None],
                    floors: dict[str, float | None] | None = None, run_id: str = "", reason: str = "",
                    accept_regression: bool = False) -> dict[str, Any]:
    path = Path(path)
    previous = load_baseline(path)
    old = previous.get("answer", {}).get("metrics", {})
    provenance = provenance or previous.get("provenance", {})
    if accept_regression and not reason.strip():
        raise ValueError("--accept-regression 必须提供非空 reason")
    if previous.get("provenance") and previous["provenance"] != provenance and not accept_regression:
        raise ValueError("baseline provenance 已变化，需显式接受实验重建")
    if old and not accept_regression:
        lowered = [key for key, value in answer_metrics.items() if value is not None and old.get(key) is not None and value < old[key]]
        if lowered:
            raise ValueError(f"普通 baseline 更新不允许指标下降: {', '.join(lowered)}")
    result = {"schema_version": 2, "provenance": provenance,
              "retrieval": {"legs": retrieval.get("legs", retrieval), "serving": retrieval.get("serving", {})},
              "answer": {"metrics": answer_metrics or old, "floors": floors if floors is not None else previous.get("answer", {}).get("floors", {})},
              "updated_at": datetime.now(timezone.utc).isoformat(), "updated_by_run": run_id,
              "update_reason": reason}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
