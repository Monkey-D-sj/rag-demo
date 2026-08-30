import pytest

from rag.eval.baseline import gate_answer_metrics, gate_with_boundary_rereview, update_baseline


def test_floor_and_relative_regression_gate():
    result = gate_answer_metrics({"faithfulness": .8, "answer_relevance": .9, "citation_accuracy": .7},
                                 {"faithfulness": .9, "answer_relevance": .9, "citation_accuracy": .9},
                                 {"faithfulness": .85, "answer_relevance": None, "citation_accuracy": None})
    assert not result["passed"]
    assert not result["metrics"]["faithfulness"]["passed"]


def test_boundary_rereview_uses_median():
    result = gate_with_boundary_rereview({"faithfulness": .95, "answer_relevance": 1., "citation_accuracy": 1.},
                                         {"faithfulness": 1., "answer_relevance": 1., "citation_accuracy": 1.},
                                         rereview=lambda key: [.98, .99])
    assert result["metrics"]["faithfulness"]["rereview_median"] == .98


def test_baseline_regression_requires_reason(tmp_path):
    path = tmp_path / "baseline.json"
    update_baseline(path, provenance={}, retrieval={}, answer_metrics={"faithfulness": .9}, run_id="r1")
    with pytest.raises(ValueError):
        update_baseline(path, provenance={}, retrieval={}, answer_metrics={"faithfulness": .8}, run_id="r2")
    updated = update_baseline(path, provenance={}, retrieval={}, answer_metrics={"faithfulness": .8}, run_id="r2",
                              accept_regression=True, reason="切换模型")
    assert updated["answer"]["metrics"]["faithfulness"] == .8

