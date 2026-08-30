from rag.eval.harness import GoldenItem
from rag.eval.suite_run import join_suite_results, render_suite_report, serving_retrieval


def test_suite_join_is_by_id_and_marks_retrieval_good_answer_bad():
    items = [GoldenItem("q2", "第二题", ["gold"]), GoldenItem("q1", "第一题", ["gold"])]
    artifacts = [{"id": "q1", "query": "第一题", "pipeline": {"contexts": [{"text": "gold"}]}},
                 {"id": "q2", "query": "第二题", "pipeline": {"contexts": [{"text": "gold"}]}}]
    answer = {"metrics": {"faithfulness": .5, "answer_relevance": .9, "citation_accuracy": .5},
              "per_query": [{"id": "q1", "faithfulness": .5, "answer_relevance": .9, "citation_accuracy": .5},
                            {"id": "q2", "faithfulness": .9, "answer_relevance": .9, "citation_accuracy": .9}]}
    retrieval = serving_retrieval(items, artifacts)
    result = join_suite_results(items, artifacts, answer, retrieval)
    assert result["per_query"][0]["id"] == "q2"
    assert result["per_query"][1]["diagnosis"] == "检索好、生成差"
    assert "Citation" in render_suite_report(result)

