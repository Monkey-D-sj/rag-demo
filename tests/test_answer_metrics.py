from rag.eval.answer_metrics import (
    CitationJudgeOutput,
    aggregate_answer_metrics,
    normalize_answer,
    score_citations,
)


def test_normalize_answer_removes_bibliography_and_inline_tokens():
    normalized = normalize_answer("事实[1][2]。\n\n[1]: 《甲》\n[2]: 《乙》")
    assert normalized.answer_body_with_citations == "事实[1][2]。"
    assert normalized.answer_body_plain == "事实。"
    assert normalized.bibliography == [{"index": 1, "text": "《甲》"}, {"index": 2, "text": "《乙》"}]


def test_citation_formula_handles_supported_unsupported_invalid_and_uncited():
    result = score_citations({"claims": [
        {"claim": "a", "is_factual": True, "citation_indices": [1], "links": [{"index": 1, "valid": True, "supported": True, "reason": ""}]},
        {"claim": "b", "is_factual": True, "citation_indices": [2], "links": [{"index": 2, "valid": True, "supported": False, "reason": "弱支撑"}]},
        {"claim": "c", "is_factual": True, "citation_indices": [9], "links": [{"index": 9, "valid": False, "supported": False, "reason": "不存在"}]},
        {"claim": "d", "is_factual": True, "citation_indices": [], "links": []},
        {"claim": "opinion", "is_factual": False, "citation_indices": [], "links": []},
    ]}, [1, 2])
    assert result["citation_accuracy"] == 1 / 4
    assert result["factual_claim_count"] == 4
    assert result["uncited_factual_claim_count"] == 1


def test_aggregate_skips_null_scores_and_reports_counts():
    result = aggregate_answer_metrics([
        {"faithfulness": 1, "answer_relevance": None, "citation_accuracy": .5},
        {"faithfulness": 0, "answer_relevance": .8, "citation_accuracy": None},
    ])
    assert result["metrics"]["faithfulness"] == .5
    assert result["counts"]["answer_relevance"]["skipped_count"] == 1


def test_no_factual_claims_is_na():
    assert score_citations(CitationJudgeOutput(claims=[]))["citation_accuracy"] is None

