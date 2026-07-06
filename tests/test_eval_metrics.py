from rag.eval.metrics import evaluate_query, normalize


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
