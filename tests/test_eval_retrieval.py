import json

import pytest

from rag.config import get_settings
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate
from rag.eval.run import BASELINE_PATH, GOLDEN_PATH, KS
from rag.models.embedding import EmbeddingModel


@pytest.mark.eval
async def test_retrieval_no_regression():
    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    assert items, "golden 集为空"

    pool, retriever = await build_retriever(settings)
    try:
        embedding = EmbeddingModel(settings)
        result = await run_eval(items, pool, embedding, retriever, ks=KS, top_k=max(KS))
    finally:
        await pool.close()

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    # 门禁只检查 fused（生产路径），vec_only/bm25_only 仅为诊断参考
    fused_baseline = baseline.get("fused", {}) if isinstance(baseline, dict) else baseline
    passed, deltas = gate(result["fused"]["aggregate"], fused_baseline)
    assert passed, f"检索指标回归：{deltas}"
