import json

import pytest

from rag.config import get_settings
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate
from rag.eval.run import BASELINE_PATH, GOLDEN_PATH, KS


@pytest.mark.eval
async def test_retrieval_no_regression():
    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    assert items, "golden 集为空"

    pool, retriever = await build_retriever(settings)
    try:
        result = await run_eval(items, retriever, ks=KS, top_k=max(KS))
    finally:
        await pool.close()

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    passed, deltas = gate(result["aggregate"], baseline)
    assert passed, f"检索指标回归：{deltas}"
