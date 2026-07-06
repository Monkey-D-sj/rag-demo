import argparse
import asyncio
import json

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"
KS = (1, 3, 5)


async def _run() -> dict:
    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    if not items:
        raise SystemExit("golden 集为空，请先构造 retrieval_golden.jsonl")
    pool, retriever = await build_retriever(settings)
    try:
        return await run_eval(items, retriever, ks=KS, top_k=max(KS))
    finally:
        await pool.close()


def _load_baseline() -> dict:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def main() -> None:
    import sys

    if sys.platform == "win32":
        import asyncio as _asyncio

        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="检索层评测")
    parser.add_argument("--update-baseline", action="store_true", help="用本次结果刷新 baseline")
    args = parser.parse_args()

    result = asyncio.run(_run())
    agg = result["aggregate"]

    print("== 检索评测聚合指标 ==")
    for key in sorted(agg):
        print(f"  {key:10s} {agg[key]:.4f}")

    if args.update_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"已更新 baseline -> {BASELINE_PATH}")
        return

    baseline = _load_baseline()
    passed, deltas = gate(agg, baseline)
    print("== 与 baseline 对比 ==")
    for key, d in deltas.items():
        print(
            f"  {key}: base={d['baseline']:.4f} cur={d['current']:.4f} "
            f"rel_drop={d['rel_drop']:+.2%}"
        )
    if not passed:
        raise SystemExit("检索指标回归：核心指标跌破容差")
    print("门禁通过")


if __name__ == "__main__":
    main()
