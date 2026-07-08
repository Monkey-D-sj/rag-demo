import argparse
import asyncio
import json

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.harness import build_retriever, load_golden, run_eval
from rag.eval.metrics import gate
from rag.models.embedding import EmbeddingModel

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"
# Top-K 评估粒度：对每条 query 分别计算 hit@k / recall@k / ndcg@k。
# 只影响报告输出，不影响门禁——门禁固定用 recall@5 和 mrr（见 gate() 默认 keys）。
KS = (1, 3, 5)

_LEG_LABEL = {"fused": "混合召回", "vec_only": "纯向量", "bm25_only": "纯 BM25"}
_LEGS = ("fused", "vec_only", "bm25_only")
_COL_WIDTH = 12


def _load_baseline() -> dict:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _print_table(result: dict) -> None:
    """三条路指标合并成一张表格。"""
    all_keys: list[str] = []
    for leg in _LEGS:
        for k in result[leg]["aggregate"]:
            if k not in all_keys:
                all_keys.append(k)

    # 表头
    header = f"{'指标':10s}"
    for leg in _LEGS:
        header += f"{_LEG_LABEL[leg]:>{_COL_WIDTH}s}"
    print(header)
    print("-" * len(header))

    # 数据行
    for key in sorted(all_keys):
        row = f"{key:10s}"
        for leg in _LEGS:
            v = result[leg]["aggregate"].get(key)
            row += f"{v:>{_COL_WIDTH}.4f}" if v is not None else f"{'—':>{_COL_WIDTH}s}"
        print(row)
    print()


def _print_gate(result: dict, baseline: dict) -> bool:
    """三条路门禁结果合并打印。"""
    all_passed = True
    rows: list[tuple[str, str, str]] = []  # (指标, 现状, 偏差)

    for leg in _LEGS:
        leg_baseline = baseline.get(leg, {})
        cur_agg = result[leg]["aggregate"]
        passed, deltas = gate(cur_agg, leg_baseline)
        if not passed:
            all_passed = False

        for key, d in deltas.items():
            arrow = "↓" if d["rel_drop"] > 0 else "↑"
            rows.append((
                f"{_LEG_LABEL[leg]}.{key}",
                f"{d['current']:.4f}",
                f"{arrow}{abs(d['rel_drop']):.1%} (base={d['baseline']:.4f})",
            ))

    if not rows:
        print("无 baseline 数据，跳过门禁（请先 --update-baseline）\n")
        return True

    header = f"{'指标':30s}{'现状':>10s}  偏差"
    print(header)
    print("-" * 65)
    for metric, cur, delta in rows:
        print(f"{metric:30s}{cur:>10s}  {delta}")
    print()

    return all_passed


def main() -> None:
    from rag.common.platform import setup_windows_loop

    setup_windows_loop()

    parser = argparse.ArgumentParser(description="检索层评测（fused / vec-only / bm25-only）")
    parser.add_argument("--update-baseline", action="store_true", help="用本次结果刷新全部 baseline")
    args = parser.parse_args()

    settings = get_settings()
    settings.check_required()
    items = load_golden(GOLDEN_PATH)
    if not items:
        raise SystemExit("golden 集为空，请先构造 retrieval_golden.jsonl")

    async def _do() -> dict:
        pool, retriever = await build_retriever(settings)
        try:
            embedding = EmbeddingModel(settings)
            return await run_eval(items, pool, embedding, retriever, ks=KS, top_k=max(KS))
        finally:
            await pool.close()

    result = asyncio.run(_do())

    # ── 三条路指标表格 ──
    _print_table(result)

    # ── 更新 baseline ──
    if args.update_baseline:
        new_baseline = {
            leg: result[leg]["aggregate"] for leg in _LEGS
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_baseline, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"已更新 baseline -> {BASELINE_PATH}")
        return

    # ── 门禁 ──
    baseline = _load_baseline()
    all_passed = _print_gate(result, baseline)

    if not all_passed:
        raise SystemExit("检索指标回归：核心指标跌破容差")
    print("全部门禁通过")


if __name__ == "__main__":
    main()
