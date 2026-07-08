import argparse
import asyncio
import json

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.harness import build_retriever, eval_out_of_scope, load_golden, run_eval
from rag.eval.metrics import classify, gate, rewrite_gate
from rag.models.embedding import EmbeddingModel

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"
# Top-K 评估粒度：对每条 query 分别计算 hit@k / recall@k / ndcg@k。
# 只影响报告输出，不影响门禁——门禁固定用 recall@5 和 mrr（见 gate() 默认 keys）。
KS = (1, 3, 5)

_LEG_LABEL = {
    "fused": "混合(改)",
    "fused_reranked": "混合重排",
    "raw": "混合(原)",
    "vec_only": "向量(改)",
    "raw_vec": "向量(原)",
    "bm25_only": "BM25(改)",
    "raw_bm25": "BM25(原)",
}
_LEG_ORDER = ("fused", "fused_reranked", "raw", "vec_only", "raw_vec", "bm25_only", "raw_bm25")
_CJK_RANGES = [
    (0x1100, 0x115F), (0x2E80, 0xA4CF), (0xA960, 0xA97F),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F),
    (0xFF01, 0xFF60), (0xFFE0, 0xFFE6), (0x1F000, 0x1F9FF),
]


def _disp_width(s: str) -> int:
    """计算字符串的终端显示宽度（CJK 字符占 2 列）。"""
    w = 0
    for ch in s:
        cp = ord(ch)
        w += 2 if any(lo <= cp <= hi for lo, hi in _CJK_RANGES) else 1
    return w


def _pad(s: str, width: int, left: bool = True) -> str:
    """按显示宽度补齐到 width。"""
    need = width - _disp_width(s)
    if need <= 0:
        return s
    return (" " * need) + s if left else s + (" " * need)


def _load_baseline() -> dict:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _print_table(result: dict) -> None:
    """多路指标合并成一张表格。"""
    legs = [l for l in _LEG_ORDER if l in result]
    col_w = max(10, 60 // len(legs))

    all_keys: list[str] = []
    for leg in legs:
        for k in result[leg]["aggregate"]:
            if k not in all_keys:
                all_keys.append(k)

    # 表头
    header = _pad("指标", 10, left=False)
    for leg in legs:
        header += _pad(_LEG_LABEL[leg], col_w)
    print(header)
    print("-" * _disp_width(header))

    # 数据行
    for key in sorted(all_keys):
        row = _pad(key, 10, left=False)
        for leg in legs:
            v = result[leg]["aggregate"].get(key)
            row += _pad(f"{v:.4f}", col_w) if v is not None else _pad("—", col_w)
        print(row)
    print()

    # 改写收益摘要（分路展示）
    if "raw" in result:
        pairs = [
            ("改写收益(混合)", "fused", "raw"),
            ("改写收益(向量)", "vec_only", "raw_vec"),
            ("改写收益(BM25)", "bm25_only", "raw_bm25"),
        ]
        lines = []
        for label, rw_leg, raw_leg in pairs:
            if rw_leg not in result or raw_leg not in result:
                continue
            rw_agg = result[rw_leg]["aggregate"]
            raw_agg = result[raw_leg]["aggregate"]
            deltas = []
            for key in sorted(raw_agg):
                if key not in rw_agg:
                    continue
                d = rw_agg[key] - raw_agg[key]
                deltas.append(f"{key}: {d:+.4f}")
            lines.append(f"  {label}: {'  '.join(deltas)}")
        print("\n".join(lines) + "\n")

    # 重排收益
    if "fused_reranked" in result and "fused" in result:
        rerank_agg = result["fused_reranked"]["aggregate"]
        fused_agg = result["fused"]["aggregate"]
        deltas = []
        for key in sorted(fused_agg):
            if key not in rerank_agg:
                continue
            d = rerank_agg[key] - fused_agg[key]
            deltas.append(f"{key}: {d:+.4f}")
        print(f"  重排收益: {'  '.join(deltas)}\n")


def _print_gate(result: dict, baseline: dict) -> bool:
    """多路门禁，含改写质量检查。"""
    all_passed = True
    rows: list[tuple[str, str, str]] = []

    legs = [l for l in ("fused", "fused_reranked", "vec_only", "bm25_only") if l in result]
    for leg in legs:
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

    # 改写质量：分路检查改写是否降低检索质量
    if "raw" in result:
        for label, rw_leg, raw_leg in [("混合", "fused", "raw"), ("向量", "vec_only", "raw_vec"), ("BM25", "bm25_only", "raw_bm25")]:
            if rw_leg not in result or raw_leg not in result:
                continue
            passed, deltas = rewrite_gate(result[raw_leg]["aggregate"], result[rw_leg]["aggregate"])
            if not passed:
                all_passed = False
            for key, d in deltas.items():
                arrow = "↓" if d["rel_drop"] > 0 else "↑"
                rows.append((
                    f"改写{label}.{key}",
                    f"{d['rewritten']:.4f}",
                    f"{arrow}{abs(d['rel_drop']):.1%} (raw={d['raw']:.4f})",
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


async def _run_classify(items, llm) -> dict:
    """范围判断评测：跑 LLM 分类 vs golden 标注。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from rag.agent.nodes.query.query import QueryRewriteOutput
    from rag.prompts.query import system_prompt

    predicted: list[bool] = []
    actual: list[bool] = []

    for item in items:
        actual.append(item.out_of_scope)

        try:
            result: QueryRewriteOutput = await llm.ainvoke_structured(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"用户查询: {item.query}\n上下文: "),
                ],
                QueryRewriteOutput,
            )
            predicted.append(result.is_out_of_scope)
        except Exception:
            predicted.append(False)  # 降级默认 in-scope

    metrics = classify(predicted, actual)
    print("== 范围判断评测（LLM 分类 vs golden 标注） ==")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1:        {metrics['f1']:.2%}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print()

    return metrics


def main() -> None:
    from rag.common.platform import setup_windows_loop

    setup_windows_loop()

    parser = argparse.ArgumentParser(description="检索层评测（含查询改写与范围判断）")
    parser.add_argument("--update-baseline", action="store_true", help="用本次结果刷新全部 baseline")
    parser.add_argument("--classify", action="store_true", help="跑范围判断 LLM 分类评测")
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
            reranker = None
            if settings.RERANK_ENABLED and settings.RERANK_BASE_URL:
                from rag.models.rerank import QwenReranker
                reranker = QwenReranker(settings)
            return await run_eval(items, pool, embedding, retriever, reranker=reranker, ks=KS, top_k=max(KS))
        finally:
            await pool.close()

    result = asyncio.run(_do())

    # ── 指标表格 ──
    _print_table(result)

    # ── 范围判断评测 ──
    if args.classify:
        from rag.models.normal import NormalModel

        llm = NormalModel(settings)

        async def _clf() -> dict:
            return await _run_classify(items, llm)

        metrics = asyncio.run(_clf())
        if metrics["f1"] < 0.9:
            raise SystemExit("范围判断 F1 不达标")

    # ── 更新 baseline ──
    if args.update_baseline:
        new_baseline = {leg: result[leg]["aggregate"] for leg in result}
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
