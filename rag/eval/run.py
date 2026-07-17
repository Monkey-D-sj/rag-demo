import argparse
import asyncio
import json
import re
import subprocess
from datetime import datetime, timezone

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.harness import build_retriever, eval_out_of_scope, load_golden, run_eval
from rag.eval.metrics import aggregate_by_category, classify, gate, rewrite_gate
from rag.models.embedding import EmbeddingModel

from rag.eval.style import (
    best_val,
    bold,
    delta_str,
    dim,
    drop_arrow,
    failure,
    green,
    red,
    success,
    worst_val,
)

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"
HISTORY_DIR = EVAL_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
# Top-K 评估粒度：对每条 query 分别计算 hit@k / recall@k / ndcg@k。
# 只影响报告输出，不影响门禁——门禁固定用 recall@5 和 mrr（见 gate() 默认 keys）。
KS = (3, 5)

# 原查询和改写查询
_LEG_LABEL = {
    "fused": "混合(改)",
    "fused_reranked": "混合重排",
    "graph_fused": "图谱融合",
    "raw": "混合(原)",
    "vec_only": "向量(改)",
    "raw_vec": "向量(原)",
    "bm25_only": "BM25(改)",
    "raw_bm25": "BM25(原)",
}
_LEG_ORDER = ("fused", "fused_reranked", "graph_fused", "raw", "vec_only", "raw_vec", "bm25_only", "raw_bm25")
_CAT_LABEL = {
    "basic": "基础召回",
    "chunk_boundary": "Chunk边界",
    "synonym": "同义词",
    "metadata": "Metadata过滤",
    "multi_chunk": "多Chunk聚合",
    "multi_doc": "多文档聚合",
    "out_of_scope": "无答案",
    "ambiguous": "歧义问题",
    "long_tail": "长尾问题",
}
_CAT_ORDER = (
    "basic", "chunk_boundary", "synonym", "metadata",
    "multi_chunk", "multi_doc", "ambiguous", "long_tail", "out_of_scope",
)
_LEG_KEYS = frozenset(_LEG_ORDER)  # result 中非 leg 键（如 "categories"）的过滤
_CJK_RANGES = [
    (0x1100, 0x115F), (0x2E80, 0xA4CF), (0xA960, 0xA97F),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F),
    (0xFF01, 0xFF60), (0xFFE0, 0xFFE6), (0x1F000, 0x1F9FF),
]


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _disp_width(s: str) -> int:
    """计算字符串的终端显示宽度（CJK 字符占 2 列，ANSI 转义不计）。"""
    s = _ANSI_RE.sub("", s)
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

    # ── 标题 ──
    print(bold("📊 指标总览"))

    # ── 计算每行最佳/最差值用于高亮 ──
    row_range: dict[str, tuple[float, float]] = {}
    for key in all_keys:
        vals = [result[leg]["aggregate"].get(key) for leg in legs]
        nums = [v for v in vals if v is not None]
        if nums:
            row_range[key] = (max(nums), min(nums))

    # ── 表头 ──
    header = _pad("指标", 10, left=False)
    for leg in legs:
        header += _pad(bold(_LEG_LABEL[leg]), col_w)
    print(header)
    print("─" * _disp_width(header))

    # ── 数据行 ──
    sorted_keys = sorted(all_keys)
    prev_prefix: str | None = None
    line_w = _disp_width(header)
    for key in sorted_keys:
        # 不同指标族之间加分隔线（hit@ / recall@ / ndcg@ / mrr）
        prefix = key.split("@")[0]
        if prev_prefix is not None and prefix != prev_prefix:
            print(dim("─" * line_w))
        prev_prefix = prefix

        row = _pad(key, 10, left=False)
        best, worst = row_range.get(key, (None, None))
        for leg in legs:
            v = result[leg]["aggregate"].get(key)
            if v is not None:
                s = f"{v:.4f}"
                if len(legs) > 1 and best is not None and v == best:
                    s = best_val(s)
                elif len(legs) > 1 and worst is not None and v == worst:
                    s = worst_val(s)
                row += _pad(s, col_w)
            else:
                row += _pad("—", col_w)
        print(row)
    print()

    # ── 改写收益摘要 ──
    if "raw" in result:
        print(bold("  🔄 改写收益"))
        pairs = [
            ("混合", "fused", "raw"),
            ("向量", "vec_only", "raw_vec"),
            ("BM25", "bm25_only", "raw_bm25"),
        ]
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
                deltas.append(f"{key}: {delta_str(d)}")
            print(f"    {label}: {'  '.join(deltas)}")
        print()

    # ── 重排收益 ──
    if "fused_reranked" in result and "fused" in result:
        print(bold("  🔀 重排收益"))
        rerank_agg = result["fused_reranked"]["aggregate"]
        fused_agg = result["fused"]["aggregate"]
        deltas = []
        for key in sorted(fused_agg):
            if key not in rerank_agg:
                continue
            d = rerank_agg[key] - fused_agg[key]
            deltas.append(f"{key}: {delta_str(d)}")
        print(f"    {'  '.join(deltas)}\n")


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
            rows.append((
                f"{_LEG_LABEL[leg]}.{key}",
                f"{d['current']:.4f}",
                f"{drop_arrow(d['rel_drop'])} (base={d['baseline']:.4f})",
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
                rows.append((
                    f"改写{label}.{key}",
                    f"{d['rewritten']:.4f}",
                    f"{drop_arrow(d['rel_drop'])} (raw={d['raw']:.4f})",
                ))

    if not rows:
        print(red("  ❌ 无 baseline 数据，请先 --update-baseline 生成基线\n"))
        return False

    print(bold("🚦 门禁检查"))
    hdr = f"{'指标':30s}{'现状':>10s}  偏差"
    print(bold(hdr))
    print("─" * 65)
    for metric, cur, delta in rows:
        # 回归行标红，正常行标绿
        icon = green("✓") if "▲" in delta else red("✗")
        print(f"{icon} {metric:28s}{cur:>10s}  {delta}")
    print()

    if all_passed:
        print(success(" 门禁通过 — 所有指标不低于 baseline"))
    else:
        print(failure(" 门禁不通过 — 存在指标退化"))
    print()

    return all_passed


def _print_category_table(result: dict) -> None:
    """按题型分类展示每条腿的指标，方便定位薄弱环节。"""
    id_to_cat = result.get("categories", {})
    if not id_to_cat:
        return

    key_metrics = ["recall@5", "mrr", "ndcg@5"]
    # 取第一个 leg 的 aggregate 来确定实际存在的指标
    sample_leg = next((result[k] for k in _LEG_ORDER if k in result), None)
    if sample_leg is None:
        return
    metrics = [m for m in key_metrics if m in sample_leg["aggregate"]]

    for leg in [l for l in _LEG_ORDER if l in result]:
        per_query = result[leg].get("per_query", [])
        if not per_query:
            continue
        by_cat = aggregate_by_category(per_query, id_to_cat)
        if not by_cat:
            continue

        cat_w = max(max(_disp_width(_CAT_LABEL.get(c, c)) for c in by_cat), 12)
        metric_w = 10

        # ── 计算每列最佳/最差 ──
        col_range: dict[str, tuple[float, float]] = {}
        for m in metrics:
            vals = [by_cat[c].get(m) for c in by_cat]
            nums = [v for v in vals if v is not None]
            if nums:
                col_range[m] = (max(nums), min(nums))

        title = f"📋 题型分类 — {_LEG_LABEL.get(leg, leg)}"
        print(f"\n{bold(title)}")

        header = _pad("类型", cat_w, left=False)
        for m in metrics:
            header += _pad(m, metric_w)
        header += _pad("数量", 6)
        print(bold(header))
        print("─" * _disp_width(header))

        for cat in _CAT_ORDER:
            if cat not in by_cat:
                continue
            agg = by_cat[cat]
            display = _CAT_LABEL.get(cat, cat)
            row = _pad(display, cat_w, left=False)
            for m in metrics:
                v = agg.get(m)
                if v is not None:
                    s = f"{v:.4f}"
                    best, worst = col_range.get(m, (None, None))
                    if len(by_cat) > 1 and best is not None and v == best:
                        s = best_val(s)
                    elif len(by_cat) > 1 and worst is not None and v == worst:
                        s = worst_val(s)
                    row += _pad(s, metric_w)
                else:
                    row += _pad("—", metric_w)
            cat_count = sum(1 for pq in per_query if id_to_cat.get(pq.get("id", "")) == cat)
            row += _pad(str(cat_count), 6)
            print(row)
        print()


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


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _save_history(result: dict) -> None:
    """每次评测保存为一个独立文件：history/YYYYMMDD-HHMMSS-{commit}.json。"""
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{ts}.json"
    legs = {k: result[k]["aggregate"] for k in result if k in _LEG_KEYS}
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_sha(),
        "legs": legs,
        "per_query": {
            leg: result[leg].get("per_query", [])
            for leg in result if leg in _LEG_KEYS and "per_query" in result[leg]
        },
        "categories": result.get("categories", {}),
        "per_category": {},
    }
    # 按类别聚合每条腿
    id_to_cat = result.get("categories", {})
    if id_to_cat:
        for leg in result:
            if leg not in _LEG_KEYS:
                continue
            per_query = result[leg].get("per_query", [])
            if per_query:
                record["per_category"][leg] = aggregate_by_category(per_query, id_to_cat)

    filepath = HISTORY_DIR / name
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"评测结果已保存 -> {filepath}\n")


def main() -> None:
    from rag.common.platform import setup_windows_loop

    setup_windows_loop()

    parser = argparse.ArgumentParser(description="检索层评测（含查询改写与范围判断）")
    parser.add_argument("--update-baseline", action="store_true", help="用本次结果刷新全部 baseline")
    parser.add_argument("--classify", action="store_true", help="跑范围判断 LLM 分类评测")
    parser.add_argument("--category", type=str, default=None,
                        help="只评测指定类别（basic, synonym, chunk_boundary, …）")
    args = parser.parse_args()

    settings = get_settings()
    items = load_golden(GOLDEN_PATH)
    if not items:
        raise SystemExit("golden 集为空，请先构造 retrieval_golden.jsonl")
    print("评测集加载完成")

    # ── 按类别筛选 ──
    if args.category:
        valid = sorted({it.category for it in items})
        items = [it for it in items if it.category == args.category]
        if not items:
            raise SystemExit(
                f"类别 '{args.category}' 无匹配条目，可用: {', '.join(valid)}"
            )
        print(f"筛选类别 '{args.category}': {len(items)} 条 (in-scope {sum(1 for it in items if not it.out_of_scope)})\n")

    async def _do() -> dict:
        pool, retriever = await build_retriever(settings)
        print("检索器初始化成功")
        try:
            embedding = EmbeddingModel(settings)
            print("embedding模型初始化成功")
            reranker = None
            if settings.RERANK_ENABLED and settings.RERANK_BASE_URL:
                from rag.models.rerank import QwenReranker
                reranker = QwenReranker(settings)
                print("rerank模型初始化成功")
            return await run_eval(items, pool, embedding, retriever, reranker=reranker, ks=KS, top_k=max(KS))
        finally:
            await pool.close()

    result = asyncio.run(_do())

    # ── 持久化历史记录 ──
    _save_history(result)

    # ── 指标表格 ──
    _print_table(result)

    # ── 题型分类评测 ──
    _print_category_table(result)

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
        new_baseline = {leg: result[leg]["aggregate"] for leg in result if leg in _LEG_KEYS}
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_baseline, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"已更新 baseline -> {BASELINE_PATH}")
        return

    # ── 门禁 ──
    baseline = _load_baseline()
    all_passed = _print_gate(result, baseline)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
