"""答案忠实度离线评测 CLI。

用法：
    uv run rag-eval-answer                    # 评测
    uv run rag-eval-answer --update-baseline  # 更新基线
    uv run rag-eval-answer --gate             # 门禁检查
    uv run rag-eval-answer --generate         # 生成静态评测集
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from rag.common.logging import get_logger
from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.answer_harness import (
    generate_answer_dataset,
    load_answer_golden,
    run_faithfulness_eval,
)
from rag.eval.harness import build_retriever, load_golden
from rag.eval.metrics import gate
from rag.eval.style import bold, delta_str, failure, green, phase, red, success

logger = get_logger()

ANSWER_GOLDEN_PATH = DATASETS_DIR / "answer_golden.jsonl"
BASELINE_PATH = EVAL_DIR / "answer_baseline.json"
HISTORY_DIR = EVAL_DIR / "history"
GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_baseline() -> dict:
    try:
        with open(BASELINE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_history(result: dict) -> None:
    """保存评测历史记录，文件名加 faithfulness- 前缀。"""
    HISTORY_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"faithfulness-{ts}.json"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_sha(),
        "aggregate": result["aggregate"],
        "per_query": result["per_query"],
    }
    filepath = HISTORY_DIR / name
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"评测结果已保存 -> {filepath}\n")


def _print_result(result: dict) -> None:
    """打印 faithfulness 评分报告。"""
    agg = result["aggregate"]
    f_mean = agg["faithfulness"]

    print(bold("📊 Faithfulness 忠实度评测"))
    print(f"  均分: {f_mean:.4f}  (共 {len(result['per_query'])} 条)")
    print()

    # 列出最低分的几条（最可疑的幻觉案例）
    per_query = sorted(result["per_query"], key=lambda x: x["faithfulness"])
    if per_query:
        print(bold("🔍 低分条目 (可能存在幻觉):"))
        for pq in per_query[:5]:
            flag = red("✗") if pq["faithfulness"] < 0.5 else ""
            print(f"  {flag} [{pq['id']}] {pq['query'][:60]:60s}  faithfulness={pq['faithfulness']:.4f}")
        print()


def _print_gate(result: dict, baseline: dict) -> bool:
    """门禁检查：faithfulness 均值不得显著退化。"""
    cur_agg = result["aggregate"]
    passed, deltas = gate(cur_agg, baseline, keys=("faithfulness",))

    if not deltas:
        print(red("  ❌ 无 baseline 数据，请先 --update-baseline 生成基线\n"))
        return False

    print(bold("🚦 门禁检查"))
    for key, d in deltas.items():
        cur = d["current"]
        base = d["baseline"]
        rel = d["rel_drop"]
        icon = green("✓") if rel <= 0.03 else red("✗")
        print(f"  {icon} {key}: {cur:.4f}  base={base:.4f}  {delta_str(-rel)}")
    print()

    if passed:
        print(success(" 门禁通过 — faithfulness 不低于 baseline"))
    else:
        print(failure(" 门禁不通过 — faithfulness 显著退化"))
    print()

    return passed


async def _run_generate() -> int:
    """从 retrieval_golden.jsonl 生成 answer_golden.jsonl。"""
    from rag.models.normal import NormalModel

    settings = get_settings()
    golden_items = load_golden(GOLDEN_PATH)
    if not golden_items:
        raise SystemExit("retrieval_golden.jsonl 为空，请先构造评测集")

    pool, retriever = await build_retriever(settings)
    llm = NormalModel(settings)

    try:
        print(phase("生成静态 Faithfulness 评测集"))
        print(f"  数据源: {GOLDEN_PATH}")
        print(f"  输出:   {ANSWER_GOLDEN_PATH}")
        print()

        count = await generate_answer_dataset(
            golden_items, retriever, llm, ANSWER_GOLDEN_PATH,
        )

        if count == 0:
            raise SystemExit("生成失败：0 条成功")
        return count
    finally:
        await pool.close()


async def _run_eval(gate_mode: bool = False) -> dict:
    """加载静态评测集，跑 faithfulness 评分。"""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if not ANSWER_GOLDEN_PATH.exists():
        raise SystemExit(
            f"评测集不存在: {ANSWER_GOLDEN_PATH}\n"
            "请先运行: uv run rag-eval-answer --generate"
        )

    settings = get_settings()
    items = load_answer_golden(ANSWER_GOLDEN_PATH)
    print(f"评测集加载完成: {len(items)} 条\n")

    # 构造 RAGAS 所需的 LLM（独立 ChatOpenAI 实例，不经过 NormalModel 重试）
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=settings.MODEL_KEY,
            model=settings.MODEL_NAME,
            base_url=settings.MODEL_URL,
            temperature=0,
            seed=42,
        )
    )

    print(phase("开始 Faithfulness 评测"))
    result = await run_faithfulness_eval(items, ragas_llm)
    print(success(" 评测完成\n"))

    return result


def main() -> None:
    from rag.common.platform import setup_windows_loop

    setup_windows_loop()

    parser = argparse.ArgumentParser(description="答案忠实度离线评测（基于 RAGAS Faithfulness）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update-baseline", action="store_true",
                       help="用本次结果更新 baseline")
    group.add_argument("--gate", action="store_true",
                       help="门禁模式：对比 baseline，退化则 exit(1)")
    group.add_argument("--generate", action="store_true",
                       help="从 retrieval_golden.jsonl 生成静态评测集")
    args = parser.parse_args()

    # ── 生成模式 ──
    if args.generate:
        asyncio.run(_run_generate())
        return

    # ── 评测模式 ──
    result = asyncio.run(_run_eval(gate_mode=args.gate))

    # ── 持久化 ──
    _save_history(result)

    # ── 输出 ──
    _print_result(result)

    # ── 更新 baseline ──
    if args.update_baseline:
        new_baseline = {"faithfulness": result["aggregate"]["faithfulness"]}
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(new_baseline, f, ensure_ascii=False, indent=2)
        print(f"已更新 baseline -> {BASELINE_PATH}")
        return

    # ── 门禁 ──
    if args.gate:
        baseline = _load_baseline()
        passed = _print_gate(result, baseline)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
