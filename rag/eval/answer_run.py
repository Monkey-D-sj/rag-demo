"""真实生产流水线答案评测 CLI。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit

from rag.config import get_settings
from rag.eval import DATASETS_DIR, EVAL_DIR
from rag.eval.answer_harness import (
    build_real_collection_context,
    collect_answers,
    dataset_fingerprint,
    sample_fingerprint,
    score_answers,
    select_answer_items,
)
from rag.eval.answer_metrics import aggregate_answer_metrics
from rag.eval.artifacts import (
    assert_resume_compatible,
    create_run,
    list_items,
    load_manifest,
    make_run_id,
    run_dir,
    update_manifest,
    write_item,
)
from rag.eval.harness import load_golden
from rag.prompts.answer_eval import PROMPT_VERSION

GOLDEN_PATH = DATASETS_DIR / "retrieval_golden.jsonl"
HISTORY_DIR = EVAL_DIR / "history"
BASELINE_PATH = EVAL_DIR / "baseline.json"


def _host(url: str) -> str:
    parsed = urlsplit(url or "")
    return f"{parsed.scheme}://{parsed.hostname}" if parsed.scheme and parsed.hostname else ""


def _prompt_hash() -> str:
    from rag.prompts.answer_eval import CITATION_JUDGE_PROMPT
    return hashlib.sha256(CITATION_JUDGE_PROMPT.encode()).hexdigest()


def _manifest(settings, items, run_id) -> dict:
    judge_name = settings.EVAL_JUDGE_MODEL_NAME or settings.MODEL_NAME
    judge_url = settings.EVAL_JUDGE_MODEL_URL or settings.MODEL_URL
    return {
        "schema_version": 2, "run_id": run_id, "status": "collecting",
        "dataset_fingerprint": dataset_fingerprint(GOLDEN_PATH),
        "sample_fingerprint": sample_fingerprint(items), "selected_ids": [x.id for x in items],
        "limit": len(items), "seed": settings.EVAL_RANDOM_SEED,
        "generator_model": settings.MODEL_NAME, "generator_url_host": _host(settings.MODEL_URL),
        "judge_model": judge_name, "judge_url_host": _host(judge_url),
        "embedding_model": settings.EMBEDDING_MODEL, "reranker_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else "disabled",
        "judge_prompt_version": PROMPT_VERSION, "judge_prompt_hash": _prompt_hash(),
        "ragas_version": _ragas_version(), "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _ragas_version() -> str:
    try:
        import ragas
        return getattr(ragas, "__version__", "unknown")
    except Exception:
        return "unavailable"


class _EvalEmbeddings:
    def __init__(self, embedding):
        self.embedding = embedding

    async def aembed_query(self, text):
        return (await self.embedding.embed([text]))[0]

    async def aembed_documents(self, texts):
        return await self.embedding.embed(list(texts))

    def embed_query(self, text):
        raise NotImplementedError("eval 使用异步 embedding")

    def embed_documents(self, texts):
        raise NotImplementedError("eval 使用异步 embedding")


def _build_scorers(settings):
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from rag.models.normal import NormalModel

    judge_configured = bool(settings.EVAL_JUDGE_MODEL_NAME)
    if judge_configured:
        judge_chat = ChatOpenAI(api_key=settings.EVAL_JUDGE_MODEL_KEY, model=settings.EVAL_JUDGE_MODEL_NAME,
                                base_url=settings.EVAL_JUDGE_MODEL_URL, temperature=0, seed=settings.EVAL_RANDOM_SEED,
                                timeout=settings.EVAL_JUDGE_TIMEOUT_SECONDS, max_retries=0)
        judge_llm = LangchainLLMWrapper(judge_chat)
        citation_judge = _StructuredJudge(judge_chat)
    else:
        citation_judge = NormalModel(settings)
        judge_llm = LangchainLLMWrapper(citation_judge._model)
    # ResponseRelevancy 在 RAGAS 0.2.x 的相似度计算是同步接口，使用同一 endpoint
    # 的 LangChain OpenAI adapter，避免在正在运行的 event loop 中嵌套 asyncio.run。
    embedding = OpenAIEmbeddings(api_key=settings.EMBEDDING_KEY, base_url=settings.EMBEDDING_URL,
                                 model=settings.EMBEDDING_MODEL, dimensions=settings.EMBEDDING_DIM,
                                 max_retries=0)
    wrapped_embedding = LangchainEmbeddingsWrapper(embedding)
    return Faithfulness(llm=judge_llm), ResponseRelevancy(llm=judge_llm, embeddings=wrapped_embedding), citation_judge


class _StructuredJudge:
    def __init__(self, model):
        self.model = model

    async def ainvoke_structured(self, messages, schema):
        return await self.model.with_structured_output(schema, method="json_mode").ainvoke(messages)


async def _score_run(run_id: str, settings, *, rescore: bool = False) -> dict:
    manifest = load_manifest(HISTORY_DIR, run_id)
    if manifest.get("status") == "completed" and not rescore:
        raise ValueError("completed run 默认不可重复执行，请使用 --rescore")
    artifacts = list_items(HISTORY_DIR, run_id)
    if not artifacts:
        raise ValueError(f"run 没有可评分 item: {run_id}")
    update_manifest(HISTORY_DIR, run_id, status="scoring")
    def needs_score(item):
        score = item.get("score") or {}
        if not score:
            return True
        for name in ("faithfulness", "answer_relevance", "citation_accuracy"):
            if score.get(f"{name}_error"):
                return True
            if score.get(name) is None and not (name == "citation_accuracy" and score.get("citation_diagnostics", {}).get("factual_claim_count") == 0):
                return True
        return False

    pending = list(artifacts) if rescore else [item for item in artifacts if needs_score(item)]
    if pending:
        faithfulness, relevance, citation = _build_scorers(settings)
        result = await score_answers(pending, faithfulness_scorer=faithfulness, relevance_scorer=relevance,
                                     citation_judge=citation, semaphore=asyncio.Semaphore(settings.EVAL_JUDGE_CONCURRENCY),
                                     timeout=settings.EVAL_JUDGE_TIMEOUT_SECONDS, max_attempts=settings.EVAL_JUDGE_MAX_ATTEMPTS)
    else:
        scores = [item["score"] for item in artifacts]
        result = {"per_query": scores, **aggregate_answer_metrics(scores)}
    if pending and len(pending) != len(artifacts):
        fresh = {row["id"]: row for row in result["per_query"]}
        merged = [fresh.get(item["id"], item.get("score", {"id": item["id"], "query": item.get("query", "")})) for item in artifacts]
        result = {"per_query": merged, **aggregate_answer_metrics(merged)}
    for row in result["per_query"]:
        item = next((x for x in artifacts if x["id"] == row["id"]), {"id": row["id"], "query": row["query"]})
        if rescore and item.get("score"):
            item.setdefault("score_attempts", []).append(item["score"])
        item["score"] = row
        write_item(HISTORY_DIR, run_id, row["id"], item)
    collection_success = sum(1 for x in artifacts if not x.get("collection_error")) / len(artifacts)
    judge_success = {key: result["counts"][key]["success_rate"] for key in result["counts"]}
    result["generation_success_rate"] = collection_success
    result["judge_success_rate"] = judge_success
    run_dir(HISTORY_DIR, run_id).joinpath("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# Answer Evaluation", "", "| Metric | Score | Evaluated | Skipped |", "|---|---:|---:|---:|"]
    for key, value in result["metrics"].items():
        counts = result["counts"][key]
        report_lines.append(f"| {key} | {'N/A' if value is None else f'{value:.4f}'} | {counts['evaluated_count']} | {counts['skipped_count']} |")
    report_lines.append(f"\nGeneration success rate: {collection_success:.4f}")
    run_dir(HISTORY_DIR, run_id).joinpath("report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    update_manifest(HISTORY_DIR, run_id, status="completed" if collection_success == 1 and all(v >= .95 for v in judge_success.values()) else "partial",
                    generation_success_rate=collection_success, judge_success_rate=judge_success)
    return result


async def _collect_run(args, settings) -> str:
    all_items = load_golden(GOLDEN_PATH)
    run_id = args.resume or make_run_id(HISTORY_DIR)
    if args.resume:
        manifest = load_manifest(HISTORY_DIR, run_id)
        selected = select_answer_items(all_items, ids=list(manifest.get("selected_ids", [])))
        expected = _manifest(settings, selected, run_id)
        expected["limit"] = manifest.get("limit", len(selected))
        expected["seed"] = manifest.get("seed", settings.EVAL_RANDOM_SEED)
        assert_resume_compatible(manifest, expected)
    else:
        ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
        selected = select_answer_items(all_items, limit=args.limit, seed=args.seed, ids=ids, all_items=args.all_items)
        create_run(HISTORY_DIR, run_id, _manifest(settings, selected, run_id))
    graph_obj, context, pool = await build_real_collection_context(settings)
    try:
        existing = {x.get("id") for x in list_items(HISTORY_DIR, run_id)}
        async def save(row):
            write_item(HISTORY_DIR, run_id, row["id"], row)
        current_items = list_items(HISTORY_DIR, run_id)
        current_by_id = {x.get("id"): x for x in current_items}
        pending = [x for x in selected if x.id not in existing or current_by_id.get(x.id, {}).get("collection_error")]
        await collect_answers(pending, graph_obj=graph_obj, context=context, run_id=run_id, on_item=save)
    finally:
        await pool.close()
    update_manifest(HISTORY_DIR, run_id, status="collected")
    return run_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="真实 LangGraph 生成端评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ids")
    parser.add_argument("--all", action="store_true", dest="all_items")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    parser.add_argument("--rescore")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--accept-regression", action="store_true")
    parser.add_argument("--reason")
    return parser


def main() -> None:
    from rag.common.platform import setup_windows_loop
    setup_windows_loop()
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    if sum(bool(x) for x in (args.score_only, args.resume, args.rescore)) > 1:
        raise SystemExit("--score-only、--resume、--rescore 不能同时使用")
    if args.run_id and not args.score_only:
        raise SystemExit("--run-id 只能与 --score-only 一起使用")
    if args.accept_regression and (not args.update_baseline or not (args.reason or "").strip()):
        raise SystemExit("--accept-regression 需要 --update-baseline 和非空 --reason")
    settings = get_settings()
    settings_seed = settings.EVAL_RANDOM_SEED if args.seed is None else args.seed
    args.seed = settings_seed
    if args.score_only:
        run_id = args.run_id or args.resume
        if not run_id:
            raise SystemExit("--score-only 需要 --run-id <run-id>")
    elif args.rescore:
        run_id = args.rescore
    else:
        run_id = asyncio.run(_collect_run(args, settings))
        print(f"collect 完成: {run_id}")
        if args.collect_only:
            return
    result = asyncio.run(_score_run(run_id, settings, rescore=bool(args.rescore)))
    print(json.dumps(result.get("metrics", {}), ensure_ascii=False, indent=2))
    from rag.eval.baseline import comparable_provenance, gate_answer_metrics, load_baseline, update_baseline
    manifest = load_manifest(HISTORY_DIR, run_id)
    provenance = {key: manifest.get(key) for key in ("sample_fingerprint", "selected_ids", "seed", "generator_model", "judge_model", "embedding_model", "judge_prompt_hash")}
    baseline = load_baseline(BASELINE_PATH)
    answer_gate = gate_answer_metrics(result.get("metrics", {}), baseline.get("answer", {}).get("metrics", {}), baseline.get("answer", {}).get("floors", {}))
    if baseline.get("provenance"):
        compatible, _ = comparable_provenance(provenance, baseline["provenance"])
        answer_gate["passed"] = answer_gate["passed"] and compatible
    if args.update_baseline:
        update_baseline(BASELINE_PATH, provenance=provenance, retrieval={}, answer_metrics=result["metrics"], run_id=run_id,
                        reason=args.reason or "answer baseline update", accept_regression=args.accept_regression)
        return
    if args.gate and not answer_gate["passed"]:
        raise SystemExit(1)
    if result.get("generation_success_rate") != 1.0 or any(v < .95 for v in result.get("judge_success_rate", {}).values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
