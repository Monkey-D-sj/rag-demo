"""基于生产 LangGraph 的可恢复答案评测 harness。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from rag.agent.type import ContextSchema
from rag.common.logging import get_logger
from rag.eval.answer_metrics import CitationJudgeOutput, aggregate_answer_metrics, normalize_answer, score_citations
from rag.eval.harness import GoldenItem
from rag.prompts.answer_eval import CITATION_JUDGE_PROMPT

logger = get_logger()


@dataclass
class AnswerGoldenItem:
    """旧 answer_golden 读取接口，仅为已有调用方提供兼容。"""
    id: str
    query: str
    answer: str
    contexts: list[str]


def load_answer_golden(path: str | Path) -> list[AnswerGoldenItem]:
    items: list[AnswerGoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if not obj.get("query"):
                raise ValueError(f"{path}:{lineno} 缺少 query")
            if not obj.get("answer"):
                raise ValueError(f"{path}:{lineno} 缺少 answer")
            contexts = obj.get("contexts")
            if not isinstance(contexts, list) or not contexts:
                raise ValueError(f"{path}:{lineno} contexts 必须是非空列表")
            items.append(AnswerGoldenItem(str(obj.get("id", lineno)), obj["query"], obj["answer"], contexts))
    return items


def eligible_items(items: list[GoldenItem]) -> list[GoldenItem]:
    return [item for item in items if not item.out_of_scope]


def dataset_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_fingerprint(items: list[GoldenItem]) -> str:
    payload = [
        {"id": item.id, "query": item.query, "rewrite_query": item.rewrite_query, "gold_snippets": item.gold_snippets,
         "out_of_scope": item.out_of_scope, "category": item.category,
         "entities": item.entities, "sub_queries": item.sub_queries}
        for item in sorted(items, key=lambda x: x.id)
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_answer_items(items: list[GoldenItem], *, limit: int | None = None, seed: int = 42,
                        ids: list[str] | None = None, all_items: bool = False) -> list[GoldenItem]:
    """稳定抽样；不修改全局 random 状态。"""
    if ids is not None and (all_items or limit is not None):
        raise ValueError("--ids、--all、--limit 互斥")
    if all_items and limit is not None:
        raise ValueError("--all 与 --limit 互斥")
    all_item_ids = [item.id for item in items]
    if len(set(all_item_ids)) != len(all_item_ids):
        raise ValueError("golden 集存在重复 ID")
    eligible = eligible_items(items)
    by_id = {item.id: item for item in eligible}
    if ids is not None:
        if len(set(ids)) != len(ids):
            raise ValueError("--ids 不能包含重复 ID")
        missing = [item_id for item_id in ids if item_id not in by_id]
        if missing:
            all_ids = {item.id for item in items}
            ineligible = [item_id for item_id in missing if item_id in all_ids]
            if ineligible:
                raise ValueError(f"指定 ID 为 ineligible: {', '.join(ineligible)}")
            raise ValueError(f"指定 ID 不存在: {', '.join(missing)}")
        return sorted((by_id[item_id] for item_id in ids), key=lambda x: x.id)
    if all_items:
        return sorted(eligible, key=lambda x: x.id)
    limit = 15 if limit is None else limit
    if limit < 0:
        raise ValueError("limit 不能为负数")
    ordered = sorted(eligible, key=lambda x: x.id)
    selected = random.Random(seed).sample(ordered, min(limit, len(ordered)))
    return sorted(selected, key=lambda x: x.id)


_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_item_id(item_id: str) -> str:
    value = _SAFE_ID_RE.sub("-", str(item_id)).strip("-")
    return value or "item"


async def collect_one(item: GoldenItem, *, graph_obj: Any, context: ContextSchema, run_id: str) -> dict[str, Any]:
    """从 graph 的 custom + values 流中收集一题，并执行一致性检查。"""
    started = time.perf_counter()
    messages: list[str] = []
    citations: list[dict] = []
    statuses: list[str] = []
    last_state: dict[str, Any] = {}
    try:
        from rag.agent.workflow import build_initial_state
        initial = build_initial_state(f"eval-{run_id}-{safe_item_id(item.id)}", item.query)
        async for mode, payload in graph_obj.astream(initial, context=context, stream_mode=["custom", "values"]):
            if mode == "values":
                if isinstance(payload, dict):
                    last_state = payload
                continue
            event = payload if isinstance(payload, dict) else {}
            event_type, data = event.get("type"), event.get("data")
            if event_type == "message" and data:
                messages.append(str(data))
            elif event_type == "citations":
                citations = list(data or [])
            elif event_type == "status" and data:
                statuses.append(str(data))
        answer = str(last_state.get("generated") or "")
        streamed = "".join(messages)
        contexts = []
        for index, row in enumerate(last_state.get("recall_vec_results") or [], 1):
            contexts.append({"index": index, "text": row.get("text", ""),
                             "document_title": row.get("filename") or row.get("document_title") or "未知文档",
                             "metadata": row.get("metadata") or {}})
        route = "cache" if last_state.get("cache_hit") else ("out_of_scope" if last_state.get("is_out_of_scope") else ("context" if last_state.get("answer_from_context") else "rag_single"))
        pipeline = {"route": route,
                    "rewrite_query": last_state.get("rewrite_query", ""), "sub_queries": last_state.get("sub_queries") or [],
                    "answer": answer, "contexts": contexts,
                    "citations": citations or list(last_state.get("citations") or []),
                    "cache_hit": bool(last_state.get("cache_hit", False)),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "statuses": statuses, "streamed_answer": streamed}
        errors: list[str] = []
        diagnostics: list[str] = []
        if last_state.get("is_out_of_scope") or last_state.get("answer_from_context"):
            errors.append("最终 state 未走知识库 RAG 路径")
        if pipeline["route"] != "rag_single":
            errors.append("未走 RAG 生成路径")
        if not answer:
            errors.append("最终 state generated 为空")
        if streamed and streamed != answer:
            diagnostics.append("state generated 与 message 拼接结果不一致，以 state 为准")
        if pipeline["cache_hit"]:
            errors.append("评测禁止 cache hit")
        citation_indices = [int(x.get("index", 0)) for x in pipeline["citations"]]
        if citation_indices and citation_indices != list(range(1, len(citation_indices) + 1)):
            errors.append("citations index 不连续")
        context_indices = [x["index"] for x in contexts]
        if context_indices != list(range(1, len(context_indices) + 1)):
            errors.append("contexts index 不连续")
        pipeline["diagnostics"] = diagnostics
        return {"id": item.id, "query": item.query, "pipeline": pipeline, "collection_error": "; ".join(errors) or None}
    except Exception as exc:
        logger.warning("答案采集失败 id=%s", item.id, exc_info=True)
        return {"id": item.id, "query": item.query, "pipeline": None,
                "collection_error": f"{type(exc).__name__}: {exc}"}


async def collect_answers(items: list[GoldenItem], *, graph_obj: Any, context: ContextSchema, run_id: str,
                          on_item: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None) -> list[dict[str, Any]]:
    results = []
    for item in items:
        result = await collect_one(item, graph_obj=graph_obj, context=context, run_id=run_id)
        results.append(result)
        if on_item:
            maybe = on_item(result)
            if asyncio.iscoroutine(maybe):
                await maybe
    return results


def _citation_context_texts(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{c['index']}] 《{c.get('document_title', '未知文档')}》\n{c.get('text', '')}" for c in contexts)


async def _call_with_retry(func, *, timeout: float, max_attempts: int, semaphore: asyncio.Semaphore):
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            async with semaphore:
                return await asyncio.wait_for(func(), timeout=timeout)
        except Exception as exc:
            last = exc
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403, 429):
                raise
            if attempt + 1 < max_attempts:
                await asyncio.sleep(min(2 ** attempt, 4))
    raise last or RuntimeError("评分失败")


async def score_answer_item(artifact: dict[str, Any], *, faithfulness_scorer: Any, relevance_scorer: Any,
                            citation_judge: Any, semaphore: asyncio.Semaphore | None = None,
                            timeout: float = 60, max_attempts: int = 3) -> dict[str, Any]:
    """三个指标独立评分；一项失败只影响该项。"""
    if artifact.get("collection_error"):
        return {"id": artifact["id"], "query": artifact["query"], "score_error": "collection failure"}
    pipeline = artifact["pipeline"]
    normalized = normalize_answer(pipeline["answer"])
    contexts = [c.get("text", "") for c in pipeline.get("contexts", [])]
    from ragas.dataset_schema import SingleTurnSample
    sample = SingleTurnSample(user_input=artifact["query"], response=normalized.answer_body_plain, retrieved_contexts=contexts)
    sem = semaphore or asyncio.Semaphore(1)
    output: dict[str, Any] = {"id": artifact["id"], "query": artifact["query"], "format_diagnostics": normalized.format_diagnostics}

    async def run_metric(name: str, fn):
        try:
            value = await _call_with_retry(fn, timeout=timeout, max_attempts=max_attempts, semaphore=sem)
            output[name] = None if value is None else float(value)
        except Exception as exc:
            output[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            output[name] = None

    async def citation_metric():
        prompt = CITATION_JUDGE_PROMPT.format(query=artifact["query"], answer=normalized.answer_body_with_citations,
                                              contexts=_citation_context_texts(pipeline.get("contexts", [])))
        messages = [SystemMessage(content=prompt), HumanMessage(content="执行引用审计并只输出 JSON。")]
        if hasattr(citation_judge, "ainvoke_structured"):
            judged = await citation_judge.ainvoke_structured(messages, CitationJudgeOutput)
        else:
            judged = await citation_judge.ainvoke(messages)
            judged = judged if isinstance(judged, (dict, CitationJudgeOutput)) else json.loads(getattr(judged, "content", judged))
        citation_result = score_citations(judged, range(1, len(pipeline.get("contexts", [])) + 1))
        output["citation_diagnostics"] = citation_result
        return citation_result["citation_accuracy"]

    await asyncio.gather(
        run_metric("faithfulness", lambda: faithfulness_scorer.single_turn_ascore(sample)),
        run_metric("answer_relevance", lambda: relevance_scorer.single_turn_ascore(sample)),
        run_metric("citation_accuracy", citation_metric),
    )
    return output


async def score_answers(artifacts: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
    scored = []
    for artifact in artifacts:
        scored.append(await score_answer_item(artifact, **kwargs))
    aggregate = aggregate_answer_metrics(scored)
    return {"per_query": scored, **aggregate}


async def run_faithfulness_eval(items: list[AnswerGoldenItem], evaluator_llm: Any) -> dict:
    """兼容旧调用方；新 CLI 不使用静态 answer golden。"""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness
    scorer = Faithfulness(llm=evaluator_llm)
    rows = []
    for item in items:
        try:
            score = await scorer.single_turn_ascore(SingleTurnSample(user_input=item.query, response=item.answer, retrieved_contexts=item.contexts))
            rows.append({"id": item.id, "query": item.query, "faithfulness": float(score)})
        except Exception:
            logger.warning("faithfulness 打分失败 id=%s", item.id, exc_info=True)
    if not rows:
        raise RuntimeError("所有条目打分均失败，无法计算 aggregate")
    return {"aggregate": {"faithfulness": sum(x["faithfulness"] for x in rows) / len(rows)}, "per_query": rows}


async def build_real_collection_context(settings: Any):
    """构造真实生产依赖；调用方负责关闭返回的 pool。"""
    from rag.agent.workflow import graph
    from rag.eval.harness import build_retriever
    from rag.models.normal import NormalModel
    pool, retriever = await build_retriever(settings)
    reranker = None
    if settings.RERANK_ENABLED and settings.RERANK_BASE_URL:
        from rag.models.rerank import QwenReranker
        reranker = QwenReranker(settings)
    context = ContextSchema(llm=NormalModel(settings), memory_manager=None, retriever=retriever,
                            reranker=reranker, pool=pool, semantic_cache=None)
    return graph, context, pool
