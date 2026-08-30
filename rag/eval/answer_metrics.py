"""生成端评测的预处理、引用判定和聚合纯函数。

本模块刻意不负责调用模型，因此可以在没有数据库、模型或 RAGAS 的环境中完整测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

from pydantic import BaseModel, Field


_BIBLIOGRAPHY_RE = re.compile(r"^\s*\[(\d+)\]:\s*(.*?)\s*$")
_CITATION_RE = re.compile(r"\[(\d+)\]")
_ILLEGAL_CITATION_RE = re.compile(r"\[([^\]\r\n]+)\]")


@dataclass
class NormalizedAnswer:
    raw_answer: str
    answer_body_with_citations: str
    answer_body_plain: str
    bibliography: list[dict] = field(default_factory=list)
    format_diagnostics: list[str] = field(default_factory=list)


def normalize_answer(answer: str) -> NormalizedAnswer:
    """剥离答案尾部 bibliography，并生成供不同 Judge 使用的两个正文版本。"""
    raw = answer or ""
    lines = raw.splitlines()
    bibliography: list[dict] = []
    body_end = len(lines)
    # bibliography 约定从第一个尾部 [N]: 行开始；中间的正文引用不会被误删。
    for i, line in enumerate(lines):
        m = _BIBLIOGRAPHY_RE.match(line)
        if m and i > 0:
            tail = lines[i:]
            if all(_BIBLIOGRAPHY_RE.match(x) for x in tail if x.strip()):
                body_end = i
                for entry in tail:
                    parsed = _BIBLIOGRAPHY_RE.match(entry)
                    if parsed:
                        bibliography.append({"index": int(parsed.group(1)), "text": parsed.group(2)})
                break
    body = "\n".join(lines[:body_end]).rstrip()
    diagnostics: list[str] = []
    for token in _ILLEGAL_CITATION_RE.findall(body):
        if not token.isdigit():
            diagnostics.append(f"非法引用格式: [{token}]")
    plain = _CITATION_RE.sub("", body)
    return NormalizedAnswer(raw, body, plain, bibliography, diagnostics)


class CitationLinkVerdict(BaseModel):
    index: int
    valid: bool
    supported: bool
    reason: str = ""


class CitationClaimVerdict(BaseModel):
    claim: str
    is_factual: bool
    citation_indices: list[int] = Field(default_factory=list)
    links: list[CitationLinkVerdict] = Field(default_factory=list)


class CitationJudgeOutput(BaseModel):
    claims: list[CitationClaimVerdict] = Field(default_factory=list)


def _as_claims(output: CitationJudgeOutput | dict | None) -> list[CitationClaimVerdict]:
    if output is None:
        return []
    if isinstance(output, CitationJudgeOutput):
        return output.claims
    return CitationJudgeOutput.model_validate(output).claims


def score_citations(
    output: CitationJudgeOutput | dict | None,
    available_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    """按计划中的 link + uncited factual claim 公式计算引用指标。"""
    available = set(available_indices or [])
    claims = _as_claims(output)
    factual = [c for c in claims if c.is_factual]
    links = [link for claim in factual for link in claim.links]
    # Judge 可能只返回 citation_indices 而省略 links，按未判定链接处理，绝不静默加分。
    for claim in factual:
        linked_indices = {link.index for link in claim.links}
        for index in claim.citation_indices:
            if index not in linked_indices:
                links.append(CitationLinkVerdict(index=index, valid=index in available, supported=False, reason="Judge 未返回该链接判定"))
    cited_claims = sum(bool(c.citation_indices or c.links) for c in factual)
    uncited_claims = len(factual) - cited_claims
    valid_links = sum(link.valid and link.index in available for link in links)
    supported_links = sum(link.valid and link.supported and link.index in available for link in links)
    denominator = len(links) + uncited_claims
    covered_claims = sum(
        any(link.valid and link.supported and link.index in available for link in claim.links)
        for claim in factual
    )
    result: dict[str, Any] = {
        "citation_accuracy": supported_links / denominator if denominator else None,
        "citation_validity": valid_links / len(links) if links else None,
        "citation_precision": supported_links / valid_links if valid_links else None,
        "citation_coverage": covered_claims / len(factual) if factual else None,
        "factual_claim_count": len(factual),
        "uncited_factual_claim_count": uncited_claims,
        "citation_link_count": len(links),
        "supported_valid_links": supported_links,
        "claims": [c.model_dump(mode="json") for c in claims],
    }
    return result


def citation_accuracy(output: CitationJudgeOutput | dict | None, available_indices: Iterable[int] | None = None) -> float | None:
    """便于调用方只取 P0 核心引用分数的薄封装。"""
    return score_citations(output, available_indices)["citation_accuracy"]


def aggregate_answer_metrics(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """对每个题目的非 null 分数做等权 macro average，并保留计数。"""
    metric_names = ("faithfulness", "answer_relevance", "citation_accuracy")
    aggregate: dict[str, Any] = {}
    counts: dict[str, dict[str, int | float]] = {}
    for name in metric_names:
        values = [float(row[name]) for row in per_query if row.get(name) is not None]
        aggregate[name] = sum(values) / len(values) if values else None
        evaluated = len(values)
        failed = sum(1 for row in per_query if row.get(name) is None and row.get(f"{name}_error"))
        skipped = len(per_query) - evaluated - failed
        counts[name] = {
            "evaluated_count": evaluated,
            "skipped_count": skipped,
            "failed_count": failed,
            "success_rate": (evaluated + skipped) / len(per_query) if per_query else 0.0,
        }
    return {"metrics": aggregate, "aggregate": aggregate, "counts": counts}


aggregate_metrics = aggregate_answer_metrics


def relative_drop(current: float, baseline: float) -> float:
    return (baseline - current) / baseline if baseline > 0 else 0.0


def median_scores(scores: list[float]) -> float:
    return float(median(scores))
