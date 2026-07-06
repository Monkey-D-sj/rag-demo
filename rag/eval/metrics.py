import math
import re

# \W(默认 Unicode)保留中文/字母/数字为词字符，去掉标点与空白；补 _ 因其属词字符
_STRIP_RE = re.compile(r"[\W_]+")


def normalize(text: str) -> str:
    """归一化用于片段包含匹配：剔除所有标点与空白（中英文），大小写折叠。"""
    return _STRIP_RE.sub("", text).casefold()


def _hit_indices(chunk_text: str, gold_norm: list[str]) -> set[int]:
    """该 chunk 归一化后命中的 gold 片段下标集合（子串包含即命中）。"""
    norm_chunk = normalize(chunk_text)
    return {i for i, g in enumerate(gold_norm) if g and g in norm_chunk}


def evaluate_query(
    retrieved_texts: list[str],
    gold_snippets: list[str],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """对单条 query 的有序检索结果算指标（纯确定性）。

    - hit@k：top-k 内是否至少命中一个 gold 片段（0/1）
    - recall@k：top-k 内命中的不同 gold 片段数 / gold 片段总数
    - mrr：首个命中 chunk 位置的倒数（无命中=0）
    - ndcg@k：二值相关性 NDCG，IDCG 以 min(k, gold 数) 个理想命中位归一化
    """
    n_gold = len(gold_snippets)
    gold_norm = [normalize(g) for g in gold_snippets]
    per_rank = [_hit_indices(t, gold_norm) for t in retrieved_texts]

    result: dict[str, float] = {}

    first_hit = next((r for r, hits in enumerate(per_rank) if hits), None)
    result["mrr"] = 1.0 / (first_hit + 1) if first_hit is not None else 0.0

    for k in ks:
        topk = per_rank[:k]
        result[f"hit@{k}"] = 1.0 if any(topk) else 0.0

        covered: set[int] = set()
        for h in topk:
            covered |= h
        result[f"recall@{k}"] = len(covered) / n_gold if n_gold else 0.0

        dcg = sum(1.0 / math.log2(r + 2) for r, h in enumerate(topk) if h)
        ideal_n = min(k, n_gold)
        idcg = sum(1.0 / math.log2(r + 2) for r in range(ideal_n))
        result[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    return result
