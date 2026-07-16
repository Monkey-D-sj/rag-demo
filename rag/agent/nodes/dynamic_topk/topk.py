from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from rag.agent.type import ContextSchema, MyState, StreamEventType, stream_event
from rag.common.logging import get_logger
from rag.config import get_settings

logger = get_logger()


def _dynamic_truncate(
    chunks: list[dict],
    default_top_k: int,
    ratio_threshold: float,
) -> list[dict]:
    """相邻分差法动态截断 rerank 结果（含 plateau 保护）。

    从第 1 名开始，比较 score[k+1] / score[k]：
      - 比值 >= threshold → gap 小，继续
      - 比值 <  threshold → 候选截断点，但需确认不是“台阶到高原”：
          若 post-gap 的分数彼此紧密（内部相邻比值均 >= threshold），
          说明它们是质量一致的第二梯队，跳过此 gap 继续扫描；
          若 post-gap 内也有 gap（持续衰减），才是真正的质量悬崖，截断。
      - 没有任何 gap 触发 → 取 min(default_top_k, len(chunks))

    score[k] 为 0 时视为无限大 gap，在 k 处截断。
    """
    if not chunks:
        return chunks

    if len(chunks) == 1:
        return chunks

    # 提取 rerank_score，缺失或非数字兜底为 0
    scores: list[float] = []
    for c in chunks:
        try:
            s = float(c.get("rerank_score", 0) or 0)
        except (ValueError, TypeError):
            s = 0.0
        scores.append(max(0.0, s))

    # 如果没有任何有效分数（全 0 或缺失），视为 rerank 未启用，硬截断
    if all(s == 0.0 for s in scores):
        return chunks[:default_top_k]

    # 钳位 ratio_threshold 到合理范围
    threshold = max(0.01, min(0.99, ratio_threshold))

    for i in range(len(scores) - 1):
        if scores[i] == 0.0:
            # score[i] 为 0，视为无限大 gap，在 i 处截断
            return chunks[:i] if i > 0 else []

        if scores[i + 1] / scores[i] < threshold:
            # 候选 gap：检查 post-gap 分数是否形成高原（内部紧密一致）
            post = scores[i + 1 :]
            if len(post) >= 2:
                plateau = True
                for j in range(len(post) - 1):
                    if post[j] == 0.0:
                        continue
                    if post[j + 1] / post[j] < threshold:
                        plateau = False
                        break
                if plateau:
                    # 台阶到第二梯队，非质量悬崖，继续扫描
                    continue
            return chunks[: i + 1]

    # 无 gap 触发，取默认 top_k
    return chunks[: min(default_top_k, len(chunks))]


async def dynamic_topk(state: MyState, runtime: Runtime[ContextSchema]) -> MyState:
    """对 rerank 后的结果做动态 top-k 截断。

    未启用时透传；异常时降级透传原始结果，不中断检索链路。
    """
    settings = get_settings()

    if not settings.RERANK_DYNAMIC_TOPK_ENABLED:
        return state

    chunks = state.get("recall_vec_results")
    if not isinstance(chunks, list):
        logger.warning("recall_vec_results 不是 list，透传")
        return state

    if not chunks:
        return state

    writer = get_stream_writer()

    try:
        before = len(chunks)
        top_k = max(1, settings.RERANK_DYNAMIC_TOPK_DEFAULT)
        filtered = _dynamic_truncate(
            chunks,
            top_k,
            settings.RERANK_DYNAMIC_TOPK_RATIO,
        )
        state["recall_vec_results"] = filtered
        after = len(filtered)

        if after == 0:
            writer(stream_event(StreamEventType.STATUS, "未找到足够相关内容"))
        else:
            writer(stream_event(StreamEventType.STATUS, f"动态筛选 {after} 条相关结果"))
        logger.debug("dynamic_topk: %d -> %d chunks", before, after)
    except Exception:
        logger.warning("动态 top-k 截断失败，透传原始结果", exc_info=True)

    return state
