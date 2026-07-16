"""答案忠实度离线评测：数据加载、RAGAS 评分、数据集生成。"""

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from rag.common.logging import get_logger
from rag.eval.harness import GoldenItem
from rag.prompts.generate import system_prompt

logger = get_logger()


@dataclass
class AnswerGoldenItem:
    """静态评测集中的单条记录。"""
    id: str
    query: str
    answer: str
    contexts: list[str]


def load_answer_golden(path) -> list[AnswerGoldenItem]:
    """读取 answer_golden.jsonl，返回评测条目列表。

    校验必填字段：id, query, answer, contexts。
    contexts 必须是非空列表。
    """
    items: list[AnswerGoldenItem] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("query"):
                raise ValueError(f"{path}:{lineno} 缺少 query")
            if not obj.get("answer"):
                raise ValueError(f"{path}:{lineno} 缺少 answer")
            contexts = obj.get("contexts")
            if not isinstance(contexts, list) or len(contexts) == 0:
                raise ValueError(f"{path}:{lineno} contexts 必须是非空列表")
            items.append(
                AnswerGoldenItem(
                    id=str(obj.get("id", lineno)),
                    query=obj["query"],
                    answer=obj["answer"],
                    contexts=contexts,
                )
            )
    return items


async def run_faithfulness_eval(
    items: list[AnswerGoldenItem],
    evaluator_llm,  # LangchainLLMWrapper — 由 CLI 层构造传入
) -> dict:
    """对静态评测集逐条打分，返回 aggregate + per_query 结果。

    返回格式：
        {
            "aggregate": {"faithfulness": 0.87},
            "per_query": [
                {"id": "q001", "query": "...", "faithfulness": 0.95},
                ...
            ],
        }

    evaluator_llm 是 RAGAS 所需的 LangchainLLMWrapper 实例，
    由 CLI 层用 ChatOpenAI 构造后传入，harness 不关心 LLM 配置。
    """
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    scorer = Faithfulness(llm=evaluator_llm)
    per_query: list[dict] = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        try:
            sample = SingleTurnSample(
                user_input=item.query,
                response=item.answer,
                retrieved_contexts=item.contexts,
            )
            score = await scorer.single_turn_ascore(sample)
            per_query.append({
                "id": item.id,
                "query": item.query,
                "faithfulness": float(score),
            })
        except Exception:
            logger.warning(
                "faithfulness 打分失败 id=%s query=%s", item.id, item.query[:60],
                exc_info=True,
            )
            continue

        if idx % 10 == 0 or idx == total:
            pct = idx * 100 // total
            print(f"  [{idx:>{len(str(total))}}/{total}] {pct:>3}% …", flush=True)

    if not per_query:
        raise RuntimeError("所有条目打分均失败，无法计算 aggregate")

    scores = [pq["faithfulness"] for pq in per_query]
    aggregate = {"faithfulness": sum(scores) / len(scores)}

    return {"aggregate": aggregate, "per_query": per_query}


async def generate_answer_dataset(
    golden_items: list[GoldenItem],
    retriever,       # KnowledgeRetriever
    llm,             # NormalModel (for ainvoke)
    output_path: Path,
    top_k: int = 5,
) -> int:
    """从 retrieval golden 集生成静态 faithfulness 评测集。

    对每条 in-scope item：
    1. retriever.search() 取 contexts
    2. llm.ainvoke() 生成 answer
    3. 写入 answer_golden.jsonl

    返回成功生成的条目数。
    """
    in_scope = [it for it in golden_items if not it.out_of_scope]
    total = len(in_scope)
    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(in_scope, 1):
            try:
                # 检索
                rw = item.rewrite_query or item.query
                rows = await retriever.search(rw, None, top_k=top_k)
                contexts = [r["text"] for r in rows]

                # 生成
                context_parts: list[str] = []
                for i, c in enumerate(rows, 1):
                    title = c.get("filename", "未知文档")
                    text = c.get("text", "")
                    context_parts.append(f"[{i}] (来源文档: {title})\n{text}")

                knowledge = "\n\n".join(context_parts)
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f'查询: {item.query}\n知识库内容: {knowledge}'),
                ]
                answer = await llm.ainvoke(messages)

                # 写入
                row = {
                    "id": item.id,
                    "query": item.query,
                    "answer": answer,
                    "contexts": contexts,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                logger.warning(
                    "生成数据集条目失败 id=%s query=%s",
                    item.id, item.query[:60],
                    exc_info=True,
                )
                continue

            if idx % 10 == 0 or idx == total:
                pct = idx * 100 // total
                print(f"  [{idx:>{len(str(total))}}/{total}] {pct:>3}% …", flush=True)

    print(f"生成完成: {count}/{total} 条 -> {output_path}")
    return count
