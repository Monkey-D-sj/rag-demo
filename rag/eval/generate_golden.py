import asyncio
import json

from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.eval import DATASETS_DIR, EVAL_KB_ID
from rag.models.normal import NormalModel

CANDIDATES_PATH = DATASETS_DIR / "retrieval_golden.candidates.jsonl"

_PROMPT = """你是检索评测数据的构造助手。下面给你一段原文，请基于它设计 1 个用户可能提出的问题，问题的答案必须能在这段原文中找到。同时抽取答案所依赖的 1~2 个原文短句（原样摘录，不要改写）作为 gold 片段。
要求：
- 问题具体、无歧义，不要出现"这段话""本文"等指代词。
- gold 片段是原文中连续的短句，尽量短且在全书中唯一。

原文：
{chunk}
"""


class GoldenCandidate(BaseModel):
    query: str = Field(description="用户问题")
    gold_snippets: list[str] = Field(description="答案所依赖的原文短句，1~2 个")


async def _load_chunks(pool) -> list[dict]:
    async with get_cursor(pool) as cur:
        await cur.execute(
            "SELECT id, chunk_index, text FROM document_chunks "
            "WHERE knowledge_base_id = %(kb)s ORDER BY chunk_index",
            {"kb": EVAL_KB_ID},
        )
        return await cur.fetchall()


async def generate() -> int:
    settings = get_settings()
    settings.check_required()
    pool = await create_pg_pool(settings)
    llm = NormalModel(settings)
    try:
        chunks = await _load_chunks(pool)
        if not chunks:
            raise SystemExit("评测 KB 为空，请先运行 python -m rag.eval.seed_corpus")
        count = 0
        with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                try:
                    cand = await llm.ainvoke_structured(
                        [_PROMPT.format(chunk=ch["text"])], GoldenCandidate
                    )
                except Exception:  # 单条失败跳过，不中断整批
                    continue
                row = {
                    "id": f"q{i:03d}",
                    "query": cand.query,
                    "gold_snippets": cand.gold_snippets,
                    "source_chunk_index": ch["chunk_index"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        return count
    finally:
        await pool.close()


def main() -> None:
    import sys

    if sys.platform == "win32":
        import asyncio as _asyncio

        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())

    n = asyncio.run(generate())
    print(f"生成候选 {n} 条 -> {CANDIDATES_PATH}")
    print("请人工筛选后另存为 retrieval_golden.jsonl（删去 source_chunk_index 字段可选）")


if __name__ == "__main__":
    main()
