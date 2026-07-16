import asyncio

from rag.config import get_settings
from rag.db.postgres import create_pg_pool, get_cursor
from rag.document import store
from rag.document.chunker import SplitStrategy, chunk
from rag.eval import DATASETS_DIR, EVAL_KB_ID
from rag.models.embedding import EmbeddingModel

CORPUS_PATH = DATASETS_DIR / "corpus.txt"


async def _reset_eval_kb(pool) -> None:
    """清空评测 KB 的既有文档与 chunk，保证 seed 幂等可重放。"""
    async with get_cursor(pool) as cur:
        await cur.execute(
            "DELETE FROM document_chunks WHERE knowledge_base_id = %(kb)s",
            {"kb": EVAL_KB_ID},
        )
        await cur.execute(
            "DELETE FROM documents WHERE knowledge_base_id = %(kb)s",
            {"kb": EVAL_KB_ID},
        )


async def seed() -> int:
    settings = get_settings()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    pieces = chunk(SplitStrategy.paragraph_semantic, text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not pieces:
        raise SystemExit("corpus.txt 切块为空")

    pool = await create_pg_pool(settings)
    try:
        await _reset_eval_kb(pool)
        doc_id = await store.create_document(
            pool,
            knowledge_base_id=EVAL_KB_ID,
            filename="corpus.txt",
            content_type="text/plain",
            size_bytes=len(text.encode("utf-8")),
            content_hash="eval-corpus-frozen",
            object_key="eval/corpus.txt",
        )

        embedding = EmbeddingModel(settings)
        embedded: list[tuple[int, str, list[float], dict]] = []
        batch = settings.EMBEDDING_BATCH_SIZE
        index = 0
        for i in range(0, len(pieces), batch):
            window = pieces[i : i + batch]
            vectors = await embedding.embed([t for t, _ in window])
            for (piece, meta), vec in zip(window, vectors):
                embedded.append((index, piece, vec, meta))
                index += 1

        await store.store_chunks_and_complete(pool, doc_id, EVAL_KB_ID, embedded)
        return len(embedded)
    finally:
        await pool.close()


def main() -> None:
    import sys

    from rag.common.platform import setup_windows_loop

    setup_windows_loop()

    n = asyncio.run(seed())
    print(f"评测语料入库完成：{n} 个 chunk -> KB {EVAL_KB_ID}")


if __name__ == "__main__":
    main()
