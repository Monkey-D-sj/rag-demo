from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.api.dependencies.agent import (
    get_llm,
    get_memory_manager,
    get_reranker,
    get_retriever,
    get_semantic_cache,
)
from rag.api.modules.chat import service, session_store
from rag.document.retriever import KnowledgeRetriever
from rag.agent.memory import MemoryManager
from rag.models.base import ChatModel
from rag.models.rerank import QwenReranker

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    session_id: str
    query: str


@chat_router.post("/")
async def chat(
    body: ChatRequest,
    request: Request,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    llm: ChatModel = Depends(get_llm),
    retriever: KnowledgeRetriever = Depends(get_retriever),
    reranker: QwenReranker | None = Depends(get_reranker),
    semantic_cache=Depends(get_semantic_cache),
):
    return StreamingResponse(
        service.stream_chat(
            body.session_id,
            body.query,
            llm=llm,
            memory_manager=memory_manager,
            retriever=retriever,
            reranker=reranker,
            pool=request.app.state.pg,
            semantic_cache=semantic_cache,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session CRUD ──

session_router = APIRouter(prefix="/sessions")


@session_router.get("/")
async def list_sessions(request: Request) -> list[dict]:
    rows = await session_store.list_sessions(request.app.state.pg)
    return [
        {
            "session_id": r["session_id"],
            "title": r["title"],
            "created_at": r["created_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
        }
        for r in rows
    ]


@session_router.post("/")
async def create_session(request: Request) -> dict:
    row = await session_store.create_session(request.app.state.pg)
    return {
        "session_id": row["session_id"],
        "title": row["title"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


@session_router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    ok = await session_store.delete_session(request.app.state.pg, session_id)
    return {"deleted": ok}


class RenameRequest(BaseModel):
    title: str


@session_router.patch("/{session_id}")
async def rename_session(session_id: str, body: RenameRequest, request: Request) -> dict:
    await session_store.set_title(request.app.state.pg, session_id, body.title)
    return {"ok": True}
