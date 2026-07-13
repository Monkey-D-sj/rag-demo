from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.api.dependencies.agent import get_llm, get_memory_manager, get_reranker, get_retriever
from rag.api.modules.chat import service
from rag.document.retriever import KnowledgeRetriever
from rag.agent.memory import MemoryManager
from rag.models.base import ChatModel
from rag.models.rerank import QwenReranker

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    session_id: str
    query: str
    knowledge_base_id: str = "00000000-0000-0000-0000-000000000001"  # 默认向后兼容


@chat_router.post("/")
async def chat(
    body: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    llm: ChatModel = Depends(get_llm),
    retriever: KnowledgeRetriever = Depends(get_retriever),
    reranker: QwenReranker | None = Depends(get_reranker),
):
    return StreamingResponse(
        service.stream_chat(
            body.session_id,
            body.query,
            knowledge_base_id=body.knowledge_base_id,
            llm=llm,
            memory_manager=memory_manager,
            retriever=retriever,
            reranker=reranker,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
