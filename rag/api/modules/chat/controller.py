from fastapi import APIRouter, Depends
from pydantic import BaseModel

from rag.api.dependence.agent import get_llm, get_memory_manager
from rag.memory import MemoryManager
from rag.models.base import ChatModel
from rag.type import ContextSchema
from rag.workflow import invoke

chat_router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    session_id: str
    query: str


@chat_router.post("/")
async def chat(
    body: ChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    llm: ChatModel = Depends(get_llm),
):
    context = ContextSchema(llm=llm, memory_manager=memory_manager)
    chunks = [chunk async for chunk in invoke(body.session_id, body.query, context)]
    return {"chunks": chunks}
