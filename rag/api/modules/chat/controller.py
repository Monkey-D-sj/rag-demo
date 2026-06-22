from fastapi import APIRouter


chat_router = APIRouter(prefix="/chat")

@chat_router.post("/")
async def chat():
	return {"message": "chat"}
