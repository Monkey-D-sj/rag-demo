from fastapi import FastAPI
from rag.api.modules.chat import chat_router


def register_modules(app: FastAPI):
	app.include_router(chat_router)
