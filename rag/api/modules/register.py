from fastapi import FastAPI

from rag.api.modules.chat import chat_router
from rag.api.modules.document import document_router
from rag.api.modules.health import health_router


def register_modules(app: FastAPI):
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(document_router)
