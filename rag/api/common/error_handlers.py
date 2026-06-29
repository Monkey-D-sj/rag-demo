from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag.api.common.schemas import ErrorResponse
from rag.common.exception import AppError


def register_error_handlers(app: FastAPI) -> None:
    """把 service 层抛出的领域异常统一映射为 HTTP 响应。"""

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail).model_dump(),
        )
