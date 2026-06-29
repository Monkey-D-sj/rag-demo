"""API 通用 Schema —— 供 exception handler / OpenAPI 文档复用。"""

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """统一错误响应体。"""

    detail: str = Field(..., description="人类可读的错误描述")
