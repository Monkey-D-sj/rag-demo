"""Document 模块请求/响应 Schema。"""

from typing import Literal

from pydantic import BaseModel, Field

# ── 文档状态字面量 ──────────────────────────────────────────

DocumentStatus = Literal["pending", "processing", "done", "failed"]


# ── Response Schemas ───────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """POST /documents/ — 文件上传已受理。"""

    document_id: str = Field(
        ...,
        description="文档 UUID",
        json_schema_extra={"example": "550e8400-e29b-41d4-a716-446655440000"},
    )
    status: Literal["pending"] = Field(
        default="pending",
        description="初始状态，等待 worker 领取入库",
    )


class DocumentStatusResponse(BaseModel):
    """GET /documents/{id} — 文档处理状态。"""

    document_id: str = Field(..., description="文档 UUID")
    filename: str = Field(..., description="原始文件名")
    status: DocumentStatus = Field(..., description="当前处理状态")
    chunk_count: int | None = Field(
        default=None,
        description="切块数量（仅 done 状态时有值）",
    )
    error: str | None = Field(
        default=None,
        description="失败原因（仅 failed 状态时有值）",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "readme.pdf",
                "status": "done",
                "chunk_count": 12,
                "error": None,
            }
        ]
    }}


class DocumentRetryResponse(BaseModel):
    """POST /documents/{id}/retry — 强制重试结果。"""

    document_id: str = Field(..., description="文档 UUID")
    status: str = Field(..., description="重试后的状态")
    message: str = Field(..., description="操作结果说明")
    retry_count: int | None = Field(
        default=None,
        description="重置后的重试计数（仅实际触发重试时有值）",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "已重置重试计数并投递入库任务",
                "retry_count": 0,
            },
            {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "message": "文档未处于 failed 状态，无需重试",
                "retry_count": None,
            },
        ]
    }}
