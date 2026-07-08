"""Document 模块请求/响应 Schema。"""

from datetime import datetime
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
    graph_status: str | None = Field(
        default=None,
        description="实体抽取状态：pending/processing/done/failed/skipped",
    )
    graph_error: str | None = Field(
        default=None,
        description="实体抽取失败原因（仅 graph_status=failed 时有值）",
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


class DocumentListItem(BaseModel):
    """GET /documents/ — 列表中的单个文档。"""

    document_id: str = Field(..., description="文档 UUID")
    knowledge_base_id: str = Field(..., description="所属知识库 UUID")
    filename: str = Field(..., description="原始文件名")
    content_type: str = Field(..., description="文件类型(txt/md/pdf/docx)")
    size_bytes: int = Field(..., description="文件大小(字节)")
    status: DocumentStatus = Field(..., description="当前处理状态")
    chunk_count: int = Field(..., description="切块数量")
    error: str | None = Field(default=None, description="失败原因(仅 failed)")
    graph_status: str | None = Field(default=None, description="实体抽取状态")
    graph_error: str | None = Field(default=None, description="实体抽取失败原因")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="最近更新时间")


class DocumentListResponse(BaseModel):
    """GET /documents/ — 分页文档列表。"""

    total: int = Field(..., description="满足过滤条件的文档总数")
    limit: int = Field(..., description="本页请求的最大条数")
    offset: int = Field(..., description="本页偏移量")
    items: list[DocumentListItem] = Field(..., description="当前页文档")


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


class GraphRetryResponse(BaseModel):
    """POST /documents/{id}/retry-graph — 仅重跑实体图抽取结果。"""

    document_id: str = Field(..., description="文档 UUID")
    graph_status: str = Field(..., description="实体抽取当前状态")
    message: str = Field(..., description="操作结果说明")

    model_config = {"json_schema_extra": {
        "example": {
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "graph_status": "pending",
            "message": "已投递实体抽取任务",
        }
    }}
