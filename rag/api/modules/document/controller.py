from typing import Annotated

from arq import ArqRedis
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import RedirectResponse
from minio import Minio
from psycopg_pool import AsyncConnectionPool

from rag.api.common.schemas import ErrorResponse
from rag.api.dependencies.db import get_pg
from rag.api.dependencies.storage import get_arq_pool, get_minio
from rag.api.modules.document import service
from rag.api.modules.document.schemas import (
    DocumentListResponse,
    DocumentRetryResponse,
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from rag.document import DEFAULT_KB_ID

document_router = APIRouter(prefix="/documents")


@document_router.post(
    "/",
    status_code=202,
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "文件类型不支持 / 文件超限"},
        500: {"model": ErrorResponse, "description": "服务端异常"},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(DEFAULT_KB_ID),
    pg=Depends(get_pg),
    minio=Depends(get_minio),
    arq_pool=Depends(get_arq_pool),
) -> DocumentUploadResponse:
    document_id = await service.ingest_upload(
        pg=pg,
        minio=minio,
        arq_pool=arq_pool,
        filename=file.filename,
        content_type=file.content_type,
        data=await file.read(),
        knowledge_base_id=knowledge_base_id,
    )
    return DocumentUploadResponse(document_id=document_id, status="pending")


@document_router.get(
    "/",
    response_model=DocumentListResponse,
)
async def list_documents(
    pg: Annotated[AsyncConnectionPool, Depends(get_pg)],
    knowledge_base_id: str | None = Query(
        None, description="按知识库 UUID 过滤;不传则返回全部"
    ),
    status: DocumentStatus | None = Query(None, description="按处理状态过滤"),
    limit: int = Query(20, ge=1, le=100, description="每页条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
) -> DocumentListResponse:
    return await service.list_documents(
        pg,
        knowledge_base_id=knowledge_base_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@document_router.get(
    "/{document_id}",
    response_model=DocumentStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "文档不存在"},
    },
)
async def get_document_status(document_id: str, pg=Depends(get_pg)):
    return await service.get_status(pg, document_id)


@document_router.get(
    "/{document_id}/download",
    responses={
        302: {"description": "重定向到 MinIO 预签名下载链接"},
        404: {"model": ErrorResponse, "description": "文档不存在"},
    },
)
async def download_document(
    document_id: str,
    pg: Annotated[AsyncConnectionPool, Depends(get_pg)],
    minio: Annotated[Minio, Depends(get_minio)],
    expires: int = Query(3600, ge=60, le=86400, description="预签名有效期（秒）"),
) -> RedirectResponse:
    url = await service.get_download_url(pg, minio, document_id, expires)
    return RedirectResponse(url)


@document_router.post(
    "/{document_id}/retry",
    status_code=202,
    response_model=DocumentRetryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "文档不存在"},
    },
)
async def retry_document(
    document_id: str,
    pg: Annotated[AsyncConnectionPool, Depends(get_pg)],
    arq_pool: Annotated[ArqRedis, Depends(get_arq_pool)],
) -> DocumentRetryResponse:
    return await service.retry_document(pg, arq_pool, document_id)
