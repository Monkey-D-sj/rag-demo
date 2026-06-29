from fastapi import APIRouter, Depends, Query
from minio import Minio

from rag.api.dependencies.storage import get_minio
from rag.api.modules.files.schemas import FileItem, FileListResponse
from rag.common.minio_client import list_objects, presigned_get_url
from rag.config import get_settings

files_router = APIRouter(prefix="/files")


@files_router.get("/", response_model=FileListResponse)
async def list_files(
    minio: Minio = Depends(get_minio),
    prefix: str = Query("", description="对象 key 前缀过滤"),
):
    """列出 MinIO bucket 中的所有对象。"""
    settings = get_settings()
    objects = await list_objects(minio, settings.minio_bucket, prefix=prefix)
    return FileListResponse(
        bucket=settings.minio_bucket,
        count=len(objects),
        items=[FileItem(**o) for o in objects],
    )


@files_router.get("/{key:path}/download")
async def download_file(
    key: str,
    minio: Minio = Depends(get_minio),
    expires: int = Query(3600, ge=60, le=86400, description="预签名有效期（秒）"),
):
    """生成预签名下载链接，浏览器 302 跳转直连 MinIO 下载。"""
    from fastapi.responses import RedirectResponse

    settings = get_settings()
    url = await presigned_get_url(minio, settings.minio_bucket, key, expires)
    return RedirectResponse(url)
