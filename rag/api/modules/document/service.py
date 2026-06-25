import hashlib

from rag.api.modules.document.exceptions import (
    DocumentNotFound,
    FileTooLarge,
    UnsupportedFileType,
)
from rag.common.minio_client import put_object
from rag.config import get_settings
from rag.document import store

ALLOWED_TYPES = {"txt", "md", "pdf"}


def _safe_filename(name: str | None) -> str:
    base = (name or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return base or "upload"


async def ingest_upload(
    *,
    pg,
    minio,
    arq_pool,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    knowledge_base_id: str,
) -> str:
    """落库 + 上传对象存储 + 投递解析任务,返回 document_id。"""
    settings = get_settings()
    name = _safe_filename(filename)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_TYPES:
        raise UnsupportedFileType(f"不支持的文件类型: {ext}")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLarge("文件超过大小上限")

    content_hash = hashlib.sha256(data).hexdigest()
    object_key = f"{knowledge_base_id}/{content_hash[:16]}-{name}"

    await put_object(
        minio,
        settings.minio_bucket,
        object_key,
        data,
        content_type or "application/octet-stream",
    )
    document_id = await store.create_document(
        pg,
        knowledge_base_id=knowledge_base_id,
        filename=name,
        content_type=ext,
        size_bytes=len(data),
        content_hash=content_hash,
        object_key=object_key,
    )
    await arq_pool.enqueue_job("ingest_document", document_id)
    return document_id


async def get_status(pg, document_id: str) -> dict:
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    return {
        "document_id": str(doc["id"]),
        "filename": doc["filename"],
        "status": doc["status"],
        "chunk_count": doc["chunk_count"],
        "error": doc["error"],
    }


async def retry_document(pg, arq_pool, document_id: str) -> dict:
    """将失败文档重置为 pending 并重新投递入库任务。"""
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    if doc["status"] not in ("failed",):
        return {
            "document_id": str(doc["id"]),
            "status": doc["status"],
            "message": "文档未处于 failed 状态，无需重试",
        }
    await store.set_status(pg, document_id, "pending")
    await arq_pool.enqueue_job("ingest_document", document_id)
    return {
        "document_id": str(doc["id"]),
        "status": "pending",
        "message": "已重新投递入库任务",
    }
