import hashlib

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from rag.api.dependence.db import get_pg
from rag.api.dependence.storage import get_arq_pool, get_minio
from rag.common.minio_client import put_object
from rag.config import get_settings
from rag.document import DEFAULT_KB_ID, store

document_router = APIRouter(prefix="/documents")

ALLOWED_TYPES = {"txt", "md", "pdf"}


def _safe_filename(name: str | None) -> str:
    base = (name or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return base or "upload"


@document_router.post("/", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(DEFAULT_KB_ID),
    pg=Depends(get_pg),
    minio=Depends(get_minio),
    arq_pool=Depends(get_arq_pool),
):
    settings = get_settings()
    filename = _safe_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件超过大小上限")

    content_hash = hashlib.sha256(data).hexdigest()
    object_key = f"{knowledge_base_id}/{content_hash[:16]}-{filename}"

    await put_object(
        minio,
        settings.minio_bucket,
        object_key,
        data,
        file.content_type or "application/octet-stream",
    )
    document_id = await store.create_document(
        pg,
        knowledge_base_id=knowledge_base_id,
        filename=filename,
        content_type=ext,
        size_bytes=len(data),
        content_hash=content_hash,
        object_key=object_key,
    )
    await arq_pool.enqueue_job("ingest_document", document_id)

    return JSONResponse(
        status_code=202, content={"document_id": document_id, "status": "pending"}
    )


@document_router.get("/{document_id}")
async def get_document_status(document_id: str, pg=Depends(get_pg)):
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {
        "document_id": str(doc["id"]),
        "filename": doc["filename"],
        "status": doc["status"],
        "chunk_count": doc["chunk_count"],
        "error": doc["error"],
    }
