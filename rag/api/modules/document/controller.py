from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from rag.api.dependence.db import get_pg
from rag.api.dependence.storage import get_arq_pool, get_minio
from rag.api.modules.document import service
from rag.document import DEFAULT_KB_ID

document_router = APIRouter(prefix="/documents")


@document_router.post("/", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(DEFAULT_KB_ID),
    pg=Depends(get_pg),
    minio=Depends(get_minio),
    arq_pool=Depends(get_arq_pool),
):
    document_id = await service.ingest_upload(
        pg=pg,
        minio=minio,
        arq_pool=arq_pool,
        filename=file.filename,
        content_type=file.content_type,
        data=await file.read(),
        knowledge_base_id=knowledge_base_id,
    )
    return JSONResponse(
        status_code=202, content={"document_id": document_id, "status": "pending"}
    )


@document_router.get("/{document_id}")
async def get_document_status(document_id: str, pg=Depends(get_pg)):
    return await service.get_status(pg, document_id)
