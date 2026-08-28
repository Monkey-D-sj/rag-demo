import hashlib

from rag.api.modules.document.exceptions import DocumentNotFound
from rag.api.modules.document.schemas import (
    DocumentListItem,
    DocumentListResponse,
    DocumentRetryResponse,
    DocumentStatusResponse,
    GraphRetryResponse,
)
from rag.common.minio_client import presigned_get_url, put_object
from rag.config import get_settings
from rag.document import store
from rag.tasks import TaskPublisher

ALLOWED_TYPES = {"txt", "md", "pdf", "docx"}


def _safe_filename(name: str | None) -> str:
    base = (name or "upload").replace("\\", "/").rsplit("/", 1)[-1]
    return base or "upload"


async def ingest_upload(
    *,
    pg,
    minio,
    task_publisher: TaskPublisher,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    knowledge_base_id: str,
) -> str:
    """落库 + 上传对象存储 + 投递解析任务，返回 document_id。"""
    settings = get_settings()
    name = _safe_filename(filename)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    content_hash = hashlib.sha256(data).hexdigest()
    object_key = f"{knowledge_base_id}/{content_hash[:16]}-{name}"

    await put_object(
        minio,
        settings.MINIO_BUCKET,
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
    await task_publisher.enqueue("ingest_document", document_id)
    return document_id


async def get_status(pg, document_id: str) -> DocumentStatusResponse:
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    return DocumentStatusResponse(
        document_id=str(doc["id"]),
        filename=doc["filename"],
        status=doc["status"],
        chunk_count=doc["chunk_count"],
        error=doc["error"],
        graph_status=doc.get("graph_status"),
        graph_error=doc.get("graph_error"),
    )


async def list_documents(
    pg,
    *,
    knowledge_base_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> DocumentListResponse:
    """分页查询文档列表,可按知识库与状态过滤。"""
    rows, total = await store.list_documents(
        pg,
        knowledge_base_id=knowledge_base_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    items = [
        DocumentListItem(
            document_id=str(row["id"]),
            knowledge_base_id=str(row["knowledge_base_id"]),
            filename=row["filename"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            error=row["error"],
            graph_status=row.get("graph_status"),
            graph_error=row.get("graph_error"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]
    return DocumentListResponse(total=total, limit=limit, offset=offset, items=items)


async def retry_document(pg, task_publisher: TaskPublisher, document_id: str) -> DocumentRetryResponse:
    """强制重试：重置 retry_count=0 并立即投递，旁路 cron 退避。

    cron 自愈对同文档每轮递增 retry_count 并按指数退避；本接口是运维入口，
    清零重试计数让文档获得完整重试预算，适合修好根因后批量恢复。
    """
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    if doc["status"] not in ("failed",):
        return DocumentRetryResponse(
            document_id=str(doc["id"]),
            status=doc["status"],
            message="文档未处于 failed 状态，无需重试",
        )
    await store.force_retry(pg, document_id)
    await task_publisher.enqueue("ingest_document", document_id)
    return DocumentRetryResponse(
        document_id=str(doc["id"]),
        status="pending",
        retry_count=0,
        message="已重置重试计数并投递入库任务",
    )


async def retry_graph(pg, task_publisher: TaskPublisher, document_id: str) -> GraphRetryResponse:
    """仅重跑实体图抽取：跳过切块+向量，直接投递 extract_document_entities。"""
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    if doc["status"] != "done":
        return GraphRetryResponse(
            document_id=str(doc["id"]),
            graph_status=doc.get("graph_status", "unknown"),
            message="文档尚未完成向量入库，请先等入库完成后再重试图抽取",
        )

    gs = doc.get("graph_status")
    if gs not in ("failed", "skipped", "pending", None):
        return GraphRetryResponse(
            document_id=str(doc["id"]),
            graph_status=gs or "unknown",
            message=f"实体抽取状态为 {gs}，无需重试",
        )

    await task_publisher.enqueue("extract_document_entities", document_id)
    return GraphRetryResponse(
        document_id=str(doc["id"]),
        graph_status="pending",
        message="已投递实体抽取任务",
    )


async def get_download_url(pg, minio, document_id: str, expires: int = 3600) -> str:
    """生成文档原始文件的预签名下载链接。"""
    settings = get_settings()
    doc = await store.get_document(pg, document_id)
    if doc is None:
        raise DocumentNotFound("文档不存在")
    return await presigned_get_url(
        minio, settings.MINIO_BUCKET, doc["object_key"], expires
    )
