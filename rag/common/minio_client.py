import asyncio
import io

from minio import Minio

from rag.config import Settings


def create_minio_client(settings: Settings) -> Minio:
    """创建 minio 客户端并确保 bucket 存在。"""
    client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )
    if not client.bucket_exists(settings.MINIO_BUCKET):
        client.make_bucket(settings.MINIO_BUCKET)
    return client


async def put_object(
    client: Minio, bucket: str, key: str, data: bytes, content_type: str
) -> None:
    """上传字节对象(同步 SDK 转线程池,避免阻塞 event loop)。"""
    await asyncio.to_thread(
        client.put_object, bucket, key, io.BytesIO(data), len(data), content_type
    )


async def get_object(client: Minio, bucket: str, key: str) -> bytes:
    """下载对象字节。"""

    def _get() -> bytes:
        resp = client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            try:
                resp.close()
            finally:
                resp.release_conn()

    return await asyncio.to_thread(_get)


async def get_object_to_file(client: Minio, bucket: str, key: str) -> str:
    """流式下载 MinIO 对象到临时文件,返回文件路径。调用方负责用完后删除。"""

    def _download() -> str:
        import tempfile

        resp = client.get_object(bucket, key)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
                for chunk in resp.stream(amt=64 * 1024):
                    f.write(chunk)
                return f.name
        finally:
            try:
                resp.close()
            finally:
                resp.release_conn()

    return await asyncio.to_thread(_download)


async def list_objects(
    client: Minio, bucket: str, prefix: str = ""
) -> list[dict]:
    """列出 bucket 中的对象元数据（线程池化）。"""

    def _list() -> list[dict]:
        return [
            {
                "key": obj.object_name,
                "size": obj.size,
                "content_type": obj.content_type or "",
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else "",
            }
            for obj in client.list_objects(bucket, prefix=prefix)
        ]

    return await asyncio.to_thread(_list)


async def presigned_get_url(
    client: Minio, bucket: str, key: str, expires_seconds: int = 3600
) -> str:
    """生成预签名下载 URL（默认 1 小时有效）。"""

    def _presign() -> str:
        return client.presigned_get_object(bucket, key, expires=expires_seconds)

    return await asyncio.to_thread(_presign)
