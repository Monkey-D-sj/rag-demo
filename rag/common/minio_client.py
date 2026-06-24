import asyncio
import io

from minio import Minio

from rag.config import Settings


def create_minio_client(settings: Settings) -> Minio:
    """创建 minio 客户端并确保 bucket 存在。"""
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)
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
