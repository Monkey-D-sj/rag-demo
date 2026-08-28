from fastapi import Request
from minio import Minio

from rag.tasks import TaskPublisher


def get_minio(request: Request) -> Minio:
    return request.app.state.minio


def get_task_publisher(request: Request) -> TaskPublisher:
    return request.app.state.task_publisher
