from arq import ArqRedis
from fastapi import Request
from minio import Minio


def get_minio(request: Request) -> Minio:
    return request.app.state.minio


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool
