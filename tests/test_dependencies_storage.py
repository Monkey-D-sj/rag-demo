from types import SimpleNamespace

from rag.api.dependencies.storage import get_minio, get_task_publisher


def test_get_minio_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(minio=sentinel)))
    assert get_minio(request) is sentinel


def test_get_task_publisher_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(task_publisher=sentinel)))
    assert get_task_publisher(request) is sentinel
