from types import SimpleNamespace

from rag.api.dependence.storage import get_arq_pool, get_minio


def test_get_minio_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(minio=sentinel)))
    assert get_minio(request) is sentinel


def test_get_arq_pool_reads_app_state():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=sentinel)))
    assert get_arq_pool(request) is sentinel
