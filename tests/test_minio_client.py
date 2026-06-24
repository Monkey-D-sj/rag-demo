import rag.common.minio_client as mc
from rag.config import Settings


class _FakeMinio:
    def __init__(self):
        self.exists = False
        self.made = []

    def bucket_exists(self, bucket):
        return self.exists

    def make_bucket(self, bucket):
        self.made.append(bucket)


def test_create_minio_makes_bucket_when_missing(monkeypatch):
    fake = _FakeMinio()
    monkeypatch.setattr(mc, "Minio", lambda *a, **k: fake)
    client = mc.create_minio_client(Settings())
    assert client is fake
    assert fake.made == ["rag-documents"]


def test_create_minio_skips_make_when_exists(monkeypatch):
    fake = _FakeMinio()
    fake.exists = True
    monkeypatch.setattr(mc, "Minio", lambda *a, **k: fake)
    mc.create_minio_client(Settings())
    assert fake.made == []


async def test_put_object_passes_bytes_and_length():
    calls = []

    class _C:
        def put_object(self, bucket, key, stream, length, content_type):
            calls.append((bucket, key, stream.read(), length, content_type))

    await mc.put_object(_C(), "bk", "k1", b"hello", "text/plain")
    assert calls == [("bk", "k1", b"hello", 5, "text/plain")]


async def test_get_object_reads_and_closes():
    closed = {"closed": False, "released": False}

    class _Resp:
        def read(self):
            return b"data"

        def close(self):
            closed["closed"] = True

        def release_conn(self):
            closed["released"] = True

    class _C:
        def get_object(self, bucket, key):
            return _Resp()

    out = await mc.get_object(_C(), "bk", "k1")
    assert out == b"data"
    assert closed == {"closed": True, "released": True}
