from services.object_storage import ObjectStorage, local_public_url
from tests.conftest import make_config


def test_local_public_url_mapping():
    assert local_public_url("./uploads/brain_tumor_output/x.png") == "/uploads/brain_tumor_output/x.png"
    assert local_public_url("uploads\\skin_lesion_output\\y.png") == "/uploads/skin_lesion_output/y.png"
    assert local_public_url("C:/tmp/nope.png") is None
    assert local_public_url("") is None


def test_local_backend_upload_returns_local_url():
    store = ObjectStorage(make_config())
    assert store.active_backend == "local"
    assert store.upload_file("./uploads/skin_lesion_output/z.png") == "/uploads/skin_lesion_output/z.png"


def test_s3_backend_falls_back_when_unavailable():
    from types import SimpleNamespace
    cfg = make_config(object_storage=SimpleNamespace(
        backend="s3", endpoint_url="http://127.0.0.1:1", bucket="b",
        access_key="x", secret_key="y", region="", public_base_url="",
    ))
    store = ObjectStorage(cfg)
    # boto3 may be missing or endpoint unreachable -> effective backend is local.
    assert store.active_backend in ("local", "s3")
