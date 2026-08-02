import pytest

from tests.conftest import make_config


def test_password_hash_and_verify():
    pytest.importorskip("passlib")
    from services.auth import hash_password, verify_password
    h = hash_password("s3cret")
    assert h != "s3cret"
    assert verify_password("s3cret", h) is True
    assert verify_password("wrong", h) is False


def test_long_password_hash_and_verify():
    pytest.importorskip("passlib")
    from services.auth import hash_password, verify_password

    long_password = "x" * 80
    hashed = hash_password(long_password)

    assert hashed != long_password
    assert verify_password(long_password, hashed) is True


def test_jwt_roundtrip():
    pytest.importorskip("jose")
    from services.auth import create_access_token, decode_access_token
    cfg = make_config()
    token = create_access_token(cfg, subject="alice", role="doctor")
    payload = decode_access_token(cfg, token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "doctor"


def test_jwt_rejects_tampered_token():
    pytest.importorskip("jose")
    from jose import JWTError
    from services.auth import create_access_token, decode_access_token
    cfg = make_config()
    token = create_access_token(cfg, subject="bob", role="patient")
    with pytest.raises(JWTError):
        decode_access_token(cfg, token + "tamper")
