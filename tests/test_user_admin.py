"""Tests for services.user_admin: listing users and safely changing roles."""

import os
import tempfile

import pytest

from tests.conftest import make_config


@pytest.fixture
def db_config():
    """A Config-like object pointing at a fresh temp SQLite DB per test."""
    import services.db as db_module

    # Reset module-level engine/session singletons so each test gets an
    # isolated DB (services.db caches _engine/_SessionLocal globally).
    db_module._engine = None
    db_module._SessionLocal = None

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_app.db")
    cfg = make_config(auth=type(
        "AuthCfg", (), {
            "enabled": True, "jwt_secret": "x", "jwt_algorithm": "HS256",
            "token_expire_minutes": 60, "database_url": f"sqlite:///{db_path}",
        }
    )())
    yield cfg

    db_module._engine = None
    db_module._SessionLocal = None


def _make_user(config, username, role="patient"):
    from services.db import init_db, is_ready, get_session
    from services.models import User
    from services.auth import hash_password

    if not is_ready():
        init_db(config)
    session = get_session()
    try:
        session.add(User(username=username, hashed_password=hash_password("Xx1!aaaa"), role=role))
        session.commit()
    finally:
        session.close()


def test_list_users_returns_no_password_hash(db_config):
    from services.user_admin import list_users

    _make_user(db_config, "alice", role="patient")
    users = list_users(db_config, )
    assert len(users) == 1
    assert users[0]["username"] == "alice"
    assert "hashed_password" not in users[0]
    assert "password" not in users[0]


def test_list_users_filters_by_role_and_search(db_config):
    from services.user_admin import list_users

    _make_user(db_config, "alice_admin", role="admin")
    _make_user(db_config, "bob_doc", role="doctor")
    _make_user(db_config, "carol", role="patient")

    assert [u["username"] for u in list_users(db_config, role="doctor")] == ["bob_doc"]
    assert [u["username"] for u in list_users(db_config, search="alice")] == ["alice_admin"]
    assert len(list_users(db_config)) == 3


def test_promote_patient_to_admin(db_config):
    from services.user_admin import update_user_role

    _make_user(db_config, "root_admin", role="admin")
    _make_user(db_config, "carol", role="patient")

    updated = update_user_role(db_config, "carol", "admin", actor_username="root_admin")
    assert updated["role"] == "admin"


def test_promote_to_doctor_resets_licence_status(db_config):
    from services.user_admin import update_user_role
    from services.doctor_verification import get_status

    _make_user(db_config, "root_admin", role="admin")
    _make_user(db_config, "carol", role="patient")

    update_user_role(db_config, "carol", "doctor", actor_username="root_admin")
    info = get_status(db_config, "carol")
    assert info["role"] == "doctor"
    assert info["doctor_status"] == "unsubmitted"  # must re-verify, never bypassed


def test_cannot_demote_last_admin(db_config):
    from services.user_admin import update_user_role

    _make_user(db_config, "only_admin", role="admin")

    with pytest.raises(PermissionError):
        update_user_role(db_config, "only_admin", "patient", actor_username="only_admin")


def test_can_demote_admin_when_another_admin_exists(db_config):
    from services.user_admin import update_user_role

    _make_user(db_config, "admin_one", role="admin")
    _make_user(db_config, "admin_two", role="admin")

    updated = update_user_role(db_config, "admin_one", "patient", actor_username="admin_two")
    assert updated["role"] == "patient"


def test_update_role_rejects_invalid_role(db_config):
    from services.user_admin import update_user_role

    _make_user(db_config, "root_admin", role="admin")
    _make_user(db_config, "carol", role="patient")

    with pytest.raises(ValueError):
        update_user_role(db_config, "carol", "superuser", actor_username="root_admin")


def test_update_role_missing_user_returns_none(db_config):
    from services.user_admin import update_user_role

    _make_user(db_config, "root_admin", role="admin")
    assert update_user_role(db_config, "ghost", "patient", actor_username="root_admin") is None
