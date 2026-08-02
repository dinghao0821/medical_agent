"""Audit logging (P3 compliance).

Writes an append-only audit record for diagnostic / sensitive operations. The
``detail`` field is PII-masked and, when an encryption key is configured,
encrypted at rest with Fernet (AES). All failures are swallowed with a warning
so auditing never breaks the main request flow.
"""

import logging

logger = logging.getLogger(__name__)

_fernet = None
_fernet_tried = False


def _get_fernet(config):
    global _fernet, _fernet_tried
    if _fernet_tried:
        return _fernet
    _fernet_tried = True
    key = getattr(getattr(config, "auth", None), "encryption_key", "") or ""
    if not key:
        _fernet = None
        return None
    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as e:
        logger.warning("Audit encryption key invalid (%s); storing detail unencrypted.", e)
        _fernet = None
    return _fernet


def write_audit(config, *, username=None, role=None, action=None, agent=None,
                session_id=None, validation_result=None, detail=None):
    """Persist one audit record. No-op (warns) on any failure."""
    auth_cfg = getattr(config, "auth", None)
    if not bool(getattr(auth_cfg, "enable_audit", False)):
        return
    try:
        from services.db import init_db, is_ready, get_session
        from services.models import AuditLog

        if not is_ready():
            if not init_db(config):
                return

        safe_detail = detail
        if safe_detail:
            if bool(getattr(auth_cfg, "enable_pii_masking", True)):
                from services.pii import mask_pii
                safe_detail = mask_pii(safe_detail)
            fernet = _get_fernet(config)
            if fernet is not None:
                try:
                    safe_detail = fernet.encrypt(safe_detail.encode("utf-8")).decode("utf-8")
                except Exception as e:
                    logger.warning("Audit detail encryption failed (%s); storing masked plaintext.", e)

        session = get_session()
        try:
            session.add(AuditLog(
                username=username,
                role=role,
                action=action,
                agent=agent,
                session_id=session_id,
                validation_result=validation_result,
                detail=safe_detail,
            ))
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning("Audit write failed (%s); continuing.", e)
