"""Authentication core (P3): password hashing + JWT (access & refresh tokens).

Third-party deps (``passlib``, ``python-jose``) are imported lazily so the app
runs even when they are not installed, as long as auth is disabled
(``ENABLE_AUTH=false``, the default).

Tokens
------
Every token carries a unique ``jti`` (JWT ID) and a ``type`` claim
("access" | "refresh"), enabling per-token revocation via
``services.token_store``. Legacy tokens minted before this change (no ``type``)
are still accepted as access tokens for backward compatibility.
"""

import time
import uuid
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

VALID_ROLES = ("patient", "doctor", "admin")
ACCESS = "access"
REFRESH = "refresh"

# Password policy for production
_MIN_PASSWORD_LENGTH = 8
_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"


def validate_password_strength(password: str) -> tuple[bool, str]:
    """Validate password meets production security requirements.
    
    Returns (is_valid, error_message).
    """
    if len(password) < _MIN_PASSWORD_LENGTH:
        return False, f"密码至少需要{_MIN_PASSWORD_LENGTH}个字符"
    if not any(c.isupper() for c in password):
        return False, "密码需包含至少一个大写字母"
    if not any(c.islower() for c in password):
        return False, "密码需包含至少一个小写字母"
    if not any(c.isdigit() for c in password):
        return False, "密码需包含至少一个数字"
    if not any(c in _SPECIAL_CHARS for c in password):
        return False, "密码需包含至少一个特殊字符"
    return True, ""

_pwd_context = None


def _get_pwd_context():
    global _pwd_context
    if _pwd_context is None:
        # Passlib 1.7.4 still reads bcrypt.__about__.__version__, which
        # bcrypt 4.1+ removed. Restore only that legacy version metadata.
        import bcrypt
        if not hasattr(bcrypt, "__about__"):
            from types import SimpleNamespace
            bcrypt.__about__ = SimpleNamespace(
                __version__=getattr(bcrypt, "__version__", "unknown")
            )
        from passlib.context import CryptContext
        _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return _pwd_context


def _normalize_password(password: str) -> bytes:
    """Prepare a password for bcrypt-compatible hashing.

    bcrypt only accepts input up to 72 bytes. Truncating at the byte boundary
    preserves compatibility for long passwords while keeping the existing
    auth semantics predictable for normal inputs.
    """
    if password is None:
        return b""
    if isinstance(password, bytes):
        encoded = password
    elif isinstance(password, str):
        encoded = password.encode("utf-8")
    else:
        encoded = str(password).encode("utf-8")
    return encoded[:72]


def hash_password(password: str) -> str:
    try:
        import bcrypt

        normalized = _normalize_password(password)
        return bcrypt.hashpw(normalized, bcrypt.gensalt()).decode("utf-8")
    except Exception as exc:
        logger.warning("bcrypt hashing failed, falling back to passlib: %s", exc)
        return _get_pwd_context().hash(_normalize_password(password).decode("utf-8", errors="ignore"))


def verify_password(password: str, hashed: str) -> bool:
    try:
        import bcrypt

        normalized = _normalize_password(password)
        if isinstance(hashed, str):
            hashed_bytes = hashed.encode("utf-8")
        else:
            hashed_bytes = hashed
        return bcrypt.checkpw(normalized, hashed_bytes)
    except Exception as e:
        logger.warning("Password verification error: %s", e)
        return False


def _new_jti() -> str:
    return uuid.uuid4().hex


def _access_ttl_minutes(config) -> int:
    """Access-token lifetime.

    When refresh tokens are enabled we use the short-lived value; otherwise we
    keep the historical ``token_expire_minutes`` so behaviour never regresses.
    """
    ac = config.auth
    if getattr(ac, "refresh_token_enabled", False):
        return int(getattr(ac, "access_token_expire_minutes", 15))
    return int(ac.token_expire_minutes)


def _encode(config, subject: str, role: str, token_type: str, ttl_minutes: int) -> str:
    try:
        from jose import jwt
    except Exception:
        from jose import jwt as jose_jwt  # type: ignore
        jwt = jose_jwt

    ac = config.auth
    now = datetime.utcnow()
    payload = {
        "sub": subject,
        "role": role,
        "type": token_type,
        "jti": _new_jti(),
        "iat": now,
        "exp": now + timedelta(minutes=int(ttl_minutes)),
    }
    return jwt.encode(payload, ac.jwt_secret, algorithm=ac.jwt_algorithm)


def create_access_token(config, subject: str, role: str) -> str:
    """Mint a signed access token (short-lived when refresh is enabled)."""
    return _encode(config, subject, role, ACCESS, _access_ttl_minutes(config))


def create_refresh_token(config, subject: str, role: str) -> str:
    """Mint a signed, long-lived refresh token."""
    ttl = int(getattr(config.auth, "refresh_token_expire_minutes", 10080))
    return _encode(config, subject, role, REFRESH, ttl)


def decode_token(config, token: str, expected_type: str = None) -> dict:
    """Decode/verify a JWT and optionally assert its ``type``.

    Raises ``jose.JWTError`` (or a ValueError for a type mismatch) on any
    invalid/expired/wrong-kind token. Legacy tokens without a ``type`` claim are
    treated as access tokens.
    """
    try:
        from jose import jwt
    except Exception:
        from jose import jwt as jose_jwt  # type: ignore
        jwt = jose_jwt

    ac = config.auth
    payload = jwt.decode(token, ac.jwt_secret, algorithms=[ac.jwt_algorithm])
    if expected_type is not None:
        actual = payload.get("type", ACCESS)  # legacy tokens => access
        if actual != expected_type:
            raise ValueError(f"Expected {expected_type} token, got {actual}")
    return payload


def decode_access_token(config, token: str) -> dict:
    """Backward-compatible helper: decode & verify an access token."""
    return decode_token(config, token, expected_type=ACCESS)


def remaining_ttl_seconds(payload: dict) -> int:
    """Seconds until the token's ``exp``; 0 if already expired/missing."""
    exp = payload.get("exp")
    if not exp:
        return 0
    # jose returns exp as an int epoch after decode.
    if isinstance(exp, datetime):
        exp = exp.timestamp()
    return max(0, int(exp - time.time()))
