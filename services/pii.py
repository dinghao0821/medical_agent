"""PII masking utilities (P3 compliance).

Best-effort redaction of common personally-identifiable information before it is
written to logs / audit records. Regex-based and conservative; not a substitute
for a full de-identification pipeline, but removes the obvious high-risk fields.
"""

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Chinese mainland phone numbers (11 digits starting with 1).
_PHONE_CN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# Generic long digit runs (e.g. ID cards / MRNs): 12+ digits (with optional X for CN ID).
_ID_LIKE = re.compile(r"(?<!\d)\d{12,18}[Xx]?(?!\d)")
# International-ish phone with separators.
_PHONE_GENERIC = re.compile(r"(?<!\w)(\+?\d[\d\-\s]{7,}\d)(?!\w)")


def mask_pii(text) -> str:
    """Return ``text`` with emails / phones / ID-like numbers masked."""
    if not text:
        return text
    if not isinstance(text, str):
        text = str(text)

    text = _EMAIL.sub("[EMAIL]", text)
    text = _ID_LIKE.sub("[ID]", text)
    text = _PHONE_CN.sub("[PHONE]", text)
    text = _PHONE_GENERIC.sub("[PHONE]", text)
    return text
