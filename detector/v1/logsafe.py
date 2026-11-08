"""Log-safe stand-ins for values that must never be written down raw.

`room` and `caller_identity` can carry a phone number (a SIP participant's
identity often does), so a log line carries `tag(value)` instead: an HMAC
under a key that exists only in this process's memory. Two lines about the
same room share a tag, so a reader can still follow one call, but the tag
cannot be reversed or matched against another process's logs.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# Per process, never persisted, never logged.
_KEY = secrets.token_bytes(32)


def tag(value: str) -> str:
    """First 16 hex characters of HMAC-SHA256(per-process key, value)."""

    return hmac.new(_KEY, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
