"""Runtime configuration, read from the environment.

Deliberately dependency-free (no pydantic-settings) so the Modal image stays
small and the mock container boots fast.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    api_token: str
    mock_mode: bool
    session_ttl_seconds: int
    otp_max_attempts: int
    otp_ttl_seconds: int
    mock_jitter_min_ms: int
    mock_jitter_max_ms: int
    speech_segments_before_decision: int

    @classmethod
    def from_env(cls) -> "Settings":
        jitter_min = _env_int("CALLBACK_MOCK_JITTER_MIN_MS", 150)
        jitter_max = _env_int("CALLBACK_MOCK_JITTER_MAX_MS", 400)
        if jitter_max < jitter_min:
            raise ValueError(
                "CALLBACK_MOCK_JITTER_MAX_MS must be >= CALLBACK_MOCK_JITTER_MIN_MS"
            )
        return cls(
            api_token=os.environ.get("CALLBACK_API_TOKEN", "dev-token"),
            mock_mode=_env_bool("CALLBACK_MOCK_MODE", True),
            session_ttl_seconds=_env_int("CALLBACK_SESSION_TTL_SECONDS", 300),
            otp_max_attempts=_env_int("CALLBACK_OTP_MAX_ATTEMPTS", 3),
            otp_ttl_seconds=_env_int("CALLBACK_OTP_TTL_SECONDS", 120),
            mock_jitter_min_ms=jitter_min,
            mock_jitter_max_ms=jitter_max,
            speech_segments_before_decision=_env_int(
                "CALLBACK_SPEECH_SEGMENTS_BEFORE_DECISION", 2
            ),
        )
