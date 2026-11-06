"""Runtime configuration, read from the environment.

Deliberately dependency-free (no pydantic-settings) so the Modal image stays
small and the mock container boots fast.
"""

from __future__ import annotations

import os
import secrets
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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_hosts(name: str) -> tuple[str, ...]:
    """A comma-separated host allowlist, lowercased, blanks dropped."""

    raw = os.environ.get(name, "")
    return tuple(host.strip().lower() for host in raw.split(",") if host.strip())


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
    # True when no CALLBACK_API_TOKEN was supplied and a random one was
    # generated. The service then refuses every caller, which is the safe
    # direction: a misnamed or empty secret must not leave the detector open.
    api_token_generated: bool = False
    # Extra idle time, on top of the TTL, before a session may be
    # evicted. 0 means derive it from the TTL, which is the safe
    # default: retention then stays at least twice the TTL however
    # the TTL is configured.
    session_eviction_grace_seconds: int = 0
    # Ceiling on a session idempotency cache. 0 means use the
    # store default.
    max_cached_responses: int = 0

    # ------------------------------------------------------- audio transport
    #
    # Every one of these defaults to the closed position. `uri` arrives in a
    # request body, so switching fetching on without naming hosts would turn
    # POST /events into a server-side request forgery primitive. See the
    # module docstring of detector/audio.py for the full threat model.
    audio_enabled: bool = False
    #: Hosts this service will fetch audio from. Empty means none, and an
    #: empty allowlist with audio_enabled on still fetches nothing over the
    #: network.
    audio_allowed_hosts: tuple[str, ...] = ()
    #: file: uris, for local development and for the demo harness. Needs
    #: audio_file_root as well: without a root, a file: uri reads anything the
    #: process can read.
    audio_allow_file_uris: bool = False
    audio_file_root: str | None = None
    #: data: uris carry the span inline. No network, so no SSRF, which is why
    #: this one is the exception that defaults on.
    audio_allow_data_uris: bool = True
    audio_max_bytes: int = 8 * 1024 * 1024
    #: Well inside the 800 ms budget shared.client gives the whole call.
    audio_fetch_timeout_ms: int = 400
    audio_max_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        jitter_min = _env_int("CALLBACK_MOCK_JITTER_MIN_MS", 150)
        jitter_max = _env_int("CALLBACK_MOCK_JITTER_MAX_MS", 400)
        if jitter_max < jitter_min:
            raise ValueError(
                "CALLBACK_MOCK_JITTER_MAX_MS must be >= CALLBACK_MOCK_JITTER_MIN_MS"
            )
        # A missing or empty token used to fall back to the literal
        # "dev-token", which is committed in this repo, so a misnamed Modal
        # secret would have left the service wide open. An empty token was
        # worse: compare_digest("", "") is True, so a bare "Bearer "
        # authenticated. Generate an unguessable token instead and fail closed.
        supplied = os.environ.get("CALLBACK_API_TOKEN", "").strip()
        generated = not supplied
        return cls(
            api_token=supplied or secrets.token_urlsafe(32),
            api_token_generated=generated,
            mock_mode=_env_bool("CALLBACK_MOCK_MODE", True),
            session_ttl_seconds=_env_int("CALLBACK_SESSION_TTL_SECONDS", 300),
            otp_max_attempts=_env_int("CALLBACK_OTP_MAX_ATTEMPTS", 3),
            otp_ttl_seconds=_env_int("CALLBACK_OTP_TTL_SECONDS", 120),
            mock_jitter_min_ms=jitter_min,
            mock_jitter_max_ms=jitter_max,
            speech_segments_before_decision=_env_int(
                "CALLBACK_SPEECH_SEGMENTS_BEFORE_DECISION", 2
            ),
            session_eviction_grace_seconds=_env_int(
                "CALLBACK_SESSION_EVICTION_GRACE_SECONDS", 0
            ),
            max_cached_responses=_env_int("CALLBACK_MAX_CACHED_RESPONSES", 0),
            audio_enabled=_env_bool("CALLBACK_AUDIO_ENABLED", False),
            audio_allowed_hosts=_env_hosts("CALLBACK_AUDIO_ALLOWED_HOSTS"),
            audio_allow_file_uris=_env_bool("CALLBACK_AUDIO_ALLOW_FILE_URIS", False),
            audio_file_root=(
                os.environ.get("CALLBACK_AUDIO_FILE_ROOT", "").strip() or None
            ),
            audio_allow_data_uris=_env_bool("CALLBACK_AUDIO_ALLOW_DATA_URIS", True),
            audio_max_bytes=_env_int("CALLBACK_AUDIO_MAX_BYTES", 8 * 1024 * 1024),
            audio_fetch_timeout_ms=_env_int("CALLBACK_AUDIO_FETCH_TIMEOUT_MS", 400),
            audio_max_seconds=_env_float("CALLBACK_AUDIO_MAX_SECONDS", 30.0),
        )
