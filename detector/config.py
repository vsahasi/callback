"""Runtime configuration, read from the environment.

Deliberately dependency-free (no pydantic-settings) so the Modal image stays
small and the mock container boots fast.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field


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


def _env_str(name: str) -> str | None:
    """A stripped string, or None when unset or blank."""

    raw = os.environ.get(name, "").strip()
    return raw or None


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
    # True when neither DETECTOR_TOKEN nor CALLBACK_API_TOKEN was supplied and
    # a random token was generated. The service then refuses every caller,
    # which is the safe direction: a misnamed or empty secret must not leave
    # the detector open.
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

    # ------------------------------------------------------------- enrolment
    #
    # The reference voices voice_match compares callers against. Enrolment
    # goes through the same audio transport as call audio, so it is off
    # whenever CALLBACK_AUDIO_ENABLED is off: one switch, no second way in.
    #
    #: Ceiling on stored identities. A put past the cap is refused rather than
    #: evicting someone else's vendor; see detector/enrolment.py.
    enrolment_max_identities: int = 10_000
    #: Name of a modal.Dict to hold the prints, so they survive a deploy and
    #: are shared across replicas. Empty keeps them in process memory, where
    #: a restart loses every enrolment.
    enrolment_dict_name: str | None = None
    #: Enrolment is not inside a live call, so it is not held to the event
    #: path's 400 ms. Pulling a whole recording out of an object store needs
    #: more than that.
    enrolment_fetch_timeout_ms: int = 5000

    # --------------------------------------------------- /v1 agent contract
    #
    #: POST /verify/bank-details exists to receive a full account number, so
    #: it is unmounted unless this is switched on. The agent never calls it.
    bank_verify_enabled: bool = False
    #: The users_on_file directory (see users_on_file.example.json). The
    #: inline JSON wins over the path; it is how the Modal secret supplies it.
    #: Both hold a real phone number, so neither is ever printed: repr=False.
    users_on_file_path: str | None = None
    users_on_file_json: str | None = field(default=None, repr=False)
    #: Twilio Verify, which sends and checks the SMS code. Real mode answers
    #: 503 otp_unavailable unless all three are set.
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = field(default=None, repr=False)
    twilio_verify_service_sid: str | None = None
    #: "" (the default) means mock mode accepts only the mock code and texts
    #: nothing. "twilio" makes mock mode send and check through Twilio Verify
    #: when the three values above are set, for the recording.
    mock_sms: str = ""

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
        #
        # DETECTOR_TOKEN is the name the agent reads, so it is the primary
        # one. CALLBACK_API_TOKEN is the old name and still works. Two
        # different values would 401 every call without a word, so that is
        # refused at startup instead.
        primary = os.environ.get("DETECTOR_TOKEN", "").strip()
        legacy = os.environ.get("CALLBACK_API_TOKEN", "").strip()
        if primary and legacy and primary != legacy:
            raise ValueError("DETECTOR_TOKEN and CALLBACK_API_TOKEN differ")
        supplied = primary or legacy
        generated = not supplied
        mock_sms = os.environ.get("CALLBACK_MOCK_SMS", "").strip().lower()
        if mock_sms not in {"", "twilio"}:
            raise ValueError('CALLBACK_MOCK_SMS must be blank or "twilio"')
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
            enrolment_max_identities=_env_int(
                "CALLBACK_ENROLMENT_MAX_IDENTITIES", 10_000
            ),
            enrolment_dict_name=(
                os.environ.get("CALLBACK_ENROLMENT_DICT_NAME", "").strip() or None
            ),
            enrolment_fetch_timeout_ms=_env_int(
                "CALLBACK_ENROLMENT_FETCH_TIMEOUT_MS", 5000
            ),
            bank_verify_enabled=_env_bool("CALLBACK_BANK_VERIFY_ENABLED", False),
            users_on_file_path=_env_str("USERS_ON_FILE_PATH"),
            users_on_file_json=_env_str("USERS_ON_FILE_JSON"),
            twilio_account_sid=_env_str("TWILIO_ACCOUNT_SID"),
            twilio_auth_token=_env_str("TWILIO_AUTH_TOKEN"),
            twilio_verify_service_sid=_env_str("TWILIO_VERIFY_SERVICE_SID"),
            mock_sms=mock_sms,
        )
