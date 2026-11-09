"""The detector's internal model of its /events dev API and of PR 3.

This is NOT the agent contract. The agent and the detector talk over
callback_shared 1.2.0 (shared/src/callback_shared/contract.py), which the
detector serves under /v1 (detector/v1). The final contract ruling of
2026-11-07 kept this file where it is, unrenamed and unchanged in shape, as
the model of the root /events, /factors and /enrol/voice routes, Eric's offline
harness and the client kit in shared/client.py.

It was derived from the Day-1 build brief: the /events and /factors
endpoints, the X-Callback-Scenario header, CONTRACT_VERSION, claimed_user_id,
the five scripted scenarios and the OTP policy. CONTRACT_VERSION stays at
0.1.0-provisional and schemas/contract.schema.json stays byte-identical.

Snapshot drift is caught by tests/contract/test_schema_snapshot.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Bump whenever any model in this module changes shape.
CONTRACT_VERSION = "0.1.0-provisional"

# Headers.
SCENARIO_HEADER = "X-Callback-Scenario"
IDEMPOTENCY_HEADER = "Idempotency-Key"

# Client-side policy. The agent must fail closed: if the detector does not
# answer inside this budget, the agent treats the call as DENY rather than
# waiting or defaulting open.
CLIENT_TIMEOUT_MS = 800
FAIL_CLOSED_DECISION = "deny"

# Mock-mode OTP. Real mode issues a random code out of band.
MOCK_OTP_CODE = "424242"


class Scenario(StrEnum):
    """The five scripted mock scenarios, selectable per request."""

    REAL_CFO = "real_cfo"
    CLONED_VOICE = "cloned_voice"
    REPLAY = "replay"
    AI_AGENT = "ai_agent"
    HUMAN_IMPOSTOR = "human_impostor"


class SessionState(StrEnum):
    """Detector session lifecycle. Terminal states are allowed, denied, expired."""

    CREATED = "created"
    LISTENING = "listening"
    CHALLENGING = "challenging"
    ALLOWED = "allowed"
    DENIED = "denied"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in {SessionState.ALLOWED, SessionState.DENIED, SessionState.EXPIRED}


class Decision(StrEnum):
    """What the agent should do right now."""

    PENDING = "pending"
    CHALLENGE = "challenge"
    ALLOW = "allow"
    DENY = "deny"


class EventType(StrEnum):
    """Events the agent pushes to POST /events."""

    SESSION_START = "session_start"
    CALLER_CLAIM = "caller_claim"
    SPEECH_SEGMENT = "speech_segment"
    OTP_REQUEST = "otp_request"
    OTP_SUBMIT = "otp_submit"
    SESSION_END = "session_end"


class NextAction(StrEnum):
    """Concrete instruction, so the agent never has to infer one."""

    CONTINUE_LISTENING = "continue_listening"
    REQUEST_OTP = "request_otp"
    COMPLETE_ALLOW = "complete_allow"
    COMPLETE_DENY = "complete_deny"


class FactorName(StrEnum):
    """Independent signals combined into the risk score."""

    VOICE_MATCH = "voice_match"
    SYNTHETIC_SPEECH = "synthetic_speech"
    REPLAY = "replay"
    AGENT_BEHAVIOUR = "agent_behaviour"
    LIVENESS = "liveness"
    OTP = "otp"


class FactorVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


class AudioEncoding(StrEnum):
    PCM_S16LE = "pcm_s16le"
    OPUS = "opus"
    FLAC = "flac"


Score = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Reject unknown fields, so contract drift fails loudly instead of silently."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class AudioRef(StrictModel):
    """A span of call audio. Audio is referenced, never inlined in JSON."""

    track_id: str = Field(description="LiveKit track identifier.")
    start_ms: int = Field(ge=0, description="Offset from session start.")
    end_ms: int = Field(ge=0, description="Exclusive end offset from session start.")
    sample_rate_hz: int = Field(ge=8000, le=48000, default=16000)
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    uri: str | None = Field(
        default=None,
        description="Optional out-of-band location (object store) for the span.",
    )


class Factor(StrictModel):
    """One signal and its contribution.

    score 1.0 means fully consistent with the claimed identity, 0.0 means fully
    inconsistent.
    """

    name: FactorName
    score: Score
    weight: Score = Field(description="Relative weight in the blended risk score.")
    verdict: FactorVerdict
    detail: str | None = Field(default=None, max_length=500)


class OtpChallenge(StrictModel):
    """Step-up state. Never carries the expected code."""

    code_length: int = Field(default=6, ge=4, le=10)
    attempts_used: int = Field(ge=0)
    attempts_remaining: int = Field(ge=0)
    expires_at: datetime


class EventRequest(StrictModel):
    """POST /events body."""

    session_id: str = Field(min_length=8, max_length=128)
    event: EventType
    sequence: int = Field(
        ge=0, description="Monotonic per session. Replays are idempotent."
    )
    occurred_at: datetime
    claimed_user_id: str | None = Field(
        default=None,
        max_length=128,
        description="Identity the caller claims. Also selects a mock scenario when "
        "the scenario header is absent.",
    )
    audio: AudioRef | None = None
    otp_code: str | None = Field(default=None, min_length=4, max_length=10)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionEnvelope(StrictModel):
    """Shared decision payload, identical in /events and /factors responses."""

    contract_version: str = CONTRACT_VERSION
    session_id: str
    state: SessionState
    decision: Decision
    risk_score: Score = Field(description="0.0 safest, 1.0 riskiest.")
    confidence: Score
    factors: list[Factor]
    next_action: NextAction
    reason: str | None = Field(default=None, max_length=500)
    scenario: Scenario | None = Field(
        default=None, description="Populated in mock mode only."
    )


class EventResponse(DecisionEnvelope):
    """POST /events 200 response."""

    sequence: int = Field(ge=0, description="Echoes the request sequence.")
    otp: OtpChallenge | None = None
    latency_ms: int = Field(ge=0, description="Server-side handling time.")
    idempotent_replay: bool = Field(
        default=False, description="True when a cached response was returned."
    )


class FactorsResponse(DecisionEnvelope):
    """GET /factors/{session_id} 200 response."""

    created_at: datetime
    updated_at: datetime
    events_seen: int = Field(ge=0)
    otp: OtpChallenge | None = None


class ErrorCode(StrEnum):
    UNAUTHORIZED = "unauthorized"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_TERMINAL = "session_terminal"
    SEQUENCE_CONFLICT = "sequence_conflict"
    OTP_EXHAUSTED = "otp_exhausted"
    OTP_EXPIRED = "otp_expired"
    INVALID_SCENARIO = "invalid_scenario"
    VALIDATION_ERROR = "validation_error"


class ErrorBody(StrictModel):
    code: ErrorCode
    message: str
    detail: str | None = None


class ErrorResponse(StrictModel):
    """Every non-2xx response body."""

    contract_version: str = CONTRACT_VERSION
    error: ErrorBody


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    contract_version: str = CONTRACT_VERSION
    mock_mode: bool


# Models frozen by the schema snapshot test, in a stable order.
SCHEMA_MODELS: tuple[type[BaseModel], ...] = (
    AudioRef,
    Factor,
    OtpChallenge,
    EventRequest,
    EventResponse,
    FactorsResponse,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
)


def schema_bundle() -> dict[str, Any]:
    """Deterministic JSON-schema bundle of the whole contract.

    Enum members are listed explicitly so that adding or renaming one is caught
    even if no model field changes.
    """

    enums: dict[str, list[str]] = {
        enum_cls.__name__: [member.value for member in enum_cls]
        for enum_cls in (
            Scenario,
            SessionState,
            Decision,
            EventType,
            NextAction,
            FactorName,
            FactorVerdict,
            AudioEncoding,
            ErrorCode,
        )
    }

    return {
        "contract_version": CONTRACT_VERSION,
        "headers": {
            "scenario": SCENARIO_HEADER,
            "idempotency": IDEMPOTENCY_HEADER,
        },
        "client_policy": {
            "timeout_ms": CLIENT_TIMEOUT_MS,
            "on_timeout": FAIL_CLOSED_DECISION,
        },
        "enums": enums,
        "models": {
            model.__name__: model.model_json_schema(mode="serialization")
            for model in SCHEMA_MODELS
        },
    }
