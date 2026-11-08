"""The wire contract between the Callback voice agent and the detection service.

This is the only thing the two sides share: every request and response body is a model here.
Unknown fields are ignored, so either side can add a response field without breaking the other.
Change this file only together, and regenerate the schema snapshot in the same change:

    cd shared && uv run python -m callback_shared.contract --write-schema tests/snapshots/contract_schema.json

The endpoints are in the Contract tab of docs/spec/2026-11-04-notion-snapshot.md. The D-numbers in
the notes below point at the decision table in docs/superpowers/plans/2026-11-05-voice-agent.md.
"""

import argparse
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"
CONTRACT_VERSION_HEADER = "X-Contract-Version"  # sent on every request to the detection service
SCENARIO_HEADER = "X-Callback-Scenario"  # mock mode: picks a scripted demo call on POST /v1/sessions
RISK_TOPIC = "risk"  # LiveKit data topic that carries a RiskUpdate about every 2 s
AUTHORIZE_TIMEOUT_S = 3.0  # /authorize answers within this; the agent treats a timeout as deny


class Channel(StrEnum):
    PHONE = "phone"
    BROWSER = "browser"


class Role(StrEnum):
    USER = "user"
    AGENT = "agent"


class Scenario(StrEnum):
    REAL_CFO = "real_cfo"
    CLONED_VOICE = "cloned_voice"
    REPLAY = "replay"
    AI_AGENT = "ai_agent"
    HUMAN_IMPOSTOR = "human_impostor"


class SpeechEventType(StrEnum):
    AGENT_SPEECH_START = "agent_speech_start"
    AGENT_SPEECH_END = "agent_speech_end"
    USER_SPEECH_START = "user_speech_start"


class ChallengeKind(StrEnum):
    DIGITS = "digits"
    PHRASE = "phrase"
    LIVENESS = "liveness"
    ENROLL = "enroll"


class ChallengeState(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"


class DecisionKind(StrEnum):
    ALLOW = "allow"
    STEP_UP = "step_up"
    DENY = "deny"


class Factor(StrEnum):
    VOICE_CHALLENGE = "voice_challenge"
    SMS_OTP = "sms_otp"
    CALLBACK_ON_FILE = "callback_on_file"
    PROBE = "probe"  # one unexpected off-script question


class OtpState(StrEnum):
    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"
    NOT_APPLICABLE = "n/a"


class CallbackResult(StrEnum):
    CONFIRMED = "confirmed"
    DENIED = "denied"
    NO_ANSWER = "no_answer"


# D1: the actions the agent sends to /authorize. ACTION_TIER says what each one needs.
class Action(StrEnum):
    HOLD_VENDOR = "hold_vendor"
    HOLD_BILL = "hold_bill"
    LIST_VENDORS = "list_vendors"
    LIST_VENDOR_ACCOUNTS = "list_vendor_accounts"
    LIST_BILLS = "list_bills"
    UPDATE_VENDOR_BANK_ACCOUNT = "update_vendor_bank_account"
    RELEASE_VENDOR_HOLD = "release_vendor_hold"
    RELEASE_BILL_HOLD = "release_bill_hold"


# The spec's action tiers: 0 protective (always allowed), 1 read, 2 money moves.
ACTION_TIER: dict[Action, int] = {
    Action.HOLD_VENDOR: 0,
    Action.HOLD_BILL: 0,
    Action.LIST_VENDORS: 1,
    Action.LIST_VENDOR_ACCOUNTS: 1,
    Action.LIST_BILLS: 1,
    Action.UPDATE_VENDOR_BANK_ACCOUNT: 2,
    Action.RELEASE_VENDOR_HOLD: 2,
    Action.RELEASE_BILL_HOLD: 2,
}


# Notes on models and enums are comments, not docstrings: pydantic copies a docstring into the JSON
# schema, so rewording a note would break the snapshot test.
class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SessionCreate(_Model):
    room: str
    caller_identity: str
    claimed_user_id: str
    channel: Channel


class SessionCreated(_Model):
    session_id: str


# D4: after the callback connects, the caller is a new participant in the same session.
class CallerUpdate(_Model):
    caller_identity: str
    channel: Channel


# All times in turns, speech events and challenges are Unix epoch ms (STT stream start + Deepgram
# word offset).
class Word(_Model):
    w: str
    start_ms: int
    end_ms: int


# D7: agent turns send words=[]. On a user turn, interrupted means the caller started while the
# agent was speaking. On an agent turn, it means the agent was cut off.
class Turn(_Model):
    role: Role
    text: str
    start_ms: int
    end_ms: int
    words: list[Word] = Field(default_factory=list)
    interrupted: bool = False


class SpeechEvent(_Model):
    type: SpeechEventType
    ts_ms: int


class ChallengeCreate(_Model):
    kind: ChallengeKind


class ChallengeIssued(_Model):
    challenge_id: str
    say: str  # the agent speaks this word for word
    expires_at: int  # epoch ms (D7)


class ChallengeStatus(_Model):
    status: ChallengeState
    phrase_match: bool | None = None
    speaker_score: float | None = None
    reason: str | None = None  # D5


class OtpSent(_Model):
    sent_to_last_four: str  # always the number on file, never one the caller gave


class OtpCheck(_Model):
    code: str


class OtpCheckResult(_Model):
    verified: bool


# D2: what the agent tells the detector's rubric about the pending Ramp change.
# Never full account numbers.
class RampContext(_Model):
    pending_action: str
    vendor_name: str | None = None
    vendor_bank_last4: str | None = None


class AuthorizeRequest(_Model):
    action: Action
    params: dict[str, Any]
    idempotency_key: str
    context: RampContext | None = None  # D2


class VoiceprintSignal(_Model):
    score: float
    threshold: float
    passed: bool
    phrase_match: bool | None = None


class DeepfakeSignal(_Model):
    spoof_prob: float
    windows: int


class TextFlag(_Model):
    type: str
    evidence: str


class TextSignal(_Model):
    risk: float
    flags: list[TextFlag] = Field(default_factory=list)


class Signals(_Model):
    voiceprint: VoiceprintSignal | None = None
    deepfake: DeepfakeSignal | None = None
    text: TextSignal | None = None
    agent_likelihood: float | None = None
    otp: OtpState = OtpState.NOT_APPLICABLE


class Decision(_Model):
    session_id: str
    action: Action
    decision: DecisionKind
    risk_score: float = 0.0
    required_factors: list[Factor] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)
    reasons: list[str] = Field(default_factory=list)
    agent_say: str = ""  # D8: the agent owns every spoken line and only logs this
    ttl_s: int = 0


class FactorReport(_Model):
    kind: Literal["callback"] = "callback"
    result: CallbackResult


# Published by the detection service on the RISK_TOPIC data topic.
class RiskUpdate(_Model):
    session_id: str
    spoof_prob: float | None = None
    text_risk: float | None = None
    flags: list[str] = Field(default_factory=list)
    agent_likelihood: float | None = None


# D3: the detection service serves the user directory.
class UserSummary(_Model):
    user_id: str
    display_name: str
    role: str
    phone_last_four: str


class UserOnFile(UserSummary):
    phone_e164: str


# D5: enrollment needs a consent record and a completion step.
class Consent(_Model):
    version: str
    verbatim: str
    ts_ms: int


class EnrollmentStart(_Model):
    session_id: str
    user_id: str
    channel: Channel
    consent: Consent


class EnrollmentStarted(_Model):
    enrollment_id: str
    prompts_required: int = 7


class EnrollmentResult(_Model):
    status: Literal["enrolled", "failed"]
    prompts_passed: int


def _models() -> list[type[_Model]]:
    """Every model in this module, so a new one can't be left out of the schema."""
    return [
        m for m in globals().values()
        if isinstance(m, type) and issubclass(m, _Model) and m is not _Model
    ]


def contract_json_schema() -> dict[str, Any]:
    """The contract version plus the JSON schema of every model, keyed by class name.

    tests/snapshots/contract_schema.json holds a copy. The snapshot test fails whenever one side
    changes the contract alone.
    """
    return {
        "version": CONTRACT_VERSION,
        "models": {m.__name__: m.model_json_schema() for m in _models()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m callback_shared.contract",
        description="Write the contract's JSON schema, the file the snapshot test compares against.",
    )
    parser.add_argument("--write-schema", type=Path, required=True, metavar="PATH")
    args = parser.parse_args()
    args.write_schema.parent.mkdir(parents=True, exist_ok=True)
    args.write_schema.write_text(json.dumps(contract_json_schema(), indent=2, sort_keys=True) + "\n")
