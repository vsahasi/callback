"""A read-only view of the /v1 session registry, for the review surface.

Served at `GET /review/api/v1/sessions` and `GET /review/api/v1/sessions/{id}`
by detector/review.py, behind the same token guard as the other review
routes. This is outside the /v1 wire contract: the agent never calls it, and
nothing in callback_shared describes it.

What it never returns: turn text or words, a challenge's `say` or expected
answer, an OTP code, a name, a full phone number, a bank number, the Ramp
context, a raw room or a raw caller identity. The number on file shows only
as its last four digits, which the agent already speaks to the caller.

Reading never touches a session: it does not count as a /v1 call for the
idle limit. The registry's expiry sweep still runs, as on every access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from callback_shared.contract import ACTION_TIER, Decision
from pydantic import BaseModel, ConfigDict, Field

from detector.v1.challenges import ChallengeRecord
from detector.v1.service import V1Service
from detector.v1.sessions import V1Session

MAX_LISTED_SESSIONS = 50


def iso(epoch_s: float | None) -> str | None:
    """Epoch seconds as ISO 8601 UTC, or None."""

    if epoch_s is None:
        return None
    return datetime.fromtimestamp(epoch_s, UTC).isoformat()


class _View(BaseModel):
    model_config = ConfigDict(extra="forbid")


class V1SessionSummary(_View):
    session_id: str
    created_at: str
    updated_at: str
    scenario: str | None
    claimed_user_id: str | None
    channel: str | None
    last_decision: str | None
    last_action: str | None
    last_reasons: list[str] = Field(default_factory=list)
    locked: bool
    ended: bool


class V1SessionList(_View):
    sessions: list[V1SessionSummary]


class V1DecisionEntry(_View):
    at: str
    action: str
    tier: int
    decision: str
    reasons: list[str]
    required_factors: list[str]
    ttl_s: int
    idempotency_key: str
    # The Decision's own signals object, as it went out on the wire.
    signals: dict[str, Any]


class V1ChallengeEntry(_View):
    challenge_id: str
    kind: str
    status: str
    phrase_match: bool | None
    speaker_score: float | None
    reason: str | None
    issued_at: str
    resolved_at: str | None


class V1OtpView(_View):
    sent: bool
    sent_to_last_four: str | None
    attempts: int
    verified: bool | None


class V1FactorEntry(_View):
    kind: str
    result: str
    at: str


class V1EventEntry(_View):
    type: str
    at: str


class V1SessionDetail(V1SessionSummary):
    decisions: list[V1DecisionEntry]
    challenges: list[V1ChallengeEntry]
    otp: V1OtpView
    factors: list[V1FactorEntry]
    events: list[V1EventEntry]


def _updated_at(session: V1Session) -> float:
    return max(
        session.last_call_at,
        session.ended_at or 0.0,
        session.history[-1].at if session.history else 0.0,
    )


def _last_gated(session: V1Session) -> Decision | None:
    """The latest tier 1 or 2 answer.

    Tier 0 holds are skipped: a denied call ends on a background hold_vendor
    allow, and the summary must still show the deny.
    """

    for record in reversed(session.history):
        if ACTION_TIER[record.decision.action] != 0:
            return record.decision
    return None


def summary(session: V1Session) -> V1SessionSummary:
    last = _last_gated(session)
    return V1SessionSummary(
        session_id=session.session_id,
        created_at=iso(session.created_at) or "",
        updated_at=iso(_updated_at(session)) or "",
        scenario=session.scenario.value if session.scenario else None,
        claimed_user_id=session.claimed_user_id,
        channel=session.channel.value if session.channel else None,
        last_decision=last.decision.value if last else None,
        last_action=last.action.value if last else None,
        last_reasons=list(last.reasons) if last else [],
        locked=session.locked_code is not None,
        ended=session.ended,
    )


def list_sessions(
    service: V1Service, limit: int = MAX_LISTED_SESSIONS
) -> V1SessionList:
    """Newest first, at most `limit` (and never more than 50)."""

    ordered = sorted(
        service.registry.sessions(), key=lambda s: s.created_at, reverse=True
    )
    cap = max(0, min(limit, MAX_LISTED_SESSIONS))
    return V1SessionList(sessions=[summary(s) for s in ordered[:cap]])


def _challenge(record: ChallengeRecord, now: float) -> V1ChallengeEntry:
    # Never record.say or record.expected: those are the answer.
    state = record.resolve(now)
    return V1ChallengeEntry(
        challenge_id=record.challenge_id,
        kind=record.kind.value,
        status=state.status.value,
        phrase_match=state.phrase_match,
        speaker_score=state.speaker_score,
        reason=state.reason,
        issued_at=iso(record.issued_at) or "",
        resolved_at=iso(state.resolved_at),
    )


def detail(service: V1Service, session_id: str) -> V1SessionDetail | None:
    """The full view of one session, or None when it is unknown or evicted."""

    session = next(
        (s for s in service.registry.sessions() if s.session_id == session_id), None
    )
    if session is None:
        return None
    now = service.registry.now()
    otp = session.otp
    user = service.directory.get(session.claimed_user_id)
    return V1SessionDetail(
        **summary(session).model_dump(),
        decisions=[
            V1DecisionEntry(
                at=iso(record.at) or "",
                action=record.decision.action.value,
                tier=ACTION_TIER[record.decision.action],
                decision=record.decision.decision.value,
                reasons=list(record.decision.reasons),
                required_factors=[f.value for f in record.decision.required_factors],
                ttl_s=record.decision.ttl_s,
                idempotency_key=record.idempotency_key,
                signals=record.decision.signals.model_dump(mode="json"),
            )
            for record in session.history
        ],
        challenges=[_challenge(c, now) for c in session.challenges.values()],
        otp=V1OtpView(
            sent=otp.sent,
            # The last four only, and only once a code went out.
            sent_to_last_four=(user.phone_last_four if otp.sent and user else None),
            attempts=otp.checks,
            verified=otp.verified if otp.checks else None,
        ),
        factors=[
            V1FactorEntry(kind="callback", result=f.result.value, at=iso(f.at) or "")
            for f in session.factor_history
        ],
        # Types and times only. /end drops events (ruling 2.6), so an ended
        # session lists none.
        events=[
            V1EventEntry(type=e.type.value, at=iso(e.ts_ms / 1000) or "")
            for e in session.events
        ],
    )
