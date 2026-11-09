"""The human review surface: evidence bundles, the dashboard, and the wiring.

The project README says a person approves before any money moves. This module
is where that person looks. It turns what the detector knows about one phone
call into a record a reviewer can act on now and an auditor can read in a year:
the decision, why it was reached, every factor with its number and its
human-readable detail, the full request timeline, the one-time-code attempt
history, and a place for the call recording and transcript to be attached.

Wiring
------
Add one call at the end of `create_app`, after the engine exists::

    from detector.review import mount as mount_review

    ...
    app.state.observability = mount_review(app, engine=engine, settings=resolved)
    return app

That single call installs the ASGI middleware, the `/metrics` route and the
`/review` routes, and registers the 401 handler those routes need. It touches
nothing already on the app.

Routes added, and the auth on each
----------------------------------
Every one of these requires a bearer token. There is no unauthenticated route
in this module, because every one of them exposes fraud-decision detail.

    GET  /metrics                                   bearer
    GET  /review                                    bearer or basic
    GET  /review/api/sessions                       bearer or basic
    GET  /review/api/sessions/{id}                  bearer or basic
    GET  /review/api/sessions/{id}/text             bearer or basic
    POST /review/api/sessions/{id}/media            bearer or basic

`/metrics` takes a bearer token only: it is scraped by a machine, and the
labels carry the decision mix, so an open scrape both publishes the deny rate
and acts as an oracle for an attacker who wants to know whether their call was
caught. `ObservabilityConfig(metrics_require_auth=False)` exists for a private
network and has to be asked for by name.

Two tokens, not one. `/metrics` takes the service token alone. The `/review`
routes take the service token or, when `CALLBACK_REVIEW_TOKEN` is set, that
second read-only token as well. The dashboard is served from Vercel and read
in a browser, so without the second one it has to hold the credential that can
authorize a payment and read a full phone number from `/v1/users/{id}`. The
review token reaches nothing else: `/v1` and `/events` check the service token
directly and never consult this module's guard, and `/metrics` is given its
own guard that does not carry the review token, because its labels publish the
decision mix and act as an oracle.

The `/review` routes additionally accept HTTP Basic carrying either token.
That is the whole reason Basic is here: a human opens the dashboard by typing a
URL, a plain navigation cannot set an `Authorization` header, and a `?token=`
query string would write the service token into every access log, proxy log and
Referer header. With Basic the browser prompts once, the token travels in a
header, and the page's own `fetch` calls inherit the credentials, so the live
refresh needs no token handling in JavaScript at all. Any username is accepted;
the token goes in either field.

An unknown session is a 404 carrying the contract error body, produced here
rather than relying on an exception handler elsewhere.

Configuration
-------------
Read from the environment by `ObservabilityConfig.from_env()`, so nothing has
to be added to `detector.config.Settings`:

    CALLBACK_LOG_ENABLED                    default true
    CALLBACK_METRICS_ENABLED                default true
    CALLBACK_METRICS_REQUIRE_AUTH           default true
    CALLBACK_REVIEW_JOURNAL_ENABLED         default true
    CALLBACK_REVIEW_JOURNAL_MAX_SESSIONS    default 256
    CALLBACK_REVIEW_JOURNAL_MAX_ENTRIES     default 200
    CALLBACK_SERVICE_NAME                   default callback-detector

Secrets
-------
No bundle, log line, metric label or dashboard pixel carries the one-time code,
the bearer token or a bank detail. The code never leaves `OtpState`; what
appears here is its length, the attempt count and the per-attempt outcome. See
the redaction notes in `detector.observability`, and `redaction_notes()` below,
which ships the same statement inside every bundle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from detector import scoring
from detector.config import Settings
from detector.errors import SessionNotFound
from detector.observability import (
    MediaAttachment,
    Observability,
    ObservabilityConfig,
    ObservabilityMiddleware,
    Redactor,
    ReviewUnauthorized,
    SessionJournal,
    SessionRecord,
    TokenAuth,
    metrics_router,
)
from detector.state import DetectorEngine, Session
from detector.v1 import review_view as v1_review
from detector.v1.review_view import V1SessionDetail, V1SessionList
from detector.v1.service import V1Service
from shared.contract import (
    CONTRACT_VERSION,
    MOCK_OTP_CODE,
    Decision,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    FactorsResponse,
)

BUNDLE_VERSION = "1"

DEFAULT_REVIEW_PREFIX = "/review"


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------- models


class ReviewModel(BaseModel):
    """Local base. These models are ours; the wire contract is frozen elsewhere."""

    model_config = ConfigDict(extra="forbid")


class EvidenceFactor(ReviewModel):
    """One signal, with everything a human needs to weigh it."""

    name: str
    verdict: str
    score: float = Field(description="1.0 fully consistent with the claim, 0.0 not.")
    risk: float = Field(description="1 - score. What this factor argues against.")
    weight: float
    risk_share: float = Field(
        description="Fraction of the blended risk this factor accounts for, 0 to 1."
    )
    detail: str | None = None
    fatal: bool = Field(
        default=False,
        description="Tripped the unambiguous-attack rule, which no code rescues.",
    )
    fatal_candidate: bool = Field(
        default=False, description="Subject to the unambiguous-attack rule at all."
    )


class EvidenceOtpAttempt(ReviewModel):
    """One code submission. The code itself is not here and never was."""

    attempt: int
    at: datetime
    outcome: str
    detail: str | None = None


class EvidenceOtp(ReviewModel):
    """Step-up state and history."""

    challenged: bool
    code_withheld: Literal[True] = Field(
        default=True,
        description="Always true. The expected code is not recorded anywhere.",
    )
    code_length: int | None = None
    attempts_used: int = 0
    attempts_remaining: int | None = None
    max_attempts: int
    expires_at: datetime | None = None
    resolved: bool | None = Field(
        default=None,
        description="True if a correct code was supplied, False if exhausted.",
    )
    attempts: list[EvidenceOtpAttempt] = Field(default_factory=list)


class EvidenceTimelineEntry(ReviewModel):
    """One request against this session, in the order it arrived."""

    at: datetime
    request_id: str
    method: str
    endpoint: str
    status: int
    outcome: str
    latency_ms: int
    call_event: str | None = None
    sequence: int | None = None
    occurred_at: datetime | None = None
    decision: str | None = None
    state: str | None = None
    risk_score: float | None = None
    next_action: str | None = None
    reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    otp_submitted: bool = False
    otp_outcome: str | None = None
    idempotent_replay: bool = False
    fail_closed_cause: str | None = None
    factor_verdicts: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceMedia(ReviewModel):
    """Hooks for the call recording and transcript the agent attaches later."""

    status: Literal["attached", "awaiting_agent"]
    attach_with: str = Field(
        description="The route the agent POSTs to once the call ends."
    )
    recording_uri: str | None = None
    recording_duration_ms: int | None = None
    transcript_uri: str | None = None
    transcript_text: str | None = None
    note: str | None = None
    attached_at: datetime | None = None


class EvidenceThresholds(ReviewModel):
    """The policy the decision was measured against, so it can be re-checked."""

    allow_below: float
    deny_at_or_above: float
    fatal_threshold: float
    fatal_factors: list[str]
    otp_weight: float
    otp_max_attempts: int
    otp_ttl_seconds: int
    session_ttl_seconds: int
    speech_segments_before_decision: int


class EvidenceCounts(ReviewModel):
    requests: int = 0
    rejected_requests: int = 0
    events_seen: int | None = None
    speech_segments: int | None = None
    otp_attempts: int = 0
    timeline_truncated: bool = False


class HumanReview(ReviewModel):
    """What the reviewer is actually being asked."""

    question: str
    recommendation: str
    guidance: str
    caveats: list[str] = Field(default_factory=list)


class EvidenceBundle(ReviewModel):
    """A complete, self-contained record of one verification call."""

    bundle_version: str = BUNDLE_VERSION
    contract_version: str = CONTRACT_VERSION
    generated_at: datetime
    session_id: str
    state_source: Literal["live_session", "journal_only"]
    mock_mode: bool

    headline: str
    decision: str
    state: str
    next_action: str | None = None
    reason: str | None = None
    why: list[str] = Field(default_factory=list)

    risk_score: float
    confidence: float
    scenario: str | None = None
    scenario_source: str | None = None
    claimed_user_id: str | None = None

    thresholds: EvidenceThresholds
    factors: list[EvidenceFactor] = Field(default_factory=list)
    otp: EvidenceOtp | None = None
    timeline: list[EvidenceTimelineEntry] = Field(default_factory=list)
    counts: EvidenceCounts
    media: EvidenceMedia
    human_review: HumanReview

    created_at: datetime | None = None
    updated_at: datetime | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    redaction: list[str] = Field(default_factory=list)


class SessionSummary(ReviewModel):
    """One row in the dashboard list."""

    session_id: str
    decision: str
    state: str
    headline: str
    reason: str | None = None
    risk_score: float | None = None
    confidence: float | None = None
    scenario: str | None = None
    claimed_user_id: str | None = None
    first_seen: datetime
    last_seen: datetime
    requests: int = 0
    rejected_requests: int = 0
    otp_attempts: int = 0
    has_media: bool = False
    live: bool = Field(
        default=False, description="Whether the session store still holds it."
    )


class SessionList(ReviewModel):
    generated_at: datetime
    count: int
    journal_sessions: int
    active_sessions: int
    journal_enabled: bool
    dropped_sessions: int = Field(
        default=0, description="Calls evicted from the journal by its size bound."
    )
    sessions: list[SessionSummary] = Field(default_factory=list)


class MediaAttachRequest(ReviewModel):
    """What the agent POSTs once it has the recording."""

    recording_uri: str | None = Field(default=None, max_length=2000)
    recording_duration_ms: int | None = Field(default=None, ge=0)
    transcript_uri: str | None = Field(default=None, max_length=2000)
    transcript_text: str | None = Field(default=None, max_length=200_000)
    note: str | None = Field(default=None, max_length=1000)


class MediaAttachResponse(ReviewModel):
    session_id: str
    media: EvidenceMedia


# -------------------------------------------------------------- the reasoning


def redaction_notes() -> list[str]:
    """What this record deliberately does not contain, and why.

    Shipped inside every bundle so the omission is on the record rather than
    looking like a gap.
    """

    return [
        (
            "The one-time code is never stored, logged or rendered. This record "
            "carries its length, the number of attempts and the outcome of each."
        ),
        (
            "The service bearer token is scrubbed from every string recorded by "
            "the observability layer, including fields nobody expected it in."
        ),
        (
            "Bank details are not part of this record. Values that look like an "
            "IBAN, or runs of nine or more digits, are masked when captured."
        ),
        (
            "This is a voice-verification record. The bank-detail change itself "
            "is checked separately and is not evidenced here."
        ),
    ]


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def build_factors(
    factors: list[Any],
) -> list[EvidenceFactor]:
    """Turn contract `Factor` models into reviewer-facing factors.

    `risk_share` is the honest version of "how much did this matter": the
    factor's weighted risk as a fraction of the total weighted risk, computed
    the same way `detector.scoring.blend_risk` computes the blend.
    """

    scored = [f for f in factors if str(_verdict_of(f)) != "not_run"]
    total = sum(float(f.weight) for f in scored) or 0.0
    weighted_total = sum(float(f.weight) * (1.0 - float(f.score)) for f in scored)

    out: list[EvidenceFactor] = []
    for factor in factors:
        name = _name_of(factor)
        score = float(factor.score)
        weight = float(factor.weight)
        risk = 1.0 - score
        share = 0.0
        if weighted_total > 0 and total > 0:
            share = (weight * risk) / weighted_total
        candidate = name in {f.value for f in scoring.FATAL_FACTORS}
        out.append(
            EvidenceFactor(
                name=name,
                verdict=str(_verdict_of(factor)),
                score=score,
                risk=risk,
                weight=weight,
                risk_share=min(1.0, max(0.0, share)),
                detail=factor.detail,
                fatal=candidate and score <= scoring.FATAL_THRESHOLD,
                fatal_candidate=candidate,
            )
        )
    out.sort(key=lambda f: (not f.fatal, -f.risk_share, f.name))
    return out


def _name_of(factor: Any) -> str:
    name = factor.name
    return name.value if hasattr(name, "value") else str(name)


def _verdict_of(factor: Any) -> str:
    verdict = factor.verdict
    return verdict.value if hasattr(verdict, "value") else str(verdict)


def headline_for(decision: str, reason: str | None) -> str:
    """One line a tired human reads correctly on the first pass."""

    verdict = decision.upper()
    if reason:
        return f"{verdict}. {reason}"
    return f"{verdict}. No reason was recorded."


def human_review_for(
    decision: str, state: str, factors: list[EvidenceFactor]
) -> HumanReview:
    """The question the reviewer is answering, and what the detector advises."""

    caveats = [
        (
            "The detector verified a voice, not a bank account. Confirm the new "
            "bank details against a source the caller does not control."
        ),
        (
            "This decision is advice. Releasing the payment is the reviewer's "
            "call and is not made here."
        ),
    ]
    if any(f.fatal for f in factors):
        caveats.insert(
            0,
            "A factor tripped the unambiguous-attack rule. A correct one-time "
            "code does not clear that, by policy.",
        )

    if decision == Decision.ALLOW.value and state == "allowed":
        return HumanReview(
            question="Release the held payment to the new bank details?",
            recommendation="consistent_with_release",
            guidance=(
                "The caller verified. Nothing in the voice analysis argues "
                "against release. Check the bank details themselves before "
                "releasing."
            ),
            caveats=caveats,
        )
    if decision == Decision.DENY.value:
        return HumanReview(
            question="Release the held payment to the new bank details?",
            recommendation="do_not_release",
            guidance=(
                "The detector refused this caller. Do not release. Reach the "
                "vendor on a number you already held before the change request "
                "and confirm by other means."
            ),
            caveats=caveats,
        )
    if decision == Decision.CHALLENGE.value:
        return HumanReview(
            question="Release the held payment to the new bank details?",
            recommendation="hold_pending_step_up",
            guidance=(
                "The voice analysis was not conclusive and a one-time code was "
                "required. Hold the payment until the code is resolved."
            ),
            caveats=caveats,
        )
    return HumanReview(
        question="Release the held payment to the new bank details?",
        recommendation="hold_call_unfinished",
        guidance=(
            "This call has not reached a decision. Hold the payment. A session "
            "that never finishes is failed closed when its time to live runs "
            "out."
        ),
        caveats=caveats,
    )


def explain(
    *,
    decision: str,
    state: str,
    risk_score: float,
    confidence: float,
    factors: list[EvidenceFactor],
    otp: EvidenceOtp | None,
    scenario: str | None,
    scenario_source: str | None,
    timeline: list[EvidenceTimelineEntry],
    settings: Settings,
    mock_mode: bool,
) -> list[str]:
    """The ordered, human-readable reasoning behind one decision.

    Deliberately written as prose a non-engineer can read out loud in a
    meeting, because that is what happens to it.
    """

    lines: list[str] = []

    if not factors:
        lines.append(
            "No factor analysis is on record for this session, so nothing here "
            "verifies the caller. Treat that as evidence against release, not "
            "as a neutral result."
        )

    fatal = next((f for f in factors if f.fatal), None)
    if fatal is not None:
        lines.append(
            f"{fatal.name} scored {fatal.score:.2f}, at or below the "
            f"{scoring.FATAL_THRESHOLD:.2f} fatal threshold. That is treated as "
            "unambiguous attack evidence: the call is refused whatever the "
            "blended score says, and a correct one-time code does not rescue it."
        )

    contributors = [
        f
        for f in factors
        if f.risk_share > 0.05 and f.verdict in {"fail", "inconclusive"}
    ][:3]
    for factor in contributors:
        detail = f" {factor.detail}" if factor.detail else ""
        lines.append(
            f"{factor.name} ({factor.verdict}, score {factor.score:.2f}) "
            f"accounts for {_pct(factor.risk_share)} of the blended risk."
            f"{detail}"
        )

    supporting = [f for f in factors if f.verdict == "pass" and f.risk_share <= 0.05]
    if supporting and decision == Decision.ALLOW.value:
        names = ", ".join(f.name for f in supporting[:4])
        lines.append(f"Supporting signals passed with little residual risk: {names}.")

    if risk_score < scoring.ALLOW_BELOW:
        lines.append(
            f"Blended risk {risk_score:.3f} is below the allow threshold "
            f"{scoring.ALLOW_BELOW:.2f}."
        )
    elif risk_score >= scoring.DENY_AT_OR_ABOVE:
        lines.append(
            f"Blended risk {risk_score:.3f} is at or above the deny threshold "
            f"{scoring.DENY_AT_OR_ABOVE:.2f}."
        )
    else:
        lines.append(
            f"Blended risk {risk_score:.3f} sits between the allow threshold "
            f"{scoring.ALLOW_BELOW:.2f} and the deny threshold "
            f"{scoring.DENY_AT_OR_ABOVE:.2f}, which is the band that calls for "
            "a one-time-code step-up."
        )

    lines.append(f"Detector confidence in this reading: {confidence:.2f}.")

    if otp is not None and otp.challenged:
        remaining = (
            otp.attempts_remaining
            if otp.attempts_remaining is not None
            else max(0, otp.max_attempts - otp.attempts_used)
        )
        resolution = "still open"
        if otp.resolved is True:
            resolution = "resolved by a correct code"
        elif otp.resolved is False:
            resolution = "exhausted without a correct code"
        lines.append(
            f"One-time-code step-up: {otp.attempts_used} of {otp.max_attempts} "
            f"attempts used, {remaining} remaining, {resolution}. The code "
            "itself is not recorded."
        )
        if otp.resolved is True and decision == Decision.DENY.value:
            lines.append(
                "A correct code was supplied and the call was still refused. "
                "Policy does not let a code override the voice evidence."
            )

    if state == "expired":
        lines.append(
            f"The session outlived its {settings.session_ttl_seconds} second "
            "time to live and was failed closed rather than left open."
        )

    causes: list[str] = []
    for entry in timeline:
        if entry.fail_closed_cause and entry.fail_closed_cause not in causes:
            causes.append(entry.fail_closed_cause)
    if causes:
        lines.append(
            "Fail-closed events recorded on this call: " + ", ".join(causes) + "."
        )

    if mock_mode and scenario:
        source = scenario_source or "unknown"
        lines.append(
            f"Mock mode. The scripted scenario was {scenario}, selected by "
            f"{source}. The state machine and the scoring are the real ones."
        )

    return lines


# -------------------------------------------------------------- the builder


class EvidenceBuilder:
    """Assembles bundles and summaries from the engine plus the journal."""

    def __init__(
        self,
        *,
        engine: DetectorEngine,
        settings: Settings,
        journal: SessionJournal,
        redactor: Redactor,
        media_route_template: str,
        clock: Any = _utcnow,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.journal = journal
        self.redactor = redactor
        self.media_route_template = media_route_template
        self.clock = clock

    # ------------------------------------------------------------- internals

    def _live(self, session_id: str) -> FactorsResponse | None:
        """Authoritative current state, or None if the store has nothing.

        Goes through `factors_for` rather than reading the store directly, for
        the same reason `GET /factors` does: a read that notices a session has
        outlived its time to live must make that durable.
        """

        try:
            return self.engine.factors_for(session_id)
        except SessionNotFound:
            return None

    def stored(self, session_id: str) -> Session | None:
        """The store's own snapshot, read without mutating it, or None."""

        try:
            return self.engine.store.get(session_id)
        except SessionNotFound:
            return None

    def _thresholds(self) -> EvidenceThresholds:
        return EvidenceThresholds(
            allow_below=scoring.ALLOW_BELOW,
            deny_at_or_above=scoring.DENY_AT_OR_ABOVE,
            fatal_threshold=scoring.FATAL_THRESHOLD,
            fatal_factors=sorted(f.value for f in scoring.FATAL_FACTORS),
            otp_weight=scoring.OTP_WEIGHT,
            otp_max_attempts=self.settings.otp_max_attempts,
            otp_ttl_seconds=self.settings.otp_ttl_seconds,
            session_ttl_seconds=self.settings.session_ttl_seconds,
            speech_segments_before_decision=(
                self.settings.speech_segments_before_decision
            ),
        )

    def media_block(
        self, attachment: MediaAttachment | None, session_id: str
    ) -> EvidenceMedia:
        route = self.media_route_template.replace("{session_id}", session_id)
        if attachment is None or not attachment.attached:
            return EvidenceMedia(status="awaiting_agent", attach_with=route)
        return EvidenceMedia(
            status="attached",
            attach_with=route,
            recording_uri=attachment.recording_uri,
            recording_duration_ms=attachment.recording_duration_ms,
            transcript_uri=attachment.transcript_uri,
            transcript_text=(
                self.redactor.scrub_text(attachment.transcript_text)
                if attachment.transcript_text
                else None
            ),
            note=(
                self.redactor.scrub_text(attachment.note) if attachment.note else None
            ),
            attached_at=attachment.attached_at,
        )

    def _timeline(self, record: SessionRecord | None) -> list[EvidenceTimelineEntry]:
        if record is None:
            return []
        return [
            EvidenceTimelineEntry(
                at=entry.at,
                request_id=entry.request_id,
                method=entry.method,
                endpoint=entry.endpoint,
                status=entry.status,
                outcome=entry.outcome,
                latency_ms=entry.latency_ms,
                call_event=entry.call_event,
                sequence=entry.sequence,
                occurred_at=entry.occurred_at,
                decision=entry.decision,
                state=entry.state,
                risk_score=entry.risk_score,
                next_action=entry.next_action,
                reason=entry.reason,
                error_code=entry.error_code,
                error_message=entry.error_message,
                otp_submitted=entry.otp_submitted,
                otp_outcome=entry.otp_outcome,
                idempotent_replay=entry.idempotent_replay,
                fail_closed_cause=entry.fail_closed_cause,
                factor_verdicts=dict(entry.factor_verdicts),
                metadata=dict(entry.metadata),
            )
            for entry in record.timeline()
        ]

    def _otp(
        self,
        live: FactorsResponse | None,
        stored: Session | None,
        record: SessionRecord | None,
    ) -> EvidenceOtp | None:
        attempts = [
            EvidenceOtpAttempt(
                attempt=item.attempt,
                at=item.at,
                outcome=item.outcome,
                detail=item.detail,
            )
            for item in (record.otp_attempts if record else [])
        ]
        challenge = live.otp if live is not None else None
        resolved = stored.otp.resolved if stored is not None and stored.otp else None

        if challenge is None and not attempts:
            return None

        if challenge is None:
            return EvidenceOtp(
                challenged=True,
                code_length=None,
                attempts_used=len(attempts),
                attempts_remaining=None,
                max_attempts=self.settings.otp_max_attempts,
                expires_at=None,
                resolved=resolved,
                attempts=attempts,
            )

        return EvidenceOtp(
            challenged=True,
            code_length=challenge.code_length,
            attempts_used=challenge.attempts_used,
            attempts_remaining=challenge.attempts_remaining,
            max_attempts=self.settings.otp_max_attempts,
            expires_at=challenge.expires_at,
            resolved=resolved,
            attempts=attempts,
        )

    # ---------------------------------------------------------------- public

    def bundle(self, session_id: str) -> EvidenceBundle:
        """The complete record for one session.

        Raises `SessionNotFound` when neither the store nor the journal has
        ever heard of it.
        """

        live = self._live(session_id)
        stored = self.stored(session_id)
        record = self.journal.get(session_id)

        if live is None and record is None:
            raise SessionNotFound(f"no session {session_id!r}")

        source: Literal["live_session", "journal_only"] = (
            "live_session" if live is not None else "journal_only"
        )

        if live is not None:
            decision = live.decision.value
            state = live.state.value
            next_action = live.next_action.value
            reason = live.reason
            risk = float(live.risk_score)
            confidence = float(live.confidence)
            scenario = live.scenario.value if live.scenario else None
            factors = build_factors(live.factors)
            created_at: datetime | None = live.created_at
            updated_at: datetime | None = live.updated_at
            events_seen: int | None = live.events_seen
        else:
            if record is None:  # pragma: no cover - the guard above rules it out
                raise SessionNotFound(f"no session {session_id!r}")
            decision = record.decision or Decision.PENDING.value
            state = record.state or "created"
            next_action = None
            reason = record.reason
            risk = float(record.risk_score or 0.0)
            confidence = float(record.confidence or 0.0)
            scenario = record.scenario
            factors = _factors_from_record(record)
            created_at = record.first_seen
            updated_at = record.last_seen
            events_seen = None

        timeline = self._timeline(record)
        otp = self._otp(live, stored, record)

        scenario_source = stored.scenario_source if stored is not None else None
        claimed = None
        if stored is not None:
            claimed = stored.claimed_user_id
        if claimed is None and record is not None:
            claimed = record.claimed_user_id

        truncated = bool(record is not None and record.requests > len(record.entries))

        counts = EvidenceCounts(
            requests=record.requests if record else 0,
            rejected_requests=record.rejected_requests if record else 0,
            events_seen=events_seen,
            speech_segments=stored.speech_segments if stored is not None else None,
            otp_attempts=record.otp_attempt_count if record else 0,
            timeline_truncated=truncated,
        )

        why = explain(
            decision=decision,
            state=state,
            risk_score=risk,
            confidence=confidence,
            factors=factors,
            otp=otp,
            scenario=scenario,
            scenario_source=scenario_source,
            timeline=timeline,
            settings=self.settings,
            mock_mode=self.settings.mock_mode,
        )

        safe_reason = self.redactor.scrub_text(reason) if reason else None

        return EvidenceBundle(
            generated_at=self.clock(),
            session_id=session_id,
            state_source=source,
            mock_mode=self.settings.mock_mode,
            headline=headline_for(decision, safe_reason),
            decision=decision,
            state=state,
            next_action=next_action,
            reason=safe_reason,
            why=why,
            risk_score=risk,
            confidence=confidence,
            scenario=scenario,
            scenario_source=scenario_source,
            claimed_user_id=self.redactor.scrub_secrets(claimed) if claimed else None,
            thresholds=self._thresholds(),
            factors=factors,
            otp=otp,
            timeline=timeline,
            counts=counts,
            media=self.media_block(record.media if record else None, session_id),
            human_review=human_review_for(decision, state, factors),
            created_at=created_at,
            updated_at=updated_at,
            first_seen=record.first_seen if record else None,
            last_seen=record.last_seen if record else None,
            redaction=redaction_notes(),
        )

    def summaries(self, limit: int = 50) -> SessionList:
        """The dashboard list, most recently active first.

        Sourced from the journal, because `SessionStore` has no listing API by
        design. A call the journal has evicted is gone from this list even if
        the store still holds the session; the bundle route still answers for
        it by id.
        """

        rows: list[SessionSummary] = []
        for record in self.journal.recent(limit):
            stored = self.stored(record.session_id)
            decision = (
                stored.decision.value
                if stored is not None
                else (record.decision or Decision.PENDING.value)
            )
            state = (
                stored.state.value
                if stored is not None
                else (record.state or "created")
            )
            reason = stored.reason if stored is not None else record.reason
            safe_reason = self.redactor.scrub_text(reason) if reason else None
            risk = float(stored.risk_score) if stored is not None else record.risk_score
            confidence = (
                float(stored.confidence) if stored is not None else record.confidence
            )
            scenario = (
                stored.scenario.value
                if stored is not None and self.settings.mock_mode
                else record.scenario
            )
            claimed = (
                stored.claimed_user_id if stored is not None else record.claimed_user_id
            )
            rows.append(
                SessionSummary(
                    session_id=record.session_id,
                    decision=decision,
                    state=state,
                    headline=headline_for(decision, safe_reason),
                    reason=safe_reason,
                    risk_score=risk,
                    confidence=confidence,
                    scenario=scenario,
                    claimed_user_id=(
                        self.redactor.scrub_secrets(claimed) if claimed else None
                    ),
                    first_seen=record.first_seen,
                    last_seen=record.last_seen,
                    requests=record.requests,
                    rejected_requests=record.rejected_requests,
                    otp_attempts=record.otp_attempt_count,
                    has_media=record.media.attached,
                    live=stored is not None,
                )
            )

        return SessionList(
            generated_at=self.clock(),
            count=len(rows),
            journal_sessions=len(self.journal),
            active_sessions=len(self.engine.store),
            journal_enabled=self.journal.enabled,
            dropped_sessions=self.journal.dropped_sessions,
            sessions=rows,
        )


def _factors_from_record(record: SessionRecord) -> list[EvidenceFactor]:
    """Rebuild factors from the journal when the store no longer has them.

    Detail text is not journalled, so it is absent here and the bundle says so
    via `state_source == "journal_only"`.
    """

    rebuilt: list[EvidenceFactor] = []
    total = 0.0
    risks: dict[str, float] = {}
    for name, score in record.factor_scores.items():
        risks[name] = 1.0 - float(score)
        total += risks[name]
    for name, verdict in record.factor_verdicts.items():
        score = float(record.factor_scores.get(name, 0.0))
        risk = 1.0 - score
        candidate = name in {f.value for f in scoring.FATAL_FACTORS}
        rebuilt.append(
            EvidenceFactor(
                name=name,
                verdict=verdict,
                score=score,
                risk=risk,
                weight=0.0,
                risk_share=(risk / total) if total > 0 else 0.0,
                detail=None,
                fatal=candidate and score <= scoring.FATAL_THRESHOLD,
                fatal_candidate=candidate,
            )
        )
    rebuilt.sort(key=lambda f: (not f.fatal, -f.risk_share, f.name))
    return rebuilt


# ------------------------------------------------------------ text rendering

_RULE = "=" * 72
_THIN = "-" * 72


def render_bundle_text(bundle: EvidenceBundle) -> str:
    """The same record as flat text, for an email, a ticket or a printout."""

    out: list[str] = []
    add = out.append

    add(_RULE)
    add("CALLBACK VERIFICATION RECORD")
    add(_RULE)
    add(f"Session          {bundle.session_id}")
    add(f"Generated        {bundle.generated_at.isoformat()}")
    add(f"Bundle version   {bundle.bundle_version}")
    add(f"Contract version {bundle.contract_version}")
    add(f"Record source    {bundle.state_source}")
    add(f"Mock mode        {'yes' if bundle.mock_mode else 'no'}")
    add("")

    add(_RULE)
    add(f"VERDICT: {bundle.decision.upper()}")
    add(_RULE)
    if bundle.reason:
        add(bundle.reason)
        add("")
    add(f"State            {bundle.state}")
    if bundle.next_action:
        add(f"Next action      {bundle.next_action}")
    add(
        f"Risk score       {bundle.risk_score:.3f}  (allow below "
        f"{bundle.thresholds.allow_below:.2f}, deny at or above "
        f"{bundle.thresholds.deny_at_or_above:.2f})"
    )
    add(f"Confidence       {bundle.confidence:.3f}")
    if bundle.scenario:
        add(f"Scenario         {bundle.scenario} (via {bundle.scenario_source})")
    if bundle.claimed_user_id:
        add(f"Claimed identity {bundle.claimed_user_id}")
    add("")

    add("WHY")
    add(_THIN)
    for index, line in enumerate(bundle.why, start=1):
        add(f"{index}. {line}")
    if not bundle.why:
        add("No reasoning was recorded.")
    add("")

    add("FACTORS")
    add(_THIN)
    if not bundle.factors:
        add("No factors were recorded for this session.")
    for factor in bundle.factors:
        flag = "  [FATAL]" if factor.fatal else ""
        add(
            f"{factor.name:<18} {factor.verdict:<13} score {factor.score:.2f}  "
            f"weight {factor.weight:.2f}  risk share {_pct(factor.risk_share)}{flag}"
        )
        if factor.detail:
            add(f"    {factor.detail}")
    add("")

    add("ONE-TIME CODE")
    add(_THIN)
    if bundle.otp is None:
        add("No step-up was issued on this call.")
    else:
        otp = bundle.otp
        add(f"Code length      {otp.code_length if otp.code_length else 'unknown'}")
        add("Code value       withheld, never recorded")
        add(f"Attempts used    {otp.attempts_used} of {otp.max_attempts}")
        if otp.attempts_remaining is not None:
            add(f"Attempts left    {otp.attempts_remaining}")
        if otp.expires_at:
            add(f"Expires at       {otp.expires_at.isoformat()}")
        resolution = "open"
        if otp.resolved is True:
            resolution = "resolved by a correct code"
        elif otp.resolved is False:
            resolution = "exhausted without a correct code"
        add(f"Resolution       {resolution}")
        for attempt in otp.attempts:
            add(
                f"  attempt {attempt.attempt} at {attempt.at.isoformat()}: "
                f"{attempt.outcome}"
            )
    add("")

    add("TIMELINE")
    add(_THIN)
    if not bundle.timeline:
        add("No requests were recorded for this session.")
    if bundle.counts.timeline_truncated:
        add(
            f"Showing the most recent {len(bundle.timeline)} of "
            f"{bundle.counts.requests} requests. Older entries were dropped by "
            "the journal size bound."
        )
    for entry in bundle.timeline:
        label = entry.call_event or entry.endpoint
        bits = [
            entry.at.isoformat(),
            f"seq {entry.sequence}" if entry.sequence is not None else "seq -",
            label,
            f"HTTP {entry.status}",
            f"{entry.latency_ms}ms",
        ]
        if entry.decision:
            bits.append(f"-> {entry.decision}/{entry.state}")
        if entry.otp_submitted:
            bits.append(f"code {entry.otp_outcome}")
        if entry.idempotent_replay:
            bits.append("idempotent replay")
        if entry.error_code:
            bits.append(f"error {entry.error_code}")
        add("  " + "  ".join(bits))
        if entry.reason:
            add(f"      {entry.reason}")
        if entry.metadata:
            add(f"      metadata {entry.metadata}")
    add("")

    add("CALL RECORDING AND TRANSCRIPT")
    add(_THIN)
    media = bundle.media
    if media.status == "awaiting_agent":
        add("Not attached yet.")
        add(f"The agent attaches them with POST {media.attach_with}")
    else:
        add(f"Recording   {media.recording_uri or 'not supplied'}")
        if media.recording_duration_ms is not None:
            add(f"Duration    {media.recording_duration_ms} ms")
        add(f"Transcript  {media.transcript_uri or 'not supplied'}")
        if media.note:
            add(f"Note        {media.note}")
        if media.attached_at:
            add(f"Attached    {media.attached_at.isoformat()}")
        if media.transcript_text:
            add("")
            add("Transcript text")
            for line in media.transcript_text.splitlines():
                add(f"  {line}")
    add("")

    add("FOR THE REVIEWER")
    add(_THIN)
    add(bundle.human_review.question)
    add(f"Detector recommendation: {bundle.human_review.recommendation}")
    add(bundle.human_review.guidance)
    for caveat in bundle.human_review.caveats:
        add(f"  - {caveat}")
    add("")

    add("COUNTS")
    add(_THIN)
    add(f"Requests recorded   {bundle.counts.requests}")
    add(f"Requests refused    {bundle.counts.rejected_requests}")
    if bundle.counts.events_seen is not None:
        add(f"Events accepted     {bundle.counts.events_seen}")
    if bundle.counts.speech_segments is not None:
        add(f"Speech segments     {bundle.counts.speech_segments}")
    add(f"Code attempts       {bundle.counts.otp_attempts}")
    add("")

    add("WHAT THIS RECORD DOES NOT CONTAIN")
    add(_THIN)
    for note in bundle.redaction:
        add(f"  - {note}")
    add("")
    add(_RULE)
    add("End of record.")
    add(_RULE)
    return "\n".join(out) + "\n"


# -------------------------------------------------------------- the dashboard

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>Callback review</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #121824;
    --panel-2: #172031;
    --line: #232d41;
    --line-soft: #1b2433;
    --text: #e8edf7;
    --muted: #93a1ba;
    --faint: #6a7a93;
    --allow: #2bd47d;
    --allow-dim: rgba(43, 212, 125, 0.14);
    --deny: #ff5a63;
    --deny-dim: rgba(255, 90, 99, 0.14);
    --warn: #ffb648;
    --warn-dim: rgba(255, 182, 72, 0.14);
    --idle: #7d8aa3;
    --idle-dim: rgba(125, 138, 163, 0.14);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
            "Helvetica Neue", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  a { color: #8ab4ff; }

  header.top {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; padding: 14px 22px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, #121a28 0%, #0d131e 100%);
    position: sticky; top: 0; z-index: 10;
  }
  .brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .brand h1 {
    font-size: 16px; font-weight: 650; letter-spacing: 0.01em; margin: 0;
  }
  .brand .sub { color: var(--muted); font-size: 13px; }
  .controls { display: flex; align-items: center; gap: 14px; font-size: 13px; }
  .controls label { display: flex; align-items: center; gap: 7px; color: var(--muted); cursor: pointer; }
  .status { color: var(--faint); font-family: var(--mono); font-size: 12px; }
  .pulse {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--allow); box-shadow: 0 0 0 0 var(--allow-dim);
  }

  main { display: grid; grid-template-columns: 340px 1fr; height: calc(100% - 57px); }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }

  aside {
    border-right: 1px solid var(--line);
    overflow-y: auto; background: var(--panel);
  }
  .list-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--faint);
    border-bottom: 1px solid var(--line-soft);
    position: sticky; top: 0; background: var(--panel);
  }
  ul#sessions { list-style: none; margin: 0; padding: 0; }
  li.card {
    padding: 13px 16px; border-bottom: 1px solid var(--line-soft);
    cursor: pointer; border-left: 3px solid transparent;
  }
  li.card:hover { background: var(--panel-2); }
  li.card.sel { background: var(--panel-2); border-left-color: currentColor; }
  li.card .row1 { display: flex; align-items: center; gap: 8px; }
  li.card .sid {
    font-family: var(--mono); font-size: 12px; color: var(--muted);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  li.card .why {
    margin-top: 6px; font-size: 13px; color: var(--text);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
  }
  li.card .meta {
    margin-top: 6px; font-size: 11px; color: var(--faint);
    font-family: var(--mono); display: flex; gap: 10px; flex-wrap: wrap;
  }
  p.empty { padding: 18px 16px; color: var(--faint); font-size: 13px; }

  .chip {
    display: inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; border: 1px solid currentColor;
  }
  .v-allow { color: var(--allow); background: var(--allow-dim); }
  .v-deny { color: var(--deny); background: var(--deny-dim); }
  .v-challenge { color: var(--warn); background: var(--warn-dim); }
  .v-pending { color: var(--idle); background: var(--idle-dim); }

  section#detail { overflow-y: auto; padding: 0 0 60px 0; }
  .placeholder {
    padding: 64px 32px; color: var(--faint); font-size: 15px; max-width: 46ch;
  }

  .verdict {
    padding: 26px 32px 24px 32px; border-bottom: 1px solid var(--line);
    border-left: 5px solid currentColor;
  }
  .verdict.v-allow { background: linear-gradient(90deg, var(--allow-dim), transparent 70%); }
  .verdict.v-deny { background: linear-gradient(90deg, var(--deny-dim), transparent 70%); }
  .verdict.v-challenge { background: linear-gradient(90deg, var(--warn-dim), transparent 70%); }
  .verdict.v-pending { background: linear-gradient(90deg, var(--idle-dim), transparent 70%); }
  .verdict .word {
    font-size: 13px; font-weight: 750; letter-spacing: 0.18em;
    text-transform: uppercase;
  }
  .verdict .reason {
    color: var(--text); font-size: 26px; line-height: 1.3; font-weight: 600;
    margin: 8px 0 0 0; max-width: 34ch;
  }
  @media (max-width: 700px) { .verdict .reason { font-size: 21px; } }
  .verdict .ask { margin-top: 14px; font-size: 14px; color: var(--muted); }
  .verdict .ask strong { color: var(--text); }

  .block { padding: 22px 32px; border-bottom: 1px solid var(--line-soft); }
  .block h2 {
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--faint); margin: 0 0 14px 0; font-weight: 700;
  }
  ol.why { margin: 0; padding-left: 22px; max-width: 86ch; }
  ol.why li { margin-bottom: 9px; }

  .kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px 22px; }
  .kv div span.k {
    display: block; font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--faint);
  }
  .kv div span.v { font-family: var(--mono); font-size: 13px; }

  .meter { margin-top: 6px; }
  .meter .track {
    position: relative; height: 8px; border-radius: 4px;
    background: #1d2636; overflow: hidden;
  }
  .meter .fill { height: 100%; border-radius: 4px; background: currentColor; }
  .meter .ticks { position: relative; height: 12px; font-size: 10px; color: var(--faint); }
  .meter .tick { position: absolute; transform: translateX(-50%); font-family: var(--mono); }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left; font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--faint); font-weight: 700;
    padding: 0 10px 8px 0; border-bottom: 1px solid var(--line);
  }
  td { padding: 10px 10px 10px 0; border-bottom: 1px solid var(--line-soft); vertical-align: top; }
  td.num { font-family: var(--mono); white-space: nowrap; }
  tr.fatal td { background: var(--deny-dim); }
  .bar { height: 6px; border-radius: 3px; background: #1d2636; width: 92px; }
  .bar > i { display: block; height: 100%; border-radius: 3px; background: currentColor; }
  .fname { font-family: var(--mono); }
  .fdetail { color: var(--muted); }
  .badge-fatal {
    color: var(--deny); font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    border: 1px solid currentColor; border-radius: 3px; padding: 0 4px; margin-left: 6px;
  }

  .notes { margin: 0; padding-left: 20px; color: var(--muted); font-size: 13px; max-width: 86ch; }
  .notes li { margin-bottom: 7px; }
  .foot { padding: 20px 32px; color: var(--faint); font-size: 12px; }
  .pill {
    display: inline-block; font-family: var(--mono); font-size: 11px;
    border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px;
    color: var(--muted);
  }
  .err { color: var(--deny); }
</style>
</head>
<body>
<header class="top">
  <div class="brand">
    <h1>Callback review</h1>
    <span class="sub">A person approves before any money moves.</span>
  </div>
  <div class="controls">
    <label><input type="checkbox" id="auto" checked> Live refresh</label>
    <span class="pulse" id="pulse"></span>
    <span class="status" id="status">starting</span>
  </div>
</header>
<main>
  <aside>
    <div class="list-head"><span>Recent calls</span><span id="count"></span></div>
    <ul id="sessions"></ul>
    <p class="empty" id="empty">No calls recorded yet. This list fills as the agent pushes events.</p>
  </aside>
  <section id="detail">
    <div class="placeholder">Pick a call on the left. The reason for the decision is the first thing you will see.</div>
  </section>
</main>
<script>
(function () {
  var BASE = window.location.pathname.replace(/\/+$/, '');
  var API = BASE + '/api/sessions';
  var REFRESH_MS = 3000;
  var selected = null;
  var listEl = document.getElementById('sessions');
  var emptyEl = document.getElementById('empty');
  var detailEl = document.getElementById('detail');
  var statusEl = document.getElementById('status');
  var countEl = document.getElementById('count');
  var autoEl = document.getElementById('auto');
  var pulseEl = document.getElementById('pulse');

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }
  function verdictClass(decision) {
    if (decision === 'allow') { return 'v-allow'; }
    if (decision === 'deny') { return 'v-deny'; }
    if (decision === 'challenge') { return 'v-challenge'; }
    return 'v-pending';
  }
  function num(value, digits) {
    if (value === null || value === undefined) { return '-'; }
    return Number(value).toFixed(digits === undefined ? 3 : digits);
  }
  function pct(value) {
    if (value === null || value === undefined) { return '-'; }
    return Math.round(Number(value) * 100) + '%';
  }
  function clock(iso) {
    if (!iso) { return '-'; }
    var d = new Date(iso);
    if (isNaN(d.getTime())) { return iso; }
    return d.toLocaleTimeString();
  }
  function setStatus(text, bad) {
    statusEl.textContent = text;
    statusEl.className = bad ? 'status err' : 'status';
  }

  function get(url) {
    return fetch(url, {
      credentials: 'same-origin',
      headers: { 'Accept': 'application/json' },
      cache: 'no-store'
    }).then(function (res) {
      if (res.status === 401) { throw new Error('unauthorized'); }
      if (res.status === 404) { throw new Error('not found'); }
      if (!res.ok) { throw new Error('HTTP ' + res.status); }
      return res.json();
    });
  }

  function meter(value, cls) {
    var wrap = el('div', 'meter ' + cls);
    var track = el('div', 'track');
    var fill = el('div', 'fill');
    fill.style.width = Math.max(0, Math.min(100, Number(value || 0) * 100)) + '%';
    track.appendChild(fill);
    wrap.appendChild(track);
    var ticks = el('div', 'ticks');
    [[30, 'allow 0.30'], [55, 'deny 0.55']].forEach(function (pair) {
      var t = el('span', 'tick', pair[1]);
      t.style.left = pair[0] + '%';
      ticks.appendChild(t);
    });
    wrap.appendChild(ticks);
    return wrap;
  }

  function kv(label, value) {
    var box = el('div');
    box.appendChild(el('span', 'k', label));
    box.appendChild(el('span', 'v', value === null || value === undefined || value === '' ? '-' : value));
    return box;
  }

  function block(title) {
    var b = el('div', 'block');
    if (title) { b.appendChild(el('h2', null, title)); }
    return b;
  }

  function renderList(data) {
    countEl.textContent = data.count + (data.journal_enabled ? '' : ' (journal off)');
    listEl.textContent = '';
    emptyEl.style.display = data.sessions.length ? 'none' : 'block';
    if (!data.journal_enabled) {
      emptyEl.textContent = 'The review journal is turned off, so no call history is kept.';
    }
    data.sessions.forEach(function (s) {
      var cls = verdictClass(s.decision);
      var li = el('li', 'card ' + cls + (s.session_id === selected ? ' sel' : ''));
      var row = el('div', 'row1');
      row.appendChild(el('span', 'chip ' + cls, s.decision));
      row.appendChild(el('span', 'sid', s.session_id));
      li.appendChild(row);
      li.appendChild(el('div', 'why', s.reason || 'No reason recorded.'));
      var meta = el('div', 'meta');
      meta.appendChild(el('span', null, 'risk ' + num(s.risk_score, 2)));
      meta.appendChild(el('span', null, s.state));
      if (s.scenario) { meta.appendChild(el('span', null, s.scenario)); }
      meta.appendChild(el('span', null, clock(s.last_seen)));
      if (!s.live) { meta.appendChild(el('span', null, 'evicted')); }
      li.appendChild(meta);
      li.addEventListener('click', function () {
        selected = s.session_id;
        renderList(data);
        loadDetail(selected);
      });
      listEl.appendChild(li);
    });
  }

  function renderFactors(factors) {
    var b = block('Why, factor by factor');
    if (!factors.length) {
      b.appendChild(el('p', 'fdetail', 'No factors were recorded for this session. Nothing here verifies the caller.'));
      return b;
    }
    var table = el('table');
    var head = el('tr');
    ['Factor', 'Verdict', 'Score', 'Share of risk', 'Weight', 'What was observed'].forEach(function (h) {
      head.appendChild(el('th', null, h));
    });
    var thead = el('thead'); thead.appendChild(head); table.appendChild(thead);
    var body = el('tbody');
    factors.forEach(function (f) {
      var tr = el('tr', f.fatal ? 'fatal' : null);
      var nameCell = el('td', 'fname');
      nameCell.appendChild(el('span', null, f.name));
      if (f.fatal) { nameCell.appendChild(el('span', 'badge-fatal', 'FATAL')); }
      tr.appendChild(nameCell);
      var vc = f.verdict === 'pass' ? 'v-allow' : (f.verdict === 'fail' ? 'v-deny' : 'v-pending');
      var vcell = el('td');
      vcell.appendChild(el('span', 'chip ' + vc, f.verdict));
      tr.appendChild(vcell);
      tr.appendChild(el('td', 'num', num(f.score, 2)));
      var shareCell = el('td', 'num ' + (f.risk_share > 0.25 ? 'v-deny' : ''));
      shareCell.appendChild(el('span', null, pct(f.risk_share)));
      var bar = el('div', 'bar');
      var inner = el('i');
      inner.style.width = Math.round(Math.min(1, f.risk_share) * 100) + '%';
      bar.appendChild(inner);
      shareCell.appendChild(bar);
      tr.appendChild(shareCell);
      tr.appendChild(el('td', 'num', num(f.weight, 2)));
      tr.appendChild(el('td', 'fdetail', f.detail || '-'));
      body.appendChild(tr);
    });
    table.appendChild(body);
    b.appendChild(table);
    return b;
  }

  function renderTimeline(bundle) {
    var b = block('Timeline');
    if (bundle.counts.timeline_truncated) {
      b.appendChild(el('p', 'fdetail',
        'Showing the most recent ' + bundle.timeline.length + ' of ' +
        bundle.counts.requests + ' requests.'));
    }
    if (!bundle.timeline.length) {
      b.appendChild(el('p', 'fdetail', 'No requests were recorded for this session.'));
      return b;
    }
    var table = el('table');
    var head = el('tr');
    ['Time', 'Seq', 'Event', 'Status', 'Latency', 'Result', 'Note'].forEach(function (h) {
      head.appendChild(el('th', null, h));
    });
    var thead = el('thead'); thead.appendChild(head); table.appendChild(thead);
    var body = el('tbody');
    bundle.timeline.forEach(function (e) {
      var tr = el('tr');
      tr.appendChild(el('td', 'num', clock(e.at)));
      tr.appendChild(el('td', 'num', e.sequence === null || e.sequence === undefined ? '-' : e.sequence));
      tr.appendChild(el('td', 'fname', e.call_event || e.endpoint));
      tr.appendChild(el('td', 'num' + (e.status >= 400 ? ' err' : ''), e.status));
      tr.appendChild(el('td', 'num', e.latency_ms + 'ms'));
      var result = el('td');
      if (e.decision) {
        result.appendChild(el('span', 'chip ' + verdictClass(e.decision), e.decision));
      } else {
        result.appendChild(el('span', 'fdetail', '-'));
      }
      tr.appendChild(result);
      var notes = [];
      if (e.otp_submitted) { notes.push('code ' + e.otp_outcome); }
      if (e.idempotent_replay) { notes.push('idempotent replay'); }
      if (e.error_code) { notes.push(e.error_code); }
      if (e.fail_closed_cause) { notes.push('fail closed: ' + e.fail_closed_cause); }
      tr.appendChild(el('td', 'fdetail', notes.join(', ') || (e.reason || '-')));
      body.appendChild(tr);
    });
    table.appendChild(body);
    b.appendChild(table);
    return b;
  }

  function renderOtp(otp) {
    var b = block('One-time code');
    if (!otp) {
      b.appendChild(el('p', 'fdetail', 'No step-up was issued on this call.'));
      return b;
    }
    var grid = el('div', 'kv');
    var resolution = otp.resolved === true ? 'resolved by a correct code'
      : (otp.resolved === false ? 'exhausted without a correct code' : 'open');
    grid.appendChild(kv('Resolution', resolution));
    grid.appendChild(kv('Attempts', otp.attempts_used + ' of ' + otp.max_attempts));
    grid.appendChild(kv('Attempts left', otp.attempts_remaining));
    grid.appendChild(kv('Code length', otp.code_length));
    grid.appendChild(kv('Code value', 'withheld, never recorded'));
    grid.appendChild(kv('Expires', clock(otp.expires_at)));
    b.appendChild(grid);
    if (otp.attempts.length) {
      var list = el('ul', 'notes');
      otp.attempts.forEach(function (a) {
        list.appendChild(el('li', null,
          'attempt ' + a.attempt + ' at ' + clock(a.at) + ': ' + a.outcome));
      });
      b.appendChild(list);
    }
    return b;
  }

  function renderMedia(media) {
    var b = block('Call recording and transcript');
    if (media.status === 'awaiting_agent') {
      b.appendChild(el('p', 'fdetail', 'Not attached yet. The agent posts them to:'));
      b.appendChild(el('div', 'pill', 'POST ' + media.attach_with));
      return b;
    }
    var grid = el('div', 'kv');
    grid.appendChild(kv('Recording', media.recording_uri));
    grid.appendChild(kv('Duration', media.recording_duration_ms === null ||
      media.recording_duration_ms === undefined ? '-' : media.recording_duration_ms + ' ms'));
    grid.appendChild(kv('Transcript', media.transcript_uri));
    grid.appendChild(kv('Attached', clock(media.attached_at)));
    b.appendChild(grid);
    if (media.note) { b.appendChild(el('p', 'fdetail', media.note)); }
    if (media.transcript_text) {
      var pre = el('pre', 'fdetail', media.transcript_text);
      pre.style.whiteSpace = 'pre-wrap';
      b.appendChild(pre);
    }
    return b;
  }

  function renderDetail(bundle) {
    var cls = verdictClass(bundle.decision);
    detailEl.textContent = '';

    var verdict = el('div', 'verdict ' + cls);
    verdict.appendChild(el('div', 'word', bundle.decision));
    verdict.appendChild(el('p', 'reason', bundle.reason || 'No reason was recorded for this decision.'));
    var ask = el('div', 'ask');
    ask.appendChild(el('strong', null, bundle.human_review.question + ' '));
    ask.appendChild(el('span', null, bundle.human_review.guidance));
    verdict.appendChild(ask);
    detailEl.appendChild(verdict);

    var why = block('Why this decision');
    var ol = el('ol', 'why');
    bundle.why.forEach(function (line) { ol.appendChild(el('li', null, line)); });
    if (!bundle.why.length) { ol.appendChild(el('li', null, 'No reasoning was recorded.')); }
    why.appendChild(ol);
    detailEl.appendChild(why);

    var scores = block('Scores');
    var grid = el('div', 'kv');
    var riskBox = el('div');
    riskBox.appendChild(el('span', 'k', 'Blended risk'));
    riskBox.appendChild(el('span', 'v', num(bundle.risk_score, 3)));
    riskBox.appendChild(meter(bundle.risk_score, cls));
    grid.appendChild(riskBox);
    grid.appendChild(kv('Confidence', num(bundle.confidence, 3)));
    grid.appendChild(kv('State', bundle.state));
    grid.appendChild(kv('Next action', bundle.next_action));
    grid.appendChild(kv('Scenario', bundle.scenario ? bundle.scenario + ' (' + (bundle.scenario_source || '?') + ')' : null));
    grid.appendChild(kv('Claimed identity', bundle.claimed_user_id));
    grid.appendChild(kv('Session', bundle.session_id));
    grid.appendChild(kv('Record source', bundle.state_source));
    grid.appendChild(kv('Updated', clock(bundle.updated_at)));
    scores.appendChild(grid);
    detailEl.appendChild(scores);

    detailEl.appendChild(renderFactors(bundle.factors));
    detailEl.appendChild(renderOtp(bundle.otp));
    detailEl.appendChild(renderTimeline(bundle));
    detailEl.appendChild(renderMedia(bundle.media));

    var notes = block('What this record does not contain');
    var ul = el('ul', 'notes');
    bundle.redaction.forEach(function (n) { ul.appendChild(el('li', null, n)); });
    notes.appendChild(ul);
    detailEl.appendChild(notes);

    var foot = el('div', 'foot');
    var link = el('a', null, 'Open the plain-text record for this session');
    link.href = API + '/' + encodeURIComponent(bundle.session_id) + '/text';
    link.target = '_blank';
    link.rel = 'noreferrer';
    foot.appendChild(link);
    foot.appendChild(el('div', null,
      'bundle ' + bundle.bundle_version + ' , contract ' + bundle.contract_version +
      ' , generated ' + clock(bundle.generated_at)));
    detailEl.appendChild(foot);
  }

  function loadDetail(sessionId) {
    if (!sessionId) { return Promise.resolve(); }
    return get(API + '/' + encodeURIComponent(sessionId)).then(renderDetail).catch(function (err) {
      detailEl.textContent = '';
      var b = block('Could not load this session');
      b.appendChild(el('p', 'err', String(err.message || err)));
      detailEl.appendChild(b);
    });
  }

  var refreshing = false;
  function refresh() {
    if (refreshing) { return; }
    refreshing = true;
    get(API + '?limit=50').then(function (data) {
      renderList(data);
      setStatus('updated ' + new Date().toLocaleTimeString());
      pulseEl.style.background = 'var(--allow)';
      if (selected) { return loadDetail(selected); }
      return null;
    }).catch(function (err) {
      pulseEl.style.background = 'var(--deny)';
      if (String(err.message) === 'unauthorized') {
        setStatus('credentials rejected, reload and sign in again', true);
      } else {
        setStatus(String(err.message || err), true);
      }
    }).then(function () { refreshing = false; });
  }

  setInterval(function () {
    if (autoEl.checked && !document.hidden) { refresh(); }
  }, REFRESH_MS);
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden && autoEl.checked) { refresh(); }
  });
  refresh();
})();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------- routing


def _not_found(session_id: str) -> JSONResponse:
    """A clean contract-shaped 404, produced here rather than by a traceback."""

    body = ErrorResponse(
        error=ErrorBody(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"no session {session_id!r}",
            detail=(
                "Neither the session store nor the review journal has a record "
                "of this session. It may have aged out of both."
            ),
        )
    )
    return JSONResponse(status_code=404, content=body.model_dump(mode="json"))


def unauthorized_handler(realm: str):
    """Handler for `ReviewUnauthorized`, registered by `mount`."""

    async def handle(_: Any, exc: Exception) -> JSONResponse:
        challenge = bool(getattr(exc, "challenge", False))
        message = getattr(exc, "message", "unauthorized")
        body = ErrorResponse(
            error=ErrorBody(code=ErrorCode.UNAUTHORIZED, message=message)
        )
        headers = (
            {"WWW-Authenticate": f'Basic realm="{realm}", charset="UTF-8"'}
            if challenge
            else None
        )
        return JSONResponse(
            status_code=401, content=body.model_dump(mode="json"), headers=headers
        )

    return handle


def _v1_not_found(session_id: str) -> JSONResponse:
    """The review surface's 404, for a /v1 session id."""

    body = ErrorResponse(
        error=ErrorBody(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"no /v1 session {session_id!r}",
            detail=(
                "The /v1 registry has no record of this session. Ended "
                "sessions are evicted an hour after they end."
            ),
        )
    )
    return JSONResponse(status_code=404, content=body.model_dump(mode="json"))


def review_router(
    *,
    builder: EvidenceBuilder,
    observability: Observability,
    auth: TokenAuth,
    prefix: str = DEFAULT_REVIEW_PREFIX,
    v1: V1Service | None = None,
) -> APIRouter:
    """Every human-facing route. All of them require the token."""

    guard = Depends(auth.browser_dependency())
    router = APIRouter(prefix=prefix, tags=["review"], dependencies=[guard])

    if v1 is not None:
        # A read-only view of the /v1 registry, so calls made over the agent
        # contract show up for a reviewer. Outside the wire contract: the
        # agent never calls these. See detector/v1/review_view.py for what
        # they never return.

        @router.get("/api/v1/sessions", response_model=V1SessionList)
        async def v1_sessions() -> V1SessionList:
            return v1_review.list_sessions(v1)

        @router.get("/api/v1/sessions/{session_id}", response_model=V1SessionDetail)
        async def v1_session(session_id: str) -> Any:
            found = v1_review.detail(v1, session_id)
            if found is None:
                return _v1_not_found(session_id)
            return found

    @router.get("", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(
            content=DASHBOARD_HTML,
            headers={
                # Nothing here should sit in a shared cache or be framed.
                "Cache-Control": "no-store",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
            },
        )

    @router.get("/api/sessions", response_model=SessionList)
    async def sessions(
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> SessionList:
        return builder.summaries(limit)

    @router.get("/api/sessions/{session_id}", response_model=EvidenceBundle)
    async def bundle(session_id: str) -> Any:
        try:
            result = builder.bundle(session_id)
        except SessionNotFound:
            return _not_found(session_id)
        observability.metrics.evidence_bundles.inc(format="json")
        return result

    @router.get("/api/sessions/{session_id}/text", response_class=PlainTextResponse)
    async def bundle_text(session_id: str) -> Any:
        try:
            result = builder.bundle(session_id)
        except SessionNotFound:
            return _not_found(session_id)
        observability.metrics.evidence_bundles.inc(format="text")
        return PlainTextResponse(
            content=render_bundle_text(result),
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/api/sessions/{session_id}/media", response_model=MediaAttachResponse)
    async def attach_media(session_id: str, payload: MediaAttachRequest) -> Any:
        known = (
            builder.journal.known(session_id) or builder.stored(session_id) is not None
        )
        if not known:
            return _not_found(session_id)
        record = builder.journal.attach_media(
            session_id,
            recording_uri=payload.recording_uri,
            recording_duration_ms=payload.recording_duration_ms,
            transcript_uri=payload.transcript_uri,
            transcript_text=payload.transcript_text,
            note=payload.note,
        )
        return MediaAttachResponse(
            session_id=session_id,
            media=builder.media_block(record.media, session_id),
        )

    return router


def mount(
    app: FastAPI,
    *,
    engine: DetectorEngine,
    settings: Settings,
    config: ObservabilityConfig | None = None,
    observability: Observability | None = None,
    prefix: str = DEFAULT_REVIEW_PREFIX,
    stream: Any = None,
    v1: V1Service | None = None,
) -> Observability:
    """Install logging, metrics and the review surface on `app`.

    Call this once, inside `create_app`, after the engine exists. Returns the
    `Observability` instance so the caller can keep it on `app.state`.

    Calling it twice on one app is refused. A second install adds a second
    middleware and a second `Observability`, so every request is logged twice
    and counted twice, while `/metrics` and the `/review` routes keep serving
    from the first instance. That is silent at request time and wrong
    everywhere, so it fails here instead, at startup. To configure the layer,
    pass the config to `create_app`, which is the only caller.
    """

    if getattr(app.state, "observability", None) is not None:
        raise RuntimeError(
            "observability is already mounted on this app. A second mount "
            "double-counts every request. Pass observability=... and "
            "log_stream=... to create_app() instead of mounting again."
        )

    resolved_config = config or ObservabilityConfig.from_env()
    redactor = Redactor(
        # The bearer tokens and the mock code are the literals this service
        # holds. A real-mode code is random per session and never leaves
        # OtpState, which is why it is not listed here. An unset review token
        # is the empty string, which add_secrets drops.
        [settings.api_token, settings.review_token, MOCK_OTP_CODE]
    )
    obs = observability or Observability(
        config=resolved_config, redactor=redactor, stream=stream
    )

    clean_prefix = prefix.rstrip("/")
    media_route = f"{clean_prefix}/api/sessions/{{session_id}}/media"
    obs.add_known_endpoints(
        [
            clean_prefix or "/",
            f"{clean_prefix}/api/sessions",
            f"{clean_prefix}/api/sessions/{{session_id}}",
            f"{clean_prefix}/api/sessions/{{session_id}}/text",
            media_route,
            f"{clean_prefix}/api/v1/sessions",
            f"{clean_prefix}/api/v1/sessions/{{session_id}}",
        ]
    )
    obs.bind_active_sessions(lambda: float(len(engine.store)))
    obs.bind_journal_size()
    obs.set_build_info(contract_version=CONTRACT_VERSION, mock_mode=settings.mock_mode)

    # Two guards, deliberately. The review token opens the dashboard and its
    # JSON and stops there, so /metrics keeps the service token alone: its
    # labels carry the decision mix, which is an oracle for a caller who wants
    # to know whether they were caught. /v1 and /events never reach this
    # class, so neither can accept the review token however this is wired.
    metrics_auth = TokenAuth(settings.api_token)
    auth = TokenAuth(settings.api_token, extra_tokens=(settings.review_token,))
    builder = EvidenceBuilder(
        engine=engine,
        settings=settings,
        journal=obs.journal,
        redactor=obs.redactor,
        media_route_template=media_route,
    )

    app.add_exception_handler(ReviewUnauthorized, unauthorized_handler(auth.realm))
    app.include_router(metrics_router(obs, metrics_auth))
    app.include_router(
        review_router(
            builder=builder,
            observability=obs,
            auth=auth,
            prefix=clean_prefix,
            v1=v1,
        )
    )
    app.add_middleware(ObservabilityMiddleware, observability=obs)

    app.state.observability = obs
    app.state.evidence = builder

    obs.logger.log(
        "observability_mounted",
        review_prefix=clean_prefix or "/",
        metrics_path="/metrics",
        metrics_require_auth=resolved_config.metrics_require_auth,
        log_enabled=resolved_config.log_enabled,
        metrics_enabled=resolved_config.metrics_enabled,
        journal_enabled=resolved_config.journal_enabled,
        contract_version=CONTRACT_VERSION,
    )
    return obs
