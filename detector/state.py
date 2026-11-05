"""Session state machine, idempotency and OTP policy.

In-memory on purpose: Day-1 serving is a single Modal container
(min_containers=1, max_containers=1), so one process owns every session. Moving
to more than one replica means swapping SessionStore for Redis or a Modal Dict,
and nothing above this module needs to change.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from detector import scoring
from detector.config import Settings
from detector.errors import (
    InvalidScenario,
    OtpExhausted,
    OtpExpired,
    SequenceConflict,
    SessionNotFound,
    SessionTerminal,
)
from detector.mock import SCRIPTS, resolve_scenario
from shared.contract import (
    MOCK_OTP_CODE,
    Decision,
    EventRequest,
    EventResponse,
    EventType,
    Factor,
    FactorName,
    FactorsResponse,
    FactorVerdict,
    NextAction,
    OtpChallenge,
    Scenario,
    SessionState,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(request: EventRequest) -> str:
    return hashlib.sha256(
        request.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()


@dataclass
class OtpState:
    code: str
    attempts_used: int
    expires_at: datetime
    resolved: bool | None = None

    def challenge(self, max_attempts: int) -> OtpChallenge:
        return OtpChallenge(
            code_length=len(self.code),
            attempts_used=self.attempts_used,
            attempts_remaining=max(0, max_attempts - self.attempts_used),
            expires_at=self.expires_at,
        )


@dataclass
class Session:
    session_id: str
    scenario: Scenario
    scenario_source: str
    created_at: datetime
    updated_at: datetime
    state: SessionState = SessionState.CREATED
    claimed_user_id: str | None = None
    events_seen: int = 0
    speech_segments: int = 0
    max_sequence: int = -1
    factors: list[Factor] = field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    decision: Decision = Decision.PENDING
    next_action: NextAction = NextAction.CONTINUE_LISTENING
    reason: str | None = None
    otp: OtpState | None = None
    # idempotency key -> (request fingerprint, stored response)
    responses: dict[str, tuple[str, EventResponse]] = field(default_factory=dict)


class SessionStore:
    """Thread-safe in-memory session map."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound(f"no session {session_id!r}")
        return session

    def get_or_create(
        self, session_id: str, scenario: Scenario, scenario_source: str
    ) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                now = _now()
                session = Session(
                    session_id=session_id,
                    scenario=scenario,
                    scenario_source=scenario_source,
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[session_id] = session
            return session

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


class DetectorEngine:
    """Applies events to sessions and produces contract responses."""

    def __init__(self, settings: Settings, store: SessionStore | None = None) -> None:
        self.settings = settings
        self.store = store or SessionStore()

    # ---------------------------------------------------------------- events

    def handle_event(
        self,
        request: EventRequest,
        *,
        scenario_header: str | None,
        idempotency_key: str | None,
    ) -> EventResponse:
        try:
            scenario, source = resolve_scenario(scenario_header, request.claimed_user_id)
        except ValueError as exc:
            raise InvalidScenario(str(exc)) from exc

        session = self.store.get_or_create(request.session_id, scenario, source)

        # An explicit header on a later event re-pins the scenario, so a fixture
        # can drive one session through a specific script end to end.
        if scenario_header:
            session.scenario = scenario
            session.scenario_source = source

        cache_key = idempotency_key or f"seq:{request.sequence}"
        fingerprint = _fingerprint(request)

        cached = session.responses.get(cache_key)
        if cached is not None:
            stored_fingerprint, stored_response = cached
            if stored_fingerprint != fingerprint:
                raise SequenceConflict(
                    f"idempotency key {cache_key!r} was already used with a "
                    "different payload"
                )
            return stored_response.model_copy(update={"idempotent_replay": True})

        if request.sequence <= session.max_sequence:
            raise SequenceConflict(
                f"sequence {request.sequence} is not greater than the last "
                f"accepted sequence {session.max_sequence}"
            )

        self._expire_if_due(session)

        if session.state.is_terminal and request.event is not EventType.SESSION_END:
            raise SessionTerminal(
                f"session is {session.state.value} and accepts no further events"
            )

        self._apply(session, request)

        session.events_seen += 1
        session.max_sequence = request.sequence
        session.updated_at = _now()

        response = self._event_response(session, request.sequence)
        session.responses[cache_key] = (fingerprint, response)
        return response

    # ------------------------------------------------------------ transitions

    def _apply(self, session: Session, request: EventRequest) -> None:
        if request.claimed_user_id:
            session.claimed_user_id = request.claimed_user_id

        match request.event:
            case EventType.SESSION_START:
                session.state = SessionState.LISTENING
                session.decision = Decision.PENDING
                session.next_action = NextAction.CONTINUE_LISTENING
                session.reason = "Session open. Gathering speech."

            case EventType.CALLER_CLAIM:
                if session.state is SessionState.CREATED:
                    session.state = SessionState.LISTENING
                session.decision = Decision.PENDING
                session.next_action = NextAction.CONTINUE_LISTENING
                session.reason = (
                    f"Claim recorded for {session.claimed_user_id!r}."
                    if session.claimed_user_id
                    else "Claim recorded."
                )

            case EventType.SPEECH_SEGMENT:
                if session.state is SessionState.CREATED:
                    session.state = SessionState.LISTENING
                session.speech_segments += 1
                if session.state is SessionState.CHALLENGING:
                    # Already stepped up. More speech does not change the gate.
                    return
                if session.speech_segments < self.settings.speech_segments_before_decision:
                    session.decision = Decision.PENDING
                    session.next_action = NextAction.CONTINUE_LISTENING
                    session.reason = (
                        f"{session.speech_segments} of "
                        f"{self.settings.speech_segments_before_decision} segments gathered."
                    )
                    return
                self._evaluate(session)

            case EventType.OTP_REQUEST:
                self._issue_otp(session)

            case EventType.OTP_SUBMIT:
                self._verify_otp(session, request.otp_code)

            case EventType.SESSION_END:
                if not session.state.is_terminal:
                    session.state = SessionState.DENIED
                    session.decision = Decision.DENY
                    session.next_action = NextAction.COMPLETE_DENY
                    session.reason = (
                        "Call ended before verification completed. Failing closed."
                    )

    def _evaluate(self, session: Session) -> None:
        script = SCRIPTS[session.scenario]
        factors = script.factor_list()
        session.confidence = script.confidence

        fatal = scoring.fatal_factor(factors)
        if fatal is not None:
            session.factors = factors
            session.risk_score = scoring.blend_risk(factors)
            session.state = SessionState.DENIED
            session.decision = Decision.DENY
            session.next_action = NextAction.COMPLETE_DENY
            session.reason = (
                f"{fatal.name.value} is unambiguous attack evidence "
                f"(score {fatal.score:.2f}). {script.reason}"
            )
            return

        risk = scoring.blend_risk(factors)
        decision = scoring.decide(risk, otp_available=script.otp_can_rescue)

        session.factors = factors
        session.risk_score = risk

        if decision is Decision.ALLOW:
            session.state = SessionState.ALLOWED
            session.decision = Decision.ALLOW
            session.next_action = NextAction.COMPLETE_ALLOW
            session.reason = script.reason
        elif decision is Decision.DENY:
            session.state = SessionState.DENIED
            session.decision = Decision.DENY
            session.next_action = NextAction.COMPLETE_DENY
            session.reason = script.reason
        else:
            self._issue_otp(session)
            session.reason = script.reason

    # ------------------------------------------------------------------- OTP

    def _issue_otp(self, session: Session) -> None:
        if session.state.is_terminal:
            raise SessionTerminal(
                f"session is {session.state.value} and cannot be challenged"
            )

        code = MOCK_OTP_CODE if self.settings.mock_mode else _random_code()
        now = _now()
        if session.otp is None or session.otp.expires_at <= now:
            session.otp = OtpState(
                code=code,
                attempts_used=0,
                expires_at=now + timedelta(seconds=self.settings.otp_ttl_seconds),
            )

        session.state = SessionState.CHALLENGING
        session.decision = Decision.CHALLENGE
        session.next_action = NextAction.REQUEST_OTP
        if not session.factors:
            session.reason = "Step-up required before a decision."

    def _verify_otp(self, session: Session, submitted: str | None) -> None:
        if session.state is not SessionState.CHALLENGING or session.otp is None:
            raise SessionTerminal(
                "no active OTP challenge; send otp_request first"
            )

        otp = session.otp
        now = _now()
        if otp.expires_at <= now:
            self._fail_closed(session, "OTP expired before it was submitted.")
            raise OtpExpired("OTP challenge expired")

        if otp.attempts_used >= self.settings.otp_max_attempts:
            self._fail_closed(session, "OTP attempt limit already reached.")
            raise OtpExhausted("OTP attempt limit reached")

        otp.attempts_used += 1
        correct = secrets.compare_digest(otp.code, submitted or "")

        behavioural = [f for f in session.factors if f.name is not FactorName.OTP]
        for factor in behavioural:
            factor.weight = factor.weight * (1.0 - scoring.OTP_WEIGHT)

        if correct:
            otp.resolved = True
            otp_factor = Factor(
                name=FactorName.OTP,
                score=1.0,
                weight=scoring.OTP_WEIGHT,
                verdict=FactorVerdict.PASS,
                detail=f"Correct code on attempt {otp.attempts_used}.",
            )
            factors = behavioural + [otp_factor]
            session.factors = factors
            session.risk_score = scoring.blend_risk(factors)
            session.state = SessionState.ALLOWED
            session.decision = Decision.ALLOW
            session.next_action = NextAction.COMPLETE_ALLOW
            session.reason = "Step-up passed. Caller verified by one-time code."
            return

        remaining = self.settings.otp_max_attempts - otp.attempts_used
        if remaining > 0:
            # Restore weights; the gate is still open for another attempt.
            for factor in behavioural:
                factor.weight = factor.weight / (1.0 - scoring.OTP_WEIGHT)
            session.decision = Decision.CHALLENGE
            session.next_action = NextAction.REQUEST_OTP
            session.reason = (
                f"Incorrect code. {remaining} attempt(s) remaining."
            )
            return

        otp.resolved = False
        otp_factor = Factor(
            name=FactorName.OTP,
            score=0.0,
            weight=scoring.OTP_WEIGHT,
            verdict=FactorVerdict.FAIL,
            detail=f"All {otp.attempts_used} attempts incorrect.",
        )
        factors = behavioural + [otp_factor]
        session.factors = factors
        session.risk_score = scoring.blend_risk(factors)
        session.state = SessionState.DENIED
        session.decision = Decision.DENY
        session.next_action = NextAction.COMPLETE_DENY
        session.reason = "Step-up failed. OTP attempts exhausted."

    def _fail_closed(self, session: Session, reason: str) -> None:
        session.state = SessionState.DENIED
        session.decision = Decision.DENY
        session.next_action = NextAction.COMPLETE_DENY
        session.reason = reason

    # -------------------------------------------------------------- lifetime

    def _expire_if_due(self, session: Session) -> None:
        if session.state.is_terminal:
            return
        age = (_now() - session.created_at).total_seconds()
        if age > self.settings.session_ttl_seconds:
            session.state = SessionState.EXPIRED
            session.decision = Decision.DENY
            session.next_action = NextAction.COMPLETE_DENY
            session.reason = "Session exceeded its time to live. Failing closed."

    # ------------------------------------------------------------- responses

    def _event_response(self, session: Session, sequence: int) -> EventResponse:
        return EventResponse(
            session_id=session.session_id,
            state=session.state,
            decision=session.decision,
            risk_score=session.risk_score,
            confidence=session.confidence,
            factors=[f.model_copy(deep=True) for f in session.factors],
            next_action=session.next_action,
            reason=session.reason,
            scenario=session.scenario if self.settings.mock_mode else None,
            sequence=sequence,
            otp=session.otp.challenge(self.settings.otp_max_attempts)
            if session.otp
            else None,
            latency_ms=0,
            idempotent_replay=False,
        )

    def factors_for(self, session_id: str) -> FactorsResponse:
        session = self.store.get(session_id)
        self._expire_if_due(session)
        return FactorsResponse(
            session_id=session.session_id,
            state=session.state,
            decision=session.decision,
            risk_score=session.risk_score,
            confidence=session.confidence,
            factors=[f.model_copy(deep=True) for f in session.factors],
            next_action=session.next_action,
            reason=session.reason,
            scenario=session.scenario if self.settings.mock_mode else None,
            created_at=session.created_at,
            updated_at=session.updated_at,
            events_seen=session.events_seen,
            otp=session.otp.challenge(self.settings.otp_max_attempts)
            if session.otp
            else None,
        )


def _random_code(length: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(length))
