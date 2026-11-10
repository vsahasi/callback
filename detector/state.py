"""Session state machine, idempotency and OTP policy.

Session storage lives behind the `SessionStore` interface in `detector.store`.
The default backend is `InMemorySessionStore`, which is process-local and
therefore single-replica; `ModalDictSessionStore` puts the same sessions in a
`modal.Dict` so the service can run on more than one container. Nothing in this
module knows which one it is talking to.

The engine mutates a session and then raises on every fail-closed path (TTL
expiry, a spent OTP attempt, an exhausted challenge). `SessionStore.update()`
commits on the way out of those raises, so a session that has been denied stays
denied no matter which replica handled the event. Read the `detector.store`
module docstring for the guarantee each backend actually provides.

Two modes, one difference in lifecycle. In mock mode the engine's decision is
the verdict the dev API publishes, and a terminal session refuses every event
but `session_end`. In real mode `/v1` decides and reads the engine's factors
through `detector.bridge`, so an allowed session keeps hearing speech for as
long as it lives: see `DetectorEngine._still_hears`.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from detector import scoring
from detector.config import Settings
from detector.errors import (
    DetectorError,
    InvalidScenario,
    OtpExhausted,
    OtpExpired,
    SequenceConflict,
    SessionTerminal,
)
from detector.evidence import (
    NO_AUDIO_REASON,
    Evidence,
    analysed_evidence,
    empty_evidence,
    scripted_evidence,
)
from detector.mock import resolve_scenario
from detector.store import (
    CachedResponse,
    InMemorySessionStore,
    OtpState,
    Session,
    SessionStore,
    StoreLimits,
)
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
    SessionState,
)

# Re-exported so `from detector.state import Session` keeps working for anyone
# who already imports it from here.
__all__ = [
    "CachedResponse",
    "DetectorEngine",
    "InMemorySessionStore",
    "OtpState",
    "Session",
    "SessionStore",
]


def _now() -> datetime:
    return datetime.now(UTC)


def _fingerprint(request: EventRequest) -> str:
    return hashlib.sha256(
        request.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()


class DetectorEngine:
    """Applies events to sessions and produces contract responses."""

    def __init__(self, settings: Settings, store: SessionStore | None = None) -> None:
        self.settings = settings
        # `store if store is not None`, never `store or ...`: a SessionStore
        # defines __len__, so an empty one is falsy and `or` would silently
        # swap an injected store for a private in-memory one.
        self.store = (
            store
            if store is not None
            else InMemorySessionStore(StoreLimits.from_settings(settings))
        )

    # ---------------------------------------------------------------- events

    def handle_event(
        self,
        request: EventRequest,
        *,
        scenario_header: str | None,
        idempotency_key: str | None,
        observed_factors: list[Factor] | None = None,
        audio_note: str | None = None,
    ) -> EventResponse:
        """Apply one event and return the decision it produces.

        `observed_factors` is real mode's input: the factors the route got by
        fetching and analysing this event's audio span. None means no span
        was analysable, which is not the same as a clean span, and mock mode
        ignores it entirely. `audio_note` says why, in words, when a span
        produced nothing. Both are keywords with defaults so every existing
        caller, and every mock-mode caller, is unaffected.
        """

        try:
            scenario, source = resolve_scenario(
                scenario_header, request.claimed_user_id
            )
        except ValueError as exc:
            raise InvalidScenario(str(exc)) from exc

        cache_key = idempotency_key or f"seq:{request.sequence}"
        fingerprint = _fingerprint(request)

        def apply(session: Session) -> EventResponse:
            """One read-modify-write of this session.

            The store may replay this against fresher state if it loses a
            commit race, so it reads everything it needs from `session` and
            keeps no state of its own. Replaying is also the behaviour we want:
            a retry re-checks the idempotency cache and the sequence guard
            against whatever another replica has already committed.
            """

            # An explicit header on a later event re-pins the scenario, so a
            # fixture can drive one session through a specific script end to end.
            if scenario_header:
                session.scenario = scenario
                session.scenario_source = source

            # Expiry is settled before the idempotency cache is consulted, not
            # after. Two things depend on that order.
            #
            # Checking it first is what stops a response cached while the call
            # was live from staying replayable after the call is over. Note that
            # _expire_if_due deliberately leaves a terminal session alone, so an
            # ALLOWED session never becomes EXPIRED and the state check alone
            # would not catch it: the guard below is written against the
            # session's age, which covers both. Without it, an identical replay
            # of a successful payload kept answering `allow` for as long as the
            # record survived.
            self._expire_if_due(session)

            if self._past_ttl(session) and request.event is not EventType.SESSION_END:
                raise SessionTerminal(
                    f"session is {session.state.value} and older than its "
                    f"{self.settings.session_ttl_seconds}s time to live; it "
                    "accepts no further events, cached replies included"
                )

            cached = session.responses.get(cache_key)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise SequenceConflict(
                        f"idempotency key {cache_key!r} was already used with a "
                        "different payload"
                    )
                return cached.response.model_copy(update={"idempotent_replay": True})

            if request.sequence <= session.max_sequence:
                raise SequenceConflict(
                    f"sequence {request.sequence} is not greater than the last "
                    f"accepted sequence {session.max_sequence}"
                )

            if session.state.is_terminal and not self._still_hears(session, request):
                raise SessionTerminal(
                    f"session is {session.state.value} and accepts no further events"
                )

            self._apply(session, request, observed_factors, audio_note)

            session.events_seen += 1
            session.max_sequence = request.sequence
            session.updated_at = _now()

            response = self._event_response(session, request.sequence)
            session.responses[cache_key] = CachedResponse(
                fingerprint=fingerprint,
                response=response,
                cached_at=session.updated_at,
            )
            return response

        return self.store.update(
            request.session_id,
            apply,
            scenario=scenario,
            scenario_source=source,
        )

    # ------------------------------------------------------------ transitions

    def _still_hears(self, session: Session, request: EventRequest) -> bool:
        """Whether a terminal session still takes this event.

        `session_end` always, so a call that is over can say so. And in real
        mode, speech on a session the engine has allowed. The voice agent
        posts one segment per caller turn for the whole call, hangs up, dials
        the number on file and posts the callback leg under the same id, while
        `/v1 .../authorize` reads `factors_for` through `detector.bridge` and
        decides. With the default two segments before a decision, an allow
        that closed the session refused every later turn with 409, so the
        callback leg was never analysed and the factors `/v1` read were frozen
        at the second span. In real mode the engine's allow is evidence, not
        the verdict, so the evidence keeps coming in: factors keep merging
        worst-of, and a later fatal span flips the session to deny.

        Mock mode is unchanged: its allow is the published dev-API verdict,
        and the integration kit expects a terminal session to refuse. A denied
        session refuses in both modes, because nothing it could hear would
        lift the deny.
        """

        if request.event is EventType.SESSION_END:
            return True
        return (
            not self.settings.mock_mode
            and session.state is SessionState.ALLOWED
            and request.event is EventType.SPEECH_SEGMENT
        )

    def _apply(
        self,
        session: Session,
        request: EventRequest,
        observed_factors: list[Factor] | None = None,
        audio_note: str | None = None,
    ) -> None:
        if request.claimed_user_id:
            session.claimed_user_id = request.claimed_user_id

        match request.event:
            case EventType.SESSION_START:
                if session.state is not SessionState.CREATED:
                    # A reconnecting agent may re-send session_start. It must
                    # not move the session backwards: doing so cleared an
                    # outstanding step-up and reported continue_listening for
                    # a caller that had already failed voice analysis.
                    return
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
                if observed_factors:
                    session.spans_analysed += 1
                evidence = self._evidence_for(session, observed_factors, audio_note)
                if session.state is SessionState.CHALLENGING:
                    # Keep analysing. Discarding speech here left the factor
                    # list empty, so a correct code could be accepted on a
                    # session whose voice was never examined. A fatal signal
                    # arriving now still denies outright.
                    self._record_evidence(session, evidence)
                    return
                if session.state is SessionState.ALLOWED:
                    # Real mode only; _still_hears refused this in mock mode.
                    # The allow has been published, so there is no step-up to
                    # offer: a span that would have asked for a code now
                    # denies instead, and a clean one leaves the allow
                    # standing with the factors brought up to date.
                    self._evaluate(session, evidence, step_up=False)
                    return
                if (
                    session.speech_segments
                    < self.settings.speech_segments_before_decision
                ):
                    # Hold the evidence, do not decide yet. Returning here
                    # without storing it meant a span analysed before the
                    # threshold was measured and then forgotten, so a caller
                    # who played a recording first and spoke second was
                    # scored only on the speaking. The next segment merges
                    # these in and the worst score per signal wins.
                    #
                    # The fatal rule is deliberately not applied here.
                    # speech_segments_before_decision is a promise to the
                    # agent about when a verdict arrives, and one more
                    # segment reaches the same deny.
                    #
                    # Real mode only. Mock mode has nothing to accumulate,
                    # because the script is a pure function of the scenario
                    # and is re-read on every event. Storing it here would
                    # publish the scripted verdict a segment early, which the
                    # integration kit would see.
                    if not self.settings.mock_mode:
                        session.factors = evidence.factors
                        session.risk_score = scoring.blend_risk(evidence.factors)
                        session.confidence = evidence.confidence
                    session.decision = Decision.PENDING
                    session.next_action = NextAction.CONTINUE_LISTENING
                    session.reason = (
                        f"{session.speech_segments} of "
                        f"{self.settings.speech_segments_before_decision} segments gathered."
                    )
                    return
                self._evaluate(session, evidence)

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

    def _evidence_for(
        self,
        session: Session,
        observed: list[Factor] | None,
        audio_note: str | None = None,
    ) -> Evidence:
        """Scripted in mock mode, measured in real mode.

        In real mode `observed` is the factor list the route produced by
        analysing this event's audio span, merged here with whatever earlier
        spans already established. None means no span was analysable, which
        is not the same as a clean span and must not be read as one.
        """

        if self.settings.mock_mode:
            return scripted_evidence(session.scenario)
        if observed:
            return analysed_evidence(
                observed, session.factors, spans=session.spans_analysed
            )
        if session.factors:
            # Nothing new, but earlier spans did produce evidence. Re-derive
            # from what is already on the session rather than throwing it
            # away, which would turn a denied call back into an unscored one.
            return analysed_evidence(
                session.factors, None, spans=session.spans_analysed
            )
        return empty_evidence(
            f"{NO_AUDIO_REASON} {audio_note}" if audio_note else NO_AUDIO_REASON
        )

    def _evaluate(
        self, session: Session, evidence: Evidence, *, step_up: bool = True
    ) -> None:
        """Turn the evidence into a decision and move the session.

        `step_up=False` is the re-evaluation of a session that has already
        allowed: the middle band and a diluted FAIL fall closed to deny, the
        way `scoring.decide` already treats an exhausted OTP, because a code
        cannot be requested after the agent has heard complete_allow.
        """

        script = evidence
        factors = evidence.factors
        session.confidence = evidence.confidence
        otp_available = script.otp_can_rescue and step_up

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
        decision = scoring.decide(risk, otp_available=otp_available)

        # A failing signal is never waved through on a good average. Weights
        # are renormalised over whichever detectors produced a score, so one
        # FAIL can be diluted below the allow threshold by passing factors
        # around it. Step up where that is possible, refuse where it is not.
        failing = scoring.first_failing_factor(factors)
        if decision is Decision.ALLOW and failing is not None:
            decision = Decision.CHALLENGE if otp_available else Decision.DENY

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

    def _record_evidence(self, session: Session, evidence: Evidence) -> None:
        """Populate factors while a step-up is already open.

        Evidence must not be discarded just because a challenge is
        outstanding. A fatal signal arriving now denies outright, and a correct
        code later has to be judged against real analysis rather than an empty
        factor list.
        """

        script = evidence
        factors = evidence.factors
        session.confidence = evidence.confidence
        session.factors = factors
        session.risk_score = scoring.blend_risk(factors)

        fatal = scoring.fatal_factor(factors)
        if fatal is not None:
            session.state = SessionState.DENIED
            session.decision = Decision.DENY
            session.next_action = NextAction.COMPLETE_DENY
            session.reason = (
                f"{fatal.name.value} is unambiguous attack evidence "
                f"(score {fatal.score:.2f}). {script.reason}"
            )

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
            raise SessionTerminal("no active OTP challenge; send otp_request first")

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
            factors = [*behavioural, otp_factor]
            session.factors = factors
            session.risk_score = scoring.blend_risk(factors)

            # A correct code moves the score; it does not override policy.
            # This used to set ALLOW unconditionally, ignoring the
            # fatal-factor rule, ignoring scenarios that must never be
            # rescued, and ignoring the score it had just computed. Combined
            # with speech being discarded while CHALLENGING, that allowed a
            # replay attack through on an empty factor list.
            # Mock mode asks the scenario whether a code may rescue the
            # caller. Real mode derives it from what was measured, which is
            # the same question asked of evidence instead of a script. This
            # used to read SCRIPTS unconditionally, so a real-mode call would
            # have been judged against whichever scenario happened to be
            # pinned to the session.
            script = (
                scripted_evidence(session.scenario)
                if self.settings.mock_mode
                else analysed_evidence(behavioural, None, spans=session.spans_analysed)
            )
            fatal = scoring.fatal_factor(behavioural)
            refusal: str | None = None
            if not behavioural:
                refusal = (
                    "No voice analysis was performed, so a correct code alone "
                    "cannot verify this caller."
                )
            elif fatal is not None:
                refusal = (
                    f"{fatal.name.value} is unambiguous attack evidence "
                    f"(score {fatal.score:.2f}), which a correct code does "
                    "not rescue."
                )
            elif not script.otp_can_rescue:
                refusal = script.reason
            elif (
                scoring.decide(session.risk_score, otp_available=False)
                is not Decision.ALLOW
            ):
                refusal = (
                    "Step-up passed but blended risk "
                    f"{session.risk_score:.3f} is still above the allow "
                    "threshold."
                )

            if refusal is not None:
                session.state = SessionState.DENIED
                session.decision = Decision.DENY
                session.next_action = NextAction.COMPLETE_DENY
                session.reason = refusal
                return

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
            session.reason = f"Incorrect code. {remaining} attempt(s) remaining."
            return

        otp.resolved = False
        otp_factor = Factor(
            name=FactorName.OTP,
            score=0.0,
            weight=scoring.OTP_WEIGHT,
            verdict=FactorVerdict.FAIL,
            detail=f"All {otp.attempts_used} attempts incorrect.",
        )
        factors = [*behavioural, otp_factor]
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

    def _past_ttl(self, session: Session, now: datetime | None = None) -> bool:
        """Whether this session has outlived its configured time to live.

        Deliberately independent of `state`: a session that reached a terminal
        decision still ages, and `handle_event` uses that to stop serving its
        cached responses once the call is over.
        """

        age = ((now or _now()) - session.created_at).total_seconds()
        return age > self.settings.session_ttl_seconds

    def _expire_if_due(self, session: Session) -> None:
        if session.state.is_terminal:
            return
        if self._past_ttl(session):
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

    def claimed_identity(self, session_id: str) -> str | None:
        """Who this call's caller says they are, if it has been recorded.

        A plain read, deliberately not through `update()`: unlike
        `factors_for` this cannot change the session, and the caller is the
        audio pipeline looking for an enrolment reference mid-call, not a
        client asking about a session. A session that does not exist yet, or a
        store that cannot answer, returns None, which leaves `voice_match`
        unrun and the decision one signal poorer. That is the safe direction.
        """

        try:
            session = self.store.get(session_id)
        except DetectorError:
            return None
        return session.claimed_user_id

    def factors_for(self, session_id: str) -> FactorsResponse:
        def apply(session: Session) -> FactorsResponse:
            # Goes through update() rather than get() because _expire_if_due can
            # move the session to EXPIRED/DENY, and a read that notices a
            # session has died must make that durable.
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

        return self.store.update(session_id, apply)


def _random_code(length: int = 6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(length))
