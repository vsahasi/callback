"""`V1Service`: every piece of /v1 state, and the operations the routes call.

The routes in detector/v1/router.py are thin: they parse, sleep the mock
jitter where the ruling says to, and call one method here. Everything lives
in one process, in memory (the Modal app is pinned to one container).

Logging duty (ruling section 2.16): log lines carry ids, codes, counts and
`tag(...)` of a room or caller identity. Never turn text or words, a `say`,
a code, a phone number, the name query, consent text, the Ramp context, a raw
room or a raw caller identity.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from callback_shared.contract import (
    ACTION_TIER,
    AUTHORIZE_SERVER_DEADLINE_S,
    AuthorizeRequest,
    CallerUpdate,
    ChallengeIssued,
    ChallengeKind,
    ChallengeState,
    ChallengeStatus,
    Channel,
    EnrollmentResult,
    EnrollmentStart,
    EnrollmentStarted,
    FactorReport,
    OtpCheckResult,
    OtpSent,
    Role,
    Scenario,
    SessionCreate,
    SessionCreated,
    SpeechEvent,
    Turn,
)
from callback_shared.contract import (
    CONTRACT_VERSION as WIRE_CONTRACT_VERSION,
)
from callback_shared.contract import (
    Decision as WireDecision,
)

from detector.analysis.dsp import build_context
from detector.analysis.voiceprint import VoicePrint, combine_prints, voiceprint_from
from detector.audio import AudioPolicy
from detector.config import Settings
from detector.enrolment import VoicePrintStore, print_from_clip, storable_enrolment
from detector.errors import DetectorError
from detector.mock import jitter_seconds
from detector.observability import StructuredLogger
from detector.v1.challenges import (
    ENROLL_HEARD,
    MAX_CHALLENGES_PER_SESSION,
    VERIFICATION_KINDS,
    ChallengeRecord,
    make_say,
    new_challenge_id,
)
from detector.v1.clips import ClipSource
from detector.v1.enrollments import (
    ConsentLog,
    EnrollmentRecord,
    new_enrollment_id,
    print_identity,
)
from detector.v1.errors import (
    ChallengeNotFound,
    ClaimMismatch,
    EnrollmentNotFound,
    IdempotencyConflict,
    InvalidScenario,
    NoOpenEnrollment,
    OtpNotSent,
    OtpNotVerified,
    OtpRateLimited,
    OtpUnavailable,
    SessionEnded,
    SessionLocked,
    SessionNotFound,
    SmsFailed,
    TooManyChallenges,
    UserNotFound,
    V1Error,
)
from detector.v1.logsafe import tag
from detector.v1.otp import OtpBackend, otp_backend_from_settings
from detector.v1.policy import (
    ChallengeView,
    SessionView,
    Verdict,
    binding_key,
    evaluate,
    internal_timeout_verdict,
    tier0_verdict,
)
from detector.v1.scenarios import (
    DEEPFAKE_WINDOWS,
    NO_AUDIO_OUTCOME,
    SCRIPTS,
    ChallengeOutcome,
    challenge_outcome,
    spoof_at,
)
from detector.v1.sessions import (
    MAX_CACHED_DECISIONS,
    MAX_DECISION_HISTORY,
    MAX_FACTOR_HISTORY,
    MAX_TURNS,
    CachedDecision,
    CallbackRecord,
    DecisionRecord,
    SessionRegistry,
    TurnRecord,
    V1Session,
    turn_key,
)
from detector.v1.users import UserDirectory

# OTP limits per session (ruling section 2.8).
OTP_SEND_INTERVAL_S = 30.0
OTP_MAX_SENDS = 3
OTP_MAX_CHECKS = 5

# Distinct version strings warned about, at most. Auth runs first, so only a
# caller holding the token can add one, but the set is bounded anyway.
_MAX_WARNED_VERSIONS = 32


def parse_scenario(raw: str | None) -> Scenario:
    """The X-Callback-Scenario value in mock mode. Absent or blank is real_cfo."""

    value = (raw or "").strip().lower()
    if not value:
        return Scenario.REAL_CFO
    try:
        return Scenario(value)
    except ValueError:
        raise InvalidScenario(
            detail="expected one of " + ", ".join(s.value for s in Scenario)
        ) from None


class V1Service:
    """The /v1 state: users, sessions, and later challenges, OTP and enrollment."""

    def __init__(
        self,
        settings: Settings,
        *,
        voiceprints: VoicePrintStore,
        clock: Callable[[], float] | None = None,
        directory: UserDirectory | None = None,
        otp: OtpBackend | None = None,
        clips: ClipSource | None = None,
        audio_policy: AudioPolicy | None = None,
        client_of: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self.settings = settings
        self.clock: Callable[[], float] = clock or time.time
        self.voiceprints = voiceprints
        # Two ways enrollment audio can arrive, and the service works with
        # either, both, or neither.
        #
        # `audio_policy` plus `client_of` read a clip the agent uploaded with
        # a turn, which is contract 1.2.0's ClipRef. The policy is the same one
        # the call path uses, including its 400 ms timeout: that budget is not
        # about this request's latency but about settling the prompt inside
        # the 700 ms a challenge waits before it resolves.
        #
        # `clips` is the hidden listener, which hears the room and needs no
        # upload. It is stretch work and None everywhere today. See
        # detector/v1/clips.py.
        self.clips = clips
        self.audio_policy = audio_policy
        self.client_of = client_of
        self.directory = (
            directory
            if directory is not None
            else UserDirectory.from_settings(settings)
        )
        self.registry = SessionRegistry(self.clock, on_evict=self._forget_session)
        self.consents = ConsentLog()
        self.enrollments: dict[str, EnrollmentRecord] = {}
        self.otp: OtpBackend = (
            otp if otp is not None else otp_backend_from_settings(settings)
        )
        self.logger: StructuredLogger | None = None
        # Authorize answers within this, jitter included (ruling 2.10).
        self.authorize_deadline_s = AUTHORIZE_SERVER_DEADLINE_S
        self._warned_versions: set[str] = set()
        self._warned_scenario_in_real_mode = False

    # ------------------------------------------------------------ plumbing

    @property
    def mock_mode(self) -> bool:
        return self.settings.mock_mode

    def bind_logger(self, logger: StructuredLogger) -> None:
        """create_app hands over the observability logger once it exists."""

        self.logger = logger

    def log(self, event: str, *, level: str = "info", **fields: Any) -> None:
        # Callers pass ids, codes, counts and tags only. The structured
        # logger's redactor is a second line, not the plan.
        if self.logger is not None:
            self.logger.log(event, level=level, **fields)

    async def jitter(self) -> None:
        """Mock-mode latency. Every route but turns, events and the poll."""

        if not self.mock_mode:
            return
        delay = jitter_seconds(
            self.settings.mock_jitter_min_ms, self.settings.mock_jitter_max_ms
        )
        if delay > 0:
            await asyncio.sleep(delay)

    def warn_version_once(self, version: str) -> None:
        if version in self._warned_versions:
            return
        if len(self._warned_versions) >= _MAX_WARNED_VERSIONS:
            return
        self._warned_versions.add(version)
        self.log(
            "v1_contract_version_differs",
            level="warning",
            presented=version[:32],
            served=WIRE_CONTRACT_VERSION,
        )

    async def aclose(self) -> None:
        await self.otp.aclose()

    # ------------------------------------------------------------ sessions

    def create_session(
        self, req: SessionCreate, scenario_header: str | None
    ) -> SessionCreated:
        if self.directory.get(req.claimed_user_id) is None:
            raise UserNotFound()
        existing = self.registry.match_room(req)
        if existing is not None:
            # A repeat create for the same live session ignores the header.
            return SessionCreated(session_id=existing.session_id)
        scenario: Scenario | None
        if self.mock_mode:
            scenario = parse_scenario(scenario_header)
        else:
            scenario = None
            if scenario_header and not self._warned_scenario_in_real_mode:
                self._warned_scenario_in_real_mode = True
                self.log(
                    "v1_scenario_header_ignored",
                    level="warning",
                    reason="real mode runs no scripts",
                )
        session = self.registry.create(req, scenario)
        self.log(
            "v1_session_created",
            session_id=session.session_id,
            room=tag(session.room),
            caller=tag(session.caller_identity),
            claimed_user_id=session.claimed_user_id,
            channel=session.channel.value,
            scenario=scenario.value if scenario else None,
            mock_mode=self.mock_mode,
        )
        return SessionCreated(session_id=session.session_id)

    def update_caller(self, session_id: str, req: CallerUpdate) -> None:
        session = self.registry.get(session_id)
        # The claim never changes: only who is on the line, and how.
        session.caller_identity = req.caller_identity
        session.channel = req.channel
        session.caller_changed_at = self.registry.now()
        self.log(
            "v1_caller_changed",
            session_id=session.session_id,
            caller=tag(req.caller_identity),
            channel=req.channel.value,
        )

    async def add_turn(self, session_id: str, turn: Turn) -> None:
        session = self.registry.get(session_id)
        key = turn_key(turn)
        if key in session.turns:
            return  # a duplicate is dropped silently
        received_at = self.registry.now()
        session.turns[key] = TurnRecord(
            role=turn.role, received_at=received_at, turn=turn
        )
        while len(session.turns) > MAX_TURNS:
            session.turns.popitem(last=False)
        if turn.role is not Role.USER:
            return
        answered = self._on_user_turn(session, received_at)
        if answered is not None:
            # Awaited rather than left to run loose, so the clip is measured
            # before this request answers 204. A challenge resolves 700 ms
            # after the turn's receive time and the agent polls after posting,
            # so settling it here is what keeps a prompt from reading `failed`
            # and then later reading `passed`.
            await self._measure_enroll_clip(session, answered, turn)

    def _on_user_turn(
        self, session: V1Session, received_at: float
    ) -> ChallengeRecord | None:
        """Bind this turn to the open challenge, and return it."""

        # Only the current challenge can be pending: a new one supersedes it.
        if session.current_challenge_id is None:
            return None
        current = session.challenges.get(session.current_challenge_id)
        if current is None:
            return None
        current.note_user_turn(received_at)
        return current

    async def _measure_enroll_clip(
        self, session: V1Session, challenge: ChallengeRecord, turn: Turn
    ) -> None:
        """Measure the caller's clip for an open enroll prompt, and pass it.

        ENROLL ONLY, and that restriction is the whole security argument. A
        `digits`, `phrase` or `liveness` challenge asks what was said and by
        whom; passing one because audio merely arrived would let anyone
        holding the service token clear a step-up by uploading any recording
        at all. An `enroll` prompt asks for neither: it records a voice, as
        `detector/v1/challenges.py` says, so "audio arrived and it is
        measurable" is the whole of its pass condition.

        Everything else here fails closed. No clip, an unfetchable clip, a
        clip with nothing measurable in it, a locked or ended session, or a
        deployment with audio switched off, and the prompt stays `no_audio`.
        A `no_audio` failure is not a voice fail (ruling F1), so a silent
        enrollment costs a retry and never a deny.
        """

        if self.mock_mode:
            return  # scripted outcomes, and mock mode writes no print
        if challenge.kind is not ChallengeKind.ENROLL:
            return
        if challenge.voice is not None:
            return  # measured already; a retry issues a new challenge
        if session.locked_code is not None or session.ended:
            return  # a denied or finished caller's voice is never recorded
        clip = turn.audio
        if clip is None or not clip.uri:
            return
        if self.audio_policy is None or not self.audio_policy.enabled:
            return

        try:
            voice, source = await print_from_clip(
                clip.uri,
                sample_rate_hz=clip.sample_rate_hz,
                encoding=clip.encoding,
                policy=self.audio_policy,
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                client=self.client_of() if self.client_of is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - a live call outlives a bad clip
            self.log(
                "v1_enroll_clip_unavailable",
                level="warning",
                session_id=session.session_id,
                challenge_id=challenge.challenge_id,
                error=type(exc).__name__,
            )
            return

        if not voice.comparable:
            self.log(
                "v1_enroll_clip_not_comparable",
                level="warning",
                session_id=session.session_id,
                challenge_id=challenge.challenge_id,
            )
            return

        if challenge.resolve(self.registry.now()).status is not ChallengeState.PENDING:
            # The fetch outlived the 700 ms resolve delay and the agent may
            # already have read this prompt as failed. An answer that has been
            # given is never taken back, so the clip is dropped.
            self.log(
                "v1_enroll_clip_too_late",
                level="warning",
                session_id=session.session_id,
                challenge_id=challenge.challenge_id,
            )
            return

        challenge.voice = voice
        challenge.outcome = ENROLL_HEARD
        self.log(
            "v1_enroll_clip_measured",
            session_id=session.session_id,
            challenge_id=challenge.challenge_id,
            prompt=challenge.enroll_prompt,
            source=source,
            voiced_seconds=round(voice.voiced_seconds, 2),
        )

    def add_event(self, session_id: str, event: SpeechEvent) -> None:
        session = self.registry.get(session_id)
        session.events.append(event)

    def report_factor(self, session_id: str, report: FactorReport) -> None:
        session = self.registry.get(session_id)
        # The latest report wins. Every report is kept for the review surface.
        record = CallbackRecord(result=report.result, at=self.registry.now())
        session.callback = record
        session.factor_history.append(record)
        del session.factor_history[:-MAX_FACTOR_HISTORY]
        self.log(
            "v1_factor_reported",
            session_id=session.session_id,
            kind=report.kind,
            result=report.result.value,
        )

    # ---------------------------------------------------------- challenges

    def _outcome_for(self, session: V1Session, kind: ChallengeKind) -> ChallengeOutcome:
        if self.mock_mode and session.scenario is not None:
            return challenge_outcome(session.scenario, kind)
        # Real mode, Round 2: no listener, so nothing can be heard.
        return NO_AUDIO_OUTCOME

    def _enroll_prompt(self, session: V1Session, now: float) -> int:
        """The prompt number the next enroll challenge asks."""

        if session.enrollment is None:
            raise NoOpenEnrollment()
        return session.enrollment.next_prompt(now)

    def create_challenge(self, session_id: str, kind: ChallengeKind) -> ChallengeIssued:
        session = self.registry.get(session_id)
        if session.locked_code is not None:
            raise SessionLocked()
        if len(session.challenges) >= MAX_CHALLENGES_PER_SESSION:
            raise TooManyChallenges()
        now = self.registry.now()
        enroll_prompt = (
            self._enroll_prompt(session, now) if kind is ChallengeKind.ENROLL else None
        )
        if session.current_challenge_id is not None:
            current = session.challenges.get(session.current_challenge_id)
            if current is not None and current.supersede(now):
                self.log(
                    "v1_challenge_superseded",
                    session_id=session.session_id,
                    challenge_id=current.challenge_id,
                )
        say, expected = make_say(kind, enroll_prompt=enroll_prompt)
        record = ChallengeRecord(
            challenge_id=new_challenge_id(),
            kind=kind,
            issued_at=now,
            outcome=self._outcome_for(session, kind),
            say=say,
            expected=expected,
            bind=session.pending_bind if kind in VERIFICATION_KINDS else None,
            enroll_prompt=enroll_prompt,
        )
        session.challenges[record.challenge_id] = record
        session.current_challenge_id = record.challenge_id
        if session.enrollment is not None and kind is ChallengeKind.ENROLL:
            session.enrollment.issued(record)
        # Never the say: it is the answer.
        self.log(
            "v1_challenge_issued",
            session_id=session.session_id,
            challenge_id=record.challenge_id,
            kind=kind.value,
            enroll_prompt=enroll_prompt,
        )
        return ChallengeIssued(
            challenge_id=record.challenge_id,
            say=say,
            expires_at=record.expires_at_ms,
        )

    def challenge_status(self, session_id: str, challenge_id: str) -> ChallengeStatus:
        session = self.registry.get(session_id)
        record = session.challenges.get(challenge_id)
        if record is None:
            raise ChallengeNotFound()
        return record.resolve(self.registry.now()).wire()

    # ----------------------------------------------------------------- OTP

    def _phone_on_file(self, session: V1Session) -> str:
        user = self.directory.get(session.claimed_user_id)
        if user is None:  # the directory is fixed at startup, so never
            raise UserNotFound()
        return user.phone_e164

    async def otp_send(self, session_id: str) -> OtpSent:
        session = self.registry.get(session_id)
        if session.locked_code is not None:
            raise SessionLocked()
        if not self.otp.available:
            raise OtpUnavailable()
        now = self.registry.now()
        record = session.otp
        if len(record.send_times) >= OTP_MAX_SENDS:
            raise OtpRateLimited(detail="at most 3 sends per session")
        if record.send_times and now - record.send_times[-1] < OTP_SEND_INTERVAL_S:
            raise OtpRateLimited(detail="at most 1 send per 30 s")
        phone = self._phone_on_file(session)
        # Reserve the slot before awaiting the provider, so a second send that
        # arrives while Twilio is answering sees it and is refused. A send that
        # failed releases it: only a code that went out counts.
        record.send_times.append(now)
        try:
            await self.otp.send(phone)
        except SmsFailed:
            record.send_times.remove(now)
            self.log(
                "v1_sms_failed",
                level="warning",
                session_id=session.session_id,
                step="send",
            )
            raise
        self.log(
            "v1_otp_sent", session_id=session.session_id, sends=len(record.send_times)
        )
        return OtpSent(sent_to_last_four=phone[-4:])

    async def otp_check(self, session_id: str, code: str) -> OtpCheckResult:
        session = self.registry.get(session_id)
        if session.locked_code is not None:
            raise SessionLocked()
        if not self.otp.available:
            raise OtpUnavailable()
        record = session.otp
        if not record.sent:
            raise OtpNotSent()
        if record.checks >= OTP_MAX_CHECKS:
            raise OtpRateLimited(detail="at most 5 checks per session")
        # Counted before the provider answers, so a failing provider cannot be
        # used to get past the check limit.
        record.checks += 1
        try:
            verified = await self.otp.check(self._phone_on_file(session), code)
        except SmsFailed:
            self.log(
                "v1_sms_failed",
                level="warning",
                session_id=session.session_id,
                step="check",
            )
            raise
        if verified:
            record.verified = True
        else:
            record.failed_checks += 1
        self.log(
            "v1_otp_checked",
            session_id=session.session_id,
            verified=verified,
            checks=record.checks,
            failed_checks=record.failed_checks,
        )
        return OtpCheckResult(verified=verified)

    # ------------------------------------------------------------ authorize

    def listening(self, session: V1Session) -> bool:
        """Real mode: whether the hidden listener hears this call.

        Always False in Round 2: the listener is stretch work (S3), so real
        mode denies tiers 1 and 2 with no_audio (Notion's DEGRADED rule).
        """

        return False

    def view(self, session: V1Session, now: float) -> SessionView:
        """A frozen picture of the session at `now`, for the policy."""

        script = (
            SCRIPTS[session.scenario]
            if self.mock_mode and session.scenario is not None
            else None
        )
        challenges = []
        for record in session.challenges.values():
            if record.kind not in VERIFICATION_KINDS:
                continue
            state = record.resolve(now)
            challenges.append(
                ChallengeView(
                    challenge_id=record.challenge_id,
                    kind=record.kind,
                    status=state.status,
                    phrase_match=state.phrase_match,
                    speaker_score=state.speaker_score,
                    reason=state.reason,
                    issued_at=record.issued_at,
                    resolved_at=state.resolved_at,
                    bind=record.bind,
                    consumed=record.consumed,
                    answered=record.answered_at is not None,
                )
            )
        return SessionView(
            session_id=session.session_id,
            mock_mode=self.mock_mode,
            listening=self.listening(session),
            locked_code=session.locked_code,
            # Real mode has no signals until the listener exists (stretch).
            voiceprint=script.voiceprint if script else None,
            spoof=(spoof_at(script.scenario, session.age_s(now)) if script else None),
            agent=script.agent if script else None,
            text=script.text if script else None,
            text_flags=script.text_flags if script else (),
            deepfake_windows=DEEPFAKE_WINDOWS,
            challenges=tuple(challenges),
            caller_changed_at=session.caller_changed_at,
            otp_sent=session.otp.sent,
            otp_verified=session.otp.verified,
            otp_failed_checks=session.otp.failed_checks,
            callback=session.callback.result if session.callback else None,
        )

    def _cached(
        self, session: V1Session, idempotency_key: str, fingerprint: str
    ) -> WireDecision | None:
        cached = session.decisions.get(idempotency_key)
        if cached is None:
            return None
        if cached.fingerprint != fingerprint:
            raise IdempotencyConflict()
        return cached.decision

    async def authorize(self, session_id: str, req: AuthorizeRequest) -> WireDecision:
        session = self.registry.get(session_id)
        fingerprint = binding_key(req.action, req.params)
        cached = self._cached(session, req.idempotency_key, fingerprint)
        if cached is not None:
            return cached  # a replay: no jitter, no new evaluation
        if ACTION_TIER[req.action] == 0:
            return self._answer_hold(session, req, fingerprint)
        try:
            async with asyncio.timeout(self.authorize_deadline_s):
                await self.jitter()
                # A same-key request may have finished during the jitter.
                cached = self._cached(session, req.idempotency_key, fingerprint)
                if cached is not None:
                    return cached
                if session.ended:  # /end arrived during the jitter
                    raise SessionEnded()
                now = self.registry.now()
                verdict = evaluate(self.view(session, now), req, now)
                self._apply(session, req.idempotency_key, fingerprint, verdict)
                return verdict.decision
        except V1Error:
            raise
        # Any failure at all denies (fail closed): TimeoutError past the
        # deadline, a bug in the policy, or a bug while recording the answer.
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(session, req, fingerprint, exc)

    def _fail_closed(
        self,
        session: V1Session,
        req: AuthorizeRequest,
        fingerprint: str,
        exc: Exception,
    ) -> WireDecision:
        """The section 2.10 deny: 200, reasons ["internal_timeout"], locked."""

        verdict = internal_timeout_verdict(session.session_id, req.action)
        # The lock first, so it holds even if the bookkeeping below fails.
        if session.locked_code is None:
            session.locked_code = verdict.lock_code
        # An answer half recorded before the failure never went out.
        if (
            session.history
            and session.history[-1].idempotency_key == req.idempotency_key
        ):
            session.history.pop()
        with contextlib.suppress(Exception):
            self.log(
                "v1_authorize_failed",
                level="error",
                session_id=session.session_id,
                action=req.action.value,
                exception=type(exc).__name__,
            )
        with contextlib.suppress(Exception):
            self._apply(session, req.idempotency_key, fingerprint, verdict)
        return verdict.decision

    def _answer_hold(
        self, session: V1Session, req: AuthorizeRequest, fingerprint: str
    ) -> WireDecision:
        """Rule 1: a hold is always allowed, so it is answered at once.

        No jitter and no deadline, so it can never time out, and no failure
        can turn it into a deny or lock the session.
        """

        now = self.registry.now()
        try:
            verdict = evaluate(self.view(session, now), req, now)
        except Exception as exc:  # noqa: BLE001
            self.log(
                "v1_authorize_failed",
                level="error",
                session_id=session.session_id,
                action=req.action.value,
                exception=type(exc).__name__,
            )
            verdict = tier0_verdict(session.session_id, req.action)
        # Recording it can fail too; the hold is allowed regardless.
        with contextlib.suppress(Exception):
            self._apply(session, req.idempotency_key, fingerprint, verdict)
        return verdict.decision

    def _apply(
        self,
        session: V1Session,
        idempotency_key: str,
        fingerprint: str,
        verdict: Verdict,
    ) -> None:
        decision = verdict.decision
        holding = ACTION_TIER[decision.action] == 0
        # A tier 0 answer never locks a session, whatever it carries.
        if (
            verdict.lock_code is not None
            and session.locked_code is None
            and not holding
        ):
            session.locked_code = verdict.lock_code
        if verdict.set_pending_bind:
            session.pending_bind = verdict.pending_bind
        if verdict.consume is not None:
            consumed = session.challenges.get(verdict.consume)
            if consumed is not None:
                consumed.consumed = True
        session.decisions[idempotency_key] = CachedDecision(
            fingerprint=fingerprint, decision=decision
        )
        while len(session.decisions) > MAX_CACHED_DECISIONS:
            session.decisions.popitem(last=False)
        session.history.append(
            DecisionRecord(
                at=self.registry.now(),
                idempotency_key=idempotency_key,
                decision=decision,
            )
        )
        del session.history[:-MAX_DECISION_HISTORY]
        # The optional detector log line (ruling C20): codes and numbers only.
        # Never the params or the Ramp context.
        self.log(
            "v1_decision",
            session_id=session.session_id,
            action=decision.action.value,
            decision=decision.decision.value,
            reason_code=decision.reasons[0] if decision.reasons else None,
            required_factors=[f.value for f in decision.required_factors] or None,
            risk_score=decision.risk_score,
            ttl_s=decision.ttl_s,
            locked=session.locked_code,
        )

    # ----------------------------------------------------------- enrollment

    def start_enrollment(self, req: EnrollmentStart) -> EnrollmentStarted:
        session = self.registry.get(req.session_id)
        if session.locked_code is not None:
            raise SessionLocked()
        if req.user_id != session.claimed_user_id or req.channel != session.channel:
            raise ClaimMismatch()
        # V4: anyone could enroll their own voice as the claimed user without it.
        if not session.otp.verified:
            raise OtpNotVerified()
        consent = self.consents.append_consent(
            user_id=req.user_id,
            channel=req.channel,
            version=req.consent.version,
            verbatim=req.consent.verbatim,
            ts_ms=req.consent.ts_ms,
        )
        if session.enrollment is not None:
            # A new start replaces the open enrollment on this session.
            self.enrollments.pop(session.enrollment.enrollment_id, None)
        record = EnrollmentRecord(
            enrollment_id=new_enrollment_id(),
            session_id=session.session_id,
            user_id=req.user_id,
            channel=req.channel,
            consent_id=consent.consent_id,
            started_at=self.registry.now(),
        )
        session.enrollment = record
        self.enrollments[record.enrollment_id] = record
        # Never the verbatim answer.
        self.log(
            "v1_enrollment_started",
            session_id=session.session_id,
            enrollment_id=record.enrollment_id,
            channel=req.channel.value,
            consent_id=consent.consent_id,
            consent_version=req.consent.version[:32],
        )
        return EnrollmentStarted(
            enrollment_id=record.enrollment_id,
            prompts_required=record.prompts_required,
        )

    def complete_enrollment(self, enrollment_id: str) -> EnrollmentResult:
        record = self.enrollments.get(enrollment_id)
        if record is None:
            raise EnrollmentNotFound()
        try:
            session = self.registry.lookup(record.session_id)
        except SessionNotFound:
            self.enrollments.pop(enrollment_id, None)
            raise EnrollmentNotFound() from None
        if record.result is not None:
            return record.result  # completing twice gives the same answer
        if session.ended:
            raise SessionEnded()
        if session.locked_code is not None:
            # A denied caller's voice is never enrolled.
            raise SessionLocked()
        session.last_call_at = self.registry.now()
        result = record.finish(self.registry.now())
        if session.enrollment is record:
            session.enrollment = None
        if result.status == "enrolled":
            self._store_enrollment_print(session, record)
        self.log(
            "v1_enrollment_completed",
            session_id=session.session_id,
            enrollment_id=record.enrollment_id,
            status=result.status,
            prompts_passed=result.prompts_passed,
        )
        return result

    def _store_enrollment_print(
        self, session: V1Session, record: EnrollmentRecord
    ) -> None:
        """Write the enrolled voice, if anything was heard.

        Ruling section 2.15: "Real-mode prints go into P3's VoicePrintStore
        under the identity '{user_id}:{channel}'. They are prints, never
        audio." And "Mock mode always enrolls and writes no print."

        Only reached when the enrollment answered `enrolled`, so consent is
        recorded, the OTP was verified, and at least five of seven prompts
        passed. That is a far stronger gate than the operator route's, which
        is why this replaces an existing print rather than conflicting with
        it: re-enrolling a vendor who has passed all of that is the intended
        flow, not an attempt to overwrite someone else's reference.

        Never raises. The ruled answer to `complete` is decided by prompts
        passed, not by whether a print landed, so a failure here cannot
        change what the agent is told. That makes it exactly the kind of
        silent gap this service tries not to have, so every path that ends
        without a stored print logs why, at `warning` when the enrollment
        said `enrolled` and a print was expected.
        """

        if self.settings.mock_mode:
            return

        def give_up(reason: str, level: str = "warning") -> None:
            self.log(
                "v1_enrollment_print_not_stored",
                level=level,
                session_id=session.session_id,
                enrollment_id=record.enrollment_id,
                reason=reason,
            )

        now = self.registry.now()
        prints: list[VoicePrint] = []
        for challenge in session.challenges.values():
            prompt = challenge.enroll_prompt
            if prompt is None or prompt not in record.passed_prompts:
                continue
            if challenge.resolve(now).status is not ChallengeState.PASSED:
                continue
            if challenge.voice is not None:
                # Measured when the turn arrived, from the clip the agent
                # uploaded with it. Already a print, so there is nothing to
                # fetch and no audio held anywhere.
                prints.append(challenge.voice)
                continue
            if self.clips is None:
                continue
            clip = self.clips.clip_for(challenge.challenge_id)
            if clip is None:
                continue
            samples, rate = clip
            try:
                prints.append(voiceprint_from(build_context(samples, rate)))
            except Exception as exc:  # noqa: BLE001 - one bad clip, not the lot
                self.log(
                    "v1_enrollment_clip_unreadable",
                    level="warning",
                    session_id=session.session_id,
                    challenge_id=challenge.challenge_id,
                    error=type(exc).__name__,
                )

        if not prints:
            if self.clips is None and self.audio_policy is None:
                # Neither way audio could arrive is configured. Not a warning:
                # it is the documented state of a deployment with no listener
                # and no audio policy, where no prompt can pass either.
                give_up("no_audio_source", level="info")
            else:
                give_up("no_clips_for_passed_prompts")
            return

        try:
            combined = combine_prints(prints)
        except ValueError:
            # Clips measured on different frequency grids. Averaging them
            # would line up different frequencies, so the voiceprint layer
            # refuses and nothing is stored.
            give_up("clips_on_different_frequency_grids")
            return

        if combined is None:
            give_up("nothing_to_combine")
            return

        enrolment = storable_enrolment(
            record.user_id,
            combined,
            source="listener",
            channel=record.channel.value,
            label=f"v1 enrollment, {len(prints)} of {record.prompts_required} prompts",
            at=datetime.fromtimestamp(now, tz=UTC),
        )
        if enrolment is None:
            # Storing it would leave voice_match inconclusive on every future
            # call while the vendor looked enrolled.
            give_up("combined_print_not_comparable")
            return

        try:
            self.voiceprints.put(enrolment, replace=True)
        except DetectorError as exc:
            # A full store, or a backend that cannot be written. The agent is
            # still told `enrolled`, because that is the ruled answer, so this
            # line is the only record that the print is missing.
            give_up(f"store_refused:{exc.message}")
            return

        self.log(
            "v1_enrollment_print_stored",
            session_id=session.session_id,
            enrollment_id=record.enrollment_id,
            channel=record.channel.value,
            quality=enrolment.quality.value,
            clips=len(prints),
            voiced_seconds=round(combined.voiced_seconds, 2),
        )

    def _forget_session(self, session: V1Session) -> None:
        """An evicted session takes its enrollments, completed ones too."""

        for enrollment_id in [
            eid
            for eid, record in self.enrollments.items()
            if record.session_id == session.session_id
        ]:
            del self.enrollments[enrollment_id]

    def delete_enrollment(self, user_id: str) -> None:
        erased = 0
        for channel in Channel:
            if self.voiceprints.delete(print_identity(user_id, channel)):
                erased += 1
        self.consents.append_revoke(
            user_id=user_id, ts_ms=round(self.registry.now() * 1000)
        )
        self.log("v1_enrollment_erased", user_id=user_id[:64], prints_erased=erased)

    # ----------------------------------------------------------------- end

    def end_session(self, session_id: str) -> None:
        session = self.registry.lookup(session_id)
        if session.ended:
            return  # idempotent
        self.registry.finish(session, at=self.registry.now(), reason="end")
        self.log("v1_session_ended", session_id=session.session_id, reason="end")
