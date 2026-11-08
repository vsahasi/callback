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
import time
from collections.abc import Callable
from typing import Any

from callback_shared.contract import (
    CONTRACT_VERSION as WIRE_CONTRACT_VERSION,
)
from callback_shared.contract import (
    CallerUpdate,
    ChallengeIssued,
    ChallengeKind,
    ChallengeStatus,
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

from detector.config import Settings
from detector.enrolment import VoicePrintStore
from detector.mock import jitter_seconds
from detector.observability import StructuredLogger
from detector.v1.challenges import (
    VERIFICATION_KINDS,
    ChallengeRecord,
    make_say,
    new_challenge_id,
)
from detector.v1.errors import (
    ChallengeNotFound,
    InvalidScenario,
    NoOpenEnrollment,
    OtpNotSent,
    OtpRateLimited,
    OtpUnavailable,
    SessionLocked,
    SmsFailed,
    UserNotFound,
)
from detector.v1.logsafe import tag
from detector.v1.otp import OtpBackend, otp_backend_from_settings
from detector.v1.scenarios import NO_AUDIO_OUTCOME, ChallengeOutcome, challenge_outcome
from detector.v1.sessions import (
    MAX_TURNS,
    CallbackRecord,
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
    ) -> None:
        self.settings = settings
        self.clock: Callable[[], float] = clock or time.time
        self.voiceprints = voiceprints
        self.directory = (
            directory
            if directory is not None
            else UserDirectory.from_settings(settings)
        )
        self.registry = SessionRegistry(self.clock)
        self.otp: OtpBackend = (
            otp if otp is not None else otp_backend_from_settings(settings)
        )
        self.logger: StructuredLogger | None = None
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

    def add_turn(self, session_id: str, turn: Turn) -> None:
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
        if turn.role is Role.USER:
            self._on_user_turn(session, received_at)

    def _on_user_turn(self, session: V1Session, received_at: float) -> None:
        # Only the current challenge can be pending: a new one supersedes it.
        if session.current_challenge_id is None:
            return
        current = session.challenges.get(session.current_challenge_id)
        if current is not None:
            current.note_user_turn(received_at)

    def add_event(self, session_id: str, event: SpeechEvent) -> None:
        session = self.registry.get(session_id)
        session.events.append(event)

    def report_factor(self, session_id: str, report: FactorReport) -> None:
        session = self.registry.get(session_id)
        # The latest report wins.
        session.callback = CallbackRecord(result=report.result, at=self.registry.now())
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

        raise NoOpenEnrollment()

    def create_challenge(self, session_id: str, kind: ChallengeKind) -> ChallengeIssued:
        session = self.registry.get(session_id)
        if session.locked_code is not None:
            raise SessionLocked()
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
        try:
            await self.otp.send(phone)
        except SmsFailed:
            self.log(
                "v1_sms_failed",
                level="warning",
                session_id=session.session_id,
                step="send",
            )
            raise
        # Only a send that went out counts toward the limits.
        record.send_times.append(now)
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

    # ----------------------------------------------------------------- end

    def end_session(self, session_id: str) -> None:
        session = self.registry.lookup(session_id)
        if session.ended:
            return  # idempotent
        self.registry.finish(session, at=self.registry.now(), reason="end")
        self.log("v1_session_ended", session_id=session.session_id, reason="end")
