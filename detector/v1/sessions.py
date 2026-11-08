"""The /v1 session registry (ruling section 2.6).

One process holds every session, in memory. The detector mints the id, the
claim is fixed at create, and a decision never ends a session: only `/end`,
the 30 minute cap or 10 idle minutes do. An ended session answers 409
`session_ended` for an hour and is then evicted, after which it is a 404.
Expiry is checked on every access, against the registry clock, so a fake
clock steps through it in tests.

`/end` drops turns, events, challenge `say` text and expected answers. It
keeps decisions, numeric signals and reason codes until eviction.
"""

from __future__ import annotations

import hashlib
import secrets
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from callback_shared.contract import (
    CallbackResult,
    Channel,
    Decision,
    Role,
    Scenario,
    SessionCreate,
    SpeechEvent,
    Turn,
)

from detector.v1.errors import (
    SessionConflict,
    SessionEnded,
    SessionNotFound,
    TooManySessions,
)

if TYPE_CHECKING:
    from detector.v1.challenges import ChallengeRecord
    from detector.v1.enrollments import EnrollmentRecord

MAX_LIVE_SESSIONS = 5
MAX_SESSION_AGE_S = 30 * 60
IDLE_LIMIT_S = 10 * 60
EVICT_AFTER_END_S = 60 * 60
MAX_TURNS = 200
MAX_EVENTS = 500
# The idempotency cache per session. A real call makes a handful of
# authorize requests; this only bounds a misbehaving client.
MAX_CACHED_DECISIONS = 256


def new_session_id() -> str:
    return "ses_" + secrets.token_hex(16)


def turn_key(turn: Turn) -> str:
    """The dedup key of a turn: sha1 of role, times and text."""

    raw = f"{turn.role.value}|{turn.start_ms}|{turn.end_ms}|{turn.text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class TurnRecord:
    role: Role
    received_at: float  # registry clock, epoch seconds
    turn: Turn


@dataclass
class OtpRecord:
    send_times: list[float] = field(default_factory=list)
    checks: int = 0
    failed_checks: int = 0
    verified: bool = False

    @property
    def sent(self) -> bool:
        return bool(self.send_times)


@dataclass
class CallbackRecord:
    result: CallbackResult
    at: float


@dataclass
class CachedDecision:
    fingerprint: str
    decision: Decision


@dataclass
class V1Session:
    # ids and claim
    session_id: str
    room: str
    caller_identity: str
    claimed_user_id: str
    channel: Channel
    # None in real mode, where no script runs.
    scenario: Scenario | None
    # The caller identity the session was created with. /caller changes
    # caller_identity; create stays idempotent on the original.
    created_caller_identity: str
    # timestamps, registry clock, epoch seconds
    created_at: float
    last_call_at: float
    ended_at: float | None = None
    end_reason: str | None = None
    caller_changed_at: float | None = None
    # The DenyReason of the first deny. Set once, never cleared.
    locked_code: str | None = None
    # speech, in memory only
    turns: OrderedDict[str, TurnRecord] = field(default_factory=OrderedDict)
    events: deque[SpeechEvent] = field(default_factory=lambda: deque(maxlen=MAX_EVENTS))
    # challenges, in issue order
    challenges: dict[str, ChallengeRecord] = field(default_factory=dict)
    current_challenge_id: str | None = None
    # step-up state
    otp: OtpRecord = field(default_factory=OtpRecord)
    callback: CallbackRecord | None = None
    # authorize state: the idempotency cache, keyed by idempotency_key, and the
    # binding key the next phrase challenge is stamped with
    decisions: OrderedDict[str, CachedDecision] = field(default_factory=OrderedDict)
    pending_bind: str | None = None
    enrollment: EnrollmentRecord | None = None

    @property
    def ended(self) -> bool:
        return self.ended_at is not None

    def age_s(self, now: float) -> float:
        return now - self.created_at


class SessionRegistry:
    """Every /v1 session, keyed by id, with the live room index beside it."""

    def __init__(
        self,
        clock: Callable[[], float],
        *,
        max_live: int = MAX_LIVE_SESSIONS,
        max_age_s: float = MAX_SESSION_AGE_S,
        idle_limit_s: float = IDLE_LIMIT_S,
        evict_after_s: float = EVICT_AFTER_END_S,
    ) -> None:
        self.clock = clock
        self.max_live = max_live
        self.max_age_s = max_age_s
        self.idle_limit_s = idle_limit_s
        self.evict_after_s = evict_after_s
        self._sessions: dict[str, V1Session] = {}
        self._live_by_room: dict[str, str] = {}

    # ------------------------------------------------------------ lifecycle

    def now(self) -> float:
        return self.clock()

    def _expire(self, session: V1Session, now: float) -> None:
        if session.ended_at is not None:
            return
        deadline = min(
            session.created_at + self.max_age_s,
            session.last_call_at + self.idle_limit_s,
        )
        if now >= deadline:
            self.finish(session, at=deadline, reason="expired")

    def _evict_if_due(self, session: V1Session, now: float) -> bool:
        if (
            session.ended_at is not None
            and now >= session.ended_at + self.evict_after_s
        ):
            self._sessions.pop(session.session_id, None)
            if self._live_by_room.get(session.room) == session.session_id:
                del self._live_by_room[session.room]
            return True
        return False

    def sweep(self) -> None:
        now = self.now()
        for session in list(self._sessions.values()):
            self._expire(session, now)
            self._evict_if_due(session, now)

    def finish(self, session: V1Session, *, at: float, reason: str) -> None:
        """End a session and drop what the ruling says /end drops."""

        if session.ended_at is not None:
            return
        session.ended_at = at
        session.end_reason = reason
        if self._live_by_room.get(session.room) == session.session_id:
            del self._live_by_room[session.room]
        session.turns.clear()
        session.events.clear()
        for challenge in session.challenges.values():
            challenge.forget_secrets()

    def live_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.ended_at is None)

    # --------------------------------------------------------------- create

    def match_room(self, req: SessionCreate) -> V1Session | None:
        """The live session a repeat create maps to, or None for a new one.

        Raises SessionConflict when the room is live under another claim or
        another caller.
        """

        self.sweep()
        existing_id = self._live_by_room.get(req.room)
        if existing_id is None:
            return None
        session = self._sessions[existing_id]
        if (
            session.claimed_user_id == req.claimed_user_id
            and session.created_caller_identity == req.caller_identity
        ):
            session.last_call_at = self.now()
            return session
        raise SessionConflict()

    def create(self, req: SessionCreate, scenario: Scenario | None) -> V1Session:
        self.sweep()
        if req.room in self._live_by_room:
            # match_room is called first; this only guards a misuse.
            raise SessionConflict()
        if self.live_count() >= self.max_live:
            raise TooManySessions()
        now = self.now()
        session = V1Session(
            session_id=new_session_id(),
            room=req.room,
            caller_identity=req.caller_identity,
            claimed_user_id=req.claimed_user_id,
            channel=req.channel,
            scenario=scenario,
            created_caller_identity=req.caller_identity,
            created_at=now,
            last_call_at=now,
        )
        self._sessions[session.session_id] = session
        self._live_by_room[req.room] = session.session_id
        return session

    # --------------------------------------------------------------- access

    def lookup(self, session_id: str) -> V1Session:
        """The session, live or ended. 404 for an unknown or evicted id."""

        now = self.now()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound()
        self._expire(session, now)
        if self._evict_if_due(session, now):
            raise SessionNotFound()
        return session

    def get(self, session_id: str) -> V1Session:
        """A live session, touched. 404 unknown, 409 session_ended."""

        session = self.lookup(session_id)
        if session.ended_at is not None:
            raise SessionEnded()
        session.last_call_at = self.now()
        return session

    def __len__(self) -> int:
        return len(self._sessions)
