"""Pluggable session persistence for the detector.

This module owns the persisted shape of a session (`Session`, `OtpState`), a
lossless JSON codec for it, and two backends:

    InMemorySessionStore   the default. One process, strict per-session
                           serialisation, no network. Single replica only.
    ModalDictSessionStore  a `modal.Dict` behind the same interface, for
                           running the detector on more than one replica.

Why the seam exists
-------------------
The detector is a verification service. Failing open is the worst possible
outcome, so the whole module is written around one rule: when the store cannot
be sure a write landed, it raises, and every error it raises maps to a non-2xx
response, which the reference client (`shared.client`) turns into DENY. There is
no code path here that answers "allow" out of uncertainty.

Concurrency, stated plainly
---------------------------
`update()` is the only supported way to change a session. It hands `apply` a
session, commits when `apply` returns, and commits *again* when `apply` raises a
`DetectorError`, because every error path in `detector.state` only ever moves a
session toward DENY and those moves must survive the exception.

    InMemorySessionStore
        Serialisable per session. A per-session re-entrant-free mutex is held
        for the whole read-modify-write, so two threads can never interleave on
        one session. Across processes it guarantees nothing at all, which is
        exactly why `detector/modal_app.py` pins the deployment to one
        container while this is the backend.

    ModalDictSessionStore
        Mutual exclusion while a lease is held, built on the one atomic
        primitive `modal.Dict` offers: `put(..., skip_if_exists=True)`.
        `modal.Dict` has no compare-and-swap and no transactions, so the race
        is *reduced*, not eliminated. The residual window and what bounds the
        damage are documented on the class. Short version: a lost update is
        possible; a fabricated ALLOW is not, because a stored DENIED/EXPIRED
        session can never be overwritten by a less restrictive one.

Growth
------
Both backends bound memory the same way, and both choices are deliberate:

  * Sessions are evicted only once they are past the engine's own TTL (so the
    engine has already turned them into EXPIRED/DENY) *and* idle for a further
    grace period. Evicting earlier could drop a session that is still mid-call.
  * `Session.responses` (the idempotency cache) is bounded two ways: entries
    older than the session TTL go, and then the count is capped with the
    *oldest* entry going first. Dropping the newest would break the retry most
    likely to arrive. A dropped entry costs a correct-but-unhelpful 409 on a
    late retry, never an allow.
  * A session created by a request that is then rejected outright is not
    persisted at all, as long as it holds no evidence. Keeping it would let any
    token holder grow the heap by one session per 409.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, runtime_checkable

from detector.config import Settings
from detector.errors import DetectorError, SequenceConflict, SessionNotFound
from detector.legs import (
    CrossLegCheck,
    LegPrint,
    cross_leg_from_json,
    cross_leg_to_json,
    legs_from_json,
    legs_to_json,
)
from shared.contract import (
    Decision,
    EventResponse,
    Factor,
    NextAction,
    OtpChallenge,
    Scenario,
    SessionState,
)

T = TypeVar("T")

# Bump when the persisted document shape changes. A replica that meets a
# document it does not understand refuses to serve it rather than silently
# dropping fields it cannot read.
STORE_FORMAT_VERSION = 1

# Default name of the backing modal.Dict.
DEFAULT_SESSION_DICT_NAME = "callback-detector-sessions"

# Idempotency-cache ceiling. A single call is tens of events, so this leaves
# several times the headroom a real call needs while capping the per-session
# footprint at a predictable number of stored responses.
DEFAULT_MAX_CACHED_RESPONSES = 128

# Floor for the post-TTL retention grace. Retention is always at least twice
# the session TTL (see StoreLimits.from_settings).
MIN_EVICTION_GRACE_SECONDS = 300


def _now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------- errors


class StoreUnavailable(RuntimeError):
    """The configured backend cannot be used at all.

    A deployment/configuration fault (the optional `modal` dependency is not
    installed, the named Dict cannot be opened), not a per-request outcome, so
    it is deliberately not a `DetectorError`: it surfaces as a 500, and the
    reference client treats any non-2xx as DENY.
    """


class StoreFormatError(RuntimeError):
    """A stored document this build cannot read.

    Also a 500 on purpose. Guessing at a document written by a different
    contract version is how a fail-closed service quietly starts failing open.
    """


class SessionVersionConflict(SequenceConflict):
    """Another writer owns this session right now.

    Subclasses `SequenceConflict` deliberately. The wire contract is frozen, so
    there is no store-specific error code to use, and `sequence_conflict` is
    both the closest fit and already fail-closed: HTTP 409, which
    `shared.client` converts into DENY. The message says what really happened.
    """


class _CommitConflict(Exception):
    """Internal: this commit lost a race. Retry `apply` on fresh state."""


# ------------------------------------------------------------ persisted model


@dataclass
class OtpState:
    """Live step-up challenge. Never leaves the service; `challenge()` does."""

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
class CachedResponse:
    """One idempotency-cache entry.

    `cached_at` is what lets the cache be bounded by age as well as by count.
    An entry may not outlive the session TTL: a response cached while a call was
    live must not stay replayable after that call has died, or a stale `allow`
    becomes a permanent free pass.
    """

    fingerprint: str
    response: EventResponse
    cached_at: datetime

    def is_stale(self, now: datetime, ttl_seconds: int) -> bool:
        """Whether this entry has outlived the session TTL.

        The authoritative refusal lives in `DetectorEngine.handle_event`, which
        rejects every cached reply once the *session* is past its TTL. This is
        the garbage-collection half: it keeps the cache from carrying entries
        that could never be served anyway.
        """

        if ttl_seconds <= 0:
            return False
        return (now - self.cached_at).total_seconds() > ttl_seconds


@dataclass
class Session:
    """Everything the detector knows about one phone call."""

    session_id: str
    scenario: Scenario
    scenario_source: str
    created_at: datetime
    updated_at: datetime
    state: SessionState = SessionState.CREATED
    claimed_user_id: str | None = None
    events_seen: int = 0
    speech_segments: int = 0
    #: How many audio spans the analysis engine actually scored. Lower than
    #: speech_segments whenever a span had no uri, failed to fetch or failed
    #: to decode, which is exactly the gap a reviewer needs to see.
    spans_analysed: int = 0
    #: One accumulated voice print per leg of the call, keyed by
    #: `detector.legs.CALLER` / `CALLBACK`. Prints, never audio. This is what
    #: lets the caller be compared against the person who answered the
    #: callback; see `detector.legs`.
    legs: dict[str, LegPrint] = field(default_factory=dict)
    #: The latest comparison of those two legs, once both carry enough speech.
    #: This record, not the `voice_match` factor it emits, is what makes a
    #: confirmed speaker mismatch fatal: the factor can be replaced in
    #: worst-score-wins merging, this cannot. See `detector.legs`.
    cross_leg: CrossLegCheck | None = None
    max_sequence: int = -1
    factors: list[Factor] = field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    decision: Decision = Decision.PENDING
    next_action: NextAction = NextAction.CONTINUE_LISTENING
    reason: str | None = None
    otp: OtpState | None = None
    # idempotency key -> cached response. Insertion-ordered: the store trims
    # from the front, which is the oldest end.
    responses: dict[str, CachedResponse] = field(default_factory=dict)
    # Optimistic-concurrency token, owned by the store. Bumped on every durable
    # write. Not part of the wire contract.
    version: int = 0


# ------------------------------------------------------------- serialisation


def _dt_to_json(value: datetime) -> str:
    return value.isoformat()


def _dt_from_json(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def otp_to_json(otp: OtpState) -> dict[str, Any]:
    return {
        "code": otp.code,
        "attempts_used": otp.attempts_used,
        "expires_at": _dt_to_json(otp.expires_at),
        "resolved": otp.resolved,
    }


def otp_from_json(doc: dict[str, Any]) -> OtpState:
    return OtpState(
        code=doc["code"],
        attempts_used=doc["attempts_used"],
        expires_at=_dt_from_json(doc["expires_at"]),
        resolved=doc["resolved"],
    )


def session_to_json(session: Session) -> dict[str, Any]:
    """A JSON-safe document for `session`.

    Round-trips losslessly through `session_from_json`: see
    tests/contract/test_store.py::test_serialisation_round_trip_is_lossless,
    which checks both object equality and byte-identical re-serialisation.
    """

    return {
        "format": STORE_FORMAT_VERSION,
        "version": session.version,
        "session_id": session.session_id,
        "scenario": session.scenario.value,
        "scenario_source": session.scenario_source,
        "created_at": _dt_to_json(session.created_at),
        "updated_at": _dt_to_json(session.updated_at),
        "state": session.state.value,
        "claimed_user_id": session.claimed_user_id,
        "events_seen": session.events_seen,
        "speech_segments": session.speech_segments,
        "spans_analysed": session.spans_analysed,
        "legs": legs_to_json(session.legs),
        "cross_leg": cross_leg_to_json(session.cross_leg),
        "max_sequence": session.max_sequence,
        "factors": [f.model_dump(mode="json") for f in session.factors],
        "risk_score": session.risk_score,
        "confidence": session.confidence,
        "decision": session.decision.value,
        "next_action": session.next_action.value,
        "reason": session.reason,
        "otp": otp_to_json(session.otp) if session.otp is not None else None,
        "responses": {
            key: {
                "fingerprint": entry.fingerprint,
                "response": entry.response.model_dump(mode="json"),
                "cached_at": _dt_to_json(entry.cached_at),
            }
            for key, entry in session.responses.items()
        },
    }


def session_from_json(doc: dict[str, Any]) -> Session:
    """Rebuild a `Session` from `session_to_json` output.

    Raises StoreFormatError on a document this build does not understand, so an
    unreadable record becomes a 500 (and therefore a client-side DENY) instead
    of a session with silently missing evidence.
    """

    recorded = doc.get("format")
    if recorded != STORE_FORMAT_VERSION:
        raise StoreFormatError(
            f"stored session format {recorded!r} is not "
            f"{STORE_FORMAT_VERSION!r}; refusing to interpret it"
        )

    try:
        return Session(
            session_id=doc["session_id"],
            scenario=Scenario(doc["scenario"]),
            scenario_source=doc["scenario_source"],
            created_at=_dt_from_json(doc["created_at"]),
            updated_at=_dt_from_json(doc["updated_at"]),
            state=SessionState(doc["state"]),
            claimed_user_id=doc["claimed_user_id"],
            events_seen=doc["events_seen"],
            speech_segments=doc["speech_segments"],
            # Tolerated as absent: a session written by an older replica
            # during a rolling deploy has no such key.
            spans_analysed=doc.get("spans_analysed", 0),
            # Likewise absent from a document written before the cross-leg
            # check existed. An empty dict means no leg has spoken yet, which
            # leaves the check unrun: the safe direction.
            legs=legs_from_json(doc.get("legs")),
            cross_leg=cross_leg_from_json(doc.get("cross_leg")),
            max_sequence=doc["max_sequence"],
            factors=[Factor.model_validate(f) for f in doc["factors"]],
            risk_score=doc["risk_score"],
            confidence=doc["confidence"],
            decision=Decision(doc["decision"]),
            next_action=NextAction(doc["next_action"]),
            reason=doc["reason"],
            otp=otp_from_json(doc["otp"]) if doc["otp"] is not None else None,
            responses={
                key: CachedResponse(
                    fingerprint=value["fingerprint"],
                    response=EventResponse.model_validate(value["response"]),
                    cached_at=_dt_from_json(value["cached_at"]),
                )
                for key, value in doc["responses"].items()
            },
            version=doc["version"],
        )
    except StoreFormatError:
        raise
    except Exception as exc:  # malformed record: refuse, do not improvise
        raise StoreFormatError(f"stored session is not readable: {exc}") from exc


# ---------------------------------------------------------------- fail-closed


# How restrictive each state is. A durable write may only move a session to a
# state at least as restrictive as the one already stored, so a DENY can never
# be clobbered by a stale ALLOW or by a stale "still listening".
_RESTRICTIVENESS: dict[SessionState, int] = {
    SessionState.CREATED: 0,
    SessionState.LISTENING: 0,
    SessionState.CHALLENGING: 0,
    SessionState.ALLOWED: 1,
    SessionState.DENIED: 2,
    SessionState.EXPIRED: 2,
}


def guard_monotonic(
    stored: SessionState, incoming: SessionState, session_id: str
) -> None:
    """Refuse a write that would make a session less restrictive.

    This is the backstop that holds even when the version check loses a race:
    whatever else a concurrent writer manages to do, it cannot talk a denied
    session back into a challengeable or allowed one.
    """

    if _RESTRICTIVENESS[incoming] < _RESTRICTIVENESS[stored]:
        raise SessionVersionConflict(
            f"refusing to move session {session_id!r} from {stored.value} back "
            f"to {incoming.value}; a concurrent writer already took the more "
            "restrictive decision"
        )


# --------------------------------------------------------------------- limits


@dataclass(frozen=True)
class StoreLimits:
    """Bounds on what a store is allowed to keep.

    session_ttl_seconds
        Mirrors `Settings.session_ttl_seconds`. A session older than this has
        already been turned terminal by `DetectorEngine._expire_if_due`, so it
        can no longer produce an ALLOW, which is what makes it safe to drop.
    eviction_grace_seconds
        Extra idle time required on top of the TTL before a session is dropped.
        Guards against evicting a record a retry is still about to reference.
    max_cached_responses
        Ceiling on `Session.responses`.
    sweep_interval_seconds
        Minimum wall time between automatic sweeps.
    """

    session_ttl_seconds: int = 300
    eviction_grace_seconds: int = MIN_EVICTION_GRACE_SECONDS
    max_cached_responses: int = DEFAULT_MAX_CACHED_RESPONSES
    sweep_interval_seconds: float = 60.0

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> StoreLimits:
        """Derive limits from the service settings.

        All three now come from `Settings`: CALLBACK_SESSION_TTL_SECONDS,
        CALLBACK_SESSION_EVICTION_GRACE_SECONDS and
        CALLBACK_MAX_CACHED_RESPONSES. The latter two treat 0 as "derive it",
        so the defaults above stay the behaviour unless somebody deliberately
        overrides them.

        The grace period is floored at both the TTL and the module minimum
        whatever is configured, so retention is never less than twice the TTL.
        A misconfiguration cannot shrink retention into the window where
        evicting a live session would hand a caller a fresh OTP budget.
        """

        ttl = max(0, int(settings.session_ttl_seconds))
        configured_grace = max(
            0, int(getattr(settings, "session_eviction_grace_seconds", 0))
        )
        # Floored at the TTL and at the module minimum, so a small or zero
        # override cannot shorten retention below what is safe to evict.
        grace = max(configured_grace, ttl, MIN_EVICTION_GRACE_SECONDS)
        configured_cache = max(0, int(getattr(settings, "max_cached_responses", 0)))
        resolved: dict[str, Any] = {
            "session_ttl_seconds": ttl,
            "eviction_grace_seconds": grace,
            "max_cached_responses": configured_cache or DEFAULT_MAX_CACHED_RESPONSES,
        }
        resolved.update(overrides)
        return cls(**resolved)

    @property
    def retention_seconds(self) -> int:
        return self.session_ttl_seconds + self.eviction_grace_seconds


@dataclass(frozen=True)
class SweepReport:
    """What one sweep did. Returned so ops and tests can assert on it."""

    swept_at: datetime
    scanned: int = 0
    skipped_busy: int = 0
    sessions_evicted: int = 0
    responses_evicted: int = 0


# ------------------------------------------------------------------ interface


class SessionStore(ABC):
    """Persistence seam for detector sessions.

    Read the module docstring for the concurrency guarantee each backend
    actually provides; it is not the same guarantee.

    `get()` returns the store's own representation of a session. On
    `InMemorySessionStore` that is the live object, so mutating it is visible
    immediately (`tests/contract/test_otp_edges.py` relies on this to backdate
    a session without sleeping). On `ModalDictSessionStore` it is a detached
    snapshot and mutating it changes nothing. Only `update()` is guaranteed to
    persist a change on every backend.
    """

    limits: StoreLimits

    # ------------------------------------------------------------- read/write

    @abstractmethod
    def get(self, session_id: str) -> Session:
        """The stored session. Raises SessionNotFound if there is none."""

    @abstractmethod
    def get_or_create(
        self, session_id: str, scenario: Scenario, scenario_source: str
    ) -> Session:
        """The stored session, created at `CREATED` if it does not exist."""

    @abstractmethod
    def save(self, session: Session) -> None:
        """Durably store `session`, checking its version first.

        Raises SessionVersionConflict if another writer got there first, or if
        the write would make the session less restrictive than what is stored.
        """

    @abstractmethod
    def update(
        self,
        session_id: str,
        apply: Callable[[Session], T],
        *,
        scenario: Scenario | None = None,
        scenario_source: str | None = None,
    ) -> T:
        """Read-modify-write one session and return whatever `apply` returned.

        `apply` may be called more than once: a backend that loses a commit race
        re-reads the session and replays `apply` against fresh state. It must
        therefore be a function of the session it is handed and must not carry
        state between calls.

        `apply` is committed when it returns normally *and* when it raises a
        `DetectorError`. That is not a convenience: `DetectorEngine` expires
        sessions and marks OTP failures by mutating the session and then
        raising, and those fail-closed mutations have to be durable.

        With `scenario` set the session is created if absent; without it a
        missing session raises SessionNotFound.
        """

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Drop one session. True if something was there."""

    @abstractmethod
    def sweep(self, *, now: datetime | None = None) -> SweepReport:
        """Evict dead sessions and trim oversized idempotency caches."""

    @abstractmethod
    def clear(self) -> None:
        """Drop everything. For tests and local resets."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored sessions."""

    # ------------------------------------------------------------- shared bits

    def new_session(
        self,
        session_id: str,
        scenario: Scenario,
        scenario_source: str,
        *,
        now: datetime | None = None,
    ) -> Session:
        moment = now or _now()
        return Session(
            session_id=session_id,
            scenario=scenario,
            scenario_source=scenario_source,
            created_at=moment,
            updated_at=moment,
        )

    def is_evictable(self, session: Session, now: datetime) -> bool:
        """Whether `session` may be dropped.

        Both conditions are load-bearing:

          * `age > session_ttl_seconds` means the engine has already turned this
            session terminal, so dropping it cannot convert a would-be DENY into
            an ALLOW. Dropping a younger session could: it would be re-created
            at CREATED on the next event and could then be allowed.
          * `idle > eviction_grace_seconds` means nothing has touched it for a
            while, so no retry is still expecting its cached response.
        """

        age = (now - session.created_at).total_seconds()
        idle = (now - session.updated_at).total_seconds()
        return age > self.limits.session_ttl_seconds and (
            idle > self.limits.eviction_grace_seconds
        )

    def holds_no_evidence(self, session: Session) -> bool:
        """Whether this session has recorded nothing at all.

        Used to decide that a session created by a request which was then
        rejected need not be kept. Such a session is byte-identical to the one
        the next event would create from scratch, so dropping it changes no
        decision; keeping it would let any token holder grow the heap by one
        session per rejected request.
        """

        return (
            session.events_seen == 0
            and session.max_sequence < 0
            and not session.responses
            and not session.factors
            and session.otp is None
        )

    def trim_responses(
        self,
        session: Session,
        *,
        now: datetime | None = None,
        thorough: bool = False,
    ) -> int:
        """Bound the idempotency cache by age and then by count.

        Age first. An entry may not outlive `session_ttl_seconds`, because the
        session itself is terminal by then: a response cached while the call was
        live must not stay replayable once the call is dead. This bound is
        never eager, precisely because it fires no sooner than the engine's own
        expiry does.

        Then count, oldest entry first. Oldest-first matters. The entry most
        likely to be retried is the newest one, so evicting from that end would
        manufacture conflicts out of exactly the retries the cache exists to
        absorb. Evicting from the old end means a *late* retry of an early event
        misses the cache, falls through to the monotonic-sequence guard and gets
        a 409. That is a wrong-but-safe answer: the agent sees a conflict, never
        an allow.

        This runs on every write, so the age pass walks only the oldest end and
        stops at the first entry still inside the TTL. That is correct because
        `responses` is insertion-ordered and `cached_at` is the session's
        `updated_at`, which only moves forward: stale entries are always a
        prefix. Scanning the whole cache instead costs one datetime subtraction
        per cached entry per event, which measured at roughly 70% added latency
        on the in-memory path. `sweep()` passes `thorough=True` to check every
        entry periodically, so an out-of-order insertion is still collected.
        """

        dropped = 0
        ttl = self.limits.session_ttl_seconds
        if ttl > 0 and session.responses:
            moment = now or _now()
            if thorough:
                stale = [
                    key
                    for key, entry in session.responses.items()
                    if entry.is_stale(moment, ttl)
                ]
                for key in stale:
                    del session.responses[key]
                dropped += len(stale)
            else:
                while session.responses:
                    oldest = next(iter(session.responses))
                    if not session.responses[oldest].is_stale(moment, ttl):
                        break
                    del session.responses[oldest]
                    dropped += 1

        limit = self.limits.max_cached_responses
        overflow = len(session.responses) - limit
        if limit > 0 and overflow > 0:
            for key in list(session.responses)[:overflow]:
                del session.responses[key]
            dropped += overflow

        return dropped


# ------------------------------------------------------------------ in-memory


# Number of striped mutexes guarding read-modify-write. Fixed, so the lock
# table cannot become the next unbounded leak, and allocation-free, so taking a
# session's lock costs a hash and an index rather than a dict insert and a
# matching delete on every event. The cost is that two sessions whose ids
# collide on a stripe serialise against each other for the duration of one
# event, which is a brief wait and never a correctness problem.
LOCK_STRIPES = 1024


class InMemorySessionStore(SessionStore):
    """Process-local sessions. The default, and the fast path.

    Guarantee: read-modify-write of a single session is serialisable within
    this process. `update()` holds that session's stripe for the whole of
    `apply`, so no two threads can interleave on one session and no update is
    ever lost. This is deliberately enforced in code rather than inherited from
    the fact that `handle_event` contains no `await`: that property is one edit
    away from vanishing (an `await` anywhere inside the engine, `post_events`
    declared `def` so FastAPI runs it in a threadpool, or more workers), and the
    invariant should not depend on it.

    Falls short: the guarantee ends at the process boundary. Two replicas with
    this backend each keep their own sessions, so the same call routed to two
    containers is two unrelated sessions, each with its own OTP attempt budget.
    That is why `detector/modal_app.py` pins the deployment to a single
    container while this backend is in use. Use ModalDictSessionStore to lift
    that pin.
    """

    def __init__(
        self,
        limits: StoreLimits | None = None,
        *,
        auto_sweep: bool = True,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.limits = limits or StoreLimits()
        self._sessions: dict[str, Session] = {}
        self._stripes: tuple[threading.Lock, ...] = tuple(
            threading.Lock() for _ in range(LOCK_STRIPES)
        )
        self._registry = threading.Lock()
        self._auto_sweep = auto_sweep
        self._clock = clock or _now
        self._last_sweep = time.monotonic()

    # ---------------------------------------------------------------- reads

    def get(self, session_id: str) -> Session:
        with self._registry:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound(f"no session {session_id!r}")
        return session

    def get_or_create(
        self, session_id: str, scenario: Scenario, scenario_source: str
    ) -> Session:
        return self._fetch_or_create(session_id, scenario, scenario_source)[0]

    def _fetch_or_create(
        self, session_id: str, scenario: Scenario, scenario_source: str
    ) -> tuple[Session, bool]:
        """The session, and whether this call is the one that created it."""

        with self._registry:
            session = self._sessions.get(session_id)
            if session is not None:
                return session, False
            session = self.new_session(
                session_id, scenario, scenario_source, now=self._clock()
            )
            self._sessions[session_id] = session
            return session, True

    def __len__(self) -> int:
        with self._registry:
            return len(self._sessions)

    # --------------------------------------------------------------- writes

    def save(self, session: Session) -> None:
        self.trim_responses(session)
        with self._registry:
            stored = self._sessions.get(session.session_id)
            if stored is not None and stored is not session:
                # A detached session (a snapshot, or one from another store).
                # Apply the same checks the distributed backend applies.
                guard_monotonic(stored.state, session.state, session.session_id)
                if stored.version != session.version:
                    raise SessionVersionConflict(
                        f"session {session.session_id!r} moved on from version "
                        f"{session.version} to {stored.version} while it was "
                        "being handled"
                    )
            session.version += 1
            self._sessions[session.session_id] = session

    def update(
        self,
        session_id: str,
        apply: Callable[[Session], T],
        *,
        scenario: Scenario | None = None,
        scenario_source: str | None = None,
    ) -> T:
        self._maybe_sweep()

        with self._stripe_for(session_id):
            if scenario is None:
                session = self.get(session_id)
                created = False
            else:
                session, created = self._fetch_or_create(
                    session_id, scenario, scenario_source or "default"
                )
            try:
                result = apply(session)
            except DetectorError:
                if created and self.holds_no_evidence(session):
                    # This request was rejected outright and the session it
                    # created holds nothing. Retaining it would be a free
                    # heap-growth primitive for any token holder.
                    self.delete(session_id)
                else:
                    # Fail-closed mutations made before the raise persist.
                    self.save(session)
                raise
            self.save(session)
            return result

    def delete(self, session_id: str) -> bool:
        with self._registry:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        with self._registry:
            self._sessions.clear()

    # --------------------------------------------------------------- sweeping

    def sweep(self, *, now: datetime | None = None) -> SweepReport:
        moment = now or self._clock()
        scanned = skipped = evicted = responses_evicted = 0

        with self._registry:
            session_ids = list(self._sessions)

        for session_id in session_ids:
            stripe = self._stripe_for(session_id)
            if not stripe.acquire(blocking=False):
                # Somebody is mid-update on this stripe. An active session is
                # not evictable anyway, and save() trims its cache on commit.
                # Never block a request to do housekeeping.
                skipped += 1
                continue
            try:
                with self._registry:
                    session = self._sessions.get(session_id)
                    if session is None:
                        continue
                    scanned += 1
                    if self.is_evictable(session, moment):
                        del self._sessions[session_id]
                        evicted += 1
                        continue
                responses_evicted += self.trim_responses(
                    session, now=moment, thorough=True
                )
            finally:
                stripe.release()

        self._last_sweep = time.monotonic()
        return SweepReport(
            swept_at=moment,
            scanned=scanned,
            skipped_busy=skipped,
            sessions_evicted=evicted,
            responses_evicted=responses_evicted,
        )

    def _maybe_sweep(self) -> None:
        if not self._auto_sweep:
            return
        interval = self.limits.sweep_interval_seconds
        if interval <= 0:
            return
        # One cheap float compare on the hot path.
        if time.monotonic() - self._last_sweep < interval:
            return
        self.sweep()

    # ----------------------------------------------------------- lock stripes

    def _stripe_for(self, session_id: str) -> threading.Lock:
        """The mutex guarding read-modify-write of this session.

        One session always maps to one stripe, which is what makes the
        serialisability guarantee hold. Two sessions may share a stripe, which
        costs them a brief wait and nothing else.
        """

        return self._stripes[hash(session_id) % len(self._stripes)]

    @property
    def lock_stripes(self) -> int:
        """Size of the fixed lock table. Here so tests can assert it is fixed."""

        return len(self._stripes)


# ----------------------------------------------------------------- modal dict


@runtime_checkable
class DictBacking(Protocol):
    """The slice of `modal.Dict` this store uses.

    Narrow on purpose: it is small enough for a test double to implement
    faithfully, including the atomicity of `put(skip_if_exists=True)`, so the
    distributed backend is testable with no network and no Modal account.
    """

    def get(self, key: Any, default: Any | None = None) -> Any: ...

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool: ...

    def pop(self, key: Any, default: Any = None) -> Any: ...

    def keys(self) -> Iterable[Any]: ...

    def clear(self) -> None: ...


class ModalDictSessionStore(SessionStore):
    """Sessions in a `modal.Dict`, so the service can run on many replicas.

    What it guarantees
    ------------------
    1. While a replica holds a session's lease, it has exclusive access to that
       session. The lease is taken with `modal.Dict.put(skip_if_exists=True)`,
       which is atomic server-side, so exactly one replica can create it.
    2. Every commit re-reads the stored record immediately before writing and
       refuses to write if the version moved. A losing commit replays `apply`
       against fresh state, up to `max_attempts` times.
    3. A commit may never make a session less restrictive than the record it is
       replacing (see `guard_monotonic`). DENIED and EXPIRED are sticky.
    4. Every unresolvable conflict raises `SessionVersionConflict`, which is
       HTTP 409, which `shared.client` turns into DENY.

    Where it falls short, precisely
    -------------------------------
    `modal.Dict` offers no compare-and-swap and no transactions, so the
    read-check-write in (2) is not atomic, and neither is stealing an expired
    lease. Two consequences, both real:

      * If a lease expires while its holder is still working (the holder was
        paused for longer than `lease_seconds`), another replica can steal it.
        The original holder's commit is then normally rejected by the version
        check, but if the steal lands inside the window between that holder's
        re-read and its write, both writes succeed and one update is lost.
      * Lease expiry uses wall-clock time, which is shared across replicas and
        therefore subject to clock skew between them. Skew widens the window
        above; it does not create a new one.

    This race cannot be closed with `modal.Dict` alone. What bounds it is
    guarantee (3): a lost update always leaves the *more restrictive* of the two
    states in place, so the failure direction is DENY. The states a lost update
    can cost you are "one more OTP attempt was recorded" or "one idempotency
    entry was not cached", never "a denied caller became allowed". A backend
    with real transactions (Redis WATCH/MULTI, or Postgres) would close it
    outright, and is the right next step if the detector ever needs more than
    fail-closed-under-contention.

    The whole store budget is kept under `shared.contract.CLIENT_TIMEOUT_MS`
    (800 ms) by default, because a store that takes longer than the client's
    patience just converts into a client-side DENY anyway.
    """

    def __init__(
        self,
        name: str = DEFAULT_SESSION_DICT_NAME,
        *,
        limits: StoreLimits | None = None,
        backing: DictBacking | None = None,
        create_if_missing: bool = True,
        environment_name: str | None = None,
        max_attempts: int = 3,
        lease_seconds: float = 5.0,
        lock_wait_seconds: float = 0.25,
        poll_seconds: float = 0.01,
        retry_backoff_seconds: float = 0.02,
        auto_sweep: bool = False,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self.limits = limits or StoreLimits()
        self.max_attempts = max(1, int(max_attempts))
        self.lease_seconds = float(lease_seconds)
        self.lock_wait_seconds = float(lock_wait_seconds)
        self.poll_seconds = float(poll_seconds)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._create_if_missing = create_if_missing
        self._environment_name = environment_name
        self._backing = backing
        # Sweeping walks every key, so it is off the request path by default.
        # Deploy it as a modal.Cron that calls sweep().
        self._auto_sweep = auto_sweep
        self._clock = clock or _now
        self._monotonic = monotonic or time.monotonic
        self._last_sweep = self._monotonic()

    # ------------------------------------------------------------- plumbing

    @property
    def backing(self) -> DictBacking:
        """The live Dict handle, opened on first use.

        Lazy so that importing and constructing this store never needs `modal`
        installed or a network round trip; only actually using it does.
        """

        if self._backing is None:
            self._backing = open_modal_dict(
                self.name,
                create_if_missing=self._create_if_missing,
                environment_name=self._environment_name,
            )
        return self._backing

    def _session_key(self, session_id: str) -> str:
        return f"s:{session_id}"

    def _lease_key(self, session_id: str) -> str:
        return f"l:{session_id}"

    def _read_doc(self, session_id: str) -> dict[str, Any] | None:
        doc = self.backing.get(self._session_key(session_id))
        if doc is None:
            return None
        if not isinstance(doc, dict):
            raise StoreFormatError(
                f"stored value for session {session_id!r} is "
                f"{type(doc).__name__}, not a document"
            )
        return doc

    # ---------------------------------------------------------------- reads

    def get(self, session_id: str) -> Session:
        """A detached snapshot. Mutating it persists nothing: use `update()`."""

        doc = self._read_doc(session_id)
        if doc is None:
            raise SessionNotFound(f"no session {session_id!r}")
        return session_from_json(doc)

    def get_or_create(
        self, session_id: str, scenario: Scenario, scenario_source: str
    ) -> Session:
        doc = self._read_doc(session_id)
        if doc is not None:
            return session_from_json(doc)

        fresh = self.new_session(
            session_id, scenario, scenario_source, now=self._clock()
        )
        fresh.version = 1
        created = self.backing.put(
            self._session_key(session_id), session_to_json(fresh), skip_if_exists=True
        )
        if created:
            return fresh
        # Another replica created it between the read and the put. Theirs wins.
        return self.get(session_id)

    def __len__(self) -> int:
        """O(number of keys). For ops and tests, not the request path."""

        return sum(1 for _ in self._iter_session_ids())

    # --------------------------------------------------------------- writes

    def save(self, session: Session) -> None:
        """Version-checked write of a detached session.

        Convenience for ops and tests. `update()` is what the engine uses: it
        holds a lease for the whole read-modify-write, which `save()` on its
        own cannot.
        """

        self.trim_responses(session)
        try:
            self._commit(
                session.session_id,
                session,
                None,
                expected_version=session.version or None,
            )
        except _CommitConflict as exc:
            raise SessionVersionConflict(
                f"refusing to write session {session.session_id!r}: {exc}"
            ) from exc

    def update(
        self,
        session_id: str,
        apply: Callable[[Session], T],
        *,
        scenario: Scenario | None = None,
        scenario_source: str | None = None,
    ) -> T:
        self._maybe_sweep()

        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                time.sleep(self.retry_backoff_seconds * attempt)

            lease = self._acquire(session_id)
            try:
                baseline = self._read_doc(session_id)
                if baseline is None:
                    if scenario is None:
                        raise SessionNotFound(f"no session {session_id!r}")
                    session = self.new_session(
                        session_id,
                        scenario,
                        scenario_source or "default",
                        now=self._clock(),
                    )
                else:
                    session = session_from_json(baseline)

                pending: DetectorError | None = None
                result: Any = None
                try:
                    result = apply(session)
                except DetectorError as exc:
                    # Commit first: the mutations made before this raise are
                    # the fail-closed ones (expired session, spent OTP
                    # attempt) and they have to be durable.
                    pending = exc

                if (
                    pending is not None
                    and baseline is None
                    and self.holds_no_evidence(session)
                ):
                    # Rejected outright, and nothing was ever written for it.
                    # Leave the Dict untouched rather than paying a key for a
                    # request that produced no evidence.
                    raise pending

                self.trim_responses(session)
                try:
                    self._commit(
                        session_id,
                        session,
                        baseline,
                        expected_version=(
                            None if baseline is None else baseline["version"]
                        ),
                    )
                except _CommitConflict:
                    continue  # replay apply() against fresh state

                if pending is not None:
                    raise pending
                return result  # type: ignore[return-value]
            finally:
                self._release(session_id, lease)

        raise SessionVersionConflict(
            f"gave up after {self.max_attempts} attempts to commit session "
            f"{session_id!r}; another replica keeps winning the race. Failing "
            "closed rather than guessing at the session state."
        )

    def _commit(
        self,
        session_id: str,
        session: Session,
        baseline: dict[str, Any] | None,
        *,
        expected_version: int | None,
    ) -> None:
        """Write `session`, or raise _CommitConflict if the record moved.

        `expected_version` is the version this write is allowed to replace, or
        None to require that no record exists yet. It is passed in rather than
        derived from whatever is in the Dict right now: reading the current
        document and then comparing it with itself would check nothing, which
        is precisely the mistake that makes an optimistic-concurrency scheme
        look like it works.

        `baseline`, when given, is only used to skip a write that would change
        nothing.

        The re-read here is what makes the version check mean anything: it
        closes the gap between `update()`'s read and this write down to the gap
        between these two statements. It does not close it entirely, because
        `modal.Dict` has no compare-and-swap. See the class docstring.
        """

        stored = self._read_doc(session_id)

        if stored is None:
            if expected_version is not None:
                # Swept or deleted under us. Refuse; the retry re-reads.
                raise _CommitConflict(
                    f"expected version {expected_version!r} but the record is gone"
                )
        else:
            if expected_version is None:
                raise _CommitConflict(
                    "expected no record but one exists at version "
                    f"{stored.get('version')!r}"
                )
            if stored.get("version") != expected_version:
                raise _CommitConflict(
                    f"version moved from {expected_version!r} to "
                    f"{stored.get('version')!r}"
                )
            try:
                guard_monotonic(
                    SessionState(stored["state"]), session.state, session_id
                )
            except SessionVersionConflict as exc:
                # Retryable: replaying apply() against the stored (more
                # restrictive) session is what produces the right answer, which
                # is normally a SessionTerminal for the caller.
                raise _CommitConflict(str(exc)) from exc

        payload = session_to_json(session)

        if baseline is not None and _without_version(payload) == _without_version(
            baseline
        ):
            # Nothing changed (an idempotent replay, say). Skip the write so it
            # does not bump the version and make other replicas retry.
            return

        payload["version"] = (expected_version or 0) + 1
        self.backing.put(self._session_key(session_id), payload)
        session.version = payload["version"]

    def delete(self, session_id: str) -> bool:
        existed = self.backing.pop(self._session_key(session_id), None) is not None
        self.backing.pop(self._lease_key(session_id), None)
        return existed

    def clear(self) -> None:
        self.backing.clear()

    # ---------------------------------------------------------------- leases

    def _acquire(self, session_id: str) -> str:
        """Take this session's lease, or fail closed.

        Blocks for at most `lock_wait_seconds`, which is deliberately well
        inside the client's 800 ms budget: refusing with a 409 the client reads
        as DENY beats holding the line open past the point the client has
        already given up.
        """

        key = self._lease_key(session_id)
        token = uuid.uuid4().hex
        deadline = self._monotonic() + self.lock_wait_seconds

        while True:
            record = {"token": token, "expires_at": time.time() + self.lease_seconds}
            if self.backing.put(key, record, skip_if_exists=True):
                return token

            held = self.backing.get(key)
            if held is None:
                continue  # released between the put and the read; try again

            if _lease_expired(held):
                # Steal it. Not atomic: two replicas can both believe they
                # stole the same dead lease. The version check in _commit is
                # what catches the loser, and guard_monotonic bounds the damage
                # if even that loses. See the class docstring.
                self.backing.put(key, record)
                return token

            if self._monotonic() >= deadline:
                raise SessionVersionConflict(
                    f"session {session_id!r} is being handled by another "
                    f"replica and did not free up within "
                    f"{self.lock_wait_seconds}s. Failing closed."
                )
            time.sleep(self.poll_seconds)

    def _release(self, session_id: str, token: str) -> None:
        key = self._lease_key(session_id)
        held = self.backing.get(key)
        if isinstance(held, dict) and held.get("token") != token:
            # Someone stole it while we worked. Leave theirs alone.
            return
        self.backing.pop(key, None)

    # -------------------------------------------------------------- sweeping

    def sweep(self, *, now: datetime | None = None) -> SweepReport:
        """Walk every key, evicting dead sessions and expired leases.

        Deploy this as a scheduled Modal function rather than letting it run on
        a request:

            @app.function(schedule=modal.Period(minutes=5))
            def sweep_sessions():
                ModalDictSessionStore().sweep()
        """

        moment = now or self._clock()
        scanned = skipped = evicted = responses_evicted = 0

        for session_id in list(self._iter_session_ids()):
            doc = self._read_doc(session_id)
            if doc is None:
                continue
            scanned += 1

            lease = self.backing.get(self._lease_key(session_id))
            if lease is not None and not _lease_expired(lease):
                skipped += 1  # in flight; leave it to its owner
                continue

            try:
                session = session_from_json(doc)
            except StoreFormatError:
                # Unreadable and nobody holds it. Dropping it is the only way
                # out, and a missing session is a fresh CREATED one, never an
                # allow.
                self.delete(session_id)
                evicted += 1
                continue

            if self.is_evictable(session, moment):
                self.delete(session_id)
                evicted += 1
                continue

            trimmed = self.trim_responses(session, now=moment, thorough=True)
            if trimmed:
                responses_evicted += trimmed
                # A conflict here is benign: the session is being written
                # by someone else anyway, and their own save() trims it.
                with contextlib.suppress(_CommitConflict, SessionVersionConflict):
                    self._commit(
                        session_id, session, doc, expected_version=doc["version"]
                    )

        for lease_id in list(self._iter_lease_ids()):
            lease = self.backing.get(self._lease_key(lease_id))
            if lease is not None and _lease_expired(lease):
                self.backing.pop(self._lease_key(lease_id), None)

        self._last_sweep = self._monotonic()
        return SweepReport(
            swept_at=moment,
            scanned=scanned,
            skipped_busy=skipped,
            sessions_evicted=evicted,
            responses_evicted=responses_evicted,
        )

    def _maybe_sweep(self) -> None:
        if not self._auto_sweep:
            return
        interval = self.limits.sweep_interval_seconds
        if interval <= 0:
            return
        if self._monotonic() - self._last_sweep < interval:
            return
        self.sweep()

    def _iter_session_ids(self) -> Iterator[str]:
        return self._iter_prefixed("s:")

    def _iter_lease_ids(self) -> Iterator[str]:
        return self._iter_prefixed("l:")

    def _iter_prefixed(self, prefix: str) -> Iterator[str]:
        try:
            keys = self.backing.keys()
        except Exception as exc:
            raise StoreUnavailable(
                f"could not list keys of modal.Dict {self.name!r}: {exc}"
            ) from exc
        try:
            for key in keys:
                if isinstance(key, str) and key.startswith(prefix):
                    yield key[len(prefix) :]
        except TypeError as exc:
            raise StoreUnavailable(
                f"modal.Dict {self.name!r} returned a non-iterable key listing "
                f"({type(keys).__name__}); sweep() needs synchronous keys()"
            ) from exc


def _without_version(doc: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != "version"}


def _lease_expired(record: Any) -> bool:
    """Treat anything unreadable as expired, so a bad lease cannot wedge a session."""

    if not isinstance(record, dict):
        return True
    try:
        return float(record["expires_at"]) <= time.time()
    except (KeyError, TypeError, ValueError):
        return True


def open_modal_dict(
    name: str, *, create_if_missing: bool, environment_name: str | None
) -> DictBacking:
    """Open a named `modal.Dict`, importing `modal` lazily.

    Public because `detector.enrolment` needs the same optional-dependency
    dance for the dict that holds voice prints.

    `modal` is an optional dependency (the `deploy` extra). This module imports
    and loads with it absent; only reaching for the distributed backend needs
    it, and when it is missing the message says exactly what to do.
    """

    try:
        import modal
    except ImportError as exc:
        raise StoreUnavailable(
            "The distributed stores need the optional 'modal' dependency. "
            "Install the deploy extra (uv sync --extra deploy), or keep the "
            "default in-process stores."
        ) from exc

    try:
        return modal.Dict.from_name(  # type: ignore[return-value]
            name,
            create_if_missing=create_if_missing,
            environment_name=environment_name,
        )
    except Exception as exc:
        raise StoreUnavailable(f"could not open modal.Dict {name!r}: {exc}") from exc


__all__ = [
    "DEFAULT_MAX_CACHED_RESPONSES",
    "DEFAULT_SESSION_DICT_NAME",
    "STORE_FORMAT_VERSION",
    "CachedResponse",
    "DictBacking",
    "InMemorySessionStore",
    "ModalDictSessionStore",
    "OtpState",
    "Session",
    "SessionStore",
    "SessionVersionConflict",
    "StoreFormatError",
    "StoreLimits",
    "StoreUnavailable",
    "SweepReport",
    "guard_monotonic",
    "open_modal_dict",
    "otp_from_json",
    "otp_to_json",
    "session_from_json",
    "session_to_json",
]
