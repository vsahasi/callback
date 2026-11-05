"""The session store: interface conformance, serialisation, eviction, races.

Two backends are exercised through the same conformance tests. The Modal one
runs against `FakeModalDict`, a stand-in for the slice of `modal.Dict` the store
uses, so the distributed backend is tested offline with no Modal account and no
cloud resources. There is deliberately no test against a live `modal.Dict`:
`from_name(create_if_missing=True)` would create a real cloud object, and the
store is written so that nothing touches the network until `.backing` is read.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from detector.config import Settings
from detector.errors import SequenceConflict, SessionNotFound, SessionTerminal
from detector.state import DetectorEngine
from detector.store import (
    STORE_FORMAT_VERSION,
    CachedResponse,
    InMemorySessionStore,
    ModalDictSessionStore,
    OtpState,
    Session,
    SessionStore,
    SessionVersionConflict,
    StoreFormatError,
    StoreLimits,
    StoreUnavailable,
    SweepReport,
    guard_monotonic,
    session_from_json,
    session_to_json,
)
from shared.contract import (
    MOCK_OTP_CODE,
    Decision,
    EventRequest,
    EventResponse,
    EventType,
    Factor,
    FactorName,
    FactorVerdict,
    NextAction,
    OtpChallenge,
    Scenario,
    SessionState,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SID = "store-session-0001"
WRONG_OTP = "000000"


def settings_for(
    *,
    session_ttl: int = 300,
    otp_ttl: int = 120,
    segments: int = 2,
) -> Settings:
    return Settings(
        api_token="store-test-token",
        mock_mode=True,
        session_ttl_seconds=session_ttl,
        otp_max_attempts=3,
        otp_ttl_seconds=otp_ttl,
        mock_jitter_min_ms=0,
        mock_jitter_max_ms=0,
        speech_segments_before_decision=segments,
    )


# A fixed timestamp, so that re-sending the same (session, event, sequence) is
# a byte-identical payload and therefore a real idempotent replay rather than a
# fingerprint mismatch. The engine never reads occurred_at.
OCCURRED_AT = datetime(2026, 11, 4, 21, 0, 0, tzinfo=UTC)


def event(
    session_id: str,
    kind: EventType,
    sequence: int,
    *,
    otp_code: str | None = None,
) -> EventRequest:
    return EventRequest(
        session_id=session_id,
        event=kind,
        sequence=sequence,
        occurred_at=OCCURRED_AT,
        otp_code=otp_code,
        metadata={},
    )


def push(
    engine: DetectorEngine,
    session_id: str,
    kind: EventType,
    sequence: int,
    *,
    scenario: Scenario = Scenario.REAL_CFO,
    otp_code: str | None = None,
    idempotency_key: str | None = None,
) -> EventResponse:
    return engine.handle_event(
        event(session_id, kind, sequence, otp_code=otp_code),
        scenario_header=scenario.value,
        idempotency_key=idempotency_key,
    )


# ------------------------------------------------------------- modal.Dict double


class FakeModalDict:
    """The slice of `modal.Dict` that `ModalDictSessionStore` uses.

    Two properties are faithful on purpose, because the store's safety argument
    rests on them:

      * `put(skip_if_exists=True)` is atomic, as it is server-side in Modal. A
        double that raced here would make the lease code look safer than it is.
      * values are deep-copied in and out, as they are when they cross a
        network. Anything that accidentally relies on sharing a live object
        fails here instead of in production.
    """

    def __init__(self) -> None:
        self._data: dict[Any, Any] = {}
        self._lock = threading.Lock()
        self.writes = 0

    def get(self, key: Any, default: Any | None = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            return copy.deepcopy(self._data[key])

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
        with self._lock:
            if skip_if_exists and key in self._data:
                return False
            self._data[key] = copy.deepcopy(value)
            self.writes += 1
            return True

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return self._data.pop(key, default)

    def keys(self) -> list[Any]:
        with self._lock:
            return list(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # test helpers -----------------------------------------------------------

    def raw(self, key: Any) -> Any:
        with self._lock:
            return copy.deepcopy(self._data.get(key))

    def session_docs(self) -> dict[str, Any]:
        with self._lock:
            return {
                key: copy.deepcopy(value)
                for key, value in self._data.items()
                if isinstance(key, str) and key.startswith("s:")
            }


class ConflictOnceDict(FakeModalDict):
    """Simulates another replica committing between our read and our write."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.session_reads = 0
        self.interfere_on = 2

    def arm(self, interfere_on: int = 2) -> None:
        self.armed = True
        self.session_reads = 0
        self.interfere_on = interfere_on

    def get(self, key: Any, default: Any | None = None) -> Any:
        if self.armed and isinstance(key, str) and key.startswith("s:"):
            self.session_reads += 1
            if self.session_reads == self.interfere_on:
                current = super().get(key)
                if isinstance(current, dict):
                    bumped = dict(current)
                    bumped["version"] = current["version"] + 1
                    super().put(key, bumped)
        return super().get(key, default)


class AlwaysConflictDict(FakeModalDict):
    """Every commit loses its race, forever. The store must fail closed."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def get(self, key: Any, default: Any | None = None) -> Any:
        if self.armed and isinstance(key, str) and key.startswith("s:"):
            current = super().get(key)
            if isinstance(current, dict):
                bumped = dict(current)
                bumped["version"] = current["version"] + 1
                super().put(key, bumped)
        return super().get(key, default)


# ------------------------------------------------------------------- backends

CONFORMANCE_LIMITS = StoreLimits(
    session_ttl_seconds=60,
    eviction_grace_seconds=60,
    max_cached_responses=8,
    sweep_interval_seconds=0.0,
)


def memory_store(limits: StoreLimits | None = None) -> InMemorySessionStore:
    return InMemorySessionStore(limits or CONFORMANCE_LIMITS, auto_sweep=False)


def modal_store(
    limits: StoreLimits | None = None,
    *,
    backing: FakeModalDict | None = None,
    **kwargs: Any,
) -> ModalDictSessionStore:
    return ModalDictSessionStore(
        "test-sessions",
        limits=limits or CONFORMANCE_LIMITS,
        backing=backing or FakeModalDict(),
        **kwargs,
    )


BACKENDS = {"memory": memory_store, "modal_dict": modal_store}


@pytest.fixture(params=sorted(BACKENDS), ids=sorted(BACKENDS))
def store(request: pytest.FixtureRequest) -> SessionStore:
    return BACKENDS[request.param]()


def seed(
    store: SessionStore,
    session_id: str,
    *,
    created_at: datetime,
    updated_at: datetime,
    state: SessionState = SessionState.DENIED,
    response_keys: int = 0,
    save: bool = True,
) -> Session:
    """Put a backdated session in the store, whichever backend it is."""

    session = store.new_session(session_id, Scenario.REAL_CFO, "test")
    session.created_at = created_at
    session.updated_at = updated_at
    session.state = state
    session.decision = (
        Decision.DENY if state is SessionState.DENIED else Decision.PENDING
    )
    for index in range(response_keys):
        session.responses[f"seq:{index}"] = a_cached(index, at=updated_at)
    if save:
        store.save(session)
    return session


def a_cached(index: int, *, at: datetime | None = None) -> CachedResponse:
    return CachedResponse(
        fingerprint=f"fingerprint-{index}",
        response=a_response(sequence=index),
        cached_at=at or datetime.now(UTC),
    )


def a_response(*, sequence: int = 1) -> EventResponse:
    return EventResponse(
        session_id=SID,
        state=SessionState.LISTENING,
        decision=Decision.PENDING,
        risk_score=0.0,
        confidence=0.0,
        factors=[],
        next_action=NextAction.CONTINUE_LISTENING,
        reason="stored",
        scenario=Scenario.REAL_CFO,
        sequence=sequence,
        otp=None,
        latency_ms=0,
        idempotent_replay=False,
    )


# ======================================================== interface conformance


def test_both_backends_implement_the_interface() -> None:
    assert issubclass(InMemorySessionStore, SessionStore)
    assert issubclass(ModalDictSessionStore, SessionStore)
    # Nothing abstract left over: instantiation would fail otherwise.
    assert isinstance(memory_store(), SessionStore)
    assert isinstance(modal_store(), SessionStore)


def test_get_on_a_missing_session_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound):
        store.get("nothing-here-0001")


def test_get_or_create_is_idempotent(store: SessionStore) -> None:
    first = store.get_or_create(SID, Scenario.CLONED_VOICE, "header")
    second = store.get_or_create(SID, Scenario.REAL_CFO, "default")

    assert first.session_id == SID
    # The second call must not re-pin the scenario of an existing session.
    assert second.scenario is Scenario.CLONED_VOICE
    assert second.scenario_source == "header"
    assert len(store) == 1


def test_update_creates_persists_and_returns(store: SessionStore) -> None:
    def apply(session: Session) -> str:
        session.speech_segments += 3
        session.state = SessionState.LISTENING
        return "applied"

    assert (
        store.update(SID, apply, scenario=Scenario.REPLAY, scenario_source="header")
        == "applied"
    )

    readback = store.get(SID)
    assert readback.speech_segments == 3
    assert readback.state is SessionState.LISTENING
    assert readback.scenario is Scenario.REPLAY
    assert readback.version >= 1


def test_update_without_a_scenario_needs_an_existing_session(
    store: SessionStore,
) -> None:
    with pytest.raises(SessionNotFound):
        store.update("nothing-here-0002", lambda session: None)


def test_update_commits_the_mutations_made_before_a_detector_error(
    store: SessionStore,
) -> None:
    """The whole point of the commit-on-raise rule.

    DetectorEngine marks a session denied and then raises. If the store rolled
    that back, the next event would meet a session that had never been denied.
    """

    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    def fail_closed(session: Session) -> None:
        session.state = SessionState.DENIED
        session.decision = Decision.DENY
        session.reason = "denied on the way out"
        raise SessionTerminal("no further events")

    with pytest.raises(SessionTerminal):
        store.update(SID, fail_closed)

    readback = store.get(SID)
    assert readback.state is SessionState.DENIED
    assert readback.decision is Decision.DENY
    assert readback.reason == "denied on the way out"


def test_delete_and_clear(store: SessionStore) -> None:
    store.get_or_create(SID, Scenario.REAL_CFO, "test")
    store.get_or_create("other-session-0001", Scenario.REAL_CFO, "test")
    assert len(store) == 2

    assert store.delete(SID) is True
    assert store.delete(SID) is False
    assert len(store) == 1

    store.clear()
    assert len(store) == 0


def test_a_stale_detached_copy_cannot_overwrite_a_newer_session(
    store: SessionStore,
) -> None:
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )
    stale = session_from_json(session_to_json(store.get(SID)))

    # Someone else advances the session.
    store.update(SID, lambda session: setattr(session, "speech_segments", 9))

    stale.speech_segments = 1
    with pytest.raises(SessionVersionConflict):
        store.save(stale)

    assert store.get(SID).speech_segments == 9


@pytest.mark.parametrize(
    "regressing_state",
    [SessionState.CREATED, SessionState.LISTENING, SessionState.CHALLENGING],
)
def test_a_denied_session_can_never_be_written_back_open(
    store: SessionStore, regressing_state: SessionState
) -> None:
    """The fail-closed backstop that survives a lost version check."""

    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )
    store.update(SID, lambda session: setattr(session, "state", SessionState.DENIED))

    # A copy whose version matches, so only the monotonicity rule can stop it.
    regressing = session_from_json(session_to_json(store.get(SID)))
    regressing.state = regressing_state

    with pytest.raises(SessionVersionConflict):
        store.save(regressing)

    assert store.get(SID).state is SessionState.DENIED


def test_a_deny_may_still_overwrite_an_allow(store: SessionStore) -> None:
    """Monotonicity runs one way only: more restrictive always wins."""

    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )
    store.update(SID, lambda session: setattr(session, "state", SessionState.ALLOWED))

    denied = session_from_json(session_to_json(store.get(SID)))
    denied.state = SessionState.DENIED
    store.save(denied)

    assert store.get(SID).state is SessionState.DENIED


def test_guard_monotonic_ranks_states_as_documented() -> None:
    guard_monotonic(SessionState.LISTENING, SessionState.ALLOWED, SID)
    guard_monotonic(SessionState.ALLOWED, SessionState.DENIED, SID)
    guard_monotonic(SessionState.DENIED, SessionState.EXPIRED, SID)

    with pytest.raises(SessionVersionConflict):
        guard_monotonic(SessionState.ALLOWED, SessionState.LISTENING, SID)
    with pytest.raises(SessionVersionConflict):
        guard_monotonic(SessionState.EXPIRED, SessionState.ALLOWED, SID)


def test_store_conflicts_are_fail_closed_http_conflicts() -> None:
    """A conflict must reach the agent as a non-2xx, which it reads as DENY.

    The wire contract is frozen, so there is no store-specific error code;
    SessionVersionConflict rides on sequence_conflict (409) rather than
    inventing one.
    """

    error = SessionVersionConflict("lost the race")
    assert isinstance(error, SequenceConflict)
    assert error.http_status == 409
    assert error.code.value == "sequence_conflict"


# =============================================================== serialisation


def a_fully_populated_session() -> Session:
    """Every field set to something non-default, including the awkward ones."""

    created = datetime(2026, 11, 4, 20, 59, 58, 123456, tzinfo=UTC)
    updated = datetime(2026, 11, 4, 21, 3, 2, 987654, tzinfo=UTC)
    expires = datetime(2026, 11, 4, 21, 5, 0, 500000, tzinfo=UTC)

    factors = [
        Factor(
            name=FactorName.VOICE_MATCH,
            # The rescaled weight the OTP path produces, not a round number.
            score=0.88,
            weight=0.35 * 0.6,
            verdict=FactorVerdict.PASS,
            detail="Embedding matches, which a clone is expected to do.",
        ),
        Factor(
            name=FactorName.SYNTHETIC_SPEECH,
            score=0.08,
            weight=0.15000000000000002,
            verdict=FactorVerdict.FAIL,
            detail=None,
        ),
        Factor(
            name=FactorName.OTP,
            score=1.0,
            weight=0.4,
            verdict=FactorVerdict.PASS,
            detail="Correct code on attempt 2.",
        ),
    ]

    cached = EventResponse(
        session_id=SID,
        state=SessionState.CHALLENGING,
        decision=Decision.CHALLENGE,
        risk_score=0.4199999999999999,
        confidence=0.86,
        factors=[factors[0].model_copy(deep=True)],
        next_action=NextAction.REQUEST_OTP,
        reason="Incorrect code. 2 attempt(s) remaining.",
        scenario=Scenario.CLONED_VOICE,
        sequence=4,
        otp=OtpChallenge(
            code_length=6,
            attempts_used=1,
            attempts_remaining=2,
            expires_at=expires,
        ),
        latency_ms=0,
        idempotent_replay=False,
    )
    plain = a_response(sequence=1)

    return Session(
        session_id=SID,
        scenario=Scenario.CLONED_VOICE,
        scenario_source="claimed_user_id substring",
        created_at=created,
        updated_at=updated,
        state=SessionState.CHALLENGING,
        claimed_user_id="cloned_voice-cfo",
        events_seen=7,
        speech_segments=3,
        max_sequence=6,
        factors=factors,
        risk_score=0.4199999999999999,
        confidence=0.86,
        decision=Decision.CHALLENGE,
        next_action=NextAction.REQUEST_OTP,
        reason="Voice matches but the audio shows synthesis artefacts.",
        otp=OtpState(
            code=MOCK_OTP_CODE, attempts_used=1, expires_at=expires, resolved=None
        ),
        responses={
            "seq:1": CachedResponse(
                fingerprint="fingerprint-one", response=plain, cached_at=created
            ),
            "agent-retry-1": CachedResponse(
                fingerprint="fingerprint-two", response=cached, cached_at=updated
            ),
        },
        version=11,
    )


def test_serialisation_round_trip_is_lossless() -> None:
    """The round-trip proof, checked three ways."""

    original = a_fully_populated_session()
    document = session_to_json(original)

    # 1. The document is genuinely JSON-safe, not merely dict-shaped.
    encoded = json.dumps(document)
    restored = session_from_json(json.loads(encoded))

    # 2. Field-by-field equality, including the pydantic models and datetimes.
    assert restored == original
    assert restored.otp == original.otp
    assert restored.factors == original.factors
    assert restored.responses == original.responses
    assert isinstance(restored.responses["seq:1"], CachedResponse)
    assert (
        restored.responses["agent-retry-1"].response.otp
        == original.responses["agent-retry-1"].response.otp
    )
    assert restored.factors[0].weight == original.factors[0].weight
    assert restored.risk_score == original.risk_score
    assert restored.created_at == original.created_at
    assert restored.otp is not None
    assert restored.otp.expires_at == original.otp.expires_at

    # 3. Re-serialising gives byte-identical output, so the codec has no
    #    lossy corner that happens to compare equal.
    assert json.dumps(session_to_json(restored), sort_keys=True) == json.dumps(
        document, sort_keys=True
    )


def test_round_trip_survives_a_real_engine_journey() -> None:
    """Round-trip a session the engine actually built, not a hand-made one."""

    engine = DetectorEngine(settings_for(), memory_store())
    push(engine, SID, EventType.SESSION_START, 1, scenario=Scenario.CLONED_VOICE)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2, scenario=Scenario.CLONED_VOICE)
    push(engine, SID, EventType.SPEECH_SEGMENT, 3, scenario=Scenario.CLONED_VOICE)
    push(
        engine,
        SID,
        EventType.OTP_SUBMIT,
        4,
        scenario=Scenario.CLONED_VOICE,
        otp_code=WRONG_OTP,
    )

    live = engine.store.get(SID)
    assert live.otp is not None and live.otp.attempts_used == 1

    document = session_to_json(live)
    restored = session_from_json(json.loads(json.dumps(document)))
    assert restored == live
    assert json.dumps(session_to_json(restored), sort_keys=True) == json.dumps(
        document, sort_keys=True
    )


A_FORMAT_THIS_BUILD_DOES_NOT_KNOW = STORE_FORMAT_VERSION + 998


def test_an_unreadable_document_is_refused_not_guessed_at() -> None:
    document = session_to_json(a_fully_populated_session())

    future = dict(document, format=A_FORMAT_THIS_BUILD_DOES_NOT_KNOW)
    with pytest.raises(StoreFormatError):
        session_from_json(future)

    truncated = dict(document)
    del truncated["factors"]
    with pytest.raises(StoreFormatError):
        session_from_json(truncated)

    corrupt = dict(document, state="not_a_state")
    with pytest.raises(StoreFormatError):
        session_from_json(corrupt)


def test_the_modal_backend_stores_json_documents() -> None:
    """Not just "it works": assert what actually lands in the Dict."""

    backing = FakeModalDict()
    engine = DetectorEngine(settings_for(), modal_store(backing=backing))

    push(engine, SID, EventType.SESSION_START, 1, scenario=Scenario.CLONED_VOICE)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2, scenario=Scenario.CLONED_VOICE)
    push(engine, SID, EventType.SPEECH_SEGMENT, 3, scenario=Scenario.CLONED_VOICE)
    allowed = push(
        engine,
        SID,
        EventType.OTP_SUBMIT,
        4,
        scenario=Scenario.CLONED_VOICE,
        otp_code=MOCK_OTP_CODE,
    )
    assert allowed.decision is Decision.ALLOW

    docs = backing.session_docs()
    assert list(docs) == [f"s:{SID}"]
    for document in docs.values():
        json.dumps(document)  # raises if anything non-JSON crept in

    assert engine.store.get(SID).state is SessionState.ALLOWED


# ================================================================ TTL sweeping


def test_a_session_past_ttl_and_grace_is_evicted(store: SessionStore) -> None:
    now = datetime.now(UTC)
    seed(
        store,
        SID,
        created_at=now - timedelta(seconds=400),
        updated_at=now - timedelta(seconds=400),
    )
    assert len(store) == 1

    report = store.sweep(now=now)

    assert isinstance(report, SweepReport)
    assert report.sessions_evicted == 1
    assert len(store) == 0


def test_a_session_still_being_used_is_never_evicted(store: SessionStore) -> None:
    """Past its TTL but touched a moment ago: the grace period protects it."""

    now = datetime.now(UTC)
    seed(
        store,
        SID,
        created_at=now - timedelta(seconds=400),
        updated_at=now - timedelta(seconds=1),
    )

    report = store.sweep(now=now)

    assert report.sessions_evicted == 0
    assert len(store) == 1


def test_a_young_session_is_never_evicted(store: SessionStore) -> None:
    """A session inside its TTL could still legitimately be allowed."""

    now = datetime.now(UTC)
    seed(
        store,
        SID,
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=400),
        state=SessionState.LISTENING,
    )

    report = store.sweep(now=now)

    assert report.sessions_evicted == 0
    assert len(store) == 1


def test_retention_is_always_at_least_twice_the_ttl() -> None:
    """The grace period is floored, so no TTL setting makes eviction eager."""

    for ttl in (0, 1, 60, 300, 1800):
        limits = StoreLimits.from_settings(settings_for(session_ttl=ttl))
        assert limits.session_ttl_seconds == ttl
        assert limits.eviction_grace_seconds >= ttl
        assert limits.retention_seconds >= 2 * ttl


def test_every_write_enforces_the_cache_bound(store: SessionStore) -> None:
    """The bound is a write-time invariant, not something a sweep catches up to."""

    now = datetime.now(UTC)
    seed(
        store,
        SID,
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=5),
        state=SessionState.LISTENING,
        response_keys=20,
    )

    assert len(store.get(SID).responses) == CONFORMANCE_LIMITS.max_cached_responses


def test_cache_entries_do_not_outlive_the_session_ttl(store: SessionStore) -> None:
    """Defect fix: a response cached while the call was live must not persist.

    Without this, a cached `allow` stays replayable for as long as the session
    document survives, which is well past the point the call is dead.
    """

    now = datetime.now(UTC)
    session = seed(
        store,
        SID,
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=5),
        state=SessionState.LISTENING,
        save=False,
    )
    # Inserted oldest-first, which is the real invariant: cached_at is the
    # session's updated_at, and that only moves forward.
    ttl = CONFORMANCE_LIMITS.session_ttl_seconds
    session.responses["stale"] = a_cached(1, at=now - timedelta(seconds=ttl + 1))
    session.responses["fresh"] = a_cached(2, at=now - timedelta(seconds=1))
    store.save(session)

    kept = store.get(SID).responses
    assert list(kept) == ["fresh"]


def test_sweeping_catches_an_out_of_order_stale_entry(store: SessionStore) -> None:
    """The write path trims the oldest end only; the sweep checks every entry."""

    now = datetime.now(UTC)
    session = seed(
        store,
        SID,
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=5),
        state=SessionState.LISTENING,
        save=False,
    )
    ttl = CONFORMANCE_LIMITS.session_ttl_seconds
    session.responses["fresh"] = a_cached(1, at=now - timedelta(seconds=1))
    session.responses["stale"] = a_cached(2, at=now - timedelta(seconds=ttl + 1))
    store.save(session)

    # The oldest-end pass stopped at "fresh", so "stale" is still there.
    assert list(store.get(SID).responses) == ["fresh", "stale"]

    assert store.sweep(now=now).responses_evicted == 1
    assert list(store.get(SID).responses) == ["fresh"]


def test_sweeping_trims_a_cache_that_grew_out_of_band() -> None:
    """Belt and braces: the sweep re-checks what write-time already enforces."""

    store = memory_store()
    now = datetime.now(UTC)
    session = seed(
        store,
        SID,
        created_at=now - timedelta(seconds=5),
        updated_at=now - timedelta(seconds=5),
        state=SessionState.LISTENING,
    )
    # Mutate the live object behind the store's back, as a future caller that
    # forgets to go through update() would.
    for index in range(20):
        session.responses[f"out-of-band-{index}"] = a_cached(index, at=now)

    report = store.sweep(now=now)

    assert report.sessions_evicted == 0
    assert report.responses_evicted == 20 - CONFORMANCE_LIMITS.max_cached_responses
    assert len(store.get(SID).responses) == CONFORMANCE_LIMITS.max_cached_responses


def test_losing_a_session_mid_call_fails_closed_not_open() -> None:
    """If a session does vanish, the caller loses the verdict, not the gate."""

    engine = DetectorEngine(settings_for(), memory_store())
    push(engine, SID, EventType.SESSION_START, 1)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2)
    decided = push(engine, SID, EventType.SPEECH_SEGMENT, 3)
    assert decided.decision is Decision.ALLOW

    # Simulate an over-eager eviction of a live session.
    assert engine.store.delete(SID) is True

    with pytest.raises(SessionNotFound):
        engine.factors_for(SID)

    # The next event rebuilds the session from nothing. It does not inherit the
    # allow, and no single event can re-grant one.
    resumed = push(engine, SID, EventType.SPEECH_SEGMENT, 4)
    assert resumed.decision is Decision.PENDING
    assert resumed.state is SessionState.LISTENING


def test_eviction_resets_the_challenge_budget() -> None:
    """The one cost of eviction, pinned so nobody shortens retention casually.

    Dropping a session drops its OTP attempt counter with it, so a caller who
    waits out the retention window gets a fresh set of attempts. Bounded at
    otp_max_attempts per retention window (>= 2x the session TTL, see
    test_retention_is_always_at_least_twice_the_ttl), and the reason a real
    deployment should key the attempt budget to the account rather than the
    session.
    """

    engine = DetectorEngine(settings_for(), memory_store())
    cloned = {"scenario": Scenario.CLONED_VOICE}

    push(engine, SID, EventType.SESSION_START, 1, **cloned)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2, **cloned)
    push(engine, SID, EventType.SPEECH_SEGMENT, 3, **cloned)
    for sequence in (4, 5, 6):
        last = push(
            engine, SID, EventType.OTP_SUBMIT, sequence, otp_code=WRONG_OTP, **cloned
        )
    assert last.decision is Decision.DENY
    assert engine.store.get(SID).state is SessionState.DENIED

    engine.store.delete(SID)

    push(engine, SID, EventType.SESSION_START, 1, **cloned)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2, **cloned)
    again = push(engine, SID, EventType.SPEECH_SEGMENT, 3, **cloned)
    assert again.decision is Decision.CHALLENGE
    assert again.otp is not None
    assert again.otp.attempts_remaining == 3


def test_an_expired_session_does_not_serve_its_cached_allow() -> None:
    """Defect regression: a cached `allow` must die with its session.

    The cache used to be consulted before expiry, so an identical replay of a
    successful payload kept returning decision=allow for the whole life of the
    process, long after the TTL had passed and the call had ended.
    """

    engine = DetectorEngine(settings_for(session_ttl=60), memory_store())
    key = "agent-retry-1"

    push(engine, SID, EventType.SESSION_START, 1)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2)
    allowed = push(engine, SID, EventType.SPEECH_SEGMENT, 3, idempotency_key=key)
    assert allowed.decision is Decision.ALLOW
    assert allowed.state is SessionState.ALLOWED

    # Replaying inside the TTL is a legitimate idempotent retry.
    replay = push(engine, SID, EventType.SPEECH_SEGMENT, 3, idempotency_key=key)
    assert replay.idempotent_replay is True
    assert replay.decision is Decision.ALLOW

    # Push the session past its time to live.
    engine.store.get(SID).created_at -= timedelta(seconds=600)

    with pytest.raises(SessionTerminal):
        push(engine, SID, EventType.SPEECH_SEGMENT, 3, idempotency_key=key)

    # Scope note. /events is the authorising path and it now refuses. /factors
    # is a read-only breakdown and still reports the decision this session
    # actually reached, because _expire_if_due deliberately leaves a terminal
    # session alone and changing that is a decision-semantics question, not a
    # storage one. What bounds it is retention: the record is evicted at
    # session_ttl + eviction_grace, after which /factors is a 404.
    after = engine.factors_for(SID)
    assert after.state is SessionState.ALLOWED

    aged = engine.store.get(SID)
    aged.updated_at -= timedelta(seconds=600)
    assert engine.store.sweep().sessions_evicted == 1
    with pytest.raises(SessionNotFound):
        engine.factors_for(SID)


@pytest.mark.parametrize(
    "kind",
    [
        EventType.SESSION_START,
        EventType.CALLER_CLAIM,
        EventType.SPEECH_SEGMENT,
        EventType.OTP_REQUEST,
    ],
    ids=lambda kind: kind.value,
)
def test_no_event_kind_can_replay_past_the_ttl(kind: EventType) -> None:
    """Whatever the agent replays, a session past its TTL answers with a refusal."""

    engine = DetectorEngine(settings_for(session_ttl=60), memory_store())
    session_id = f"expired-{kind.value}-01"

    first = push(engine, session_id, kind, 1)
    assert first.idempotent_replay is False

    # Inside the TTL the replay is served, which is the contract.
    assert push(engine, session_id, kind, 1).idempotent_replay is True

    engine.store.get(session_id).created_at -= timedelta(seconds=600)

    with pytest.raises(SessionTerminal):
        push(engine, session_id, kind, 1)


def test_session_end_is_the_one_event_an_aged_session_still_accepts() -> None:
    """Unchanged behaviour: the event that means "the call is over" gets through."""

    engine = DetectorEngine(settings_for(session_ttl=60), memory_store())
    push(engine, SID, EventType.SESSION_START, 1)
    engine.store.get(SID).created_at -= timedelta(seconds=600)

    ended = push(engine, SID, EventType.SESSION_END, 2)
    assert ended.decision is Decision.DENY
    assert ended.state is SessionState.EXPIRED


def test_an_aged_cache_entry_is_dropped_from_storage_too() -> None:
    """The garbage-collection half of the cache TTL."""

    limits = StoreLimits(
        session_ttl_seconds=60,
        eviction_grace_seconds=60,
        max_cached_responses=64,
        sweep_interval_seconds=0.0,
    )
    store = InMemorySessionStore(limits, auto_sweep=False)
    engine = DetectorEngine(settings_for(session_ttl=60), store)

    push(engine, SID, EventType.SESSION_START, 1)
    stored = store.get(SID)
    assert len(stored.responses) == 1

    entry = stored.responses["seq:1"]
    assert entry.is_stale(datetime.now(UTC), 60) is False
    stored.responses["seq:1"] = CachedResponse(
        fingerprint=entry.fingerprint,
        response=entry.response,
        cached_at=entry.cached_at - timedelta(seconds=600),
    )

    assert store.sweep().responses_evicted == 1
    assert store.get(SID).responses == {}


def test_rejected_requests_do_not_retain_a_session_each(store: SessionStore) -> None:
    """Defect regression: a 409 used to cost one retained session.

    `otp_submit` on an unknown session id is rejected outright. It created a
    session on the way to being rejected, so any token holder could grow the
    heap one session per rejected request with no rate limit in the way.
    """

    engine = DetectorEngine(settings_for(), store)

    for index in range(50):
        with pytest.raises(SessionTerminal):
            push(
                engine,
                f"rejected-session-{index:04d}",
                EventType.OTP_SUBMIT,
                1,
                otp_code=MOCK_OTP_CODE,
            )

    assert len(store) == 0


def test_an_invalid_scenario_header_retains_nothing(store: SessionStore) -> None:
    from detector.errors import InvalidScenario

    for index in range(10):
        with pytest.raises(InvalidScenario):
            engine = DetectorEngine(settings_for(), store)
            engine.handle_event(
                event(f"bad-scenario-{index:04d}", EventType.SESSION_START, 1),
                scenario_header="not_a_real_scenario",
                idempotency_key=None,
            )

    assert len(store) == 0


def test_a_rejection_never_discards_a_session_that_holds_evidence(
    store: SessionStore,
) -> None:
    """The discard rule is narrow: only sessions with nothing recorded go."""

    engine = DetectorEngine(settings_for(), store)

    push(engine, SID, EventType.SESSION_START, 1)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2)
    assert push(engine, SID, EventType.SPEECH_SEGMENT, 3).decision is Decision.ALLOW

    # Every further event is rejected, and the decided session must survive it.
    for sequence in range(4, 20):
        with pytest.raises(SessionTerminal):
            push(engine, SID, EventType.SPEECH_SEGMENT, sequence)

    assert len(store) == 1
    assert store.get(SID).state is SessionState.ALLOWED
    assert store.get(SID).events_seen == 3


def test_a_session_that_only_collects_rejections_still_gets_swept() -> None:
    """updated_at only advances on an accepted event, so rejections go idle."""

    store = memory_store()
    engine = DetectorEngine(settings_for(session_ttl=60), store)

    push(engine, SID, EventType.SESSION_START, 1)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2)
    push(engine, SID, EventType.SPEECH_SEGMENT, 3)

    for sequence in range(4, 10):
        with pytest.raises(SessionTerminal):
            push(engine, SID, EventType.SPEECH_SEGMENT, sequence)

    stored = store.get(SID)
    last_accepted = stored.updated_at
    assert (datetime.now(UTC) - last_accepted).total_seconds() < 5

    # Age it past retention and confirm the rejections did not keep it alive.
    stored.created_at -= timedelta(seconds=600)
    stored.updated_at -= timedelta(seconds=600)
    assert store.sweep().sessions_evicted == 1
    assert len(store) == 0


def test_auto_sweep_bounds_memory_without_anyone_calling_sweep() -> None:
    """Growth is bounded by the store itself, with no wiring in app.py."""

    limits = StoreLimits(
        session_ttl_seconds=1,
        eviction_grace_seconds=1,
        max_cached_responses=8,
        sweep_interval_seconds=30.0,
    )
    store = InMemorySessionStore(limits, auto_sweep=True)

    old = datetime.now(UTC) - timedelta(seconds=600)
    for index in range(25):
        seed(store, f"dead-session-{index:04d}", created_at=old, updated_at=old)
    assert len(store) == 25

    # Pretend the sweep interval has elapsed, then make one ordinary call.
    store._last_sweep -= limits.sweep_interval_seconds + 1
    store.update(
        "live-session-0001",
        lambda session: None,
        scenario=Scenario.REAL_CFO,
        scenario_source="header",
    )

    assert len(store) == 1
    assert list(store._sessions) == ["live-session-0001"]


# ================================================= idempotency cache bounding


def test_the_idempotency_cache_is_capped_oldest_first(store: SessionStore) -> None:
    def fill(session: Session) -> None:
        for index in range(30):
            session.responses[f"seq:{index}"] = a_cached(index)

    store.update(SID, fill, scenario=Scenario.REAL_CFO, scenario_source="header")

    kept = list(store.get(SID).responses)
    limit = CONFORMANCE_LIMITS.max_cached_responses
    assert len(kept) == limit
    # The newest keys survive; the oldest are the ones that went.
    assert kept == [f"seq:{index}" for index in range(30 - limit, 30)]


def test_a_dropped_cache_entry_becomes_a_conflict_never_an_allow() -> None:
    """The documented cost of bounding the cache, pinned.

    A late retry whose cached response has been evicted falls through to the
    monotonic-sequence guard and gets a 409. Wrong-but-safe: the agent reads any
    non-2xx as DENY, so the worst case is a refused retry, not a free pass.
    """

    limits = StoreLimits(
        session_ttl_seconds=300,
        eviction_grace_seconds=300,
        max_cached_responses=2,
        sweep_interval_seconds=0.0,
    )
    engine = DetectorEngine(
        settings_for(), InMemorySessionStore(limits, auto_sweep=False)
    )

    push(engine, SID, EventType.SESSION_START, 1)
    push(engine, SID, EventType.SPEECH_SEGMENT, 2)
    push(engine, SID, EventType.SPEECH_SEGMENT, 3)

    cached = list(engine.store.get(SID).responses)
    assert cached == ["seq:2", "seq:3"]

    # The newest retry still works, which is the retry that actually happens.
    replay = push(engine, SID, EventType.SPEECH_SEGMENT, 3)
    assert replay.idempotent_replay is True
    assert replay.decision is Decision.ALLOW

    # The evicted one degrades to a conflict, not to a fresh evaluation.
    with pytest.raises(SequenceConflict):
        push(engine, SID, EventType.SESSION_START, 1)


def test_the_cache_cannot_grow_past_the_bound_over_a_long_call() -> None:
    limits = StoreLimits(
        session_ttl_seconds=3600,
        eviction_grace_seconds=3600,
        max_cached_responses=16,
        sweep_interval_seconds=0.0,
    )
    store = InMemorySessionStore(limits, auto_sweep=False)
    engine = DetectorEngine(settings_for(segments=500), store)

    push(engine, SID, EventType.SESSION_START, 1)
    for sequence in range(2, 120):
        push(engine, SID, EventType.SPEECH_SEGMENT, sequence)
        assert len(store.get(SID).responses) <= 16

    assert len(store.get(SID).responses) == 16


# ================================================== in-memory concurrency path


def test_concurrent_updates_to_one_session_never_lose_a_write() -> None:
    store = memory_store()
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    threads_count, per_thread = 8, 50

    def bump(session: Session) -> None:
        current = session.events_seen
        # Widen the read-modify-write window so an unsynchronised store loses.
        for _ in range(50):
            pass
        session.events_seen = current + 1

    def worker() -> None:
        for _ in range(per_thread):
            store.update(SID, bump)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert store.get(SID).events_seen == threads_count * per_thread


def test_only_one_thread_is_ever_inside_one_session() -> None:
    store = memory_store()
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    inside = 0
    peak = 0
    counter_lock = threading.Lock()

    def apply(session: Session) -> None:
        nonlocal inside, peak
        with counter_lock:
            inside += 1
            peak = max(peak, inside)
        try:
            session.speech_segments += 1
        finally:
            with counter_lock:
                inside -= 1

    threads = [
        threading.Thread(target=lambda: [store.update(SID, apply) for _ in range(40)])
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak == 1, f"{peak} threads were inside the same session at once"
    assert store.get(SID).speech_segments == 6 * 40


def test_different_sessions_do_not_block_each_other() -> None:
    """Per-session locking, not one global lock."""

    store = memory_store()

    # Pick ids on distinct stripes, so the test measures the design and not a
    # hash collision. Collisions are legitimate (and harmless) at runtime, but
    # they would make this assertion flaky.
    chosen: dict[int, str] = {}
    index = 0
    while len(chosen) < 4:
        session_id = f"parallel-session-{index:04d}"
        chosen.setdefault(id(store._stripe_for(session_id)), session_id)
        index += 1

    arrived = threading.Barrier(4, timeout=5.0)

    def apply(session: Session) -> None:
        # Deadlocks (and fails on timeout) if one lock covers every session.
        arrived.wait()
        session.events_seen += 1

    threads = [
        threading.Thread(
            target=store.update,
            args=(session_id, apply),
            kwargs={"scenario": Scenario.REAL_CFO, "scenario_source": "header"},
        )
        for session_id in chosen.values()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(store) == 4


def test_racing_events_on_one_session_produce_exactly_one_winner() -> None:
    """Fail-closed under a race: duplicate sequences conflict, never double-apply."""

    engine = DetectorEngine(settings_for(), memory_store())
    accepted: list[EventResponse] = []
    conflicts: list[Exception] = []
    results_lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            response = engine.handle_event(
                event(SID, EventType.SESSION_START, 1),
                scenario_header=Scenario.REAL_CFO.value,
                idempotency_key=f"agent-retry-{index}",
            )
        except SequenceConflict as exc:
            with results_lock:
                conflicts.append(exc)
        else:
            with results_lock:
                accepted.append(response)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 1
    assert len(conflicts) == 11
    assert engine.store.get(SID).events_seen == 1


def test_the_lock_table_is_fixed_size() -> None:
    """The mutexes are striped, so they cannot become the next leak."""

    store = memory_store()
    engine = DetectorEngine(settings_for(), store)
    stripes = store.lock_stripes

    for index in range(200):
        with pytest.raises(SessionNotFound):
            engine.factors_for(f"unknown-session-{index:04d}")
    assert store.lock_stripes == stripes

    def worker(index: int) -> None:
        push(engine, f"real-session-{index:04d}", EventType.SESSION_START, 1)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(store) == 40
    assert store.lock_stripes == stripes


def test_one_session_always_maps_to_one_stripe() -> None:
    """The serialisability guarantee rests on this mapping being stable."""

    store = memory_store()
    for index in range(5000):
        session_id = f"stability-{index:06d}"
        assert store._stripe_for(session_id) is store._stripe_for(session_id)


def test_concurrent_journeys_on_separate_sessions_all_decide_correctly() -> None:
    """The realistic shape: many calls at once, one session each."""

    engine = DetectorEngine(settings_for(), memory_store())
    outcomes: dict[str, Decision] = {}
    outcomes_lock = threading.Lock()
    expected = {
        Scenario.REAL_CFO: Decision.ALLOW,
        Scenario.CLONED_VOICE: Decision.CHALLENGE,
        Scenario.REPLAY: Decision.DENY,
        Scenario.AI_AGENT: Decision.DENY,
        Scenario.HUMAN_IMPOSTOR: Decision.CHALLENGE,
    }

    def worker(index: int, scenario: Scenario) -> None:
        session_id = f"journey-{scenario.value}-{index:03d}"
        push(engine, session_id, EventType.SESSION_START, 1, scenario=scenario)
        push(engine, session_id, EventType.SPEECH_SEGMENT, 2, scenario=scenario)
        final = push(engine, session_id, EventType.SPEECH_SEGMENT, 3, scenario=scenario)
        with outcomes_lock:
            outcomes[session_id] = final.decision

    threads = [
        threading.Thread(target=worker, args=(index, scenario))
        for index in range(6)
        for scenario in expected
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outcomes) == 30
    for session_id, decision in outcomes.items():
        scenario = Scenario(session_id.split("journey-")[1].rsplit("-", 1)[0])
        assert decision is expected[scenario], session_id


# ================================================= modal.Dict concurrency path


def test_a_lease_is_taken_and_released() -> None:
    backing = FakeModalDict()
    store = modal_store(backing=backing)

    def apply(session: Session) -> None:
        # The lease exists for exactly as long as the read-modify-write.
        assert backing.raw(f"l:{SID}") is not None

    store.update(SID, apply, scenario=Scenario.REAL_CFO, scenario_source="header")
    assert backing.raw(f"l:{SID}") is None


def test_a_held_lease_makes_the_other_replica_fail_closed() -> None:
    backing = FakeModalDict()
    replica_a = modal_store(backing=backing, lock_wait_seconds=0.01)
    replica_b = modal_store(backing=backing, lock_wait_seconds=0.01)

    replica_a.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    def hold(session: Session) -> str:
        with pytest.raises(SessionVersionConflict):
            replica_b.update(SID, lambda other: None)
        return "held"

    assert replica_a.update(SID, hold) == "held"


def test_an_expired_lease_does_not_wedge_a_session_forever() -> None:
    """A replica that dies holding a lease must not lock the caller out."""

    backing = FakeModalDict()
    store = modal_store(backing=backing, lock_wait_seconds=0.01)
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    backing.put(f"l:{SID}", {"token": "dead-replica", "expires_at": 0.0})
    store.update(SID, lambda session: setattr(session, "events_seen", 4))

    assert store.get(SID).events_seen == 4


@pytest.mark.parametrize("junk", [None, "not-a-dict", {}, {"expires_at": "soon"}])
def test_an_unreadable_lease_counts_as_expired(junk: Any) -> None:
    backing = FakeModalDict()
    store = modal_store(backing=backing, lock_wait_seconds=0.01)
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )
    if junk is not None:
        backing.put(f"l:{SID}", junk)

    store.update(SID, lambda session: setattr(session, "events_seen", 7))
    assert store.get(SID).events_seen == 7


def test_a_lost_commit_race_replays_apply_against_fresh_state() -> None:
    backing = ConflictOnceDict()
    store = modal_store(backing=backing)
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    calls: list[int] = []

    def apply(session: Session) -> int:
        calls.append(session.version)
        session.events_seen = session.events_seen + 1
        return session.version

    backing.arm(interfere_on=2)
    store.update(SID, apply)

    assert len(calls) == 2, "the store should have replayed apply() once"
    assert store.get(SID).events_seen == 1


def test_an_unwinnable_race_fails_closed_after_the_retry_budget() -> None:
    backing = AlwaysConflictDict()
    store = modal_store(backing=backing, max_attempts=3, retry_backoff_seconds=0.0)
    store.update(
        SID, lambda s: None, scenario=Scenario.REAL_CFO, scenario_source="header"
    )

    backing.armed = True
    attempts: list[int] = []

    with pytest.raises(SessionVersionConflict) as caught:
        store.update(SID, lambda session: attempts.append(1))

    assert len(attempts) == 3
    assert "3 attempts" in str(caught.value)
    assert caught.value.http_status == 409


def test_an_unchanged_session_is_not_rewritten() -> None:
    """An idempotent replay must not bump the version and start a retry storm."""

    backing = FakeModalDict()
    store = modal_store(backing=backing)
    engine = DetectorEngine(settings_for(), store)

    push(engine, SID, EventType.SESSION_START, 1)
    version_before = store.get(SID).version
    writes_before = backing.writes

    replay = push(engine, SID, EventType.SESSION_START, 1)

    assert replay.idempotent_replay is True
    assert store.get(SID).version == version_before
    # Only the lease put and nothing else: the session document is untouched.
    assert backing.writes == writes_before + 1


def test_two_replicas_sharing_one_dict_agree_on_the_session() -> None:
    """The whole reason this backend exists, driven through the real engine."""

    backing = FakeModalDict()
    config = settings_for()
    replicas = [
        DetectorEngine(config, modal_store(backing=backing, lock_wait_seconds=2.0))
        for _ in range(3)
    ]

    journey = [
        EventType.SESSION_START,
        EventType.SPEECH_SEGMENT,
        EventType.SPEECH_SEGMENT,
    ]
    for index, kind in enumerate(journey):
        # Each event is handled by a different replica, as a load balancer
        # would do it.
        engine = replicas[index % len(replicas)]
        final = push(engine, SID, kind, index + 1)

    assert final.decision is Decision.ALLOW
    assert final.state is SessionState.ALLOWED

    # Every replica reads the same session out of the shared Dict.
    for engine in replicas:
        readback = engine.factors_for(SID)
        assert readback.decision is Decision.ALLOW
        assert readback.events_seen == 3


def test_replicas_racing_one_session_never_fabricate_an_allow() -> None:
    """Many replicas, one session, concurrent events. Nothing ends up open."""

    backing = FakeModalDict()
    config = settings_for()
    engines = [
        DetectorEngine(config, modal_store(backing=backing, lock_wait_seconds=2.0))
        for _ in range(4)
    ]

    accepted: list[EventResponse] = []
    refused: list[Exception] = []
    results_lock = threading.Lock()

    def worker(index: int) -> None:
        engine = engines[index % len(engines)]
        kind = EventType.SESSION_END if index % 4 == 3 else EventType.SPEECH_SEGMENT
        try:
            response = push(engine, SID, kind, index + 1)
        except (SequenceConflict, SessionTerminal) as exc:
            with results_lock:
                refused.append(exc)
        else:
            with results_lock:
                accepted.append(response)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert accepted or refused
    # No update was lost: the session saw exactly the accepted events.
    final = backing.get(f"s:{SID}")
    assert final["events_seen"] == len(accepted)

    # And whichever order they landed in, a session that reached a terminal
    # deny stayed there.
    if any(response.state is SessionState.DENIED for response in accepted):
        assert SessionState(final["state"]) in {
            SessionState.DENIED,
            SessionState.EXPIRED,
        }


# ================================================== degrading without modal


def test_constructing_the_modal_store_touches_nothing() -> None:
    """No network, no Dict, no credentials until the backing is actually used."""

    store = ModalDictSessionStore("never-opened")
    assert store.name == "never-opened"
    assert store.limits.max_cached_responses > 0


def test_the_module_works_with_modal_uninstalled() -> None:
    """Imports cleanly, and says exactly what is missing when it cannot work."""

    script = """
import sys

class BlockModal:
    def find_spec(self, name, path=None, target=None):
        if name == "modal" or name.startswith("modal."):
            raise ImportError("modal is blocked for this test")
        return None

sys.meta_path.insert(0, BlockModal())
for name in [n for n in sys.modules if n == "modal" or n.startswith("modal.")]:
    del sys.modules[name]

import detector.store as store
from detector.state import DetectorEngine

# The default backend still works with modal absent.
assert isinstance(store.InMemorySessionStore(), store.SessionStore)

handle = store.ModalDictSessionStore("unused")
try:
    handle.backing
except store.StoreUnavailable as exc:
    assert "modal" in str(exc), exc
    print("DEGRADED-CLEANLY")
else:
    raise AssertionError("expected StoreUnavailable")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        # The point of this test is to inspect the outcome, not to raise on it.
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DEGRADED-CLEANLY" in completed.stdout


def test_a_broken_backing_surfaces_as_store_unavailable() -> None:
    class NoKeys(FakeModalDict):
        def keys(self) -> list[Any]:
            raise RuntimeError("Dict is not hydrated")

    store = modal_store(backing=NoKeys())
    with pytest.raises(StoreUnavailable):
        store.sweep()
