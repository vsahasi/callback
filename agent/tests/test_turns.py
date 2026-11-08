import asyncio
import inspect
import itertools
import json
import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager

import httpx
import pytest
import respx
from callback_shared.contract import Role, SpeechEvent, SpeechEventType, Turn, Word

import callback_agent.turns as turns_module
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.turns import (
    EpochAnchor,
    FinalSegment,
    Forwarder,
    build_agent_turn,
    build_user_turn,
    speech_event,
)

# Three distinct items to forward. A's text stands in for a spoken code: it must never reach a log.
A = Turn(role=Role.USER, text="my code is 482913", start_ms=1_000, end_ms=2_000)
B = SpeechEvent(type=SpeechEventType.AGENT_SPEECH_START, ts_ms=2_100)
C = Turn(role=Role.AGENT, text="Thank you.", start_ms=2_100, end_ms=2_900)

FOUR_TWO = FinalSegment(
    "four two", 0.40, 0.90, (("four", 0.42, 0.56), ("two", 0.60, 0.88)), t0=1000.0
)


# --- epoch ms ---------------------------------------------------------------------------------


def test_anchor_uses_the_first_frame_only():
    a = EpochAnchor()
    a.observe_frame(0.02, 1000.02)
    a.observe_frame(0.02, 1005.00)
    assert a.t0 == pytest.approx(1000.0) and a.to_ms(1.5) == 1001500


def test_anchor_has_no_time_before_the_first_frame():
    a = EpochAnchor()
    assert a.t0 is None
    with pytest.raises(ValueError, match="frame"):
        a.to_ms(1.0)


def test_user_turn_words_are_epoch_ms():
    turn = build_user_turn([FOUR_TWO], interrupted=False)
    assert (turn.role, turn.start_ms, turn.end_ms) == (Role.USER, 1000400, 1000900)
    assert turn.words == [
        Word(w="four", start_ms=1000420, end_ms=1000560),
        Word(w="two", start_ms=1000600, end_ms=1000880),
    ]
    assert (turn.text, turn.interrupted) == ("four two", False)


def test_user_turn_merges_segments():
    seven = FinalSegment("seven", 2.10, 2.50, (("seven", 2.12, 2.48),), t0=1000.0)
    turn = build_user_turn([FOUR_TWO, seven], interrupted=True)
    assert turn.text == "four two seven"  # joined with a space
    assert (turn.start_ms, turn.end_ms) == (1000400, 1002500)  # start of the first, end of the last
    assert [w.w for w in turn.words] == ["four", "two", "seven"]
    assert turn.words[-1] == Word(w="seven", start_ms=1002120, end_ms=1002480)
    assert turn.interrupted is True


def test_each_segment_is_placed_with_its_own_t0():
    # A handoff restarts the STT clock, so two segments of one turn can hang off different anchors.
    later = FinalSegment("seven", 0.10, 0.30, (("seven", 0.12, 0.28),), t0=1005.0)
    turn = build_user_turn([FOUR_TWO, later], interrupted=False)
    assert (turn.start_ms, turn.end_ms) == (1000400, 1005300)
    assert [(w.start_ms, w.end_ms) for w in turn.words] == [
        (1000420, 1000560),
        (1000600, 1000880),
        (1005120, 1005280),
    ]


def test_times_stay_exact_at_current_epoch_magnitudes():
    # Epoch seconds are about 1.8e9, where a float resolves 0.2 microseconds: ms must not drift.
    seg = FinalSegment("four", 0.40, 0.60, (("four", 0.42, 0.56),), t0=1_793_000_000.0)
    turn = build_user_turn([seg], interrupted=False)
    assert (turn.start_ms, turn.end_ms) == (1_793_000_000_400, 1_793_000_000_600)
    assert turn.words == [Word(w="four", start_ms=1_793_000_000_420, end_ms=1_793_000_000_560)]


def test_user_turn_needs_a_segment():
    with pytest.raises(ValueError, match="segment"):
        build_user_turn([], interrupted=False)


def test_agent_turn_has_no_words():
    turn = build_agent_turn("Thank you.", 1000.25, 1002.5, interrupted=True)
    assert turn.words == []
    assert turn == Turn(
        role=Role.AGENT,
        text="Thank you.",
        start_ms=1000250,
        end_ms=1002500,
        words=[],
        interrupted=True,
    )


def test_speech_event_is_epoch_ms_rounded_to_the_nearest():
    event = speech_event(SpeechEventType.USER_SPEECH_START, 1_793_000_000.4216)
    assert event == SpeechEvent(type=SpeechEventType.USER_SPEECH_START, ts_ms=1_793_000_000_422)


# --- the forwarder ----------------------------------------------------------------------------


class FakeDetector:
    """Stands in for DetectorClient.

    `script` has one entry per call, in order: an exception to raise, or None to succeed. Calls past
    its end succeed. A test can assign an Event to `gate` to hold every call until it sets it.
    """

    def __init__(self, script: Iterable[Exception | None] = ()) -> None:
        self._script = iter(script)
        self.attempts: list[tuple[str, Turn | SpeechEvent]] = []  # every call, failed or not
        self.posted: list[tuple[str, Turn | SpeechEvent]] = []  # the calls that went through
        self.methods: list[str] = []  # which method each of those calls used
        self.gate: asyncio.Event | None = None

    async def post_turn(self, session_id: str, turn: Turn) -> None:
        await self._call("post_turn", session_id, turn)

    async def post_event(self, session_id: str, event: SpeechEvent) -> None:
        await self._call("post_event", session_id, event)

    async def _call(self, method: str, session_id: str, item: Turn | SpeechEvent) -> None:
        self.attempts.append((session_id, item))
        if self.gate is not None:
            await self.gate.wait()
        failure = next(self._script, None)
        if failure is not None:
            raise failure
        self.posted.append((session_id, item))
        self.methods.append(method)


@pytest.fixture(autouse=True)
def pauses(monkeypatch):
    """The wait before a retry is recorded, not slept, so no test sits through it."""
    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(turns_module, "sleep", record)
    return waited


@asynccontextmanager
async def running(detector: FakeDetector | DetectorClient, **kwargs):
    """A Forwarder with its consumer started. On the way out it is closed, which must end run()."""
    forwarder = Forwarder(detector, **kwargs)
    consumer = asyncio.create_task(forwarder.run())
    try:
        yield forwarder
    finally:
        await forwarder.aclose()
        await consumer


async def flush(forwarder: Forwarder) -> None:
    """Wait for the consumer to finish what it holds.

    The limits are generous on purpose: a drain that works returns the moment the work is done, so
    the limits cost nothing, and a drain that doesn't fails here instead of passing slowly.
    """
    await asyncio.wait_for(forwarder.drain(timeout_s=60), timeout=10)


async def test_forwarder_buffers_until_attached_then_posts_in_order():
    detector = FakeDetector()
    async with running(detector) as forwarder:
        forwarder.submit(A)
        forwarder.submit(B)
        forwarder.submit(C)
        await asyncio.sleep(0.01)  # time enough for a consumer that ignored the missing session
        assert detector.attempts == []
        forwarder.attach("s1")
        await flush(forwarder)
    assert detector.posted == [("s1", A), ("s1", B), ("s1", C)]
    assert detector.methods == ["post_turn", "post_event", "post_turn"]  # each to its endpoint


async def test_forwarder_retries_once_then_drops_and_keeps_going(pauses):
    detector = FakeDetector([DetectorError("503"), DetectorError("503")])
    async with running(detector) as forwarder:
        forwarder.attach("s1")
        forwarder.submit(A)
        forwarder.submit(B)
        await flush(forwarder)
    assert detector.attempts == [("s1", A), ("s1", A), ("s1", B)]  # A tried twice, then B
    assert detector.posted == [("s1", B)]
    assert pauses == [0.2]  # one wait, before the retry


async def test_forwarder_retry_that_works_posts_the_item_once(pauses):
    detector = FakeDetector([DetectorError("blip")])
    async with running(detector) as forwarder:
        forwarder.attach("s1")
        forwarder.submit(A)
        await flush(forwarder)
    assert detector.attempts == [("s1", A), ("s1", A)]
    assert detector.posted == [("s1", A)]
    assert pauses == [0.2]


async def test_forwarder_buffer_is_bounded():
    detector = FakeDetector()
    async with running(detector, max_buffer=2) as forwarder:
        forwarder.submit(A)
        forwarder.submit(B)
        forwarder.submit(C)  # the buffer is full: A, the oldest, goes
        forwarder.attach("s1")
        await flush(forwarder)
    assert detector.posted == [("s1", B), ("s1", C)]


def test_submit_never_raises_when_the_detector_is_down():
    # No event loop runs here, and every call to the detector would fail. submit must not touch
    # either: it only buffers, and run() does the posting.
    detector = FakeDetector(itertools.repeat(DetectorError("down")))
    forwarder = Forwarder(detector, max_buffer=2)
    assert forwarder.submit(A) is None
    forwarder.submit(B)
    forwarder.submit(C)  # past the limit
    forwarder.attach("s1")
    forwarder.submit(A)
    assert detector.attempts == []


async def test_forwarder_posts_as_items_arrive_once_attached():
    detector = FakeDetector()
    async with running(detector) as forwarder:
        forwarder.attach("s1")
        forwarder.submit(A)
        await flush(forwarder)
        assert detector.posted == [("s1", A)]
        forwarder.submit(B)  # the consumer went back to sleep when the buffer ran dry
        await flush(forwarder)
    assert detector.posted == [("s1", A), ("s1", B)]


async def test_a_consumer_that_starts_late_still_posts_what_was_buffered():
    detector = FakeDetector()
    forwarder = Forwarder(detector)
    forwarder.submit(A)
    forwarder.attach("s1")
    consumer = asyncio.create_task(forwarder.run())
    try:
        await flush(forwarder)
    finally:
        await forwarder.aclose()
        await consumer
    assert detector.posted == [("s1", A)]


async def test_forwarder_survives_an_unexpected_error():
    detector = FakeDetector([RuntimeError("boom")])
    async with running(detector) as forwarder:
        forwarder.attach("s1")
        forwarder.submit(A)
        forwarder.submit(B)
        await flush(forwarder)
    assert detector.attempts == [("s1", A), ("s1", B)]  # a bug is not retried: it would repeat
    assert detector.posted == [("s1", B)]


@pytest.mark.parametrize(
    "script",
    [
        pytest.param([DetectorError("HTTP 503 on POST /turns")] * 2, id="detector_down"),
        pytest.param([RuntimeError("boom")], id="unexpected_error"),
    ],
)
async def test_a_dropped_item_is_logged_but_its_text_is_not(script, caplog):
    caplog.set_level(logging.DEBUG, logger="callback_agent.turns")
    async with running(FakeDetector(script)) as forwarder:
        forwarder.attach("s1")
        forwarder.submit(A)
        await flush(forwarder)
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "482913" not in caplog.text


BASE = "https://detector.test"
WIRE_TURN = {
    "role": "user",
    "text": "four two",
    "start_ms": 1000400,
    "end_ms": 1000900,
    "words": [
        {"w": "four", "start_ms": 1000420, "end_ms": 1000560},
        {"w": "two", "start_ms": 1000600, "end_ms": 1000880},
    ],
    "interrupted": True,
}


async def test_forwarder_works_with_the_real_detector_client():
    # FakeDetector could drift from DetectorClient. This pins the method names, the argument order
    # and the error the forwarder relies on, against the real client with only its HTTP mocked.
    turn = build_user_turn([FOUR_TWO], interrupted=True)
    event = speech_event(SpeechEventType.USER_SPEECH_START, 1000.123)
    with respx.mock(base_url=BASE) as api:
        turns = api.post("/v1/sessions/s1/turns").mock(
            side_effect=[httpx.Response(503), httpx.Response(204)]
        )
        events = api.post("/v1/sessions/s1/events").respond(204)
        client = DetectorClient(BASE, "tok")
        try:
            async with running(client) as forwarder:
                forwarder.attach("s1")
                forwarder.submit(turn)  # refused once with a 503, then accepted on the retry
                forwarder.submit(event)
                await flush(forwarder)
        finally:
            await client.aclose()
    assert [json.loads(c.request.content) for c in turns.calls] == [WIRE_TURN, WIRE_TURN]
    assert [json.loads(c.request.content) for c in events.calls] == [
        {"type": "user_speech_start", "ts_ms": 1000123}
    ]


# --- drain and close --------------------------------------------------------------------------


async def test_drain_with_nothing_pending_returns_at_once():
    await flush(Forwarder(FakeDetector()))


async def test_drain_waits_for_the_post_in_flight():
    detector = FakeDetector()
    detector.gate = asyncio.Event()
    async with running(detector) as forwarder:
        try:
            forwarder.attach("s1")
            forwarder.submit(A)
            drained = asyncio.create_task(flush(forwarder))
            await asyncio.sleep(0.01)
            # A has left the buffer and is mid-post: the buffer is empty, but drain is not done.
            assert detector.attempts == [("s1", A)]
            assert not drained.done()
            detector.gate.set()
            await drained
            assert detector.posted == [("s1", A)]
        finally:
            detector.gate.set()


async def test_drain_gives_up_at_the_timeout_without_raising(caplog):
    detector = FakeDetector()
    async with running(detector) as forwarder:
        forwarder.submit(A)  # nothing is attached, so A cannot go out
        await forwarder.drain(timeout_s=0.05)
        assert detector.attempts == []
        assert any(r.levelno >= logging.WARNING for r in caplog.records)
        forwarder.attach("s1")  # A is still buffered, and a late attach still flushes it
        await flush(forwarder)
    assert detector.posted == [("s1", A)]


async def test_aclose_ends_run_and_drops_what_is_buffered():
    detector = FakeDetector()
    forwarder = Forwarder(detector)
    consumer = asyncio.create_task(forwarder.run())
    forwarder.submit(A)  # no session yet, so A waits in the buffer
    await asyncio.sleep(0.01)
    await forwarder.aclose()
    assert consumer.done() and consumer.exception() is None  # run() returned: it was not cancelled
    forwarder.attach("s1")
    forwarder.submit(B)  # a closed forwarder ignores both
    await asyncio.sleep(0.01)
    assert detector.attempts == []
    await flush(forwarder)  # B was not buffered, so there is nothing to wait for


async def test_run_on_a_closed_forwarder_returns_at_once():
    forwarder = Forwarder(FakeDetector())
    await forwarder.aclose()
    await asyncio.wait_for(forwarder.run(), timeout=10)


def test_defaults_match_the_plan():
    assert inspect.signature(Forwarder).parameters["max_buffer"].default == 200
    assert inspect.signature(Forwarder.drain).parameters["timeout_s"].default == 2.0


def test_a_buffer_that_holds_nothing_is_refused():
    with pytest.raises(ValueError, match="max_buffer"):
        Forwarder(FakeDetector(), max_buffer=0)
