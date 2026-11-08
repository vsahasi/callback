"""wire_turn_forwarding turns AgentSession events into forwarded turns and speech events, and
finish_call ends the forwarding when the call ends.

The handlers are driven with the real event models, through a stand-in session that only records
who subscribed to what. A few tests use a real AgentSession (never started), to prove that the event
names exist and that no handler can raise into the session's own code.
"""

import asyncio
import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest
import respx
from callback_shared.contract import (
    Channel,
    Role,
    Scenario,
    SessionCreate,
    SpeechEvent,
    SpeechEventType,
    Turn,
    Word,
)
from livekit import rtc
from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
)
from livekit.agents.llm import AgentHandoff, ChatMessage

import callback_agent.session_wiring as wiring
from callback_agent.config import Settings
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.redact import redact_turn
from callback_agent.session_wiring import (
    build_forwarding,
    call_state_for,
    finish_call,
    turn_logging_enabled,
    wire_turn_forwarding,
)
from callback_agent.state import AGENT_SPEECH_HISTORY, CallState
from callback_agent.turns import FinalSegment, Forwarder, build_agent_turn, build_user_turn

# The text stands in for a spoken code: it must never reach a log.
FOUR_TWO = FinalSegment(
    "four two", 0.40, 0.90, (("four", 0.42, 0.56), ("two", 0.60, 0.88)), t0=1000.0
)
SEVEN = FinalSegment("seven", 2.10, 2.50, (("seven", 2.12, 2.48),), t0=1000.0)
# A turn nobody answered: its FINAL was heard, but LiveKit dropped the turn.
OKAY = FinalSegment("okay", 0.10, 0.30, (("okay", 0.12, 0.28),), t0=900.0)
CODE = FinalSegment("my code is 482913", 0.40, 0.90, (("my", 0.42, 0.50),), t0=1000.0)


def new_state(**fields) -> CallState:
    return CallState(room_name="room", caller_identity="caller", channel=Channel.BROWSER, **fields)


class FakeSession:
    """The part of an AgentSession that wire_turn_forwarding uses: on() and agent_state."""

    def __init__(self, agent_state: str = "listening") -> None:
        self.agent_state = agent_state
        self.handlers = defaultdict(list)

    def on(self, event: str, callback=None):
        self.handlers[event].append(callback)
        return callback

    def fire(self, event: str, ev) -> None:
        for handler in self.handlers[event]:
            handler(ev)


class FakeForwarder:
    def __init__(self, fail: Exception | None = None) -> None:
        self.sent: list[Turn | SpeechEvent] = []
        self._fail = fail

    def submit(self, item) -> None:
        if self._fail is not None:
            raise self._fail
        self.sent.append(item)


TURN = Turn(role=Role.USER, text="my code is 482913", start_ms=1_000, end_ms=2_000)


class Detector:
    """Stands in for DetectorClient and records what the end of a call does to it, in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.end_session_error: Exception | None = None
        self.hang_on_end_session = False
        self.aclose_error: Exception | None = None
        self.post_gate: asyncio.Event | None = None
        self.post_started = asyncio.Event()

    async def post_turn(self, session_id: str, turn: Turn) -> None:
        self.post_started.set()
        if self.post_gate is not None:
            await self.post_gate.wait()
        self.calls.append("post_turn")

    async def post_event(self, session_id: str, event: SpeechEvent) -> None:
        self.calls.append("post_event")

    async def end_session(self, session_id: str) -> None:
        self.calls.append(f"end_session {session_id}")
        if self.hang_on_end_session:
            await asyncio.Event().wait()
        if self.end_session_error is not None:
            raise self.end_session_error

    async def aclose(self) -> None:
        self.calls.append("aclose")
        if self.aclose_error is not None:
            raise self.aclose_error


def item_added(item, created_at: float = 1000.0) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=item, created_at=created_at)


def agent_changed(old: str, new: str, at: float = 1000.5) -> AgentStateChangedEvent:
    return AgentStateChangedEvent(old_state=old, new_state=new, created_at=at)


def user_changed(old: str, new: str, at: float = 1000.5) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state=old, new_state=new, created_at=at)


def said(text: str | None, role: str = "user", **fields) -> ChatMessage:
    return ChatMessage(role=role, content=[] if text is None else [text], **fields)


def spoken_at(first_word_s: float, text: str = "four two") -> FinalSegment:
    """A caller's FINAL whose first word starts at `first_word_s` (Unix seconds)."""
    words = tuple((w, 0.3 * i, 0.3 * i + 0.2) for i, w in enumerate(text.split()))
    return FinalSegment(text, -0.02, words[-1][2] + 0.02, words, t0=first_word_s)


def agent_speaks(session, start: float, end: float | None = None) -> None:
    """The agent starts speaking at `start` and, if `end` is given, stops then (Unix seconds)."""
    session.fire("agent_state_changed", agent_changed("listening", "speaking", at=start))
    if end is not None:
        session.fire("agent_state_changed", agent_changed("speaking", "listening", at=end))


def redacted(turn: Turn) -> Turn:
    """What the wiring forwards for `turn`: the same turn with its numbers redacted."""
    return redact_turn(turn, exempt_short_runs=False)


def message_arrives(wired, *segments: FinalSegment) -> Turn:
    """The caller's message arrives with these segments pending. Returns the user turn forwarded."""
    wired.state.pending_segments.extend(segments)
    before = len(wired.sent)
    text = " ".join(segment.text for segment in segments)
    wired.session.fire("conversation_item_added", item_added(said(text)))
    [turn] = wired.sent[before:]
    return turn


@pytest.fixture
def wired():
    session, state, forwarder = FakeSession(), new_state(), FakeForwarder()
    wire_turn_forwarding(session, state, forwarder)
    return SimpleNamespace(session=session, state=state, sent=forwarder.sent)


# --- user messages ----------------------------------------------------------------------------


def test_a_user_message_forwards_the_pending_segments_as_one_user_turn(wired):
    wired.state.pending_segments.extend([FOUR_TWO])
    wired.session.fire("conversation_item_added", item_added(said("four two")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO], interrupted=False))]
    # "four two" is a number, so it is forwarded as one [NUM] word with the times of both.
    assert wired.sent[0].start_ms == 1000400 and wired.sent[0].words == [
        Word(w="[NUM]", start_ms=1000420, end_ms=1000880)
    ]
    assert wired.state.pending_segments == []


def test_the_segments_of_one_message_are_merged(wired):
    wired.state.pending_segments.extend([FOUR_TWO, SEVEN])
    wired.session.fire("conversation_item_added", item_added(said("four two seven")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO, SEVEN], interrupted=False))]
    assert wired.state.pending_segments == []


def test_typed_input_becomes_a_turn_without_words(wired):
    wired.session.fire(
        "conversation_item_added", item_added(said("hello there", created_at=1000.25))
    )
    assert wired.sent == [
        Turn(role=Role.USER, text="hello there", start_ms=1000250, end_ms=1000250, words=[])
    ]


def test_a_message_with_no_text_forwards_nothing(wired):
    wired.session.fire("conversation_item_added", item_added(said(None)))
    wired.session.fire("conversation_item_added", item_added(said("   ")))
    assert wired.sent == []


def test_a_segment_nobody_answered_is_not_merged_into_the_next_turn(wired, caplog):
    # A turn that ends while an uninterruptible prompt plays is dropped by LiveKit: no message
    # follows its FINAL, so its segment would otherwise lead the next turn.
    wired.state.pending_segments.extend([OKAY, FOUR_TWO])
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        wired.session.fire("conversation_item_added", item_added(said("four two")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO], interrupted=False))]
    assert wired.state.pending_segments == []
    assert "okay" not in caplog.text and any(r.levelno == logging.WARNING for r in caplog.records)


def test_segments_that_come_after_the_message_wait_for_their_own(wired):
    wired.state.pending_segments.extend([FOUR_TWO, SEVEN])
    wired.session.fire("conversation_item_added", item_added(said("four two")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO], interrupted=False))]
    assert wired.state.pending_segments == [SEVEN]


def test_a_message_the_segments_do_not_match_takes_them_all(wired):
    # Something rewrote the message after the FINALs: the words are still the best record of it.
    wired.state.pending_segments.extend([FOUR_TWO, SEVEN])
    wired.session.fire("conversation_item_added", item_added(said("four two [removed]")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO, SEVEN], interrupted=False))]
    assert wired.state.pending_segments == []


def test_whitespace_in_the_message_does_not_hide_a_match(wired):
    wired.state.pending_segments.extend([OKAY, FOUR_TWO])
    wired.session.fire("conversation_item_added", item_added(said("  four   two ")))
    assert wired.sent == [redacted(build_user_turn([FOUR_TWO], interrupted=False))]


def test_the_same_words_twice_still_pick_the_latest(wired):
    yes_1 = FinalSegment("yes", 0.1, 0.3, (("yes", 0.1, 0.3),), t0=900.0)
    yes_2 = FinalSegment("yes", 0.1, 0.3, (("yes", 0.1, 0.3),), t0=1000.0)
    wired.state.pending_segments.extend([yes_1, yes_2])
    wired.session.fire("conversation_item_added", item_added(said("yes")))
    assert wired.sent == [build_user_turn([yes_2], interrupted=False)]


# --- assistant messages -----------------------------------------------------------------------


def test_an_assistant_message_forwards_an_agent_turn_from_its_metrics(wired):
    spoken = said(
        "Thank you.",
        role="assistant",
        interrupted=True,
        metrics={"started_speaking_at": 1000.25, "stopped_speaking_at": 1002.5},
    )
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1003.0))
    assert wired.sent == [
        Turn(
            role=Role.AGENT,
            text="Thank you.",
            start_ms=1000250,
            end_ms=1002500,
            words=[],
            interrupted=True,
        )
    ]


def test_an_assistant_message_without_metrics_falls_back_to_created_at(wired):
    spoken = said("Thank you.", role="assistant", created_at=1000.25)
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1002.5))
    assert wired.sent == [build_agent_turn("Thank you.", 1000.25, 1002.5, interrupted=False)]


def test_an_agent_turn_never_ends_before_it_starts(wired):
    spoken = said("Hi.", role="assistant", created_at=1005.0)
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1002.0))
    assert wired.sent[0].start_ms == wired.sent[0].end_ms == 1005000


def test_an_assistant_message_with_no_text_forwards_nothing(wired):
    wired.session.fire("conversation_item_added", item_added(said(None, role="assistant")))
    assert wired.sent == []


@pytest.mark.parametrize("role", ["system", "developer"])
def test_messages_that_nobody_said_are_ignored(role, wired):
    wired.state.pending_segments.append(FOUR_TWO)
    wired.session.fire("conversation_item_added", item_added(said("be nice", role=role)))
    assert wired.sent == [] and wired.state.pending_segments == [FOUR_TWO]


def test_an_agent_handoff_marker_is_ignored(wired):
    wired.session.fire("conversation_item_added", item_added(AgentHandoff(new_agent_id="agent")))
    assert wired.sent == []


# --- speech events ----------------------------------------------------------------------------


def test_the_agent_starting_to_speak_is_a_speech_start(wired):
    wired.session.fire("agent_state_changed", agent_changed("listening", "speaking", at=1000.5))
    assert wired.sent == [SpeechEvent(type=SpeechEventType.AGENT_SPEECH_START, ts_ms=1000500)]


def test_the_agent_leaving_speaking_is_a_speech_end(wired):
    wired.session.fire("agent_state_changed", agent_changed("speaking", "listening", at=1002.25))
    wired.session.fire("agent_state_changed", agent_changed("speaking", "thinking", at=1003.0))
    assert [(e.type, e.ts_ms) for e in wired.sent] == [
        (SpeechEventType.AGENT_SPEECH_END, 1002250),
        (SpeechEventType.AGENT_SPEECH_END, 1003000),
    ]


def test_agent_changes_that_do_not_touch_speaking_send_nothing(wired):
    wired.session.fire("agent_state_changed", agent_changed("initializing", "listening"))
    wired.session.fire("agent_state_changed", agent_changed("listening", "thinking"))
    wired.session.fire("agent_state_changed", agent_changed("thinking", "listening"))
    assert wired.sent == []


def test_the_caller_starting_to_speak_is_a_speech_start(wired):
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=1001.5))
    assert wired.sent == [SpeechEvent(type=SpeechEventType.USER_SPEECH_START, ts_ms=1001500)]


def test_other_caller_changes_send_nothing(wired):
    wired.session.fire("user_state_changed", user_changed("speaking", "listening"))
    wired.session.fire("user_state_changed", user_changed("listening", "away"))
    assert wired.sent == []


def test_a_whole_exchange_goes_out_in_order(wired):
    session, state = wired.session, wired.state
    session.fire("agent_state_changed", agent_changed("listening", "speaking", at=1000.0))
    spoken = said(
        "Who am I speaking with?",
        role="assistant",
        metrics={"started_speaking_at": 1000.0, "stopped_speaking_at": 1002.0},
    )
    session.fire("conversation_item_added", item_added(spoken, created_at=1002.0))
    session.fire("agent_state_changed", agent_changed("speaking", "listening", at=1002.0))
    session.fire("user_state_changed", user_changed("listening", "speaking", at=1003.0))
    state.pending_segments.append(FOUR_TWO)
    session.fire("conversation_item_added", item_added(said("four two"), created_at=1004.0))
    kinds = [
        i.type.value if isinstance(i, SpeechEvent) else f"{i.role.value} turn" for i in wired.sent
    ]
    assert kinds == [
        "agent_speech_start",
        "agent turn",
        "agent_speech_end",
        "user_speech_start",
        "user turn",
    ]


# --- interrupted: the caller began while the agent was speaking -----------------------------


def test_a_barge_in_then_a_pause_then_more_speech_is_still_an_interruption(wired):
    # VAD flips the caller to listening and back inside one Flux turn. The turn began over the
    # agent all the same, and only its first word says so.
    session = wired.session
    agent_speaks(session, 1000.0)  # the agent is mid-sentence
    session.fire("user_state_changed", user_changed("listening", "speaking", at=1000.6))
    session.fire("agent_state_changed", agent_changed("speaking", "listening", at=1001.2))
    session.fire("user_state_changed", user_changed("speaking", "listening", at=1002.0))  # a pause
    session.fire("user_state_changed", user_changed("listening", "speaking", at=1002.8))  # more
    words = (("one", 0.62, 0.80), ("two", 0.90, 1.10), ("three", 2.90, 3.20))
    turn = message_arrives(wired, FinalSegment("one two three", 0.60, 3.30, words, t0=1000.0))
    assert turn.interrupted is True
    assert turn.words[0].start_ms == 1000620


def test_speech_that_starts_after_the_agent_stopped_is_not_an_interruption(wired):
    agent_speaks(wired.session, 1000.0, 1002.0)
    assert message_arrives(wired, spoken_at(1002.4)).interrupted is False


def test_typed_input_is_never_an_interruption(wired):
    agent_speaks(wired.session, 1000.0)  # the agent is speaking right now
    wired.session.fire("conversation_item_added", item_added(said("hello", created_at=1000.5)))
    [turn] = wired.sent[-1:]
    assert turn.role == Role.USER and turn.words == [] and turn.interrupted is False


@pytest.mark.parametrize(
    ("first_word_ms", "interrupted"),
    [(999_999, False), (1_000_000, True), (1_001_999, True), (1_002_000, False)],
    ids=["a ms before", "the first ms", "the last ms", "the end"],
)
def test_an_agent_speech_covers_its_first_millisecond_and_not_its_end(
    first_word_ms, interrupted, wired
):
    agent_speaks(wired.session, 1000.0, 1002.0)
    assert message_arrives(wired, spoken_at(first_word_ms / 1000)).interrupted is interrupted


def test_a_turn_that_starts_while_the_agent_is_still_speaking_is_an_interruption(wired):
    agent_speaks(wired.session, 1000.0)  # still speaking when the message arrives
    assert message_arrives(wired, spoken_at(1005.0)).interrupted is True


def test_a_turn_that_began_before_the_agent_did_is_not_an_interruption(wired):
    agent_speaks(wired.session, 1000.0, 1002.0)
    assert message_arrives(wired, spoken_at(999.5)).interrupted is False


def test_an_older_agent_speech_still_counts(wired):
    for start in (1000.0, 1010.0, 1020.0):
        agent_speaks(wired.session, start, start + 2.0)
    assert message_arrives(wired, spoken_at(1011.0)).interrupted is True


def test_the_first_word_of_a_turn_decides_not_a_later_one(wired):
    agent_speaks(wired.session, 1000.0, 1002.0)
    agent_speaks(wired.session, 1004.0, 1006.0)
    began_over_the_agent = (spoken_at(1001.0, "one two"), spoken_at(1003.0, "three"))
    assert message_arrives(wired, *began_over_the_agent).interrupted is True
    began_in_the_quiet = (spoken_at(1002.5, "one two"), spoken_at(1004.5, "three"))
    assert message_arrives(wired, *began_in_the_quiet).interrupted is False


def test_a_turn_without_word_times_is_placed_by_where_its_window_starts(wired):
    agent_speaks(wired.session, 1000.0, 1002.0)
    no_words = FinalSegment("four two", 1.0, 1.9, (), t0=1000.0)  # the window starts at 1001.0
    assert message_arrives(wired, no_words).interrupted is True


def test_a_segment_dropped_as_stale_does_not_decide(wired):
    agent_speaks(
        wired.session, 900.0, 901.0
    )  # the stale OKAY (first word at 900.12) is inside this
    wired.state.pending_segments.append(OKAY)
    assert message_arrives(wired, FOUR_TWO).interrupted is False


def test_agent_speech_is_kept_as_intervals_in_epoch_ms(wired):
    agent_speaks(wired.session, 1000.0, 1002.25)
    agent_speaks(wired.session, 1005.0)
    spans = [(i.start_ms, i.end_ms) for i in wired.state.agent_speech]
    assert spans == [(1000000, 1002250), (1005000, None)]


def test_the_speech_events_and_the_intervals_use_the_same_milliseconds(wired):
    agent_speaks(wired.session, 1000.0004, 1002.2506)
    [started, ended] = [e.ts_ms for e in wired.sent]
    assert [(i.start_ms, i.end_ms) for i in wired.state.agent_speech] == [(started, ended)]


def test_a_second_start_while_the_agent_is_speaking_keeps_the_first(wired):
    agent_speaks(wired.session, 1000.0)
    agent_speaks(wired.session, 1001.0)
    assert [(i.start_ms, i.end_ms) for i in wired.state.agent_speech] == [(1000000, None)]


def test_a_stop_stamped_before_its_start_is_a_speech_of_no_length(wired):
    agent_speaks(
        wired.session, 1001.0, 1000.0
    )  # clocks that disagree must not make a negative span
    [speech] = wired.state.agent_speech
    assert (speech.start_ms, speech.end_ms) == (1001000, 1001000)
    assert message_arrives(wired, spoken_at(1000.5)).interrupted is False


def test_a_stop_with_nothing_open_opens_nothing_and_is_still_forwarded(wired):
    wired.session.fire("agent_state_changed", agent_changed("speaking", "listening", at=1000.0))
    assert list(wired.state.agent_speech) == []
    assert [e.type for e in wired.sent] == [SpeechEventType.AGENT_SPEECH_END]


def test_the_speech_history_is_bounded_and_keeps_the_newest(wired):
    for k in range(AGENT_SPEECH_HISTORY + 5):
        agent_speaks(wired.session, 1000.0 + 10 * k, 1001.0 + 10 * k)
    history = wired.state.agent_speech
    assert len(history) == AGENT_SPEECH_HISTORY
    assert history[0].start_ms == 1_050_000  # the first five were dropped
    assert history[-1].start_ms == (1000 + 10 * (AGENT_SPEECH_HISTORY + 4)) * 1000
    assert message_arrives(wired, spoken_at(1000.5)).interrupted is False  # long gone


def test_each_call_gets_its_own_speech_history():
    first, second = new_state(), new_state()
    first.agent_speech.append(SimpleNamespace(start_ms=1, end_ms=2))
    assert len(second.agent_speech) == 0


# --- close ------------------------------------------------------------------------------------


def test_closing_the_session_marks_the_call_as_closing(wired):
    assert wired.state.closing is False
    wired.session.fire("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
    assert wired.state.closing is True


# --- no detector ------------------------------------------------------------------------------


def test_without_a_forwarder_the_bookkeeping_still_happens(caplog):
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, None, log_turns=True)
    agent_speaks(session, 999.0)
    state.pending_segments.append(FOUR_TWO)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        session.fire("conversation_item_added", item_added(said("four two")))
    session.fire("close", CloseEvent(reason=CloseReason.USER_INITIATED))
    assert state.pending_segments == [] and state.closing is True
    assert [(i.start_ms, i.end_ms) for i in state.agent_speech] == [(999000, None)]
    assert "interrupted=True" in caplog.text  # the turn was judged against the agent's speech


# --- with the real forwarder ---------------------------------------------------------------


async def test_the_wiring_feeds_the_real_forwarder_which_sends_in_order_once_attached():
    detector = Detector()
    session, state, forwarder = FakeSession(), new_state(), Forwarder(detector)
    wire_turn_forwarding(session, state, forwarder)
    consumer = asyncio.create_task(forwarder.run())
    try:
        # Before the caller is identified the forwarder only buffers (D9).
        session.fire("agent_state_changed", agent_changed("listening", "speaking", at=1000.0))
        state.pending_segments.append(FOUR_TWO)
        session.fire("conversation_item_added", item_added(said("four two")))
        await asyncio.sleep(0)
        assert detector.calls == []
        forwarder.attach("s1")
        await asyncio.wait_for(forwarder.drain(60), timeout=10)
    finally:
        await forwarder.aclose()
        await consumer
    assert detector.calls == ["post_event", "post_turn"]


# --- handlers never raise ---------------------------------------------------------------------


def test_a_forwarder_that_raises_cannot_break_a_handler(caplog):
    # Forwarder.submit never raises, but the handlers must not depend on that.
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, FakeForwarder(fail=RuntimeError("down 482913")))
    state.pending_segments.append(CODE)
    with caplog.at_level(logging.ERROR, logger="callback_agent.session_wiring"):
        session.fire("conversation_item_added", item_added(said("my code is 482913")))
        session.fire("conversation_item_added", item_added(said("Hi.", role="assistant")))
        session.fire("agent_state_changed", agent_changed("listening", "speaking"))
        session.fire("user_state_changed", user_changed("listening", "speaking"))
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 4
    assert "482913" not in caplog.text  # nor in an exception's message
    assert state.pending_segments == []  # the failed turn does not poison the next one


def test_a_handler_error_does_not_stop_the_next_turn():
    session, state, forwarder = FakeSession(), new_state(), FakeForwarder()
    wire_turn_forwarding(session, state, forwarder)
    state.pending_segments.append(FinalSegment("bad", 0.0, 0.1, (), t0=None))
    session.fire("conversation_item_added", item_added(said("bad")))
    state.pending_segments.append(FOUR_TWO)
    session.fire("conversation_item_added", item_added(said("four two")))
    assert forwarder.sent == [redacted(build_user_turn([FOUR_TWO], interrupted=False))]


async def test_the_real_session_delivers_every_event_to_the_handlers():
    session, state, forwarder = AgentSession(), new_state(), FakeForwarder()
    wire_turn_forwarding(session, state, forwarder)
    state.pending_segments.append(FOUR_TWO)
    session.emit("agent_state_changed", agent_changed("listening", "speaking"))
    session.emit("user_state_changed", user_changed("listening", "speaking"))
    session.emit("conversation_item_added", item_added(said("four two")))
    session.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
    assert [type(i).__name__ for i in forwarder.sent] == ["SpeechEvent", "SpeechEvent", "Turn"]
    assert state.closing is True and state.pending_segments == []


async def test_a_type_error_in_a_handler_does_not_reach_the_session(caplog):
    # rtc.EventEmitter lets a TypeError out of emit(), into whatever code emitted the event.
    session, state = AgentSession(), new_state()
    wire_turn_forwarding(session, state, FakeForwarder())
    state.pending_segments.append(FinalSegment("my code is 482913", 0.0, 0.1, (), t0=None))
    with caplog.at_level(logging.ERROR):
        session.emit("conversation_item_added", item_added(said("my code is 482913")))
    assert any("TypeError" in r.getMessage() for r in caplog.records)
    assert "482913" not in caplog.text


# --- CALLBACK_LOG_TURNS -----------------------------------------------------------------------


def test_logging_turns_names_the_role_times_word_count_and_flag_but_not_the_text(caplog):
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, FakeForwarder(), log_turns=True)
    state.pending_segments.append(CODE)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        agent_speaks(session, 1000.0)  # the caller's first word, at 1000.42, is over the agent
        session.fire("conversation_item_added", item_added(said("my code is 482913")))
    assert [r.getMessage() for r in caplog.records][-1] == (
        "forwarding turn: role=user start_ms=1000400 end_ms=1000900 words=1 interrupted=True"
    )
    assert "482913" not in caplog.text


def test_logging_turns_covers_agent_turns_and_events(caplog):
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, FakeForwarder(), log_turns=True)
    spoken = said(
        "Thank you.",
        role="assistant",
        metrics={"started_speaking_at": 1000.25, "stopped_speaking_at": 1002.5},
    )
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        session.fire("conversation_item_added", item_added(spoken))
        session.fire("agent_state_changed", agent_changed("speaking", "listening", at=1002.5))
        session.fire("user_state_changed", user_changed("listening", "speaking", at=1003.5))
    assert [r.getMessage() for r in caplog.records] == [
        "forwarding turn: role=agent start_ms=1000250 end_ms=1002500 words=0 interrupted=False",
        "forwarding event: type=agent_speech_end ts_ms=1002500",
        "forwarding event: type=user_speech_start ts_ms=1003500",
    ]
    assert "Thank you" not in caplog.text


def test_turns_are_not_logged_unless_asked(caplog, wired):
    wired.state.pending_segments.append(FOUR_TWO)
    with caplog.at_level(logging.DEBUG, logger="callback_agent.session_wiring"):
        wired.session.fire("conversation_item_added", item_added(said("four two")))
        wired.session.fire("agent_state_changed", agent_changed("listening", "speaking"))
    assert caplog.records == [] and len(wired.sent) == 2


def test_logging_works_without_a_detector(caplog):
    # The live timing check runs before a detector exists, so the log must not depend on one.
    session, state = FakeSession(), new_state()
    wire_turn_forwarding(session, state, None, log_turns=True)
    state.pending_segments.append(FOUR_TWO)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        session.fire("conversation_item_added", item_added(said("four two")))
    assert len(caplog.records) == 1


@pytest.mark.parametrize(
    ("env", "enabled"),
    [
        ({}, False),
        ({"CALLBACK_LOG_TURNS": ""}, False),
        ({"CALLBACK_LOG_TURNS": "   "}, False),
        ({"CALLBACK_LOG_TURNS": "1"}, True),
        ({"CALLBACK_LOG_TURNS": "true"}, True),
        ({"CALLBACK_LOG_TURNS": "0"}, True),  # any non-empty value turns it on
    ],
)
def test_turn_logging_is_on_for_any_non_empty_value(env, enabled):
    assert turn_logging_enabled(env) is enabled


# --- the call's state -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "channel"),
    [
        (rtc.ParticipantKind.PARTICIPANT_KIND_SIP, Channel.PHONE),
        (rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, Channel.BROWSER),
        (rtc.ParticipantKind.PARTICIPANT_KIND_INGRESS, Channel.BROWSER),
        (rtc.ParticipantKind.PARTICIPANT_KIND_AGENT, Channel.BROWSER),
    ],
)
def test_the_call_state_follows_the_participant(kind, channel):
    caller = SimpleNamespace(identity="caller-1", kind=kind)
    state = call_state_for("room-1", caller)
    assert (state.room_name, state.caller_identity, state.channel) == (
        "room-1",
        "caller-1",
        channel,
    )
    assert state.pending_segments == [] and len(state.agent_speech) == 0
    assert state.session_id is None


def test_each_call_gets_its_own_segment_list():
    first, second = new_state(), new_state()
    first.pending_segments.append(FOUR_TWO)
    assert second.pending_segments == []


# --- the detector client and forwarder of a call ----------------------------------------------


def settings(**fields) -> Settings:
    return Settings(agent_name="callback", cartesia_voice_id="voice", **fields)


@pytest.mark.parametrize(
    "fields",
    [{}, {"detector_url": "https://detector.test"}, {"detector_token": "tok"}],
    ids=["neither", "only_url", "only_token"],
)
def test_without_both_detector_settings_nothing_is_forwarded(fields, caplog):
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        assert build_forwarding(settings(**fields)) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


async def test_with_a_detector_configured_the_call_gets_a_client_and_a_forwarder():
    both = settings(
        detector_url="https://detector.test",
        detector_token="tok",
        detector_scenario=Scenario.REAL_CFO,
    )
    detector, forwarder = build_forwarding(both)
    try:
        assert isinstance(detector, DetectorClient) and isinstance(forwarder, Forwarder)
        with respx.mock(base_url="https://detector.test") as api:
            route = api.post("/v1/sessions").respond(200, json={"session_id": "s1"})
            await detector.create_session(
                SessionCreate(
                    room="r", caller_identity="c", claimed_user_id="u", channel=Channel.PHONE
                )
            )
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer tok"
        assert request.headers["x-callback-scenario"] == "real_cfo"
    finally:
        await detector.aclose()


# --- the end of the call ----------------------------------------------------------------------


@pytest.fixture
def quick(monkeypatch):
    """Short limits, so that a step that hangs is given up on in a blink."""
    for name in ("CLOSE_WAIT_S", "DRAIN_S", "END_SESSION_S", "FORWARDER_CLOSE_S"):
        monkeypatch.setattr(wiring, name, 0.05)


@pytest.fixture
async def call():
    """A forwarder with its consumer running, attached to session s1, and the state of that call."""
    detector = Detector()
    forwarder = Forwarder(detector)
    consumer = asyncio.create_task(forwarder.run())
    forwarder.attach("s1")
    yield SimpleNamespace(
        detector=detector,
        forwarder=forwarder,
        consumer=consumer,
        state=new_state(session_id="s1"),
    )
    if detector.post_gate is not None:
        detector.post_gate.set()
    await forwarder.aclose()
    await asyncio.gather(consumer, return_exceptions=True)  # finish_call may have cancelled it


async def end(call, **kwargs) -> None:
    await asyncio.wait_for(
        finish_call(call.state, call.forwarder, call.detector, consumer=call.consumer, **kwargs),
        timeout=10,
    )


async def test_the_end_of_a_call_sends_what_is_left_then_ends_the_session_then_closes(call):
    call.forwarder.submit(TURN)
    await end(call)
    assert call.detector.calls == ["post_turn", "end_session s1", "aclose"]
    assert call.consumer.done() and call.consumer.exception() is None


async def test_a_call_that_never_identified_anyone_has_nothing_to_send_or_end(call, monkeypatch):
    monkeypatch.setattr(wiring, "DRAIN_S", 60)  # a drain here would sit for a minute
    call.state.session_id = None
    call.forwarder.submit(TURN)
    await end(call)
    assert call.detector.calls == ["aclose"]
    assert call.consumer.done()


async def test_a_session_that_fails_to_end_is_logged_and_the_rest_still_closes(call, caplog):
    call.detector.end_session_error = DetectorError("POST /v1/sessions/s1/end: HTTP 503")
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        await end(call)
    assert call.detector.calls == ["end_session s1", "aclose"]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert call.consumer.done()


async def test_a_session_that_hangs_on_end_is_given_up_on(call, quick, caplog):
    call.detector.hang_on_end_session = True
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        await end(call)
    assert call.detector.calls == ["end_session s1", "aclose"]
    assert "could not end the detector's session: timed out" in caplog.text


async def test_a_timeout_raised_by_the_work_itself_is_logged_like_any_other(call, caplog):
    call.detector.end_session_error = TimeoutError("the detector's own timer")
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        await end(call)
    assert call.detector.calls == ["end_session s1", "aclose"]
    assert "could not end the detector's session: timed out" in caplog.text


async def test_a_post_that_hangs_cannot_hold_the_call_open(call, quick):
    call.detector.post_gate = asyncio.Event()  # the detector never answers
    call.forwarder.submit(TURN)
    await asyncio.wait_for(call.detector.post_started.wait(), timeout=10)
    await end(call)  # the drain gives up, the session is ended, aclose gives up on the post
    assert call.detector.calls == ["end_session s1", "aclose"]
    assert call.consumer.done()  # cancelled, not left behind


async def test_the_end_waits_for_the_session_to_finish_closing(call, monkeypatch):
    # LiveKit runs its own close next to ours; its last turns arrive while it closes.
    monkeypatch.setattr(wiring, "CLOSE_WAIT_S", 60)
    closed = asyncio.Event()
    ending = asyncio.create_task(end(call, session_closed=closed))
    for _ in range(5):
        await asyncio.sleep(0)
    assert not ending.done() and call.detector.calls == []
    call.forwarder.submit(TURN)  # a last turn, made while the session closes
    closed.set()
    await ending
    assert call.detector.calls == ["post_turn", "end_session s1", "aclose"]


async def test_the_end_does_not_wait_forever_for_a_session_that_never_closes(call, quick):
    await end(call, session_closed=asyncio.Event())
    assert call.detector.calls == ["end_session s1", "aclose"]


async def test_an_already_closed_session_costs_no_wait(call, monkeypatch):
    monkeypatch.setattr(wiring, "CLOSE_WAIT_S", 60)
    closed = asyncio.Event()
    closed.set()
    await end(call, session_closed=closed)
    assert call.detector.calls == ["end_session s1", "aclose"]


async def test_a_detector_that_fails_to_close_does_not_raise(call, caplog):
    call.detector.aclose_error = RuntimeError("already closed")
    with caplog.at_level(logging.WARNING, logger="callback_agent.session_wiring"):
        await end(call)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


async def test_a_forwarder_that_fails_to_drain_does_not_stop_the_session_from_ending(
    call, monkeypatch
):
    async def broken(timeout_s=2.0):
        raise RuntimeError("bug")

    monkeypatch.setattr(call.forwarder, "drain", broken)
    await end(call)
    assert call.detector.calls == ["end_session s1", "aclose"]


def test_the_limits_are_what_the_brief_says():
    assert wiring.DRAIN_S == 2.0 and wiring.FORWARDER_CLOSE_S == 3.0


# --- redaction of what is forwarded ---

ACCOUNT = FinalSegment(
    "the account is 0 0 0 1 2 3 4 5 6 7 8 9",
    0.0,
    2.0,
    tuple(
        (w, 0.1 * i, 0.1 * i + 0.08)
        for i, w in enumerate(
            ["the", "account", "is", "0", "0", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        )
    ),
    t0=1000.0,
)


def test_a_forwarded_user_turn_has_a_bank_number_redacted(wired):
    turn = message_arrives(wired, ACCOUNT)
    assert turn.text == "the account is [NUM]"
    assert [w.w for w in turn.words] == ["the", "account", "is", "[NUM]"]


def test_a_forwarded_agent_turn_is_redacted_too(wired):
    spoken = said("the account ending 6789", role="assistant", created_at=1000.25)
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1002.5))
    assert wired.sent[0].text == "the account ending [NUM]"


def test_the_exempt_window_covers_only_the_first_user_turn(wired):
    wired.state.answer_exempt = True
    first = message_arrives(wired, spoken_at(1000.0, "four two four two four two"))
    assert first.text == "four two four two four two"
    assert wired.state.answer_exempt is False
    second = message_arrives(wired, spoken_at(1010.0, "four two four two four two"))
    assert second.text == "[NUM]"


def test_an_agent_turn_does_not_use_up_the_window(wired):
    wired.state.answer_exempt = True
    spoken = said("please repeat after me", role="assistant", created_at=1000.25)
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1002.5))
    assert wired.state.answer_exempt is True


def test_an_agent_turn_is_never_exempt(wired):
    wired.state.answer_exempt = True
    spoken = said("four two four two four two", role="assistant", created_at=1000.25)
    wired.session.fire("conversation_item_added", item_added(spoken, created_at=1002.5))
    assert wired.sent[0].text == "[NUM]"
