"""TimedSttMixin records each FINAL_TRANSCRIPT of the STT stream as a FinalSegment on the CallState.

Agent.default.stt_node needs a live STT, so these tests swap it for a stand-in. The stand-in reads
every audio frame it is given, then yields the events a test scripted. The wall clock is pinned, so
the anchor of a stt_node call is known.
"""

import logging
from types import SimpleNamespace

import pytest
from callback_shared.contract import Channel, Role, Turn, Word
from fake_llm import FakeLLM
from livekit import rtc
from livekit.agents import Agent, AgentSession, ConversationItemAddedEvent, ModelSettings, stt
from livekit.agents.llm import ChatMessage
from livekit.agents.types import TimedString

import callback_agent.agents as agents_module
from callback_agent.agents import TimedSttMixin
from callback_agent.session_wiring import wire_turn_forwarding
from callback_agent.state import CallState
from callback_agent.turns import FinalSegment

FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT
INTERIM = stt.SpeechEventType.INTERIM_TRANSCRIPT


def new_state() -> CallState:
    return CallState(room_name="room", caller_identity="caller", channel=Channel.BROWSER)


class ProbeAgent(TimedSttMixin, Agent):
    """An Agent that belongs to the session a test hands it, so no session has to run."""

    def __init__(self, session) -> None:
        super().__init__(instructions="test")
        self._probe_session = session

    @property
    def session(self):
        return self._probe_session


class RealSessionAgent(TimedSttMixin, Agent):
    """The same mixin on an ordinary Agent, so that Agent.session is the real property."""

    def __init__(self) -> None:
        super().__init__(instructions="test")


def probe_agent(state: CallState) -> ProbeAgent:
    return ProbeAgent(SimpleNamespace(userdata=state))


def frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(24000, 1, 480)  # 20 ms


def word(text: str, start: float, end: float, offset=None) -> TimedString:
    extra = {} if offset is None else {"start_time_offset": offset}
    return TimedString(text, start_time=start, end_time=end, **extra)


def event(kind, text: str, start: float, end: float, words, created_at: float = 1000.9):
    data = stt.SpeechData(language="en", text=text, start_time=start, end_time=end, words=words)
    return stt.SpeechEvent(type=kind, alternatives=[data], created_at=created_at)


def four_two(offset: float | None = None, shift: float = 0.0) -> stt.SpeechEvent:
    """The FINAL for "four two", with `shift` seconds added to every time, as an offset does."""
    words = [
        word("four", 0.42 + shift, 0.56 + shift, offset),
        word("two", 0.60 + shift, 0.88 + shift, offset),
    ]
    return event(FINAL, "four two", 0.40 + shift, 0.90 + shift, words)


def assert_segment(segment: FinalSegment, text, start, end, words, t0) -> None:
    """Segment times went through subtractions, so they are compared to the nearest microsecond."""
    assert segment.text == text
    assert segment.start_s == pytest.approx(start) and segment.end_s == pytest.approx(end)
    assert segment.t0 == pytest.approx(t0)
    assert [w[0] for w in segment.words] == [w[0] for w in words]
    assert [t for w in segment.words for t in w[1:]] == pytest.approx(
        [t for w in words for t in w[1:]]
    )


FOUR_TWO_WORDS = (("four", 0.42, 0.56), ("two", 0.60, 0.88))


def grown_by(growth: float) -> stt.SpeechEvent:
    """The FINAL for "four two" once the stream's offset, 3 s at the first frame, has grown."""
    return four_two(offset=3.0 + growth, shift=3.0 + growth)


def switch_records(caplog) -> list[logging.LogRecord]:
    """The records in which the mixin says a stream has switched to the offset of a final."""
    return [
        r
        for r in caplog.records
        if r.name == "callback_agent.agents" and "after its transcript arrived" in r.getMessage()
    ]


@pytest.fixture
def clock(monkeypatch):
    """Pins the wall clock the anchor reads. The first 20 ms frame then puts t0 at 1000.0."""
    now = SimpleNamespace(value=1000.02)
    monkeypatch.setattr(agents_module, "wall_clock", lambda: now.value)
    return now


@pytest.fixture
def stream_offset(monkeypatch):
    """What the mixin reads off the STT stream at the first frame. A test sets `value` or `error`.

    The stand-in node runs without a pipeline, so without this fixture the read finds no stream
    and the mixin falls back to the offset on the first word.
    """
    read = SimpleNamespace(value=None, error=None, reads=0)

    def fake_read():
        read.reads += 1
        if read.error is not None:
            raise read.error
        return read.value

    monkeypatch.setattr(agents_module, "_stream_start_time_offset", fake_read)
    return read


@pytest.fixture
def stream_started_at(monkeypatch):
    """When the attempt feeding the STT stream began, as the mixin reads it at the first frame.

    A test sets `value` or `error`. Without this fixture the read finds no stream, and the mixin
    counts no wait for the first frame.
    """
    read = SimpleNamespace(value=None, error=None, reads=0)

    def fake_read():
        read.reads += 1
        if read.error is not None:
            raise read.error
        return read.value

    monkeypatch.setattr(agents_module, "_stream_started_at", fake_read)
    return read


@pytest.fixture
def stt_script(monkeypatch, clock):
    """What the stand-in for Agent.default.stt_node yields: a test fills `events`.

    The stand-in reads all the audio, then yields the events, each at its own moment on the pinned
    wall clock: `arrivals[i]` if the test gave one, else `latency` seconds after the audio ended,
    which is late enough that no word can end after it.
    """
    script = SimpleNamespace(events=[], frames=[], arrivals=[], latency=10.0)

    async def stand_in(agent, audio, model_settings):
        async for f in audio:
            script.frames.append(f)
        heard_until = clock.value
        for i, ev in enumerate(script.events):
            arrives = (
                script.arrivals[i] if i < len(script.arrivals) else heard_until + script.latency
            )
            clock.value = arrives
            yield ev

    monkeypatch.setattr(Agent.default, "stt_node", stand_in)
    return script


async def run_stt_node(agent, frames):
    async def audio():
        for f in frames:
            yield f

    return [ev async for ev in agent.stt_node(audio(), ModelSettings())]


async def test_a_final_transcript_becomes_a_segment_on_the_call_state(clock, stt_script):
    state = new_state()
    agent = probe_agent(state)
    stt_script.events = [event(INTERIM, "four", 0.40, 0.60, [word("four", 0.42, 0.56)]), four_two()]
    await run_stt_node(agent, [frame(), frame()])
    [segment] = state.pending_segments  # the interim one added nothing
    assert_segment(segment, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_every_event_still_reaches_the_framework_untouched(clock, stt_script):
    events = [
        event(INTERIM, "four", 0.40, 0.60, [word("four", 0.42, 0.56)]),
        four_two(),
        "a plain string",
    ]
    stt_script.events = list(events)
    passed = await run_stt_node(probe_agent(new_state()), [frame()])
    assert len(passed) == 3 and all(a is b for a, b in zip(passed, events, strict=True))


async def test_the_segment_is_recorded_before_the_framework_sees_the_final(clock, stt_script):
    # The framework acts on a FINAL the moment it is yielded, so its segment has to be there first.
    state, recorded_when_seen = new_state(), []
    stt_script.events = [four_two()]

    async def audio():
        yield frame()

    async for _event in probe_agent(state).stt_node(audio(), ModelSettings()):
        recorded_when_seen.append(len(state.pending_segments))
    assert recorded_when_seen == [1]


async def test_segments_pile_up_in_the_order_they_arrive(clock, stt_script):
    state = new_state()
    seven = event(FINAL, "seven", 2.10, 2.50, [word("seven", 2.12, 2.48)])
    stt_script.events = [four_two(), seven]
    await run_stt_node(probe_agent(state), [frame()])
    assert [s.text for s in state.pending_segments] == ["four two", "seven"]


async def test_each_stt_node_call_has_an_anchor_of_its_own(clock, stt_script):
    # A handoff starts a new stt_node and a new Deepgram clock: the second call's t0 comes from its
    # own first frame, not from the first call's.
    state = new_state()
    agent = probe_agent(state)
    stt_script.events = [four_two()]
    clock.value = 1000.02
    await run_stt_node(agent, [frame(), frame()])
    clock.value = 1060.02
    await run_stt_node(agent, [frame()])
    assert [s.t0 for s in state.pending_segments] == pytest.approx([1000.0, 1060.0])


async def test_only_the_first_frame_of_a_call_sets_the_anchor(clock, stt_script):
    state = new_state()
    stt_script.events = [four_two()]

    async def audio():
        yield frame()
        clock.value = 1500.0  # later frames arrive later: they must not move t0
        yield frame()

    [_ async for _ in probe_agent(state).stt_node(audio(), ModelSettings())]
    assert state.pending_segments[0].t0 == pytest.approx(1000.0)


async def test_a_final_before_any_frame_falls_back_to_when_the_event_was_made(clock, stt_script):
    # No frame has been seen, so there is no t0: the notes' fallback ends the segment at created_at.
    state = new_state()
    stt_script.events = [
        event(FINAL, "four two", 0.40, 0.90, [word("four", 0.42, 0.56)], created_at=1000.9)
    ]
    await run_stt_node(probe_agent(state), [])
    assert state.pending_segments[0].t0 == pytest.approx(1000.9 - 0.90)


async def test_the_offset_read_off_the_stream_is_taken_out(clock, stt_script, stream_offset):
    # Agent.default.stt_node sets stream.start_time_offset to the time since the input started, and
    # the Deepgram plugin adds it to every time. Epoch ms is the stream start plus Deepgram's own
    # offset, so the 3 s must go before t0 is added.
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_a_retry_before_any_word_leaves_the_words_on_true_time(
    clock, stt_script, stream_offset
):
    # The websocket dropped and was retried before the first word. The framework added the first
    # connection's lifetime (1 s) to the stream's offset, which is now 4 s. Only the 3 s it had at
    # the first frame belong to the framework; the 1 s really passed, so the words sit 1 s later.
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [four_two(offset=4.0, shift=4.0)]
    await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 1.0, e + 1.0) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 1.40, 1.90, later, t0=1000.0)


async def test_a_jump_in_the_offset_after_a_word_was_heard_is_kept(
    clock, stt_script, stream_offset
):
    stream_offset.value = 3.0
    state = new_state()
    first = event(INTERIM, "four", 3.40, 3.60, [word("four", 3.42, 3.56, 3.0)])
    after_reconnect = four_two(offset=8.0, shift=8.0)
    stt_script.events = [first, after_reconnect]
    await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 5.0, e + 5.0) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 5.40, 5.90, later, t0=1000.0)


async def test_the_offset_of_the_stream_wins_over_the_one_on_the_words(
    clock, stt_script, stream_offset
):
    # The stream's own offset is what the framework gave it. A word that carries a different one
    # has seen it grow, and that growth is time that passed: it must not become the offset.
    stream_offset.value = 0.0  # also: a zero read is a value, not a gap to fill from the words
    state = new_state()
    stt_script.events = [four_two(offset=2.0, shift=2.0)]
    await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 2.0, e + 2.0) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 2.40, 2.90, later, t0=1000.0)


async def test_the_stream_is_read_once_at_the_first_frame_of_each_call(
    clock, stt_script, stream_offset
):
    stream_offset.value = 3.0
    agent = probe_agent(new_state())
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    await run_stt_node(agent, [frame(), frame(), frame()])
    assert stream_offset.reads == 1
    await run_stt_node(
        agent, [frame()]
    )  # a handoff or a rebuilt pipeline: a new call, a new stream
    assert stream_offset.reads == 2


async def test_no_frame_means_no_read(clock, stt_script, stream_offset):
    stt_script.events = [four_two()]
    await run_stt_node(probe_agent(new_state()), [])
    assert stream_offset.reads == 0


# --- a word cannot end after its transcript arrived ---------------------------------------------
#
# With the first-frame offset a stream whose first connection failed while connecting puts every
# word late: the growth is dead time, because that connection never took a frame. The words of a
# first-frame t0 of 1000.0 and a growth of 0.4 s end at 1001.28 here.


async def test_a_word_that_ends_after_its_transcript_arrived_switches_the_stream_to_its_offset(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [grown_by(0.4)]
    stt_script.arrivals = [
        1001.0
    ]  # a word that ends at 1001.28 cannot be in a transcript from 1001.0
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    [record] = switch_records(caplog)
    assert record.levelno == logging.WARNING


async def test_a_switch_lasts_for_the_connection_that_caused_it(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    state = new_state()
    # The first final is impossible with the first-frame offset: its words end at 1001.28, after
    # it arrived (1001.0). The stream switches. So does the second final, which carries the same
    # 3.4 s: it is the same connection, although its words, arriving at 1010, would pass the check.
    # The third carries 3.9 s, so it is another connection. It is judged against the first-frame
    # offset again: its words would end at 1001.78, long before it arrived (1020), so the 0.9 s of
    # growth could be audio, as that of any drop after a failed connect, and stays in whole.
    stt_script.events = [grown_by(0.4), grown_by(0.4), grown_by(0.9)]
    stt_script.arrivals = [1001.0, 1010.0, 1020.0]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    first, second, third = state.pending_segments
    assert_segment(first, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    assert_segment(second, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    later = tuple((w, s + 0.9, e + 0.9) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(third, "four two", 1.30, 1.80, later, t0=1000.0)
    assert len(switch_records(caplog)) == 1


async def test_growth_after_a_switch_that_cannot_be_audio_switches_again(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    state = new_state()
    # The first final switches the stream. The second is another connection, with 3.9 s: kept whole
    # its 0.9 s of growth would put its words at 1001.78, after it arrived (1001.2). Dead time
    # again, so the stream switches again and places that final by its own offset.
    stt_script.events = [grown_by(0.4), grown_by(0.9)]
    stt_script.arrivals = [1001.0, 1001.2]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    for segment in state.pending_segments:
        assert_segment(segment, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    assert len(switch_records(caplog)) == 2  # one line per switch


async def test_a_switch_does_not_spend_the_wait_for_the_first_frame(
    clock, stt_script, stream_offset, stream_started_at, caplog
):
    stream_offset.value, stream_started_at.value = 3.0, 999.7  # a 0.3 s wait
    state = new_state()
    # 1.0 s of growth, 0.3 s of it the wait: 0.7 s is dead, so the stream switches. A later
    # connection's growth counts that first attempt whole, the wait included, so the wait is left
    # out of it again: of the 1.5 s the second final carries 1.2 s stays in.
    stt_script.events = [grown_by(1.0), grown_by(1.5)]
    stt_script.arrivals = [1001.2, 1020.0]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    first, second = state.pending_segments
    assert_segment(first, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    later = tuple((w, s + 1.2, e + 1.2) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(second, "four two", 1.60, 2.10, later, t0=1000.0)  # not 1.5 s of it, nor 0.5
    assert len(switch_records(caplog)) == 1


async def test_a_word_that_ends_before_its_transcript_arrived_keeps_the_first_frame_offset(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [grown_by(0.4)]
    stt_script.arrivals = [1002.0]  # the word ends at 1001.28: possible, so the growth was audio
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 0.4, e + 0.4) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 0.80, 1.30, later, t0=1000.0)
    assert switch_records(caplog) == []


@pytest.mark.parametrize(
    ("ends_after_arrival_s", "switches"), [(0.049, False), (0.051, True)], ids=["49 ms", "51 ms"]
)
async def test_a_word_may_end_50_ms_after_its_transcript_arrived(
    ends_after_arrival_s, switches, clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    stt_script.events = [grown_by(0.4)]
    stt_script.arrivals = [1001.28 - ends_after_arrival_s]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(new_state()), [frame()])
    assert bool(switch_records(caplog)) is switches


async def test_the_check_looks_at_the_last_word_not_at_the_window(
    clock, stt_script, stream_offset, caplog
):
    # The window can run on past the speech, into the silence the model waited through, so it can
    # sit close to the arrival whatever happened to the offset. Only a word's end is evidence.
    stream_offset.value = 3.0
    words = [word("four", 3.82, 3.96, 3.4), word("two", 4.00, 4.28, 3.4)]
    stt_script.events = [
        event(FINAL, "four two", 3.80, 5.90, words)
    ]  # a window to 2.9 s, words to 1.28 s
    stt_script.arrivals = [1001.5]
    state = new_state()
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    assert state.pending_segments[0].end_s == pytest.approx(2.9)  # the growth stayed in
    assert switch_records(caplog) == []


async def test_a_final_whose_words_carry_no_offset_cannot_be_followed(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [four_two(offset=None, shift=3.4)]  # 0.4 s of growth the words do not name
    stt_script.arrivals = [1001.0]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 0.4, e + 0.4) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 0.80, 1.30, later, t0=1000.0)
    assert switch_records(caplog) == []


async def test_a_stream_that_could_not_be_read_never_switches(clock, stt_script, caplog):
    # It follows the offset on the first word that carries one, so there is no first-frame offset
    # to be impossible. Growth after that word is kept, as it always was, however late it ends.
    state = new_state()
    first = event(INTERIM, "four", 3.40, 3.60, [word("four", 3.42, 3.56, 3.0)])
    stt_script.events = [first, grown_by(0.4)]
    stt_script.arrivals = [1000.5, 1001.0]  # the final could not hold a word that ends at 1001.28
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 0.4, e + 0.4) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 0.80, 1.30, later, t0=1000.0)
    assert switch_records(caplog) == []


async def test_an_impossible_end_with_no_growth_is_not_a_switch(
    clock, stt_script, stream_offset, caplog
):
    # Then the offset is not what is wrong (the anchor may be: audio that waited before it was
    # sent), and following it would change nothing but the state of the stream.
    stream_offset.value = 3.0
    state = new_state()
    stt_script.events = [grown_by(0.0)]
    stt_script.arrivals = [1000.5]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    assert switch_records(caplog) == []


async def test_each_call_switches_on_its_own(clock, stt_script, stream_offset, caplog):
    stream_offset.value = 3.0
    state = new_state()
    agent = probe_agent(state)
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        stt_script.events, stt_script.arrivals = [grown_by(0.4)], [1001.0]
        await run_stt_node(agent, [frame()])
        clock.value = 1060.02  # a handoff: a new stream with a first frame of its own
        stt_script.events, stt_script.arrivals = [grown_by(0.4)], [1062.0]  # possible here
        await run_stt_node(agent, [frame()])
    first, second = state.pending_segments
    assert_segment(first, "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    later = tuple((w, s + 0.4, e + 0.4) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(second, "four two", 0.80, 1.30, later, t0=1060.0)  # not switched
    assert len(switch_records(caplog)) == 1


async def test_the_switch_is_logged_without_the_transcript(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = 3.0
    words = [
        word("my", 3.82, 3.96, 3.4),
        word("code", 4.00, 4.28, 3.4),
        word("482913", 4.30, 4.28 + 0.3, 3.4),
    ]
    stt_script.events = [event(FINAL, "my code 482913", 3.80, 4.60, words)]
    stt_script.arrivals = [1001.0]
    with caplog.at_level(logging.DEBUG, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(new_state()), [frame()])
    assert len(switch_records(caplog)) == 1
    assert "482913" not in caplog.text


# --- the wait for the first frame ---------------------------------------------------------------
#
# The stream is made before its audio comes. The base class dates each attempt from its own start,
# so after a retry that wait is in the growth of the offset, though no audio and no Deepgram clock
# ran during it. Here the attempt began at 999.7 and the first sample of audio was captured at
# 1000.0 (t0): a wait of 0.3 s.


async def test_the_wait_for_the_first_frame_is_left_out_of_the_growth(
    clock, stt_script, stream_offset, stream_started_at, caplog
):
    # A growth of 0.9 s is the 0.3 s wait, 0.5 s of audio and the 0.1 s before the retry. The wait
    # was never on Deepgram's clock, so only 0.6 s stays in. Counted whole, the words would end at
    # 1001.78, after the transcript that holds them arrived (at 1001.6), and that would look like a
    # connection that failed while connecting.
    stream_offset.value, stream_started_at.value = 3.0, 999.7
    state = new_state()
    stt_script.events = [grown_by(0.9)]
    stt_script.arrivals = [1001.6]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 0.6, e + 0.6) for w, s, e in FOUR_TWO_WORDS)
    # (0.6 and not 0.58: the wait runs up to the start of the audio, t0, and not to the arrival of
    # the frame that carried its last 20 ms)
    assert_segment(state.pending_segments[0], "four two", 1.00, 1.50, later, t0=1000.0)
    assert switch_records(caplog) == []


async def test_a_wait_with_no_growth_is_not_taken_out_of_the_times(
    clock, stt_script, stream_offset, stream_started_at
):
    # Nothing was added to the offset, so the wait is not in it.
    stream_offset.value, stream_started_at.value = 3.0, 999.7
    state = new_state()
    stt_script.events = [grown_by(0.0)]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_the_wait_cannot_take_out_more_than_the_growth(
    clock, stt_script, stream_offset, stream_started_at
):
    stream_offset.value, stream_started_at.value = 3.0, 999.7  # a 0.3 s wait ...
    state = new_state()
    stt_script.events = [grown_by(0.2)]  # ... in a growth of 0.2 s, which cannot be
    stt_script.arrivals = [1001.0]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_a_connect_failure_longer_than_the_wait_is_still_seen(
    clock, stt_script, stream_offset, stream_started_at, caplog
):
    # 1.0 s of growth, 0.3 s of it the wait: 0.7 s is still dead, so the words would end at 1001.58,
    # after the transcript (1001.2) arrived.
    stream_offset.value, stream_started_at.value = 3.0, 999.7
    state = new_state()
    stt_script.events = [grown_by(1.0)]
    stt_script.arrivals = [1001.2]
    with caplog.at_level(logging.WARNING, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    assert len(switch_records(caplog)) == 1


async def test_audio_captured_before_the_attempt_began_counts_no_wait(
    clock, stt_script, stream_offset, stream_started_at
):
    # Frames that waited for a stream (a backlog) were captured before its attempt began: no wait.
    stream_offset.value, stream_started_at.value = 3.0, 1000.5
    state = new_state()
    stt_script.events = [grown_by(0.4)]
    stt_script.arrivals = [1005.0]
    await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 0.4, e + 0.4) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 0.80, 1.30, later, t0=1000.0)


async def test_the_start_of_the_attempt_is_read_once_at_the_first_frame_of_each_call(
    clock, stt_script, stream_offset, stream_started_at
):
    stream_offset.value, stream_started_at.value = 3.0, 999.7
    agent = probe_agent(new_state())
    stt_script.events = [grown_by(0.0)]
    await run_stt_node(agent, [frame(), frame(), frame()])
    assert stream_started_at.reads == 1
    await run_stt_node(agent, [frame()])  # a new stream, with an attempt of its own
    assert stream_started_at.reads == 2


async def test_the_start_of_the_attempt_is_not_read_when_the_stream_offset_is_not(
    clock, stt_script, stream_started_at
):
    stt_script.events = [four_two()]
    await run_stt_node(probe_agent(new_state()), [frame()])  # no pipeline: no stream to read
    assert stream_started_at.reads == 0


async def test_a_start_time_that_cannot_be_read_counts_no_wait_and_says_so_once(
    clock, stt_script, stream_offset, stream_started_at, caplog
):
    stream_offset.value = 3.0
    stream_started_at.error = AttributeError("no start_time, my code is 482913")
    state = new_state()
    stt_script.events = [grown_by(0.3)]
    stt_script.arrivals = [1005.0]
    with caplog.at_level(logging.DEBUG, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame(), frame(), frame()])
    # Without the wait the growth stays in whole, as it did before the wait was measured.
    later = tuple((w, s + 0.3, e + 0.3) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 0.70, 1.20, later, t0=1000.0)
    [record] = [r for r in caplog.records if r.name == "callback_agent.agents"]
    assert record.levelno == logging.ERROR and "AttributeError" in record.getMessage()
    assert "482913" not in caplog.text


async def test_an_unreadable_stream_falls_back_to_the_first_words_offset_and_says_so_once(
    clock, stt_script, stream_offset, caplog
):
    stream_offset.value = None  # what the read gives when it finds no stream
    state = new_state()
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    with caplog.at_level(logging.DEBUG, logger="callback_agent.agents"):
        await run_stt_node(probe_agent(state), [frame(), frame(), frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    [record] = [r for r in caplog.records if r.name == "callback_agent.agents"]
    assert record.levelno == logging.WARNING and "first word" in record.getMessage()


async def test_a_failing_read_falls_back_to_the_first_words_offset_and_says_so_once(
    clock, stt_script, stream_offset, caplog
):
    # The read leans on private names of livekit-agents. When they go, STT must not go with them,
    # and the log must say what failed without quoting the exception's message.
    stream_offset.error = AttributeError("no attribute _recognize_stream, my code is 482913")
    state = new_state()
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    with caplog.at_level(logging.DEBUG, logger="callback_agent.agents"):
        passed = await run_stt_node(probe_agent(state), [frame(), frame(), frame()])
    assert len(passed) == 1
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)
    [record] = [r for r in caplog.records if r.name == "callback_agent.agents"]
    assert record.levelno == logging.ERROR and "AttributeError" in record.getMessage()
    assert "482913" not in caplog.text


async def test_without_a_readable_stream_the_offset_on_the_first_word_is_taken_out(
    clock, stt_script
):
    # The fallback: nothing here runs a pipeline, so the read finds no stream.
    state = new_state()
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_without_a_readable_stream_a_jump_after_the_first_word_is_kept(clock, stt_script):
    # The fallback, with a reconnect after a word was heard: that word's offset is the stream's.
    state = new_state()
    first = event(INTERIM, "four", 3.40, 3.60, [word("four", 3.42, 3.56, 3.0)])
    after_reconnect = four_two(offset=8.0, shift=8.0)
    stt_script.events = [first, after_reconnect]
    await run_stt_node(probe_agent(state), [frame()])
    later = tuple((w, s + 5.0, e + 5.0) for w, s, e in FOUR_TWO_WORDS)
    assert_segment(state.pending_segments[0], "four two", 5.40, 5.90, later, t0=1000.0)


async def test_words_with_no_offset_are_used_as_they_are(clock, stt_script):
    state = new_state()
    stt_script.events = [four_two(offset=None)]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, FOUR_TWO_WORDS, t0=1000.0)


async def test_a_final_with_no_words_keeps_its_text_and_window(clock, stt_script):
    state = new_state()
    stt_script.events = [event(FINAL, "four two", 0.40, 0.90, None)]
    await run_stt_node(probe_agent(state), [frame()])
    assert_segment(state.pending_segments[0], "four two", 0.40, 0.90, (), t0=1000.0)


@pytest.mark.parametrize(
    "kind",
    [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ],
)
async def test_only_final_transcripts_are_recorded(kind, clock, stt_script):
    state = new_state()
    stt_script.events = [event(kind, "four two", 0.40, 0.90, [word("four", 0.42, 0.56)])]
    await run_stt_node(probe_agent(state), [frame()])
    assert state.pending_segments == []


async def test_a_final_without_text_or_alternatives_records_nothing(clock, stt_script):
    state = new_state()
    stt_script.events = [
        event(FINAL, "", 0.40, 0.90, [word("four", 0.42, 0.56)]),
        stt.SpeechEvent(type=FINAL, alternatives=[]),
    ]
    await run_stt_node(probe_agent(state), [frame()])
    assert state.pending_segments == []


async def test_a_final_that_cannot_be_recorded_is_dropped_and_stt_goes_on(
    clock, stt_script, caplog
):
    state = new_state()
    broken = event(
        FINAL, "my code is 482913", 0.40, 0.90, [TimedString("my")]
    )  # no word times at all
    stt_script.events = [broken, four_two()]
    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        passed = await run_stt_node(probe_agent(state), [frame()])
    assert passed[0] is broken and len(passed) == 2  # the stream never noticed
    assert [s.text for s in state.pending_segments] == ["four two"]  # only the bad one is lost
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "482913" not in caplog.text  # the log says what failed, not what was said


async def test_a_session_without_userdata_does_not_break_stt(clock, stt_script, caplog):
    class NoUserdata:
        @property
        def userdata(self):
            raise ValueError("AgentSession userdata is not set")  # what the real session does

    stt_script.events = [four_two()]
    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        passed = await run_stt_node(ProbeAgent(NoUserdata()), [frame()])
    assert len(passed) == 1
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


async def test_a_failure_is_logged_without_its_message(clock, stt_script, caplog):
    # An exception's message can quote what was being handled, and that is a caller's turn.
    class Quoting:
        @property
        def userdata(self):
            raise RuntimeError("could not file: my code is 482913")

    stt_script.events = [four_two()]
    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        await run_stt_node(ProbeAgent(Quoting()), [frame()])
    assert "RuntimeError" in caplog.text
    assert "482913" not in caplog.text


async def test_a_frame_that_cannot_anchor_the_clock_does_not_break_stt(clock, stt_script, caplog):
    class BadFrame:
        @property
        def duration(self):
            raise RuntimeError("no duration")

    state = new_state()
    stt_script.events = [four_two()]
    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        passed = await run_stt_node(probe_agent(state), [BadFrame(), BadFrame(), BadFrame()])
    assert len(passed) == 1 and len(stt_script.frames) == 3  # audio kept flowing
    assert (
        len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    )  # logged once, not per frame
    # No t0, so the segment hangs off created_at, and is still recorded.
    assert state.pending_segments[0].t0 == pytest.approx(1000.9 - 0.90)


async def test_the_tap_and_the_segments_work_together(clock, stt_script):
    tapped, state = [], new_state()
    agent = ProbeAgent(SimpleNamespace(userdata=state))
    agent._frame_tap = tapped.append
    stt_script.events = [four_two()]
    frames = [frame(), frame()]
    await run_stt_node(agent, frames)
    assert tapped == frames and len(state.pending_segments) == 1


async def test_it_works_through_a_real_session(clock, stt_script):
    state = new_state()
    async with AgentSession(llm=FakeLLM(fake_responses=[]), userdata=state) as session:
        agent = RealSessionAgent()
        await session.start(agent)
        stt_script.events = [four_two()]
        await run_stt_node(agent, [frame()])
    assert [s.text for s in state.pending_segments] == ["four two"]


async def test_a_spoken_turn_runs_from_the_stt_event_to_epoch_ms(clock, stt_script):
    # The whole path: the mixin hears the FINAL as the framework reports it (3 s of offset on
    # every time), and the wiring forwards the user's message as the detector will receive it.
    state, sent, handlers = new_state(), [], {}
    session = SimpleNamespace(
        on=lambda event, cb=None: handlers.setdefault(event, cb), agent_state="x"
    )
    wire_turn_forwarding(session, state, SimpleNamespace(submit=sent.append))
    stt_script.events = [four_two(offset=3.0, shift=3.0)]
    await run_stt_node(probe_agent(state), [frame()])
    said = ConversationItemAddedEvent(item=ChatMessage(role="user", content=["four two"]))
    handlers["conversation_item_added"](said)
    assert sent == [
        Turn(
            role=Role.USER,
            text="four two",
            start_ms=1000400,
            end_ms=1000900,
            words=[
                Word(w="four", start_ms=1000420, end_ms=1000560),
                Word(w="two", start_ms=1000600, end_ms=1000880),
            ],
        )
    ]
