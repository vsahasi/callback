"""The caller's speech reaches the detector: the mixin feeds the capture on the frame clock, the
wiring cuts segments at the speech events and stamps the leg, the forwarder carries them under the
turns' attach rule and fails quietly, and nothing anywhere holds the audio or its base64.

The `clock` and `stt_script` fixtures come from test_stt_segments, as parameters, which ruff reads
as redefinitions of the imported names (F811)."""

# ruff: noqa: F811

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import respx
from callback_shared.contract import Channel, Role, SpeechEvent, SpeechEventType, Turn
from livekit import rtc
from livekit.agents import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    UserStateChangedEvent,
)
from livekit.agents.llm import ChatMessage
from test_stt_segments import ProbeAgent, clock, four_two, run_stt_node, stt_script  # noqa: F401
from test_turns import FakeDetector, flush, pauses, running  # noqa: F401

from callback_agent.audio_segments import AudioSegment, SegmentCapture
from callback_agent.callback_leg import CALLEE_PREFIX
from callback_agent.detector import DetectorClient, DetectorError
from callback_agent.session_wiring import wire_turn_forwarding
from callback_agent.state import CallState
from callback_agent.transcript import Transcript
from callback_agent.turns import MAX_BUFFERED_AUDIO_S, FinalSegment, Forwarder

T0 = 1_700_000_000.0


EVENT = SpeechEvent(type=SpeechEventType.AGENT_SPEECH_START, ts_ms=2_100)
A_TURN = Turn(role=Role.AGENT, text="Thank you.", start_ms=2_100, end_ms=2_900)


def new_state(**fields) -> CallState:
    return CallState(room_name="room", caller_identity="caller", channel=Channel.PHONE, **fields)


def segment(seconds: float = 1.0, track: str = "caller", start_ms: int = 1_700_000_000_000):
    return AudioSegment(
        track_id=track,
        start_ms=start_ms,
        end_ms=start_ms + round(seconds * 1000),
        samples=np.full(round(seconds * 24000), 1000, dtype=np.int16),
        sample_rate_hz=24000,
    )


def frame(ms: int = 20, sample_rate: int = 24000) -> rtc.AudioFrame:
    n = sample_rate * ms // 1000
    return rtc.AudioFrame(np.full(n, 1000, dtype=np.int16).tobytes(), sample_rate, 1, n)


class RecordingCapture(SegmentCapture):
    def __init__(self) -> None:
        super().__init__()
        self.fed: list[tuple[float, float]] = []  # (start_s, duration_s) of every frame fed

    def add_frame(self, frame: rtc.AudioFrame, start_s: float) -> None:
        self.fed.append((start_s, frame.duration))
        super().add_frame(frame, start_s)


# --- the mixin feeds the capture on the frame clock ----------------------------------------------


async def test_every_frame_reaches_the_capture_with_its_start_on_the_anchors_clock(
    clock, stt_script
):
    state = new_state()
    state.audio_capture = RecordingCapture()
    stt_script.events = [four_two()]
    await run_stt_node(ProbeAgent(SimpleNamespace(userdata=state)), [frame(), frame(), frame()])
    # The pinned clock puts the first 20 ms frame's t0 at 1000.0; each frame starts where the
    # last ended, whatever the wall clock says meanwhile.
    assert [s for s, _ in state.audio_capture.fed] == pytest.approx([1000.0, 1000.02, 1000.04])


async def test_each_stt_node_call_feeds_from_its_own_anchor(clock, stt_script):
    state = new_state()
    state.audio_capture = RecordingCapture()
    stt_script.events = [four_two()]
    agent = ProbeAgent(SimpleNamespace(userdata=state))
    await run_stt_node(agent, [frame(), frame()])
    clock.value = 1060.02  # a task handoff: a new stt_node call, a new first frame
    await run_stt_node(agent, [frame()])
    assert [s for s, _ in state.audio_capture.fed] == pytest.approx([1000.0, 1000.02, 1060.0])


async def test_a_call_state_without_a_capture_is_fine(clock, stt_script):
    state = new_state()
    stt_script.events = [four_two()]
    passed = await run_stt_node(ProbeAgent(SimpleNamespace(userdata=state)), [frame()])
    assert len(passed) == 1 and len(state.pending_segments) == 1


async def test_a_capture_that_fails_is_given_up_without_touching_the_audio(
    clock, stt_script, caplog
):
    class Broken(SegmentCapture):
        def add_frame(self, frame, start_s):
            raise RuntimeError("the caller said 482913")

    state = new_state()
    state.audio_capture = Broken()
    stt_script.events = [four_two()]
    with caplog.at_level(logging.ERROR, logger="callback_agent.agents"):
        passed = await run_stt_node(
            ProbeAgent(SimpleNamespace(userdata=state)), [frame(), frame(), frame()]
        )
    assert len(passed) == 1 and len(stt_script.frames) == 3
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # once, not per frame
    assert "RuntimeError" in caplog.text and "482913" not in caplog.text


# --- the wiring cuts segments at the speech events and stamps the leg ----------------------------


class FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}

    def on(self, event, callback=None):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def fire(self, event, ev) -> None:
        for handler in self.handlers.get(event, []):
            handler(ev)


class FakeForwarder:
    def __init__(self) -> None:
        self.sent = []

    def submit(self, item) -> None:
        self.sent.append(item)


def user_changed(old: str, new: str, at: float) -> UserStateChangedEvent:
    return UserStateChangedEvent(old_state=old, new_state=new, created_at=at)


def wire(**kwargs):
    session, state, forwarder = FakeSession(), new_state(), FakeForwarder()
    wire_turn_forwarding(session, state, forwarder, **kwargs)
    return SimpleNamespace(session=session, state=state, sent=forwarder.sent)


def segments(wired) -> list[AudioSegment]:
    """The segments forwarded, leaving the USER_SPEECH_START events and the turns aside."""
    return [item for item in wired.sent if isinstance(item, AudioSegment)]


def hear(wired, seconds: float, start_s: float = T0) -> None:
    """`seconds` of frames go through the capture, as the mixin would feed them."""
    for k in range(round(seconds * 50)):
        wired.state.audio_capture.add_frame(frame(), start_s + k * 0.02)


def test_the_wiring_puts_a_capture_on_the_call_state():
    wired = wire()
    assert isinstance(wired.state.audio_capture, SegmentCapture)


def test_the_callers_speech_becomes_one_segment_from_speaking_to_listening():
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 2.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.0))
    [item] = segments(wired)
    assert item.track_id == "caller"
    assert (item.start_ms, item.end_ms) == (1_700_000_000_000, 1_700_000_002_000)


def test_the_user_turn_commit_also_closes_an_open_segment():
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 2.0)
    message = ChatMessage(role="user", content=["hello there"])
    wired.session.fire(
        "conversation_item_added", ConversationItemAddedEvent(item=message, created_at=T0 + 2.0)
    )
    kinds = [type(item).__name__ for item in wired.sent]
    assert kinds == ["SpeechEvent", "AudioSegment", "Turn"]  # the audio goes out before its turn
    assert not wired.state.audio_capture.open


def test_the_callback_legs_segments_carry_the_callback_track():
    wired = wire()
    wired.state.caller_identity = CALLEE_PREFIX + "0a1b2c3d"  # the callee answered
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 1.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 1.0))
    [item] = segments(wired)
    assert item.track_id == "callback"


def test_nothing_is_cut_while_a_callback_is_being_dialed():
    # Between the hang-up on the caller and the callee's answer the stream carries a silence frame
    # and the callee's first instants: nobody's voice to judge.
    wired = wire()
    wired.state.callback_in_progress = True
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 2.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.0))
    assert segments(wired) == [] and not wired.state.audio_capture.open


def test_a_segment_is_logged_by_its_leg_and_times_only(caplog):
    wired = wire(log_turns=True)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
        hear(wired, 2.0)
        wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2))
    [record] = [r for r in caplog.records if "audio" in r.getMessage()]
    assert "caller" in record.getMessage() and "2000" in record.getMessage()
    assert max(len(line) for line in caplog.text.splitlines()) < 200


def test_the_transcript_records_a_segments_leg_and_duration_and_nothing_else(tmp_path):
    transcript = Transcript(tmp_path / "call.jsonl")
    wired = wire(transcript=transcript)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 2.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.0))
    [record] = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert record == {
        "kind": "audio_segment",
        "leg": "caller",
        "start_ms": 1_700_000_000_000,
        "end_ms": 1_700_000_002_000,
        "duration_ms": 2000,
    }
    assert len(transcript.path.read_text()) < 200


def agent_speaks(wired, start_s: float, end_s: float | None = None) -> None:
    wired.session.fire("agent_state_changed", agent_changed("listening", "speaking", at=start_s))
    if end_s is not None:
        agent_stops(wired, end_s)


def agent_stops(wired, at: float) -> None:
    wired.session.fire("agent_state_changed", agent_changed("speaking", "listening", at=at))


def agent_changed(old: str, new: str, at: float) -> AgentStateChangedEvent:
    return AgentStateChangedEvent(old_state=old, new_state=new, created_at=at)


def commits(wired, text: str, at: float, *segments: FinalSegment) -> None:
    """The user's message arrives, made of `segments` (none: typed input, nothing interrupted)."""
    wired.state.pending_segments.extend(segments)
    message = ChatMessage(role="user", content=[text])
    wired.session.fire(
        "conversation_item_added", ConversationItemAddedEvent(item=message, created_at=at)
    )


def final(text: str, start_s: float, end_s: float) -> FinalSegment:
    """A FINAL whose words run from `start_s` to `end_s`, epoch seconds (t0 is the epoch)."""
    return FinalSegment(
        text=text, start_s=start_s, end_s=end_s, words=((text, start_s, end_s),), t0=0
    )


# --- the callback leg: the open segment dies with the relink, the leg is stamped at the end -------


def test_the_leg_is_stamped_when_the_segment_ends_not_when_it_begins():
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 1.0)
    wired.state.caller_identity = CALLEE_PREFIX + "0a1b2c3d"
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 1.0))
    [item] = segments(wired)
    assert item.track_id == "callback"


def test_a_segment_open_when_the_callback_begins_is_discarded_not_sent():
    # The caller was mid-sentence when the leg relinked: nothing of that stretch, and nothing of
    # the callee's first instants, is a segment of anyone's.
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 1.0)
    wired.state.callback_in_progress = True
    wired.state.audio_capture.discard()  # what dial_and_link does at the relink
    hear(wired, 2.0, start_s=T0 + 1.0)  # the silence frame and the callee's first instants
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 2.0))
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 3.0))
    assert segments(wired) == [] and not wired.state.audio_capture.open
    wired.state.callback_in_progress = False
    wired.state.caller_identity = CALLEE_PREFIX + "0a1b2c3d"
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 10.0))
    hear(wired, 2.0, start_s=T0 + 10.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 12.0))
    [item] = segments(wired)
    assert item.track_id == "callback" and item.start_ms == 1_700_000_010_000


def test_a_user_state_change_during_the_dial_also_discards_an_open_segment():
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 1.0)
    wired.state.callback_in_progress = True
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 1.0))
    assert not wired.state.audio_capture.open
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.0))
    assert segments(wired) == []


# --- speech heard over the agent's own voice: echo, unless the turn was an interruption ----------


def test_speech_that_began_over_the_agent_is_dropped_when_the_turn_did_not_interrupt(caplog):
    wired = wire()
    agent_speaks(wired, T0, None)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 1.0))
    hear(wired, 0.7, start_s=T0 + 1.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 1.7))
    assert segments(wired) == []  # held: echo of the agent's line, or a barge-in, not known yet
    agent_stops(wired, T0 + 3.0)
    with caplog.at_level(logging.INFO, logger="callback_agent.session_wiring"):
        commits(wired, "yes please", T0 + 6.0)  # typed-like: no words, so nothing interrupted
        wired.session.fire("close", None)
    assert segments(wired) == []
    assert "1" in caplog.text and "echo" in caplog.text.lower()


def test_speech_that_began_over_the_agent_goes_out_when_its_turn_interrupted():
    wired = wire()
    agent_speaks(wired, T0, None)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 1.0))
    hear(wired, 1.5, start_s=T0 + 1.0)
    agent_stops(wired, T0 + 1.5)  # the agent gives way
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.5))
    assert segments(wired) == []
    commits(wired, "stop", T0 + 2.6, final("stop", T0 + 1.2, T0 + 2.4))
    kinds = [type(item).__name__ for item in wired.sent]
    assert kinds[-2:] == ["AudioSegment", "Turn"]  # the audio goes out just before its turn
    assert wired.sent[-1].interrupted is True
    assert segments(wired)[0].start_ms == 1_700_000_001_000


def test_speech_that_began_after_the_agent_stopped_is_sent_at_once():
    wired = wire()
    agent_speaks(wired, T0, T0 + 2.0)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 2.4))
    hear(wired, 2.0, start_s=T0 + 2.4)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 4.4))
    assert len(segments(wired)) == 1


# --- a deny has locked the call: nothing more is cut --------------------------------------------


def test_no_segment_is_cut_once_a_deny_has_locked_the_call():
    wired = wire()
    wired.state.locked = True
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 2.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 2.0))
    assert segments(wired) == [] and not wired.state.audio_capture.open


# --- the timing the detector's agent_behaviour signal reads goes with each segment --------------


def test_a_segment_carries_the_reply_latency_the_durations_and_the_barge_ins():
    wired = wire()
    agent_speaks(wired, T0, T0 + 2.0)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 2.4))
    hear(wired, 2.0, start_s=T0 + 2.4)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 4.4))
    [first] = segments(wired)
    assert first.metadata == {
        "turn_latencies_ms": [400],
        "response_durations_ms": [2000],
        "barge_ins": 0,
    }
    # A barge-in counts once its turn commits, and the next segment carries the count.
    agent_speaks(wired, T0 + 5.0, None)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 6.0))
    hear(wired, 1.0, start_s=T0 + 6.0)
    agent_stops(wired, T0 + 6.3)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 7.0))
    commits(wired, "wait", T0 + 7.1, final("wait", T0 + 6.1, T0 + 6.9))
    _, second = segments(wired)
    assert second.metadata["barge_ins"] == 1
    assert second.metadata["turn_latencies_ms"] == [400]  # a barge-in is not a reply latency
    assert second.metadata["response_durations_ms"] == [2000, 1000]
    agent_speaks(wired, T0 + 8.0, T0 + 9.0)
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0 + 9.5))
    hear(wired, 1.0, start_s=T0 + 9.5)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 10.5))
    third = segments(wired)[-1]
    assert third.metadata["turn_latencies_ms"] == [400, 500]
    assert third.metadata["barge_ins"] == 1


def test_a_first_segment_with_no_agent_speech_before_it_has_no_latency():
    wired = wire()
    wired.session.fire("user_state_changed", user_changed("listening", "speaking", at=T0))
    hear(wired, 1.0)
    wired.session.fire("user_state_changed", user_changed("speaking", "listening", at=T0 + 1.0))
    [item] = segments(wired)
    assert item.metadata["turn_latencies_ms"] == []


# --- the forwarder carries segments under the turns' rules ---------------------------------------


class AudioDetector(FakeDetector):
    """FakeDetector that also takes speech segments. `audio_script` fails them in order."""

    def __init__(self, audio_script=()) -> None:
        super().__init__()
        self._audio_script = iter(audio_script)
        self.segments: list[tuple[str, int, str | None, AudioSegment]] = []

    async def post_speech_segment(self, session_id, segment, *, sequence, claimed_user_id):
        failure = next(self._audio_script, None)
        if failure is not None:
            raise failure
        self.segments.append((session_id, sequence, claimed_user_id, segment))


async def test_nothing_is_posted_before_the_session_attaches_then_segments_go_in_order():
    detector = AudioDetector()
    async with running(detector) as forwarder:
        first, second = segment(1.0), segment(2.0, start_ms=1_700_000_005_000)
        forwarder.submit(first)
        forwarder.submit(second)
        await asyncio.sleep(0.01)
        assert detector.segments == []
        forwarder.attach("ses_1", claimed_user_id="u_eric")
        await flush(forwarder)
    assert detector.segments == [
        ("ses_1", 0, "u_eric", first),
        ("ses_1", 1, "u_eric", second),  # the sequence counts up per call
    ]


async def test_audio_buffered_with_no_session_is_dropped_on_close_and_never_sent(caplog):
    detector = AudioDetector()
    forwarder = Forwarder(detector)
    consumer = asyncio.create_task(forwarder.run())
    forwarder.submit(segment(3.0))
    await asyncio.sleep(0.01)
    with caplog.at_level(logging.INFO, logger="callback_agent.turns"):
        await forwarder.aclose()
    await consumer
    assert detector.segments == []
    assert "1 dropped unsent" in caplog.text


async def test_the_unattached_buffer_keeps_only_the_last_60_s_of_audio(caplog):
    assert MAX_BUFFERED_AUDIO_S == 60.0
    detector = AudioDetector()
    with caplog.at_level(logging.INFO, logger="callback_agent.turns"):
        async with running(detector) as forwarder:
            pieces = [segment(20.0, start_ms=1_700_000_000_000 + k * 20_000) for k in range(4)]
            for piece in pieces:
                forwarder.submit(piece)
            forwarder.submit(EVENT)
            forwarder.attach("ses_1")
            await flush(forwarder)
    assert [s for _, _, _, s in detector.segments] == pieces[1:]  # the oldest 20 s went
    assert detector.posted == [("ses_1", EVENT)]  # the turns and events are untouched
    assert "1 dropped unsent" in caplog.text


async def test_a_refused_post_is_logged_once_with_its_status_and_the_call_goes_on(caplog):
    detector = AudioDetector(
        [DetectorError("POST /events: HTTP 404"), DetectorError("POST /events: ReadTimeout")]
    )
    with caplog.at_level(logging.INFO, logger="callback_agent.turns"):
        async with running(detector) as forwarder:
            forwarder.attach("ses_1")
            forwarder.submit(segment(1.0))
            forwarder.submit(segment(1.0))
            forwarder.submit(segment(1.0))
            forwarder.submit(A_TURN)
            await flush(forwarder)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "HTTP 404" in warnings[0].getMessage()
    assert len(detector.segments) == 1  # the third went through
    assert detector.posted == [("ses_1", A_TURN)]  # the turn was not held up
    assert "2 refused or failed" in caplog.text
    assert max(len(line) for line in caplog.text.splitlines()) < 200


async def test_a_bug_in_the_audio_path_is_logged_once_too(caplog):
    detector = AudioDetector([RuntimeError("boom"), RuntimeError("boom")])
    with caplog.at_level(logging.INFO, logger="callback_agent.turns"):
        async with running(detector) as forwarder:
            forwarder.attach("ses_1")
            forwarder.submit(segment(1.0))
            forwarder.submit(segment(1.0))
            await flush(forwarder)
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


async def test_the_real_client_posts_the_segment_to_events_and_a_404_raises_nothing(caplog):
    base = "https://detector.test"
    with respx.mock(base_url=base) as api:
        events = api.post("/events").mock(
            side_effect=[httpx.Response(404, json={"error": {"code": "session_not_found"}})]
        )
        client = DetectorClient(base, "tok")
        try:
            with caplog.at_level(logging.DEBUG, logger="callback_agent.turns"):
                async with running(client) as forwarder:
                    forwarder.attach("ses_0123456789", claimed_user_id="u_eric")
                    forwarder.submit(segment(1.0))
                    await flush(forwarder)
        finally:
            await client.aclose()
    body = json.loads(events.calls.last.request.content)
    assert body["session_id"] == "ses_0123456789" and body["claimed_user_id"] == "u_eric"
    assert body["event"] == "speech_segment" and body["sequence"] == 0
    assert body["audio"]["uri"].startswith("data:audio/wav;base64,UklGR")  # "RIFF"
    assert "UklGR" not in caplog.text  # no base64 in any log line
    assert max(len(line) for line in caplog.text.splitlines()) < 200
