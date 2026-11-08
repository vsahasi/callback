"""TimedSttMixin through the REAL Agent.default.stt_node, inside a real AgentSession.

Nothing between the audio and the mixin is a stand-in: frames go through the session's own audio
pipeline, which runs the mixin's stt_node, which wraps the framework's default node. Only the STT is
fake, and it has no network. Its stream follows what matters here of the Deepgram v2 plugin: every
time it reports is the seconds of audio it has received on the current connection plus
`self.start_time_offset`, and each word repeats that offset in TimedString.start_time_offset. When
the connection drops, the base class retries after 0.1 s and adds the connection's lifetime to the
offset, as it does for the real plugin.

A connection can fail in two ways. After it has taken audio it drops: the next connection starts
where this one stopped. While connecting it fails before taking a frame, because the plugin connects
before it starts to send: the frames wait in the input channel and the next connection starts at the
first of them. The base class adds the lifetime and the wait to the offset either way.

A stream can also exist for a while before its first frame (`idle_s`): at the start of a call the
session creates it inside start(), and the caller's audio is forwarded only once start() has
returned. That wait is part of the first connection's lifetime, which the base class adds to the
offset after a retry, but Deepgram's clock does not run during it.

A script can also give a second word, said on the connection after the one that said the first:
that one drops once it has taken enough audio, so the next connection starts where it stopped.

Every frame carries the moment its audio was captured, so the true time of a word is known.
"""

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from callback_shared.contract import Channel
from livekit import rtc
from livekit.agents import Agent, AgentSession, APIConnectionError, stt
from livekit.agents.cli.log import JsonFormatter
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, TimedString
from livekit.agents.voice import io as livekit_io

import callback_agent.agents as agents_module
from callback_agent.agents import TimedSttMixin
from callback_agent.safelog import PiiExtraFilter, install_pii_filter
from callback_agent.state import CallState

SAMPLE_RATE = 16000
FRAME_S = 0.02
STREAM_DELAY_S = (
    0.3  # the stream is made this late after the session starts: its offset is at least this
)
WORD_S = 0.2  # how long the word the fake stream hears lasts


@dataclass
class Script:
    """What the fake stream does. Times are seconds of audio received on a connection."""

    word_at: float  # where the word starts on the connection that reports it
    speak_on: int = 1  # the connection that reports it
    drop_after_s: float | None = None  # the first connection drops once it has received this much
    connect_fail_after_s: float | None = (
        None  # or it fails this long into connecting, with no audio
    )
    # How long the stream exists before its first frame arrives: no audio, and no Deepgram clock.
    idle_s: float = 0.0
    # A second word, on the connection after the one that said the first. That one drops once it
    # has received `first_drops_after_s`, with audio taken, so the next starts where it stopped.
    second_word_at: float | None = None
    first_drops_after_s: float = 1.5
    # How long after the word ends the transcript arrives. A real final trails the speech by the
    # time the model needs to be sure the turn is over. 0.25 s keeps the two readings of a
    # reconnect apart with room to spare: one that carried audio ends 0.1 s late and moves with the
    # jitter of the frame it dropped on (50 ms at most, measured), far from the 50 ms of slack
    # beyond the latency; a failed connect of 0.3 s ends 0.4 s late and does not move at all.
    final_latency_s: float = 0.25


@dataclass
class Reported:
    truth_s: float  # when the word was really captured
    start_s: float  # the start time the stream reported: Deepgram's offset plus the framework's
    offset_s: float  # the stream's start_time_offset when it reported


class ScriptedStream(stt.RecognizeStream):
    def __init__(self, *, stt, conn_options, script: Script) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._script = script
        self.attempts = 0
        self.created_at = time.time()
        self.first_frame_after_s: float | None = None  # how long the stream waited for its audio
        self.offset_at_first_frame: float | None = None
        self.reported: Reported | None = None  # the first word
        self.reports: list[Reported] = []  # every word, in the order it was said

    def push_frame(self, frame) -> None:
        if self.offset_at_first_frame is None:
            self.offset_at_first_frame = self.start_time_offset  # the offset the first frame meets
            self.first_frame_after_s = time.time() - self.created_at
        super().push_frame(frame)

    async def _run(self) -> None:
        self.attempts += 1
        if self.attempts == 1 and self._script.connect_fail_after_s is not None:
            await asyncio.sleep(self._script.connect_fail_after_s)  # the handshake hangs ...
            raise APIConnectionError("simulated connect failure")  # ... and no frame was taken
        received, zero = 0.0, None  # zero: when the first sample of this connection was captured
        async for frame in self._input_ch:
            if isinstance(frame, self._FlushSentinel):
                continue
            if zero is None:
                zero = frame.userdata["captured_at"]
            received += frame.duration
            drop = self._script.drop_after_s
            if self.attempts == 1 and drop is not None and received >= drop:
                raise APIConnectionError("simulated drop")  # retryable: the base class retries
            if self.attempts == self._script.speak_on:
                if self.reported is None and self._heard(received, self._script.word_at):
                    self._report(zero, self._script.word_at)
                elif (
                    self._script.second_word_at is not None
                    and received >= self._script.first_drops_after_s
                ):
                    raise APIConnectionError("simulated drop")  # after it took audio
            elif (
                self.attempts == self._script.speak_on + 1
                and self._script.second_word_at is not None
                and len(self.reports) == 1
                and self._heard(received, self._script.second_word_at)
            ):
                self._report(zero, self._script.second_word_at)

    def _heard(self, received: float, word_at: float) -> bool:
        """The connection has taken the word, and the final latency after it."""
        return received >= word_at + WORD_S + self._script.final_latency_s

    def _report(self, zero: float, raw_start: float) -> None:
        offset = self.start_time_offset
        word = TimedString(
            "four",
            start_time=raw_start + offset,
            end_time=raw_start + WORD_S + offset,
            start_time_offset=offset,
        )
        data = stt.SpeechData(
            language="en",
            text="four",
            start_time=raw_start - 0.02 + offset,
            end_time=raw_start + WORD_S + 0.02 + offset,
            words=[word],
        )
        report = Reported(zero + raw_start, raw_start + offset, offset)
        self.reports.append(report)
        self.reported = self.reported or report
        final = stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[data])
        self._event_ch.send_nowait(final)


class ScriptedSTT(stt.STT):
    def __init__(self, script: Script) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._script = script
        self.streams: list[ScriptedStream] = []

    async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options):
        raise NotImplementedError

    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        stream = ScriptedStream(stt=self, conn_options=conn_options, script=self._script)
        self.streams.append(stream)
        return stream


class PacedAudio(livekit_io.AudioInput):
    """Hands the session 20 ms frames in real time, once `may_start` is set.

    Frame k is captured at start + k * 20 ms and reaches the agent when its audio is over. Pacing
    follows absolute deadlines, so a slow loop makes frames late but never shifts the capture times.
    The first frame comes `idle_s` after `may_start`, which is when the stream is made.
    """

    def __init__(self, may_start: asyncio.Event, frames: int, idle_s: float = 0.0) -> None:
        super().__init__(label="paced test audio")
        self._may_start = may_start
        self._idle_s = idle_s
        self._frames = frames
        self._sent = 0
        self._start = 0.0

    async def __anext__(self) -> rtc.AudioFrame:
        if self._sent == 0:
            await self._may_start.wait()
            await asyncio.sleep(self._idle_s)  # the stream exists, and has no audio yet
            self._start = time.time()
        if self._sent >= self._frames:
            raise StopAsyncIteration
        captured_at = self._start + self._sent * FRAME_S
        self._sent += 1
        await asyncio.sleep(max(0.0, captured_at + FRAME_S - time.time()))
        frame = rtc.AudioFrame.create(SAMPLE_RATE, 1, round(SAMPLE_RATE * FRAME_S))
        frame.userdata["captured_at"] = captured_at
        return frame


class LateStreamAgent(TimedSttMixin, Agent):
    """Makes its STT stream `delay_s` after the session starts, then lets the audio begin.

    The framework offsets a stream's times by the time since the session started, which is a few
    milliseconds when everything is warm. Making the stream late makes that offset large and
    certain, and the audio starts only once the stream exists, so no frame waits for it.
    """

    def __init__(self, delay_s: float, audio_may_start: asyncio.Event) -> None:
        super().__init__(instructions="test")
        self._delay_s = delay_s
        self._audio_may_start = audio_may_start

    async def stt_node(self, audio, model_settings):
        await asyncio.sleep(self._delay_s)
        self._audio_may_start.set()
        async for event in super().stt_node(audio, model_settings):
            yield event


@pytest.fixture
def read_offsets(monkeypatch):
    """Spies on what the mixin reads off the STT stream; the real read still happens."""
    real = agents_module._stream_start_time_offset
    seen: list[float | None] = []

    def spy():
        value = real()
        seen.append(value)
        return value

    monkeypatch.setattr(agents_module, "_stream_start_time_offset", spy)
    return seen


@pytest.fixture
def read_started_at(monkeypatch):
    """Spies on when the mixin reads the attempt to have begun; the real read still happens."""
    real = agents_module._stream_started_at
    seen: list[float | None] = []

    def spy():
        value = real()
        seen.append(value)
        return value

    monkeypatch.setattr(agents_module, "_stream_started_at", spy)
    return seen


class SegmentList(list):
    """CallState.pending_segments that says when the segments a test waits for have arrived."""

    def __init__(self, expected: int = 1) -> None:
        super().__init__()
        self.expected = expected
        self.arrived = asyncio.Event()

    def append(self, item) -> None:
        super().append(item)
        if len(self) >= self.expected:
            self.arrived.set()


async def hear(script: Script, words: int = 1) -> SimpleNamespace:
    """Run one fake caller through a real session; return what the mixin made and the truth.

    `error_s` and the rest describe the first word; `errors_s` has the error of each of `words`.
    """
    state = CallState(room_name="room", caller_identity="caller", channel=Channel.BROWSER)
    state.pending_segments = segments = SegmentList(words)
    fake_stt = ScriptedSTT(script)
    audio_may_start = asyncio.Event()
    session = AgentSession(stt=fake_stt, userdata=state)
    session.input.audio = PacedAudio(audio_may_start, frames=200, idle_s=script.idle_s)
    try:
        await session.start(LateStreamAgent(STREAM_DELAY_S, audio_may_start))
        await asyncio.wait_for(segments.arrived.wait(), timeout=20)
        segment, stream = segments[0], fake_stt.streams[0]
    finally:
        await session.aclose()
    reported = stream.reported
    return SimpleNamespace(
        stream=stream,
        reported=reported,
        t0=segment.t0,
        # How far from the truth each rule puts the word's epoch time, in seconds.
        error_s=segment.t0 + segment.words[0][1] - reported.truth_s,
        uncorrected_error_s=segment.t0 + reported.start_s - reported.truth_s,
        errors_s=[
            seg.t0 + seg.words[0][1] - rep.truth_s
            for seg, rep in zip(segments, stream.reports, strict=True)
        ],
    )


def switches(caplog) -> list[logging.LogRecord]:
    """The records in which the mixin says a stream has switched to the offset of a final."""
    return [
        r
        for r in caplog.records
        if r.name == "callback_agent.agents" and "after its transcript arrived" in r.getMessage()
    ]


async def test_a_word_lands_on_true_time_with_the_real_default_node(read_offsets, caplog):
    heard = await hear(Script(word_at=0.2))
    # The framework really did offset this stream, so there is something to take out ...
    assert heard.stream.offset_at_first_frame >= STREAM_DELAY_S - 0.05
    assert heard.uncorrected_error_s > 0.25  # (t0 + the reported time would be this late)
    # ... the mixin read exactly that offset off the real stream at the first frame ...
    assert read_offsets == [pytest.approx(heard.stream.offset_at_first_frame, abs=0.005)]
    assert not [r for r in caplog.records if r.name == "callback_agent.agents"]  # no fallback
    # ... and the word lands where it was spoken.
    assert abs(heard.error_s) < 0.15


async def test_words_after_a_retry_that_came_before_any_word_still_land_on_true_time(
    read_offsets, caplog
):
    # The first connection drops before anything was heard. The framework adds that connection's
    # lifetime (and the 0.1 s before the retry) to the offset. Latching the offset off the first
    # word would take that growth out as well, and put every later word that early.
    heard = await hear(Script(word_at=0.1, speak_on=2, drop_after_s=0.5))
    assert heard.stream.attempts == 2
    assert heard.reported.offset_s - heard.stream.offset_at_first_frame >= 0.5  # it did grow
    # What is left is the wait before the retry: the framework counts it as time that passed.
    assert -0.15 < heard.error_s < 0.35
    assert heard.uncorrected_error_s > heard.error_s + 0.25  # the offset was still taken out
    assert not switches(caplog)  # the growth was audio: the first-frame offset stays


@pytest.mark.parametrize("fails_after_s", [0.3, 1.0])
async def test_words_after_a_connect_failure_land_on_true_time(fails_after_s, read_offsets, caplog):
    # The first connection fails while connecting, so it took no frame: the frames wait, and the
    # next connection starts at the first one. The base class still adds that attempt's lifetime and
    # the 0.1 s wait to the offset, and none of it is audio. Taking out only the first-frame offset
    # would put every word that late: 0.4 s and 1.1 s here. A word cannot end after its transcript
    # arrived, so the mixin sees that and follows the offset the transcript carries.
    caplog.set_level(logging.WARNING, logger="callback_agent.agents")
    heard = await hear(Script(word_at=0.2, speak_on=2, connect_fail_after_s=fails_after_s))
    assert heard.stream.attempts == 2
    growth = heard.reported.offset_s - read_offsets[0]
    assert growth >= fails_after_s + 0.1 - 0.02  # the offset grew, though no frame went with it
    # The growth, 0.4 s and 1.1 s, is more than the final's 0.25 s latency plus the 50 ms of slack,
    # so the check sees it and the error is only what the first frame arrived late by: a few
    # milliseconds, and 150 ms leaves room for a loaded machine. Without the switch the error is
    # the growth itself. Growth that hid inside latency plus slack (0.3 s here) would not be seen,
    # and would cost no more than that.
    assert abs(heard.error_s) < 0.15
    assert len(switches(caplog)) == 1


# --- a stream that waits for its first frame ---------------------------------------------------
#
# At the start of a call the session makes the STT stream inside start(), and the caller's audio is
# forwarded only after start() has returned, so the stream waits for its first frame. The base class
# dates every attempt from its own start, so after a retry that wait is in the growth of the offset,
# though no audio and no Deepgram clock ran during it. These tests give the stream a wait of 0.3 s.

IDLE_S = 0.3


def assert_waited(heard, read_started_at) -> float:
    """The stream waited, and the mixin measured the wait that really happened. Returns that wait.

    The wait is not exactly IDLE_S: a sleep runs at least that long and, on a loaded machine, up to
    0.1 s more (0.403 s was seen at load average 9). So the mixin's reading is compared with what
    the harness saw for itself, from the making of the stream to its first frame, and the
    expectations that depend on the size of the wait are computed from it.
    """
    stream = heard.stream
    assert stream.first_frame_after_s >= IDLE_S - 0.01  # a sleep is never short
    # RecognizeStream.start_time is the moment the attempt began. The framework seeds it when the
    # stream is made and again as the retry loop starts, a tick later; 0.15 s is far below the 0.3 s
    # by which the session's own start (or any other clock) would differ.
    assert read_started_at == [pytest.approx(stream.created_at, abs=0.15)]
    # The wait runs up to the first sample of audio, a frame before the frame arrived.
    waited = stream.first_frame_after_s - FRAME_S
    assert heard.t0 - read_started_at[0] == pytest.approx(waited, abs=0.15)
    return waited


async def test_a_stream_that_waited_for_its_first_frame_and_never_reconnected_is_unchanged(
    read_offsets, read_started_at, caplog
):
    heard = await hear(Script(word_at=0.2, idle_s=IDLE_S))
    assert_waited(heard, read_started_at)
    # No retry, so nothing was added to the offset and nothing is taken out of it for the wait.
    assert heard.reported.offset_s == pytest.approx(read_offsets[0], abs=0.005)
    assert not [r for r in caplog.records if r.name == "callback_agent.agents"]
    assert abs(heard.error_s) < 0.15


@pytest.mark.parametrize("drops_after_s", [0.5, 2.0])
async def test_a_drop_after_audio_is_not_taken_for_dead_time_by_a_stream_that_waited(
    drops_after_s, read_offsets, read_started_at, caplog
):
    # The first connection carries audio and drops. The growth holds the 0.3 s wait, the audio, and
    # the 0.1 s before the retry. Counting the wait as audio puts every word 0.4 s late, which is
    # more than the final's 0.25 s latency plus the 50 ms of slack, so the check used to fire and
    # follow the live offset: that took the first connection's audio out as well, and put the words
    # early by all of it (-0.5 s here, -2.0 s for the longer one, and without bound in a long call).
    caplog.set_level(logging.WARNING, logger="callback_agent.agents")
    script = Script(word_at=0.1, speak_on=2, drop_after_s=drops_after_s, idle_s=IDLE_S)
    heard = await hear(script)
    assert_waited(heard, read_started_at)
    assert heard.stream.attempts == 2
    growth = heard.reported.offset_s - read_offsets[0]
    assert growth >= IDLE_S + drops_after_s + 0.1 - 0.05  # the wait is in the growth
    assert not switches(caplog)
    # With the wait left out, what is left is the 0.1 s the framework waits before the retry, which
    # the audio does not take, plus the delivery jitter of the frame the connection dropped on. The
    # error was 0.101 to 0.106 s in six runs at load average 6 to 9, and 0.159 s once at load 11:
    # 59 ms of jitter. The window leaves 90 ms above that spike and 50 ms below the quiet value.
    # The false switch it guards against costs 0.5 s or more, and it is asserted on its own.
    assert -0.05 < heard.error_s < 0.25


@pytest.mark.parametrize("fails_after_s", [0.3, 1.0])
async def test_a_connect_failure_in_a_stream_that_waited_is_still_corrected(
    fails_after_s, read_offsets, read_started_at, caplog
):
    # The failed attempt takes no frame, so all of its lifetime is dead time. Of that, 0.3 s is the
    # wait for the first frame, which is left out of the growth like any other; what remains for the
    # check is the failure beyond the wait plus the 0.1 s before the retry.
    caplog.set_level(logging.WARNING, logger="callback_agent.agents")
    heard = await hear(
        Script(word_at=0.2, speak_on=2, connect_fail_after_s=fails_after_s, idle_s=IDLE_S)
    )
    waited = assert_waited(heard, read_started_at)
    assert heard.stream.attempts == 2
    remains = max(0.0, fails_after_s + 0.1 - waited)  # from the wait that really happened
    if remains > 0.25 + 0.05 + 0.1:
        # About 0.8 s: more than the latency plus slack with 100 ms to spare, so it is seen, and
        # the stream goes over to the offset the transcript carries.
        assert len(switches(caplog)) == 1
        assert abs(heard.error_s) < 0.15
    else:
        # About 0.1 s, or none if the wait outlasted the failure: inside latency plus slack, so it
        # is not seen, and it costs no more than it is, plus the 60 ms of jitter seen under load.
        assert not switches(caplog)
        assert remains - 0.05 < heard.error_s < remains + 0.15


# --- what follows a switch ---------------------------------------------------------------------


@pytest.mark.parametrize("idle_s", [0.0, IDLE_S])
async def test_a_drop_that_carried_audio_after_a_connect_failure_is_late_by_the_retry_wait_only(
    idle_s, read_offsets, read_started_at, caplog
):
    # The first connection fails while connecting, 1.0 s in, and takes no frame. The second catches
    # up on the frames that waited, hears the first word, and drops after carrying on. The first
    # word ends after its transcript arrived with the first-frame offset, so the mixin places that
    # connection by the offset on its transcripts. The third starts where the second stopped,
    # which is further in than the second's lifetime says: the frames that waited are audio by
    # then. Taking the growth out again would put its word early by all of it (-1.5 s here, and
    # still -1.0 s, the failed connect, if the stream were only re-based on the first word's
    # offset). Judged again against the first-frame offset, the word is late by the 0.1 s the
    # framework waits before a retry, which is what a drop that carries audio costs.
    caplog.set_level(logging.WARNING, logger="callback_agent.agents")
    script = Script(
        word_at=0.2,
        speak_on=2,
        connect_fail_after_s=1.0,
        second_word_at=0.1,
        first_drops_after_s=1.5,
        idle_s=idle_s,
    )
    heard = await hear(script, words=2)
    if idle_s:
        assert_waited(heard, read_started_at)
    assert heard.stream.attempts == 3
    first, second = heard.stream.reports
    assert first.offset_s - read_offsets[0] >= 1.0 + 0.1 - 0.02  # the failed connect, all dead
    assert second.offset_s - first.offset_s >= 0.4  # the second connection ran: it carried audio
    assert len(switches(caplog)) == 1  # the first word's, and not the second's
    first_error, second_error = heard.errors_s
    # The first word is exact up to what the first frame arrived late by; 150 ms leaves room for a
    # loaded machine, as in the connect failure test above.
    assert abs(first_error) < 0.15
    # The second is late by the 0.1 s of the retry wait plus the delivery jitter of the frame the
    # second connection dropped on: 0.101 to 0.103 s in the runs on a quiet machine, and the
    # window of the drop test above (59 ms of jitter seen at load average 11) is used here.
    assert -0.05 < second_error < 0.25


async def test_the_frameworks_own_log_of_a_transcript_prints_no_transcript_once_the_filter_is_in():
    # At DEBUG the framework logs every final transcript, with the text in an lk.pii.user_transcript
    # extra, and LiveKit's formatters print every extra. This is its real record, not a made-up one.
    out, root = io.StringIO(), logging.getLogger()
    handler = logging.StreamHandler(out)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    framework = logging.getLogger("livekit.agents")
    level = framework.level
    framework.setLevel(logging.DEBUG)
    try:
        install_pii_filter()
        await hear(Script(word_at=0.2))
    finally:
        framework.setLevel(level)
        root.removeHandler(handler)
        for owner in [root, framework, *root.handlers]:
            for f in [f for f in owner.filters if isinstance(f, PiiExtraFilter)]:
                owner.removeFilter(f)
    printed = out.getvalue()
    assert "received user transcript" in printed  # the framework did log the transcript ...
    assert "lk.pii" not in printed  # ... and the text did not get out with it
