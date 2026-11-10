"""The caller's speech, cut into segments for the detector to hear.

The detection service measures voices on its /events interface, one `speech_segment` at a time, and
until now the agent sent it text and timings only. This module turns the frames the agent already
sees on their way to the STT (TimedSttMixin taps them) into segments: one per stretch of the
caller speaking, bounded by the session's own speech events, capped at MAX_SEGMENT_S of audio, in
the detector's wire shape (a WAV container, pcm_s16le, mono, 16 kHz, inline as a data: URI).

Everything here is pure and runs on the event loop. SegmentCapture is the one stateful piece: it
keeps a short pre-roll of recent frames (a speech start is reported a few hundred ms after the
fact), the open segment, and nothing else. A finished segment is handed to the sink, which is the
session wiring's emit, and from there to the forwarder; nothing here posts, logs or writes.

Times are Unix epoch ms on the frame clock: the mixin's anchor (the first frame of an stt_node
call) plus the audio heard since, which is the clock the turns' word times are on, so a segment
lines up with its turn. Nothing here holds text, and a segment prints as its leg and its times.
"""

import base64
import io
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from livekit import rtc

MAX_SEGMENT_S = 20.0  # a longer stretch of speech is split into consecutive segments
MIN_SEGMENT_S = 0.5  # shorter is dropped: the detector marks under 0.25 s unusable anyway
PREROLL_S = 1.0  # how far back a late speech start can reach for frames already heard
WIRE_SAMPLE_RATE_HZ = 16000  # what the detector's analysis runs at
TRACK_CALLER = "caller"  # the inbound leg
TRACK_CALLBACK = "callback"  # the leg dialed to the number on file
TIMING_HISTORY = 32  # how many reply latencies and durations a segment's metadata carries


def _epoch_ms(epoch_s: float) -> int:
    return round(epoch_s * 1000)  # the one rule turns.py uses for every time


def frame_to_mono(frame: rtc.AudioFrame) -> np.ndarray:
    """The frame's samples as mono int16: channels are averaged."""
    samples = np.frombuffer(frame.data, dtype=np.int16)
    channels = frame.num_channels
    if channels <= 1:
        return samples.copy()  # the frame's buffer is reused: keep our own
    return np.rint(samples.reshape(-1, channels).mean(axis=1)).astype(np.int16)


def resample(samples: np.ndarray, from_hz: int, to_hz: int) -> np.ndarray:
    """`samples` at `to_hz`, by linear interpolation. Dependency-free; good enough for voice."""
    if from_hz == to_hz or samples.size == 0:
        return samples
    count = max(1, round(samples.size * to_hz / from_hz))
    positions = np.arange(count) * (samples.size / count)
    out = np.interp(positions, np.arange(samples.size), samples.astype(np.float64))
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def wav_bytes(samples: np.ndarray, sample_rate_hz: int) -> bytes:
    """A WAV container (RIFF/WAVE, pcm_s16le, mono) over `samples`."""
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate_hz)
        w.writeframes(np.ascontiguousarray(samples, dtype=np.int16).tobytes())
    return out.getvalue()


@dataclass
class CallTiming:
    """The conversation's timing so far, for the detector's agent_behaviour signal. Numbers only.

    `turn_latencies_ms` is the gap from the end of the agent's line to the start of the caller's
    reply, `response_durations_ms` how long each reply ran, `barge_ins` how many of the caller's
    turns began over the agent. The session wiring keeps one per call and a segment carries a
    snapshot when it is cut, so the detector reads what the agent knew by then. The detector
    needs three latencies before the signal runs, and the last TIMING_HISTORY are enough.
    """

    turn_latencies_ms: list[int] = field(default_factory=list)
    response_durations_ms: list[int] = field(default_factory=list)
    barge_ins: int = 0

    def reply(self, *, latency_ms: int | None, duration_ms: int) -> None:
        """The caller replied: `latency_ms` after the agent's last line, if there was one."""
        if latency_ms is not None:
            self.turn_latencies_ms.append(int(latency_ms))
            del self.turn_latencies_ms[:-TIMING_HISTORY]
        self.response_durations_ms.append(int(duration_ms))
        del self.response_durations_ms[:-TIMING_HISTORY]

    def barge_in(self) -> None:
        self.barge_ins += 1

    def metadata(self) -> dict[str, Any]:
        """The `metadata` of a speech_segment event: the detector's CallMetadata keys."""
        return {
            "turn_latencies_ms": list(self.turn_latencies_ms),
            "response_durations_ms": list(self.response_durations_ms),
            "barge_ins": self.barge_ins,
        }


@dataclass(eq=False)
class AudioSegment:
    """One stretch of the caller speaking, as heard: mono int16 at the frames' own rate.

    `track_id` is the leg it was heard on, stamped when the segment is cut, and `metadata` the
    call's timing by then (CallTiming.metadata()), None until the wiring stamps it.
    """

    track_id: str
    start_ms: int
    end_ms: int
    samples: np.ndarray = field(repr=False)
    sample_rate_hz: int
    metadata: dict[str, Any] | None = None

    @property
    def duration_ms(self) -> int:
        return round(self.samples.size * 1000 / self.sample_rate_hz)

    def wav_bytes(self, sample_rate_hz: int = WIRE_SAMPLE_RATE_HZ) -> bytes:
        """The segment as a WAV at `sample_rate_hz`."""
        samples = resample(self.samples, self.sample_rate_hz, sample_rate_hz)
        return wav_bytes(samples, sample_rate_hz)

    def __str__(self) -> str:
        return f"{self.track_id} audio segment at {self.start_ms} ms ({self.duration_ms} ms)"


def speech_segment_body(
    session_id: str, segment: AudioSegment, *, sequence: int, claimed_user_id: str | None
) -> dict[str, Any]:
    """The body of a `speech_segment` event on the detector's /events interface.

    `audio` goes last: the detector's journal reads only the first 16 KiB of a body, and the
    other fields have to be in it. `occurred_at` is the segment's end. `metadata` is the call's
    timing (CallTiming), when the segment carries one: the detector's agent_behaviour signal
    reads it, and nothing in it is text.

    `start_ms` and `end_ms` are Unix epoch ms on the turns' clock, so the segment lines up with
    the turn it belongs to, and `track_id` names the leg ('caller' or 'callback'). The detector's
    schema words them as offsets and a LiveKit track id; nothing there reads them yet, and this is
    the reading its author is asked to adopt (noted on AudioRef in shared/contract.py).
    """
    wav = segment.wav_bytes()
    body: dict[str, Any] = {
        "session_id": session_id,
        "event": "speech_segment",
        "sequence": sequence,
        "occurred_at": datetime.fromtimestamp(segment.end_ms / 1000, tz=UTC).isoformat(),
        "claimed_user_id": claimed_user_id,
    }
    if segment.metadata is not None:
        body["metadata"] = segment.metadata
    body["audio"] = {
        "track_id": segment.track_id,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "sample_rate_hz": WIRE_SAMPLE_RATE_HZ,
        "encoding": "pcm_s16le",  # a WAV is detected by its magic bytes, under this name
        "uri": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"),
    }
    return body


@dataclass
class _Chunk:
    start_s: float
    samples: np.ndarray
    rate: int

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.rate

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


class SegmentCapture:
    """Cuts the frames it is fed into AudioSegments, at the speech events it is told about.

    add_frame runs in the STT path, once per frame, and does nothing but keep the frame (as mono
    int16) in the pre-roll or in the open segment. begin() opens a segment at a time, reaching
    back into the pre-roll for the frames since then; end() closes it at a time, hands back what
    came after, and gives the segment to `sink`. A segment that reaches `max_segment_s` goes to
    the sink at once and the next one opens where it ended. Nothing is kept for long: the
    pre-roll is `preroll_s` of audio, and the open segment at most `max_segment_s`.
    """

    def __init__(
        self,
        *,
        sink: Callable[[AudioSegment], None] | None = None,
        max_segment_s: float = MAX_SEGMENT_S,
        min_segment_s: float = MIN_SEGMENT_S,
        preroll_s: float = PREROLL_S,
    ) -> None:
        self.sink = sink
        self._max_s = max_segment_s
        self._min_s = min_segment_s
        self._preroll_s = preroll_s
        self._recent: deque[_Chunk] = deque()
        self._recent_s = 0.0
        self._open: list[_Chunk] | None = None
        self._open_samples = 0  # at the rate of the open segment's first frame
        self._track = TRACK_CALLER
        self.dropped_short = 0  # segments under min_segment_s, not sent
        self.emitted = 0

    @property
    def open(self) -> bool:
        return self._open is not None

    @property
    def buffered_s(self) -> float:
        """Seconds of audio held: the pre-roll and the open segment."""
        open_s = 0.0 if not self._open else self._open_samples / self._open[0].rate
        return self._recent_s + open_s

    def add_frame(self, frame: rtc.AudioFrame, start_s: float) -> None:
        """Keep one frame, which began at `start_s` (epoch seconds on the frame clock)."""
        chunk = _Chunk(start_s, frame_to_mono(frame), frame.sample_rate)
        if self._open is None:
            self._recent.append(chunk)
            self._recent_s += chunk.duration_s
            while self._recent_s > self._preroll_s and len(self._recent) > 1:
                self._recent_s -= self._recent.popleft().duration_s
            return
        self._append_open(chunk)

    def _append_open(self, chunk: _Chunk) -> None:
        """Add a chunk to the open segment, cutting sample-exactly where it reaches the cap."""
        assert self._open is not None
        if self._open and chunk.rate != self._open[0].rate:
            rate = self._open[0].rate
            chunk = _Chunk(chunk.start_s, resample(chunk.samples, chunk.rate, rate), rate)
        room = round(self._max_s * chunk.rate) - self._open_samples
        if chunk.samples.size < room:
            self._open.append(chunk)
            self._open_samples += chunk.samples.size
            return
        head = _Chunk(chunk.start_s, chunk.samples[:room], chunk.rate)
        self._open.append(head)
        track = self._track
        self._emit(self._open)
        self._open, self._open_samples = [], 0  # the speech goes on: the next piece starts here
        self._track = track
        if chunk.samples.size > room:
            self._append_open(_Chunk(head.end_s, chunk.samples[room:], chunk.rate))

    def begin(self, at_s: float, *, track_id: str) -> None:
        """Speech began at `at_s`. The frames heard since then come out of the pre-roll."""
        if self._open is not None:
            return
        self._track = track_id
        self._open, self._open_samples = [], 0
        for chunk in self._recent:
            if chunk.end_s > at_s:
                self._append_open(chunk)
        self._recent.clear()
        self._recent_s = 0.0

    def discard(self) -> None:
        """Drop the open segment and the pre-roll, unheard: the call changed hands (a relink)."""
        self._open, self._open_samples = None, 0
        self._recent.clear()
        self._recent_s = 0.0

    def end(self, at_s: float) -> None:
        """Speech ended at `at_s`. What was heard after goes back to the pre-roll."""
        if self._open is None:
            return
        kept = [c for c in self._open if c.start_s < at_s]
        after = [c for c in self._open if c.start_s >= at_s]
        self._open, self._open_samples = None, 0
        for chunk in after:
            self._recent.append(chunk)
            self._recent_s += chunk.duration_s
        self._emit(kept)

    def _emit(self, chunks: list[_Chunk]) -> None:
        if not chunks:
            return
        rate = chunks[0].rate
        samples = np.concatenate([c.samples for c in chunks])  # one rate: _append_open saw to it
        duration_s = samples.size / rate
        if duration_s < self._min_s:
            self.dropped_short += 1
            return
        start_s = chunks[0].start_s
        segment = AudioSegment(
            track_id=self._track,
            start_ms=_epoch_ms(start_s),
            end_ms=_epoch_ms(start_s + duration_s),
            samples=samples,
            sample_rate_hz=rate,
        )
        self.emitted += 1
        if self.sink is not None:
            self.sink(segment)
