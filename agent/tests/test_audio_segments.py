"""audio_segments turns the caller's frames into speech segments: 16 kHz mono WAV, cut at the
speech events, capped at 20 s, with epoch ms from the frame clock, and the /events body for one."""

import base64
import io
import json
import wave

import numpy as np
import pytest
from livekit import rtc

from callback_agent.audio_segments import (
    MAX_SEGMENT_S,
    MIN_SEGMENT_S,
    TIMING_HISTORY,
    AudioSegment,
    CallTiming,
    SegmentCapture,
    frame_to_mono,
    resample,
    speech_segment_body,
    wav_bytes,
)

T0 = 1_700_000_000.0  # the epoch second the frame clock starts at


def frame(sample_rate: int, channels: int, ms: int, fill: int = 1000) -> rtc.AudioFrame:
    """One frame of `ms` ms, every sample `fill`, so that a resample keeps a known level."""
    n = sample_rate * ms // 1000
    pcm = np.full(n * channels, fill, dtype=np.int16)
    return rtc.AudioFrame(pcm.tobytes(), sample_rate, channels, n)


def frames(seconds: float, sample_rate: int = 24000, channels: int = 1, ms: int = 20):
    """`seconds` of frames, each tagged with its start on the frame clock (epoch seconds)."""
    out = []
    count = round(seconds * 1000 / ms)
    for k in range(count):
        out.append((frame(sample_rate, channels, ms), T0 + k * ms / 1000))
    return out


def read_wav(data: bytes) -> tuple[int, int, int, np.ndarray]:
    with wave.open(io.BytesIO(data)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), samples


def capture(**kwargs) -> tuple[SegmentCapture, list[AudioSegment]]:
    out: list[AudioSegment] = []
    return SegmentCapture(sink=out.append, **kwargs), out


def speak(cap: SegmentCapture, seconds: float, *, track="caller", begin_at=None, end_at=None):
    """Caller speaks for `seconds` from T0: begin, every frame, end."""
    cap.begin(T0 if begin_at is None else begin_at, track_id=track)
    for f, at in frames(seconds):
        cap.add_frame(f, at)
    cap.end(T0 + seconds if end_at is None else end_at)


# --- frames to WAV -------------------------------------------------------------------------------


def test_a_stereo_48k_frame_becomes_mono_at_its_own_rate():
    pcm = np.array([100, 300, -200, 200], dtype=np.int16)  # two stereo samples
    f = rtc.AudioFrame(pcm.tobytes(), 48000, 2, 2)
    mono = frame_to_mono(f)
    assert mono.dtype == np.int16 and mono.tolist() == [200, 0]


def test_resample_keeps_the_duration_and_the_level():
    out = resample(np.full(48000, 1000, dtype=np.int16), 48000, 16000)
    assert out.dtype == np.int16 and out.size == 16000
    assert out.min() == 1000 and out.max() == 1000


def test_resample_at_the_same_rate_is_the_input():
    x = np.arange(100, dtype=np.int16)
    assert resample(x, 16000, 16000) is x


def test_48k_stereo_frames_become_a_16k_mono_wav_of_the_right_length():
    cap, out = capture()
    cap.begin(T0, track_id="caller")
    for f, at in frames(3.0, sample_rate=48000, channels=2):
        cap.add_frame(f, at)
    cap.end(T0 + 3.0)
    [segment] = out
    data = segment.wav_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    channels, width, rate, samples = read_wav(data)
    assert (channels, width, rate) == (1, 2, 16000)
    assert samples.size == 3 * 16000
    assert samples.min() == 1000 and samples.max() == 1000
    assert segment.sample_rate_hz == 48000 and segment.duration_ms == 3000


def test_wav_bytes_is_a_valid_riff_header_over_the_samples():
    data = wav_bytes(np.array([1, -1, 2], dtype=np.int16), 16000)
    assert len(data) == 44 + 6
    assert read_wav(data)[3].tolist() == [1, -1, 2]


# --- segment boundaries --------------------------------------------------------------------------


def test_a_segment_runs_from_the_speech_start_to_the_speech_end_in_epoch_ms():
    cap, out = capture()
    speak(cap, 2.0)
    [segment] = out
    assert segment.track_id == "caller"
    assert (segment.start_ms, segment.end_ms) == (1_700_000_000_000, 1_700_000_002_000)
    assert segment.duration_ms == 2000


def test_the_frames_before_the_start_event_arrived_are_kept_as_a_preroll():
    # VAD and Flux report a speech start a few hundred ms after the fact, and by then those frames
    # have passed the tap. They are kept for a while, so a late begin() still gets them.
    cap, out = capture()
    all_frames = frames(2.0)
    for f, at in all_frames[:15]:  # 300 ms heard before anyone said the caller was speaking
        cap.add_frame(f, at)
    cap.begin(T0 + 0.1, track_id="caller")  # the event says speech began at 100 ms
    for f, at in all_frames[15:]:
        cap.add_frame(f, at)
    cap.end(T0 + 2.0)
    [segment] = out
    assert segment.start_ms == 1_700_000_000_100  # from the first frame that overlaps the start
    assert segment.end_ms == 1_700_000_002_000
    assert segment.duration_ms == 1900


def test_frames_after_the_end_event_time_are_cut_off_the_segment():
    cap, out = capture()
    speak(cap, 2.0, end_at=T0 + 1.5)  # the end event is dated 500 ms before the last frame
    [segment] = out
    assert segment.end_ms == 1_700_000_001_500 and segment.duration_ms == 1500


def test_frames_outside_speech_make_no_segment():
    cap, out = capture()
    for f, at in frames(5.0):
        cap.add_frame(f, at)
    cap.end(T0 + 5.0)  # no begin
    assert out == [] and not cap.open


def test_a_segment_shorter_than_the_minimum_is_dropped_and_counted():
    cap, out = capture()
    speak(cap, MIN_SEGMENT_S / 2)
    assert out == [] and cap.dropped_short == 1


def test_a_second_begin_while_open_changes_nothing():
    cap, out = capture()
    cap.begin(T0, track_id="caller")
    for f, at in frames(1.0):
        cap.add_frame(f, at)
    cap.begin(T0 + 0.5, track_id="callback")
    for f, at in frames(1.0):
        cap.add_frame(f, at + 1.0)
    cap.end(T0 + 2.0)
    [segment] = out
    assert segment.track_id == "caller" and segment.duration_ms == 2000


def test_a_long_stretch_is_split_into_consecutive_segments_of_at_most_20_s():
    assert MAX_SEGMENT_S == 20.0
    cap, out = capture()
    speak(cap, 45.0)
    assert [s.duration_ms for s in out] == [20000, 20000, 5000]
    assert [s.start_ms for s in out] == [
        1_700_000_000_000,
        1_700_000_020_000,
        1_700_000_040_000,
    ]
    assert [s.end_ms for s in out] == [1_700_000_020_000, 1_700_000_040_000, 1_700_000_045_000]
    assert all(s.wav_bytes()[:4] == b"RIFF" for s in out)


def test_the_full_pieces_go_out_while_the_speech_goes_on():
    cap, out = capture()
    cap.begin(T0, track_id="caller")
    for f, at in frames(21.0):
        cap.add_frame(f, at)
    assert len(out) == 1 and out[0].duration_ms == 20000  # before end()
    assert cap.open


def test_frames_at_another_rate_inside_a_segment_are_resampled_into_it():
    cap, out = capture()
    cap.begin(T0, track_id="caller")
    for f, at in frames(1.0, sample_rate=24000):
        cap.add_frame(f, at)
    for f, at in frames(1.0, sample_rate=48000):
        cap.add_frame(f, at + 1.0)
    cap.end(T0 + 2.0)
    [segment] = out
    assert segment.sample_rate_hz == 24000 and segment.duration_ms == 2000


def test_the_preroll_is_bounded():
    cap, _ = capture(preroll_s=1.0)
    for f, at in frames(30.0):
        cap.add_frame(f, at)
    assert cap.buffered_s == pytest.approx(1.0, abs=0.05)


# --- the /events body ----------------------------------------------------------------------------


def test_the_body_is_the_detectors_speech_segment_event_with_the_audio_inline_and_last():
    cap, out = capture()
    speak(cap, 1.0, track="callback")
    [segment] = out
    body = speech_segment_body(
        "ses_0123456789abcdef", segment, sequence=3, claimed_user_id="u_eric"
    )
    assert list(body) == [
        "session_id",
        "event",
        "sequence",
        "occurred_at",
        "claimed_user_id",
        "audio",
    ]
    assert body["session_id"] == "ses_0123456789abcdef"
    assert body["event"] == "speech_segment"
    assert body["sequence"] == 3
    assert body["occurred_at"] == "2023-11-14T22:13:21+00:00"  # the segment's end, ISO 8601 UTC
    assert body["claimed_user_id"] == "u_eric"
    audio = body["audio"]
    assert list(audio) == ["track_id", "start_ms", "end_ms", "sample_rate_hz", "encoding", "uri"]
    assert audio["track_id"] == "callback"
    assert (audio["start_ms"], audio["end_ms"]) == (segment.start_ms, segment.end_ms)
    assert audio["sample_rate_hz"] == 16000 and audio["encoding"] == "pcm_s16le"
    prefix = "data:audio/wav;base64,"
    assert audio["uri"].startswith(prefix)
    assert base64.b64decode(audio["uri"][len(prefix) :], validate=True) == segment.wav_bytes()
    json.dumps(body)  # it is plain JSON


def test_the_body_carries_the_segments_timing_as_metadata_before_the_audio():
    cap, out = capture()
    speak(cap, 1.0)
    [segment] = out
    segment.metadata = {
        "turn_latencies_ms": [400, 650],
        "response_durations_ms": [1000],
        "barge_ins": 1,
    }
    body = speech_segment_body("ses_0123456789abcdef", segment, sequence=0, claimed_user_id=None)
    assert list(body)[-2:] == ["metadata", "audio"]
    assert body["metadata"] == {
        "turn_latencies_ms": [400, 650],
        "response_durations_ms": [1000],
        "barge_ins": 1,
    }


def test_call_timing_keeps_numbers_only_and_the_latest_few():
    timing = CallTiming()
    for k in range(40):
        timing.reply(latency_ms=100 + k, duration_ms=1000)
    timing.barge_in()
    data = timing.metadata()
    assert list(data) == ["turn_latencies_ms", "response_durations_ms", "barge_ins"]
    assert len(data["turn_latencies_ms"]) == TIMING_HISTORY
    assert data["turn_latencies_ms"][-1] == 139
    assert data["barge_ins"] == 1
    numbers = data["turn_latencies_ms"] + data["response_durations_ms"]
    assert all(isinstance(v, int) for v in numbers)
    timing.reply(latency_ms=None, duration_ms=500)  # a reply with nothing to measure it from
    assert len(timing.metadata()["turn_latencies_ms"]) == TIMING_HISTORY
    assert timing.metadata()["response_durations_ms"][-1] == 500


def test_discard_drops_the_open_segment_and_what_came_before_it():
    cap, out = capture()
    cap.begin(T0, track_id="caller")
    for f, at in frames(1.0):
        cap.add_frame(f, at)
    cap.discard()
    assert not cap.open and cap.buffered_s == 0.0
    cap.end(T0 + 1.0)
    assert out == [] and cap.emitted == 0


def test_the_body_takes_no_claimed_user():
    cap, out = capture()
    speak(cap, 1.0)
    body = speech_segment_body("ses_0123456789abcdef", out[0], sequence=0, claimed_user_id=None)
    assert body["claimed_user_id"] is None


def test_a_segment_describes_itself_without_its_audio():
    cap, out = capture()
    speak(cap, 1.0)
    text = str(out[0])
    assert "caller" in text and "1000" in text and len(text) < 120
