import numpy as np
import pytest
from livekit import rtc

from callback_agent.audio_check import BandwidthProbe, high_band_ratio

SR = 24000  # the rate RoomIO hands the STT node by default


def tone(hz: float, seconds: float, amplitude: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * hz * t)


def frames(signal: np.ndarray, sr: int = SR, ms: int = 50) -> list[rtc.AudioFrame]:
    """Cut a float signal (full scale is 1.0) into mono 16-bit frames, like the room input does."""
    pcm = np.round(signal * 32767).astype(np.int16)
    n = sr * ms // 1000
    return [rtc.AudioFrame(pcm[i : i + n].tobytes(), sr, 1, n) for i in range(0, len(pcm) - n + 1, n)]


def feed(probe: BandwidthProbe, signal: np.ndarray) -> None:
    for frame in frames(signal):
        probe.add(frame)


@pytest.mark.parametrize("sr", [16000, 24000, 48000])
def test_high_band_ratio(sr):
    t = np.arange(sr) / sr
    low, high = np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 6000 * t)
    assert high_band_ratio(low, sr) < 0.01
    assert high_band_ratio(high, sr) > 0.9
    assert high_band_ratio(low + high, sr) == pytest.approx(0.5, abs=0.05)


def test_high_band_ratio_of_silence_is_zero_not_nan():
    assert high_band_ratio(np.zeros(16000), 16000) == 0.0


def test_probe_ignores_silence():
    probe = BandwidthProbe(SR)
    feed(probe, np.zeros(10 * SR))
    assert probe.verdict() is None


def test_probe_waits_for_enough_speech():
    probe = BandwidthProbe(SR)
    feed(probe, tone(1000, 9.0))
    assert probe.verdict() is None
    feed(probe, tone(1000, 1.5))
    assert probe.verdict() is not None


def test_probe_does_not_count_quiet_frames():
    probe = BandwidthProbe(SR)
    # RMS of a sine is amplitude / sqrt(2). Amplitude 0.01 is 0.7% of full scale, below the 1% bar.
    feed(probe, tone(6000, 20.0, amplitude=0.01))
    assert probe.verdict() is None
    # Quiet and loud frames interleaved: only the loud 1 kHz frames count, and only they are measured.
    for _ in range(25):
        feed(probe, tone(6000, 0.5, amplitude=0.01))
        feed(probe, tone(1000, 0.5))
    result = probe.verdict()
    assert result is not None and result["verdict"] == "narrowband"


@pytest.mark.parametrize(
    ("high_amplitude", "expected"),
    [
        (0.0, "narrowband"),
        (0.0213, "narrowband"),  # 6 kHz carries 0.5% of the energy next to a 0.3 tone at 1 kHz
        (0.0429, "wideband"),  # 2%
        (0.1, "wideband"),
    ],
)
def test_probe_threshold(high_amplitude, expected):
    probe = BandwidthProbe(SR)
    feed(probe, tone(1000, 10.5) + tone(6000, 10.5, amplitude=high_amplitude))
    result = probe.verdict()
    assert result is not None
    assert result["verdict"] == expected
    assert (result["high_band_ratio"] > 0.01) == (expected == "wideband")


def test_narrowband_speech_with_pauses_is_not_mistaken_for_wideband():
    # Noise limited to 300-3400 Hz in half-second bursts with silence between: the joins between
    # the bursts the probe keeps must not leak energy into the 4-8 kHz band.
    rng = np.random.default_rng(7)
    burst_len = SR // 2
    spectrum = np.fft.rfft(rng.standard_normal(burst_len))
    freqs = np.fft.rfftfreq(burst_len, 1 / SR)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0
    burst = np.fft.irfft(spectrum, burst_len)
    burst = 0.3 * burst / np.max(np.abs(burst))
    probe = BandwidthProbe(SR)
    for _ in range(30):
        feed(probe, burst)
        feed(probe, np.zeros(SR // 4))
    result = probe.verdict()
    assert result is not None
    assert result["verdict"] == "narrowband"
    assert result["high_band_ratio"] < 0.001


def test_verdict_is_final_once_it_exists():
    probe = BandwidthProbe(SR)
    feed(probe, tone(1000, 10.5))
    first = probe.verdict()
    feed(probe, tone(6000, 30.0))
    assert probe.verdict() == first


@pytest.mark.parametrize(
    "frame",
    [
        rtc.AudioFrame(bytes(2 * 160), 16000, 1, 160),  # another sample rate
        rtc.AudioFrame(bytes(2 * 2 * 1200), SR, 2, 1200),  # stereo
    ],
    ids=["other-rate", "stereo"],
)
def test_probe_refuses_frames_it_cannot_measure(frame):
    with pytest.raises(ValueError):
        BandwidthProbe(SR).add(frame)
