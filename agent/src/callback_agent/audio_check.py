"""The HD-voice check: does the caller's audio carry any energy above 4 kHz?

A call carried as G.711 (8 kHz) has none; a G.722 call (16 kHz) has plenty. LiveKit exposes no codec
field and the room track is Opus either way, so measuring the audio is the only way to know.
"""

import numpy as np
from livekit import rtc

_LOWEST_HZ = 100  # keeps DC and mains hum out of the total
WIDEBAND_RATIO = 0.01  # above this share of energy in 4-8 kHz, the audio is wideband
_SPEECH_RMS = (
    0.01 * 32768
)  # a frame has to be louder than 1% of full scale (16-bit) to count as speech


def high_band_ratio(
    samples: np.ndarray, sample_rate: int, *, split_hz: int = 4000, top_hz: int = 8000
) -> float:
    """The energy in [split_hz, top_hz) over the energy in [100 Hz, top_hz), 0.0 for silence."""
    x = np.asarray(samples, dtype=np.float64)
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1 / sample_rate)
    total = power[(freqs >= _LOWEST_HZ) & (freqs < top_hz)].sum()
    if total == 0:
        return 0.0
    return float(power[(freqs >= split_hz) & (freqs < top_hz)].sum() / total)


class BandwidthProbe:
    """Collects `seconds` of speech from mono 16-bit frames, then says whether it is wideband.

    Frames at or below 1% of full scale (RMS) are skipped, so silence and line noise never count.
    The verdict does not change once it exists.
    """

    def __init__(self, sample_rate: int, *, seconds: float = 10.0) -> None:
        self._sample_rate = sample_rate
        self._wanted = round(seconds * sample_rate)
        self._kept = 0
        self._chunks: list[np.ndarray] = []
        self._ratio: float | None = None

    def add(self, frame: rtc.AudioFrame) -> None:
        if frame.sample_rate != self._sample_rate or frame.num_channels != 1:
            raise ValueError(
                f"BandwidthProbe takes mono audio at {self._sample_rate} Hz, "
                f"got {frame.num_channels} channel(s) at {frame.sample_rate} Hz"
            )
        if self._kept >= self._wanted:
            return
        samples = np.frombuffer(frame.data, dtype=np.int16)
        if np.sqrt(np.mean(samples.astype(np.float64) ** 2)) <= _SPEECH_RMS:
            return
        self._chunks.append(samples.copy())
        self._kept += samples.size

    def verdict(self) -> dict[str, float | str] | None:
        """None until enough speech is in, then the ratio and the verdict.

        The verdict is "wideband" above 1% of the energy in 4-8 kHz, otherwise "narrowband".
        """
        if self._ratio is None:
            if self._kept < self._wanted:
                return None
            speech = np.concatenate(self._chunks)[: self._wanted]
            self._ratio = high_band_ratio(speech, self._sample_rate)
            self._chunks = []
        return {
            "high_band_ratio": self._ratio,
            "verdict": "wideband" if self._ratio > WIDEBAND_RATIO else "narrowband",
        }
