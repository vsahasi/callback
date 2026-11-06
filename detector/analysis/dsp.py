"""Numpy signal-processing primitives shared by the classical detectors.

Everything here is plain numpy. No scipy, no librosa, no torch. The FFT,
autocorrelation, filterbank-free band energies, cepstrum and run-length
helpers below are the whole toolbox the detectors in this package stand on,
which keeps the detector service installable without a second CUDA wheel.

Two conventions hold throughout:

* Frames. One framing grid is computed once per call (`build_context`) and
  every detector reads the same `Spectrogram`, `Activity` and `PitchTrack`.
  That guarantees a frame index means the same instant in every module, and
  it means the STFT is paid for once.
* Scale invariance. Apart from `quantisation_step`, which deliberately looks
  at the raw sample lattice, every measurement is a ratio or a difference in
  dB relative to the active-speech level. A detector must not change its mind
  because the caller turned the gain up.

Nothing in here raises on degenerate input. Silence, a single sample, pure
noise, a DC ramp and a 4 kHz sample rate all produce finite numbers; it is the
detectors' job to notice the numbers are uninformative and say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EPS",
    "MAX_WORKING_RATE",
    "MIN_USABLE_RATE",
    "Activity",
    "AudioBlock",
    "AudioContext",
    "FrameGrid",
    "PitchTrack",
    "Spectrogram",
    "amplitude_perturbation_db",
    "band_energy",
    "build_context",
    "circular_std",
    "clamp01",
    "envelope_periodicity",
    "f0_perturbation_pct",
    "frames",
    "interior_gap_mask",
    "linear_slope",
    "modulation_spectrum",
    "pearson",
    "prepare",
    "quantisation_step",
    "real_cepstrum",
    "resample_fft",
    "rolloff_hz",
    "safe_db",
    "smooth_mask",
    "spectral_flatness",
    "true_runs",
    "wrap_phase",
]

EPS = 1e-12

# Below this the band structure of speech is gone and nothing in this package
# is meaningful. 4 kHz keeps 2 kHz of bandwidth, which is thin but analysable.
MIN_USABLE_RATE = 4000

# Analysis runs at the native rate up to here. Phone audio is 8 kHz, LiveKit
# gives 16 kHz or 48 kHz; anything above is decimated because nothing this
# package measures lives above 8 kHz and the extra samples only cost time.
MAX_WORKING_RATE = 16_000

# The reference band for every "is this band-limited" question. It is inside
# the 8 kHz-available range, so the same numbers mean the same thing whether
# the call arrived at 8 kHz or 16 kHz.
SPEECH_BAND_LO = 300.0
SPEECH_BAND_HI = 2000.0
HIGH_BAND_LO = 2000.0
HIGH_BAND_HI = 3800.0
LOW_BAND_LO = 70.0
LOW_BAND_HI = 250.0


def clamp01(value: float) -> float:
    """Clamp to [0, 1] and turn a non-finite value into 0.5.

    A NaN score would propagate into the risk blend and out to a reviewer as
    a confident-looking number, so non-finite collapses to "no opinion".
    """

    if not np.isfinite(value):
        return 0.5
    return float(min(1.0, max(0.0, value)))


def safe_db(value: float | np.ndarray, floor: float = -200.0) -> np.ndarray:
    """20*log10 with a floor, so silence is a number rather than -inf."""

    magnitude = np.abs(np.asarray(value, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 20.0 * np.log10(np.maximum(magnitude, EPS))
    return np.maximum(np.nan_to_num(out, nan=floor, neginf=floor), floor)


def wrap_phase(angles: np.ndarray) -> np.ndarray:
    """Wrap radians into (-pi, pi]."""

    return (np.asarray(angles) + np.pi) % (2.0 * np.pi) - np.pi


def circular_std(angles: np.ndarray, axis: int = -1) -> np.ndarray:
    """Circular standard deviation in radians, 0 for a perfectly fixed angle.

    sqrt(-2 ln R) where R is the resultant length. Unbounded above, so it is
    clipped at 4 rad (R ~ 3e-4), far past anything a detector cares about.
    """

    data = np.asarray(angles, dtype=np.float64)
    if data.size == 0:
        return np.zeros(())
    resultant = np.abs(np.mean(np.exp(1j * data), axis=axis))
    resultant = np.clip(resultant, 1e-4, 1.0)
    return np.sqrt(np.maximum(-2.0 * np.log(resultant), 0.0))


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation, 0.0 when either side is constant or empty."""

    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    size = min(left.size, right.size)
    if size < 3:
        return 0.0
    left = left[:size] - left[:size].mean()
    right = right[:size] - right[:size].mean()
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator <= EPS:
        return 0.0
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))


def linear_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Least-squares slope dy/dx, 0.0 when x is degenerate."""

    x_data = np.asarray(x_values, dtype=np.float64).ravel()
    y_data = np.asarray(y_values, dtype=np.float64).ravel()
    if x_data.size < 2 or x_data.size != y_data.size:
        return 0.0
    x_centred = x_data - x_data.mean()
    variance = float(np.sum(x_centred**2))
    if variance <= EPS:
        return 0.0
    return float(np.sum(x_centred * (y_data - y_data.mean())) / variance)


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) spans of consecutive True values."""

    flags = np.asarray(mask, dtype=bool)
    if flags.size == 0:
        return []
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1).tolist()
    stops = np.flatnonzero(edges == -1).tolist()
    return list(zip(starts, stops, strict=True))


def interior_gap_mask(speech: np.ndarray, erode: int = 2) -> np.ndarray:
    """Non-speech frames bounded by speech on both sides, eroded at the edges.

    This is the right mask for anything that asks a question about the gaps
    *between words*: a reverberant tail, a synthetic noise floor, background
    drift. Leading and trailing silence is excluded because it carries the
    line's own noise rather than anything about the talker, and the frames
    immediately next to speech are excluded because a 32 ms analysis window
    smears across the boundary.
    """

    flags = np.asarray(speech, dtype=bool)
    gap = np.zeros(flags.size, dtype=bool)
    for start, stop in true_runs(~flags):
        if start == 0 or stop == flags.size:
            continue
        inner_start = start + erode
        inner_stop = stop - erode
        if inner_stop > inner_start:
            gap[inner_start:inner_stop] = True
    return gap


def envelope_periodicity(
    envelope: np.ndarray,
    hop_s: float,
    min_lag_s: float = 0.25,
    max_lag_s: float = 2.5,
) -> tuple[float, float]:
    """Strongest periodicity in the amplitude envelope: (strength, period).

    Autocorrelation of the mean-removed envelope, normalised by the zero lag
    and corrected for the shrinking overlap at long lags. A talker who
    alternates speech and pause on a fixed schedule scores high; spontaneous
    speech does not, because nothing repeats.

    Returns (0.0, 0.0) when the recording is too short to contain two cycles
    of the shortest lag considered.
    """

    data = np.asarray(envelope, dtype=np.float64).ravel()
    size = data.size
    if size < 16 or hop_s <= 0.0:
        return 0.0, 0.0
    # size // 3 rather than size // 2: at longer lags so few sample pairs
    # overlap that the bias correction below starts amplifying noise into
    # spurious long-period "rhythms".
    max_lag = min(int(max_lag_s / hop_s), size // 3)
    min_lag = max(1, int(min_lag_s / hop_s))
    if max_lag <= min_lag:
        return 0.0, 0.0

    centred = data - data.mean()
    n_fft = int(2 ** int(np.ceil(np.log2(max(2 * size, 4)))))
    transform = np.fft.rfft(centred, n_fft)
    autocorr = np.fft.irfft(transform * np.conjugate(transform), n_fft)[:size]
    if autocorr[0] <= EPS:
        return 0.0, 0.0
    autocorr = autocorr / autocorr[0]

    lags = np.arange(min_lag, max_lag + 1)
    # Fewer sample pairs overlap at long lags, which biases the raw
    # correlation down. max_lag is capped at size/2 so this correction can
    # never exceed a factor of two.
    overlap = np.maximum(size - lags, 1)
    corrected = autocorr[lags] * np.minimum(size / overlap, 1.3)
    best = int(np.argmax(corrected))
    return float(np.clip(corrected[best], 0.0, 1.0)), float(lags[best] * hop_s)


def smooth_mask(mask: np.ndarray, min_run: int, max_gap: int) -> np.ndarray:
    """Bridge gaps up to `max_gap`, then drop runs shorter than `min_run`.

    Applied in that order on purpose: a word split by one quiet frame should
    survive as one word rather than being deleted as two fragments.
    """

    out = np.asarray(mask, dtype=bool).copy()
    if out.size == 0:
        return out
    for start, stop in true_runs(~out):
        interior = start > 0 and stop < out.size
        if interior and (stop - start) <= max_gap:
            out[start:stop] = True
    for start, stop in true_runs(out):
        if (stop - start) < min_run:
            out[start:stop] = False
    return out


def frames(signal: np.ndarray, win: int, hop: int) -> np.ndarray:
    """`(n_frames, win)` view of overlapping frames, zero-padded if needed."""

    data = np.asarray(signal, dtype=np.float64).ravel()
    if win <= 0 or hop <= 0:
        raise ValueError(f"win and hop must be positive, got {win} and {hop}")
    if data.size < win:
        data = np.pad(data, (0, win - data.size))
    windows = np.lib.stride_tricks.sliding_window_view(data, win)
    return np.ascontiguousarray(windows[::hop])


def resample_fft(signal: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited resample to `n_out` samples by truncating the spectrum.

    The same algorithm as `scipy.signal.resample`. Good enough for decimating
    48 kHz call audio to the working rate; it is not used in any measurement
    path, only to normalise the sample rate before one.
    """

    data = np.asarray(signal, dtype=np.float64).ravel()
    n_in = data.size
    if n_out <= 0 or n_in == 0:
        return np.zeros(max(n_out, 0), dtype=np.float64)
    if n_out == n_in:
        return data
    spectrum = np.fft.rfft(data)
    out_bins = n_out // 2 + 1
    keep = min(out_bins, spectrum.size)
    resized = np.zeros(out_bins, dtype=np.complex128)
    resized[:keep] = spectrum[:keep]
    if n_out % 2 == 0 and n_out < n_in and out_bins > 0:
        # The Nyquist bin of an even-length real signal must be real.
        resized[-1] = resized[-1].real
    return np.fft.irfft(resized, n_out) * (n_out / n_in)


def spectral_flatness(values: np.ndarray) -> float:
    """Geometric over arithmetic mean of non-negative values, in [0, 1].

    1.0 is perfectly flat (white), near 0 is one dominant peak. Used on
    modulation spectra, where a sharp peak means a metronomic rhythm.
    """

    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size < 2:
        return 1.0
    data = np.maximum(data, 0.0)
    arithmetic = float(np.mean(data))
    if arithmetic <= EPS:
        return 1.0
    geometric = float(np.exp(np.mean(np.log(data + EPS))))
    return clamp01(geometric / arithmetic)


def band_energy(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    """Summed power in [lo, hi) Hz. 0.0 when the band is outside Nyquist."""

    frequencies = np.asarray(freqs, dtype=np.float64)
    spectrum = np.asarray(power, dtype=np.float64)
    if frequencies.size == 0 or spectrum.size == 0:
        return 0.0
    selected = (frequencies >= lo) & (frequencies < hi)
    if not np.any(selected):
        return 0.0
    return float(np.sum(spectrum[selected]))


def rolloff_hz(freqs: np.ndarray, power: np.ndarray, fraction: float = 0.95) -> float:
    """Frequency below which `fraction` of the spectral energy lies."""

    frequencies = np.asarray(freqs, dtype=np.float64)
    spectrum = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    total = float(np.sum(spectrum))
    if total <= EPS or frequencies.size == 0:
        return 0.0
    cumulative = np.cumsum(spectrum) / total
    index = int(np.searchsorted(cumulative, fraction))
    index = min(index, frequencies.size - 1)
    return float(frequencies[index])


def real_cepstrum(log_spectrum: np.ndarray) -> np.ndarray:
    """Real cepstrum of a one-sided log spectrum.

    Index q is a delay of q / sample_rate seconds, so a fixed early
    reflection at 2.4 ms appears as a peak near q = 0.0024 * sample_rate.
    """

    data = np.asarray(log_spectrum, dtype=np.float64).ravel()
    if data.size < 4:
        return np.zeros(4, dtype=np.float64)
    n_time = 2 * (data.size - 1)
    return np.fft.irfft(data.astype(np.complex128), n_time)


def quantisation_step(signal: np.ndarray) -> tuple[float, float, float]:
    """Infer the sample lattice: (relative step, on-lattice fraction, bits).

    A signal that has been decoded, played out, re-recorded and requantised
    at a coarser depth than the capture path sits on a visibly coarse
    lattice. This is a real forensic cue, but a fragile one: any lossy codec
    or resampling in the chain destroys it, so detectors that use it must
    treat a negative result as no evidence rather than as evidence of
    innocence.

    Returns (0.0, 0.0, 0.0) when there is not enough distinct amplitude to
    infer anything.
    """

    data = np.asarray(signal, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if data.size < 256 or peak <= EPS:
        return 0.0, 0.0, 0.0
    unique = np.unique(np.abs(data))
    unique = unique[unique > peak * 1e-9]
    if unique.size < 16:
        # Fewer than 16 distinct levels is either a synthetic test tone or
        # near-silence. Either way the lattice says nothing about replay.
        return 0.0, 0.0, 0.0
    gaps = np.diff(unique)
    gaps = gaps[gaps > peak * 1e-9]
    if gaps.size == 0:
        return 0.0, 0.0, 0.0
    step = float(np.percentile(gaps, 5.0))
    if step <= peak * 1e-9:
        return 0.0, 0.0, 0.0
    ratios = data / step
    residual = np.abs(ratios - np.round(ratios))
    on_lattice = float(np.mean(residual < 1e-4))
    relative = step / peak
    bits = float(np.log2(1.0 / max(relative, 1e-12)))
    return relative, on_lattice, bits


def modulation_spectrum(
    envelope: np.ndarray, hop_s: float, f_max: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of the amplitude envelope, up to `f_max` Hz.

    The 2-8 Hz region is the syllable rate. Natural speech puts a broad hump
    there; a template-driven talker puts a spike.
    """

    data = np.asarray(envelope, dtype=np.float64).ravel()
    if data.size < 8 or hop_s <= 0.0:
        return np.zeros(0), np.zeros(0)
    centred = data - data.mean()
    window = np.hanning(centred.size)
    spectrum = np.abs(np.fft.rfft(centred * window))
    freqs = np.fft.rfftfreq(centred.size, d=hop_s)
    selected = (freqs > 0.4) & (freqs <= f_max)
    if not np.any(selected):
        return np.zeros(0), np.zeros(0)
    return freqs[selected], spectrum[selected]


# ---------------------------------------------------------------------------
# Framing grid and the per-call analysis context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameGrid:
    """One framing configuration, shared by every detector in a call."""

    win_ms: float = 32.0
    hop_ms: float = 10.0

    def win(self, sample_rate: int) -> int:
        return max(32, round(self.win_ms * sample_rate / 1000.0))

    def hop(self, sample_rate: int) -> int:
        return max(8, round(self.hop_ms * sample_rate / 1000.0))

    def n_fft(self, sample_rate: int) -> int:
        """Next power of two at or above the window, doubled.

        Doubling keeps the bin spacing at roughly 15 Hz at both 8 kHz and
        16 kHz, so a frequency-domain ripple measured in bins means the same
        thing at either rate.
        """

        win = self.win(sample_rate)
        return int(2 ** (int(np.ceil(np.log2(max(win, 32)))) + 1))


DEFAULT_GRID = FrameGrid()


@dataclass(frozen=True)
class AudioBlock:
    """Conditioned audio plus the facts about how bad the input was."""

    samples: np.ndarray
    # The same audio before DC removal and before any resampling. Only the
    # requantisation cue uses it: subtracting a mean that is not a multiple of
    # the quantisation step destroys the very lattice that cue looks for, and
    # resampling destroys it outright.
    raw_samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    duration_s: float
    peak: float
    rms: float
    dc_offset: float
    clipped_fraction: float
    non_finite_fraction: float
    resampled: bool
    too_short: bool
    rate_too_low: bool

    @property
    def usable(self) -> bool:
        """False when nothing downstream can be trusted at all."""

        return not (self.too_short or self.rate_too_low) and self.rms > EPS


@dataclass(frozen=True)
class Spectrogram:
    spectrum: np.ndarray  # (n_frames, n_bins) complex
    power: np.ndarray  # (n_frames, n_bins) real
    freqs: np.ndarray  # (n_bins,)
    rms: np.ndarray  # (n_frames,) linear frame RMS
    energy_db: np.ndarray  # (n_frames,)
    hop_s: float
    win_s: float
    sample_rate: int
    n_fft: int

    @property
    def n_frames(self) -> int:
        return int(self.power.shape[0])


@dataclass(frozen=True)
class Activity:
    """Energy-based speech/silence segmentation."""

    speech: np.ndarray  # bool per frame
    energy_db: np.ndarray
    threshold_db: float
    speech_db: float
    floor_db: float
    hop_s: float
    has_structure: bool  # False for silence, a tone, or stationary noise

    @property
    def speech_frames(self) -> int:
        return int(np.count_nonzero(self.speech))

    @property
    def speech_seconds(self) -> float:
        return self.speech_frames * self.hop_s

    @property
    def dynamic_range_db(self) -> float:
        return float(self.speech_db - self.floor_db)


@dataclass(frozen=True)
class PitchTrack:
    """Short-term autocorrelation pitch, 10 ms hop, parabolically refined."""

    f0: np.ndarray  # Hz, 0.0 where unvoiced
    voiced: np.ndarray  # bool per frame
    strength: np.ndarray  # normalised autocorrelation peak in [0, 1]
    hop_s: float

    @property
    def voiced_frames(self) -> int:
        return int(np.count_nonzero(self.voiced))

    @property
    def voiced_seconds(self) -> float:
        return self.voiced_frames * self.hop_s

    @property
    def median_f0(self) -> float:
        values = self.f0[self.voiced]
        if values.size == 0:
            return 0.0
        return float(np.median(values))

    @property
    def median_period_s(self) -> float:
        median = self.median_f0
        return 1.0 / median if median > 0.0 else 0.0

    def semitone_spread(self) -> float:
        """Interquartile F0 spread in semitones. 0.0 for a monotone."""

        values = self.f0[self.voiced]
        values = values[values > 0.0]
        if values.size < 8:
            return 0.0
        low, high = np.percentile(values, [10.0, 90.0])
        if low <= 0.0:
            return 0.0
        return float(12.0 * np.log2(high / low))


@dataclass(frozen=True)
class AudioContext:
    """Everything the detectors need, computed once."""

    block: AudioBlock
    spec: Spectrogram
    activity: Activity
    pitch: PitchTrack
    grid: FrameGrid = DEFAULT_GRID
    notes: tuple[str, ...] = field(default=())

    @property
    def sample_rate(self) -> int:
        return self.block.sample_rate

    @property
    def nyquist(self) -> float:
        return self.block.sample_rate / 2.0

    def mean_log_spectrum(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Mean of the per-frame log power spectrum over selected frames.

        Averaging the *log* spectrum across frames is what makes a fixed
        channel response separable from voiced harmonics: the harmonic comb
        moves with F0 across the utterance and smears, while a room or
        loudspeaker response is the same in every frame and survives.
        """

        power = self.spec.power
        if power.size == 0:
            return np.zeros(self.spec.freqs.size)
        if mask is None:
            selected = power
        else:
            flags = np.asarray(mask, dtype=bool)
            selected = power[flags] if np.any(flags) else power
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log10(np.maximum(selected, EPS))
        return np.nan_to_num(np.mean(logs, axis=0))


def prepare(audio: object, sample_rate: int) -> AudioBlock:
    """Coerce any mono-ish audio into an `AudioBlock`, never raising.

    Accepts a numpy array, a torch tensor (duck-typed, so torch is not
    imported), a nested sequence or a memoryview. Multi-channel input is
    downmixed. Non-finite samples are zeroed and counted. The DC offset is
    removed, because a battery-powered handset preamp commonly has one and it
    would otherwise dominate the low-band energy measurement.

    The signal is NOT level-normalised: `quantisation_step` needs the raw
    lattice, and every other measurement here is already scale-invariant.
    """

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    data = _to_numpy(audio)
    if data.ndim > 1:
        # (channels, samples) or (samples, channels): collapse the short axis.
        axis = int(np.argmin(data.shape))
        data = np.mean(data, axis=axis)
    data = np.asarray(data, dtype=np.float64).ravel()

    finite = np.isfinite(data)
    non_finite_fraction = float(1.0 - np.mean(finite)) if data.size else 1.0
    if non_finite_fraction > 0.0:
        data = np.where(finite, data, 0.0)

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > EPS:
        # Clipping shows up as a flat top: many samples sitting on the peak.
        clipped_fraction = float(np.mean(np.abs(data) >= peak * 0.999))
    else:
        clipped_fraction = 0.0

    raw_samples = data.copy()
    dc_offset = float(np.mean(data)) if data.size else 0.0
    data = data - dc_offset

    original_rate = int(sample_rate)
    working_rate = original_rate
    resampled = False
    if original_rate > MAX_WORKING_RATE and data.size > 0:
        n_out = round(data.size * MAX_WORKING_RATE / original_rate)
        data = resample_fft(data, max(n_out, 1))
        working_rate = MAX_WORKING_RATE
        resampled = True

    duration = data.size / working_rate if working_rate > 0 else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0

    return AudioBlock(
        samples=data,
        raw_samples=raw_samples,
        sample_rate=working_rate,
        original_sample_rate=original_rate,
        duration_s=duration,
        peak=peak,
        rms=rms,
        dc_offset=dc_offset,
        clipped_fraction=clipped_fraction,
        non_finite_fraction=non_finite_fraction,
        resampled=resampled,
        too_short=duration < 0.25,
        rate_too_low=original_rate < MIN_USABLE_RATE,
    )


def _to_numpy(audio: object) -> np.ndarray:
    """Duck-typed conversion, so torch stays an optional dependency."""

    if isinstance(audio, np.ndarray):
        return audio
    detach = getattr(audio, "detach", None)
    if callable(detach):  # torch.Tensor and anything that mimics one
        tensor = detach()
        to_cpu = getattr(tensor, "cpu", None)
        if callable(to_cpu):
            tensor = to_cpu()
        as_numpy = getattr(tensor, "numpy", None)
        if callable(as_numpy):
            return np.asarray(as_numpy())
        return np.asarray(tensor)
    as_numpy = getattr(audio, "numpy", None)
    if callable(as_numpy):
        return np.asarray(as_numpy())
    return np.asarray(audio)


def compute_spectrogram(
    block: AudioBlock, grid: FrameGrid = DEFAULT_GRID
) -> Spectrogram:
    """Hann-windowed STFT power and the per-frame RMS, on the shared grid."""

    rate = block.sample_rate
    win = grid.win(rate)
    hop = grid.hop(rate)
    n_fft = grid.n_fft(rate)

    framed = frames(block.samples, win, hop)
    window = np.hanning(win)
    normaliser = float(np.sum(window**2)) or 1.0
    windowed = framed * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=-1)
    power = (spectrum.real**2 + spectrum.imag**2) / normaliser
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / rate)
    rms = np.sqrt(np.mean(framed**2, axis=-1))

    return Spectrogram(
        spectrum=spectrum,
        power=power,
        freqs=freqs,
        rms=rms,
        energy_db=safe_db(rms),
        hop_s=hop / rate,
        win_s=win / rate,
        sample_rate=rate,
        n_fft=n_fft,
    )


def detect_activity(spec: Spectrogram) -> Activity:
    """Adaptive energy VAD with hysteresis.

    The threshold sits 8 dB above the noise floor but never more than 25 dB
    below the loud end, which keeps quiet voiced tails inside the speech mask
    on a noisy line. When the dynamic range is under 6 dB there is no
    speech/silence structure to find at all (silence, a tone, stationary
    noise), which is reported rather than guessed at.
    """

    energy_db = np.asarray(spec.energy_db, dtype=np.float64)
    if energy_db.size == 0:
        return Activity(
            speech=np.zeros(0, dtype=bool),
            energy_db=energy_db,
            threshold_db=-200.0,
            speech_db=-200.0,
            floor_db=-200.0,
            hop_s=spec.hop_s,
            has_structure=False,
        )

    floor_db = float(np.percentile(energy_db, 10.0))
    loud_db = float(np.percentile(energy_db, 95.0))
    dynamic_range = loud_db - floor_db
    has_structure = dynamic_range >= 6.0

    if not has_structure:
        # No usable split. Treat anything above the absolute noise gate as
        # "material" so downstream measures still have frames to work with,
        # and flag that the segmentation is not trustworthy.
        speech = energy_db > (loud_db - 3.0)
        speech_db = float(np.median(energy_db[speech])) if np.any(speech) else loud_db
        return Activity(
            speech=speech,
            energy_db=energy_db,
            threshold_db=loud_db - 3.0,
            speech_db=speech_db,
            floor_db=floor_db,
            hop_s=spec.hop_s,
            has_structure=False,
        )

    threshold = max(floor_db + 8.0, loud_db - 25.0)
    raw = energy_db > threshold
    min_run = max(2, round(0.03 / max(spec.hop_s, 1e-6)))
    max_gap = max(1, round(0.03 / max(spec.hop_s, 1e-6)))
    speech = smooth_mask(raw, min_run=min_run, max_gap=max_gap)
    if not np.any(speech):
        speech = raw
    speech_db = float(np.median(energy_db[speech])) if np.any(speech) else loud_db

    return Activity(
        speech=speech,
        energy_db=energy_db,
        threshold_db=threshold,
        speech_db=speech_db,
        floor_db=floor_db,
        hop_s=spec.hop_s,
        has_structure=True,
    )


def track_pitch(
    block: AudioBlock,
    activity: Activity,
    grid: FrameGrid = DEFAULT_GRID,
    f_min: float = 60.0,
    f_max: float = 400.0,
) -> PitchTrack:
    """Centre-clipped autocorrelation pitch on a 40 ms window, 10 ms hop.

    The window is longer than the shared 32 ms analysis window so that two
    full periods of a 60 Hz voice fit. The peak lag is refined by parabolic
    interpolation, which matters: without it the F0 estimate is quantised to
    the sample grid and the perturbation measures read that quantisation as
    jitter.
    """

    rate = block.sample_rate
    hop = grid.hop(rate)
    win = max(grid.win(rate), round(0.040 * rate))
    n_frames = activity.speech.size

    # The pitch window is longer than the shared analysis window, so without
    # padding this grid would come out one frame short and every per-frame
    # comparison downstream would need an offset. Pad instead.
    signal = block.samples
    if n_frames > 0:
        needed = (n_frames - 1) * hop + win
        if signal.size < needed:
            signal = np.pad(signal, (0, needed - signal.size))
    framed = frames(signal, win, hop)
    if n_frames > 0 and framed.shape[0] > n_frames:
        framed = framed[:n_frames]

    count = framed.shape[0]
    f0 = np.zeros(count, dtype=np.float64)
    strength = np.zeros(count, dtype=np.float64)

    min_lag = max(2, int(np.floor(rate / f_max)))
    max_lag = min(win - 2, int(np.ceil(rate / f_min)))
    if max_lag <= min_lag + 1 or count == 0:
        return PitchTrack(
            f0=f0,
            voiced=np.zeros(count, dtype=bool),
            strength=strength,
            hop_s=hop / rate,
        )

    n_corr = int(2 ** int(np.ceil(np.log2(2 * win))))
    centred = framed - framed.mean(axis=-1, keepdims=True)

    # Low-pass to 1 kHz first. On telephone audio the 300 Hz high-pass has
    # already removed the fundamental of most adult voices, so the loudest
    # thing in the frame is a formant; without this the autocorrelation locks
    # onto the formant period and the whole track is an octave or two out.
    # Only harmonics 1-7 of a 140 Hz voice are needed to recover the period.
    low_passed = _low_pass_frames(centred, rate, 1000.0)

    # Centre clipping flattens what is left of the formant ripple, which
    # sharpens the periodic peak.
    clip_level = 0.3 * np.max(np.abs(low_passed), axis=-1, keepdims=True)
    clipped = np.sign(low_passed) * np.maximum(np.abs(low_passed) - clip_level, 0.0)

    transform = np.fft.rfft(clipped, n=n_corr, axis=-1)
    autocorr = np.fft.irfft(transform * np.conjugate(transform), n=n_corr, axis=-1)
    zero_lag = autocorr[:, 0:1]
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.nan_to_num(autocorr / np.maximum(zero_lag, EPS))

    window_slice = normalised[:, min_lag : max_lag + 1]
    best = _octave_corrected_peak(window_slice, min_lag)
    peak_value = window_slice[np.arange(count), best]
    lags = best + min_lag

    # Parabolic refinement around the integer peak.
    left = normalised[np.arange(count), np.maximum(lags - 1, 0)]
    right = normalised[np.arange(count), np.minimum(lags + 1, n_corr - 1)]
    denominator = left - 2.0 * peak_value + right
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(
            np.abs(denominator) > EPS, 0.5 * (left - right) / denominator, 0.0
        )
    offset = np.clip(np.nan_to_num(offset), -0.5, 0.5)
    refined_lag = lags + offset

    voiced = (peak_value > 0.35) & (refined_lag > 1.0)
    if activity.speech.size == count:
        voiced &= activity.speech
    with np.errstate(divide="ignore", invalid="ignore"):
        f0 = np.where(voiced, rate / np.maximum(refined_lag, 1.0), 0.0)
    f0 = np.nan_to_num(f0)
    voiced &= (f0 >= f_min) & (f0 <= f_max)
    f0 = np.where(voiced, f0, 0.0)
    f0 = _repair_octave_jumps(f0, voiced)
    voiced &= (f0 >= f_min) & (f0 <= f_max)
    f0 = np.where(voiced, f0, 0.0)

    # Plausibility gate. One talker's F0 over a few seconds does not span
    # more than about an octave, so a frame nearly two octaves from the
    # utterance median is a tracking error, not a voice. Those frames arrive
    # in runs (a fricative, the quiet edge of a syllable), which is why the
    # local-median repair above cannot catch them: inside the run the local
    # median is wrong too. Unvoicing them costs a little coverage and buys a
    # pitch range that means something.
    if int(np.count_nonzero(voiced)) >= 8:
        median_f0 = float(np.median(f0[voiced]))
        if median_f0 > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(voiced, f0 / median_f0, 1.0)
            ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
            voiced &= (ratio >= 1.0 / 1.78) & (ratio <= 1.78)
            f0 = np.where(voiced, f0, 0.0)

    return PitchTrack(
        f0=f0,
        voiced=voiced,
        strength=np.clip(np.nan_to_num(peak_value), 0.0, 1.0),
        hop_s=hop / rate,
    )


def _low_pass_frames(
    framed: np.ndarray, sample_rate: int, cutoff_hz: float
) -> np.ndarray:
    """Zero-phase low-pass each frame, for pitch analysis only."""

    length = framed.shape[-1]
    if length < 8 or cutoff_hz <= 0.0:
        return framed
    spectrum = np.fft.rfft(framed, axis=-1)
    freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    response = 1.0 / (1.0 + (freqs / cutoff_hz) ** 8)
    return np.fft.irfft(spectrum * response, n=length, axis=-1)


def _octave_corrected_peak(window: np.ndarray, min_lag: int) -> np.ndarray:
    """Autocorrelation peak index, with multiples of the period folded down.

    For a periodic signal the autocorrelation peaks at every multiple of the
    period, and over a 40 ms window those peaks are nearly the same height.
    The global maximum therefore lands on a multiple about as often as on the
    period itself, which halves or thirds the reported F0 and makes the
    perturbation measurements meaningless: one halved period reads as a 100%
    jitter event.

    Neither "take the global maximum" nor "take the shortest strong peak"
    survives contact with speech. The first reports an octave too low on a
    clean harmonic stack; the second reports an octave too high on speech,
    where formant residue and aspiration leave strong peaks at half the
    period. So this does what production pitch trackers do: start from the
    global maximum and test its exact submultiples. A candidate is accepted
    only if it is an integer division of the chosen lag and still carries 80%
    of the peak, which an unrelated noise spike cannot be.
    """

    count, width = window.shape
    best = np.argmax(window, axis=-1)
    if width < 3:
        return best[:count]

    corrected = best.copy()
    for frame in range(count):
        index = int(best[frame])
        peak = float(window[frame, index])
        if peak <= 0.0:
            continue
        lag = min_lag + index
        for divisor in (3, 2):
            candidate = round(lag / divisor) - min_lag
            if candidate < 1 or candidate >= width - 1:
                continue
            # One bin of slack around the submultiple, for the rounding.
            low = max(candidate - 1, 0)
            high = min(candidate + 2, width)
            local = int(np.argmax(window[frame, low:high])) + low
            if float(window[frame, local]) >= 0.80 * peak:
                corrected[frame] = local
                break
    return corrected[:count]


def _repair_octave_jumps(
    f0: np.ndarray, voiced: np.ndarray, half_window: int = 7
) -> np.ndarray:
    """Snap outlying frames onto the octave nearest the local median.

    The per-frame correction in `_octave_corrected_peak` decides each frame
    in isolation, and on real speech a few frames still land an octave out:
    aspiration noise and the quiet edges of a syllable leave a strong
    autocorrelation peak at half the period. Those frames do not corrupt the
    perturbation measure, which rejects them as tracking errors, but they do
    corrupt the pitch *range*, which is a liveness cue.

    This only ever multiplies a frame's F0 by a power-of-small-integer factor
    and only when the frame is already more than 25% away from its
    neighbours, so genuine micro-variation (fractions of a percent) and
    genuine contours (tens of percent) pass through untouched.
    """

    flags = np.asarray(voiced, dtype=bool)
    if int(np.count_nonzero(flags)) < 5:
        return f0

    out = np.asarray(f0, dtype=np.float64).copy()
    indices = np.flatnonzero(flags)
    for position, frame in enumerate(indices):
        low = max(position - half_window, 0)
        high = min(position + half_window + 1, indices.size)
        reference = float(np.median(f0[indices[low:high]]))
        value = float(f0[frame])
        if reference <= 0.0 or value <= 0.0:
            continue
        ratio = value / reference
        if 0.8 <= ratio <= 1.25:
            continue
        best = value
        best_distance = abs(float(np.log2(ratio)))
        for factor in (0.25, 1.0 / 3.0, 0.5, 2.0, 3.0, 4.0):
            candidate = value * factor
            distance = abs(float(np.log2(candidate / reference)))
            # Only when the alternative is clearly closer, so a frame in the
            # middle between two octaves is left alone.
            if distance < best_distance - 0.08:
                best = candidate
                best_distance = distance
        out[frame] = best
    return out


def f0_perturbation_pct(
    pitch: PitchTrack, min_run: int = 5, outlier: float = 0.20
) -> float:
    """Relative average perturbation (RAP) of the glottal period, in percent.

        RAP = mean |T_i - mean(T_{i-1}, T_i, T_{i+1})| / mean T

    RAP rather than the simpler mean |T_i - T_{i+1}| for one specific
    reason: the simple version also measures the pitch *contour*. A talker
    gliding five semitones across a syllable changes the period by most of a
    percent per frame, so the simple statistic would mostly report how
    expressive the delivery was, and the pitch range is already a separate
    cue. Subtracting a three-point moving average removes any slope and
    leaves the frame-to-frame wobble.

    Adjacent frames whose period changes by more than `outlier` are dropped
    as tracking errors rather than scored as enormous jitter. That is
    standard practice in jitter measurement and it is necessary here: one
    octave error would otherwise swamp a whole utterance.

    Measured on a 10 ms frame grid, not period-synchronously, so the numbers
    are not comparable with published clinical jitter values. They are
    comparable between two recordings measured this way, which is all the
    detectors use them for.

    The working range is narrow. On the synthetic corpus a messy talker with
    the jitter switched off still reads about 0.18%, which is this
    estimator's own noise on non-stationary audio, against about 0.37% with
    jitter on. Anything built on this cue has to tolerate a factor of two,
    not a factor of ten.

    Returns 0.0 when no voiced run was long enough, which callers must
    distinguish from a genuine zero.
    """

    deviations: list[float] = []
    periods_kept: list[float] = []
    for start, stop in true_runs(pitch.voiced):
        if (stop - start) < min_run:
            continue
        values = pitch.f0[start:stop]
        values = values[values > 0.0]
        if values.size < min_run:
            continue
        periods = 1.0 / values
        steady = np.abs(np.diff(periods)) / periods[:-1] <= outlier
        for index in range(1, periods.size - 1):
            if not (steady[index - 1] and steady[index]):
                continue
            local = float(np.mean(periods[index - 1 : index + 2]))
            deviations.append(abs(float(periods[index]) - local))
            periods_kept.append(float(periods[index]))
    if not deviations:
        return 0.0
    mean_period = float(np.mean(periods_kept))
    if mean_period <= EPS:
        return 0.0
    return float(100.0 * float(np.mean(deviations)) / mean_period)


def amplitude_perturbation_db(
    spec: Spectrogram, pitch: PitchTrack, min_run: int = 5
) -> float:
    """Shimmer analogue of `f0_perturbation_pct`, in dB.

    Three-point detrended for the same reason: without it the measurement is
    dominated by the syllable envelope rising and falling, which says nothing
    about whether a larynx produced the sound.
    """

    energies = np.asarray(spec.energy_db, dtype=np.float64)
    voiced = np.asarray(pitch.voiced, dtype=bool)
    size = min(energies.size, voiced.size)
    if size == 0:
        return 0.0
    energies = energies[:size]
    voiced = voiced[:size]

    deviations: list[float] = []
    for start, stop in true_runs(voiced):
        if (stop - start) < min_run:
            continue
        segment = energies[start:stop]
        local = (segment[:-2] + segment[1:-1] + segment[2:]) / 3.0
        deviations.extend(np.abs(segment[1:-1] - local).tolist())
    if not deviations:
        return 0.0
    return float(np.mean(deviations))


def build_context(
    audio: object, sample_rate: int, grid: FrameGrid = DEFAULT_GRID
) -> AudioContext:
    """Condition the audio and compute the shared analysis products once."""

    block = prepare(audio, sample_rate)
    spec = compute_spectrogram(block, grid)
    activity = detect_activity(spec)
    pitch = track_pitch(block, activity, grid)

    notes: list[str] = []
    if block.rate_too_low:
        notes.append(f"sample rate {block.original_sample_rate} Hz is below 4 kHz")
    if block.too_short:
        notes.append(f"only {block.duration_s * 1000:.0f} ms of audio")
    if block.resampled:
        notes.append(
            f"decimated from {block.original_sample_rate} Hz to {block.sample_rate} Hz"
        )
    if block.non_finite_fraction > 0.0:
        notes.append(f"{block.non_finite_fraction:.1%} of samples were not finite")
    if block.clipped_fraction > 0.01:
        notes.append(f"{block.clipped_fraction:.1%} of samples clipped")
    if not activity.has_structure:
        notes.append("no speech/silence structure (silence, tone or steady noise)")

    return AudioContext(
        block=block,
        spec=spec,
        activity=activity,
        pitch=pitch,
        grid=grid,
        notes=tuple(notes),
    )
