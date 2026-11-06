"""A deterministic synthetic corpus, because the real one is not reachable.

The ASVspoof 2019 LA corpus lives on a university server that is down, so
there is no real attack audio to calibrate against. Everything in this module
exists to fill that gap honestly: it generates speech-like signals in which
each artefact the detectors look for is present or absent *by construction*,
so a test can assert that a detector moves in the right direction without
anyone having to agree on an absolute threshold.

What this corpus is good for:

* Proving a detector responds to the physics it claims to measure. Convolving
  a signal with a room impulse response really does add a comb and a tail,
  and the replay detector really must notice.
* Fixing thresholds to a sane starting point, and documenting where they came
  from.
* Regression safety. These signals are seeded, so a refactor that changes a
  measurement shows up immediately.

What it is emphatically NOT good for:

* Estimating a false-accept or false-reject rate. These are parametric
  signals, not speech. A detector's margin here says nothing about its margin
  on a real call.
* Validating the synthesis detector against modern TTS. `tts_speech` below is
  a deliberately naive synthesiser: fixed harmonic phases, no jitter, static
  formants. A 2024-era neural vocoder shares almost none of those weaknesses,
  and anyone reading these margins as evidence against such a system is
  reading them wrong.

Everything is `numpy.random.default_rng(seed)`: same seed, same samples, on
any machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LIVE",
    "TTS",
    "SpeechStyle",
    "clipped",
    "dc_offset",
    "impulse_train",
    "live_speech",
    "pure_tone",
    "replay_of",
    "room_impulse_response",
    "silence",
    "telephone_channel",
    "tts_speech",
    "white_noise",
]

# The band a telephone network imposes, applied to every generator so that
# "clean" means "clean phone call" and the band-limiting cues are not
# trivially separable.
TELEPHONE_BAND = (300.0, 3400.0)


@dataclass(frozen=True)
class SpeechStyle:
    """How messy the generated talker is.

    Each field maps onto exactly one thing a detector measures, so a test can
    change one and watch one score move.
    """

    f0_hz: float = 132.0
    contour_semitones: float = 5.0  # pitch range of the contour
    jitter_pct: float = 1.2  # period-to-period perturbation
    shimmer_db: float = 1.6  # amplitude perturbation
    phase_drift_rad: float = 0.9  # per-harmonic slow phase wander
    formant_motion: float = 1.0  # 0 = static vocal tract
    aspiration: float = 0.03  # breathiness mixed into voiced frames
    floor_rel_db: float = -40.0  # inter-word floor, relative to speech RMS
    floor_drift_db: float = 4.0  # how much that floor moves
    breaths: bool = True
    breath_rel_db: float = -30.0
    rhythm_jitter: float = 0.8  # 0 = perfectly uniform speech/pause
    speech_run_s: float = 0.45
    pause_run_s: float = 0.25


# A live human at a handset: everything wobbles.
LIVE = SpeechStyle()

# A naive synthesiser: nothing wobbles, the floor is digital silence and the
# rhythm is a metronome.
TTS = SpeechStyle(
    f0_hz=132.0,
    contour_semitones=1.5,
    jitter_pct=0.0,
    shimmer_db=0.0,
    phase_drift_rad=0.0,
    formant_motion=0.12,
    aspiration=0.0,
    floor_rel_db=-75.0,
    floor_drift_db=0.05,
    breaths=False,
    rhythm_jitter=0.0,
)


# ---------------------------------------------------------------------------
# Speech-like generators
# ---------------------------------------------------------------------------


def live_speech(
    duration_s: float = 4.0,
    sample_rate: int = 8000,
    seed: int = 0,
    style: SpeechStyle = LIVE,
) -> np.ndarray:
    """Speech-like audio with human micro-variation, breath and rough rhythm."""

    return _render(duration_s, sample_rate, seed, style)


def tts_speech(
    duration_s: float = 4.0,
    sample_rate: int = 8000,
    seed: int = 0,
    style: SpeechStyle = TTS,
) -> np.ndarray:
    """The same machinery with every source of irregularity switched off."""

    return _render(duration_s, sample_rate, seed, style)


def _render(
    duration_s: float, sample_rate: int, seed: int, style: SpeechStyle
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = max(round(duration_s * sample_rate), sample_rate // 4)
    out = np.zeros(total, dtype=np.float64)
    active = np.zeros(total, dtype=bool)
    gaps: list[tuple[int, int]] = []

    cursor = 0
    index = 0
    while cursor < total:
        speech_len = style.speech_run_s * (
            1.0 + style.rhythm_jitter * float(rng.uniform(-0.55, 1.25))
        )
        n_speech = min(round(speech_len * sample_rate), total - cursor)
        if n_speech < sample_rate // 20:
            break
        segment = _voiced_segment(n_speech, sample_rate, style, rng, index)
        if index % 3 == 2:
            segment = _append_fricative(segment, sample_rate, rng)
        out[cursor : cursor + n_speech] += segment
        active[cursor : cursor + n_speech] = True
        cursor += n_speech

        pause_len = style.pause_run_s * (
            1.0 + style.rhythm_jitter * float(rng.uniform(-0.6, 1.9))
        )
        n_pause = min(round(pause_len * sample_rate), total - cursor)
        if n_pause <= 0:
            break
        gaps.append((cursor, cursor + n_pause))
        cursor += n_pause
        index += 1

    speech_rms = float(np.sqrt(np.mean(out[active] ** 2))) if np.any(active) else 1.0
    if speech_rms <= 0.0:
        speech_rms = 1.0
    out = out / speech_rms * 0.12  # speech RMS at about -18 dBFS

    if style.breaths:
        _insert_breaths(out, gaps, sample_rate, style, rng)

    out = out + _floor_noise(total, sample_rate, style, rng)
    return telephone_channel(out, sample_rate)


def _voiced_segment(
    n_samples: int,
    sample_rate: int,
    style: SpeechStyle,
    rng: np.random.Generator,
    index: int,
) -> np.ndarray:
    """A harmonic stack through a moving formant filter, with an envelope."""

    time = np.arange(n_samples, dtype=np.float64) / sample_rate

    # Pitch contour: one slow arch plus, for a live talker, a random walk.
    arch = np.sin(np.pi * np.linspace(0.0, 1.0, n_samples)) - 0.5
    walk = (
        _smooth_noise(n_samples, sample_rate, 1.5, rng)
        if style.contour_semitones
        else 0.0
    )
    semitones = style.contour_semitones * (0.6 * arch + 0.4 * np.asarray(walk))
    f0 = style.f0_hz * 2.0 ** (semitones / 12.0)

    # Jitter is held for one glottal period at a time, which is what makes it
    # period-to-period perturbation rather than a slow vibrato.
    if style.jitter_pct > 0.0:
        hold = max(1, round(sample_rate / style.f0_hz))
        blocks = int(np.ceil(n_samples / hold))
        values = rng.normal(0.0, 1.0, blocks)
        jitter = np.repeat(values, hold)[:n_samples]
        f0 = f0 * (1.0 + jitter * style.jitter_pct / 100.0 * 1.4)

    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    source = np.zeros(n_samples, dtype=np.float64)
    n_harmonics = max(1, int(min(40, (sample_rate / 2.2) / style.f0_hz)))
    for harmonic in range(1, n_harmonics + 1):
        amplitude = 1.0 / harmonic
        offset = float(rng.uniform(0.0, 2.0 * np.pi))
        if style.phase_drift_rad > 0.0:
            drift = style.phase_drift_rad * _smooth_noise(
                n_samples, sample_rate, 7.0, rng
            )
        else:
            drift = 0.0
        source += amplitude * np.sin(harmonic * phase + offset + drift)

    if style.aspiration > 0.0:
        source = source + style.aspiration * rng.normal(0.0, 1.0, n_samples) * np.max(
            np.abs(source)
        )

    # Shimmer, also held for one period.
    if style.shimmer_db > 0.0:
        hold = max(1, round(sample_rate / style.f0_hz))
        blocks = int(np.ceil(n_samples / hold))
        gains_db = rng.normal(0.0, style.shimmer_db, blocks)
        gains = np.repeat(10.0 ** (gains_db / 20.0), hold)[:n_samples]
        source = source * gains

    source = _formant_filter(source, sample_rate, style, rng, index)

    # A syllable envelope with soft edges, so offsets are abrupt but not a
    # discontinuity.
    ramp_n = max(2, int(0.012 * sample_rate))
    envelope = np.ones(n_samples)
    envelope[:ramp_n] = np.linspace(0.0, 1.0, ramp_n)
    envelope[-ramp_n:] = np.linspace(1.0, 0.0, ramp_n)
    envelope = envelope * (0.75 + 0.25 * np.sin(2.0 * np.pi * 2.3 * time + index))
    return source * envelope


def _formant_filter(
    signal: np.ndarray,
    sample_rate: int,
    style: SpeechStyle,
    rng: np.random.Generator,
    index: int,
) -> np.ndarray:
    """Overlap-add resonator filtering with per-block formant frequencies.

    Zero-phase on purpose: the harmonic phase structure set up by the caller
    is what the phase cue measures, so the vocal-tract stage must not be the
    thing that scrambles it.
    """

    win = max(64, round(0.032 * sample_rate))
    win += win % 2
    hop = win // 2
    if signal.size < win:
        signal = np.pad(signal, (0, win - signal.size))
    window = np.hanning(win)
    freqs = np.fft.rfftfreq(win, d=1.0 / sample_rate)

    n_blocks = 1 + (signal.size - win) // hop
    out = np.zeros(signal.size + win, dtype=np.float64)
    base_f1 = 520.0 + 120.0 * np.sin(index)
    base_f2 = 1450.0 + 260.0 * np.cos(index)
    swing = style.formant_motion
    for block in range(n_blocks):
        start = block * hop
        position = block / max(n_blocks - 1, 1)
        f1 = base_f1 * (1.0 + swing * 0.45 * np.sin(2.0 * np.pi * position))
        f2 = base_f2 * (1.0 + swing * 0.30 * np.cos(2.0 * np.pi * position + 1.0))
        response = (
            _resonator(freqs, f1, 9.0)
            * _resonator(freqs, f2, 11.0)
            * _resonator(freqs, 2650.0, 13.0)
        )
        spectrum = np.fft.rfft(signal[start : start + win] * window)
        out[start : start + win] += np.fft.irfft(spectrum * response, win)
    del rng
    return out[: signal.size]


def _resonator(freqs: np.ndarray, centre: float, quality: float) -> np.ndarray:
    """Magnitude response of a second-order resonator."""

    ratio = freqs / max(centre, 1.0)
    denominator = np.sqrt((1.0 - ratio**2) ** 2 + (ratio / quality) ** 2)
    return 1.0 / np.maximum(denominator, 1e-6)


def _append_fricative(
    segment: np.ndarray, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """Tack a band-limited noise burst onto the end of a syllable."""

    n_noise = max(8, round(0.07 * sample_rate))
    if segment.size <= n_noise:
        return segment
    noise = rng.normal(0.0, 1.0, n_noise)
    noise = _band_shape(
        noise, sample_rate, 1800.0, min(3400.0, sample_rate / 2.2), order=3
    )
    peak = float(np.max(np.abs(noise))) or 1.0
    level = float(np.max(np.abs(segment))) * 0.35
    out = segment.copy()
    taper = np.linspace(1.0, 0.0, n_noise)
    out[-n_noise:] = out[-n_noise:] * taper + noise / peak * level
    return out


def _insert_breaths(
    out: np.ndarray,
    gaps: list[tuple[int, int]],
    sample_rate: int,
    style: SpeechStyle,
    rng: np.random.Generator,
) -> None:
    """Put a high-tilted noise burst in roughly every other gap."""

    speech_rms = 0.12
    level = speech_rms * 10.0 ** (style.breath_rel_db / 20.0)
    for position, (start, stop) in enumerate(gaps):
        if position % 2 == 1:
            continue
        n_breath = min(
            round(float(rng.uniform(0.14, 0.26)) * sample_rate), stop - start
        )
        if n_breath < int(0.08 * sample_rate):
            continue
        noise = rng.normal(0.0, 1.0, n_breath)
        noise = _band_shape(
            noise, sample_rate, 500.0, min(3400.0, sample_rate / 2.2), order=2
        )
        # Rising tilt: breath is high-frequency dominant, which is what
        # separates it from a quiet voiced tail.
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n_breath, d=1.0 / sample_rate)
        spectrum = spectrum * (1.0 + 2.5 * freqs / max(freqs[-1], 1.0))
        noise = np.fft.irfft(spectrum, n_breath)
        peak = float(np.max(np.abs(noise))) or 1.0
        shape = np.hanning(n_breath)
        offset = start + (stop - start - n_breath) // 2
        out[offset : offset + n_breath] += noise / peak * level * 2.0 * shape


def _floor_noise(
    total: int, sample_rate: int, style: SpeechStyle, rng: np.random.Generator
) -> np.ndarray:
    """A background bed at `floor_rel_db` below the speech RMS, with drift."""

    speech_rms = 0.12
    level = speech_rms * 10.0 ** (style.floor_rel_db / 20.0)
    noise = rng.normal(0.0, 1.0, total)
    if style.floor_drift_db > 0.0:
        drift_db = style.floor_drift_db * _smooth_noise(total, sample_rate, 0.35, rng)
        noise = noise * 10.0 ** (drift_db / 20.0)
    return noise * level


def _smooth_noise(
    n_samples: int, sample_rate: int, cutoff_hz: float, rng: np.random.Generator
) -> np.ndarray:
    """Unit-ish amplitude noise band-limited to `cutoff_hz`."""

    if n_samples < 4:
        return np.zeros(n_samples)
    noise = rng.normal(0.0, 1.0, n_samples)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spectrum = spectrum / (1.0 + (freqs / max(cutoff_hz, 0.01)) ** 4)
    smoothed = np.fft.irfft(spectrum, n_samples)
    peak = float(np.max(np.abs(smoothed)))
    if peak <= 0.0:
        return np.zeros(n_samples)
    return smoothed / peak


# ---------------------------------------------------------------------------
# Channels and degradations
# ---------------------------------------------------------------------------


def telephone_channel(
    signal: np.ndarray, sample_rate: int, band: tuple[float, float] = TELEPHONE_BAND
) -> np.ndarray:
    """300-3400 Hz band, the shape every genuine call already has."""

    top = min(band[1], sample_rate / 2.05)
    return _band_shape(signal, sample_rate, band[0], top, order=4)


def room_impulse_response(
    sample_rate: int = 8000,
    seed: int = 0,
    rt60_s: float = 0.55,
    reflections: tuple[tuple[float, float], ...] = (
        (0.00215, 0.60),
        (0.00340, 0.45),
        (0.00610, 0.30),
    ),
    tail_gain: float = 0.12,
) -> np.ndarray:
    """Direct path, a few strong early reflections, then a diffuse tail.

    The early reflection delays are chosen to sit away from the glottal
    period of an adult voice (4.5-10 ms), so a test that fires the comb cue
    is not just rediscovering the talker's pitch.
    """

    rng = np.random.default_rng(seed)
    length = max(round(rt60_s * 1.2 * sample_rate), 16)
    response = np.zeros(length, dtype=np.float64)
    response[0] = 1.0
    for delay_s, gain in reflections:
        index = round(delay_s * sample_rate)
        if 0 < index < length:
            response[index] += gain
    tail_start = min(round(0.008 * sample_rate), length - 1)
    time = np.arange(length - tail_start, dtype=np.float64) / sample_rate
    decay = 10.0 ** (-3.0 * time / max(rt60_s, 1e-3))
    response[tail_start:] += rng.normal(0.0, 1.0, time.size) * decay * tail_gain
    return response


def replay_of(
    signal: np.ndarray,
    sample_rate: int = 8000,
    seed: int = 0,
    *,
    rt60_s: float = 0.55,
    speaker_band: tuple[float, float] = (260.0, 3000.0),
    requantise_bits: int = 10,
    noise_rel_db: float = -46.0,
    post_channel: bool = True,
) -> np.ndarray:
    """Play `signal` out of a small loudspeaker into a phone.

    The chain is room, loudspeaker response, mild speaker distortion, capture
    noise, requantisation, and then (with `post_channel`) the telephone
    network again.

    `post_channel=True` is the realistic case and is the default. It also
    destroys the sample lattice, which is exactly why the requantisation cue
    reports "no evidence" rather than "clean" on the realistic fixture.
    Pass `post_channel=False` for the uncompressed-PCM case where that cue
    can actually fire.
    """

    rng = np.random.default_rng(seed + 977)
    data = np.asarray(signal, dtype=np.float64).ravel()
    if data.size == 0:
        return data

    response = room_impulse_response(sample_rate, seed=seed, rt60_s=rt60_s)
    played = _convolve(data, response)[: data.size]
    played = _band_shape(
        played,
        sample_rate,
        speaker_band[0],
        min(speaker_band[1], sample_rate / 2.05),
        order=4,
    )

    peak = float(np.max(np.abs(played))) or 1.0
    played = played / peak * 0.7
    played = np.tanh(played * 1.6) / 1.6  # loudspeaker compression

    played = played + rng.normal(0.0, 1.0, played.size) * 10.0 ** (noise_rel_db / 20.0)

    peak = float(np.max(np.abs(played))) or 1.0
    played = played / peak * 0.5
    if requantise_bits > 0:
        step = 0.5 / float(2 ** (requantise_bits - 1))
        played = np.round(played / step) * step

    if post_channel:
        played = telephone_channel(played, sample_rate)
    return played


def _band_shape(
    signal: np.ndarray, sample_rate: int, lo: float, hi: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth-magnitude band shaping in the FFT domain."""

    data = np.asarray(signal, dtype=np.float64).ravel()
    if data.size < 8:
        return data
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(data.size, d=1.0 / sample_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        high_pass = 1.0 / np.sqrt(
            1.0 + (max(lo, 0.0) / np.maximum(freqs, 1e-6)) ** (2 * order)
        )
        low_pass = 1.0 / np.sqrt(1.0 + (freqs / max(hi, 1.0)) ** (2 * order))
    shaped = np.nan_to_num(high_pass * low_pass)
    return np.fft.irfft(spectrum * shaped, data.size)


def _convolve(signal: np.ndarray, response: np.ndarray) -> np.ndarray:
    length = signal.size + response.size - 1
    size = int(2 ** int(np.ceil(np.log2(max(length, 2)))))
    product = np.fft.rfft(signal, size) * np.fft.rfft(response, size)
    return np.fft.irfft(product, size)[:length]


# ---------------------------------------------------------------------------
# Degenerate inputs, for robustness tests
# ---------------------------------------------------------------------------


def silence(duration_s: float = 3.0, sample_rate: int = 8000) -> np.ndarray:
    return np.zeros(max(round(duration_s * sample_rate), 1), dtype=np.float64)


def white_noise(
    duration_s: float = 3.0, sample_rate: int = 8000, seed: int = 0, level: float = 0.1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_samples = max(round(duration_s * sample_rate), 1)
    return rng.normal(0.0, level, n_samples)


def pure_tone(
    duration_s: float = 3.0,
    sample_rate: int = 8000,
    frequency: float = 440.0,
    amplitude: float = 0.4,
) -> np.ndarray:
    n_samples = max(round(duration_s * sample_rate), 1)
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * frequency * time)


def impulse_train(
    duration_s: float = 3.0, sample_rate: int = 8000, rate_hz: float = 4.0
) -> np.ndarray:
    n_samples = max(round(duration_s * sample_rate), 1)
    out = np.zeros(n_samples, dtype=np.float64)
    stride = max(1, round(sample_rate / max(rate_hz, 0.1)))
    out[::stride] = 1.0
    return out


def clipped(signal: np.ndarray, limit: float = 0.25) -> np.ndarray:
    return np.clip(np.asarray(signal, dtype=np.float64), -limit, limit)


def dc_offset(signal: np.ndarray, offset: float = 0.3) -> np.ndarray:
    return np.asarray(signal, dtype=np.float64) + offset
