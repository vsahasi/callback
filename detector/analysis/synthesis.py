"""Synthetic and vocoded speech detection from classical cues.

A neural countermeasure trained on ASVspoof would do this better. With no
corpus available, these are the classical measurements that still carry real
information about whether a waveform was produced by a larynx or by a model:

* Micro-prosody. A human larynx cannot hold a pitch steady; the period
  wobbles by a fraction of a percent from one cycle to the next (jitter) and
  so does the amplitude (shimmer). Concatenative and many neural vocoders
  reconstruct a smooth F0 contour and leave that wobble out.
* Phase. A vocoder that reconstructs a waveform from a magnitude spectrogram,
  or that drives an excitation from a smooth pulse train, produces
  unnaturally predictable phase: the instantaneous frequency of each harmonic
  barely moves between frames. Natural phonation does not behave that way.
  Measured against the synthetic corpus this cue turns out to be a
  stationarity measure rather than a phase measure: no single generator knob
  moves it much, and it only reaches its synthetic value once the pitch
  contour, the formants and the micro-variation are frozen together. It
  separates well, but not for the one reason its name suggests.
* Spectral variance. Over-smoothing is the classic generative artefact. A
  model trained with an L1 or L2 spectral loss regresses towards the
  conditional mean, which narrows the frame-to-frame spread of the log
  spectrum.
* Modulation spectrum. Natural speech has a broad 2-8 Hz syllable hump. A
  template-driven or metronomic talker concentrates that energy into a narrow
  spike.
* The noise floor between words. A live handset always has comfort noise, and
  that noise drifts. Synthesised audio often has a floor that is both too
  quiet and too constant.

Honest limits:

* Modern neural TTS deliberately models jitter, shimmer and phase. A 2024-era
  system with a diffusion or flow vocoder can beat every cue here. A high
  score from this detector therefore does not mean "definitely human"; it
  means "no classical synthesis artefact was found", and the `detail` string
  says so.
* All six cues need voiced speech. On under about half a second of voiced
  audio the verdict is `inconclusive` whatever the score.
* The cues are not independent, and the calibration harness shows by how
  much. Freezing the pitch contour alone moves the phase cue, the spectral
  variance cue and the pitch-range cue over in `liveness`; so does freezing
  the formants. Several cues firing here is therefore not several
  independent confirmations, which is why `blend` is given a lower
  `sharpness` here than the replay detector gets.
* The pitch-perturbation cue has a narrow working range. On the synthetic
  corpus a messy talker with the jitter switched off still measures about
  0.18%, which is this estimator's own noise, against about 0.37% with
  jitter on. The amplitude-perturbation cue is worse, about 1.5x. Both are
  weighted below the phase, rhythm and noise-floor cues for that reason, and
  neither is decisive on its own.
"""

from __future__ import annotations

import numpy as np

from detector.analysis.cues import (
    Cue,
    DetectorResult,
    blend,
    inconclusive,
    ramp,
    summarise,
    temper,
    verdict_for,
)
from detector.analysis.dsp import (
    EPS,
    AudioContext,
    amplitude_perturbation_db,
    circular_std,
    envelope_periodicity,
    f0_perturbation_pct,
    interior_gap_mask,
    modulation_spectrum,
    true_runs,
    wrap_phase,
)

__all__ = ["analyse_synthesis"]

MIN_VOICED_SECONDS = 0.45

# Independent cues compound; see cues.blend.
SHARPNESS = 2.6

# The pitch-perturbation ramp, in percent. These two numbers are the least
# trustworthy constants in the package. They come from synthetic audio, and
# the measurement is frame-averaged rather than period-synchronous, so the
# scale is compressed well below published clinical jitter figures: the
# generator in `synthetic` produces 1.2% period-to-period jitter and this
# measurement reads it as about 0.24%. A genuine caller with a steady voice
# could plausibly land near the guilty end. Re-derive both from real audio
# before anyone relies on a fail from this cue alone.
JITTER_INNOCENT_PCT = 0.22
JITTER_GUILTY_PCT = 0.05


def analyse_synthesis(ctx: AudioContext) -> DetectorResult:
    """Score how consistent the recording is with natural phonation (1.0) or
    with a synthesiser (0.0)."""

    block = ctx.block
    if not block.usable:
        reason = "; ".join(ctx.notes) or "no usable audio"
        return inconclusive(f"Synthesis analysis not possible: {reason}.")

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    for builder in (
        _jitter_cue,
        _shimmer_cue,
        _phase_cue,
        _spectral_variance_cue,
        _modulation_cue,
        _noise_floor_cue,
    ):
        cue, measures = builder(ctx)
        cues.append(cue)
        measurements.update(measures)

    raw_score = blend(cues, sharpness=SHARPNESS)
    measurable = sum(1 for cue in cues if cue.measurable)
    agreeing = sum(1 for cue in cues if cue.measurable and cue.penalty >= 0.5)

    confident = (
        ctx.pitch.voiced_seconds >= MIN_VOICED_SECONDS
        and ctx.activity.has_structure
        and measurable >= 4
        and block.sample_rate >= 8000
    )
    score = temper(raw_score, confident=confident)

    measurements["cues_measurable"] = float(measurable)
    measurements["cues_agreeing"] = float(agreeing)
    measurements["voiced_seconds"] = ctx.pitch.voiced_seconds
    measurements["score_before_tempering"] = raw_score

    detail = summarise("Synthesis cues:", cues)
    if not confident:
        detail = (
            f"{detail} Low confidence: only {ctx.pitch.voiced_seconds:.2f}s voiced and "
            f"{measurable} of {len(cues)} cues measurable."
        )
    elif score >= 0.62:
        detail = f"{detail} No classical artefact found, which a modern neural vocoder can also achieve."

    return DetectorResult(
        score=score,
        verdict=verdict_for(score, confident=confident),
        detail=detail[:500],
        measurements=measurements,
        cues=tuple(cues),
    )


# ---------------------------------------------------------------------------
# Micro-prosody
# ---------------------------------------------------------------------------


def _jitter_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Frame-to-frame F0 period perturbation."""

    if ctx.pitch.voiced_seconds < 0.25:
        note = "not enough voiced speech to measure pitch perturbation"
        return _unmeasurable("f0_perturbation", 0.20, note), {}

    value = f0_perturbation_pct(ctx.pitch)
    measures = {"f0_perturbation_pct": value, "median_f0_hz": ctx.pitch.median_f0}
    if value <= 0.0:
        note = "no voiced run long enough to measure pitch perturbation"
        return _unmeasurable("f0_perturbation", 0.20, note), measures

    penalty = ramp(value, JITTER_INNOCENT_PCT, JITTER_GUILTY_PCT)
    if penalty >= 0.25:
        note = (
            f"pitch period perturbation only {value:.3f}% (no larynx holds a note that "
            "steady)"
        )
    else:
        note = f"pitch period perturbation {value:.3f}%, in the human range"

    return (
        Cue(
            name="f0_perturbation", value=value, penalty=penalty, weight=0.20, note=note
        ),
        measures,
    )


def _shimmer_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Frame-to-frame amplitude perturbation over voiced runs."""

    value = amplitude_perturbation_db(ctx.spec, ctx.pitch)
    measures = {"amplitude_perturbation_db": value}
    if value <= 0.0:
        note = "no voiced run long enough to measure amplitude perturbation"
        return _unmeasurable("amplitude_perturbation", 0.10, note), measures

    penalty = ramp(value, 0.23, 0.10)
    if penalty >= 0.25:
        note = f"amplitude perturbation only {value:.2f} dB between frames, unnaturally even"
        return (
            Cue(
                name="amplitude_perturbation",
                value=value,
                penalty=penalty,
                weight=0.10,
                note=note,
            ),
            measures,
        )

    note = f"amplitude perturbation {value:.2f} dB between frames"
    return (
        Cue(
            name="amplitude_perturbation",
            value=value,
            penalty=penalty,
            weight=0.10,
            note=note,
        ),
        measures,
    )


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def _phase_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Dispersion of the instantaneous-frequency change on harmonic bins.

    The expected phase advance for bin k over one hop is removed first, so
    what remains is the deviation from a steady sinusoid. Taking a second
    difference across frames gives the frame-to-frame change in
    instantaneous frequency. In natural phonation that quantity jitters; in a
    waveform reconstructed from a smooth spectrogram it barely moves.

    What it measures in practice is broader than phase. Zeroing the
    per-harmonic phase wander in the synthetic generator barely moves this
    number (1.26 to 1.34 rad); the drop to 0.38 rad needs the pitch contour,
    the formant motion and the micro-variation frozen as well. Treat it as a
    stationarity measure read off the phase, and do not count it as evidence
    independent of the spectral-variance cue.
    """

    spec = ctx.spec
    phase = np.angle(spec.spectrum)
    if phase.shape[0] < 10 or phase.shape[1] < 8:
        return _unmeasurable("phase_steadiness", 0.26, "too few frames for phase"), {}

    bins = np.arange(phase.shape[1], dtype=np.float64)
    hop_samples = spec.hop_s * spec.sample_rate
    expected = 2.0 * np.pi * bins * hop_samples / spec.n_fft

    pooled: list[np.ndarray] = []
    for start, stop in true_runs(np.asarray(ctx.activity.speech, dtype=bool)):
        if (stop - start) < 8:
            continue
        segment = phase[start:stop]
        first = wrap_phase(np.diff(segment, axis=0) - expected)
        pooled.append(wrap_phase(np.diff(first, axis=0)))
    if not pooled:
        note = "no speech run long enough (80 ms) to measure phase behaviour"
        return _unmeasurable("phase_steadiness", 0.26, note), {}

    deltas = np.concatenate(pooled, axis=0)
    if deltas.shape[0] < 6:
        return _unmeasurable("phase_steadiness", 0.26, "too few phase samples"), {}

    dispersion = np.asarray(circular_std(deltas, axis=0), dtype=np.float64)
    selected = _harmonic_bins(ctx)
    if int(np.count_nonzero(selected)) < 4:
        note = "no bins carry enough energy to measure phase behaviour"
        return _unmeasurable("phase_steadiness", 0.26, note), {}

    value = float(np.median(dispersion[selected]))
    measures = {
        "phase_dispersion_rad": value,
        "phase_bins": float(np.count_nonzero(selected)),
    }

    penalty = ramp(value, 0.90, 0.30)
    if penalty >= 0.25:
        note = (
            f"harmonic phase advances with only {value:.2f} rad of jitter, the signature "
            "of a vocoder reconstruction"
        )
    else:
        note = f"harmonic phase jitter {value:.2f} rad, natural"

    return (
        Cue(
            name="phase_steadiness",
            value=value,
            penalty=penalty,
            weight=0.26,
            note=note,
        ),
        measures,
    )


def _harmonic_bins(ctx: AudioContext) -> np.ndarray:
    """Local spectral maxima inside the speech band, i.e. harmonic peaks.

    Restricting to peaks rather than to "any loud bin" matters for the phase
    cue. The valleys between harmonics are noise-dominated, and the phase of
    a noise bin is uniformly random in every recording, synthetic or not, so
    including them buries the signal under a constant noise term.
    """

    freqs = ctx.spec.freqs
    power = ctx.spec.power
    speech = np.asarray(ctx.activity.speech, dtype=bool)
    selected_frames = power[speech] if np.any(speech) else power
    if selected_frames.size == 0 or freqs.size < 5:
        return np.zeros(freqs.size, dtype=bool)
    mean_power = np.mean(selected_frames, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_db = 10.0 * np.log10(np.maximum(mean_power, EPS))
    mean_db = np.nan_to_num(mean_db, neginf=-200.0)

    peaks = np.zeros(freqs.size, dtype=bool)
    peaks[1:-1] = (mean_db[1:-1] >= mean_db[:-2]) & (mean_db[1:-1] >= mean_db[2:])
    in_band = (freqs >= 200.0) & (freqs <= min(3400.0, ctx.nyquist))
    loud = mean_db >= (float(np.max(mean_db)) - 22.0)
    selected = peaks & in_band & loud
    if int(np.count_nonzero(selected)) >= 4:
        return selected
    # A spectrum with no resolvable harmonic structure (whispered, very
    # noisy, or a tone) falls back to loud in-band bins so the cue can still
    # be attempted; its confidence gate decides whether to believe it.
    return in_band & loud


# ---------------------------------------------------------------------------
# Spectral variance and rhythm
# ---------------------------------------------------------------------------


def _spectral_variance_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Spread over time of the *shape* of the log spectrum.

    Each frame's log spectrum is centred on its own mean before the variance
    over time is taken, so the syllable envelope rising and falling drops
    out. Without that normalisation this measures loudness dynamics, which
    every talker has, instead of formant movement, which is the thing an
    over-smoothed generative model loses.
    """

    power = ctx.spec.power
    speech = np.asarray(ctx.activity.speech, dtype=bool)
    frames_used = power[speech] if np.any(speech) else power
    if frames_used.shape[0] < 8:
        return _unmeasurable("spectral_variance", 0.14, "too few speech frames"), {}

    selected = _harmonic_bins(ctx)
    if int(np.count_nonzero(selected)) < 4:
        return _unmeasurable(
            "spectral_variance", 0.14, "no bins with enough energy"
        ), {}

    with np.errstate(divide="ignore", invalid="ignore"):
        logs = 10.0 * np.log10(np.maximum(frames_used[:, selected], EPS))
    logs = np.nan_to_num(logs, neginf=-200.0)
    shape = logs - logs.mean(axis=1, keepdims=True)

    value = float(np.median(np.std(shape, axis=0)))
    measures = {"spectral_variance_db": value}

    penalty = ramp(value, 8.5, 5.0)
    if penalty >= 0.25:
        note = (
            f"spectral shape moves by only {value:.1f} dB over time: the vocal tract is "
            "barely changing, as an over-smoothed model leaves it"
        )
    else:
        note = f"spectral shape moves by {value:.1f} dB over time"

    return (
        Cue(
            name="spectral_variance",
            value=value,
            penalty=penalty,
            weight=0.14,
            note=note,
        ),
        measures,
    )


def _modulation_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Is the speech/pause rhythm itself periodic?

    The first version of this cue used the spectral flatness of the 2-8 Hz
    modulation spectrum, on the theory that a metronomic talker concentrates
    energy in a spike. It measured the opposite of what was intended: a
    perfectly periodic rhythm produces a *comb* of modulation harmonics,
    which is flatter than the single broad low-frequency hump spontaneous
    speech produces. The flatness statistic ranked the metronome as more
    natural than the human.

    Autocorrelating the envelope directly asks the question the cue was
    always about: does the pattern repeat, and on what period. A rendered
    file alternating 450 ms of speech with 250 ms of pause repeats every
    700 ms and scores high. Spontaneous speech repeats nothing.
    """

    envelope = np.asarray(ctx.spec.rms, dtype=np.float64)
    duration = envelope.size * ctx.spec.hop_s
    if duration < 1.8:
        note = f"only {duration:.1f}s of audio: too short for a rhythm to repeat in"
        return _unmeasurable("modulation_regularity", 0.14, note), {}

    strength, period = envelope_periodicity(
        envelope, ctx.spec.hop_s, min_lag_s=0.25, max_lag_s=min(2.5, duration / 2.2)
    )
    freqs, magnitude = modulation_spectrum(envelope, ctx.spec.hop_s, f_max=14.0)
    measures = {"rhythm_periodicity": strength, "rhythm_period_s": period}
    if freqs.size:
        measures["modulation_peak_hz"] = float(freqs[int(np.argmax(magnitude))])
    if strength <= 0.0:
        note = "envelope too short to autocorrelate"
        return _unmeasurable("modulation_regularity", 0.14, note), measures

    penalty = ramp(strength, 0.52, 0.82)
    if penalty >= 0.25:
        note = (
            f"the speech/pause pattern repeats every {period * 1000:.0f} ms "
            f"(correlation {strength:.2f}), a rendered rhythm rather than speech"
        )
    else:
        note = f"speech/pause pattern does not repeat (best correlation {strength:.2f})"

    return (
        Cue(
            name="modulation_regularity",
            value=strength,
            penalty=penalty,
            weight=0.14,
            note=note,
        ),
        measures,
    )


def _noise_floor_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Level and steadiness of the floor between words.

    Two failure modes, both weighted into one cue: a floor far below
    anything a telephone line produces, and a floor that does not drift at
    all. Measured on the interior gaps only, because frames straddling a
    speech boundary swing by tens of dB and would be read as drift.
    """

    activity = ctx.activity
    speech = np.asarray(activity.speech, dtype=bool)
    gap = interior_gap_mask(speech, erode=2)
    if int(np.count_nonzero(gap)) < 6 or not activity.has_structure:
        note = "no gap between words long enough to inspect the noise floor"
        return _unmeasurable("noise_floor", 0.16, note), {}

    gap_db = activity.energy_db[gap]
    level = float(np.median(gap_db)) - activity.speech_db
    steadiness = float(np.std(gap_db))
    measures = {"noise_floor_rel_db": level, "noise_floor_std_db": steadiness}

    quiet_penalty = ramp(level, -52.0, -70.0)
    flat_penalty = ramp(steadiness, 1.6, 0.5)
    penalty = 0.5 * quiet_penalty + 0.5 * flat_penalty

    if penalty >= 0.25:
        note = (
            f"floor between words {abs(level):.0f} dB down and varying by only "
            f"{steadiness:.1f} dB, cleaner than a telephone line"
        )
    else:
        note = (
            f"floor between words {abs(level):.0f} dB down, varying {steadiness:.1f} dB"
        )

    return (
        Cue(name="noise_floor", value=level, penalty=penalty, weight=0.16, note=note),
        measures,
    )


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
