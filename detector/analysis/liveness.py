"""Liveness: is there a person breathing into a handset right now?

Real speech is messy in ways that are hard to fake and easy to measure. A
live talker on a phone call produces audible breath intakes, speech runs and
pauses of wildly uneven length, a pitch contour that drifts, and a background
that moves because the room and the handset move. A recording played back, or
a synthesiser reading a script, produces none of that, or produces it on a
schedule.

The five cues here, in the order they matter:

* Breath events. Short noise-like bursts in the gaps, spectrally tilted
  towards the high end and well below the speech level. Their absence over a
  few seconds is weak evidence; their presence is good evidence of a live
  talker close to a microphone.
* Pause and run irregularity. The coefficient of variation of speech-run and
  gap durations. Spontaneous speech scatters; a script does not.
* Pitch micro-variation, as in `synthesis`.
* Pitch contour range, in semitones. A monotone delivery is characteristic of
  a reader or a synthesiser.
* Background drift. The level of the floor between words moves on a live
  line and does not on a rendered file.

Honest limits:

* This is the weakest module in the package on short audio. Three seconds of
  a genuine caller saying "yes, that is correct" contains perhaps one pause
  and no breath at all, and the correct answer there is `inconclusive`, which
  is what it returns. Below 2.5 seconds the breath and irregularity cues are
  marked unmeasurable outright rather than scored as absent.
* Two of the five cues (pitch perturbation and contour range) are also used
  by `synthesis`, so a synthesiser is penalised by both factors and the risk
  blend double-counts that evidence. Nothing here corrects for it; the
  weights in `engine` were chosen knowing it.
* "Liveness" here is acoustic only. It cannot tell a live human reading an
  attacker's script from a live human speaking for themselves. That is what
  the interactive challenge in the call flow is for.
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
    f0_perturbation_pct,
    interior_gap_mask,
    true_runs,
)

# Imported rather than restated: this module and `synthesis` read the same
# pitch-perturbation measurement, and two copies of the threshold would
# eventually disagree. The double counting that creates in the risk blend is
# documented in `engine`.
from detector.analysis.synthesis import JITTER_GUILTY_PCT, JITTER_INNOCENT_PCT

__all__ = ["analyse_liveness"]

# Below this there is not enough call to look for breath or rhythm in.
MIN_DURATION_FOR_RHYTHM_S = 2.5
BREATH_MIN_S = 0.06
BREATH_MAX_S = 0.45

# Independent cues compound; see cues.blend.
SHARPNESS = 2.2


def analyse_liveness(ctx: AudioContext) -> DetectorResult:
    """Score how consistent the recording is with a live talker (1.0) or with
    a rendered or replayed file (0.0)."""

    block = ctx.block
    if not block.usable:
        reason = "; ".join(ctx.notes) or "no usable audio"
        return inconclusive(f"Liveness analysis not possible: {reason}.")

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    for builder in (
        _breath_cue,
        _rhythm_cue,
        _micro_variation_cue,
        _contour_cue,
        _background_cue,
    ):
        cue, measures = builder(ctx)
        cues.append(cue)
        measurements.update(measures)

    raw_score = blend(cues, sharpness=SHARPNESS)
    measurable = sum(1 for cue in cues if cue.measurable)
    agreeing = sum(1 for cue in cues if cue.measurable and cue.penalty >= 0.5)

    confident = (
        block.duration_s >= MIN_DURATION_FOR_RHYTHM_S
        and ctx.activity.has_structure
        and ctx.pitch.voiced_seconds >= 0.5
        and measurable >= 4
        and block.sample_rate >= 8000
    )
    score = temper(raw_score, confident=confident)

    measurements["cues_measurable"] = float(measurable)
    measurements["cues_agreeing"] = float(agreeing)
    measurements["duration_s"] = block.duration_s
    measurements["score_before_tempering"] = raw_score

    detail = summarise("Liveness:", cues)
    if not confident:
        detail = (
            f"{detail} Low confidence: {block.duration_s:.1f}s of audio, "
            f"{ctx.pitch.voiced_seconds:.1f}s voiced, {measurable} of {len(cues)} cues "
            "measurable."
        )

    return DetectorResult(
        score=score,
        verdict=verdict_for(score, confident=confident),
        detail=detail[:500],
        measurements=measurements,
        cues=tuple(cues),
    )


# ---------------------------------------------------------------------------
# Breath
# ---------------------------------------------------------------------------


def _breath_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Rate of breath-like noise bursts, per second of recording.

    A breath is quiet, unvoiced, tilted towards the high end, and lasts
    between about 60 and 450 ms. Those four conditions together are
    restrictive enough that fricatives inside words and line clicks do not
    usually qualify.
    """

    if ctx.block.duration_s < MIN_DURATION_FOR_RHYTHM_S:
        note = (
            f"only {ctx.block.duration_s:.1f}s of audio: too short to conclude anything "
            "from the absence of breath"
        )
        return _unmeasurable("breath_events", 0.14, note), {}

    activity = ctx.activity
    spec = ctx.spec
    freqs = spec.freqs
    power = spec.power
    if power.shape[0] < 8:
        return _unmeasurable("breath_events", 0.18, "too few frames"), {}

    high = (freqs >= 1200.0) & (freqs <= min(3400.0, ctx.nyquist))
    low = (freqs >= 200.0) & (freqs < 1000.0)
    if int(np.count_nonzero(high)) < 4 or int(np.count_nonzero(low)) < 4:
        note = "bandwidth too narrow to separate breath from voice"
        return _unmeasurable("breath_events", 0.14, note), {}

    with np.errstate(divide="ignore", invalid="ignore"):
        tilt = 10.0 * np.log10(
            np.maximum(np.sum(power[:, high], axis=1), EPS)
            / np.maximum(np.sum(power[:, low], axis=1), EPS)
        )
    tilt = np.nan_to_num(tilt, neginf=-200.0, posinf=200.0)

    strength = _aligned(ctx.pitch.strength, power.shape[0])
    energy = activity.energy_db[: power.shape[0]]
    speech = _aligned(activity.speech, power.shape[0]).astype(bool)

    # Only the interior of a gap, eroded hard. The 32 ms analysis window
    # smears speech energy several frames past the end of a word, and those
    # smeared frames are quiet, unvoiced and high-tilted: indistinguishable
    # from a breath by every other test here. Counting them produced the
    # same event rate whether or not the recording contained any breath at
    # all, which is to say the cue was measuring nothing.
    interior = interior_gap_mask(speech, erode=4)
    candidate = (
        interior
        # 6 dB above the floor, not 4: a drifting line floor crosses 4 dB on
        # its own.
        & (energy > activity.floor_db + 6.0)
        & (energy < activity.speech_db - 8.0)
        & (tilt > -6.0)
        & (strength < 0.35)
    )

    min_frames = max(1, round(BREATH_MIN_S / max(spec.hop_s, 1e-6)))
    max_frames = max(min_frames, round(BREATH_MAX_S / max(spec.hop_s, 1e-6)))
    events = [
        (start, stop)
        for start, stop in true_runs(candidate)
        if min_frames <= (stop - start) <= max_frames
    ]
    rate = len(events) / max(ctx.block.duration_s, EPS)
    measures = {"breath_events": float(len(events)), "breath_rate_per_s": rate}

    # Absence of breath is only as damning as the recording is long. Three
    # seconds of "yes, that is correct" legitimately contains none, so the
    # penalty for finding none is scaled by duration and reaches full
    # strength only at six seconds.
    absence_weight = ramp(ctx.block.duration_s, MIN_DURATION_FOR_RHYTHM_S, 6.0)
    penalty = ramp(rate, 0.22, 0.0) * absence_weight
    measures["breath_absence_weight"] = absence_weight
    if penalty >= 0.25:
        note = (
            f"no breath events in {ctx.block.duration_s:.1f}s of call "
            f"({len(events)} candidates); a talker at a handset normally has one"
        )
    elif rate <= 0.0:
        note = (
            f"no breath events, but {ctx.block.duration_s:.1f}s is short enough that "
            "that proves little"
        )
    else:
        note = f"{len(events)} breath-like events ({rate:.2f}/s)"

    return (
        Cue(name="breath_events", value=rate, penalty=penalty, weight=0.14, note=note),
        measures,
    )


# ---------------------------------------------------------------------------
# Rhythm
# ---------------------------------------------------------------------------


def _rhythm_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Coefficient of variation of speech-run and gap durations.

    Uniform alternation is the fingerprint of a rendered file. Spontaneous
    speech has runs that differ by a factor of several.

    The first and last speech runs are excluded. Both are cut by where the
    recording happens to start and stop rather than by the talker, and one
    truncated run is enough to turn a metronomic delivery into an
    apparently irregular one: on the synthetic corpus a 5 s crop of
    perfectly uniform speech measured 16% variation against 2% for the same
    speech cropped at a segment boundary.
    """

    if ctx.block.duration_s < MIN_DURATION_FOR_RHYTHM_S:
        note = f"only {ctx.block.duration_s:.1f}s of audio: no rhythm to measure"
        return _unmeasurable("pause_irregularity", 0.26, note), {}

    activity = ctx.activity
    speech = np.asarray(activity.speech, dtype=bool)
    hop = max(activity.hop_s, 1e-6)
    speech_runs = [
        (stop - start) * hop
        for start, stop in true_runs(speech)
        if start > 0 and stop < speech.size
    ]
    gap_runs = [
        (stop - start) * hop
        for start, stop in true_runs(~speech)
        if start > 0 and stop < speech.size
    ]

    if len(speech_runs) < 3 or len(gap_runs) < 2:
        note = (
            f"{len(speech_runs)} speech runs and {len(gap_runs)} interior pauses: too few "
            "to judge rhythm"
        )
        return _unmeasurable("pause_irregularity", 0.26, note), {
            "speech_runs": float(len(speech_runs)),
            "interior_pauses": float(len(gap_runs)),
        }

    speech_cv = _coefficient_of_variation(speech_runs)
    gap_cv = _coefficient_of_variation(gap_runs)
    value = 0.5 * (speech_cv + gap_cv)
    measures = {
        "speech_run_cv": speech_cv,
        "pause_cv": gap_cv,
        "rhythm_cv": value,
        "speech_runs": float(len(speech_runs)),
        "interior_pauses": float(len(gap_runs)),
    }

    penalty = ramp(value, 0.14, 0.04)
    if penalty >= 0.25:
        note = (
            f"speech and pause lengths vary by only {value * 100:.0f}% "
            "(near-uniform alternation, not spontaneous speech)"
        )
    else:
        note = f"speech and pause lengths vary by {value * 100:.0f}%"

    return (
        Cue(
            name="pause_irregularity",
            value=value,
            penalty=penalty,
            weight=0.26,
            note=note,
        ),
        measures,
    )


def _coefficient_of_variation(values: list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size < 2:
        return 0.0
    mean = float(np.mean(data))
    if mean <= EPS:
        return 0.0
    return float(np.std(data) / mean)


# ---------------------------------------------------------------------------
# Pitch behaviour
# ---------------------------------------------------------------------------


def _micro_variation_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Pitch period perturbation, as a liveness rather than synthesis cue."""

    if ctx.pitch.voiced_seconds < 0.25:
        note = "not enough voiced speech to measure pitch micro-variation"
        return _unmeasurable("pitch_micro_variation", 0.18, note), {}

    value = f0_perturbation_pct(ctx.pitch)
    measures = {"f0_perturbation_pct": value}
    if value <= 0.0:
        note = "no voiced run long enough to measure pitch micro-variation"
        return _unmeasurable("pitch_micro_variation", 0.18, note), measures

    penalty = ramp(value, JITTER_INNOCENT_PCT, JITTER_GUILTY_PCT)
    if penalty >= 0.25:
        note = f"pitch held to within {value:.2f}%, steadier than a larynx manages"
    else:
        note = f"pitch wanders by {value:.2f}% between frames"

    return (
        Cue(
            name="pitch_micro_variation",
            value=value,
            penalty=penalty,
            weight=0.18,
            note=note,
        ),
        measures,
    )


def _contour_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """10-90 percentile F0 spread in semitones."""

    if ctx.pitch.voiced_frames < 12:
        note = "not enough voiced frames to measure a pitch contour"
        return _unmeasurable("pitch_contour", 0.20, note), {}

    value = ctx.pitch.semitone_spread()
    measures = {"f0_spread_semitones": value, "median_f0_hz": ctx.pitch.median_f0}
    penalty = ramp(value, 3.5, 0.8)
    if penalty >= 0.25:
        note = f"pitch contour spans only {value:.1f} semitones, near monotone"
    else:
        note = f"pitch contour spans {value:.1f} semitones"

    return (
        Cue(name="pitch_contour", value=value, penalty=penalty, weight=0.20, note=note),
        measures,
    )


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def _background_cue(ctx: AudioContext) -> tuple[Cue, dict[str, float]]:
    """Drift of the inter-word floor level.

    A live acoustic environment moves. A rendered file's floor is either
    digital silence or one constant noise bed.
    """

    activity = ctx.activity
    gap = interior_gap_mask(np.asarray(activity.speech, dtype=bool), erode=2)
    if int(np.count_nonzero(gap)) < 8 or not activity.has_structure:
        note = "no gap between words long enough to watch the background"
        return _unmeasurable("background_drift", 0.22, note), {}

    gap_db = activity.energy_db[gap]
    value = float(np.std(gap_db))
    measures = {"background_drift_db": value}

    penalty = ramp(value, 1.1, 0.35)
    if penalty >= 0.25:
        note = f"background level varies by only {value:.1f} dB, a rendered noise bed"
    else:
        note = f"background level varies by {value:.1f} dB, a live environment"

    return (
        Cue(
            name="background_drift",
            value=value,
            penalty=penalty,
            weight=0.22,
            note=note,
        ),
        measures,
    )


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Widen every True run by `radius` frames on each side."""

    flags = np.asarray(mask, dtype=bool)
    if radius <= 0 or flags.size == 0:
        return flags
    out = flags.copy()
    for start, stop in true_runs(flags):
        out[max(start - radius, 0) : min(stop + radius, flags.size)] = True
    return out


def _aligned(values: np.ndarray, size: int) -> np.ndarray:
    """Trim or edge-pad a per-frame array to `size` entries.

    The pitch tracker uses a longer window than the shared spectrogram, so
    its frame count can be one short. Aligning here keeps every per-frame
    comparison index-safe without each caller re-deriving the offset.
    """

    data = np.asarray(values)
    if data.size == size:
        return data
    if data.size > size:
        return data[:size]
    if data.size == 0:
        return np.zeros(size, dtype=data.dtype if data.dtype != object else np.float64)
    return np.concatenate([data, np.repeat(data[-1:], size - data.size)])


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
