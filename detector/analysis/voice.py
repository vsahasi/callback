"""Voice match against an enrolled reference, with classical features only.

READ THIS BEFORE TRUSTING IT. This is not speaker verification. Speaker
verification needs a trained embedding; this module compares two recordings
on pitch register and long-term spectral shape, which is enough to notice
that the caller is obviously a different person and nothing more. Two adults
of the same sex with similar pitch will match here even if a listener could
tell them apart instantly.

It exists because `FactorName.VOICE_MATCH` carries the largest weight in the
risk blend and the alternative was to leave it `not_run` on every call,
which throws away the one case classical features do catch: the
`human_impostor` attack, where someone with a different voice reads the
attacker's script. For that case the measurements below are genuinely
informative. For a close match they are not, and the `detail` string says so
rather than implying a positive identification.

With no reference recording the factor is `not_run`.
"""

from __future__ import annotations

import numpy as np

from detector.analysis.cues import (
    Cue,
    DetectorResult,
    blend,
    inconclusive,
    not_run,
    ramp,
    summarise,
    temper,
    verdict_for,
)
from detector.analysis.dsp import EPS, AudioContext, pearson

__all__ = ["analyse_voice_match"]

MIN_VOICED_SECONDS = 0.6

# Pitch register carries most of the weight because the long-term spectrum is
# confounded by the channel: a different handset moves it as much as a
# different talker does.
WEIGHT_PITCH = 0.7
WEIGHT_SHAPE = 0.3

# Independent cues compound; see cues.blend.
SHARPNESS = 2.6


def analyse_voice_match(
    probe: AudioContext, reference: AudioContext | None
) -> DetectorResult:
    """Compare the caller against an enrolled reference.

    Returns `not_run` when there is no reference, which is the normal state
    of this service until an enrolment recording exists for the vendor.
    """

    if reference is None:
        return not_run(
            "No enrolled reference recording for the claimed identity, so no voice "
            "comparison was made. This factor carries no weight in the decision."
        )
    if not probe.block.usable or not reference.block.usable:
        return inconclusive(
            "Voice comparison not possible: "
            + ("; ".join(probe.notes + reference.notes) or "unusable audio")
            + "."
        )

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    pitch_cue, pitch_measures = _pitch_cue(probe, reference)
    cues.append(pitch_cue)
    measurements.update(pitch_measures)

    spectrum_cue, spectrum_measures = _spectrum_cue(probe, reference)
    cues.append(spectrum_cue)
    measurements.update(spectrum_measures)

    raw_score = blend(cues, sharpness=SHARPNESS)
    measurable = sum(1 for cue in cues if cue.measurable)
    confident = (
        measurable == 2
        and probe.pitch.voiced_seconds >= MIN_VOICED_SECONDS
        and reference.pitch.voiced_seconds >= MIN_VOICED_SECONDS
    )
    score = temper(raw_score, confident=confident)
    measurements["cues_measurable"] = float(measurable)
    measurements["score_before_tempering"] = raw_score

    detail = summarise("Voice comparison:", cues)
    if not confident:
        detail = (
            f"{detail} Low confidence: {probe.pitch.voiced_seconds:.1f}s voiced in the "
            f"call and {reference.pitch.voiced_seconds:.1f}s in the reference."
        )
    elif score >= 0.62:
        detail = (
            f"{detail} Classical features only: this rules out an obviously different "
            "voice, it does not confirm identity."
        )

    return DetectorResult(
        score=score,
        verdict=verdict_for(score, confident=confident),
        detail=detail[:500],
        measurements=measurements,
        cues=tuple(cues),
    )


def _pitch_cue(
    probe: AudioContext, reference: AudioContext
) -> tuple[Cue, dict[str, float]]:
    """Median F0 distance in semitones.

    Semitones rather than Hz, because a 30 Hz gap means something different
    at 100 Hz than at 220 Hz.
    """

    probe_f0 = probe.pitch.median_f0
    reference_f0 = reference.pitch.median_f0
    if probe_f0 <= 0.0 or reference_f0 <= 0.0:
        note = "no voiced speech on one side, so pitch register could not be compared"
        return _unmeasurable("pitch_register", WEIGHT_PITCH, note), {}

    distance = float(abs(12.0 * np.log2(probe_f0 / reference_f0)))
    measures = {
        "probe_f0_hz": probe_f0,
        "reference_f0_hz": reference_f0,
        "f0_distance_semitones": distance,
    }

    penalty = ramp(distance, 2.5, 7.5)
    if penalty >= 0.25:
        note = (
            f"caller's pitch register is {distance:.1f} semitones from the reference "
            f"({probe_f0:.0f} Hz against {reference_f0:.0f} Hz)"
        )
    else:
        note = (
            f"pitch register within {distance:.1f} semitones of the reference "
            f"({probe_f0:.0f} Hz against {reference_f0:.0f} Hz)"
        )

    return (
        Cue(
            name="pitch_register",
            value=distance,
            penalty=penalty,
            weight=WEIGHT_PITCH,
            note=note,
        ),
        measures,
    )


def _spectrum_cue(
    probe: AudioContext, reference: AudioContext
) -> tuple[Cue, dict[str, float]]:
    """Correlation of the two long-term average log spectra.

    Heavily confounded by the channel: a different handset or codec moves
    this number as much as a different talker does. Weighted below pitch for
    that reason.
    """

    probe_spectrum = _ltas(probe)
    reference_spectrum = _ltas(reference)
    if probe_spectrum is None or reference_spectrum is None:
        note = "bandwidth too narrow to compare spectral shape"
        return _unmeasurable("spectral_shape", WEIGHT_SHAPE, note), {}

    size = min(probe_spectrum.size, reference_spectrum.size)
    if size < 16:
        note = "too few shared frequency bins to compare spectral shape"
        return _unmeasurable("spectral_shape", WEIGHT_SHAPE, note), {}

    correlation = pearson(probe_spectrum[:size], reference_spectrum[:size])
    measures = {"ltas_correlation": correlation}

    penalty = ramp(correlation, 0.80, 0.25)
    if penalty >= 0.25:
        note = (
            f"long-term spectrum correlates only r={correlation:.2f} with the reference "
            "(may be a different handset rather than a different talker)"
        )
    else:
        note = f"long-term spectrum correlates r={correlation:.2f} with the reference"

    return (
        Cue(
            name="spectral_shape",
            value=correlation,
            penalty=penalty,
            weight=WEIGHT_SHAPE,
            note=note,
        ),
        measures,
    )


def _ltas(ctx: AudioContext) -> np.ndarray | None:
    """Mean log power spectrum over speech frames, 300 Hz to 3.4 kHz, in dB.

    Normalised to zero mean so overall level drops out; what remains is
    shape.
    """

    freqs = ctx.spec.freqs
    band = (freqs >= 300.0) & (freqs <= min(3400.0, ctx.nyquist))
    if int(np.count_nonzero(band)) < 16:
        return None
    spectrum = ctx.mean_log_spectrum(ctx.activity.speech)[band] * 10.0
    spread = float(np.std(spectrum))
    if spread <= EPS:
        return None
    return spectrum - float(np.mean(spectrum))


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
