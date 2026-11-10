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

With no reference the factor is `not_run`.

The reference side is a `VoicePrint`, not a recording
----------------------------------------------------
The comparison reads exactly five things from the reference: whether the
audio was usable, its conditioning notes, its median pitch, how much voiced
speech it held, and its long-term average spectrum. `VoicePrint` is those
five things and nothing else, which is why the service can hold a reference
without holding a vendor's voice. See `detector.analysis.voiceprint`.

One consequence worth stating: the probe is reduced to a print too, so the
comparison is symmetric by construction rather than by care, and a change to
how either side is measured cannot apply to only one of them.
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
from detector.analysis.dsp import AudioContext, pearson
from detector.analysis.voiceprint import VoicePrint, voiceprint_from

__all__ = ["analyse_voice_match", "compare_prints"]

MIN_VOICED_SECONDS = 0.6

# Pitch register carries most of the weight because the long-term spectrum is
# confounded by the channel: a different handset moves it as much as a
# different talker does.
WEIGHT_PITCH = 0.7
WEIGHT_SHAPE = 0.3

# Independent cues compound; see cues.blend.
SHARPNESS = 2.6

NO_REFERENCE = (
    "No enrolled reference recording for the claimed identity, so no voice "
    "comparison was made. This factor carries no weight in the decision."
)


def analyse_voice_match(
    probe: AudioContext, reference: VoicePrint | None
) -> DetectorResult:
    """Compare the caller against an enrolled reference print.

    Returns `not_run` when there is no reference, which is the normal state
    of this service for any identity that has not been enrolled.
    """

    if reference is None:
        return not_run(NO_REFERENCE)

    return compare_prints(voiceprint_from(probe), reference)


def compare_prints(caller: VoicePrint, reference: VoicePrint) -> DetectorResult:
    """The comparison itself, print against print.

    Split out of `analyse_voice_match` so that the cross-leg check in
    `detector.legs` can score two prints that were accumulated over a call
    without ever holding either side's audio. Same cues, same weights, same
    thresholds: a caller is judged against the person who answered the
    callback exactly as they would be judged against an enrolment.
    """

    if not caller.usable or not reference.usable:
        return inconclusive(
            "Voice comparison not possible: "
            + ("; ".join(caller.notes + reference.notes) or "unusable audio")
            + "."
        )

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    pitch_cue, pitch_measures = _pitch_cue(caller, reference)
    cues.append(pitch_cue)
    measurements.update(pitch_measures)

    spectrum_cue, spectrum_measures = _spectrum_cue(caller, reference)
    cues.append(spectrum_cue)
    measurements.update(spectrum_measures)

    raw_score = blend(cues, sharpness=SHARPNESS)
    measurable = sum(1 for cue in cues if cue.measurable)
    confident = (
        measurable == 2
        and caller.voiced_seconds >= MIN_VOICED_SECONDS
        and reference.voiced_seconds >= MIN_VOICED_SECONDS
    )
    score = temper(raw_score, confident=confident)
    measurements["cues_measurable"] = float(measurable)
    measurements["score_before_tempering"] = raw_score
    # The confidence gate `verdict_for` was given, recorded so a caller can
    # read it rather than re-derive it. `detector.legs` refuses a call only on
    # a FAIL that was confident here.
    measurements["confident"] = 1.0 if confident else 0.0

    detail = summarise("Voice comparison:", cues)
    if not confident:
        detail = (
            f"{detail} Low confidence: {caller.voiced_seconds:.1f}s voiced in the "
            f"call and {reference.voiced_seconds:.1f}s in the reference."
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
    caller: VoicePrint, reference: VoicePrint
) -> tuple[Cue, dict[str, float]]:
    """Median F0 distance in semitones.

    Semitones rather than Hz, because a 30 Hz gap means something different
    at 100 Hz than at 220 Hz.
    """

    if not caller.has_pitch or not reference.has_pitch:
        note = "no voiced speech on one side, so pitch register could not be compared"
        return _unmeasurable("pitch_register", WEIGHT_PITCH, note), {}

    probe_f0 = caller.median_f0_hz
    reference_f0 = reference.median_f0_hz
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
    caller: VoicePrint, reference: VoicePrint
) -> tuple[Cue, dict[str, float]]:
    """Correlation of the two long-term average log spectra.

    Heavily confounded by the channel: a different handset or codec moves
    this number as much as a different talker does. Weighted below pitch for
    that reason.
    """

    if not caller.has_spectrum or not reference.has_spectrum:
        note = "bandwidth too narrow to compare spectral shape"
        return _unmeasurable("spectral_shape", WEIGHT_SHAPE, note), {}

    if not caller.shares_grid_with(reference):
        # Correlating these would line up bin 40 of one spectrum against a
        # different frequency in the other and report the result as a
        # similarity. Refusing is the only honest option; the pitch cue is
        # unaffected and still carries 0.7.
        note = (
            f"reference was analysed on a {reference.bin_hz:.1f} Hz frequency grid and "
            f"this call on a {caller.bin_hz:.1f} Hz one, so spectral shape is not "
            "comparable"
        )
        return _unmeasurable("spectral_shape", WEIGHT_SHAPE, note), {}

    probe_spectrum = caller.spectrum()
    reference_spectrum = reference.spectrum()
    size = min(probe_spectrum.size, reference_spectrum.size)
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


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
