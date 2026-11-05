"""Liveness.

The weakest module in the package on short audio, so most of these tests are
about it refusing to over-claim: no breath in three seconds has to be weak
evidence, not a denial.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.dsp import build_context  # noqa: E402
from detector.analysis.liveness import MIN_DURATION_FOR_RHYTHM_S, analyse_liveness  # noqa: E402
from detector.analysis.synthetic import (  # noqa: E402
    SpeechStyle,
    live_speech,
    pure_tone,
    silence,
    tts_speech,
    white_noise,
)
from shared.contract import FactorVerdict  # noqa: E402

RATE = 8000

# Measured margin on this corpus is about 0.85.
LIVENESS_MARGIN = 0.45
CLEAN_FLOOR = 0.70


def _result(signal: np.ndarray, rate: int = RATE):
    return analyse_liveness(build_context(signal, rate))


def _cue(signal: np.ndarray, name: str, rate: int = RATE):
    for cue in _result(signal, rate).cues:
        if cue.name == name:
            return cue
    raise AssertionError(f"no cue named {name}")


# ---------------------------------------------------------------------------
# The headline direction
# ---------------------------------------------------------------------------


def test_a_live_talker_scores_materially_above_a_rendered_file() -> None:
    live = _result(live_speech(5.0, RATE, seed=0))
    rendered = _result(tts_speech(5.0, RATE, seed=0))

    assert live.score >= CLEAN_FLOOR
    assert live.score - rendered.score >= LIVENESS_MARGIN
    assert live.verdict is FactorVerdict.PASS
    assert rendered.verdict is FactorVerdict.FAIL


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_direction_holds_across_utterances(seed: int) -> None:
    live = _result(live_speech(5.0, RATE, seed=seed)).score
    rendered = _result(tts_speech(5.0, RATE, seed=seed)).score
    assert live - rendered >= LIVENESS_MARGIN


# ---------------------------------------------------------------------------
# Each cue against the knob that should move it
# ---------------------------------------------------------------------------


def test_breath_bursts_are_found_when_they_are_there() -> None:
    # Honest about what this cue manages: it finds roughly twice as many
    # events with breath as without, not all of them and not only them. A
    # drifting line floor in a long gap still produces the occasional false
    # candidate, which is why the cue carries the smallest weight in the
    # module and why its failure direction is to miss rather than to accuse.
    with_breath = _cue(live_speech(5.0, RATE, seed=0), "breath_events")
    without = _cue(
        live_speech(5.0, RATE, seed=0, style=SpeechStyle(breaths=False)),
        "breath_events",
    )
    assert with_breath.value > without.value
    assert with_breath.penalty < 0.15
    # A rendered file with neither breath nor a drifting floor is the clean
    # negative case.
    assert _cue(tts_speech(5.0, RATE, seed=0), "breath_events").value == 0.0


def test_uniform_alternation_is_detected_as_non_spontaneous() -> None:
    uniform = SpeechStyle(rhythm_jitter=0.0)
    irregular_cue = _cue(live_speech(5.0, RATE, seed=0), "pause_irregularity")
    uniform_cue = _cue(
        live_speech(5.0, RATE, seed=0, style=uniform), "pause_irregularity"
    )

    # Speech and pause lengths vary by a quarter against under 2%.
    assert irregular_cue.value > 10.0 * uniform_cue.value
    assert uniform_cue.penalty > 0.9
    assert irregular_cue.penalty == 0.0


def test_a_monotone_delivery_is_detected() -> None:
    monotone = SpeechStyle(contour_semitones=1.0)
    moving_cue = _cue(live_speech(5.0, RATE, seed=0), "pitch_contour")
    flat_cue = _cue(live_speech(5.0, RATE, seed=0, style=monotone), "pitch_contour")

    assert moving_cue.value > flat_cue.value + 2.0
    assert flat_cue.penalty > 0.8
    assert moving_cue.penalty == 0.0


def test_a_rendered_noise_bed_is_detected() -> None:
    moving_cue = _cue(live_speech(5.0, RATE, seed=0), "background_drift")
    assert moving_cue.penalty == 0.0
    assert _cue(tts_speech(5.0, RATE, seed=0), "background_drift").penalty > 0.8


# ---------------------------------------------------------------------------
# Honesty: short audio must not produce a confident denial
# ---------------------------------------------------------------------------


def test_absence_of_breath_is_weak_evidence_on_a_short_call() -> None:
    short = _cue(tts_speech(2.8, RATE, seed=0), "breath_events")
    long = _cue(tts_speech(6.5, RATE, seed=0), "breath_events")

    # The same absence, weighted by how much call there was to find a breath
    # in. Three seconds of "yes, that is correct" legitimately contains none.
    assert short.value == 0.0
    assert long.value == 0.0
    assert short.penalty < 0.25
    assert long.penalty > 0.8
    assert "proves little" in short.note


def test_the_rhythm_cues_refuse_to_run_on_a_short_call() -> None:
    short = _result(live_speech(2.0, RATE, seed=0))
    assert short.verdict is FactorVerdict.INCONCLUSIVE
    for name in ("breath_events", "pause_irregularity"):
        cue = _cue(live_speech(2.0, RATE, seed=0), name)
        assert not cue.measurable
    assert f"{2.0:.1f}s" in short.detail or "Low confidence" in short.detail


def test_audio_with_no_pauses_at_all_is_not_judged_on_rhythm() -> None:
    # A continuous signal has no speech/pause pattern, and inventing a
    # coefficient of variation from one run would be guessing.
    cue = _cue(pure_tone(3.5, RATE), "pause_irregularity")
    assert not cue.measurable
    assert "too few" in cue.note or "no rhythm" in cue.note


@pytest.mark.parametrize(
    "signal",
    [
        silence(3.0, RATE),
        white_noise(3.0, RATE, seed=1),
        pure_tone(3.0, RATE),
        np.zeros(400),
        live_speech(4.0, RATE, seed=0)[:1600],
    ],
    ids=["silence", "noise", "tone", "tiny", "short"],
)
def test_unanalysable_audio_is_inconclusive_and_near_no_opinion(
    signal: np.ndarray,
) -> None:
    result = analyse_liveness(build_context(signal, RATE))
    assert result.verdict is FactorVerdict.INCONCLUSIVE
    assert np.isfinite(result.score)
    assert 0.2 < result.score < 0.8


def test_the_confidence_threshold_matches_the_documented_one() -> None:
    # The module says it needs 2.5 s before it will commit; the test holds it
    # to that rather than to a number buried in a branch.
    assert pytest.approx(2.5) == MIN_DURATION_FOR_RHYTHM_S
    assert (
        analyse_liveness(
            build_context(
                live_speech(MIN_DURATION_FOR_RHYTHM_S - 0.5, RATE, seed=0), RATE
            )
        ).verdict
        is FactorVerdict.INCONCLUSIVE
    )


def test_detail_stays_inside_the_contract_limit() -> None:
    for signal in (
        live_speech(5.0, RATE, seed=0),
        tts_speech(5.0, RATE, seed=0),
        silence(3.0, RATE),
    ):
        assert len(_result(signal).detail) <= 500
