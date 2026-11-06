"""Synthetic and vocoded speech detection.

The corpus here is a deliberately naive synthesiser: fixed harmonic phases,
no jitter, no shimmer, static formants, a metronomic rhythm and a digital
silence floor. Separating that from the messy generator is a real test of
each cue's direction, and it is not evidence that the detector would separate
modern neural TTS. See the module docstrings in `detector.analysis.synthesis`
and `detector.analysis.synthetic` for what these margins do and do not mean.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.dsp import build_context  # noqa: E402
from detector.analysis.synthesis import analyse_synthesis  # noqa: E402
from detector.analysis.synthetic import (  # noqa: E402
    LIVE,
    TTS,
    SpeechStyle,
    live_speech,
    pure_tone,
    replay_of,
    silence,
    tts_speech,
    white_noise,
)
from shared.contract import FactorVerdict  # noqa: E402

RATE = 8000

# A naive synthesiser must cost at least this much score against the messy
# generator. The measured margin on this corpus is about 0.89.
SYNTHESIS_MARGIN = 0.50
CLEAN_FLOOR = 0.70


def _result(signal: np.ndarray, rate: int = RATE):
    return analyse_synthesis(build_context(signal, rate))


def _cue(signal: np.ndarray, name: str, rate: int = RATE):
    for cue in _result(signal, rate).cues:
        if cue.name == name:
            return cue
    raise AssertionError(f"no cue named {name}")


# ---------------------------------------------------------------------------
# The headline direction
# ---------------------------------------------------------------------------


def test_synthetic_speech_scores_materially_below_natural_speech() -> None:
    natural = _result(live_speech(4.0, RATE, seed=0))
    synthetic_voice = _result(tts_speech(4.0, RATE, seed=0))

    assert natural.score >= CLEAN_FLOOR
    assert natural.score - synthetic_voice.score >= SYNTHESIS_MARGIN
    assert natural.verdict is FactorVerdict.PASS
    assert synthetic_voice.verdict is FactorVerdict.FAIL


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_direction_holds_across_utterances(seed: int) -> None:
    natural = _result(live_speech(4.0, RATE, seed=seed)).score
    synthetic_voice = _result(tts_speech(4.0, RATE, seed=seed)).score
    assert natural - synthetic_voice >= SYNTHESIS_MARGIN


def test_a_replayed_human_is_not_called_synthetic() -> None:
    # The two attacks are different accusations. A recording of a real person
    # played out of a speaker is still a real person's voice, and saying
    # otherwise would put the wrong explanation in front of a reviewer.
    replayed = replay_of(live_speech(4.0, RATE, seed=0), RATE, seed=5)
    assert _result(replayed).score >= CLEAN_FLOOR
    assert _result(replayed).verdict is FactorVerdict.PASS


# ---------------------------------------------------------------------------
# One generator knob at a time
# ---------------------------------------------------------------------------


def test_removing_the_micro_prosody_moves_the_perturbation_cue() -> None:
    # Only a factor of two, not a factor of ten: with the jitter switched off
    # the measurement still reads about 0.18%, which is the estimator's own
    # noise on non-stationary audio. That narrow working range is the honest
    # limit of this cue and is documented in dsp.f0_perturbation_pct.
    flat = SpeechStyle(jitter_pct=0.0, shimmer_db=0.0)
    natural_cue = _cue(live_speech(4.0, RATE, seed=0), "f0_perturbation")
    flat_cue = _cue(live_speech(4.0, RATE, seed=0, style=flat), "f0_perturbation")

    assert natural_cue.value > 1.4 * flat_cue.value
    assert natural_cue.penalty == 0.0
    assert flat_cue.penalty > 0.15

    # Only when everything is frozen does the cue become decisive.
    assert _cue(tts_speech(4.0, RATE, seed=0), "f0_perturbation").penalty > 0.9


def test_the_phase_cue_measures_stationarity_not_phase_alone() -> None:
    # Zeroing the per-harmonic phase wander on its own barely moves this
    # number. It falls only once the contour, the formants and the
    # micro-variation are frozen together, which is what the docstring now
    # says it measures. Asserting the honest relationship rather than the
    # one the cue's name implies.
    natural = _cue(live_speech(4.0, RATE, seed=0), "phase_steadiness")
    drift_off = _cue(
        live_speech(4.0, RATE, seed=0, style=SpeechStyle(phase_drift_rad=0.0)),
        "phase_steadiness",
    )
    frozen = _cue(
        live_speech(
            4.0,
            RATE,
            seed=0,
            style=SpeechStyle(
                phase_drift_rad=0.0,
                jitter_pct=0.0,
                shimmer_db=0.0,
                aspiration=0.0,
                formant_motion=0.12,
                contour_semitones=1.5,
            ),
        ),
        "phase_steadiness",
    )

    assert drift_off.value == pytest.approx(natural.value, rel=0.3)
    assert natural.value > 2.5 * frozen.value
    assert natural.penalty == 0.0
    assert frozen.penalty > 0.6


def test_freezing_the_vocal_tract_moves_the_spectral_variance_cue() -> None:
    static = SpeechStyle(formant_motion=0.0)
    moving_cue = _cue(live_speech(4.0, RATE, seed=0), "spectral_variance")
    static_cue = _cue(live_speech(4.0, RATE, seed=0, style=static), "spectral_variance")
    assert moving_cue.value > static_cue.value + 1.0
    assert moving_cue.penalty < static_cue.penalty


def test_a_metronomic_rhythm_is_detected_as_a_repeating_pattern() -> None:
    metronome = SpeechStyle(rhythm_jitter=0.0, speech_run_s=0.45, pause_run_s=0.25)
    natural_cue = _cue(live_speech(5.0, RATE, seed=0), "modulation_regularity")
    fixed_cue = _cue(
        live_speech(5.0, RATE, seed=0, style=metronome), "modulation_regularity"
    )

    assert fixed_cue.value > natural_cue.value + 0.2
    assert fixed_cue.penalty > 0.5
    assert natural_cue.penalty < 0.3
    # The detail names the period it found, so a reviewer can sanity-check it.
    assert "repeats every" in fixed_cue.note


def test_a_digital_silence_floor_is_detected() -> None:
    quiet = SpeechStyle(floor_rel_db=-80.0, floor_drift_db=0.0)
    natural_cue = _cue(live_speech(4.0, RATE, seed=0), "noise_floor")
    quiet_cue = _cue(live_speech(4.0, RATE, seed=0, style=quiet), "noise_floor")

    assert quiet_cue.value < natural_cue.value - 20.0
    # Half of this cue is "too quiet" and half is "too steady". Dropping the
    # floor alone trips only the first half, because the breath bursts this
    # generator still puts in the gaps keep the level moving. Both halves
    # trip together on the full synthetic style.
    assert quiet_cue.penalty >= 0.45
    assert natural_cue.penalty < 0.2
    assert _cue(tts_speech(4.0, RATE, seed=0), "noise_floor").penalty > 0.9


def test_the_two_styles_differ_only_as_documented() -> None:
    # Guards the corpus: if LIVE and TTS stopped differing on micro-prosody
    # the tests above would pass for the wrong reason.
    assert LIVE.jitter_pct > 0.0
    assert TTS.jitter_pct == 0.0
    assert LIVE.phase_drift_rad > TTS.phase_drift_rad
    assert LIVE.rhythm_jitter > TTS.rhythm_jitter
    assert LIVE.floor_rel_db > TTS.floor_rel_db
    assert LIVE.breaths and not TTS.breaths


# ---------------------------------------------------------------------------
# Honesty
# ---------------------------------------------------------------------------


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
    result = analyse_synthesis(build_context(signal, RATE))
    assert result.verdict is FactorVerdict.INCONCLUSIVE
    assert np.isfinite(result.score)
    assert 0.2 < result.score < 0.8


def test_an_untuned_sample_rate_is_never_confident() -> None:
    result = analyse_synthesis(build_context(live_speech(4.0, 4000, seed=0), 4000))
    assert result.verdict is FactorVerdict.INCONCLUSIVE


def test_a_clean_pass_admits_what_it_cannot_rule_out() -> None:
    detail = _result(live_speech(4.0, RATE, seed=0)).detail
    assert "neural vocoder" in detail
    assert len(detail) <= 500


def test_a_fail_names_the_measurements_behind_it() -> None:
    detail = _result(tts_speech(4.0, RATE, seed=0)).detail
    assert "perturbation" in detail or "phase" in detail or "repeats" in detail
    assert "cues flagged" in detail
    assert len(detail) <= 500


def test_measurements_are_recorded_for_recalibration() -> None:
    measurements = _result(live_speech(4.0, RATE, seed=0)).measurements
    for key in (
        "f0_perturbation_pct",
        "phase_dispersion_rad",
        "spectral_variance_db",
        "rhythm_periodicity",
        "noise_floor_rel_db",
        "cues_measurable",
        "score_before_tempering",
    ):
        assert key in measurements
        assert np.isfinite(measurements[key])
