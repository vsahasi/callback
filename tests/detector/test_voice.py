"""Voice match against an enrolled reference.

The module under test is explicit that it is not speaker verification, and
these tests hold it to exactly that claim: catch an obviously different
voice, decline to confirm identity, and do nothing at all without a
reference.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.dsp import build_context  # noqa: E402
from detector.analysis.synthetic import SpeechStyle, live_speech, silence  # noqa: E402
from detector.analysis.voice import analyse_voice_match  # noqa: E402
from detector.analysis.voiceprint import voiceprint_from  # noqa: E402
from shared.contract import FactorVerdict  # noqa: E402

RATE = 8000


def _compare(probe: np.ndarray, reference: np.ndarray | None):
    return analyse_voice_match(
        build_context(probe, RATE),
        voiceprint_from(build_context(reference, RATE))
        if reference is not None
        else None,
    )


def test_without_a_reference_the_factor_does_not_run() -> None:
    result = _compare(live_speech(4.0, RATE, seed=0), None)
    assert result.verdict is FactorVerdict.NOT_RUN
    assert "no voice comparison" in result.detail.lower()
    assert "carries no weight" in result.detail


def test_the_same_talker_matches() -> None:
    result = _compare(live_speech(4.0, RATE, seed=0), live_speech(4.0, RATE, seed=7))
    assert result.verdict is FactorVerdict.PASS
    assert result.score >= 0.8
    assert result.measurements["f0_distance_semitones"] < 2.0


def test_a_different_register_is_caught() -> None:
    impostor = live_speech(4.0, RATE, seed=0)
    enrolled = live_speech(4.0, RATE, seed=7, style=SpeechStyle(f0_hz=215.0))

    matched = _compare(impostor, live_speech(4.0, RATE, seed=7))
    mismatched = _compare(impostor, enrolled)

    assert mismatched.measurements["f0_distance_semitones"] > 5.0
    assert matched.score - mismatched.score >= 0.5
    assert mismatched.verdict is FactorVerdict.FAIL


def test_a_pass_refuses_to_claim_identity() -> None:
    detail = _compare(
        live_speech(4.0, RATE, seed=0), live_speech(4.0, RATE, seed=7)
    ).detail
    assert "does not confirm identity" in detail
    assert len(detail) <= 500


def test_a_silent_side_is_inconclusive_not_a_match() -> None:
    result = _compare(live_speech(4.0, RATE, seed=0), silence(3.0, RATE))
    assert result.verdict is FactorVerdict.INCONCLUSIVE
    assert np.isfinite(result.score)
    assert 0.2 < result.score < 0.8


def test_too_little_voiced_audio_is_never_confident() -> None:
    result = _compare(
        live_speech(4.0, RATE, seed=0)[:2400], live_speech(4.0, RATE, seed=7)
    )
    assert result.verdict is FactorVerdict.INCONCLUSIVE
    assert "Low confidence" in result.detail


def test_the_comparison_is_symmetric_in_pitch_distance() -> None:
    low = live_speech(4.0, RATE, seed=0)
    high = live_speech(4.0, RATE, seed=7, style=SpeechStyle(f0_hz=215.0))
    forward = _compare(low, high).measurements["f0_distance_semitones"]
    backward = _compare(high, low).measurements["f0_distance_semitones"]
    assert forward == pytest.approx(backward, abs=1.0)
