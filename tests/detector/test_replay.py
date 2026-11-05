"""Replay detection.

Every assertion here is about direction and margin, never about an absolute
score. The margins are stated as named constants so that the intent survives
a re-tuning: if a future threshold change breaks one of these, the test says
which physical relationship stopped holding rather than which number moved.

The corpus is `detector.analysis.synthetic`, where each artefact is present
or absent by construction. Read its module docstring before trusting a margin
here as evidence about real calls.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.dsp import build_context  # noqa: E402
from detector.analysis.replay import analyse_replay  # noqa: E402
from detector.analysis.synthetic import (  # noqa: E402
    SpeechStyle,
    clipped,
    live_speech,
    replay_of,
    room_impulse_response,
    silence,
    telephone_channel,
    tts_speech,
    white_noise,
)
from shared.contract import FactorVerdict  # noqa: E402

RATE = 8000

# A loudspeaker in the path must cost at least this much score. Measured
# margin on this corpus is about 0.95; the assertion is deliberately slack.
REPLAY_MARGIN = 0.55

# Clean phone audio must not be accused.
CLEAN_FLOOR = 0.70


def _score(signal: np.ndarray, rate: int = RATE) -> float:
    return analyse_replay(build_context(signal, rate)).score


def _cue(signal: np.ndarray, name: str, rate: int = RATE):
    result = analyse_replay(build_context(signal, rate))
    for cue in result.cues:
        if cue.name == name:
            return cue
    raise AssertionError(f"no cue named {name}")


# ---------------------------------------------------------------------------
# The headline direction
# ---------------------------------------------------------------------------


def test_replay_scores_materially_below_the_clean_original() -> None:
    clean = live_speech(4.0, RATE, seed=0)
    replayed = replay_of(clean, RATE, seed=5)

    clean_score = _score(clean)
    replay_score = _score(replayed)

    assert clean_score >= CLEAN_FLOOR
    assert clean_score - replay_score >= REPLAY_MARGIN
    assert analyse_replay(build_context(clean, RATE)).verdict is FactorVerdict.PASS
    assert analyse_replay(build_context(replayed, RATE)).verdict is FactorVerdict.FAIL


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_direction_holds_across_different_utterances(seed: int) -> None:
    clean = live_speech(4.0, RATE, seed=seed)
    replayed = replay_of(clean, RATE, seed=seed + 100)
    assert _score(clean) - _score(replayed) >= REPLAY_MARGIN


def test_a_dry_room_is_still_caught_by_the_comb_alone() -> None:
    # RT60 of 180 ms leaves little tail, so the detector has to lean on the
    # fixed early reflections. This is the harder of the two replay cases.
    clean = live_speech(4.0, RATE, seed=0)
    dry = replay_of(clean, RATE, seed=5, rt60_s=0.18)
    assert _score(clean) - _score(dry) >= REPLAY_MARGIN
    assert _cue(dry, "fixed_comb").penalty > 0.5


def test_replay_of_synthetic_speech_is_caught_too() -> None:
    # The two attack signatures are independent: a cloned voice played out of
    # a speaker must trip the replay detector as well.
    assert _score(replay_of(tts_speech(4.0, RATE, seed=0), RATE, seed=5)) < 0.25


# ---------------------------------------------------------------------------
# Individual cues respond to the physics they claim to measure
# ---------------------------------------------------------------------------


def test_the_comb_cue_responds_to_an_early_reflection() -> None:
    clean = live_speech(4.0, RATE, seed=0)
    reflected = telephone_channel(
        np.convolve(clean, room_impulse_response(RATE, seed=1, tail_gain=0.0))[
            : clean.size
        ],
        RATE,
    )
    assert _cue(reflected, "fixed_comb").value > 2.0 * _cue(clean, "fixed_comb").value
    assert _cue(clean, "fixed_comb").penalty == 0.0


def test_the_tail_cue_needs_the_gap_to_be_speech_shaped() -> None:
    clean = live_speech(4.0, RATE, seed=0)
    replayed = replay_of(clean, RATE, seed=5)

    clean_cue = _cue(clean, "speech_shaped_tail")
    replay_cue = _cue(replayed, "speech_shaped_tail")

    # The gaps are both louder and spectrally matched to the speech.
    assert replay_cue.value > clean_cue.value + 8.0
    assert replay_cue.penalty > 0.4
    assert clean_cue.penalty < 0.2


def test_loud_stationary_noise_alone_does_not_trip_the_tail_cue() -> None:
    # A noisy line raises the level of the gaps without making them
    # speech-shaped, and must not be read as reverberation.
    clean = live_speech(4.0, RATE, seed=0)
    noisy = clean + telephone_channel(white_noise(4.0, RATE, seed=9, level=0.02), RATE)
    assert _cue(noisy, "speech_shaped_tail").penalty < 0.35
    assert _score(noisy) >= CLEAN_FLOOR


def test_the_decay_cue_measures_the_room_not_the_talker() -> None:
    clean = live_speech(4.0, RATE, seed=0)
    replayed = replay_of(clean, RATE, seed=5)
    # dB per second, negative. A handset offset is limited by the analysis
    # window; a room is limited by its RT60.
    assert _cue(clean, "offset_decay").value < -500.0
    assert _cue(replayed, "offset_decay").value > -300.0


def test_the_requantisation_cue_fires_only_on_uncompressed_pcm() -> None:
    clean = live_speech(4.0, RATE, seed=0)
    pcm = replay_of(clean, RATE, seed=5, requantise_bits=10, post_channel=False)
    coded = replay_of(clean, RATE, seed=5, requantise_bits=10, post_channel=True)

    pcm_cue = _cue(pcm, "requantisation")
    assert pcm_cue.measurable
    assert pcm_cue.value == pytest.approx(9.0, abs=1.0)
    assert pcm_cue.penalty > 0.9

    # Pushing it back through a codec erases the lattice. That has to read as
    # "no evidence", not as a clean bill of health.
    assert not _cue(coded, "requantisation").measurable


def test_the_corroborating_cues_never_vote_innocent() -> None:
    # playback_highband measures nothing on this corpus (-21 dB either way),
    # which is exactly why it is marked guilt-only: a cue that cannot
    # discriminate must not dilute the ones that can.
    clean_cue = _cue(live_speech(4.0, RATE, seed=0), "playback_highband")
    assert clean_cue.one_sided
    assert not clean_cue.counts


# ---------------------------------------------------------------------------
# Honesty: when it cannot measure, it says so
# ---------------------------------------------------------------------------


def test_hard_clipping_disables_the_comb_cue_instead_of_faking_one() -> None:
    # Clipping generates its own harmonic distortion, which looks exactly
    # like a comb. Scoring it would deny a payment over a hot microphone.
    squashed = clipped(live_speech(4.0, RATE, seed=0), limit=0.02)
    cue = _cue(squashed, "fixed_comb")
    assert not cue.measurable
    assert "clip" in cue.note
    assert _score(squashed) >= CLEAN_FLOOR


@pytest.mark.parametrize(
    "signal",
    [
        silence(3.0, RATE),
        white_noise(3.0, RATE, seed=1),
        np.zeros(400),
        live_speech(4.0, RATE, seed=0)[:800],
    ],
    ids=["silence", "noise", "tiny", "short"],
)
def test_unanalysable_audio_is_inconclusive_and_near_no_opinion(
    signal: np.ndarray,
) -> None:
    result = analyse_replay(build_context(signal, RATE))
    assert result.verdict is FactorVerdict.INCONCLUSIVE
    assert np.isfinite(result.score)
    # An inconclusive verdict has to come with a score that does not decide
    # anything, because detector.scoring blends inconclusive factors at face
    # value.
    assert 0.2 < result.score < 0.8


def test_a_short_call_says_why_it_is_unsure() -> None:
    result = analyse_replay(build_context(live_speech(4.0, RATE, seed=0)[:3200], RATE))
    assert "Low confidence" in result.detail
    assert len(result.detail) <= 500


def test_detail_names_the_measurement_not_just_a_label() -> None:
    replayed = replay_of(live_speech(4.0, RATE, seed=0), RATE, seed=5)
    detail = analyse_replay(build_context(replayed, RATE)).detail
    assert "ms spectral comb" in detail
    assert "dB/s" in detail or "reverberant tail" in detail
    # The speakerphone false-positive risk is disclosed to the reviewer.
    assert "speakerphone" in detail
    assert len(detail) <= 500


def test_three_agreeing_cues_are_allowed_to_reach_the_fatal_band() -> None:
    replayed = replay_of(live_speech(4.0, RATE, seed=0), RATE, seed=5)
    result = analyse_replay(build_context(replayed, RATE))
    assert result.measurements["strong_cues_agreeing"] >= 3.0
    assert result.measurements["score_floor"] <= 0.02
    # detector.scoring denies outright at or below 0.10.
    assert result.score <= 0.10


def test_one_agreeing_cue_cannot_reach_the_fatal_band() -> None:
    # A single room cue is floored well above the fatal threshold, so no one
    # noisy measurement can deny a payment by itself.
    quiet_room = replay_of(
        live_speech(4.0, RATE, seed=0),
        RATE,
        seed=5,
        rt60_s=0.02,
        speaker_band=(80.0, 3900.0),
    )
    result = analyse_replay(build_context(quiet_room, RATE))
    if result.measurements["strong_cues_agreeing"] <= 1.0:
        assert result.score > 0.10


def test_a_high_pitched_talker_is_not_mistaken_for_a_room() -> None:
    # A 215 Hz voice has a 4.7 ms glottal period, inside the early-reflection
    # quefrency band. Without the pitch exclusion its own harmonics would
    # read as a fixed comb.
    high = live_speech(4.0, RATE, seed=3, style=SpeechStyle(f0_hz=215.0))
    assert _cue(high, "fixed_comb").penalty < 0.4
    assert _score(high) >= CLEAN_FLOOR
