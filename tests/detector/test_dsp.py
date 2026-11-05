"""Signal-processing primitives.

These are the measurements every detector stands on, so they are tested
directly against signals whose answer is known by construction rather than
only through a detector's score.

Skipped wholesale when numpy is absent, so the dependency-free part of the
suite stays green.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.dsp import (  # noqa: E402  (after importorskip by design)
    MAX_WORKING_RATE,
    PitchTrack,
    amplitude_perturbation_db,
    build_context,
    clamp01,
    envelope_periodicity,
    f0_perturbation_pct,
    frames,
    interior_gap_mask,
    linear_slope,
    pearson,
    prepare,
    quantisation_step,
    real_cepstrum,
    resample_fft,
    rolloff_hz,
    smooth_mask,
    spectral_flatness,
    true_runs,
)
from detector.analysis.synthetic import (  # noqa: E402
    live_speech,
    pure_tone,
    silence,
    telephone_channel,
    tts_speech,
    white_noise,
)

RATE = 8000


def _harmonic(f0: float, duration_s: float = 1.5, rate: int = RATE) -> np.ndarray:
    """A band-limited harmonic stack at a fixed fundamental."""

    time = np.arange(int(duration_s * rate), dtype=np.float64) / rate
    signal = np.zeros_like(time)
    for harmonic in range(1, 16):
        signal += np.sin(2.0 * np.pi * f0 * harmonic * time) / harmonic
    return telephone_channel(signal, rate)


# ---------------------------------------------------------------------------
# Scalars and shapes
# ---------------------------------------------------------------------------


def test_clamp01_turns_non_finite_into_no_opinion() -> None:
    assert clamp01(0.3) == pytest.approx(0.3)
    assert clamp01(-2.0) == 0.0
    assert clamp01(7.0) == 1.0
    # A NaN score would reach a reviewer looking like a number, so it has to
    # collapse to "no opinion", not to 0.0 or 1.0.
    assert clamp01(float("nan")) == 0.5
    assert clamp01(float("inf")) == 0.5


def test_true_runs_and_smoothing() -> None:
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert true_runs(mask) == [(1, 3), (4, 5), (7, 10)]
    assert true_runs(np.zeros(0, dtype=bool)) == []

    # One quiet frame inside a word is bridged; an isolated frame is dropped.
    smoothed = smooth_mask(mask, min_run=2, max_gap=1)
    assert smoothed.tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    ]


def test_frames_pads_short_input_rather_than_raising() -> None:
    framed = frames(np.arange(5.0), win=8, hop=4)
    assert framed.shape == (1, 8)
    assert framed[0, 5:].tolist() == [0.0, 0.0, 0.0]

    framed = frames(np.arange(20.0), win=8, hop=4)
    assert framed.shape == (4, 8)
    assert framed[1].tolist() == list(range(4, 12))


def test_pearson_and_slope_degenerate_cases() -> None:
    assert pearson(np.zeros(10), np.zeros(10)) == 0.0
    assert pearson(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert pearson(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)
    assert pearson(np.arange(2.0), np.arange(2.0)) == 0.0  # too short to mean anything

    assert linear_slope(np.arange(5.0), 3.0 * np.arange(5.0) + 1.0) == pytest.approx(
        3.0
    )
    assert linear_slope(np.zeros(5), np.arange(5.0)) == 0.0
    assert linear_slope(np.array([1.0]), np.array([1.0])) == 0.0


def test_spectral_flatness_bounds() -> None:
    assert spectral_flatness(np.ones(32)) == pytest.approx(1.0)
    spike = np.zeros(32)
    spike[7] = 1.0
    assert spectral_flatness(spike) < 0.05
    assert spectral_flatness(np.array([1.0])) == 1.0


def test_rolloff_and_cepstrum_handle_empty_input() -> None:
    assert rolloff_hz(np.zeros(0), np.zeros(0)) == 0.0
    assert rolloff_hz(np.array([0.0, 100.0]), np.zeros(2)) == 0.0
    assert real_cepstrum(np.zeros(2)).size == 4


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_prepare_accepts_lists_and_multichannel() -> None:
    block = prepare([0.1, -0.2, 0.3, -0.4] * 100, RATE)
    assert block.samples.ndim == 1
    assert block.sample_rate == RATE

    stereo = np.stack([np.ones(800), -np.ones(800)])
    assert np.allclose(prepare(stereo, RATE).samples, 0.0)


def test_prepare_accepts_a_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    signal = live_speech(1.0, RATE, seed=0)
    from_numpy = prepare(signal, RATE)
    from_tensor = prepare(torch.from_numpy(signal), RATE)
    assert np.allclose(from_numpy.samples, from_tensor.samples)


def test_prepare_records_rather_than_hides_bad_input() -> None:
    signal = live_speech(2.0, RATE, seed=0)

    dirty = signal.copy()
    dirty[::50] = np.nan
    dirty[1::97] = np.inf
    block = prepare(dirty, RATE)
    assert block.non_finite_fraction > 0.0
    assert np.all(np.isfinite(block.samples))

    offset = prepare(signal + 0.4, RATE)
    assert offset.dc_offset == pytest.approx(0.4, abs=0.02)
    assert abs(float(np.mean(offset.samples))) < 1e-9

    squashed = prepare(np.clip(signal, -0.01, 0.01), RATE)
    assert squashed.clipped_fraction > 0.05


def test_prepare_decimates_only_above_the_working_rate() -> None:
    at_16k = prepare(live_speech(1.0, 16000, seed=0), 16000)
    assert not at_16k.resampled
    assert at_16k.sample_rate == 16000

    at_48k = prepare(live_speech(1.0, 48000, seed=0), 48000)
    assert at_48k.resampled
    assert at_48k.sample_rate == MAX_WORKING_RATE
    assert at_48k.duration_s == pytest.approx(1.0, abs=0.01)

    # 4 kHz is the documented minimum usable rate, so it is not "too low";
    # anything under it is.
    assert not prepare(live_speech(1.0, 4000, seed=0), 4000).rate_too_low
    assert prepare(live_speech(1.0, 3000, seed=0), 3000).rate_too_low
    assert prepare(np.zeros(100), RATE).too_short


def test_prepare_rejects_a_nonsense_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        prepare(np.zeros(800), 0)


def test_resample_fft_preserves_frequency() -> None:
    tone = pure_tone(1.0, RATE, frequency=300.0)
    upsampled = resample_fft(tone, 2 * tone.size)
    spectrum = np.abs(np.fft.rfft(upsampled))
    peak_hz = np.fft.rfftfreq(upsampled.size, d=1.0 / (2 * RATE))[
        int(np.argmax(spectrum))
    ]
    assert peak_hz == pytest.approx(300.0, abs=5.0)
    assert np.array_equal(resample_fft(tone, tone.size), tone)
    assert resample_fft(np.zeros(0), 10).size == 10


# ---------------------------------------------------------------------------
# Activity, pitch and perturbation
# ---------------------------------------------------------------------------


def test_activity_finds_structure_in_speech_but_not_in_noise() -> None:
    speech = build_context(live_speech(4.0, RATE, seed=0), RATE).activity
    assert speech.has_structure
    assert 0.8 < speech.speech_seconds < 4.0
    assert speech.dynamic_range_db > 20.0

    for flat in (
        silence(2.0, RATE),
        white_noise(2.0, RATE, seed=1),
        pure_tone(2.0, RATE),
    ):
        assert not build_context(flat, RATE).activity.has_structure


@pytest.mark.parametrize("f0", [110.0, 160.0, 220.0])
def test_pitch_tracker_finds_the_fundamental_not_a_formant(f0: float) -> None:
    # The telephone high-pass removes the fundamental of every one of these
    # voices, so a naive autocorrelation locks onto a harmonic. Within 4% is
    # a real test that the octave handling works.
    pitch = build_context(_harmonic(f0), RATE).pitch
    assert pitch.voiced_seconds > 0.5
    assert pitch.median_f0 == pytest.approx(f0, rel=0.04)


def test_pitch_contour_spread_tracks_the_generated_range() -> None:
    steady = build_context(_harmonic(150.0), RATE).pitch
    assert steady.semitone_spread() < 1.0

    # The live generator is configured for a five-semitone contour.
    moving = build_context(live_speech(4.0, RATE, seed=0), RATE).pitch
    assert 2.5 < moving.semitone_spread() < 9.0


def test_pitch_track_is_empty_for_silence() -> None:
    pitch = build_context(silence(2.0, RATE), RATE).pitch
    assert pitch.voiced_frames == 0
    assert pitch.median_f0 == 0.0
    assert pitch.median_period_s == 0.0
    assert pitch.semitone_spread() == 0.0


def _track(f0_values: list[float]) -> PitchTrack:
    f0 = np.asarray(f0_values, dtype=np.float64)
    return PitchTrack(
        f0=f0,
        voiced=f0 > 0.0,
        strength=np.full(f0.size, 0.9),
        hop_s=0.01,
    )


def test_rap_ignores_a_pitch_glide_but_measures_a_wobble() -> None:
    # A pure contour: the period changes every frame, but smoothly. RAP must
    # read nearly zero, or the cue would be measuring how expressive the
    # delivery was.
    glide = _track(np.linspace(120.0, 150.0, 40).tolist())
    assert f0_perturbation_pct(glide) < 0.05

    # Alternating +-1% periods. The three-point detrend leaves two thirds of
    # the peak-to-peak difference, so 2% peak-to-peak reads as about 1.33%.
    base = 1.0 / 120.0
    alternating = [1.0 / (base * (1.0 + (0.01 if n % 2 else -0.01))) for n in range(40)]
    assert 1.0 < f0_perturbation_pct(_track(alternating)) < 1.7

    assert f0_perturbation_pct(_track([120.0] * 40)) == 0.0  # no wobble at all
    assert f0_perturbation_pct(_track([120.0, 120.0])) == 0.0  # no run long enough


def test_rap_discards_octave_errors_instead_of_scoring_them() -> None:
    values = [120.0] * 20
    values[10] = 240.0  # one halved period: a tracking error, not jitter
    # Without rejection this single frame would read as a ~100% perturbation.
    assert f0_perturbation_pct(_track(values)) < 0.1


def test_amplitude_perturbation_needs_a_voiced_run() -> None:
    ctx = build_context(silence(2.0, RATE), RATE)
    assert amplitude_perturbation_db(ctx.spec, ctx.pitch) == 0.0

    live = build_context(live_speech(4.0, RATE, seed=0), RATE)
    synthetic_voice = build_context(tts_speech(4.0, RATE, seed=0), RATE)
    assert amplitude_perturbation_db(live.spec, live.pitch) > amplitude_perturbation_db(
        synthetic_voice.spec, synthetic_voice.pitch
    )


# ---------------------------------------------------------------------------
# Gaps, rhythm and the sample lattice
# ---------------------------------------------------------------------------


def test_interior_gap_mask_excludes_the_edges() -> None:
    speech = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
    gap = interior_gap_mask(speech, erode=2)
    # Only the eroded middle of the interior gap survives; the leading and
    # trailing silence is dropped.
    assert gap.tolist() == [False] * 6 + [True, True] + [False] * 6
    assert not interior_gap_mask(np.ones(5, dtype=bool)).any()


def test_envelope_periodicity_separates_a_metronome_from_noise() -> None:
    hop = 0.01
    frame_count = 600  # six seconds
    period_frames = 70  # 700 ms
    square = np.array(
        [1.0 if (n % period_frames) < 45 else 0.05 for n in range(frame_count)]
    )
    strength, period = envelope_periodicity(square, hop)
    assert strength > 0.6
    assert period == pytest.approx(0.70, abs=0.03)

    rng = np.random.default_rng(0)
    strength, _ = envelope_periodicity(rng.random(frame_count), hop)
    assert strength < 0.5

    assert envelope_periodicity(np.ones(4), hop) == (0.0, 0.0)
    assert envelope_periodicity(np.ones(600), 0.0) == (0.0, 0.0)


def test_quantisation_step_reads_a_coarse_lattice() -> None:
    rng = np.random.default_rng(0)
    step = 2.0**-8
    coarse = np.round(rng.uniform(-0.5, 0.5, 4000) / step) * step
    relative, on_lattice, bits = quantisation_step(coarse)
    assert on_lattice > 0.999
    assert relative == pytest.approx(step / np.max(np.abs(coarse)), rel=0.05)
    assert 6.5 < bits < 8.5

    # Ordinary float audio is not on any lattice, which must read as "no
    # evidence" rather than as a clean bill of health.
    _, float_lattice, _ = quantisation_step(rng.uniform(-0.5, 0.5, 4000))
    assert float_lattice < 0.5

    assert quantisation_step(np.zeros(4000)) == (0.0, 0.0, 0.0)
    assert quantisation_step(np.ones(10)) == (0.0, 0.0, 0.0)


def test_mean_log_spectrum_is_finite_for_silence() -> None:
    ctx = build_context(silence(1.0, RATE), RATE)
    spectrum = ctx.mean_log_spectrum()
    assert spectrum.size == ctx.spec.freqs.size
    assert np.all(np.isfinite(spectrum))


def test_context_notes_name_every_conditioning_problem() -> None:
    notes = build_context(live_speech(1.0, 48000, seed=0), 48000).notes
    assert any("decimated" in note for note in notes)

    notes = build_context(np.zeros(100), RATE).notes
    assert any("ms of audio" in note for note in notes)

    notes = build_context(live_speech(1.0, 3000, seed=0), 3000).notes
    assert any("below 4 kHz" in note for note in notes)


def test_harmonic_helper_is_actually_harmonic() -> None:
    # Guards the test fixture itself: if _harmonic stopped producing a
    # periodic signal the pitch tests above would pass for the wrong reason.
    signal = _harmonic(150.0, 0.5)
    assert signal.size == int(0.5 * RATE)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / RATE)
    peak_hz = freqs[int(np.argmax(spectrum))]
    assert peak_hz % 150.0 < 5.0 or 150.0 - (peak_hz % 150.0) < 5.0
    assert math.isfinite(float(np.max(np.abs(signal))))
