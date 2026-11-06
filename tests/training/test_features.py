"""Front end tests. CPU only, synthetic waveforms, no dataset.

Skipped wholesale when torch is absent, so the dependency-free part of the
suite stays green.
"""

from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from training.features import (  # noqa: E402  (after importorskip by design)
    LFCC,
    FeatureConfig,
    LogMelSpectrogram,
    build_frontend,
    compute_deltas,
    dct_matrix,
    hz_to_mel,
    linear_filterbank,
    mel_filterbank,
    mel_to_hz,
)

SAMPLE_RATE = 16_000
CROP = 16_000  # 1 second


def _waveform(batch: int = 2, samples: int = CROP, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, samples, generator=generator)


def _tone(frequency: float, samples: int = CROP, rate: int = SAMPLE_RATE):
    time = torch.arange(samples, dtype=torch.float32)
    return torch.sin(2 * math.pi * frequency * time / rate)


# ---------------------------------------------------------------------------
# Scales, filterbanks, DCT
# ---------------------------------------------------------------------------


class TestMelScale:
    def test_round_trips(self):
        for hz in (0.0, 100.0, 1000.0, 4000.0, 8000.0):
            assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, abs=1e-6)

    def test_zero_maps_to_zero(self):
        assert hz_to_mel(0.0) == 0.0
        assert mel_to_hz(0.0) == 0.0

    def test_is_monotonically_increasing(self):
        values = [hz_to_mel(hz) for hz in range(0, 8000, 250)]
        assert all(a < b for a, b in itertools.pairwise(values))

    def test_matches_the_published_htk_anchor(self):
        """The HTK formula 2595 * log10(1 + f/700) maps 1000 Hz to about
        999.99 mel, which is the anchor the constant was chosen for."""

        assert hz_to_mel(1000.0) == pytest.approx(999.99, abs=0.1)


class TestFilterbanks:
    def test_mel_bank_shape(self):
        bank = mel_filterbank(80, 512, SAMPLE_RATE)
        assert bank.shape == (80, 257)

    def test_linear_bank_shape(self):
        bank = linear_filterbank(20, 512, SAMPLE_RATE)
        assert bank.shape == (20, 257)

    @pytest.mark.parametrize("builder", [mel_filterbank, linear_filterbank])
    def test_filters_are_non_negative(self, builder):
        assert (builder(20, 512, SAMPLE_RATE) >= 0).all()

    @pytest.mark.parametrize("builder", [mel_filterbank, linear_filterbank])
    def test_each_filter_peaks_at_one(self, builder):
        """Peak normalisation, matching torchaudio's default norm=None.

        The analytic triangle peaks at exactly 1.0, but the bank is
        sampled at FFT bin centres and an apex almost never lands on a
        bin, so the sampled maximum approaches 1.0 from below. torchaudio
        behaves identically. What matters is that no filter exceeds 1.0
        (which would mean the normalisation is wrong) and that none is
        far short of it (which would mean a filter is degenerate).
        """

        bank = builder(20, 512, SAMPLE_RATE)
        peaks = bank.max(dim=1).values
        assert bool((peaks <= 1.0 + 1e-6).all()), peaks
        assert bool((peaks > 0.85).all()), peaks

    def test_linear_bank_centres_are_evenly_spaced(self):
        """That is the whole point of LFCC: uniform resolution, so the high
        band where vocoder artefacts live is not compressed away."""

        bank = linear_filterbank(10, 512, SAMPLE_RATE)
        centres = bank.argmax(dim=1).to(torch.float32)
        gaps = centres[1:] - centres[:-1]
        assert gaps.std() < 1.0

    def test_mel_bank_centres_are_not_evenly_spaced(self):
        bank = mel_filterbank(10, 512, SAMPLE_RATE)
        centres = bank.argmax(dim=1).to(torch.float32)
        gaps = centres[1:] - centres[:-1]
        assert gaps.std() > 1.0

    def test_a_band_limit_confines_the_bank(self):
        bank = linear_filterbank(10, 512, SAMPLE_RATE, f_min=0.0, f_max=4000.0)
        # 4000 Hz is bin 128 of 257; nothing above it should be active.
        assert bank[:, 140:].abs().max() == 0.0


class TestDctMatrix:
    def test_shape(self):
        assert dct_matrix(20, 40).shape == (20, 40)

    def test_is_orthonormal_when_square(self):
        # Build in float64 rather than casting a float32 matrix up: the
        # cast preserves float32 rounding (~1e-8), which cannot meet a
        # 1e-10 tolerance. dct_matrix computes in float64 internally and
        # takes a dtype, so ask it for the precision being asserted.
        matrix = dct_matrix(8, 8, dtype=torch.float64)
        product = matrix @ matrix.T
        assert torch.allclose(product, torch.eye(8, dtype=torch.float64), atol=1e-10)

    def test_first_row_is_the_scaled_mean(self):
        """The zeroth DCT-II coefficient of x is sum(x) / sqrt(N)."""

        matrix = dct_matrix(4, 16).to(torch.float64)
        x = torch.arange(16, dtype=torch.float64)
        assert (matrix @ x)[0] == pytest.approx(x.sum().item() / math.sqrt(16))

    def test_a_constant_signal_has_no_energy_above_the_zeroth_coefficient(self):
        matrix = dct_matrix(8, 16).to(torch.float64)
        coefficients = matrix @ torch.ones(16, dtype=torch.float64)
        assert coefficients[1:].abs().max() < 1e-10

    def test_requesting_more_outputs_than_inputs_is_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            dct_matrix(10, 5)


class TestComputeDeltas:
    def test_preserves_shape(self):
        features = torch.randn(2, 20, 50)
        assert compute_deltas(features).shape == features.shape

    def test_a_constant_signal_has_zero_delta(self):
        features = torch.ones(1, 3, 20)
        assert compute_deltas(features).abs().max() < 1e-6

    def test_a_unit_ramp_has_unit_delta(self):
        """For x[t] = t the regression formula gives
        sum(n * 2n) / (2 * sum n^2) = 1 exactly, in the interior where the
        replicate padding does not reach."""

        ramp = torch.arange(40, dtype=torch.float32).view(1, 1, 40)
        deltas = compute_deltas(ramp, width=5)
        interior = deltas[0, 0, 2:-2]
        assert torch.allclose(interior, torch.ones_like(interior), atol=1e-5)

    def test_the_sign_follows_the_slope(self):
        """A descending kernel would negate the derivative, which trains fine
        and is silently wrong against every other delta implementation."""

        falling = torch.arange(40, 0, -1, dtype=torch.float32).view(1, 1, 40)
        deltas = compute_deltas(falling, width=5)
        assert deltas[0, 0, 2:-2].max() < 0

    def test_even_or_tiny_widths_are_rejected(self):
        features = torch.randn(1, 2, 10)
        with pytest.raises(ValueError, match="odd and at least 3"):
            compute_deltas(features, width=4)
        with pytest.raises(ValueError, match="odd and at least 3"):
            compute_deltas(features, width=1)

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_deltas(torch.randn(2, 1, 20, 30))


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    def test_rejects_an_unknown_frontend(self):
        with pytest.raises(ValueError, match="unknown frontend"):
            FeatureConfig(name="mfcc")  # type: ignore[arg-type]

    def test_rejects_a_window_longer_than_the_fft(self):
        with pytest.raises(ValueError, match="must not exceed n_fft"):
            FeatureConfig(n_fft=256, win_length=400)

    def test_rejects_more_cepstra_than_filters(self):
        with pytest.raises(ValueError, match="cannot exceed n_filter"):
            FeatureConfig(n_filter=10, n_ceps=20)

    def test_rejects_a_non_positive_hop(self):
        with pytest.raises(ValueError, match="hop_length"):
            FeatureConfig(hop_length=0)

    def test_n_features_accounts_for_deltas(self):
        assert FeatureConfig(name="lfcc", n_ceps=20, with_deltas=True).n_features == 60
        assert FeatureConfig(name="lfcc", n_ceps=20, with_deltas=False).n_features == 20
        assert FeatureConfig(name="logmel", n_mels=80).n_features == 80

    def test_f_max_defaults_to_nyquist(self):
        config = FeatureConfig(sample_rate=16_000)
        assert config.top_hz == 8000.0


# ---------------------------------------------------------------------------
# Front ends
# ---------------------------------------------------------------------------


class TestLogMel:
    def test_output_shape(self):
        config = FeatureConfig(name="logmel", n_mels=80)
        frontend = LogMelSpectrogram(config)
        features = frontend(_waveform())
        # center=True, hop 160, 16000 samples -> 101 frames.
        assert features.shape == (2, 1, 80, 101)

    def test_reports_its_feature_count(self):
        assert LogMelSpectrogram(FeatureConfig(name="logmel")).n_features == 80

    def test_output_is_finite_for_silence(self):
        """log(0) is the obvious way to produce -inf; the epsilon prevents
        it."""

        frontend = LogMelSpectrogram(FeatureConfig(name="logmel", normalise=False))
        features = frontend(torch.zeros(1, CROP))
        assert torch.isfinite(features).all()

    def test_normalisation_zero_means_each_coefficient(self):
        frontend = LogMelSpectrogram(FeatureConfig(name="logmel", normalise=True))
        features = frontend(_waveform())
        assert features.mean(dim=-1).abs().max() < 1e-4

    def test_a_tone_concentrates_energy_near_its_own_mel_band(self):
        frontend = LogMelSpectrogram(
            FeatureConfig(name="logmel", n_mels=40, normalise=False)
        )
        low = frontend(_tone(300.0).unsqueeze(0))
        high = frontend(_tone(6000.0).unsqueeze(0))
        low_peak = low[0, 0].mean(dim=-1).argmax().item()
        high_peak = high[0, 0].mean(dim=-1).argmax().item()
        assert low_peak < high_peak


class TestLfcc:
    def test_output_shape_with_deltas(self):
        frontend = LFCC(FeatureConfig(name="lfcc", n_ceps=20, with_deltas=True))
        assert frontend(_waveform()).shape == (2, 1, 60, 101)

    def test_output_shape_without_deltas(self):
        frontend = LFCC(FeatureConfig(name="lfcc", n_ceps=20, with_deltas=False))
        assert frontend(_waveform()).shape == (2, 1, 20, 101)

    def test_reports_its_feature_count(self):
        frontend = LFCC(FeatureConfig(name="lfcc", n_ceps=20, with_deltas=True))
        assert frontend.n_features == 60

    def test_output_is_finite_for_silence(self):
        frontend = LFCC(FeatureConfig(name="lfcc", normalise=False))
        assert torch.isfinite(frontend(torch.zeros(1, CROP))).all()

    def test_fewer_cepstra_narrows_the_output(self):
        frontend = LFCC(
            FeatureConfig(name="lfcc", n_filter=20, n_ceps=12, with_deltas=False)
        )
        assert frontend(_waveform()).shape[2] == 12

    def test_energy_replacement_tracks_loudness(self):
        """With with_energy the zeroth coefficient is the log frame energy, so
        scaling the waveform must shift it."""

        frontend = LFCC(
            FeatureConfig(
                name="lfcc",
                with_energy=True,
                with_deltas=False,
                normalise=False,
            )
        )
        quiet = frontend((0.01 * _waveform(1)).contiguous())
        loud = frontend(_waveform(1))
        assert loud[0, 0, 0].mean() > quiet[0, 0, 0].mean()

    def test_deltas_of_a_steady_signal_are_near_zero(self):
        frontend = LFCC(FeatureConfig(name="lfcc", with_deltas=True, normalise=False))
        features = frontend(_tone(1000.0).unsqueeze(0))
        n_ceps = 20
        # Static block varies; the delta blocks should be nearly flat for a
        # stationary tone, away from the edges.
        deltas = features[0, 0, n_ceps : 2 * n_ceps, 5:-5]
        statics = features[0, 0, :n_ceps, 5:-5]
        assert deltas.abs().mean() < statics.abs().mean()


class TestSharedFrontendBehaviour:
    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_accepts_unbatched_batched_and_channelled_input(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        flat = frontend(torch.randn(CROP))
        batched = frontend(torch.randn(1, CROP))
        channelled = frontend(torch.randn(1, 1, CROP))
        assert flat.shape == batched.shape == channelled.shape

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_downmixes_multichannel_audio(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        assert frontend(torch.randn(2, 2, CROP)).shape[0] == 2

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_rejects_a_rank_five_tensor(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        with pytest.raises(ValueError, match="1, 2 or 3 dims"):
            frontend(torch.randn(1, 1, 1, 1, CROP))

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_rejects_audio_shorter_than_the_window(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        with pytest.raises(ValueError, match="shorter than win_length"):
            frontend(torch.randn(1, 100))

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_is_deterministic(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        waveform = _waveform()
        assert torch.equal(frontend(waveform), frontend(waveform))

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_frame_count_scales_with_input_length(self, name):
        frontend = build_frontend(FeatureConfig(name=name))
        short = frontend(torch.randn(1, 8_000))
        long = frontend(torch.randn(1, 32_000))
        assert long.shape[-1] > short.shape[-1]

    def test_build_frontend_resolves_the_name(self):
        assert isinstance(build_frontend(FeatureConfig(name="lfcc")), LFCC)
        assert isinstance(
            build_frontend(FeatureConfig(name="logmel")), LogMelSpectrogram
        )

    def test_build_frontend_defaults_to_lfcc(self):
        assert isinstance(build_frontend(), LFCC)

    @pytest.mark.parametrize("name", ["lfcc", "logmel"])
    def test_buffers_are_not_persisted_into_a_state_dict(self, name):
        """Filterbanks and DCT matrices are derived from the config, so
        persisting them would bloat every checkpoint and, worse, let a
        checkpoint carry a filterbank that disagrees with its own config."""

        frontend = build_frontend(FeatureConfig(name=name))
        assert frontend.state_dict() == {}
