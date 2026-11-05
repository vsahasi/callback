"""Model tests: shapes, length-agnosticism, gradients, score convention.

CPU only, synthetic waveforms, no dataset. Skipped wholesale when torch is
absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.features import FeatureConfig  # noqa: E402
from training.model import (  # noqa: E402
    LCNN,
    CountermeasureModel,
    MaxFeatureMap2d,
    ModelConfig,
    build_model,
    count_parameters,
    score_from_logits,
)

CROP = 16_000


def _waveform(batch: int = 4, samples: int = CROP, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, samples, generator=generator)


# ---------------------------------------------------------------------------
# Max-Feature-Map
# ---------------------------------------------------------------------------


class TestMaxFeatureMap:
    def test_halves_the_channel_axis(self):
        assert MaxFeatureMap2d()(torch.randn(2, 8, 5, 5)).shape == (2, 4, 5, 5)

    def test_takes_the_elementwise_maximum_of_the_two_halves(self):
        x = torch.tensor([[[[1.0]], [[2.0]], [[5.0]], [[0.0]]]])  # (1, 4, 1, 1)
        out = MaxFeatureMap2d()(x)
        assert out.shape == (1, 2, 1, 1)
        assert out.flatten().tolist() == [5.0, 2.0]

    def test_works_on_a_two_dimensional_tensor_for_the_head(self):
        assert MaxFeatureMap2d(dim=1)(torch.randn(3, 10)).shape == (3, 5)

    def test_an_odd_channel_count_is_rejected(self):
        with pytest.raises(ValueError, match="even channel count"):
            MaxFeatureMap2d()(torch.randn(1, 7, 4, 4))

    def test_passes_gradient_through_the_winning_half(self):
        x = torch.tensor([[1.0, 2.0, 5.0, 0.0]], requires_grad=True)
        MaxFeatureMap2d(dim=1)(x).sum().backward()
        # Winners are index 2 (5 > 1) and index 1 (2 > 0).
        assert x.grad.flatten().tolist() == [0.0, 1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_rejects_a_wrong_length_channel_ladder(self):
        with pytest.raises(ValueError, match="5 entries"):
            ModelConfig(channels=(32, 48))

    def test_rejects_a_non_positive_pooled_freq(self):
        with pytest.raises(ValueError, match="pooled_freq"):
            ModelConfig(pooled_freq=0)

    def test_rejects_an_out_of_range_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            ModelConfig(dropout=1.0)


# ---------------------------------------------------------------------------
# LCNN trunk
# ---------------------------------------------------------------------------


class TestLcnn:
    def test_maps_a_feature_image_to_two_logits(self):
        model = LCNN()
        assert model(torch.randn(4, 1, 60, 101)).shape == (4, 2)

    def test_embedding_dimension_matches_the_config(self):
        model = LCNN(ModelConfig(embedding_dim=80))
        assert model.embed(torch.randn(2, 1, 60, 101)).shape == (2, 80)

    @pytest.mark.parametrize("frames", [32, 64, 101, 256, 404])
    def test_is_agnostic_to_the_time_axis(self, frames):
        """A model trained on 4 s crops has to score a call segment of
        whatever length arrives."""

        model = LCNN().eval()
        assert model(torch.randn(2, 1, 60, frames)).shape == (2, 2)

    @pytest.mark.parametrize("bins", [20, 40, 60, 80])
    def test_is_agnostic_to_the_frequency_axis(self, bins):
        """So the log-mel and LFCC front ends share one back end."""

        model = LCNN().eval()
        assert model(torch.randn(2, 1, bins, 101)).shape == (2, 2)

    def test_a_feature_map_too_small_to_pool_is_a_clear_error(self):
        model = LCNN()
        with pytest.raises(ValueError, match="too small for four"):
            model(torch.randn(2, 1, 8, 8))

    def test_wrong_rank_is_rejected(self):
        with pytest.raises(ValueError, match="4-D"):
            LCNN()(torch.randn(2, 60, 101))

    def test_temporal_variation_changes_the_embedding(self):
        """Variance over time is a cue in itself: synthesis artefacts tend to
        be unnaturally stationary, which is why the pooling layer keeps a
        standard deviation branch and not just a mean. A steady input and a
        strongly fluctuating one must not land on the same embedding."""

        model = LCNN().eval()
        steady = torch.ones(1, 1, 60, 101)
        wobbly = torch.ones(1, 1, 60, 101)
        wobbly[..., ::2] = 3.0
        wobbly[..., 1::2] = -1.0  # same mean as steady, far more variance
        assert not torch.allclose(model.embed(steady), model.embed(wobbly))

    def test_gradients_reach_the_first_convolution(self):
        model = LCNN()
        model(torch.randn(4, 1, 60, 101)).sum().backward()
        first_conv = model.trunk[0][0]
        assert first_conv.weight.grad is not None
        assert first_conv.weight.grad.abs().sum() > 0

    def test_batch_norm_can_be_switched_off(self):
        model = LCNN(ModelConfig(batch_norm=False)).eval()
        # Without BatchNorm a batch of one is fine in train mode too.
        model.train()
        assert model(torch.randn(1, 1, 60, 101)).shape == (1, 2)


# ---------------------------------------------------------------------------
# CountermeasureModel
# ---------------------------------------------------------------------------


class TestCountermeasureModel:
    def test_waveform_to_logits(self):
        model = CountermeasureModel().eval()
        assert model(_waveform()).shape == (4, 2)

    @pytest.mark.parametrize("frontend", ["lfcc", "logmel"])
    def test_both_front_ends_wire_up(self, frontend):
        model = CountermeasureModel(FeatureConfig(name=frontend)).eval()
        assert model(_waveform(2)).shape == (2, 2)

    def test_accepts_a_channel_dimension(self):
        model = CountermeasureModel().eval()
        assert model(_waveform(2).unsqueeze(1)).shape == (2, 2)

    def test_parameter_count_stays_modest(self):
        """The architecture choice was partly about size. A refactor that
        inflates it by an order of magnitude should fail here rather than show
        up as an out-of-memory error on the box."""

        total = count_parameters(CountermeasureModel())
        assert 50_000 < total < 1_000_000

    def test_the_front_end_contributes_no_parameters(self):
        """Filterbanks and the DCT are fixed linear algebra, not learned."""

        model = CountermeasureModel()
        assert count_parameters(model.frontend) == 0

    def test_a_training_step_moves_the_loss(self):
        torch.manual_seed(0)
        model = CountermeasureModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()
        waveform = _waveform(8, 8_000)
        target = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

        # Both measurements are taken in train mode on purpose. This model
        # has seven BatchNorm layers, so a train-mode loss and an
        # eval-mode loss are computed from different statistics and are
        # not comparable: train mode normalises with this batch, eval mode
        # uses running averages which, after a handful of
        # high-learning-rate steps on one synthetic batch, still lag far
        # behind the activations. Comparing across the two measures
        # BatchNorm warm-up, not learning, and it reported a rising loss
        # for a model whose training loss was in fact collapsing.
        model.train()
        with torch.no_grad():
            before = criterion(model(waveform), target).item()
        for _ in range(5):
            optimizer.zero_grad()
            criterion(model(waveform), target).backward()
            optimizer.step()
        with torch.no_grad():
            after = criterion(model(waveform), target).item()
        assert after < before, f"loss did not fall: {before:.4f} -> {after:.4f}"

    def test_every_parameter_is_reached_by_backpropagation(self):
        """A parameter with no gradient tensor at all is a layer wired into the
        module but not into the forward pass."""

        model = CountermeasureModel()
        model(_waveform(4, 8_000)).sum().backward()
        unreached = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is None
        ]
        assert unreached == []

    def test_almost_every_parameter_gets_a_non_zero_gradient(self):
        """MFM routes gradient to the winning half only, so a handful of
        weights can legitimately come out at zero on one batch. A large block
        of zeros would mean a dead branch."""

        model = CountermeasureModel()
        model(_waveform(4, 8_000)).sum().backward()
        grads = [p.grad.abs().sum().item() for p in model.parameters()]
        live = sum(1 for value in grads if value > 0)
        assert live >= 0.9 * len(grads)

    def test_score_returns_one_value_per_utterance(self):
        model = CountermeasureModel()
        scores = model.score(_waveform(3))
        assert scores.shape == (3,)

    def test_score_leaves_the_training_flag_as_it_found_it(self):
        model = CountermeasureModel()
        model.train()
        model.score(_waveform(2))
        assert model.training is True
        model.eval()
        model.score(_waveform(2))
        assert model.training is False

    def test_eval_mode_is_deterministic_despite_dropout(self):
        model = CountermeasureModel().eval()
        waveform = _waveform(2)
        assert torch.equal(model(waveform), model(waveform))

    def test_build_model_is_the_same_thing(self):
        assert isinstance(build_model(), CountermeasureModel)


class TestCheckpointRoundTrip:
    def test_config_dict_describes_both_halves(self):
        model = CountermeasureModel(
            FeatureConfig(name="logmel", n_mels=40), ModelConfig(dropout=0.1)
        )
        payload = model.config_dict()
        assert payload["feature"]["name"] == "logmel"
        assert payload["feature"]["n_mels"] == 40
        assert payload["model"]["dropout"] == 0.1

    def test_a_checkpoint_rebuilds_an_identical_model(self, tmp_path):
        """A model that silently gets log-mel at inference after being trained
        on LFCC is a bug that reads as "the deployed detector is worse than the
        paper". The config travels with the weights so it cannot happen."""

        original = CountermeasureModel(
            FeatureConfig(name="logmel", n_mels=40), ModelConfig(dropout=0.1)
        ).eval()
        path = tmp_path / "ckpt.pt"
        torch.save(
            {"model": original.state_dict(), "config": original.config_dict()},
            path,
        )

        payload = torch.load(path, weights_only=False)
        restored = CountermeasureModel.from_config_dict(payload["config"])
        restored.load_state_dict(payload["model"])
        restored.eval()

        waveform = _waveform(2)
        assert torch.equal(original(waveform), restored(waveform))

    def test_channels_survive_a_json_round_trip_as_a_list(self):
        """torch.save keeps the tuple, but a config that has been through JSON
        comes back as a list, and ModelConfig validates on length either
        way."""

        import json

        payload = json.loads(json.dumps(CountermeasureModel().config_dict()))
        restored = CountermeasureModel.from_config_dict(payload)
        assert restored.model_config.channels == (32, 48, 64, 32, 32)

    def test_a_malformed_config_is_rejected(self):
        with pytest.raises(ValueError, match="must hold"):
            CountermeasureModel.from_config_dict({"feature": "lfcc", "model": "lcnn"})


# ---------------------------------------------------------------------------
# Score convention
# ---------------------------------------------------------------------------


class TestScoreFromLogits:
    def test_returns_the_bonafide_log_probability(self):
        logits = torch.tensor([[0.0, 0.0]])
        # Equal logits give a probability of 0.5, so log(0.5).
        assert score_from_logits(logits).item() == pytest.approx(
            torch.log(torch.tensor(0.5)).item()
        )

    def test_higher_means_more_bonafide(self):
        """The convention the metrics module and the official ASVspoof
        packages both expect. Getting it backwards turns an EER of 0.05 into
        0.95."""

        confident_bonafide = torch.tensor([[-5.0, 5.0]])
        confident_spoof = torch.tensor([[5.0, -5.0]])
        assert (
            score_from_logits(confident_bonafide).item()
            > score_from_logits(confident_spoof).item()
        )

    def test_is_bounded_above_by_zero(self):
        scores = score_from_logits(torch.randn(100, 2))
        assert scores.max().item() <= 0.0

    def test_is_monotonic_in_the_logit_difference(self):
        differences = torch.linspace(-10, 10, 50)
        logits = torch.stack([torch.zeros_like(differences), differences], dim=1)
        scores = score_from_logits(logits)
        assert torch.all(scores[1:] > scores[:-1])

    def test_rejects_the_wrong_shape(self):
        with pytest.raises(ValueError, match="batch, 2"):
            score_from_logits(torch.randn(4))
        with pytest.raises(ValueError, match="batch, 2"):
            score_from_logits(torch.randn(4, 1))
