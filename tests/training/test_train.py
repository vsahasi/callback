"""Training loop tests, including the end-to-end smoke test.

Everything runs on CPU against synthetic random audio: no dataset, no network,
no audio files. Skipped wholesale when torch is absent.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.datasets import SyntheticWaveformDataset, collate_batch  # noqa: E402
from training.metrics import AsvErrorRates, read_score_file  # noqa: E402
from training.model import CountermeasureModel  # noqa: E402
from training.train import (  # noqa: E402
    TrainConfig,
    Trainer,
    build_parser,
    config_from_args,
    evaluate,
    main,
    run,
    seed_everything,
)


def _smoke_config(output_dir: Path, **overrides) -> TrainConfig:
    payload = {
        "smoke_test": True,
        "output_dir": output_dir,
        "device": "cpu",
        "amp": False,
        "epochs": 2,
        **overrides,
    }
    return TrainConfig(**payload)


def _loader(n_items: int = 16, batch_size: int = 8, seed: int = 0):
    from torch.utils.data import DataLoader

    dataset = SyntheticWaveformDataset(n_items=n_items, crop_samples=8_000, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_batch,
        num_workers=0,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Config and CLI
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_feature_config_is_derived_consistently(self):
        config = TrainConfig(frontend="logmel", n_mels=40)
        feature = config.feature_config()
        assert feature.name == "logmel"
        assert feature.n_mels == 40

    def test_no_asv_rates_means_eer_only(self):
        assert TrainConfig().asv_error_rates() is None

    def test_all_three_asv_rates_build_an_operating_point(self):
        config = TrainConfig(asv_p_miss=0.05, asv_p_fa=0.01, asv_p_miss_spoof=0.9)
        rates = config.asv_error_rates()
        assert isinstance(rates, AsvErrorRates)
        assert rates.p_miss == 0.05

    def test_a_partial_asv_operating_point_is_rejected(self):
        """Silently defaulting the missing rate would produce a plausible but
        meaningless t-DCF."""

        config = TrainConfig(asv_p_miss=0.05)
        with pytest.raises(ValueError, match="all three ASV error rates"):
            config.asv_error_rates()

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("bfloat16", torch.bfloat16),
            ("bf16", torch.bfloat16),
            ("float16", torch.float16),
            ("fp16", torch.float16),
        ],
    )
    def test_amp_dtype_resolution(self, name, expected):
        assert TrainConfig(amp_dtype=name).autocast_dtype() is expected

    def test_an_unknown_amp_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="amp_dtype"):
            TrainConfig(amp_dtype="float8").autocast_dtype()

    def test_an_explicit_device_overrides_autodetection(self):
        assert TrainConfig(device="cpu").resolved_device() == torch.device("cpu")

    def test_to_json_is_serialisable(self):
        payload = TrainConfig(output_dir=Path("runs/x")).to_json()
        assert payload["output_dir"] == "runs/x"
        json.dumps(payload)  # must not raise


class TestCli:
    def test_smoke_test_flag_parses(self):
        assert config_from_args(["--smoke-test"]).smoke_test is True

    def test_defaults_match_the_dataclass(self):
        assert config_from_args([]) == TrainConfig()

    def test_negative_flags_turn_features_off(self):
        config = config_from_args(
            [
                "--no-augment",
                "--no-amp",
                "--no-deltas",
                "--no-balanced-sampling",
                "--no-class-weighted-loss",
            ]
        )
        assert config.augment is False
        assert config.amp is False
        assert config.with_deltas is False
        assert config.balanced_sampling is False
        assert config.class_weighted_loss is False

    def test_lr_alias(self):
        assert config_from_args(["--lr", "0.01"]).learning_rate == 0.01

    def test_paths_are_parsed_as_paths(self):
        config = config_from_args(["--output-dir", "runs/abc"])
        assert config.output_dir == Path("runs/abc")

    def test_an_unknown_frontend_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            config_from_args(["--frontend", "mfcc"])

    def test_every_config_field_the_parser_sets_exists(self):
        parsed = vars(build_parser().parse_args([]))
        unknown = set(parsed) - set(TrainConfig.__dataclass_fields__)
        assert unknown == set()


class TestSeeding:
    def test_is_reproducible(self):
        seed_everything(123)
        first = torch.randn(5)
        seed_everything(123)
        assert torch.equal(first, torch.randn(5))

    def test_different_seeds_differ(self):
        seed_everything(1)
        first = torch.randn(5)
        seed_everything(2)
        assert not torch.equal(first, torch.randn(5))

    def test_deterministic_mode_sets_the_cudnn_flags(self):
        seed_everything(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        # Leave the process as the default path found it.
        seed_everything(0, deterministic=False)
        torch.use_deterministic_algorithms(False)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_returns_metrics_and_raw_scores(self):
        model = CountermeasureModel()
        loader = _loader()
        result, raw = evaluate(model, loader, torch.device("cpu"))
        assert 0.0 <= result.eer <= 1.0
        assert result.min_tdcf is None
        assert len(raw["score"]) == len(raw["utterance_id"]) == 16
        assert set(raw["target"]) == {0, 1}

    def test_adds_tdcf_when_given_an_asv_operating_point(self):
        model = CountermeasureModel()
        result, _ = evaluate(
            model,
            _loader(),
            torch.device("cpu"),
            asv=AsvErrorRates(p_miss=0.05, p_fa=0.01, p_miss_spoof=0.9),
        )
        assert result.min_tdcf is not None
        assert 0.0 <= result.min_tdcf <= 1.0

    def test_breaks_scores_down_by_attack(self):
        model = CountermeasureModel()
        result, _ = evaluate(model, _loader(), torch.device("cpu"))
        assert result.per_attack_eer is not None
        assert "SYN" in result.per_attack_eer

    def test_leaves_the_model_in_eval_mode(self):
        model = CountermeasureModel()
        model.train()
        evaluate(model, _loader(), torch.device("cpu"))
        assert model.training is False

    def test_max_batches_truncates_the_pass(self):
        model = CountermeasureModel()
        # 32 items, 16 bonafide then 16 spoof, batch 8: three batches is the
        # fewest that still covers both classes.
        _, raw = evaluate(
            model, _loader(n_items=32), torch.device("cpu"), max_batches=3
        )
        assert len(raw["score"]) == 24

    def test_a_single_class_pass_is_a_loud_failure(self):
        """An EER computed over one class is meaningless, so it must not be
        silently reported as 0.0."""

        model = CountermeasureModel()
        # The synthetic set is ordered bonafide-then-spoof, so one batch of the
        # first half is bonafide only.
        with pytest.raises(ValueError, match="needs both classes"):
            evaluate(model, _loader(n_items=16), torch.device("cpu"), max_batches=1)

    def test_is_deterministic(self):
        model = CountermeasureModel()
        first, _ = evaluate(model, _loader(), torch.device("cpu"))
        second, _ = evaluate(model, _loader(), torch.device("cpu"))
        assert first.eer == second.eer


# ---------------------------------------------------------------------------
# Trainer mechanics
# ---------------------------------------------------------------------------


class TestTrainerMechanics:
    def _trainer(self, tmp_path, **overrides):
        config = _smoke_config(tmp_path, **overrides)
        return Trainer(
            config,
            CountermeasureModel(config.feature_config(), config.model_config()),
            _loader(seed=0),
            _loader(seed=1),
            class_weights=(1.0, 1.0),
        )

    def test_warmup_then_cosine_decay(self, tmp_path):
        trainer = self._trainer(tmp_path, epochs=10, warmup_epochs=2)
        schedule = [trainer._lr_at(epoch) for epoch in range(10)]
        peak = trainer.config.learning_rate
        assert schedule[0] == pytest.approx(peak / 2)
        assert schedule[1] == pytest.approx(peak)
        # After warmup it only decays. The first pair spans the warmup ramp,
        # which rises, so the check starts at index 1.
        assert all(a >= b for a, b in itertools.pairwise(schedule[1:]))
        assert schedule[-1] < peak

    def test_no_warmup_starts_at_the_peak(self, tmp_path):
        trainer = self._trainer(tmp_path, epochs=10, warmup_epochs=0)
        assert trainer._lr_at(0) == pytest.approx(trainer.config.learning_rate)

    def test_the_scaler_is_off_for_bfloat16(self, tmp_path):
        """bf16 has fp32's exponent range, so gradients do not underflow and
        loss scaling is pure overhead."""

        trainer = self._trainer(tmp_path, amp=True, amp_dtype="bfloat16")
        assert trainer.scaler.is_enabled() is False

    def test_the_scaler_is_off_on_cpu_even_for_float16(self, tmp_path):
        trainer = self._trainer(tmp_path, amp=True, amp_dtype="float16")
        assert trainer.amp_enabled is False
        assert trainer.scaler.is_enabled() is False

    def test_a_training_epoch_returns_its_statistics(self, tmp_path):
        trainer = self._trainer(tmp_path)
        stats = trainer.train_epoch()
        assert math.isfinite(stats["loss"])
        assert 0.0 <= stats["accuracy"] <= 1.0
        assert stats["samples"] == 16
        assert stats["seconds"] >= 0.0

    def test_an_epoch_with_clipping_enabled_stays_finite_and_updates(self):
        """Clipping is on the gradient norm, and AdamW then normalises the
        step, so the observable contract is narrow: training still progresses
        and nothing becomes NaN or inf."""

        for clip in (0.0, 1e-6, 5.0):
            config = TrainConfig(device="cpu", amp=False, grad_clip=clip, log_every=0)
            trainer = Trainer(
                config,
                CountermeasureModel(),
                _loader(),
                _loader(),
                class_weights=(1.0, 1.0),
            )
            before = [p.detach().clone() for p in trainer.model.parameters()]
            stats = trainer.train_epoch()
            assert math.isfinite(stats["loss"])
            after = list(trainer.model.parameters())
            assert all(torch.isfinite(p).all() for p in after)
            assert any(
                not torch.equal(a, b.detach())
                for a, b in zip(before, after, strict=False)
            )

    def test_a_non_finite_loss_stops_the_run(self, tmp_path):
        """Training on through a NaN loss poisons every weight, so the loop
        refuses rather than continuing."""

        trainer = self._trainer(tmp_path)

        class Exploding(torch.nn.Module):
            def forward(self, logits, target):
                return logits.sum() * float("inf")

        trainer.criterion = Exploding()
        with pytest.raises(RuntimeError, match="stopping rather than"):
            trainer.train_epoch()

    def test_an_empty_training_loader_is_a_loud_failure(self, tmp_path):
        trainer = self._trainer(tmp_path, max_train_batches=0)
        with pytest.raises(RuntimeError, match="no batches"):
            trainer.train_epoch()

    def test_the_class_weighted_loss_is_wired_up(self, tmp_path):
        trainer = self._trainer(tmp_path)
        weighted = Trainer(
            trainer.config,
            CountermeasureModel(),
            _loader(),
            _loader(),
            class_weights=(0.5, 2.0),
        )
        assert weighted.criterion.weight is not None
        assert weighted.criterion.weight.tolist() == [0.5, 2.0]

    def test_class_weighting_can_be_switched_off(self, tmp_path):
        config = _smoke_config(tmp_path, class_weighted_loss=False)
        trainer = Trainer(
            config,
            CountermeasureModel(),
            _loader(),
            _loader(),
            class_weights=(0.5, 2.0),
        )
        assert trainer.criterion.weight is None


class TestEarlyStopping:
    def test_improvement_resets_the_counter(self, tmp_path):
        from training.train import _EarlyStopping

        stopper = _EarlyStopping(patience=2)
        assert stopper.update(0.5, 0) is True
        assert stopper.update(0.6, 1) is False
        assert stopper.update(0.4, 2) is True
        assert stopper.epochs_without_improvement == 0
        assert stopper.best_epoch == 2
        assert not stopper.should_stop

    def test_fires_after_the_patience_is_exhausted(self):
        from training.train import _EarlyStopping

        stopper = _EarlyStopping(patience=2)
        stopper.update(0.5, 0)
        stopper.update(0.6, 1)
        assert not stopper.should_stop
        stopper.update(0.7, 2)
        assert stopper.should_stop

    def test_lower_is_better(self):
        from training.train import _EarlyStopping

        stopper = _EarlyStopping(patience=5)
        stopper.update(0.3, 0)
        stopper.update(0.9, 1)
        assert stopper.best == 0.3


# ---------------------------------------------------------------------------
# The smoke test itself
# ---------------------------------------------------------------------------


class TestSmokeRun:
    def test_runs_end_to_end_and_writes_its_artefacts(self, tmp_path):
        """This is the test that proves the whole pipeline is wired: front end,
        model, loss, AMP wiring, metrics, checkpointing and score file, with no
        dataset on disk."""

        result = run(_smoke_config(tmp_path))

        assert result["epochs_run"] == 2
        assert 0.0 <= result["best_dev_eer"] <= 1.0
        assert result["best_epoch"] in (0, 1)

        assert (tmp_path / "last.pt").exists()
        assert (tmp_path / "best.pt").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "history.json").exists()
        assert (tmp_path / "dev_scores.txt").exists()

    def test_the_score_file_is_readable_by_the_metrics_reader(self, tmp_path):
        """The score file is the artefact the official ASVspoof evaluation
        package consumes, so it has to parse."""

        run(_smoke_config(tmp_path))
        scores = read_score_file(tmp_path / "dev_scores.txt")
        assert len(scores) == 32
        assert all(key.startswith("SYN_") for key in scores)
        assert all(value <= 0.0 for value in scores.values())

    def test_history_records_one_entry_per_epoch(self, tmp_path):
        run(_smoke_config(tmp_path))
        history = json.loads((tmp_path / "history.json").read_text())
        assert len(history) == 2
        assert [record["epoch"] for record in history] == [0, 1]
        assert all(math.isfinite(record["train_loss"]) for record in history)
        assert all(0.0 <= record["dev_eer"] <= 1.0 for record in history)

    def test_the_saved_config_describes_the_run(self, tmp_path):
        run(_smoke_config(tmp_path))
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["smoke_test"] is True
        assert config["frontend"] == "lfcc"

    def test_the_checkpoint_is_self_describing(self, tmp_path):
        run(_smoke_config(tmp_path))
        payload = torch.load(tmp_path / "best.pt", weights_only=False)
        assert payload["format_version"] == 1
        assert "model" in payload
        assert payload["model_config"]["feature"]["name"] == "lfcc"
        restored = CountermeasureModel.from_config_dict(payload["model_config"])
        restored.load_state_dict(payload["model"])

    def test_the_loss_falls_on_separable_synthetic_data(self, tmp_path):
        """A smoke run that only proves nothing crashed is weak evidence. The
        synthetic classes are separable, so the loss must actually drop."""

        result = run(_smoke_config(tmp_path, epochs=4))
        losses = [record["train_loss"] for record in result["history"]]
        assert losses[-1] < losses[0]

    def test_the_logmel_front_end_also_runs_end_to_end(self, tmp_path):
        result = run(_smoke_config(tmp_path, frontend="logmel", n_mels=40))
        assert result["epochs_run"] == 2

    def test_tdcf_is_reported_when_asv_rates_are_supplied(self, tmp_path):
        result = run(
            _smoke_config(
                tmp_path,
                asv_p_miss=0.05,
                asv_p_fa=0.01,
                asv_p_miss_spoof=0.9,
            )
        )
        assert result["final_dev"]["min_tdcf"] is not None

    def test_main_returns_zero_for_the_cli_smoke_path(self, tmp_path):
        exit_code = main(
            [
                "--smoke-test",
                "--output-dir",
                str(tmp_path),
                "--device",
                "cpu",
                "--no-amp",
            ]
        )
        assert exit_code == 0
        assert (tmp_path / "best.pt").exists()

    def test_main_reports_missing_data_without_a_traceback(self, tmp_path):
        """A real run against an unpacked-nothing data dir must explain itself
        and exit 1, not raise."""

        exit_code = main(
            [
                "--data-dir",
                str(tmp_path / "empty"),
                "--output-dir",
                str(tmp_path / "out"),
                "--device",
                "cpu",
            ]
        )
        assert exit_code == 1


class TestResume:
    def test_resuming_restores_the_epoch_and_the_best_score(self, tmp_path):
        first = run(_smoke_config(tmp_path, epochs=2))

        resumed_config = _smoke_config(tmp_path, epochs=4, resume=tmp_path / "last.pt")
        second = run(resumed_config)

        # Epochs 2 and 3 only: the first two are not repeated.
        assert [record["epoch"] for record in second["history"]][-2:] == [2, 3]
        assert second["best_dev_eer"] <= first["best_dev_eer"]

    def test_resuming_restores_the_optimizer_state(self, tmp_path):
        config = _smoke_config(tmp_path, epochs=1)
        model = CountermeasureModel(config.feature_config(), config.model_config())
        trainer = Trainer(config, model, _loader(), _loader(), class_weights=(1.0, 1.0))
        trainer.train_epoch()
        trainer.save_checkpoint("ckpt.pt")
        saved = trainer.optimizer.state_dict()["state"]

        fresh = Trainer(
            config,
            CountermeasureModel(config.feature_config(), config.model_config()),
            _loader(),
            _loader(),
            class_weights=(1.0, 1.0),
        )
        assert fresh.optimizer.state_dict()["state"] == {}
        fresh.load_checkpoint(tmp_path / "ckpt.pt")

        restored = fresh.optimizer.state_dict()["state"]
        assert set(restored) == set(saved)
        assert fresh.epoch == 1

    def test_model_only_load_leaves_the_optimizer_untouched(self, tmp_path):
        config = _smoke_config(tmp_path, epochs=1)
        trainer = Trainer(
            config,
            CountermeasureModel(config.feature_config(), config.model_config()),
            _loader(),
            _loader(),
            class_weights=(1.0, 1.0),
        )
        trainer.train_epoch()
        trainer.save_checkpoint("ckpt.pt")

        fresh = Trainer(
            config,
            CountermeasureModel(config.feature_config(), config.model_config()),
            _loader(),
            _loader(),
            class_weights=(1.0, 1.0),
        )
        fresh.load_checkpoint(tmp_path / "ckpt.pt", model_only=True)
        assert fresh.optimizer.state_dict()["state"] == {}
        assert fresh.epoch == 0

    def test_a_checkpoint_save_is_atomic(self, tmp_path):
        """Written to a temporary file and renamed, so an interrupted save
        cannot leave a truncated checkpoint where the good one was."""

        config = _smoke_config(tmp_path, epochs=1)
        trainer = Trainer(
            config,
            CountermeasureModel(config.feature_config(), config.model_config()),
            _loader(),
            _loader(),
            class_weights=(1.0, 1.0),
        )
        path = trainer.save_checkpoint("ckpt.pt")
        assert path.exists()
        assert not path.with_suffix(path.suffix + ".tmp").exists()
