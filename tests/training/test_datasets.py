"""Dataset, cropping and augmentation tests.

The audio loader is injected throughout, so nothing here reads a file from
disk except the few cases that write their own WAV-free synthetic "audio" via a
fake loader. Skipped wholesale when torch is absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from training.datasets import (  # noqa: E402  (after importorskip by design)
    AugmentChain,
    CropMode,
    DatasetConfig,
    MusanNoiseAugment,
    RirReverbAugment,
    SyntheticWaveformDataset,
    UtteranceDataset,
    balanced_sampler,
    build_augmentation,
    collate_batch,
    dataset_for_subset,
    fixed_length_crop,
    repeat_pad,
)
from training.manifest import Label, Manifest, Subset, Utterance  # noqa: E402

CROP = 16_000


def _utterances(n_bonafide: int = 2, n_spoof: int = 3, subset=Subset.TRAIN):
    items = [
        Utterance(f"B{i}", Path(f"B{i}.flac"), Label.BONAFIDE, subset)
        for i in range(n_bonafide)
    ]
    items += [
        Utterance(f"S{i}", Path(f"S{i}.flac"), Label.SPOOF, subset, f"A0{i % 3 + 1}")
        for i in range(n_spoof)
    ]
    return Manifest(tuple(items))


def _loader(samples: int = 20_000, rate: int = 16_000):
    """A fake audio loader: deterministic per-path pseudo-audio, no files."""

    def load(path: Path):
        seed = abs(hash(str(path))) % (2**31)
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(samples, generator=generator), rate

    return load


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


class TestRepeatPad:
    def test_tiles_a_short_waveform_to_exact_length(self):
        padded = repeat_pad(torch.tensor([1.0, 2.0, 3.0]), 8)
        assert padded.tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]

    def test_truncates_a_long_waveform(self):
        assert repeat_pad(torch.arange(10.0), 4).tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_never_introduces_silence(self):
        """Zero padding would teach the model that silence predicts "short
        file", and in ASVspoof 2019 LA file length correlates with class."""

        padded = repeat_pad(torch.full((100,), 0.5), 1000)
        assert (padded == 0.5).all()

    def test_an_empty_waveform_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            repeat_pad(torch.tensor([]), 10)

    def test_a_multidimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            repeat_pad(torch.randn(2, 10), 5)


class TestFixedLengthCrop:
    @pytest.mark.parametrize("mode", [CropMode.RANDOM, CropMode.CENTRE])
    @pytest.mark.parametrize("samples", [500, 4000, 4096, 10_000])
    def test_always_returns_the_requested_length(self, mode, samples):
        cropped = fixed_length_crop(torch.randn(samples), 4096, mode)
        assert cropped.shape == (4096,)

    def test_centre_crop_is_centred(self):
        cropped = fixed_length_crop(torch.arange(10.0), 4, CropMode.CENTRE)
        assert cropped.tolist() == [3.0, 4.0, 5.0, 6.0]

    def test_centre_crop_is_deterministic(self):
        waveform = torch.randn(10_000)
        first = fixed_length_crop(waveform, 4000, CropMode.CENTRE)
        second = fixed_length_crop(waveform, 4000, CropMode.CENTRE)
        assert torch.equal(first, second)

    def test_random_crop_varies_with_the_generator(self):
        import random as _random

        waveform = torch.arange(10_000, dtype=torch.float32)
        crops = {
            tuple(
                fixed_length_crop(waveform, 100, CropMode.RANDOM, _random.Random(seed))[
                    :1
                ].tolist()
            )
            for seed in range(20)
        }
        assert len(crops) > 1

    def test_random_crop_is_reproducible_for_a_given_seed(self):
        import random as _random

        waveform = torch.arange(10_000, dtype=torch.float32)
        first = fixed_length_crop(waveform, 100, CropMode.RANDOM, _random.Random(42))
        second = fixed_length_crop(waveform, 100, CropMode.RANDOM, _random.Random(42))
        assert torch.equal(first, second)

    def test_a_crop_is_a_contiguous_slice_of_the_source(self):
        import random as _random

        waveform = torch.arange(1000, dtype=torch.float32)
        cropped = fixed_length_crop(waveform, 50, CropMode.RANDOM, _random.Random(1))
        gaps = cropped[1:] - cropped[:-1]
        assert torch.allclose(gaps, torch.ones_like(gaps))

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown crop mode"):
            fixed_length_crop(torch.randn(1000), 100, "sideways")

    def test_a_non_positive_length_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            fixed_length_crop(torch.randn(100), 0)


# ---------------------------------------------------------------------------
# Augmentation, present and absent
# ---------------------------------------------------------------------------


class TestAugmentationIsOptional:
    def test_musan_is_disabled_when_the_directory_is_none(self):
        assert not MusanNoiseAugment(None).enabled

    def test_musan_is_disabled_when_the_directory_is_missing(self, tmp_path):
        assert not MusanNoiseAugment(tmp_path / "nope").enabled

    def test_musan_is_disabled_when_the_directory_holds_no_audio(self, tmp_path):
        (tmp_path / "noise").mkdir()
        assert not MusanNoiseAugment(tmp_path).enabled

    def test_rir_is_disabled_when_the_directory_is_none(self):
        assert not RirReverbAugment(None).enabled

    def test_rir_is_disabled_when_the_directory_is_missing(self, tmp_path):
        assert not RirReverbAugment(tmp_path / "nope").enabled

    @pytest.mark.parametrize("builder", [MusanNoiseAugment, RirReverbAugment])
    def test_a_disabled_augmentation_is_the_identity(self, builder):
        augment = builder(None)
        waveform = torch.randn(1000)
        assert torch.equal(augment(waveform), waveform)

    def test_the_default_chain_is_disabled_with_no_corpora(self):
        chain = build_augmentation(None, None)
        assert not chain.enabled
        waveform = torch.randn(1000)
        assert torch.equal(chain(waveform), waveform)

    def test_zero_probability_disables_an_otherwise_present_corpus(self, tmp_path):
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        (noise_dir / "a.wav").write_bytes(b"not really audio")
        assert not MusanNoiseAugment(tmp_path, probability=0.0).enabled


class TestAugmentationWhenPresent:
    @staticmethod
    def _fake_corpus(tmp_path: Path, subdir: str, count: int = 3) -> Path:
        directory = tmp_path / subdir
        directory.mkdir(parents=True)
        for index in range(count):
            # Content is never read: the loader is injected.
            (directory / f"{index}.wav").write_bytes(b"placeholder")
        return tmp_path

    def test_musan_activates_and_changes_the_waveform(self, tmp_path):
        root = self._fake_corpus(tmp_path, "noise")
        augment = MusanNoiseAugment(
            root, probability=1.0, loader=_loader(samples=5_000)
        )
        assert augment.enabled
        waveform = torch.randn(CROP)
        assert not torch.equal(augment(waveform), waveform)

    def test_musan_preserves_length(self, tmp_path):
        root = self._fake_corpus(tmp_path, "noise")
        augment = MusanNoiseAugment(
            root, probability=1.0, loader=_loader(samples=5_000)
        )
        assert augment(torch.randn(CROP)).shape == (CROP,)

    def test_musan_respects_the_requested_snr(self, tmp_path):
        """Mix at a fixed SNR and measure the residual's power against the
        signal's. A 10 dB request should put the noise 10 dB down."""

        root = self._fake_corpus(tmp_path, "noise")
        augment = MusanNoiseAugment(
            root,
            probability=1.0,
            snr_range=(10.0, 10.0),
            loader=_loader(samples=CROP),
        )
        waveform = torch.randn(CROP)
        mixed = augment(waveform)
        residual = mixed - waveform
        measured = 10 * math.log10(
            waveform.pow(2).mean().item() / residual.pow(2).mean().item()
        )
        assert measured == pytest.approx(10.0, abs=0.5)

    def test_musan_skips_the_speech_category_by_default(self, tmp_path):
        """Mixing MUSAN speech into a single-speaker clip makes babble, which
        the label no longer describes."""

        self._fake_corpus(tmp_path, "speech")
        assert not MusanNoiseAugment(tmp_path, probability=1.0).enabled

    def test_musan_survives_an_unreadable_noise_file(self, tmp_path):
        root = self._fake_corpus(tmp_path, "noise")

        def exploding(path):
            raise OSError("corrupt")

        augment = MusanNoiseAugment(root, probability=1.0, loader=exploding)
        waveform = torch.randn(1000)
        assert torch.equal(augment(waveform), waveform)

    def test_rir_activates_and_changes_the_waveform(self, tmp_path):
        root = self._fake_corpus(tmp_path, "simulated_rirs")
        augment = RirReverbAugment(root, probability=1.0, loader=_loader(samples=500))
        assert augment.enabled
        waveform = torch.randn(CROP)
        assert not torch.equal(augment(waveform), waveform)

    def test_rir_preserves_length_and_loudness(self, tmp_path):
        root = self._fake_corpus(tmp_path, "simulated_rirs")
        augment = RirReverbAugment(root, probability=1.0, loader=_loader(samples=500))
        waveform = torch.randn(CROP)
        reverbed = augment(waveform)
        assert reverbed.shape == (CROP,)
        source_rms = waveform.pow(2).mean().sqrt()
        result_rms = reverbed.pow(2).mean().sqrt()
        assert result_rms.item() == pytest.approx(source_rms.item(), rel=1e-4)

    def test_a_unit_impulse_leaves_the_waveform_untouched(self, tmp_path):
        """FFT convolution must agree with the time-domain definition, or the
        reverb is just noise. Convolving with a delta at lag 0 is the identity,
        and the RMS restoration is exactly 1, so this is an exact check of the
        whole convolution path."""

        root = self._fake_corpus(tmp_path, "simulated_rirs", count=1)

        def delta(path):
            return torch.tensor([1.0, 0.0, 0.0, 0.0]), 16_000

        augment = RirReverbAugment(root, probability=1.0, loader=delta)
        waveform = torch.randn(4_096)
        assert torch.allclose(augment(waveform), waveform, atol=1e-5)

    def test_a_delayed_impulse_delays_the_waveform(self, tmp_path):
        """A delta at lag 1 is a one-sample delay. The RMS restoration rescales
        by a factor that differs from 1 only by the one sample shifted off the
        end, which is negligible over 4096 samples."""

        root = self._fake_corpus(tmp_path, "simulated_rirs", count=1)

        def delayed_delta(path):
            return torch.tensor([0.0, 1.0, 0.0, 0.0]), 16_000

        augment = RirReverbAugment(root, probability=1.0, loader=delayed_delta)
        waveform = torch.randn(4_096)
        result = augment(waveform)
        expected = torch.cat([torch.zeros(1), waveform[:-1]])
        assert torch.allclose(result, expected, atol=1e-2)
        assert result[0].abs().item() < 1e-3

    def test_rir_survives_an_unreadable_file(self, tmp_path):
        root = self._fake_corpus(tmp_path, "simulated_rirs")

        def exploding(path):
            raise OSError("corrupt")

        augment = RirReverbAugment(root, probability=1.0, loader=exploding)
        waveform = torch.randn(1000)
        assert torch.equal(augment(waveform), waveform)


class TestAugmentChain:
    def test_is_enabled_when_any_member_is(self, tmp_path):
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        (noise_dir / "a.wav").write_bytes(b"x")
        chain = AugmentChain(
            [
                RirReverbAugment(None),
                MusanNoiseAugment(
                    tmp_path, probability=1.0, loader=_loader(samples=1000)
                ),
            ]
        )
        assert chain.enabled

    def test_applies_members_in_order(self):
        calls: list[str] = []

        class Recorder:
            def __init__(self, name):
                self.name = name
                self.enabled = True

            def __call__(self, waveform, generator=None):
                calls.append(self.name)
                return waveform

        AugmentChain([Recorder("first"), Recorder("second")])(torch.randn(10))
        assert calls == ["first", "second"]

    def test_the_default_chain_reverberates_before_adding_noise(self):
        """A room colours the voice, then the microphone picks up ambient
        noise that was never in the room's impulse response."""

        chain = build_augmentation(None, None)
        assert isinstance(chain.augmentations[0], RirReverbAugment)
        assert isinstance(chain.augmentations[1], MusanNoiseAugment)


# ---------------------------------------------------------------------------
# UtteranceDataset
# ---------------------------------------------------------------------------


class TestUtteranceDataset:
    def test_length_matches_the_manifest(self):
        dataset = UtteranceDataset(_utterances(), loader=_loader())
        assert len(dataset) == 5

    def test_an_empty_manifest_is_rejected(self):
        with pytest.raises(ValueError, match="empty manifest"):
            UtteranceDataset(Manifest(()), loader=_loader())

    def test_item_keys_and_shapes(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        item = dataset[0]
        assert set(item) == {
            "waveform",
            "target",
            "utterance_id",
            "attack_id",
            "index",
        }
        assert item["waveform"].shape == (CROP,)
        assert item["waveform"].dtype == torch.float32
        assert item["target"].dtype == torch.int64

    def test_targets_follow_the_label(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        assert dataset[0]["target"].item() == 1  # bonafide
        assert dataset[4]["target"].item() == 0  # spoof

    def test_bonafide_attack_id_collates_as_a_dash(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        assert dataset[0]["attack_id"] == "-"
        assert dataset[4]["attack_id"].startswith("A")

    def test_eval_mode_is_deterministic_across_repeated_reads(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        assert torch.equal(dataset[0]["waveform"], dataset[0]["waveform"])

    def test_eval_mode_ignores_the_epoch(self):
        """A dev EER that wobbles between epochs for crop reasons is not
        measuring the model."""

        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        first = dataset[0]["waveform"].clone()
        dataset.set_epoch(7)
        assert torch.equal(dataset[0]["waveform"], first)

    def test_train_mode_changes_the_crop_between_epochs(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=1_000, train=True),
            loader=_loader(samples=50_000),
        )
        crops = set()
        for epoch in range(10):
            dataset.set_epoch(epoch)
            crops.add(tuple(dataset[0]["waveform"][:3].tolist()))
        assert len(crops) > 1

    def test_train_mode_is_reproducible_for_a_given_seed_and_epoch(self):
        def build(seed):
            dataset = UtteranceDataset(
                _utterances(),
                DatasetConfig(crop_samples=1_000, train=True),
                loader=_loader(samples=50_000),
                seed=seed,
            )
            dataset.set_epoch(3)
            return dataset[0]["waveform"]

        assert torch.equal(build(11), build(11))
        assert not torch.equal(build(11), build(12))

    def test_peak_normalisation_bounds_the_waveform(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False, peak_normalise=True),
            loader=_loader(),
        )
        assert dataset[0]["waveform"].abs().max().item() == pytest.approx(1.0)

    def test_a_sample_rate_mismatch_is_a_loud_failure(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, sample_rate=16_000, train=False),
            loader=_loader(rate=8_000),
        )
        with pytest.raises(ValueError, match="8000 Hz but the config expects"):
            dataset[0]

    def test_the_rate_check_can_be_waived(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False, resample_check=False),
            loader=_loader(rate=8_000),
        )
        assert dataset[0]["waveform"].shape == (CROP,)

    def test_augmentation_runs_in_train_mode_only(self):
        seen: list[int] = []

        class Counter:
            enabled = True

            def __call__(self, waveform, generator=None):
                seen.append(1)
                return waveform

        for train in (True, False):
            seen.clear()
            dataset = UtteranceDataset(
                _utterances(),
                DatasetConfig(crop_samples=CROP, train=train),
                augmentation=Counter(),
                loader=_loader(),
            )
            dataset[0]
            assert len(seen) == (1 if train else 0)

    def test_a_length_changing_augmentation_is_corrected(self):
        class Shrinker:
            enabled = True

            def __call__(self, waveform, generator=None):
                return waveform[:100]

        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=True),
            augmentation=Shrinker(),
            loader=_loader(),
        )
        assert dataset[0]["waveform"].shape == (CROP,)

    def test_crop_seconds_is_reported(self):
        config = DatasetConfig(crop_samples=64_600, sample_rate=16_000)
        assert config.crop_seconds == pytest.approx(4.0375)

    def test_crop_mode_follows_the_train_flag(self):
        assert DatasetConfig(train=True).crop_mode == CropMode.RANDOM
        assert DatasetConfig(train=False).crop_mode == CropMode.CENTRE


class TestDatasetForSubset:
    def test_train_split_gets_random_crops_and_augmentation(self):
        manifest = _utterances() + _utterances(1, 1, Subset.DEV)
        dataset = dataset_for_subset(
            manifest,
            Subset.TRAIN,
            crop_samples=CROP,
            augmentation=build_augmentation(None, None),
            loader=_loader(),
        )
        assert len(dataset) == 5
        assert dataset.config.train is True
        assert dataset.augmentation is not None

    def test_dev_split_is_deterministic_and_unaugmented(self):
        manifest = _utterances() + _utterances(1, 1, Subset.DEV)
        dataset = dataset_for_subset(
            manifest,
            Subset.DEV,
            crop_samples=CROP,
            augmentation=build_augmentation(None, None),
            loader=_loader(),
        )
        assert len(dataset) == 2
        assert dataset.config.train is False
        assert dataset.augmentation is None


# ---------------------------------------------------------------------------
# Synthetic dataset, sampler, collate
# ---------------------------------------------------------------------------


class TestSyntheticWaveformDataset:
    def test_shape_and_keys(self):
        dataset = SyntheticWaveformDataset(n_items=8, crop_samples=4_000)
        item = dataset[0]
        assert item["waveform"].shape == (4_000,)
        assert set(item) == {
            "waveform",
            "target",
            "utterance_id",
            "attack_id",
            "index",
        }

    def test_holds_both_classes(self):
        dataset = SyntheticWaveformDataset(n_items=8, crop_samples=1_000)
        targets = {dataset[i]["target"].item() for i in range(len(dataset))}
        assert targets == {0, 1}

    def test_never_degenerates_to_one_class(self):
        for fraction in (0.0, 1.0):
            dataset = SyntheticWaveformDataset(
                n_items=4, crop_samples=500, bonafide_fraction=fraction
            )
            targets = {dataset[i]["target"].item() for i in range(4)}
            assert targets == {0, 1}

    def test_is_deterministic_for_a_given_seed(self):
        a = SyntheticWaveformDataset(n_items=4, crop_samples=500, seed=5)
        b = SyntheticWaveformDataset(n_items=4, crop_samples=500, seed=5)
        assert torch.equal(a[2]["waveform"], b[2]["waveform"])

    def test_different_seeds_give_different_audio(self):
        a = SyntheticWaveformDataset(n_items=4, crop_samples=500, seed=5)
        b = SyntheticWaveformDataset(n_items=4, crop_samples=500, seed=6)
        assert not torch.equal(a[2]["waveform"], b[2]["waveform"])

    def test_the_classes_are_separable_so_a_smoke_run_shows_learning(self):
        """A smoke test whose EER stays at 0.5 proves only that nothing
        crashed, so the synthetic classes differ by construction."""

        dataset = SyntheticWaveformDataset(
            n_items=8, crop_samples=4_000, separable=True
        )
        bonafide = [
            dataset[i]["waveform"] for i in range(8) if dataset[i]["target"].item() == 1
        ]
        spoof = [
            dataset[i]["waveform"] for i in range(8) if dataset[i]["target"].item() == 0
        ]

        # The spoof class carries a 3 kHz tone, so its spectrum is peakier.
        def peakiness(waveform):
            spectrum = torch.fft.rfft(waveform).abs()
            return (spectrum.max() / spectrum.mean()).item()

        assert min(peakiness(w) for w in spoof) > max(peakiness(w) for w in bonafide)

    def test_a_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            SyntheticWaveformDataset(n_items=0)


class TestBalancedSampler:
    def test_oversamples_the_minority_class(self):
        manifest = _utterances(n_bonafide=1, n_spoof=9)
        sampler = balanced_sampler(manifest, generator=torch.Generator().manual_seed(0))
        drawn = list(sampler)
        assert len(drawn) == 10
        bonafide_draws = sum(1 for i in drawn if manifest[i].is_bonafide)
        # Unweighted, the expectation would be 1 in 10.
        assert bonafide_draws >= 2

    def test_is_reproducible_for_a_given_generator_seed(self):
        manifest = _utterances(n_bonafide=2, n_spoof=8)
        first = list(
            balanced_sampler(manifest, generator=torch.Generator().manual_seed(7))
        )
        second = list(
            balanced_sampler(manifest, generator=torch.Generator().manual_seed(7))
        )
        assert first == second


class TestCollateBatch:
    def test_stacks_tensors_and_keeps_strings_as_lists(self):
        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=CROP, train=False),
            loader=_loader(),
        )
        batch = collate_batch([dataset[0], dataset[1], dataset[4]])
        assert batch["waveform"].shape == (3, CROP)
        assert batch["target"].shape == (3,)
        assert batch["index"].tolist() == [0, 1, 4]
        assert batch["utterance_id"] == ["B0", "B1", "S2"]
        assert isinstance(batch["attack_id"], list)

    def test_works_as_a_dataloader_collate_fn(self):
        from torch.utils.data import DataLoader

        dataset = UtteranceDataset(
            _utterances(),
            DatasetConfig(crop_samples=4_000, train=False),
            loader=_loader(),
        )
        loader = DataLoader(
            dataset, batch_size=2, collate_fn=collate_batch, num_workers=0
        )
        batch = next(iter(loader))
        assert batch["waveform"].shape == (2, 4_000)
        assert len(batch["utterance_id"]) == 2
