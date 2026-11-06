"""Torch datasets over a `Manifest`, with optional MUSAN and RIR augmentation.

Design points that matter:

* **Fixed-length crops.** Countermeasures are trained on a fixed number of
  samples (64600 at 16 kHz, about 4.03 s, is the ASVspoof baseline length), so
  a batch is a plain stacked tensor. Train mode takes a random crop, eval mode
  a deterministic centre crop, so a dev EER does not wobble between epochs for
  reasons that have nothing to do with the model.
* **Repeat padding, not zero padding.** Short utterances are tiled to length,
  which is what the ASVspoof baselines do. Zero padding teaches the model that
  silence predicts "short file", and in ASVspoof 2019 LA file length correlates
  with class, so that shortcut is worth a lot of EER and generalises to
  nothing.
* **Augmentation is an optional hook that no-ops when the corpus is absent.**
  MUSAN and RIRS_NOISES are 11 GB of downloads that may not have landed. Both
  augmenters construct fine with a missing directory, report
  `enabled == False`, and pass audio through untouched. Nothing in the training
  path branches on whether the data exists.
* **Audio loading is injectable.** The default loader tries soundfile then
  torchaudio and raises a clear error if neither is installed; tests inject a
  synthetic loader, so the whole dataset path is exercised with no audio files
  at all.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler

from training.manifest import Label, Manifest, Subset, Utterance

__all__ = [
    "AudioLoader",
    "AugmentChain",
    "Augmentation",
    "CropMode",
    "DatasetConfig",
    "MusanNoiseAugment",
    "RirReverbAugment",
    "SyntheticWaveformDataset",
    "UtteranceDataset",
    "balanced_sampler",
    "build_augmentation",
    "collate_batch",
    "dataset_for_subset",
    "default_audio_loader",
    "fixed_length_crop",
    "repeat_pad",
]

# ASVspoof baseline crop: 64600 samples at 16 kHz.
DEFAULT_CROP_SAMPLES = 64_600


class CropMode:
    """Crop strategy names. A class rather than an Enum so the strings stay
    usable directly in argparse and config files."""

    RANDOM = "random"
    CENTRE = "centre"


class AudioLoader(Protocol):
    """Reads a path, returns `(waveform (T,), sample_rate)` as float32 mono."""

    def __call__(self, path: Path) -> tuple[Tensor, int]: ...


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def default_audio_loader(path: Path) -> tuple[Tensor, int]:
    """Read one audio file to a float32 mono tensor.

    Tries soundfile first (it reads FLAC and WAV, it is a small wheel, and it
    does not have to match the torch build), then torchaudio. Raises an
    actionable error when neither is installed rather than failing deep inside
    a DataLoader worker.
    """

    try:
        import soundfile  # type: ignore[import-not-found]
    except ImportError:
        soundfile = None  # type: ignore[assignment]

    if soundfile is not None:
        data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data).mean(dim=1)
        return waveform.contiguous(), int(sample_rate)

    try:
        import torchaudio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "no audio backend available. Install soundfile (preferred: it is "
            "independent of the torch build) or torchaudio:\n"
            "    uv add soundfile\n"
            f"while trying to read {path}"
        ) from exc

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform.mean(dim=0).contiguous(), int(sample_rate)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def repeat_pad(waveform: Tensor, length: int) -> Tensor:
    """Tile `waveform` until it is at least `length` samples, then truncate.

    Repeat rather than zero padding: see the module docstring.
    """

    if waveform.dim() != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {tuple(waveform.shape)}")
    samples = waveform.size(0)
    if samples == 0:
        raise ValueError("cannot pad an empty waveform")
    if samples >= length:
        return waveform[:length]
    repeats = math.ceil(length / samples)
    return waveform.repeat(repeats)[:length]


def fixed_length_crop(
    waveform: Tensor,
    length: int,
    mode: str = CropMode.CENTRE,
    generator: random.Random | None = None,
) -> Tensor:
    """Crop or repeat-pad to exactly `length` samples.

    `mode=CropMode.RANDOM` picks a uniform start offset using `generator`
    (defaulting to the module-level `random`, which `train.py` seeds).
    `mode=CropMode.CENTRE` is deterministic: the same file always yields the
    same crop, which is what makes a dev EER comparable across epochs.
    """

    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if waveform.dim() != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {tuple(waveform.shape)}")

    samples = waveform.size(0)
    if samples < length:
        return repeat_pad(waveform, length)
    if samples == length:
        return waveform

    if mode == CropMode.RANDOM:
        rng = generator or random
        start = rng.randint(0, samples - length)
    elif mode == CropMode.CENTRE:
        start = (samples - length) // 2
    else:
        raise ValueError(
            f"unknown crop mode {mode!r}; expected "
            f"{CropMode.RANDOM!r} or {CropMode.CENTRE!r}"
        )
    return waveform[start : start + length]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class Augmentation(Protocol):
    """Waveform in, waveform out, same length. `enabled` says if it does
    anything; a disabled augmentation must be the identity."""

    enabled: bool

    def __call__(
        self, waveform: Tensor, generator: random.Random | None = None
    ) -> Tensor: ...


def _collect_audio(
    root: Path | None,
    patterns: Sequence[str],
    exclude_dirs: Sequence[str] = (),
) -> list[Path]:
    """Recursively glob for audio under `root`. Empty list when absent.

    `exclude_dirs` drops any file with one of those names among its path
    components, so an exclusion survives the recursive fallback glob rather
    than being quietly undone by it.
    """

    if root is None:
        return []
    directory = Path(root)
    if not directory.is_dir():
        return []
    blocked = {name.lower() for name in exclude_dirs}
    found: list[Path] = []
    for pattern in patterns:
        for path in sorted(directory.rglob(pattern)):
            if blocked and blocked & {part.lower() for part in path.parts}:
                continue
            found.append(path)
    return found


class MusanNoiseAugment:
    """Additive noise from MUSAN at a random SNR.

    MUSAN splits into `noise/`, `music/` and `speech/`. Only `noise` and
    `music` are used by default: mixing in MUSAN `speech` creates babble, which
    turns a single-speaker bonafide clip into something the label no longer
    describes. The exclusion is enforced by `exclude_dirs` on the fallback glob
    too, so an unusual unpack layout cannot smuggle speech back in.

    A no-op when the MUSAN directory is missing or holds no usable audio.
    """

    def __init__(
        self,
        musan_dir: Path | str | None,
        *,
        snr_range: tuple[float, float] = (5.0, 20.0),
        probability: float = 0.5,
        categories: Sequence[str] = ("noise", "music"),
        exclude_dirs: Sequence[str] = ("speech",),
        patterns: Sequence[str] = ("*.wav", "*.flac"),
        loader: AudioLoader | None = None,
        max_files: int | None = None,
    ) -> None:
        self.snr_range = snr_range
        self.probability = probability
        self.loader = loader or default_audio_loader

        # Never exclude a directory the caller explicitly asked for.
        wanted = {name.lower() for name in categories}
        excluded = tuple(name for name in exclude_dirs if name.lower() not in wanted)

        files: list[Path] = []
        root = Path(musan_dir) if musan_dir is not None else None
        if root is not None and root.is_dir():
            for category in categories:
                subdir = root / category
                files.extend(
                    _collect_audio(
                        subdir if subdir.is_dir() else None, patterns, excluded
                    )
                )
            if not files:  # flat layout, or category names differ
                files = _collect_audio(root, patterns, excluded)
        self.files = files[:max_files] if max_files else files
        self.enabled = bool(self.files) and probability > 0.0

    def __call__(
        self, waveform: Tensor, generator: random.Random | None = None
    ) -> Tensor:
        if not self.enabled:
            return waveform
        rng = generator or random
        if rng.random() >= self.probability:
            return waveform

        noise_path = self.files[rng.randrange(len(self.files))]
        try:
            noise, _ = self.loader(noise_path)
        except Exception:  # noqa: BLE001 - a bad noise file must not kill a run
            return waveform
        if noise.numel() == 0:
            return waveform

        noise = repeat_pad(noise.to(waveform.dtype), waveform.size(0))
        snr_db = rng.uniform(*self.snr_range)
        return _mix_at_snr(waveform, noise, snr_db)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        state = f"{len(self.files)} files" if self.enabled else "disabled"
        return f"MusanNoiseAugment({state})"


class RirReverbAugment:
    """Reverberation by convolution with a room impulse response.

    Uses FFT convolution and keeps the original length by taking the first `T`
    samples of the full convolution, then restores the input RMS so the
    augmentation changes the spectrum without changing loudness (loudness is a
    cue the model should not learn from).

    A no-op when the RIRS_NOISES directory is missing or holds no audio.
    """

    def __init__(
        self,
        rirs_dir: Path | str | None,
        *,
        probability: float = 0.3,
        patterns: Sequence[str] = ("*.wav",),
        subdirs: Sequence[str] = ("simulated_rirs", "real_rirs_isotropic_noises"),
        loader: AudioLoader | None = None,
        max_files: int | None = None,
    ) -> None:
        self.probability = probability
        self.loader = loader or default_audio_loader

        files: list[Path] = []
        root = Path(rirs_dir) if rirs_dir is not None else None
        if root is not None and root.is_dir():
            for subdir in subdirs:
                candidate = root / subdir
                if candidate.is_dir():
                    files.extend(_collect_audio(candidate, patterns))
            if not files:
                files = _collect_audio(root, patterns)
        self.files = files[:max_files] if max_files else files
        self.enabled = bool(self.files) and probability > 0.0

    def __call__(
        self, waveform: Tensor, generator: random.Random | None = None
    ) -> Tensor:
        if not self.enabled:
            return waveform
        rng = generator or random
        if rng.random() >= self.probability:
            return waveform

        rir_path = self.files[rng.randrange(len(self.files))]
        try:
            rir, _ = self.loader(rir_path)
        except Exception:  # noqa: BLE001 - a bad RIR must not kill a run
            return waveform
        if rir.numel() == 0:
            return waveform

        return _convolve_keep_length(waveform, rir.to(waveform.dtype))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        state = f"{len(self.files)} files" if self.enabled else "disabled"
        return f"RirReverbAugment({state})"


class AugmentChain:
    """Apply augmentations in order. `enabled` is true if any member is."""

    def __init__(self, augmentations: Iterable[Augmentation]) -> None:
        self.augmentations = [a for a in augmentations if a is not None]
        self.enabled = any(a.enabled for a in self.augmentations)

    def __call__(
        self, waveform: Tensor, generator: random.Random | None = None
    ) -> Tensor:
        for augmentation in self.augmentations:
            waveform = augmentation(waveform, generator)
        return waveform

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"AugmentChain({self.augmentations!r})"


def _mix_at_snr(signal: Tensor, noise: Tensor, snr_db: float) -> Tensor:
    """Add `noise` to `signal` so the result has the requested SNR in dB."""

    signal_power = signal.pow(2).mean()
    noise_power = noise.pow(2).mean()
    if signal_power <= 0 or noise_power <= 0:
        return signal
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(target_noise_power / noise_power)
    return signal + scale * noise


def _convolve_keep_length(signal: Tensor, impulse: Tensor) -> Tensor:
    """FFT convolution truncated to the input length, RMS preserved."""

    length = signal.size(0)
    n_fft = 1 << math.ceil(math.log2(length + impulse.size(0) - 1))
    spectrum = torch.fft.rfft(signal, n=n_fft) * torch.fft.rfft(impulse, n=n_fft)
    convolved = torch.fft.irfft(spectrum, n=n_fft)[:length]

    source_rms = signal.pow(2).mean().sqrt()
    result_rms = convolved.pow(2).mean().sqrt()
    if result_rms > 0 and source_rms > 0:
        convolved = convolved * (source_rms / result_rms)
    return convolved.to(signal.dtype)


def build_augmentation(
    musan_dir: Path | str | None = None,
    rirs_dir: Path | str | None = None,
    *,
    noise_probability: float = 0.5,
    reverb_probability: float = 0.3,
    snr_range: tuple[float, float] = (5.0, 20.0),
    loader: AudioLoader | None = None,
    max_files: int | None = None,
) -> AugmentChain:
    """The standard chain: reverb then additive noise.

    Order matters physically: a room colours the voice, then the microphone
    picks up ambient noise that was never in the room's impulse response. Doing
    it the other way round reverberates the noise too, which is wrong and, more
    practically, makes the noise SNR no longer the SNR you asked for.

    Returns a chain whose `enabled` is False when neither corpus is present, in
    which case it is the identity function.
    """

    return AugmentChain(
        [
            RirReverbAugment(
                rirs_dir,
                probability=reverb_probability,
                loader=loader,
                max_files=max_files,
            ),
            MusanNoiseAugment(
                musan_dir,
                snr_range=snr_range,
                probability=noise_probability,
                loader=loader,
                max_files=max_files,
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Everything that shapes a batch of waveforms."""

    sample_rate: int = 16_000
    crop_samples: int = DEFAULT_CROP_SAMPLES
    train: bool = True
    # Loud and quiet recordings are not a spoofing cue, so level is normalised
    # away. Peak rather than RMS, because RMS normalisation of a clip that is
    # mostly silence amplifies the noise floor.
    peak_normalise: bool = True
    resample_check: bool = True

    @property
    def crop_mode(self) -> str:
        return CropMode.RANDOM if self.train else CropMode.CENTRE

    @property
    def crop_seconds(self) -> float:
        return self.crop_samples / self.sample_rate


class UtteranceDataset(Dataset):
    """Fixed-length waveform crops from a `Manifest`.

    Each item is a dict, not a tuple, so adding a field later does not silently
    reorder anyone's unpacking:

        {
          "waveform":     float32 (crop_samples,)
          "target":       int64 scalar, 1 = bonafide, 0 = spoof
          "utterance_id": str
          "attack_id":    str        ("-" for bonafide, so it collates)
          "index":        int64 scalar
        }

    Augmentation applies in train mode only. The per-item RNG is derived from
    the dataset seed, the current epoch and the item index, so a DataLoader
    with several workers still produces reproducible augmentation, two workers
    never draw the same noise file for the same item, and (because `train.py`
    calls `set_epoch`) the random crop of a given file differs from epoch to
    epoch instead of being frozen at the first draw. In eval mode the crop is
    the deterministic centre crop and the epoch has no effect at all.
    """

    def __init__(
        self,
        manifest: Manifest | Sequence[Utterance],
        config: DatasetConfig | None = None,
        *,
        augmentation: Augmentation | None = None,
        loader: AudioLoader | None = None,
        seed: int = 0,
        epoch: int = 0,
    ) -> None:
        self.utterances: tuple[Utterance, ...] = (
            manifest.utterances if isinstance(manifest, Manifest) else tuple(manifest)
        )
        if not self.utterances:
            raise ValueError("UtteranceDataset was given an empty manifest")
        self.config = config or DatasetConfig()
        self.augmentation = augmentation
        self.loader = loader or default_audio_loader
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.utterances)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation and random-crop stream.

        Call once per epoch before building the DataLoader iterator. Workers
        receive the dataset by fork or pickle at iterator creation, so setting
        it mid-epoch has no effect on the workers already running.
        """

        self.epoch = int(epoch)

    def _rng(self, index: int) -> random.Random:
        mixed = (
            (self.seed + 1) * 1_000_003
            ^ (self.epoch + 1) * 15_485_863
            ^ (index + 1) * 2_654_435_761
        )
        return random.Random(mixed)

    def __getitem__(self, index: int) -> dict[str, object]:
        utterance = self.utterances[index]
        config = self.config
        rng = self._rng(index)

        waveform, sample_rate = self.loader(utterance.path)
        waveform = waveform.to(torch.float32)
        if waveform.dim() != 1:
            waveform = waveform.reshape(-1)

        if config.resample_check and sample_rate != config.sample_rate:
            raise ValueError(
                f"{utterance.path} is {sample_rate} Hz but the config expects "
                f"{config.sample_rate} Hz. ASVspoof LA is 16 kHz throughout; "
                "resample the corpus rather than resampling per batch, or set "
                "resample_check=False if you know what you are doing."
            )

        waveform = fixed_length_crop(
            waveform, config.crop_samples, config.crop_mode, rng
        )

        if config.train and self.augmentation is not None:
            waveform = self.augmentation(waveform, rng)
            # Augmentation can change the length if an augmenter misbehaves.
            if waveform.size(0) != config.crop_samples:
                waveform = fixed_length_crop(
                    waveform, config.crop_samples, CropMode.CENTRE
                )

        if config.peak_normalise:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

        return {
            "waveform": waveform,
            "target": torch.tensor(utterance.target, dtype=torch.long),
            "utterance_id": utterance.utterance_id,
            "attack_id": utterance.attack_id or "-",
            "index": torch.tensor(index, dtype=torch.long),
        }


class SyntheticWaveformDataset(Dataset):
    """Random audio with deterministic labels, for the smoke test.

    Not a model of anything: it exists so `train.py --smoke-test` can prove the
    pipeline is wired end to end in seconds with no dataset on disk. The two
    classes are separable on purpose (spoof gets a faint tone added and a
    narrower amplitude distribution), so a smoke run shows a loss that falls
    and an EER that moves. A smoke test where EER stays at 0.5 would prove only
    that nothing crashed.
    """

    def __init__(
        self,
        n_items: int = 64,
        *,
        crop_samples: int = 16_000,
        sample_rate: int = 16_000,
        bonafide_fraction: float = 0.5,
        seed: int = 0,
        separable: bool = True,
    ) -> None:
        if n_items <= 0:
            raise ValueError("n_items must be positive")
        self.n_items = n_items
        self.crop_samples = crop_samples
        self.sample_rate = sample_rate
        self.seed = seed
        self.separable = separable
        n_bonafide = max(1, min(n_items - 1, round(n_items * bonafide_fraction)))
        self.targets = [
            Label.BONAFIDE.target if i < n_bonafide else Label.SPOOF.target
            for i in range(n_items)
        ]

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        generator = torch.Generator().manual_seed(
            (self.seed * 7919 + index * 104_729) % (2**31)
        )
        waveform = torch.randn(
            self.crop_samples, generator=generator, dtype=torch.float32
        )
        if self.separable and target == Label.SPOOF.target:
            # A faint fixed tone plus a tighter amplitude range: a trivially
            # learnable spectral artefact, which is the point.
            time = torch.arange(self.crop_samples, dtype=torch.float32)
            tone = torch.sin(2 * math.pi * 3_000.0 * time / self.sample_rate)
            waveform = 0.6 * waveform + 0.8 * tone
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        return {
            "waveform": waveform,
            "target": torch.tensor(target, dtype=torch.long),
            "utterance_id": f"SYN_{index:06d}",
            "attack_id": "-" if target == Label.BONAFIDE.target else "SYN",
            "index": torch.tensor(index, dtype=torch.long),
        }


def balanced_sampler(
    manifest: Manifest, *, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    """Inverse-frequency sampler over a manifest.

    ASVspoof LA train is roughly 1 bonafide to 9 spoof. Left alone, the easy
    minimum is "say spoof", and the loss falls nicely while the EER does not.
    """

    weights = manifest.sample_weights()
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def collate_batch(items: Sequence[dict[str, object]]) -> dict[str, object]:
    """Collate that keeps the string fields as lists of strings.

    The default collate would try to stack `utterance_id`. It happens to
    produce a list of strings for str values, but spelling it out here means
    the batch keys are a contract rather than an accident.
    """

    return {
        "waveform": torch.stack([item["waveform"] for item in items]),  # type: ignore[arg-type]
        "target": torch.stack([item["target"] for item in items]),  # type: ignore[arg-type]
        "index": torch.stack([item["index"] for item in items]),  # type: ignore[arg-type]
        "utterance_id": [str(item["utterance_id"]) for item in items],
        "attack_id": [str(item["attack_id"]) for item in items],
    }


def dataset_for_subset(
    manifest: Manifest,
    subset: Subset,
    *,
    crop_samples: int = DEFAULT_CROP_SAMPLES,
    sample_rate: int = 16_000,
    augmentation: Augmentation | None = None,
    loader: AudioLoader | None = None,
    seed: int = 0,
) -> UtteranceDataset:
    """Dataset for one split, with train/eval behaviour chosen by the split.

    Only `Subset.TRAIN` gets random crops and augmentation. Everything else is
    deterministic, which is what makes dev and eval numbers comparable between
    runs.
    """

    is_train = subset is Subset.TRAIN
    return UtteranceDataset(
        manifest.filter(subset=subset),
        DatasetConfig(
            sample_rate=sample_rate, crop_samples=crop_samples, train=is_train
        ),
        augmentation=augmentation if is_train else None,
        loader=loader,
        seed=seed,
    )
