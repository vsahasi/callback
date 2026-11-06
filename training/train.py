"""Training loop for the Callback countermeasure.

    # prove the whole pipeline is wired, no dataset, a few seconds
    uv run python -m training.train --smoke-test

    # a real run once ASVspoof 2019 LA is unpacked
    uv run python -m training.train \
        --data-dir ~/data/callback --epochs 50 --batch-size 64

    # resume from the last checkpoint of a run
    uv run python -m training.train --resume runs/lfcc-lcnn/last.pt

What is here: AMP, gradient clipping, cosine LR with warmup, class-weighted
cross entropy, checkpointing on best dev EER, early stopping, deterministic
seeding, and a resume path that restores the optimizer, the scaler, the epoch
counter and the early-stopping state (restoring the weights alone and calling
it a resume quietly throws away the optimizer's momentum and the LR schedule
position).

Device and precision notes for the target box (RTX 5080, sm_120, torch
2.15.0+cu132 under WSL2):

* AMP uses the `torch.amp` API with an explicit device type, not the
  deprecated `torch.cuda.amp`.
* bfloat16 is the default autocast dtype on CUDA here rather than float16.
  Blackwell has full-rate bf16, and bf16 has the same exponent range as fp32,
  so the GradScaler is not strictly needed. It is still constructed and used
  when `--amp-dtype float16` is chosen, so both paths work.
* `torch.stft` does not accept half precision, so `features.py` disables
  autocast around the transform. The convolutional trunk still runs in low
  precision, which is where the time goes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training import metrics as metrics_mod
from training.datasets import (
    DEFAULT_CROP_SAMPLES,
    SyntheticWaveformDataset,
    balanced_sampler,
    build_augmentation,
    collate_batch,
    dataset_for_subset,
)
from training.features import FeatureConfig
from training.manifest import (
    Manifest,
    Subset,
    discover_roots,
    load_asvspoof2019_la,
)
from training.metrics import AsvErrorRates, CmMetrics, evaluate_cm_scores
from training.model import (
    CountermeasureModel,
    ModelConfig,
    count_parameters,
    score_from_logits,
)

__all__ = [
    "TrainConfig",
    "Trainer",
    "evaluate",
    "main",
    "seed_everything",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Every knob, in one place, so a run is reproducible from one dict.

    Serialised into every checkpoint, so a checkpoint says what produced it.
    """

    # data
    data_dir: Path | None = None
    crop_samples: int = DEFAULT_CROP_SAMPLES
    sample_rate: int = 16_000
    num_workers: int = 4
    balanced_sampling: bool = True
    augment: bool = True
    noise_probability: float = 0.5
    reverb_probability: float = 0.3

    # front end and model
    frontend: str = "lfcc"
    n_mels: int = 80
    n_filter: int = 20
    n_ceps: int = 20
    with_deltas: bool = True
    dropout: float = 0.3

    # optimisation
    epochs: int = 50
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 5.0
    class_weighted_loss: bool = True
    label_smoothing: float = 0.0

    # precision
    amp: bool = True
    amp_dtype: str = "bfloat16"

    # bookkeeping
    seed: int = 1234
    deterministic: bool = False
    output_dir: Path = Path("runs/lfcc-lcnn")
    patience: int = 10
    log_every: int = 50
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    resume: Path | None = None
    device: str = "auto"

    # t-DCF needs an ASV operating point. None means "report EER only",
    # which is the honest default until the ASV score files are parsed.
    asv_p_miss: float | None = None
    asv_p_fa: float | None = None
    asv_p_miss_spoof: float | None = None
    tdcf_version: str = "2019"

    smoke_test: bool = False

    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(
            name=self.frontend,  # type: ignore[arg-type]
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_filter=self.n_filter,
            n_ceps=self.n_ceps,
            with_deltas=self.with_deltas,
        )

    def model_config(self) -> ModelConfig:
        return ModelConfig(dropout=self.dropout)

    def asv_error_rates(self) -> AsvErrorRates | None:
        values = (self.asv_p_miss, self.asv_p_fa, self.asv_p_miss_spoof)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                "min t-DCF needs all three ASV error rates "
                "(--asv-p-miss, --asv-p-fa, --asv-p-miss-spoof) or none of "
                "them. Derive them from the ASVspoof ASV score files with "
                "manifest.parse_asv_scores + metrics.obtain_asv_error_rates."
            )
        return AsvErrorRates(
            p_miss=float(self.asv_p_miss),  # type: ignore[arg-type]
            p_fa=float(self.asv_p_fa),  # type: ignore[arg-type]
            p_miss_spoof=float(self.asv_p_miss_spoof),  # type: ignore[arg-type]
        )

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def autocast_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
        }
        if self.amp_dtype not in mapping:
            raise ValueError(
                f"amp_dtype must be one of {sorted(mapping)}, got {self.amp_dtype!r}"
            )
        return mapping[self.amp_dtype]

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, torch and (if present) numpy.

    `deterministic=True` also pins cuDNN to deterministic algorithms and sets
    the cuBLAS workspace variable CUDA needs for reproducible GEMMs. It costs
    real throughput, so it is off by default: the seeding above is enough to
    reproduce a run's data order and initialisation, which is what matters for
    comparing two configs.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy  # type: ignore[import-not-found]

        numpy.random.seed(seed)
    except ImportError:
        pass

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker its own, derived seed."""

    base = torch.initial_seed() % (2**31)
    random.seed(base + worker_id)
    try:
        import numpy  # type: ignore[import-not-found]

        numpy.random.seed((base + worker_id) % (2**32))
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    asv: AsvErrorRates | None = None,
    tdcf_version: str = "2019",
    max_batches: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    per_attack: bool = True,
) -> tuple[CmMetrics, dict[str, list[Any]]]:
    """Score a loader and compute EER (and min t-DCF when ASV rates are
    given).

    Returns `(metrics, raw)` where `raw` carries the utterance ids, scores and
    targets, so the caller can write an ASVspoof-format score file and have the
    official evaluation package check these numbers.
    """

    model.eval()
    utterance_ids: list[str] = []
    scores: list[float] = []
    targets: list[int] = []
    attacks: list[str] = []

    use_amp = autocast_dtype is not None and device.type == "cuda"
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        waveform = batch["waveform"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=use_amp
        ):
            logits = model(waveform)
        batch_scores = score_from_logits(logits.float())

        scores.extend(batch_scores.detach().cpu().tolist())
        targets.extend(batch["target"].detach().cpu().tolist())
        utterance_ids.extend(batch["utterance_id"])
        attacks.extend(batch.get("attack_id", ["-"] * len(batch_scores)))

    bonafide = [s for s, t in zip(scores, targets, strict=True) if t == 1]
    spoof = [s for s, t in zip(scores, targets, strict=True) if t == 0]
    if not bonafide or not spoof:
        raise ValueError(
            f"evaluation needs both classes, got {len(bonafide)} bonafide and "
            f"{len(spoof)} spoof trials. Increase --max-eval-batches or check "
            "the manifest."
        )

    by_attack: dict[str, list[float]] | None = None
    if per_attack:
        by_attack = {}
        for score, target, attack in zip(scores, targets, attacks, strict=True):
            if target == 1 or attack in ("-", ""):
                continue
            by_attack.setdefault(attack, []).append(score)
        by_attack = by_attack or None

    result = evaluate_cm_scores(
        bonafide,
        spoof,
        asv=asv,
        tdcf_version=tdcf_version,
        spoof_scores_by_attack=by_attack,
    )
    raw = {
        "utterance_id": utterance_ids,
        "score": scores,
        "target": targets,
        "attack_id": attacks,
    }
    return result, raw


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class _EarlyStopping:
    patience: int
    best: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record `value`; return True when it is a new best (lower is
        better)."""

        if value < self.best:
            self.best = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


class Trainer:
    """Owns the loop, the checkpoints and the early-stopping state."""

    def __init__(
        self,
        config: TrainConfig,
        model: CountermeasureModel,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        *,
        class_weights: Sequence[float] | None = None,
    ) -> None:
        self.config = config
        self.device = config.resolved_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        weight = (
            torch.tensor(list(class_weights), dtype=torch.float32, device=self.device)
            if class_weights and config.class_weighted_loss
            else None
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=config.label_smoothing
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.autocast_dtype = config.autocast_dtype()
        self.amp_enabled = config.amp and self.device.type == "cuda"
        # GradScaler only matters for float16. bf16 has fp32's exponent range,
        # so gradients do not underflow and scaling is a no-op cost. Enabled
        # exactly when it is needed.
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp_enabled and self.autocast_dtype is torch.float16,
        )

        self.epoch = 0
        self.global_step = 0
        self.early_stopping = _EarlyStopping(patience=config.patience)
        self.output_dir = Path(config.output_dir)
        self.history: list[dict[str, Any]] = []

    # -- schedule ---------------------------------------------------------

    def _lr_at(self, epoch: int) -> float:
        """Linear warmup then cosine decay to 1% of the peak LR."""

        config = self.config
        if config.warmup_epochs > 0 and epoch < config.warmup_epochs:
            return config.learning_rate * (epoch + 1) / config.warmup_epochs
        span = max(1, config.epochs - config.warmup_epochs)
        progress = min(1.0, (epoch - config.warmup_epochs) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.learning_rate * (0.01 + 0.99 * cosine)

    def _apply_lr(self, epoch: int) -> float:
        lr = self._lr_at(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # -- one epoch --------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        config = self.config
        self.model.train()
        lr = self._apply_lr(self.epoch)

        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)  # type: ignore[union-attr]

        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        started = time.monotonic()

        for batch_index, batch in enumerate(self.train_loader):
            if (
                config.max_train_batches is not None
                and batch_index >= config.max_train_batches
            ):
                break

            waveform = batch["waveform"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.amp_enabled,
            ):
                logits = self.model(waveform)
                loss = self.criterion(logits.float(), target)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"loss is {loss.item()} at step {self.global_step}; "
                    "stopping rather than poisoning the weights. Lower the "
                    "learning rate or switch --amp-dtype to bfloat16."
                )

            self.scaler.scale(loss).backward()
            if config.grad_clip > 0:
                # Unscale before clipping, or the clip threshold is applied to
                # scaled gradients and means nothing.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((logits.argmax(dim=1) == target).sum().item())
            total_seen += batch_size
            self.global_step += 1

            if config.log_every and self.global_step % config.log_every == 0:
                print(
                    f"  epoch {self.epoch} step {self.global_step} "
                    f"loss {total_loss / max(1, total_seen):.4f} "
                    f"acc {total_correct / max(1, total_seen):.3f}",
                    flush=True,
                )

        if total_seen == 0:
            raise RuntimeError(
                "the training loader produced no batches; check the manifest "
                "and --max-train-batches"
            )

        return {
            "loss": total_loss / total_seen,
            "accuracy": total_correct / total_seen,
            "lr": lr,
            "seconds": time.monotonic() - started,
            "samples": float(total_seen),
        }

    # -- checkpoints ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "model_config": self.model.config_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "train_config": self.config.to_json(),
            "best_dev_eer": self.early_stopping.best,
            "best_epoch": self.early_stopping.best_epoch,
            "epochs_without_improvement": (
                self.early_stopping.epochs_without_improvement
            ),
            "history": self.history,
        }

    def save_checkpoint(self, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        # Write to a temporary file then rename, so an interrupted save cannot
        # leave a truncated checkpoint where the good one used to be.
        staging = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), staging)
        staging.replace(path)
        return path

    def load_checkpoint(self, path: Path | str, *, model_only: bool = False) -> None:
        """Restore a run: weights, optimizer, scaler, epoch, early stopping.

        `model_only=True` loads just the weights, which is what you want when
        reloading the best checkpoint for a final scoring pass rather than
        resuming training.

        `torch.load(weights_only=False)` because a checkpoint here carries the
        config dicts and the history list, not only tensors. That lets
        `torch.load` execute pickled code, so only load checkpoints this
        project wrote.
        """

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if model_only:
            return
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.epoch = int(payload.get("epoch", 0)) + 1
        self.global_step = int(payload.get("global_step", 0))
        self.early_stopping.best = float(payload.get("best_dev_eer", math.inf))
        self.early_stopping.best_epoch = int(payload.get("best_epoch", -1))
        self.early_stopping.epochs_without_improvement = int(
            payload.get("epochs_without_improvement", 0)
        )
        self.history = list(payload.get("history", []))
        print(
            f"resumed from {path}: next epoch {self.epoch}, "
            f"best dev EER so far {self.early_stopping.best:.5f}",
            flush=True,
        )

    # -- the loop ---------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        config = self.config
        asv = config.asv_error_rates()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(config.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(
            f"device {self.device}  "
            f"params {count_parameters(self.model):,}  "
            f"amp {self.amp_enabled} ({config.amp_dtype})  "
            f"scaler {self.scaler.is_enabled()}",
            flush=True,
        )

        while self.epoch < config.epochs:
            train_stats = self.train_epoch()
            dev_metrics, _ = evaluate(
                self.model,
                self.dev_loader,
                self.device,
                asv=asv,
                tdcf_version=config.tdcf_version,
                max_batches=config.max_eval_batches,
                autocast_dtype=self.autocast_dtype if self.amp_enabled else None,
            )

            improved = self.early_stopping.update(dev_metrics.eer, self.epoch)
            record = {
                "epoch": self.epoch,
                "train_loss": train_stats["loss"],
                "train_accuracy": train_stats["accuracy"],
                "lr": train_stats["lr"],
                "seconds": train_stats["seconds"],
                "dev_eer": dev_metrics.eer,
                "dev_min_tdcf": dev_metrics.min_tdcf,
                "improved": improved,
            }
            self.history.append(record)

            print(
                f"epoch {self.epoch:3d}  "
                f"loss {train_stats['loss']:.4f}  "
                f"acc {train_stats['accuracy']:.3f}  "
                f"lr {train_stats['lr']:.2e}  "
                f"dev {dev_metrics.summary()}  "
                f"{train_stats['seconds']:.1f}s"
                f"{'  <- best' if improved else ''}",
                flush=True,
            )

            self.save_checkpoint("last.pt")
            if improved:
                self.save_checkpoint("best.pt")

            if self.early_stopping.should_stop:
                print(
                    f"early stop: {config.patience} epochs without a dev EER "
                    f"improvement (best {self.early_stopping.best:.5f} at "
                    f"epoch {self.early_stopping.best_epoch})",
                    flush=True,
                )
                self.epoch += 1
                break

            self.epoch += 1

        (self.output_dir / "history.json").write_text(
            json.dumps(self.history, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "best_dev_eer": self.early_stopping.best,
            "best_epoch": self.early_stopping.best_epoch,
            "epochs_run": len(self.history),
            "output_dir": str(self.output_dir),
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    sampler: Any = None,
    num_workers: int = 0,
    device: torch.device | None = None,
    drop_last: bool = False,
) -> DataLoader:
    pin = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=drop_last,
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )


def build_smoke_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    """Synthetic train and dev loaders. No dataset, no network, no audio
    files."""

    crop = min(config.crop_samples, 16_000)
    train_set = SyntheticWaveformDataset(
        n_items=64, crop_samples=crop, sample_rate=config.sample_rate, seed=config.seed
    )
    dev_set = SyntheticWaveformDataset(
        n_items=32,
        crop_samples=crop,
        sample_rate=config.sample_rate,
        seed=config.seed + 1,
    )
    device = config.resolved_device()
    train_loader = _make_loader(
        train_set,
        config.batch_size,
        shuffle=True,
        num_workers=0,
        device=device,
        # BatchNorm1d in the head cannot handle a batch of 1 in train mode.
        drop_last=True,
    )
    dev_loader = _make_loader(
        dev_set, config.eval_batch_size, num_workers=0, device=device
    )
    return train_loader, dev_loader, (1.0, 1.0)


def build_real_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    """Train and dev loaders over ASVspoof 2019 LA.

    Raises with an actionable message when the corpus is not unpacked, rather
    than producing an empty loader and a confusing error twenty lines later.
    """

    roots = discover_roots(config.data_dir)
    if roots.asvspoof2019_la is None:
        raise FileNotFoundError(
            "ASVspoof 2019 LA is not unpacked.\n"
            f"{roots.describe()}\n"
            "Fetch it with `uv run python -m training.download --only "
            "asvspoof2019_la`, unpack asvspoof2019_LA.zip in the data dir, "
            "then re-run. To prove the pipeline without data, use "
            "--smoke-test."
        )

    manifest: Manifest = load_asvspoof2019_la(
        roots.asvspoof2019_la, (Subset.TRAIN, Subset.DEV)
    )
    train_manifest = manifest.for_training()
    # The dev DataLoader is built from the full manifest by dataset_for_subset
    # below, so a separate dev manifest is not needed here.
    print(f"manifest: {manifest.counts}", flush=True)

    augmentation = (
        build_augmentation(
            roots.musan,
            roots.rirs_noises,
            noise_probability=config.noise_probability,
            reverb_probability=config.reverb_probability,
        )
        if config.augment
        else None
    )
    if augmentation is not None and not augmentation.enabled:
        print(
            "augmentation requested but neither MUSAN nor RIRS_NOISES is "
            "unpacked; running clean",
            flush=True,
        )

    train_set = dataset_for_subset(
        manifest,
        Subset.TRAIN,
        crop_samples=config.crop_samples,
        sample_rate=config.sample_rate,
        augmentation=augmentation,
        seed=config.seed,
    )
    dev_set = dataset_for_subset(
        manifest,
        Subset.DEV,
        crop_samples=config.crop_samples,
        sample_rate=config.sample_rate,
        seed=config.seed,
    )

    device = config.resolved_device()
    sampler = (
        balanced_sampler(
            train_manifest,
            generator=torch.Generator().manual_seed(config.seed),
        )
        if config.balanced_sampling
        else None
    )
    train_loader = _make_loader(
        train_set,
        config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        device=device,
        drop_last=True,
    )
    dev_loader = _make_loader(
        dev_set,
        config.eval_batch_size,
        num_workers=config.num_workers,
        device=device,
    )
    return train_loader, dev_loader, train_manifest.class_weights()


def run(config: TrainConfig) -> dict[str, Any]:
    """Build everything from a config and train. The programmatic entry
    point."""

    if config.smoke_test:
        config = _smoke_overrides(config)

    seed_everything(config.seed, config.deterministic)

    if config.smoke_test:
        train_loader, dev_loader, class_weights = build_smoke_loaders(config)
    else:
        train_loader, dev_loader, class_weights = build_real_loaders(config)

    model = CountermeasureModel(config.feature_config(), config.model_config())
    trainer = Trainer(
        config,
        model,
        train_loader,
        dev_loader,
        class_weights=class_weights,
    )
    if config.resume is not None:
        trainer.load_checkpoint(config.resume)

    result = trainer.fit()

    # Score the BEST checkpoint, not whatever the last epoch left in memory.
    # Reporting the final-epoch model under the best epoch's EER is a classic
    # quiet inconsistency between a training log and a results table.
    best_path = Path(config.output_dir) / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path, model_only=True)

    # Write a score file for the final dev pass, in the format the official
    # ASVspoof evaluation package reads. This is the artefact that lets the
    # numbers above be checked by code nobody here wrote.
    dev_metrics, raw = evaluate(
        trainer.model,
        dev_loader,
        trainer.device,
        asv=config.asv_error_rates(),
        tdcf_version=config.tdcf_version,
        max_batches=config.max_eval_batches,
    )
    metrics_mod.write_score_file(
        Path(config.output_dir) / "dev_scores.txt",
        raw["utterance_id"],
        raw["score"],
    )
    result["final_dev"] = {
        "eer": dev_metrics.eer,
        "min_tdcf": dev_metrics.min_tdcf,
        "per_attack_eer": dict(dev_metrics.per_attack_eer or {}),
    }
    print(f"final dev {dev_metrics.summary()}", flush=True)
    return result


def _smoke_overrides(config: TrainConfig) -> TrainConfig:
    """Shrink a config so the smoke test finishes in seconds on CPU.

    Deliberately explicit rather than a separate code path: the smoke test runs
    the real `Trainer`, the real loss, the real AMP wiring, the real
    checkpointing and the real metrics. Only the sizes change. A smoke test
    that uses its own loop proves nothing about the loop you ship.

    A replacement is applied only where the caller left the field at its
    default, so `--smoke-test --epochs 10` really runs ten epochs instead of
    silently running two.
    """

    defaults = TrainConfig()
    replacements: dict[str, Any] = {
        "epochs": 2,
        "batch_size": 8,
        "eval_batch_size": 8,
        # 16000 samples at hop 160 is 101 frames, and the LCNN trunk needs at
        # least 16 frames to survive four pooling stages.
        "crop_samples": 16_000,
        "num_workers": 0,
        "warmup_epochs": 1,
        "patience": 5,
        "log_every": 0,
        "augment": False,
        "balanced_sampling": False,
        "learning_rate": 1e-3,
        "output_dir": Path("runs/smoke-test"),
    }
    payload = vars(config).copy()
    for key, value in replacements.items():
        if payload[key] == getattr(defaults, key):  # untouched by the caller
            payload[key] = value
    return TrainConfig(**payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m training.train",
        description="Train the Callback audio anti-spoofing countermeasure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    data.add_argument("--crop-samples", type=int, default=defaults.crop_samples)
    data.add_argument("--sample-rate", type=int, default=defaults.sample_rate)
    data.add_argument("--num-workers", type=int, default=defaults.num_workers)
    data.add_argument(
        "--no-balanced-sampling", dest="balanced_sampling", action="store_false"
    )
    data.add_argument("--no-augment", dest="augment", action="store_false")
    data.add_argument(
        "--noise-probability", type=float, default=defaults.noise_probability
    )
    data.add_argument(
        "--reverb-probability", type=float, default=defaults.reverb_probability
    )

    front = parser.add_argument_group("front end and model")
    front.add_argument(
        "--frontend", choices=("lfcc", "logmel"), default=defaults.frontend
    )
    front.add_argument("--n-mels", type=int, default=defaults.n_mels)
    front.add_argument("--n-filter", type=int, default=defaults.n_filter)
    front.add_argument("--n-ceps", type=int, default=defaults.n_ceps)
    front.add_argument("--no-deltas", dest="with_deltas", action="store_false")
    front.add_argument("--dropout", type=float, default=defaults.dropout)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=defaults.epochs)
    optim.add_argument("--batch-size", type=int, default=defaults.batch_size)
    optim.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    optim.add_argument(
        "--learning-rate", "--lr", type=float, default=defaults.learning_rate
    )
    optim.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    optim.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    optim.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    optim.add_argument(
        "--no-class-weighted-loss",
        dest="class_weighted_loss",
        action="store_false",
    )
    optim.add_argument(
        "--label-smoothing", type=float, default=defaults.label_smoothing
    )

    precision = parser.add_argument_group("precision")
    precision.add_argument("--no-amp", dest="amp", action="store_false")
    precision.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default=defaults.amp_dtype,
    )

    book = parser.add_argument_group("bookkeeping")
    book.add_argument("--seed", type=int, default=defaults.seed)
    book.add_argument("--deterministic", action="store_true")
    book.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    book.add_argument("--patience", type=int, default=defaults.patience)
    book.add_argument("--log-every", type=int, default=defaults.log_every)
    book.add_argument("--max-train-batches", type=int, default=None)
    book.add_argument("--max-eval-batches", type=int, default=None)
    book.add_argument(
        "--resume", type=Path, default=None, help="checkpoint to resume from"
    )
    book.add_argument("--device", default=defaults.device)

    tdcf = parser.add_argument_group("t-DCF (needs all three or none)")
    tdcf.add_argument("--asv-p-miss", type=float, default=None)
    tdcf.add_argument("--asv-p-fa", type=float, default=None)
    tdcf.add_argument("--asv-p-miss-spoof", type=float, default=None)
    tdcf.add_argument(
        "--tdcf-version", choices=("2019", "2021"), default=defaults.tdcf_version
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "run the whole pipeline on synthetic random audio in seconds, "
            "with no dataset and no network"
        ),
    )
    return parser


def config_from_args(argv: Sequence[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    fields = set(TrainConfig.__dataclass_fields__)
    payload = {k: v for k, v in vars(args).items() if k in fields}
    return TrainConfig(**payload)


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(argv)
    try:
        result = run(config)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    print(
        f"done. best dev EER {result['best_dev_eer'] * 100:.3f}% at epoch "
        f"{result['best_epoch']}; checkpoints in {result['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
