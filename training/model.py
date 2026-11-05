"""A compact LCNN countermeasure, front end included.

ARCHITECTURE CHOICE: LCNN over AASIST
-------------------------------------
Both were considered. LCNN wins here for three concrete reasons:

1. **It is the architecture that matches the front end this project asked
   for.** AASIST consumes raw waveform through a learned sinc filterbank and a
   spectro-temporal graph attention stack. Choosing it would make
   `features.py` dead code and leave the configurable log-mel/LFCC front end
   with nothing downstream. LCNN is a 2-D convolutional network over a
   time-frequency image, which is exactly what that front end produces.

2. **Size and training cost.** At the default settings this model is under a
   quarter of a million parameters (`count_parameters` reports the exact
   figure, and a unit test holds it under 1 M so a refactor cannot quietly
   inflate it). On an RTX 5080 with 16 GB that leaves plenty of VRAM for a
   large batch or a second concurrent experiment. AASIST is a similar
   parameter count but its graph attention layers are far more memory-hungry
   per sample, and the published recipe needs more epochs to converge.

3. **It is the right risk profile for a hackathon.** The Max-Feature-Map
   activation is the only unusual component and it is six lines. The LCNN +
   LFCC combination was a top-performing single system in ASVspoof 2019 LA (the
   STC submission) and remains the ASVspoof 2021 baseline, so a disappointing
   result points at the data pipeline rather than at an architecture that was
   reimplemented from a paper under time pressure.

What LCNN gives up: AASIST generalises better to unseen attacks, which is
exactly what matters for the In-the-Wild evaluation. If the 2019-to-In-the-Wild
gap turns out to be the blocker, swapping in an AASIST or an SSL front end
(wav2vec 2.0 features plus a light back end) is the next move, and
`CountermeasureModel` is the seam: anything with the same
`(B, T) -> (B, 2)` signature drops in.

Max-Feature-Map (MFM)
---------------------
MFM replaces ReLU with an elementwise maximum over two halves of the channel
axis, so a 2C-channel convolution produces C channels. It is a learned
competitive activation rather than a fixed threshold: no gradient is lost on
the negative side, and the channel count halves, which is where LCNN's
parameter efficiency comes from.

Scoring convention
------------------
The model emits 2 logits, ordered to match `manifest.Label.target`: index 0 is
spoof, index 1 is bonafide. `score_from_logits` returns the bonafide
log-probability, so higher means more bonafide, which is the convention
`training.metrics` and the official ASVspoof packages expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from training.features import FeatureConfig, build_frontend

__all__ = [
    "LCNN",
    "CountermeasureModel",
    "MaxFeatureMap2d",
    "ModelConfig",
    "build_model",
    "count_parameters",
    "score_from_logits",
]


class MaxFeatureMap2d(nn.Module):
    """Elementwise max over two halves of the channel axis. `2C -> C`."""

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        size = x.size(self.dim)
        if size % 2 != 0:
            raise ValueError(
                f"MaxFeatureMap2d needs an even channel count on dim "
                f"{self.dim}, got {size}"
            )
        first, second = torch.split(x, size // 2, dim=self.dim)
        return torch.maximum(first, second)


def _mfm_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Sequential:
    """Convolution producing `2 * out_channels`, then MFM down to
    `out_channels`."""

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        MaxFeatureMap2d(),
    )


@dataclass
class ModelConfig:
    """Back end shape and regularisation."""

    # Channel widths after each MFM stage. The default is the LCNN-9 ladder
    # from the ASVspoof baselines.
    channels: tuple[int, ...] = (32, 48, 64, 32, 32)
    # Frequency bins the trunk output is adaptively pooled to before the head.
    # Fixed so the head's input size does not depend on n_mels or n_ceps: the
    # four 2x2 pooling stages take LFCC-60 down to 3 bins and log-mel-80 down
    # to 5, and the adaptive pool makes both 3.
    pooled_freq: int = 3
    embedding_dim: int = 80
    dropout: float = 0.3
    n_classes: int = 2
    batch_norm: bool = True

    def __post_init__(self) -> None:
        if len(self.channels) != 5:
            raise ValueError(
                f"channels must have 5 entries for this LCNN-9 trunk, got "
                f"{len(self.channels)}"
            )
        if self.pooled_freq < 1:
            raise ValueError("pooled_freq must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class LCNN(nn.Module):
    """Light CNN with MFM activations over a `(B, 1, F, T)` feature image.

    Length-agnostic by construction. The trunk is all convolution and pooling,
    then the frequency axis is adaptively pooled to `pooled_freq` bins and the
    time axis is collapsed by mean-and-standard-deviation statistics pooling.
    So a model trained on 4 s crops can score a 2 s or a 10 s utterance with no
    reshaping, which matters because real calls are not 4.03 s long.

    Statistics pooling (mean and std, concatenated) rather than mean alone:
    the variance of a feature over time is itself a cue, since synthesis
    artefacts tend to be unnaturally stationary.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c1, c2, c3, c4, c5 = self.config.channels

        def norm(channels: int) -> nn.Module:
            return nn.BatchNorm2d(channels) if self.config.batch_norm else nn.Identity()

        self.trunk = nn.Sequential(
            # Block 1
            _mfm_conv(in_channels, c1, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            # Block 2: 1x1 bottleneck then 3x3
            _mfm_conv(c1, c1, kernel_size=1),
            norm(c1),
            _mfm_conv(c1, c2, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            norm(c2),
            # Block 3
            _mfm_conv(c2, c2, kernel_size=1),
            norm(c2),
            _mfm_conv(c2, c3, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            # Block 4
            _mfm_conv(c3, c3, kernel_size=1),
            norm(c3),
            _mfm_conv(c3, c4, kernel_size=3, padding=1),
            norm(c4),
            # Block 5
            _mfm_conv(c4, c4, kernel_size=1),
            norm(c4),
            _mfm_conv(c4, c5, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        # Pool the frequency axis to a fixed size, keep the time axis.
        self.freq_pool = nn.AdaptiveAvgPool2d((self.config.pooled_freq, None))

        pooled = c5 * self.config.pooled_freq * 2  # x2 for mean and std
        self.head = nn.Sequential(
            nn.Linear(pooled, self.config.embedding_dim * 2),
            MaxFeatureMap2d(dim=1),
            nn.BatchNorm1d(self.config.embedding_dim)
            if self.config.batch_norm
            else nn.Identity(),
            nn.Dropout(self.config.dropout),
        )
        self.classifier = nn.Linear(self.config.embedding_dim, self.config.n_classes)

    def embed(self, features: Tensor) -> Tensor:
        """`(B, 1, F, T)` features to a `(B, embedding_dim)` utterance
        embedding."""

        if features.dim() != 4:
            raise ValueError(
                "LCNN expects a 4-D (batch, channel, freq, time) feature "
                f"tensor, got shape {tuple(features.shape)}"
            )
        if features.size(-1) < 16 or features.size(-2) < 16:
            raise ValueError(
                "feature map is too small for four 2x2 pooling stages: got "
                f"{tuple(features.shape)[-2:]} (freq, time), need at least "
                "16x16. Use a longer crop or fewer feature dimensions."
            )

        x = self.trunk(features)
        x = self.freq_pool(x)  # (B, C, pooled_freq, T')
        x = x.flatten(1, 2)  # (B, C * pooled_freq, T')
        mean = x.mean(dim=-1)
        # correction=0 (the biased variance) so a single remaining time step
        # gives 0 rather than NaN, which happens with very short inputs.
        std = x.var(dim=-1, correction=0).clamp_min(1e-10).sqrt()
        pooled = torch.cat([mean, std], dim=1)
        return self.head(pooled)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.embed(features))


class CountermeasureModel(nn.Module):
    """Front end plus LCNN: raw waveform in, 2 logits out.

    Keeping the front end inside the module means the STFT runs on the GPU in
    the forward pass, the DataLoader only ships raw audio, and a checkpoint
    carries the feature configuration it was trained with. A model that
    silently gets log-mel at inference after being trained on LFCC is the kind
    of bug that shows up as "the deployed detector is worse than the paper",
    and this makes it structurally impossible.
    """

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        self.frontend = build_frontend(self.feature_config)
        self.backbone = LCNN(self.model_config, in_channels=1)

    def forward(self, waveform: Tensor) -> Tensor:
        """`(B, T)` or `(B, 1, T)` waveform to `(B, 2)` logits."""

        return self.backbone(self.frontend(waveform))

    @torch.no_grad()
    def score(self, waveform: Tensor) -> Tensor:
        """`(B,)` countermeasure scores. Higher means more bonafide."""

        was_training = self.training
        self.eval()
        try:
            return score_from_logits(self(waveform))
        finally:
            if was_training:
                self.train()

    def config_dict(self) -> dict[str, object]:
        """Serialisable config, stored alongside the weights in a
        checkpoint."""

        return {
            "feature": vars(self.feature_config).copy(),
            "model": vars(self.model_config).copy(),
        }

    @classmethod
    def from_config_dict(cls, payload: dict[str, object]) -> CountermeasureModel:
        """Rebuild from `config_dict`, so a checkpoint is self-describing."""

        feature = payload.get("feature") or {}
        model = payload.get("model") or {}
        if not isinstance(feature, dict) or not isinstance(model, dict):
            raise ValueError(
                "checkpoint config must hold 'feature' and 'model' dicts, got "
                f"{type(feature).__name__} and {type(model).__name__}"
            )
        model = dict(model)
        if "channels" in model:
            model["channels"] = tuple(model["channels"])  # type: ignore[index]
        return cls(FeatureConfig(**feature), ModelConfig(**model))


def score_from_logits(logits: Tensor) -> Tensor:
    """Bonafide log-probability from `(B, 2)` logits. Higher = more bonafide.

    The log-softmax of the bonafide class, which is what the ASVspoof baselines
    submit. Monotonic in the logit difference, so EER is identical either way;
    the log-probability is used because it is bounded above by 0 and therefore
    easier to eyeball in a score file.
    """

    if logits.dim() != 2 or logits.size(1) < 2:
        raise ValueError(f"expected (batch, 2) logits, got shape {tuple(logits.shape)}")
    return torch.log_softmax(logits, dim=1)[:, 1]


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if p.requires_grad or not trainable_only
    )


def build_model(
    feature_config: FeatureConfig | None = None,
    model_config: ModelConfig | None = None,
) -> CountermeasureModel:
    return CountermeasureModel(feature_config, model_config)
