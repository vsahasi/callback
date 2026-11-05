"""Waveform front ends: log-mel and LFCC, selectable by config.

Both are `nn.Module`s that take a batch of waveforms `(B, T)` or `(B, 1, T)`
and return `(B, 1, F, N)`: a single-channel time-frequency image ready for a
2-D convolutional countermeasure. They live on whatever device the model does,
so the DataLoader only ever moves raw audio and the STFT runs on the GPU.

WHY NO TORCHAUDIO
-----------------
Everything here is implemented against plain `torch`: `torch.stft` plus
hand-built filterbank and DCT matrices. torchaudio would shorten this file by
maybe sixty lines, and in exchange it would pin a second CUDA-compiled wheel
that has to match torch exactly. On a Blackwell sm_120 box running
torch 2.15.0+cu132 that is the dependency most likely to have no matching
build, and a front end is not worth a blocked environment. The filterbank and
DCT are a few dozen lines of linear algebra that are easy to test directly.

LFCC is the classic ASVspoof countermeasure front end and is the default here:
the LA attacks leave their strongest traces in fine spectral detail and in the
very high band, which a mel scale deliberately smooths away. Log-mel is
provided for comparison and because it is what most pretrained audio models
expect, should the model later be swapped for one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "LFCC",
    "FeatureConfig",
    "LogMelSpectrogram",
    "build_frontend",
    "compute_deltas",
    "dct_matrix",
    "hz_to_mel",
    "linear_filterbank",
    "mel_filterbank",
    "mel_to_hz",
]

FrontendName = Literal["lfcc", "logmel"]


@dataclass
class FeatureConfig:
    """Front end configuration.

    Defaults are the ASVspoof LFCC baseline settings: 16 kHz audio, 20 ms
    window, 10 ms hop, 20 cepstral coefficients plus first and second deltas,
    giving 60 feature dimensions.
    """

    name: FrontendName = "lfcc"
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 320  # 20 ms at 16 kHz
    hop_length: int = 160  # 10 ms at 16 kHz
    f_min: float = 0.0
    f_max: float | None = None  # defaults to Nyquist
    # LFCC
    n_filter: int = 20
    n_ceps: int = 20
    with_deltas: bool = True
    with_energy: bool = True
    # log-mel
    n_mels: int = 80
    # shared
    log_eps: float = 1e-8
    power: float = 2.0
    center: bool = True
    normalise: bool = True  # per-utterance mean/variance normalisation

    def __post_init__(self) -> None:
        if self.name not in ("lfcc", "logmel"):
            raise ValueError(
                f"unknown frontend {self.name!r}; expected 'lfcc' or 'logmel'"
            )
        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length ({self.win_length}) must not exceed n_fft ({self.n_fft})"
            )
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.n_ceps > self.n_filter:
            raise ValueError(
                f"n_ceps ({self.n_ceps}) cannot exceed n_filter "
                f"({self.n_filter}); there are only n_filter log-energies to "
                "take a DCT of"
            )

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def top_hz(self) -> float:
        return self.f_max if self.f_max is not None else self.nyquist

    @property
    def n_features(self) -> int:
        """Feature dimension of the output, so the model can be sized."""

        if self.name == "logmel":
            return self.n_mels
        base = self.n_ceps
        return base * 3 if self.with_deltas else base


# ---------------------------------------------------------------------------
# Filterbanks and DCT, as plain tensor builders so they can be tested alone
# ---------------------------------------------------------------------------


def hz_to_mel(hz: float) -> float:
    """HTK mel scale, the one Kaldi and torchaudio's `htk` mode use."""

    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _triangular_bank(
    edges: list[float],
    n_freqs: int,
    sample_rate: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Triangular filterbank from `len(edges) - 2` overlapping triangles.

    `edges` are band edge frequencies in Hz, length `n_filter + 2`. Returns
    `(n_filter, n_freqs)`, to be applied as `bank @ spectrogram`.
    """

    n_filter = len(edges) - 2
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, dtype=dtype)
    bank = torch.zeros(n_filter, n_freqs, dtype=dtype)

    for index in range(n_filter):
        left, centre, right = edges[index], edges[index + 1], edges[index + 2]
        if right <= left:  # degenerate band, leave it empty
            continue
        rising = (freqs - left) / max(centre - left, 1e-12)
        falling = (right - freqs) / max(right - centre, 1e-12)
        bank[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)

    return bank


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """`(n_mels, n_fft // 2 + 1)` triangular mel filterbank, HTK scale.

    Not area-normalised (no Slaney normalisation), matching torchaudio's
    default `norm=None`, so each filter peaks at 1.0.
    """

    top = f_max if f_max is not None else sample_rate / 2.0
    mel_edges = torch.linspace(hz_to_mel(f_min), hz_to_mel(top), n_mels + 2)
    edges = [mel_to_hz(float(value)) for value in mel_edges]
    return _triangular_bank(edges, n_fft // 2 + 1, sample_rate, dtype)


def linear_filterbank(
    n_filter: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """`(n_filter, n_fft // 2 + 1)` linearly spaced triangular filterbank.

    This is the "L" in LFCC: identical machinery to the mel bank with the
    warping removed, so resolution is uniform across the band and the high
    frequencies where vocoder artefacts live are not compressed.
    """

    top = f_max if f_max is not None else sample_rate / 2.0
    edges = torch.linspace(f_min, top, n_filter + 2).tolist()
    return _triangular_bank(edges, n_fft // 2 + 1, sample_rate, dtype)


def dct_matrix(n_out: int, n_in: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Orthonormal DCT-II matrix, `(n_out, n_in)`.

    Applied as `dct @ log_energies`. Orthonormal scaling (the 1/sqrt(4N) and
    1/sqrt(2N) factors) matches `scipy.fftpack.dct(..., norm='ortho')` and
    torchaudio's `create_dct(norm='ortho')`, so coefficients are comparable
    with published LFCC setups.
    """

    if n_out > n_in:
        raise ValueError(f"n_out ({n_out}) cannot exceed n_in ({n_in})")
    rows = torch.arange(n_out, dtype=torch.float64).unsqueeze(1)
    cols = torch.arange(n_in, dtype=torch.float64).unsqueeze(0)
    matrix = torch.cos(math.pi * rows * (2.0 * cols + 1.0) / (2.0 * n_in))
    matrix *= math.sqrt(2.0 / n_in)
    matrix[0] *= math.sqrt(0.5)
    return matrix.to(dtype)


def compute_deltas(features: Tensor, width: int = 5) -> Tensor:
    """Regression-coefficient deltas along the last (time) axis.

    The standard HTK/Kaldi formula

        d_t = sum_{n=1..N} n * (x_{t+n} - x_{t-n}) / (2 * sum_{n=1..N} n^2)

    with `N = (width - 1) // 2` and edge frames replicated. Operates on
    `(B, F, T)` and returns the same shape.
    """

    if width < 3 or width % 2 == 0:
        raise ValueError(f"width must be odd and at least 3, got {width}")
    if features.dim() != 3:
        raise ValueError(
            "compute_deltas expects a 3-D (batch, feature, time) tensor, got "
            f"shape {tuple(features.shape)}"
        )
    half = (width - 1) // 2
    denominator = 2.0 * sum(n * n for n in range(1, half + 1))

    padded = nn.functional.pad(features, (half, half), mode="replicate")
    # conv1d is cross-correlation, so out[t] = sum_k w[k] * x[t + k - half].
    # The weight at offset +n must therefore be +n and at offset -n must be
    # -n, giving an ascending kernel [-N .. 0 .. +N]. A descending kernel
    # produces the negated derivative, which trains just as well and is
    # silently wrong against every other delta implementation.
    kernel = torch.arange(-half, half + 1, dtype=features.dtype, device=features.device)
    kernel = kernel.view(1, 1, -1) / denominator

    shape = padded.shape
    flat = padded.reshape(-1, 1, shape[-1])
    out = nn.functional.conv1d(flat, kernel)
    return out.reshape(*features.shape)


# ---------------------------------------------------------------------------
# Front ends
# ---------------------------------------------------------------------------


class _SpectrogramBase(nn.Module):
    """Shared STFT plumbing: window, power spectrogram, shape handling."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "window", torch.hamming_window(config.win_length), persistent=False
        )

    @staticmethod
    def _to_batched_mono(waveform: Tensor) -> Tensor:
        """Accept `(T)`, `(B, T)` or `(B, C, T)`; return `(B, T)`."""

        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        if waveform.dim() == 2:
            return waveform
        if waveform.dim() == 3:
            if waveform.size(1) == 1:
                return waveform.squeeze(1)
            return waveform.mean(dim=1)  # downmix multi-channel
        raise ValueError(
            f"expected a waveform with 1, 2 or 3 dims, got shape "
            f"{tuple(waveform.shape)}"
        )

    def power_spectrogram(self, waveform: Tensor) -> Tensor:
        """`(B, n_fft // 2 + 1, N)` magnitude-to-the-power spectrogram."""

        config = self.config
        batched = self._to_batched_mono(waveform)
        if batched.size(-1) < config.win_length:
            raise ValueError(
                f"waveform is {batched.size(-1)} samples, shorter than "
                f"win_length={config.win_length}; crop or pad it first"
            )

        window = self.window
        # Autocast would hand torch.stft a half-precision input, which it does
        # not support. Force the transform to run in the window's dtype.
        with torch.autocast(device_type=batched.device.type, enabled=False):
            spectrum = torch.stft(
                batched.to(window.dtype),
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                window=window,
                center=config.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        return spectrum.abs().pow(config.power)

    @staticmethod
    def _normalise(features: Tensor) -> Tensor:
        """Per-utterance, per-coefficient mean and variance normalisation."""

        mean = features.mean(dim=-1, keepdim=True)
        # correction=0 (the biased variance) so a single-frame input gives 0
        # rather than NaN. The clamp then keeps the division finite.
        std = features.var(dim=-1, correction=0, keepdim=True).clamp_min(1e-10).sqrt()
        return (features - mean) / std


class LogMelSpectrogram(_SpectrogramBase):
    """Log-mel filterbank energies, `(B, 1, n_mels, N)`."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        config = config or FeatureConfig(name="logmel")
        super().__init__(config)
        self.register_buffer(
            "filterbank",
            mel_filterbank(
                config.n_mels,
                config.n_fft,
                config.sample_rate,
                config.f_min,
                config.f_max,
            ),
            persistent=False,
        )

    @property
    def n_features(self) -> int:
        return self.config.n_mels

    def forward(self, waveform: Tensor) -> Tensor:
        spectrogram = self.power_spectrogram(waveform)
        energies = torch.matmul(self.filterbank, spectrogram)
        features = torch.log(energies + self.config.log_eps)
        if self.config.normalise:
            features = self._normalise(features)
        return features.unsqueeze(1)


class LFCC(_SpectrogramBase):
    """Linear frequency cepstral coefficients, `(B, 1, F, N)`.

    `F` is `n_ceps`, or `3 * n_ceps` with deltas enabled (static, delta,
    delta-delta stacked along the feature axis in that order).

    With `with_energy` the zeroth cepstral coefficient is replaced by the log
    frame energy of the linear filterbank, which is the usual LFCC convention
    and carries loudness explicitly rather than as a DCT artefact.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        config = config or FeatureConfig(name="lfcc")
        super().__init__(config)
        self.register_buffer(
            "filterbank",
            linear_filterbank(
                config.n_filter,
                config.n_fft,
                config.sample_rate,
                config.f_min,
                config.f_max,
            ),
            persistent=False,
        )
        self.register_buffer(
            "dct", dct_matrix(config.n_ceps, config.n_filter), persistent=False
        )

    @property
    def n_features(self) -> int:
        return self.config.n_ceps * (3 if self.config.with_deltas else 1)

    def forward(self, waveform: Tensor) -> Tensor:
        config = self.config
        spectrogram = self.power_spectrogram(waveform)
        energies = torch.matmul(self.filterbank, spectrogram)
        log_energies = torch.log(energies + config.log_eps)
        cepstra = torch.matmul(self.dct, log_energies)

        if config.with_energy:
            frame_energy = torch.log(energies.sum(dim=1, keepdim=True) + config.log_eps)
            cepstra = torch.cat([frame_energy, cepstra[:, 1:, :]], dim=1)

        if config.with_deltas:
            delta = compute_deltas(cepstra)
            delta2 = compute_deltas(delta)
            cepstra = torch.cat([cepstra, delta, delta2], dim=1)

        if config.normalise:
            cepstra = self._normalise(cepstra)
        return cepstra.unsqueeze(1)


def build_frontend(config: FeatureConfig | None = None) -> _SpectrogramBase:
    """Config to front end. The one place the name string is resolved."""

    config = config or FeatureConfig()
    if config.name == "logmel":
        return LogMelSpectrogram(config)
    if config.name == "lfcc":
        return LFCC(config)
    raise ValueError(f"unknown frontend {config.name!r}")  # pragma: no cover
