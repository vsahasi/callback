"""ASVspoof countermeasure metrics: EER and minimum normalised t-DCF.

Deliberately pure standard library. No numpy, no torch. Three reasons:

1. These numbers are the project's scoreboard. Every reported result depends on
   them, so they must be unit-testable in the default dev environment, today,
   with no optional dependency installed and no dataset on disk.
2. The arithmetic is O(n log n) in the number of trials and runs on score
   lists, not on tensors. There is nothing to vectorise that matters at
   ASVspoof scale (the 2021 LA eval set is about 180k trials; this takes under
   a second).
3. It keeps the scoring path importable from the serving process
   (`detector/`), which has no ML dependencies at all.

Any sequence of floats is accepted: a list, a tuple, a numpy array or a torch
tensor (anything exposing `tolist()` or plain iteration).

Score convention, matching the official ASVspoof packages: a countermeasure
score is higher for bonafide and lower for spoof. Pass bonafide scores as
`target_scores` and spoof scores as `nontarget_scores`.

Reference implementations this file reproduces:

- `compute_det_curve` / `compute_eer`: the `eval_metrics.py` shipped with the
  ASVspoof 2019 evaluation package, reused verbatim in the ASVspoof 2021
  package and in the AASIST and LCNN baselines.
- `compute_tdcf_2019`: `evaluate_tDCF_asvspoof19.py`, the ASVspoof 2019
  tandem detection cost function.
- `compute_tdcf_2021`: the revised t-DCF of the ASVspoof 2021 evaluation plan.

The 2019 form is the one every published LA number is quoted against, so it is
the default. See the note on `compute_tdcf_2021` for what to re-check once the
official 2021 keys and evaluation package are on disk.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ASV_COST_MODEL_2019",
    "ASV_COST_MODEL_2021",
    "AsvErrorRates",
    "CmMetrics",
    "DetCurve",
    "compute_det_curve",
    "compute_eer",
    "compute_tdcf_2019",
    "compute_tdcf_2021",
    "evaluate_cm_scores",
    "obtain_asv_error_rates",
    "per_attack_eer",
    "read_score_file",
    "split_scores_by_label",
    "write_score_file",
]


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

# ASVspoof 2019 evaluation plan, Table 1. Used by evaluate_tDCF_asvspoof19.py.
ASV_COST_MODEL_2019: Mapping[str, float] = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,  # prior of a target (genuine) trial
    "Pnon": (1 - 0.05) * 0.01,  # prior of a nontarget (zero-effort impostor)
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}

# ASVspoof 2021 evaluation plan. Same priors, renamed and reduced costs: the
# ASV miss/false-alarm costs collapse into a single pair and the spoof false
# alarm gets its own cost.
ASV_COST_MODEL_2021: Mapping[str, float] = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss": 1.0,
    "Cfa": 10.0,
    "Cfa_spoof": 10.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_float_list(scores: Any, *, name: str) -> list[float]:
    """Coerce lists, tuples, numpy arrays and torch tensors to list[float]."""

    if scores is None:
        raise ValueError(f"{name} must not be None")
    if hasattr(scores, "detach"):  # torch tensor, drop autograd first
        scores = scores.detach()
    if hasattr(scores, "tolist"):  # numpy array or torch tensor
        scores = scores.tolist()
    if isinstance(scores, (float, int)):
        raise TypeError(f"{name} must be a sequence of scores, got a scalar")
    values = [float(value) for value in scores]
    for value in values:
        if math.isnan(value):
            raise ValueError(f"{name} contains NaN")
    return values


@dataclass(frozen=True)
class DetCurve:
    """A detection error tradeoff curve, one point per candidate threshold.

    `p_miss[i]` is the false rejection rate and `p_fa[i]` the false acceptance
    rate at point `i`, which rejects the `i` lowest-sorted scores and accepts
    the rest. So `thresholds[i]` is the highest REJECTED score and the
    operating rule is "accept when score > threshold", not `>=`. That is a
    detail of the reference implementation worth knowing before comparing a
    threshold against anything: for perfectly separable scores the EER
    threshold comes back as the top spoof score, not the bottom bonafide one.

    The curve has `len(all_scores) + 1` points: one leading point below the
    lowest score, where everything is accepted, plus one per observed score.
    """

    p_miss: tuple[float, ...]
    p_fa: tuple[float, ...]
    thresholds: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.thresholds)


def compute_det_curve(target_scores: Any, nontarget_scores: Any) -> DetCurve:
    """DET curve for a two-class score set.

    A direct transcription of `compute_det_curve` from the official ASVspoof
    `eval_metrics.py`, including its two conventions that matter for exact
    agreement with published numbers:

    * The sort is stable (numpy's `kind="mergesort"`), so on tied scores the
      target trials, which are concatenated first, are ordered before the
      nontarget trials. Python's `sorted` is stable, so sorting
      `(score, group_index)` pairs by score alone reproduces this.
    * The leading threshold is `lowest_score - 0.001`, a sentinel representing
      "accept everything". The 0.001 is not meaningful; it is copied so that
      threshold values returned by `compute_eer` match the reference
      implementation bit for bit.
    """

    targets = _as_float_list(target_scores, name="target_scores")
    nontargets = _as_float_list(nontarget_scores, name="nontarget_scores")

    if not targets:
        raise ValueError("target_scores is empty; cannot form a DET curve")
    if not nontargets:
        raise ValueError("nontarget_scores is empty; cannot form a DET curve")

    n_target = len(targets)
    n_nontarget = len(nontargets)
    n_scores = n_target + n_nontarget

    # label 1 = target (bonafide), label 0 = nontarget (spoof).
    labelled = [(score, 1) for score in targets]
    labelled += [(score, 0) for score in nontargets]
    labelled.sort(key=lambda pair: pair[0])  # stable, so ties keep target first

    p_miss = [0.0]
    p_fa = [1.0]
    thresholds = [labelled[0][0] - 0.001]

    target_seen = 0
    for index, (score, label) in enumerate(labelled, start=1):
        target_seen += label
        nontarget_remaining = n_nontarget - (index - target_seen)
        p_miss.append(target_seen / n_target)
        p_fa.append(nontarget_remaining / n_nontarget)
        thresholds.append(score)

    assert len(p_miss) == n_scores + 1
    return DetCurve(tuple(p_miss), tuple(p_fa), tuple(thresholds))


def compute_eer(target_scores: Any, nontarget_scores: Any) -> tuple[float, float]:
    """Equal error rate and the threshold at which it is reached.

    Returns `(eer, threshold)`. The EER is the mean of the false rejection and
    false acceptance rates at the curve point where they are closest, which is
    the official definition used by every ASVspoof baseline. On a discrete
    score set the two rates rarely coincide exactly, so the mean of the two is
    taken rather than interpolating.

    Known values, all covered by the unit tests:

    * Perfectly separable scores (every bonafide above every spoof): 0.0
    * Interleaved scores (bonafide and spoof alternating): exactly 0.5
    * Fully inverted scores (every spoof above every bonafide): 1.0

    The last is worth knowing: this metric is not folded at 0.5, so a value
    above 0.5 means the score polarity is flipped. If you see 0.9, negate your
    scores rather than celebrating.
    """

    curve = compute_det_curve(target_scores, nontarget_scores)
    best_index = min(
        range(len(curve)),
        key=lambda i: abs(curve.p_miss[i] - curve.p_fa[i]),
    )
    eer = (curve.p_miss[best_index] + curve.p_fa[best_index]) / 2.0
    return eer, curve.thresholds[best_index]


# ---------------------------------------------------------------------------
# ASV error rates, the other half of a tandem metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsvErrorRates:
    """Operating point of the automatic speaker verification system.

    t-DCF is a *tandem* cost: it scores the countermeasure given a fixed ASV
    system. These three rates are that fixed system, measured at the ASV's own
    EER threshold on the same trial set.

    * `p_miss`: target speakers rejected by ASV
    * `p_fa`: zero-effort impostors accepted by ASV
    * `p_miss_spoof`: spoofed trials rejected by ASV (so `1 - p_miss_spoof` is
      the share of spoofs ASV would accept on its own)

    ASVspoof ships the ASV scores needed to compute these alongside the CM
    protocols. Nothing here invents them: `obtain_asv_error_rates` derives them
    from real scores, and no numeric defaults are hardcoded, because a wrong
    hardcoded ASV operating point silently shifts every t-DCF you report.
    """

    p_miss: float
    p_fa: float
    p_miss_spoof: float

    @property
    def p_fa_spoof(self) -> float:
        """Share of spoofed trials ASV accepts. Used by the 2021 cost form."""

        return 1.0 - self.p_miss_spoof

    def __post_init__(self) -> None:
        for field_name in ("p_miss", "p_fa", "p_miss_spoof"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"AsvErrorRates.{field_name} must be in [0, 1], got {value}"
                )


def obtain_asv_error_rates(
    target_asv: Any,
    nontarget_asv: Any,
    spoof_asv: Any,
    asv_threshold: float | None = None,
) -> tuple[AsvErrorRates, float]:
    """ASV error rates at a threshold, defaulting to the ASV EER threshold.

    Mirrors `obtain_asv_error_rates` in the official package. The acceptance
    rule is `score >= threshold`.

    Returns `(rates, threshold_used)`.
    """

    targets = _as_float_list(target_asv, name="target_asv")
    nontargets = _as_float_list(nontarget_asv, name="nontarget_asv")
    spoofs = _as_float_list(spoof_asv, name="spoof_asv")

    if asv_threshold is None:
        _, asv_threshold = compute_eer(targets, nontargets)

    p_fa = sum(1 for s in nontargets if s >= asv_threshold) / len(nontargets)
    p_miss = sum(1 for s in targets if s < asv_threshold) / len(targets)
    if not spoofs:
        raise ValueError(
            "spoof_asv is empty; t-DCF needs the ASV scores of spoofed trials"
        )
    p_miss_spoof = sum(1 for s in spoofs if s < asv_threshold) / len(spoofs)

    return AsvErrorRates(p_miss=p_miss, p_fa=p_fa, p_miss_spoof=p_miss_spoof), (
        asv_threshold
    )


# ---------------------------------------------------------------------------
# t-DCF
# ---------------------------------------------------------------------------


def compute_tdcf_2019(
    target_scores: Any,
    nontarget_scores: Any,
    asv: AsvErrorRates,
    cost_model: Mapping[str, float] = ASV_COST_MODEL_2019,
) -> tuple[float, float]:
    """Minimum normalised t-DCF, ASVspoof 2019 form.

    From the 2019 evaluation plan, the cost at CM threshold t is

        t-DCF(t) = C1 * Pmiss_cm(t) + C2 * Pfa_cm(t)

        C1 = Ptar * (Cmiss_cm - Cmiss_asv * Pmiss_asv)
             - Pnon * Cfa_asv * Pfa_asv
        C2 = Cfa_cm * Pspoof * (1 - Pmiss_spoof_asv)

    normalised by `min(C1, C2)`, which is the cost of the best possible
    threshold-free system (accept everything, or reject everything). So:

    * 0.0 means the countermeasure separates the classes perfectly.
    * 1.0 means it is worth no more than a constant decision.
    * The normalised value can never exceed 1.0, because the curve always
      contains the two endpoints (Pmiss=0, Pfa=1) and (Pmiss=1, Pfa=0).

    Returns `(min_tdcf_norm, threshold)`.

    Raises `ValueError` when the ASV operating point makes C1 or C2 negative or
    makes `min(C1, C2)` zero. Both mean the metric is undefined rather than
    bad, and the official implementation exits on the same conditions. In
    particular an ASV that rejects every spoof on its own (`p_miss_spoof == 1`)
    drives C2 to zero: there is then no spoof left for the countermeasure to
    catch, so there is no cost to minimise.
    """

    curve = compute_det_curve(target_scores, nontarget_scores)

    c1 = (
        cost_model["Ptar"]
        * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * asv.p_miss)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * asv.p_fa
    )
    c2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1.0 - asv.p_miss_spoof)

    if c1 < 0.0 or c2 < 0.0:
        raise ValueError(
            "t-DCF weights must be non-negative; got "
            f"C1={c1!r}, C2={c2!r}. Check the ASV error rates and cost model."
        )
    denominator = min(c1, c2)
    if denominator <= 0.0:
        raise ValueError(
            "t-DCF normalisation is min(C1, C2) and it is zero "
            f"(C1={c1!r}, C2={c2!r}), so the normalised cost is undefined. "
            "This happens when the ASV system alone already rejects every "
            "spoofed trial, or when the priors are degenerate."
        )

    # Strict `<`, so the earliest of several equal minima wins. This matches
    # numpy's argmin in the reference implementation and only affects which
    # threshold is reported, never the cost.
    best_index = 0
    best_cost = math.inf
    for index in range(len(curve)):
        cost = c1 * curve.p_miss[index] + c2 * curve.p_fa[index]
        if cost < best_cost:
            best_cost = cost
            best_index = index

    return best_cost / denominator, curve.thresholds[best_index]


def compute_tdcf_2021(
    target_scores: Any,
    nontarget_scores: Any,
    asv: AsvErrorRates,
    cost_model: Mapping[str, float] = ASV_COST_MODEL_2021,
) -> tuple[float, float]:
    """Minimum normalised t-DCF, ASVspoof 2021 form.

    The 2021 evaluation plan reworked the cost so that the ASV system's own
    errors contribute a threshold-independent floor C0, which the
    countermeasure cannot influence:

        t-DCF(t) = C0 + C1 * Pmiss_cm(t) + C2 * Pfa_cm(t)

        C0 = Ptar * Cmiss * Pmiss_asv + Pnon * Cfa * Pfa_asv
        C1 = Ptar * Cmiss - C0
        C2 = Pspoof * Cfa_spoof * Pfa_spoof_asv

    normalised by `C0 + min(C1, C2)`.

    ASSUMPTION TO VERIFY. This form is written from the 2021 evaluation plan,
    not checked against the official `evaluate_2021_LA.py`, because the keys
    archive is not on disk yet. Once `keys/LA/CM/trial_metadata.txt` and the
    official evaluation package are unpacked, run both on the same score file
    and confirm they agree to about 1e-9. If they do not, this function is the
    thing to fix; `compute_tdcf_2019` and `compute_eer` are transcriptions of
    code that has been stable since 2019 and are far less likely to be wrong.

    Because C0 is a floor the countermeasure cannot reduce, the normalised
    value here does NOT reach 0.0 for a perfect countermeasure. It bottoms out
    at `C0 / (C0 + min(C1, C2))`. That is a property of the 2021 metric, not a
    bug.

    Returns `(min_tdcf_norm, threshold)`.
    """

    curve = compute_det_curve(target_scores, nontarget_scores)

    c0 = (
        cost_model["Ptar"] * cost_model["Cmiss"] * asv.p_miss
        + cost_model["Pnon"] * cost_model["Cfa"] * asv.p_fa
    )
    c1 = cost_model["Ptar"] * cost_model["Cmiss"] - c0
    c2 = cost_model["Pspoof"] * cost_model["Cfa_spoof"] * asv.p_fa_spoof

    if c1 < 0.0 or c2 < 0.0:
        raise ValueError(
            f"t-DCF weights must be non-negative; got C0={c0!r}, C1={c1!r}, C2={c2!r}."
        )
    denominator = c0 + min(c1, c2)
    if denominator <= 0.0:
        raise ValueError(
            "t-DCF normalisation C0 + min(C1, C2) is zero "
            f"(C0={c0!r}, C1={c1!r}, C2={c2!r}); the normalised cost is "
            "undefined."
        )

    best_index = 0
    best_cost = math.inf
    for index in range(len(curve)):
        cost = c0 + c1 * curve.p_miss[index] + c2 * curve.p_fa[index]
        if cost < best_cost:
            best_cost = cost
            best_index = index

    return best_cost / denominator, curve.thresholds[best_index]


# ---------------------------------------------------------------------------
# Convenience: one call, everything a training run wants to log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CmMetrics:
    """Everything a countermeasure evaluation pass reports.

    `min_tdcf` is None when no ASV operating point was supplied. That is the
    honest state for a dev-set pass before the ASV scores are parsed, and it
    keeps the training loop from logging a made-up number.
    """

    eer: float
    eer_threshold: float
    n_bonafide: int
    n_spoof: int
    min_tdcf: float | None = None
    tdcf_threshold: float | None = None
    tdcf_version: str | None = None
    per_attack_eer: Mapping[str, float] | None = None

    def summary(self) -> str:
        parts = [f"EER {self.eer * 100:.3f}%"]
        if self.min_tdcf is not None:
            parts.append(f"min t-DCF {self.min_tdcf:.5f} ({self.tdcf_version})")
        parts.append(f"n={self.n_bonafide}+{self.n_spoof}")
        return "  ".join(parts)


def per_attack_eer(
    bonafide_scores: Any,
    spoof_scores_by_attack: Mapping[str, Any],
) -> dict[str, float]:
    """EER of each attack in isolation, against the shared bonafide pool.

    This is how ASVspoof reports per-attack breakdowns: the bonafide trials are
    not partitioned, every attack is scored against all of them. Attacks with
    no trials are skipped rather than reported as 0.0.
    """

    bonafide = _as_float_list(bonafide_scores, name="bonafide_scores")
    results: dict[str, float] = {}
    for attack, scores in sorted(spoof_scores_by_attack.items()):
        values = _as_float_list(scores, name=f"spoof_scores[{attack}]")
        if not values:
            continue
        results[attack] = compute_eer(bonafide, values)[0]
    return results


def evaluate_cm_scores(
    bonafide_scores: Any,
    spoof_scores: Any,
    *,
    asv: AsvErrorRates | None = None,
    tdcf_version: str = "2019",
    spoof_scores_by_attack: Mapping[str, Any] | None = None,
) -> CmMetrics:
    """EER plus, when an ASV operating point is given, minimum t-DCF."""

    bonafide = _as_float_list(bonafide_scores, name="bonafide_scores")
    spoof = _as_float_list(spoof_scores, name="spoof_scores")

    eer, eer_threshold = compute_eer(bonafide, spoof)

    min_tdcf: float | None = None
    tdcf_threshold: float | None = None
    version: str | None = None
    if asv is not None:
        if tdcf_version == "2019":
            min_tdcf, tdcf_threshold = compute_tdcf_2019(bonafide, spoof, asv)
        elif tdcf_version == "2021":
            min_tdcf, tdcf_threshold = compute_tdcf_2021(bonafide, spoof, asv)
        else:
            raise ValueError(
                f"tdcf_version must be '2019' or '2021', got {tdcf_version!r}"
            )
        version = tdcf_version

    breakdown = (
        per_attack_eer(bonafide, spoof_scores_by_attack)
        if spoof_scores_by_attack
        else None
    )

    return CmMetrics(
        eer=eer,
        eer_threshold=eer_threshold,
        n_bonafide=len(bonafide),
        n_spoof=len(spoof),
        min_tdcf=min_tdcf,
        tdcf_threshold=tdcf_threshold,
        tdcf_version=version,
        per_attack_eer=breakdown,
    )


# ---------------------------------------------------------------------------
# Score file I/O, in the ASVspoof submission format
# ---------------------------------------------------------------------------


def write_score_file(
    path: str | Path,
    utterance_ids: Iterable[str],
    scores: Iterable[float],
) -> Path:
    """Write `utt_id score` lines, the format the official packages consume.

    Keeping this here means a training run can emit a file that the official
    ASVspoof evaluation scripts read directly, which is the cross-check that
    proves this module right once the real keys are on disk.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ids = list(utterance_ids)
    values = list(scores)
    if len(ids) != len(values):
        raise ValueError(f"got {len(ids)} utterance ids but {len(values)} scores")
    with target.open("w", encoding="utf-8") as handle:
        for utterance_id, score in zip(ids, values, strict=True):
            handle.write(f"{utterance_id} {float(score):.6f}\n")
    return target


def read_score_file(path: str | Path) -> dict[str, float]:
    """Read `utt_id score` lines back. Blank lines and `#` comments skipped."""

    scores: dict[str, float] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(
                f"{path}:{line_number}: expected 'utt_id score', got {raw!r}"
            )
        # Take the last field as the score: some submissions carry extra
        # columns between the id and the score.
        scores[fields[0]] = float(fields[-1])
    return scores


def split_scores_by_label(
    scores: Mapping[str, float],
    labels: Mapping[str, str],
    *,
    bonafide_label: str = "bonafide",
) -> tuple[list[float], list[float], list[str]]:
    """Split a score dict into bonafide and spoof lists using a label map.

    Returns `(bonafide, spoof, missing)` where `missing` lists utterance ids
    present in `labels` but absent from `scores`. A silently truncated score
    file is the classic way to report an EER that is too good, so the caller is
    handed the gap instead of having it hidden.
    """

    bonafide: list[float] = []
    spoof: list[float] = []
    missing: list[str] = []
    for utterance_id, label in labels.items():
        if utterance_id not in scores:
            missing.append(utterance_id)
            continue
        if label == bonafide_label:
            bonafide.append(scores[utterance_id])
        else:
            spoof.append(scores[utterance_id])
    return bonafide, spoof, missing
