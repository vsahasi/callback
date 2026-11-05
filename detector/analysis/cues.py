"""Cue arithmetic: how a measurement becomes a score, a verdict and a detail.

Every detector in this package works the same way. It takes a handful of
physical measurements, maps each onto a penalty in [0, 1] with an explicit
innocent-to-guilty ramp, blends the penalties by weight, and turns the result
into a `Factor`-shaped triple.

Two rules are enforced here rather than left to each detector:

* A score is "consistency with the claimed identity", matching
  `shared.contract.Factor`. 1.0 means nothing suspicious was found, 0.0 means
  the evidence is damning. Penalties are therefore inverted on the way out.
* An unmeasurable cue contributes nothing. It is dropped from the blend, not
  scored as innocent. "I could not look" and "I looked and it was clean" are
  different facts, and collapsing them is how a detector ends up confidently
  allowing a fraudulent payment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from detector.analysis.dsp import clamp01
from shared.contract import FactorVerdict

__all__ = [
    "Cue",
    "DetectorResult",
    "blend",
    "inconclusive",
    "not_run",
    "ramp",
    "summarise",
    "temper",
    "verdict_for",
]

# A score at or above this is a clean bill of health, at or below the lower
# bound is an accusation, and the band between is honest uncertainty. These
# are deliberately wide: the detectors here are uncalibrated against real
# attack audio, so the inconclusive band is where most short, noisy calls
# belong.
PASS_AT_OR_ABOVE = 0.62
FAIL_AT_OR_BELOW = 0.32

# A one-sided cue below this penalty is treated as silent rather than as a
# vote of confidence. See `Cue.one_sided`.
ONE_SIDED_FLOOR = 0.20


def ramp(value: float, innocent: float, guilty: float) -> float:
    """Map a measurement onto a penalty in [0, 1].

    0.0 at `innocent`, 1.0 at `guilty`, linear in between, clamped outside.
    Works in either direction, so `ramp(x, 2.0, 0.2)` penalises small values
    and `ramp(x, 0.2, 2.0)` penalises large ones.
    """

    if not np.isfinite(value) or not np.isfinite(innocent) or not np.isfinite(guilty):
        return 0.0
    if innocent == guilty:
        return 0.0
    return clamp01((value - innocent) / (guilty - innocent))


@dataclass(frozen=True)
class Cue:
    """One measurement and what it implies.

    `measurable` False means the recording could not support the measurement
    (no voiced speech, no silence to look at, band outside Nyquist). Such a
    cue keeps its value for the calibration record but is excluded from the
    blend.

    `one_sided` True means the cue is evidence of guilt only, and its
    absence proves nothing. A bot that answers the question correctly is not
    thereby shown to be human, so `prompt_tracking` not firing must not vote
    innocent and dilute the cues that did fire. Such a cue is dropped from
    the blend when its penalty is negligible, exactly as an unmeasurable one
    is.
    """

    name: str
    value: float
    penalty: float
    weight: float
    note: str = ""
    measurable: bool = True
    one_sided: bool = False

    @property
    def counts(self) -> bool:
        if not self.measurable:
            return False
        return not (self.one_sided and self.penalty < ONE_SIDED_FLOOR)

    @property
    def contribution(self) -> float:
        return self.weight * self.penalty if self.counts else 0.0


@dataclass(frozen=True)
class DetectorResult:
    """What one detector concluded, before it becomes a `Factor`."""

    score: float
    verdict: FactorVerdict
    detail: str
    measurements: dict[str, float] = field(default_factory=dict)
    cues: tuple[Cue, ...] = ()


def blend(
    cues: list[Cue] | tuple[Cue, ...], floor: float = 0.0, sharpness: float = 2.2
) -> float:
    """Combine cue penalties into a score in [0, 1].

        evidence = sum(weight * penalty) / sum(weight)   over measurable cues
        score    = max(exp(-sharpness * evidence), floor)

    The exponential, rather than a plain weighted mean, is there because
    independent evidence compounds. Three cues each pointing at a
    loudspeaker in the path is not the average of three opinions, it is three
    multiplicative likelihood ratios, and a weighted mean can never be more
    damning than its loudest term. `sharpness` sets how fast agreement
    compounds; it is tuned per detector against the synthetic corpus and
    printed by the calibration harness.

    `floor` is the lowest score the blend may return. The replay and
    agent_behaviour factors are fatal at or below 0.10 in
    `detector.scoring`, so those detectors keep the floor above that and
    lower it only when several independent cues agree. One noisy measurement
    must not be able to deny a payment on its own.

    Unmeasurable cues are dropped from both sums, so a cue that could not be
    computed neither helps nor hurts.
    """

    usable = [cue for cue in cues if cue.counts and cue.weight > 0.0]
    total_weight = sum(cue.weight for cue in usable)
    if total_weight <= 0.0:
        return 0.5
    evidence = sum(cue.contribution for cue in usable) / total_weight
    score = float(np.exp(-max(sharpness, 0.0) * clamp01(evidence)))
    return clamp01(max(score, floor))


def temper(score: float, *, confident: bool, shrink: float = 0.4) -> float:
    """Pull an unconfident score towards 0.5.

    `detector.scoring.blend_risk` includes `inconclusive` factors at their
    face value; only `not_run` is excluded. So a detector that says
    "inconclusive" while reporting 0.03 still denies the payment, which
    defeats the point of the verdict. Tempering makes the number agree with
    the word: an unconfident detector contributes a weak opinion, not a
    confident one wearing a disclaimer.
    """

    if confident:
        return clamp01(score)
    return clamp01(0.5 + (clamp01(score) - 0.5) * shrink)


def verdict_for(score: float, *, confident: bool) -> FactorVerdict:
    """Score to verdict, with an explicit confidence gate.

    `confident` False forces `inconclusive` whatever the score says. Callers
    pass False when the recording is too short, too quiet or too degraded for
    the measurements to mean anything, which is the common case on three
    seconds of 8 kHz telephone audio.
    """

    if not confident:
        return FactorVerdict.INCONCLUSIVE
    if score >= PASS_AT_OR_ABOVE:
        return FactorVerdict.PASS
    if score <= FAIL_AT_OR_BELOW:
        return FactorVerdict.FAIL
    return FactorVerdict.INCONCLUSIVE


def inconclusive(
    reason: str, measurements: dict[str, float] | None = None
) -> DetectorResult:
    """A detector that ran but learned nothing. Score 0.5, no opinion."""

    return DetectorResult(
        score=0.5,
        verdict=FactorVerdict.INCONCLUSIVE,
        detail=reason[:500],
        measurements=measurements or {},
    )


def not_run(reason: str) -> DetectorResult:
    """A detector that could not run at all.

    Score 0.5 is recorded for the audit trail, but a `not_run` factor is
    excluded from the risk blend by `detector.scoring.blend_risk`, so the
    number never reaches a decision.
    """

    return DetectorResult(
        score=0.5,
        verdict=FactorVerdict.NOT_RUN,
        detail=reason[:500],
        measurements={},
    )


def summarise(headline: str, cues: list[Cue] | tuple[Cue, ...], limit: int = 3) -> str:
    """Build a `Factor.detail` string: headline, then the loudest evidence.

    The detail lands in front of a human deciding whether to release a held
    payment, so it names the measurement and its value, not just a label.
    """

    ranked = sorted(
        (cue for cue in cues if cue.measurable and cue.note),
        key=lambda cue: cue.contribution,
        reverse=True,
    )
    material = [cue for cue in ranked if cue.penalty >= 0.25][:limit]
    if not material:
        clean = [cue.note for cue in ranked[:limit]]
        body = "; ".join(clean) if clean else "no measurable anomalies"
        return f"{headline} {body}."[:500]
    body = "; ".join(cue.note for cue in material)
    agreeing = sum(1 for cue in cues if cue.measurable and cue.penalty >= 0.5)
    measured = sum(1 for cue in cues if cue.measurable)
    return f"{headline} {body}. {agreeing} of {measured} cues flagged."[:500]
