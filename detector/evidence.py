"""What the state machine knows about a caller, and where it came from.

`detector.state` used to read `detector.mock.SCRIPTS` directly, in two places.
That was the whole reason real mode did not exist: there was no seam, so there
was nowhere for measured factors to enter. This module is that seam. Both
modes produce the same `Evidence` value and the state machine stops caring
which one it got.

    mock mode   scripted_evidence(scenario)         reads SCRIPTS
    real mode   analysed_evidence(factors, prior)   from detector.analysis

Combining spans
---------------
A call arrives as several `speech_segment` events, each with its own span of
audio. The analysis engine is deliberately stateless and scores one span at a
time, so something has to combine them.

This takes the lowest score seen per factor, and the matching verdict. Two
reasons. It is the conservative direction, which is the direction this service
leans everywhere else: one clearly bad span is evidence, and averaging it
against three clean ones is how a short burst of a replayed recording gets
smoothed away. And it needs no state of its own, because the running factors
already live on the session, so it behaves the same whether one container
handles the call or four do.

The cost is real and worth naming: a single unlucky span pulls the whole call
down and there is no way back up. For a thirty-second verification call with
a handful of spans that is the right trade. For a long call it would not be.

Deriving otp_can_rescue
-----------------------
In mock mode each scenario says whether a one-time code should be able to
rescue the caller. In real mode nothing says it, so it is derived: a code
cannot rescue a call where the replay or the agent-behaviour signal failed.
Those are the two signals the scoring layer already treats as different in
kind, and the reasoning carries over. Someone playing a recording down the
line, or a bot driving the conversation, is not made legitimate by also
holding the phone that receives the code. A failing voice or synthesis signal
is different: that is the case a code exists for.
"""

from __future__ import annotations

from dataclasses import dataclass

from detector.mock import SCRIPTS
from detector.scoring import speaker_mismatch
from shared.contract import Factor, FactorName, FactorVerdict, Scenario

__all__ = [
    "NO_AUDIO_REASON",
    "Evidence",
    "analysed_evidence",
    "combine_factors",
    "empty_evidence",
    "scripted_evidence",
]


# The two signals a correct one-time code cannot talk its way past. Same pair
# the scoring layer treats as fatal below 0.10, for the same reason.
UNRESCUABLE: frozenset[FactorName] = frozenset(
    {FactorName.REPLAY, FactorName.AGENT_BEHAVIOUR}
)

NO_AUDIO_REASON = (
    "No audio could be analysed for this call, so there is nothing to "
    "verify the caller against. Failing closed."
)


@dataclass(frozen=True)
class Evidence:
    """Everything the state machine needs to reach a decision.

    Identical in shape whether it was scripted or measured, which is the
    point: `_evaluate` has no branch on mock mode.
    """

    factors: list[Factor]
    confidence: float
    reason: str
    otp_can_rescue: bool

    @property
    def is_empty(self) -> bool:
        return not self.factors


def scripted_evidence(scenario: Scenario) -> Evidence:
    """Mock mode. The five scripted scenarios, unchanged."""

    script = SCRIPTS[scenario]
    return Evidence(
        factors=script.factor_list(),
        confidence=script.confidence,
        reason=script.reason,
        otp_can_rescue=script.otp_can_rescue,
    )


def empty_evidence(reason: str = NO_AUDIO_REASON) -> Evidence:
    """Real mode with nothing measured.

    An empty factor list blends to 1.0, not 0.0, so this denies. That is the
    fix for the second fail-open: a detector that scored nothing used to read
    as a detector that found nothing wrong.
    """

    return Evidence(factors=[], confidence=0.0, reason=reason, otp_can_rescue=False)


def combine_factors(previous: list[Factor], incoming: list[Factor]) -> list[Factor]:
    """Merge a new span's factors into the running set, worst score wins.

    Order follows `incoming` first so the engine's stable factor order is
    preserved, then anything only `previous` had, so a detector that ran on an
    earlier span and not on this one is not quietly dropped.
    """

    merged: dict[FactorName, Factor] = {f.name: f for f in previous}
    for factor in incoming:
        held = merged.get(factor.name)
        if held is None:
            merged[factor.name] = factor
            continue
        # NOT_RUN carries a placeholder score, so it never wins on being low.
        if held.verdict is FactorVerdict.NOT_RUN:
            merged[factor.name] = factor
        elif factor.verdict is FactorVerdict.NOT_RUN:
            continue
        elif factor.score < held.score:
            merged[factor.name] = factor

    ordered: list[Factor] = []
    seen: set[FactorName] = set()
    for factor in incoming:
        ordered.append(merged[factor.name])
        seen.add(factor.name)
    for factor in previous:
        if factor.name not in seen:
            ordered.append(merged[factor.name])
            seen.add(factor.name)
    return ordered


def _confidence_for(factors: list[Factor]) -> float:
    """How much of the picture actually got measured.

    The share of total weight that produced a real verdict. Three of five
    detectors running is a genuinely less confident answer than five, and the
    reviewer should see that rather than a number that looks the same either
    way.
    """

    if not factors:
        return 0.0
    total = sum(f.weight for f in factors)
    if total <= 0.0:
        return 0.0
    scored = sum(f.weight for f in factors if f.verdict is not FactorVerdict.NOT_RUN)
    return round(scored / total, 4)


def _reason_for(factors: list[Factor], spans: int) -> str:
    """One sentence a reviewer can read, built from what actually happened."""

    failing = [f for f in factors if f.verdict is FactorVerdict.FAIL]
    ran = [f for f in factors if f.verdict is not FactorVerdict.NOT_RUN]
    not_run = [f for f in factors if f.verdict is FactorVerdict.NOT_RUN]

    span_word = "span" if spans == 1 else "spans"
    scope = f"{len(ran)} of {len(factors)} signals over {spans} audio {span_word}"

    if failing:
        worst = min(failing, key=lambda f: f.score)
        others = [f.name.value for f in failing if f.name is not worst.name]
        tail = f", and also {', '.join(others)}" if others else ""
        detail = worst.detail or "no detail recorded"
        return (
            f"{worst.name.value} failed at {worst.score:.2f}{tail}. {detail} ({scope})"
        )

    if not_run:
        missing = ", ".join(f.name.value for f in not_run)
        return (
            f"No signal failed, but {missing} could not be measured, so this is "
            f"a partial answer ({scope})"
        )

    return f"Every signal passed ({scope})"


def analysed_evidence(
    incoming: list[Factor],
    previous: list[Factor] | None = None,
    *,
    spans: int = 1,
) -> Evidence:
    """Real mode. Measured factors, merged with whatever came before.

    `spans` is how many audio spans have now been analysed for this call. It
    only reaches the reason string, so a reviewer can tell a verdict drawn
    from one span from one drawn from four.
    """

    factors = combine_factors(list(previous or []), list(incoming))
    if not factors:
        return empty_evidence()

    # A cross-leg speaker mismatch is unrescuable for the same reason the two
    # names in UNRESCUABLE are: the code would go to the phone the real vendor
    # is holding, which says nothing about who rang in.
    rescuable = speaker_mismatch(factors) is None and not any(
        f.name in UNRESCUABLE and f.verdict is FactorVerdict.FAIL for f in factors
    )
    return Evidence(
        factors=factors,
        confidence=_confidence_for(factors),
        reason=_reason_for(factors, spans),
        otp_can_rescue=rescuable,
    )
