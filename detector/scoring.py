"""Risk blending and the decision thresholds.

Shared by mock mode and (later) real model mode, so a scenario fixture and a
real inference path cannot disagree about how a set of factors becomes a
decision.
"""

from __future__ import annotations

from shared.contract import Decision, Factor, FactorName, FactorVerdict

# Blended-risk thresholds. Below ALLOW_BELOW the caller passes, at or above
# DENY_AT_OR_ABOVE the caller is refused, anything between triggers OTP step-up.
ALLOW_BELOW = 0.30
DENY_AT_OR_ABOVE = 0.55

# Some signals are unambiguous attack evidence. If one of these scores at or
# below FATAL_THRESHOLD the call is denied regardless of the blended score,
# because no amount of good voice match should rescue a detected replay.
FATAL_FACTORS: frozenset[FactorName] = frozenset(
    {FactorName.REPLAY, FactorName.AGENT_BEHAVIOUR}
)
FATAL_THRESHOLD = 0.10

# When OTP step-up resolves, it carries this share of the blended score and the
# behavioural factors are scaled to the remainder.
OTP_WEIGHT = 0.40


def blend_risk(factors: list[Factor]) -> float:
    """Weighted mean of per-factor risk, where risk is 1 - score.

    Weights are renormalised, so callers do not have to make them sum to 1.
    """

    scored = [f for f in factors if f.verdict is not FactorVerdict.NOT_RUN]
    total_weight = sum(f.weight for f in scored)
    if total_weight <= 0.0:
        return 0.0
    weighted = sum(f.weight * (1.0 - f.score) for f in scored)
    return _clamp(weighted / total_weight)


def blend_risk_with_otp(behavioural: list[Factor], otp_factor: Factor) -> float:
    """Blend behavioural factors with a resolved OTP factor."""

    base = blend_risk(behavioural)
    otp_risk = 1.0 - otp_factor.score
    return _clamp((1.0 - OTP_WEIGHT) * base + OTP_WEIGHT * otp_risk)


def fatal_factor(factors: list[Factor]) -> Factor | None:
    """The first unambiguous-attack factor, if any tripped."""

    for factor in factors:
        if factor.name in FATAL_FACTORS and factor.score <= FATAL_THRESHOLD:
            return factor
    return None


def decide(risk_score: float, *, otp_available: bool = True) -> Decision:
    """Map a blended risk score onto a decision.

    When OTP step-up is no longer available (attempts exhausted or already
    used), the middle band falls closed to DENY rather than looping.
    """

    if risk_score < ALLOW_BELOW:
        return Decision.ALLOW
    if risk_score >= DENY_AT_OR_ABOVE:
        return Decision.DENY
    return Decision.CHALLENGE if otp_available else Decision.DENY


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
