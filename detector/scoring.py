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

# A `voice_match` factor whose detail opens with this is the cross-leg check
# in `detector.legs` reporting that the person who rang in and the person who
# answered the callback are not the same voice. That is fatal too, and for a
# reason the blend cannot see: the callback leg is the vendor by construction,
# so a mismatch is the impostor case even when every other signal is clean,
# and a one-time code sent to the vendor's phone proves nothing about the
# caller. The marker is the detail prefix rather than a new `FactorName`
# because the wire contract is frozen by a schema snapshot, and only this
# service writes factor details.
SPEAKER_MISMATCH_PREFIX = "speaker_mismatch:"

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
        # Nothing could be scored. This used to return 0.0, which decide()
        # reads as ALLOW, so a total inference failure authenticated the
        # caller. "No evidence" and "perfect evidence" must never be the same
        # number in a verification service: fail closed at maximum risk.
        return 1.0
    weighted = sum(f.weight * (1.0 - f.score) for f in scored)
    return _clamp(weighted / total_weight)


def blend_risk_with_otp(behavioural: list[Factor], otp_factor: Factor) -> float:
    """Blend behavioural factors with a resolved OTP factor."""

    base = blend_risk(behavioural)
    otp_risk = 1.0 - otp_factor.score
    return _clamp((1.0 - OTP_WEIGHT) * base + OTP_WEIGHT * otp_risk)


def first_failing_factor(factors: list[Factor]) -> Factor | None:
    """The first factor that outright failed, if any.

    Distinct from `fatal_factor`, which is about two specific signals being
    unambiguous attack evidence. This is weaker and broader: any detector
    saying FAIL means something is wrong, and a wrong-looking call must not be
    waved through just because the weighted average came out low.

    That matters most when only some detectors ran. Weights are renormalised
    over the factors that produced a score, so a single failing signal can be
    diluted below the allow threshold by a handful of passing ones, or
    conversely carry far more weight than its nominal share. Neither should be
    able to turn a FAIL into an allow.
    """

    for factor in factors:
        if factor.verdict is FactorVerdict.FAIL:
            return factor
    return None


def speaker_mismatch(factors: list[Factor]) -> Factor | None:
    """The cross-leg `voice_match` failure, if the legs were found to differ.

    Only a FAIL verdict carrying the `SPEAKER_MISMATCH_PREFIX` counts. An
    enrolment-based `voice_match` failure is not one: that case is what a
    step-up exists for, and it keeps its challenge.
    """

    for factor in factors:
        if (
            factor.name is FactorName.VOICE_MATCH
            and factor.verdict is FactorVerdict.FAIL
            and (factor.detail or "").startswith(SPEAKER_MISMATCH_PREFIX)
        ):
            return factor
    return None


def fatal_factor(factors: list[Factor]) -> Factor | None:
    """The first unambiguous-attack factor, if any tripped.

    Two kinds: a replay or agent-behaviour signal at or below
    `FATAL_THRESHOLD`, and a cross-leg speaker mismatch (see
    `speaker_mismatch`).
    """

    for factor in factors:
        if factor.name in FATAL_FACTORS and factor.score <= FATAL_THRESHOLD:
            return factor
    return speaker_mismatch(factors)


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
