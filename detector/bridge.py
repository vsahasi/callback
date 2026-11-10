"""The seam between the /events pipeline and the /v1 policy.

The agent posts one speech segment per caller turn to POST /events, under the
same session id `/v1/sessions` returned, and `DetectorEngine` scores them.
Nothing used to carry those scores back to `/v1`, so real mode reached the
policy with every signal unset and denied tiers 1 and 2 with `no_audio`.

This module is where the two contracts meet, and the only place they are
allowed to. `detector/v1` serves `callback_shared` and must never import the
internal `shared.contract`, because the two define different models under the
same names and a stray import would put an internal shape on the agent's wire
(see tests/v1/test_imports.py). So the translation happens here and `/v1`
receives `EngineSignals`, which is nothing but floats, a bool and, when the
engine has already decided, one of the policy's own deny codes.

Polarity is the thing to get right. A contract `Factor` scores 1.0 for "fully
consistent with the claimed identity", so the policy's `voiceprint`, which is
also high-good, passes straight through, while its `spoof` and `agent` are
risks and come back inverted.

One thing crosses by name rather than as a number. The engine's cross-leg
check (`detector.legs`) compares whoever rang in against whoever answered the
callback to the number on file, and a confirmed mismatch is the engine's own
fatal deny: no one-time code or callback can rescue it, because they would
reach the phone the real vendor is holding. As a number it arrived at the
policy as a voiceprint under 0.45, which is one voice fail of the two the
policy wants before it says `speaker_mismatch`, so the policy stepped the
caller up for exactly the factors the mismatch says prove nothing. So the
confirmed mismatch is carried as `EngineSignals.fatal`, the deny code the
policy must return and lock the session with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callback_shared.contract import DenyReason

from detector.errors import DetectorError
from shared.contract import FactorName, FactorVerdict

#: The signals the /v1 policy reads. One gate per question it asks.
SPOOF_SIGNALS = (
    FactorName.SYNTHETIC_SPEECH,
    FactorName.REPLAY,
    FactorName.LIVENESS,
)

#: The deny code a confirmed cross-leg mismatch crosses the seam as. The
#: policy's own code for "this is not the claimed speaker", so the agent sees
#: one name for it whichever side found it.
SPEAKER_MISMATCH: str = DenyReason.SPEAKER_MISMATCH.value


@dataclass(frozen=True)
class EngineSignals:
    """What /events measured for one session, in the /v1 policy's terms.

    `heard` is what `V1Service.listening` reports, so an empty one still
    denies `no_audio`. The three numbers follow the policy's polarity:
    `voiceprint` is high-good, `spoof` and `agent` are high-bad.

    `fatal` is a deny code the engine has already settled on, or None. Set
    only for a confirmed cross-leg mismatch (`Session.cross_leg.confirmed`),
    and the policy denies with it, locking the session, whatever else it
    reads: challenges, codes and callbacks are what it says cannot help.
    """

    heard: bool = False
    voiceprint: float | None = None
    spoof: float | None = None
    agent: float | None = None
    fatal: str | None = None


NOT_HEARD = EngineSignals()


def signals_from_engine(engine: Any | None, session_id: str) -> EngineSignals:
    """Read what /events scored for `session_id`, or NOT_HEARD.

    `engine` is a `detector.state.DetectorEngine`, typed loosely so this
    module does not import it. None, an unknown session, or a session where
    nothing has been scored yet all come back NOT_HEARD, which leaves the
    policy exactly where it was before this bridge: denying `no_audio`.

    Two reads: the factor list (`factors_for`) and the session's cross-leg
    record (`cross_leg_for`). The record is read on its own because the
    factor it emits competes with the enrolment comparison in worst-score
    merging and can be replaced there; the session's own field cannot.
    """

    if engine is None:
        return NOT_HEARD
    try:
        answer = engine.factors_for(session_id)
    except DetectorError:
        return NOT_HEARD  # nothing posted under this id yet
    check = engine.cross_leg_for(session_id)
    return signals_from_factors(
        answer.factors,
        speaker_mismatch=check is not None and bool(check.confirmed),
    )


def signals_from_factors(
    factors: Any, *, speaker_mismatch: bool = False
) -> EngineSignals:
    """The mapping itself, over an iterable of contract Factors.

    `speaker_mismatch` is whether the session's cross-leg check is confirmed.
    It sets `fatal`, and counts as heard on its own: a confirmed mismatch
    means both legs were measured, whatever the factor list says.
    """

    scored = {
        factor.name: factor.score
        for factor in factors
        if factor.verdict is not FactorVerdict.NOT_RUN
    }
    if not scored and not speaker_mismatch:
        return NOT_HEARD
    # The policy has one spoof gate, and three signals answer the question it
    # asks: is this a live human speaking now. Take the worst of whichever
    # ran, so a replay or a dead-room liveness failure still sinks the call
    # when the synthesis check alone looks clean. That is the same rule
    # `detector.scoring.first_failing_factor` applies on the /events side.
    spoof_scores = [scored[name] for name in SPOOF_SIGNALS if name in scored]
    agent = scored.get(FactorName.AGENT_BEHAVIOUR)
    return EngineSignals(
        heard=True,
        voiceprint=scored.get(FactorName.VOICE_MATCH),
        spoof=1.0 - min(spoof_scores) if spoof_scores else None,
        agent=None if agent is None else 1.0 - agent,
        fatal=SPEAKER_MISMATCH if speaker_mismatch else None,
    )
