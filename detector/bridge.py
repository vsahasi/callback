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
receives `EngineSignals`, which is nothing but floats and a bool.

Polarity is the thing to get right. A contract `Factor` scores 1.0 for "fully
consistent with the claimed identity", so the policy's `voiceprint`, which is
also high-good, passes straight through, while its `spoof` and `agent` are
risks and come back inverted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from detector.errors import DetectorError
from shared.contract import FactorName, FactorVerdict

#: The signals the /v1 policy reads. One gate per question it asks.
SPOOF_SIGNALS = (
    FactorName.SYNTHETIC_SPEECH,
    FactorName.REPLAY,
    FactorName.LIVENESS,
)


@dataclass(frozen=True)
class EngineSignals:
    """What /events measured for one session, in the /v1 policy's terms.

    `heard` is what `V1Service.listening` reports, so an empty one still
    denies `no_audio`. The three numbers follow the policy's polarity:
    `voiceprint` is high-good, `spoof` and `agent` are high-bad.
    """

    heard: bool = False
    voiceprint: float | None = None
    spoof: float | None = None
    agent: float | None = None


NOT_HEARD = EngineSignals()


def signals_from_engine(engine: Any | None, session_id: str) -> EngineSignals:
    """Read what /events scored for `session_id`, or NOT_HEARD.

    `engine` is a `detector.state.DetectorEngine`, typed loosely so this
    module does not import it. None, an unknown session, or a session where
    nothing has been scored yet all come back NOT_HEARD, which leaves the
    policy exactly where it was before this bridge: denying `no_audio`.
    """

    if engine is None:
        return NOT_HEARD
    try:
        answer = engine.factors_for(session_id)
    except DetectorError:
        return NOT_HEARD  # nothing posted under this id yet
    return signals_from_factors(answer.factors)


def signals_from_factors(factors: Any) -> EngineSignals:
    """The mapping itself, over an iterable of contract Factors."""

    scored = {
        factor.name: factor.score
        for factor in factors
        if factor.verdict is not FactorVerdict.NOT_RUN
    }
    if not scored:
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
    )
