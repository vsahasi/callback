"""The five scripted scenarios.

Each script is a fixed set of factor readings. They run through the real state
machine and the real scoring code in detector.scoring, so a fixture that passes
here exercises the same decision path a real model will.

Factor weights within a script sum to 1.0. Scores are "consistency with the
claimed identity": 1.0 fully consistent, 0.0 fully inconsistent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shared.contract import (
    Factor,
    FactorName,
    FactorVerdict,
    Scenario,
)


@dataclass(frozen=True)
class ScenarioScript:
    scenario: Scenario
    factors: tuple[Factor, ...]
    confidence: float
    reason: str
    # True when a correct OTP should rescue the caller. False for scenarios the
    # service refuses outright (detected replay, a non-human agent on the line).
    otp_can_rescue: bool

    def factor_list(self) -> list[Factor]:
        return [f.model_copy(deep=True) for f in self.factors]


def _f(
    name: FactorName,
    score: float,
    weight: float,
    verdict: FactorVerdict,
    detail: str,
) -> Factor:
    return Factor(name=name, score=score, weight=weight, verdict=verdict, detail=detail)


SCRIPTS: dict[Scenario, ScenarioScript] = {
    Scenario.REAL_CFO: ScenarioScript(
        scenario=Scenario.REAL_CFO,
        factors=(
            _f(FactorName.VOICE_MATCH, 0.96, 0.35, FactorVerdict.PASS,
               "Speaker embedding matches the enrolled reference."),
            _f(FactorName.SYNTHETIC_SPEECH, 0.97, 0.25, FactorVerdict.PASS,
               "No vocoder or codec artefacts detected."),
            _f(FactorName.REPLAY, 0.95, 0.15, FactorVerdict.PASS,
               "Channel response consistent with a live handset."),
            _f(FactorName.AGENT_BEHAVIOUR, 0.94, 0.15, FactorVerdict.PASS,
               "Turn-taking latency and prosody in the human range."),
            _f(FactorName.LIVENESS, 0.93, 0.10, FactorVerdict.PASS,
               "Spontaneous breath and filler events present."),
        ),
        confidence=0.95,
        reason="Voice matches the enrolled reference with no spoofing indicators.",
        otp_can_rescue=True,
    ),
    Scenario.CLONED_VOICE: ScenarioScript(
        scenario=Scenario.CLONED_VOICE,
        factors=(
            _f(FactorName.VOICE_MATCH, 0.88, 0.35, FactorVerdict.PASS,
               "Embedding matches, which a high-quality clone is expected to do."),
            _f(FactorName.SYNTHETIC_SPEECH, 0.08, 0.25, FactorVerdict.FAIL,
               "Vocoder phase artefacts and unnaturally flat spectral tilt."),
            _f(FactorName.REPLAY, 0.82, 0.15, FactorVerdict.PASS,
               "No replay channel signature."),
            _f(FactorName.AGENT_BEHAVIOUR, 0.55, 0.15, FactorVerdict.INCONCLUSIVE,
               "Turn-taking slightly fast but within tolerance."),
            _f(FactorName.LIVENESS, 0.30, 0.10, FactorVerdict.FAIL,
               "No breath events across the sampled segments."),
        ),
        confidence=0.86,
        reason="Voice matches but the audio shows synthesis artefacts. Step-up required.",
        otp_can_rescue=True,
    ),
    Scenario.REPLAY: ScenarioScript(
        scenario=Scenario.REPLAY,
        factors=(
            _f(FactorName.VOICE_MATCH, 0.91, 0.35, FactorVerdict.PASS,
               "Genuine voice, consistent with a recording of the real speaker."),
            _f(FactorName.SYNTHETIC_SPEECH, 0.76, 0.25, FactorVerdict.PASS,
               "Not synthesised. The source audio is real."),
            _f(FactorName.REPLAY, 0.04, 0.15, FactorVerdict.FAIL,
               "Double-compression and loudspeaker channel signature detected."),
            _f(FactorName.AGENT_BEHAVIOUR, 0.62, 0.15, FactorVerdict.INCONCLUSIVE,
               "Responses do not track the prompts."),
            _f(FactorName.LIVENESS, 0.06, 0.10, FactorVerdict.FAIL,
               "No interactive liveness. Audio is pre-recorded."),
        ),
        confidence=0.93,
        reason="Replayed recording detected. Refused without step-up.",
        otp_can_rescue=False,
    ),
    Scenario.AI_AGENT: ScenarioScript(
        scenario=Scenario.AI_AGENT,
        factors=(
            _f(FactorName.VOICE_MATCH, 0.35, 0.35, FactorVerdict.FAIL,
               "Embedding does not match the enrolled reference."),
            _f(FactorName.SYNTHETIC_SPEECH, 0.10, 0.25, FactorVerdict.FAIL,
               "Text-to-speech signature across all segments."),
            _f(FactorName.REPLAY, 0.88, 0.15, FactorVerdict.PASS,
               "Live stream, not a replay."),
            _f(FactorName.AGENT_BEHAVIOUR, 0.06, 0.15, FactorVerdict.FAIL,
               "Sub-200ms turn latency and no disfluencies. Automated caller."),
            _f(FactorName.LIVENESS, 0.22, 0.10, FactorVerdict.FAIL,
               "No breath or background movement."),
        ),
        confidence=0.97,
        reason="Automated caller detected. Refused without step-up.",
        otp_can_rescue=False,
    ),
    Scenario.HUMAN_IMPOSTOR: ScenarioScript(
        scenario=Scenario.HUMAN_IMPOSTOR,
        factors=(
            _f(FactorName.VOICE_MATCH, 0.18, 0.35, FactorVerdict.FAIL,
               "Live human, but the embedding does not match the enrolled reference."),
            _f(FactorName.SYNTHETIC_SPEECH, 0.95, 0.25, FactorVerdict.PASS,
               "Natural speech, no synthesis artefacts."),
            _f(FactorName.REPLAY, 0.93, 0.15, FactorVerdict.PASS,
               "Live handset channel."),
            _f(FactorName.AGENT_BEHAVIOUR, 0.90, 0.15, FactorVerdict.PASS,
               "Human turn-taking and disfluencies."),
            _f(FactorName.LIVENESS, 0.94, 0.10, FactorVerdict.PASS,
               "Breath and background movement present."),
        ),
        confidence=0.88,
        reason="A live human whose voice does not match the claim. Step-up required.",
        otp_can_rescue=True,
    ),
}


def resolve_scenario(
    header_value: str | None, claimed_user_id: str | None
) -> tuple[Scenario, str]:
    """Pick a scenario. The header wins, then claimed_user_id, then the default.

    Returns the scenario and a short note on how it was chosen, which lands in
    the response so a fixture failure is easy to diagnose.

    Raises ValueError if the header names something that is not a scenario.
    """

    if header_value:
        candidate = header_value.strip().lower()
        try:
            return Scenario(candidate), "header"
        except ValueError as exc:
            raise ValueError(
                f"unknown scenario {header_value!r}; expected one of "
                + ", ".join(s.value for s in Scenario)
            ) from exc

    if claimed_user_id:
        needle = claimed_user_id.strip().lower()
        for scenario in Scenario:
            if scenario.value == needle:
                return scenario, "claimed_user_id exact"
        for scenario in Scenario:
            if scenario.value in needle:
                return scenario, "claimed_user_id substring"

    return Scenario.REAL_CFO, "default"


def jitter_seconds(min_ms: int, max_ms: int, rng: random.Random | None = None) -> float:
    """Simulated per-request latency, in seconds."""

    if max_ms <= 0:
        return 0.0
    source = rng or random
    return source.uniform(min_ms, max_ms) / 1000.0
