"""The five mock scripts for /v1 (ruling section 2.11), as frozen data.

They reproduce Notion's mock table. They run through the policy in
detector/v1/policy.py with no special cases: every outcome in the demo comes
from the same rules a real call would meet. They are separate from the
/events scripts in detector/mock.py, which stay as they are.

`cloned_voice` is the one script that moves: its spoof score rises from 0.30
to 0.92 once the session is 6.0 s old on the registry clock, so a fast first
authorize steps up and the clone is caught a few seconds in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from callback_shared.contract import ChallengeKind, ChallengeState, Scenario

# When the cloned voice's spoof score rises, in seconds of session age.
SPOOF_RISE_AT_S = 6.0
# The deepfake signal's window count in mock mode.
DEEPFAKE_WINDOWS = 8


@dataclass(frozen=True)
class ChallengeOutcome:
    """What a challenge resolves to, once its answer is in."""

    status: ChallengeState
    phrase_match: bool | None
    speaker_score: float | None
    reason: str | None


@dataclass(frozen=True)
class Script:
    scenario: Scenario
    voiceprint: float  # the passive voiceprint score
    spoof: float
    agent: float
    text: float
    text_flags: tuple[str, ...]
    challenge: ChallengeOutcome  # digits, phrase and liveness
    enroll: ChallengeOutcome
    # cloned_voice only: the spoof score from SPOOF_RISE_AT_S on.
    spoof_after_rise: float | None = None


def _passed(speaker: float) -> ChallengeOutcome:
    return ChallengeOutcome(ChallengeState.PASSED, True, speaker, None)


# Enrollment has no reference voice to score against, so no speaker score.
_ENROLL_PASSED = ChallengeOutcome(ChallengeState.PASSED, True, None, None)

SCRIPTS: Mapping[Scenario, Script] = MappingProxyType(
    {
        Scenario.REAL_CFO: Script(
            scenario=Scenario.REAL_CFO,
            voiceprint=0.83,
            spoof=0.04,
            agent=0.05,
            text=0.10,
            text_flags=(),
            challenge=_passed(0.83),
            enroll=_ENROLL_PASSED,
        ),
        Scenario.CLONED_VOICE: Script(
            scenario=Scenario.CLONED_VOICE,
            voiceprint=0.78,
            spoof=0.30,
            spoof_after_rise=0.92,
            agent=0.10,
            text=0.20,
            text_flags=(),
            challenge=_passed(0.78),
            enroll=_ENROLL_PASSED,
        ),
        Scenario.REPLAY: Script(
            scenario=Scenario.REPLAY,
            voiceprint=0.80,
            spoof=0.45,
            agent=0.10,
            text=0.20,
            text_flags=(),
            challenge=ChallengeOutcome(
                ChallengeState.FAILED, False, 0.80, "phrase_mismatch"
            ),
            enroll=_ENROLL_PASSED,
        ),
        Scenario.AI_AGENT: Script(
            scenario=Scenario.AI_AGENT,
            voiceprint=0.70,
            spoof=0.85,
            agent=0.90,
            text=0.40,
            text_flags=(),
            challenge=_passed(0.70),
            enroll=_ENROLL_PASSED,
        ),
        Scenario.HUMAN_IMPOSTOR: Script(
            scenario=Scenario.HUMAN_IMPOSTOR,
            voiceprint=0.29,
            spoof=0.05,
            agent=0.05,
            text=0.80,
            text_flags=("urgency", "verification_refusal"),
            challenge=ChallengeOutcome(
                ChallengeState.FAILED, True, 0.29, "speaker_mismatch"
            ),
            enroll=_ENROLL_PASSED,
        ),
    }
)

# Real mode, until the listener exists: no audio, so every challenge fails
# with no_audio, which never counts as a voice fail.
NO_AUDIO_OUTCOME = ChallengeOutcome(ChallengeState.FAILED, None, None, "no_audio")


def spoof_at(scenario: Scenario, session_age_s: float) -> float:
    script = SCRIPTS[scenario]
    if script.spoof_after_rise is not None and session_age_s >= SPOOF_RISE_AT_S:
        return script.spoof_after_rise
    return script.spoof


def challenge_outcome(scenario: Scenario, kind: ChallengeKind) -> ChallengeOutcome:
    script = SCRIPTS[scenario]
    if kind is ChallengeKind.ENROLL:
        return script.enroll
    return script.challenge
