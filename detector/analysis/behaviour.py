"""Automated-caller detection from call timing, not from audio.

Whether the thing on the line is a bot is a question about conversation, not
about waveforms. A text-to-speech pipeline answers a prompt in 80-250 ms with
almost no variance, because the latency is a fixed sum of ASR, inference and
synthesis. A human takes 300 ms to 1.5 s and varies enormously: they think,
they interrupt, they say "sorry, what?". A replayed recording is worse than
either, because it answers on its own schedule and its answers stop tracking
the prompts.

None of that is in the audio, so this module takes metadata and returns
`not_run` without it. That is deliberate. Inventing a behavioural score from
a waveform would put a confident-looking number in front of a reviewer that
nothing measured, and `agent_behaviour` is one of the two fatal factors in
`detector.scoring`: at or below 0.10 it denies a payment outright. A fabricated
number there could refuse a genuine vendor.

The agent supplies the timings through `EventRequest.metadata`; see
`CallMetadata.from_mapping` for the keys it reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from detector.analysis.cues import (
    Cue,
    DetectorResult,
    blend,
    not_run,
    ramp,
    summarise,
    temper,
    verdict_for,
)
from detector.analysis.dsp import EPS

__all__ = ["CallMetadata", "analyse_behaviour"]

MIN_TURNS_FOR_LATENCY = 3
MIN_TURNS_FOR_FATAL = 4
SINGLE_CUE_FLOOR = 0.12
CORROBORATED_FLOOR = 0.02

# Independent cues compound; see cues.blend.
SHARPNESS = 2.6

# Cue weights. prompt_tracking carries the most because it is the only cue
# here that catches a replayed recording: a recording's timing can look
# perfectly human while its answers have nothing to do with the questions.
WEIGHT_LATENCY = 0.35
WEIGHT_CONSISTENCY = 0.25
WEIGHT_TRACKING = 0.40


@dataclass(frozen=True)
class CallMetadata:
    """Conversation timing observed by the agent, in milliseconds.

    `turn_latencies_ms` is the gap between the end of each agent prompt and
    the start of the caller's reply. `prompt_relevance` is an optional 0-1
    score per turn from whatever the agent uses to decide the reply answered
    the question; it is the only cue that can catch a replayed recording,
    because a recording's timing can look human while its content does not
    track what was asked.
    """

    turn_latencies_ms: Sequence[float] = ()
    response_durations_ms: Sequence[float] = ()
    prompt_relevance: Sequence[float] = ()
    barge_ins: int | None = None

    @property
    def has_timing(self) -> bool:
        return len(self.turn_latencies_ms) >= MIN_TURNS_FOR_LATENCY

    @property
    def has_relevance(self) -> bool:
        return len(self.prompt_relevance) >= 2

    @property
    def is_empty(self) -> bool:
        return not (self.turn_latencies_ms or self.prompt_relevance)

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, object] | None) -> CallMetadata:
        """Parse the agent's free-form `EventRequest.metadata` dict.

        Reads `turn_latencies_ms`, `response_durations_ms`,
        `prompt_relevance` and `barge_ins`. Anything missing, mistyped or
        non-finite is dropped rather than raising: a malformed metadata blob
        must degrade this factor to `not_run`, not fail the request.
        """

        if not metadata:
            return cls()
        barge = metadata.get("barge_ins")
        return cls(
            turn_latencies_ms=_numbers(metadata.get("turn_latencies_ms")),
            response_durations_ms=_numbers(metadata.get("response_durations_ms")),
            prompt_relevance=_numbers(metadata.get("prompt_relevance")),
            barge_ins=int(barge) if isinstance(barge, (int, float)) else None,
        )


def _numbers(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        value = float(item)
        if np.isfinite(value):
            out.append(value)
    return tuple(out)


def analyse_behaviour(metadata: CallMetadata | None) -> DetectorResult:
    """Score how consistent the conversation timing is with a human (1.0) or
    an automated caller (0.0). `not_run` without usable metadata."""

    if metadata is None or metadata.is_empty:
        return not_run(
            "No turn-taking metadata supplied, so no behavioural measurement was made. "
            "Pass turn_latencies_ms (and optionally prompt_relevance) in the event "
            "metadata to enable this factor."
        )
    if not metadata.has_timing and not metadata.has_relevance:
        return not_run(
            f"Only {len(metadata.turn_latencies_ms)} turn latencies supplied; at least "
            f"{MIN_TURNS_FOR_LATENCY} are needed before timing means anything."
        )

    cues: list[Cue] = []
    measurements: dict[str, float] = {}

    latency_cues, latency_measures = _latency_cues(metadata)
    cues.extend(latency_cues)
    measurements.update(latency_measures)

    relevance_cue, relevance_measures = _relevance_cue(metadata)
    cues.append(relevance_cue)
    measurements.update(relevance_measures)

    turns = len(metadata.turn_latencies_ms)
    latency_penalties = [
        cue.penalty
        for cue in cues
        if cue.measurable and cue.name in ("turn_latency", "latency_consistency")
    ]
    corroborated = (
        turns >= MIN_TURNS_FOR_FATAL
        and len(latency_penalties) == 2
        and all(penalty >= 0.7 for penalty in latency_penalties)
    )
    floor = CORROBORATED_FLOOR if corroborated else SINGLE_CUE_FLOOR
    raw_score = blend(cues, floor=floor, sharpness=SHARPNESS)

    measurable = sum(1 for cue in cues if cue.measurable)
    confident = measurable >= 2 and turns >= MIN_TURNS_FOR_LATENCY
    score = temper(raw_score, confident=confident)
    measurements["turns"] = float(turns)
    measurements["cues_measurable"] = float(measurable)
    measurements["score_floor"] = floor
    measurements["score_before_tempering"] = raw_score

    detail = summarise(f"Turn-taking over {turns} turns:", cues)
    if not confident:
        detail = (
            f"{detail} Low confidence: {turns} turns and {measurable} measurable cues."
        )

    return DetectorResult(
        score=score,
        verdict=verdict_for(score, confident=confident),
        detail=detail[:500],
        measurements=measurements,
        cues=tuple(cues),
    )


def _latency_cues(metadata: CallMetadata) -> tuple[list[Cue], dict[str, float]]:
    latencies = np.asarray(metadata.turn_latencies_ms, dtype=np.float64)
    latencies = latencies[np.isfinite(latencies) & (latencies >= 0.0)]
    if latencies.size < MIN_TURNS_FOR_LATENCY:
        note = f"{latencies.size} usable turn latencies, fewer than {MIN_TURNS_FOR_LATENCY}"
        return (
            [
                _unmeasurable("turn_latency", WEIGHT_LATENCY, note),
                _unmeasurable("latency_consistency", WEIGHT_CONSISTENCY, note),
            ],
            {"turn_latency_count": float(latencies.size)},
        )

    median = float(np.median(latencies))
    mean = float(np.mean(latencies))
    spread = float(np.std(latencies) / max(mean, EPS))

    median_penalty = ramp(median, 320.0, 140.0)
    if median_penalty >= 0.25:
        median_note = f"median reply latency {median:.0f} ms, below human reaction time"
    elif median > 3000.0:
        median_note = (
            f"median reply latency {median:.0f} ms, unusually slow but not machine-like"
        )
    else:
        median_note = f"median reply latency {median:.0f} ms, human"

    spread_penalty = ramp(spread, 0.28, 0.06)
    if spread_penalty >= 0.25:
        spread_note = (
            f"reply latency varies by only {spread * 100:.0f}% across turns, a fixed "
            "processing pipeline rather than a person"
        )
    else:
        spread_note = f"reply latency varies by {spread * 100:.0f}% across turns"

    cues = [
        Cue(
            name="turn_latency",
            value=median,
            penalty=median_penalty,
            weight=WEIGHT_LATENCY,
            note=median_note,
        ),
        Cue(
            name="latency_consistency",
            value=spread,
            penalty=spread_penalty,
            weight=WEIGHT_CONSISTENCY,
            note=spread_note,
        ),
    ]
    return cues, {
        "median_latency_ms": median,
        "latency_cv": spread,
        "turn_latency_count": float(latencies.size),
    }


def _relevance_cue(metadata: CallMetadata) -> tuple[Cue, dict[str, float]]:
    values = np.asarray(metadata.prompt_relevance, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        note = (
            "no prompt-relevance scores supplied, so content tracking was not checked"
        )
        return _unmeasurable("prompt_tracking", WEIGHT_TRACKING, note), {}

    mean = float(np.clip(np.mean(values), 0.0, 1.0))
    penalty = ramp(mean, 0.65, 0.20)
    if penalty >= 0.25:
        note = (
            f"replies scored {mean:.2f} for answering the prompt: the caller is not "
            "tracking the questions"
        )
    else:
        note = f"replies scored {mean:.2f} for answering the prompt"

    return (
        Cue(
            name="prompt_tracking",
            value=mean,
            penalty=penalty,
            weight=WEIGHT_TRACKING,
            note=note,
            # Guilt-only: replies that answer the prompts are what a competent
            # voice bot produces too, so a clean score here must not dilute
            # the latency cues.
            one_sided=True,
        ),
        {"prompt_relevance_mean": mean, "prompt_relevance_count": float(values.size)},
    )


def _unmeasurable(name: str, weight: float, note: str) -> Cue:
    return Cue(
        name=name, value=0.0, penalty=0.0, weight=weight, note=note, measurable=False
    )
