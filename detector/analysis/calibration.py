"""Calibration harness: every detector's score across every synthetic condition.

    uv run python -m detector.analysis.calibration
    uv run python -m detector.analysis.calibration --measurements

This is the table the thresholds in `replay`, `synthesis`, `liveness` and
`behaviour` were chosen from, and it is the table to re-read when real audio
finally arrives. It prints scores, not pass/fail, precisely so that a shift in
a measurement is visible before it becomes a flipped verdict.

Read it with the caveat from `detector.analysis.synthetic` in mind: these are
parametric signals with the artefacts built in by construction. The margins
below say "the detector responds to the physics it claims to measure". They
say nothing about a false-accept rate on real calls.

`CONDITIONS` is also what `tests/detector` asserts against, so the harness and
the tests cannot drift apart.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from detector.analysis import synthetic
from detector.analysis.behaviour import CallMetadata
from detector.analysis.engine import AnalysisEngine, AnalysisReport
from detector.scoring import blend_risk, decide, fatal_factor
from shared.contract import Decision, FactorName

__all__ = ["CONDITIONS", "Condition", "Row", "build", "main", "run"]

AUDIO_FACTORS: tuple[FactorName, ...] = (
    FactorName.SYNTHETIC_SPEECH,
    FactorName.REPLAY,
    FactorName.LIVENESS,
)

# Turn-taking metadata, the only input the behaviour detector takes.
HUMAN_TURNS = CallMetadata(
    turn_latencies_ms=(640.0, 1180.0, 410.0, 2050.0, 830.0),
    prompt_relevance=(0.91, 0.88, 0.95, 0.83, 0.9),
)
BOT_TURNS = CallMetadata(
    turn_latencies_ms=(145.0, 152.0, 139.0, 148.0, 151.0),
    prompt_relevance=(0.92, 0.9, 0.93, 0.89, 0.91),
)
REPLAY_TURNS = CallMetadata(
    turn_latencies_ms=(910.0, 1420.0, 600.0, 1750.0),
    prompt_relevance=(0.22, 0.14, 0.31, 0.18),
)


@dataclass(frozen=True)
class Condition:
    """One synthetic recording plus what we expect of it, in words."""

    name: str
    make: Callable[[], np.ndarray]
    sample_rate: int = 8000
    metadata: CallMetadata | None = None
    reference: Callable[[], np.ndarray] | None = None
    reference_sample_rate: int | None = None
    expectation: str = ""

    def audio(self) -> np.ndarray:
        return self.make()


def _live(duration: float = 4.0, seed: int = 0, rate: int = 8000) -> np.ndarray:
    return synthetic.live_speech(duration, rate, seed)


def _tts(duration: float = 4.0, seed: int = 0, rate: int = 8000) -> np.ndarray:
    return synthetic.tts_speech(duration, rate, seed)


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        name="live_4s",
        make=lambda: _live(4.0, 0),
        expectation="baseline: every detector should score high",
    ),
    Condition(
        name="live_3s",
        make=lambda: _live(3.0, 1),
        expectation="a realistic demo segment; liveness may go inconclusive",
    ),
    Condition(
        name="live_1p5s",
        make=lambda: _live(1.5, 2),
        expectation="too short: expect inconclusive verdicts, not low scores",
    ),
    Condition(
        name="live_16k",
        make=lambda: _live(4.0, 0, 16000),
        sample_rate=16000,
        expectation="same signal at a wider rate: scores should barely move",
    ),
    Condition(
        name="replay_of_live",
        make=lambda: synthetic.replay_of(_live(4.0, 0), 8000, seed=5),
        expectation="replay should fall hard; synthesis should not",
    ),
    Condition(
        name="replay_of_live_pcm",
        make=lambda: synthetic.replay_of(
            _live(4.0, 0), 8000, seed=5, post_channel=False
        ),
        expectation="as above, plus a measurable coarse sample lattice",
    ),
    Condition(
        name="replay_dry_room",
        make=lambda: synthetic.replay_of(_live(4.0, 0), 8000, seed=5, rt60_s=0.18),
        expectation="a harder replay: little reverb, so only the comb and band cues fire",
    ),
    Condition(
        name="tts_4s",
        make=lambda: _tts(4.0, 0),
        expectation="synthesis and liveness should fall; replay should not",
    ),
    Condition(
        name="tts_3s",
        make=lambda: _tts(3.0, 1),
        expectation="as above on a demo-length segment",
    ),
    Condition(
        name="replay_of_tts",
        make=lambda: synthetic.replay_of(_tts(4.0, 0), 8000, seed=5),
        expectation="both attack signatures at once: all three should fall",
    ),
    Condition(
        name="silence",
        make=lambda: synthetic.silence(3.0),
        expectation="no opinion: inconclusive everywhere, no NaN",
    ),
    Condition(
        name="white_noise",
        make=lambda: synthetic.white_noise(3.0, seed=3),
        expectation="no opinion: there is no speech to analyse",
    ),
    Condition(
        name="pure_tone",
        make=lambda: synthetic.pure_tone(3.0),
        expectation="no speech/silence structure: inconclusive",
    ),
    Condition(
        name="impulse_train",
        make=lambda: synthetic.impulse_train(3.0),
        expectation="pathological input: must not raise or return NaN",
    ),
    Condition(
        name="clipped_live",
        make=lambda: synthetic.clipped(_live(4.0, 0), limit=0.02),
        expectation="hard clipping: scores may move, verdicts should stay honest",
    ),
    Condition(
        name="dc_offset_live",
        make=lambda: synthetic.dc_offset(_live(4.0, 0), offset=0.3),
        expectation="DC is removed in conditioning: scores close to live_4s",
    ),
    Condition(
        name="live_4khz",
        make=lambda: _live(4.0, 0, 4000),
        sample_rate=4000,
        expectation="untuned rate: expect inconclusive, not a confident verdict",
    ),
    Condition(
        name="live_48khz",
        make=lambda: _live(4.0, 0, 48000),
        sample_rate=48000,
        expectation="decimated to 16 kHz in conditioning, then as live_16k",
    ),
    Condition(
        name="live_120ms",
        make=lambda: _live(4.0, 0)[:960],
        expectation="below the usable floor: every factor inconclusive",
    ),
    Condition(
        name="live_human_turns",
        make=lambda: _live(4.0, 0),
        metadata=HUMAN_TURNS,
        expectation="behaviour should pass on varied, human-latency turns",
    ),
    Condition(
        name="live_bot_turns",
        make=lambda: _live(4.0, 0),
        metadata=BOT_TURNS,
        expectation="behaviour should fail on 150 ms metronomic turns",
    ),
    Condition(
        name="replay_replay_turns",
        make=lambda: synthetic.replay_of(_live(4.0, 0), 8000, seed=5),
        metadata=REPLAY_TURNS,
        expectation="replay audio plus replies that do not track the prompts",
    ),
    Condition(
        name="live_matched_reference",
        make=lambda: _live(4.0, 0),
        reference=lambda: _live(4.0, 7),
        expectation="voice_match should pass: same synthetic talker parameters",
    ),
    Condition(
        name="live_mismatched_reference",
        make=lambda: _live(4.0, 0),
        reference=lambda: synthetic.live_speech(
            4.0, 8000, 7, synthetic.SpeechStyle(f0_hz=215.0)
        ),
        expectation="voice_match should fail: a register an octave up",
    ),
)


@dataclass
class Row:
    """One condition's results, flattened for printing."""

    name: str
    expectation: str
    scores: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    measurements: dict[str, dict[str, float]] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    risk: float = 0.0
    decision: str = ""
    fatal: str = ""


def build(condition: Condition, engine: AnalysisEngine | None = None) -> AnalysisReport:
    """Run one condition through the engine."""

    engine = engine or AnalysisEngine()
    reference = condition.reference() if condition.reference else None
    return engine.analyse_report(
        condition.audio(),
        condition.sample_rate,
        metadata=condition.metadata,
        reference_audio=reference,
        reference_sample_rate=condition.reference_sample_rate or condition.sample_rate,
    )


def run(conditions: tuple[Condition, ...] = CONDITIONS) -> list[Row]:
    """Run every condition and collect the numbers."""

    engine = AnalysisEngine()
    rows: list[Row] = []
    for condition in conditions:
        report = build(condition, engine)
        row = Row(name=condition.name, expectation=condition.expectation)
        for factor in report.factors:
            row.scores[factor.name.value] = factor.score
            row.verdicts[factor.name.value] = factor.verdict.value
        row.measurements = report.measurements
        row.elapsed_ms = report.elapsed_ms

        # Run the real scoring code, so the harness shows what the service
        # would do rather than only what the detectors measured.
        tripped = fatal_factor(report.factors)
        row.fatal = tripped.name.value if tripped else ""
        row.risk = blend_risk(report.factors)
        row.decision = Decision.DENY.value if tripped else decide(row.risk).value
        rows.append(row)
    return rows


_COLUMNS: tuple[tuple[str, str], ...] = (
    (FactorName.VOICE_MATCH.value, "voice"),
    (FactorName.SYNTHETIC_SPEECH.value, "synth"),
    (FactorName.REPLAY.value, "replay"),
    (FactorName.AGENT_BEHAVIOUR.value, "behav"),
    (FactorName.LIVENESS.value, "live"),
)

_VERDICT_MARK = {"pass": "P", "fail": "F", "inconclusive": "?", "not_run": "-"}

# The measurements worth printing per factor. Everything else stays in the
# report for whoever is debugging one detector.
_HEADLINE_MEASURES: dict[str, tuple[str, ...]] = {
    FactorName.REPLAY.value: (
        "comb_peak_ratio",
        "gap_minus_speech_db",
        "tail_shape_correlation",
        "offset_decay_db_per_s",
        "high_band_db",
        "in_band_tilt_db",
        "effective_bits",
    ),
    FactorName.SYNTHETIC_SPEECH.value: (
        "f0_perturbation_pct",
        "amplitude_perturbation_db",
        "phase_dispersion_rad",
        "spectral_variance_db",
        "rhythm_periodicity",
        "noise_floor_rel_db",
    ),
    FactorName.LIVENESS.value: (
        "breath_rate_per_s",
        "rhythm_cv",
        "f0_spread_semitones",
        "background_drift_db",
    ),
    FactorName.AGENT_BEHAVIOUR.value: (
        "median_latency_ms",
        "latency_cv",
        "prompt_relevance_mean",
    ),
    FactorName.VOICE_MATCH.value: ("f0_distance_semitones", "ltas_correlation"),
}


def format_table(rows: list[Row]) -> str:
    """Scores with a verdict mark, plus the decision they produce."""

    header = (
        f"{'condition':<26}"
        + "".join(f"{label:>12}" for _, label in _COLUMNS)
        + f"{'risk':>8}{'decision':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _ in _COLUMNS:
            score = row.scores.get(key)
            mark = _VERDICT_MARK.get(row.verdicts.get(key, ""), "?")
            cells.append("          --" if score is None else f"{score:>10.2f} {mark}")
        suffix = f"{row.risk:>8.2f}{row.decision:>11}"
        lines.append(f"{row.name:<26}" + "".join(cells) + suffix)
    lines.append("")
    lines.append("P pass   F fail   ? inconclusive   - not_run")
    lines.append(
        "Score is consistency with the claimed identity: 1.0 clean, 0.0 damning."
    )
    lines.append(
        "risk and decision come from detector.scoring, so this is what the service "
        "would actually do."
    )
    return "\n".join(lines)


def format_measurements(rows: list[Row]) -> str:
    """The raw numbers behind each score, for re-tuning a single ramp."""

    lines: list[str] = []
    for row in rows:
        lines.append(f"\n{row.name}  ({row.expectation})")
        lines.append(f"  analysed in {row.elapsed_ms:.1f} ms")
        for factor, keys in _HEADLINE_MEASURES.items():
            values = row.measurements.get(factor, {})
            shown = [
                f"{key}={values[key]:.3g}"
                for key in keys
                if key in values and np.isfinite(values[key])
            ]
            if shown:
                lines.append(f"  {factor:<18} " + "  ".join(shown))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m detector.analysis.calibration",
        description=(
            "Print each classical detector's score across the synthetic calibration "
            "conditions."
        ),
    )
    parser.add_argument(
        "--measurements",
        action="store_true",
        help="also print the raw measurements behind each score",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="run only conditions whose name contains this substring",
    )
    args = parser.parse_args(argv)

    conditions = CONDITIONS
    if args.only:
        conditions = tuple(c for c in CONDITIONS if args.only in c.name)
        if not conditions:
            print(f"no condition matches {args.only!r}", file=sys.stderr)
            return 2

    rows = run(conditions)
    print(format_table(rows))
    if args.measurements:
        print(format_measurements(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
