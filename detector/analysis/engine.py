"""The entry point: audio in, `list[Factor]` out.

    from detector.analysis import AnalysisEngine, CallMetadata

    engine = AnalysisEngine()
    factors = engine.analyse(pcm, 8000, metadata=CallMetadata(...))

`factors` is exactly the shape `detector.mock` produces, so it goes straight
into `detector.scoring.blend_risk`, `fatal_factor` and `decide` with no
adaptation. The engine never emits an `otp` factor: OTP is a state-machine
step owned by the service, not a measurement.

WEIGHTS
-------
`DEFAULT_WEIGHTS` matches the scripted scenarios in `detector.mock` exactly,
so swapping the mock for this engine does not also silently change how
factors are combined. They sum to 1.0. `detector.scoring.blend_risk`
renormalises over the factors that actually ran, so a `not_run` factor costs
nothing and redistributes its weight rather than counting as innocent.

WHAT IS AND IS NOT INDEPENDENT
------------------------------
The weighted blend assumes the factors are independent evidence. Two of them
are not. `synthetic_speech` and `liveness` both read pitch micro-variation,
so a synthesiser is charged for the same measurement twice and lands lower
than the arithmetic suggests. The weights were chosen with that in mind
(liveness is the lightest factor at 0.10) but it is not corrected for, and it
is the first thing to revisit once real attack audio exists to calibrate
against.

WHAT RENORMALISATION DOES TO A SINGLE FAILING SIGNAL
----------------------------------------------------
`detector.scoring.blend_risk` renormalises over the factors that ran, and
`decide` allows below a blended risk of 0.30. So the fewer detectors run, the
more each one is worth, and a failing signal can be either amplified or
diluted depending on what else ran. Both directions are live:

With no reference voice and no turn-taking metadata only three factors run,
`replay` carries 0.30 of the renormalised weight, and a replay score above
about 0.10 still cannot push the blend past the allow threshold on its own.
Worse, `voice_match` not running also removed the *failing* case: a different
talker reading the attacker's script passes synthesis, replay and liveness,
so with nothing enrolled that call blends to 0.05 and is allowed.

With a reference voice all four audio factors run, `voice_match` carries 0.35,
and a clone that matches it dilutes the signals that caught the synthesis. A
synthesiser that denied at 0.59 on three factors challenges at 0.35 on four.
That one is intended: see `detector.mock`'s `cloned_voice` scenario, which has
always scored `voice_match` as a pass and called for a step-up.

What keeps all of this safe is not the blend. It is that
`detector.scoring.first_failing_factor` refuses to let any `fail` verdict
become an allow, and that `detector.evidence.UNRESCUABLE` withholds the
step-up for `replay` and `agent_behaviour`, which drops the middle band to
deny. The remaining cheap mitigation, outside this package: have the agent
always send `turn_latencies_ms` so `agent_behaviour` runs too and the weights
stop being renormalised at all.

FAILURE POLICY
--------------
Every detector runs inside a guard. A detector that raises becomes a
`not_run` factor carrying the exception type, and the rest of the analysis
continues. A verification service that returns a 500 because one cue hit a
degenerate frame is worse than one that returns four factors and an honest
gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from detector.analysis.behaviour import CallMetadata, analyse_behaviour
from detector.analysis.cues import DetectorResult, not_run
from detector.analysis.dsp import DEFAULT_GRID, FrameGrid, build_context
from detector.analysis.liveness import analyse_liveness
from detector.analysis.replay import analyse_replay
from detector.analysis.synthesis import analyse_synthesis
from detector.analysis.voice import analyse_voice_match
from detector.analysis.voiceprint import VoicePrint, voiceprint_from
from shared.contract import Factor, FactorName, FactorVerdict

__all__ = [
    "DEFAULT_WEIGHTS",
    "AnalysisEngine",
    "AnalysisReport",
    "CallMetadata",
    "analyse_call",
]

# Identical to the per-scenario weights in detector.mock, deliberately.
DEFAULT_WEIGHTS: dict[FactorName, float] = {
    FactorName.VOICE_MATCH: 0.35,
    FactorName.SYNTHETIC_SPEECH: 0.25,
    FactorName.REPLAY: 0.15,
    FactorName.AGENT_BEHAVIOUR: 0.15,
    FactorName.LIVENESS: 0.10,
}

# The order factors are emitted in. Stable, so a diff of two analyses lines up.
FACTOR_ORDER: tuple[FactorName, ...] = (
    FactorName.VOICE_MATCH,
    FactorName.SYNTHETIC_SPEECH,
    FactorName.REPLAY,
    FactorName.AGENT_BEHAVIOUR,
    FactorName.LIVENESS,
)


@dataclass(frozen=True)
class AnalysisReport:
    """Factors plus the raw numbers behind them.

    `measurements` is keyed by factor name, then by measurement name. It is
    not part of the wire contract; it exists so the calibration harness and
    an observability sink can see what the detectors actually measured
    instead of only the score they produced.
    """

    factors: list[Factor]
    measurements: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    duration_s: float = 0.0
    sample_rate: int = 0
    elapsed_ms: float = 0.0
    #: The analysed span reduced to a `VoicePrint`: pitch register and
    #: long-term spectrum, nothing that carries what was said. This is what
    #: the service keeps of a span once the samples are gone, so that the
    #: two legs of a callback can be compared without storing audio.
    voice: VoicePrint | None = None

    def score(self, name: FactorName) -> float:
        for factor in self.factors:
            if factor.name is name:
                return factor.score
        raise KeyError(f"no factor named {name}")

    def verdict(self, name: FactorName) -> FactorVerdict:
        for factor in self.factors:
            if factor.name is name:
                return factor.verdict
        raise KeyError(f"no factor named {name}")


class AnalysisEngine:
    """Classical signal-processing detector. Stateless and deterministic.

    Stateless: no model, no warm-up, nothing cached between calls, so one
    instance is safe to share across requests and across threads.

    Deterministic: the same samples and sample rate always give the same
    factors. There is no randomness anywhere in the analysis path, which is
    what makes a disputed decision reproducible from the stored audio.
    """

    def __init__(
        self,
        weights: dict[FactorName, float] | None = None,
        grid: FrameGrid = DEFAULT_GRID,
    ) -> None:
        self._weights = dict(weights or DEFAULT_WEIGHTS)
        missing = [name for name in FACTOR_ORDER if name not in self._weights]
        if missing:
            raise ValueError(f"weights missing for {[name.value for name in missing]}")
        total = sum(self._weights[name] for name in FACTOR_ORDER)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"factor weights must sum to 1.0, got {total:.6f}")
        self._grid = grid

    @property
    def weights(self) -> dict[FactorName, float]:
        return dict(self._weights)

    def analyse(
        self,
        audio: object,
        sample_rate: int,
        *,
        metadata: CallMetadata | None = None,
        reference: VoicePrint | None = None,
        reference_audio: object | None = None,
        reference_sample_rate: int | None = None,
    ) -> list[Factor]:
        """Analyse one span of mono call audio.

        `audio` may be a numpy array, a torch tensor, or any nested sequence
        of numbers; multi-channel input is downmixed. `sample_rate` is in Hz.

        `metadata` carries the agent's turn-taking timings. Without it the
        `agent_behaviour` factor is `not_run` rather than guessed.

        `reference` is the enrolled voice print of the claimed identity, which
        is what the service holds; `reference_audio` is an enrolment recording
        to extract one from on the spot, which is what the calibration harness
        and the tests hand over. `reference` wins if both arrive. Without
        either, the `voice_match` factor is `not_run`.

        Returns the five audio and behaviour factors in `FACTOR_ORDER`, ready
        for `detector.scoring`. Never raises on degenerate audio.
        """

        return self.analyse_report(
            audio,
            sample_rate,
            metadata=metadata,
            reference=reference,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
        ).factors

    def analyse_report(
        self,
        audio: object,
        sample_rate: int,
        *,
        metadata: CallMetadata | None = None,
        reference: VoicePrint | None = None,
        reference_audio: object | None = None,
        reference_sample_rate: int | None = None,
    ) -> AnalysisReport:
        """`analyse`, plus the raw measurements behind each factor."""

        started = time.perf_counter()
        ctx = build_context(audio, sample_rate, self._grid)

        reference_print = reference
        if reference_print is None and reference_audio is not None:
            # Extracting here costs a full context build per call, which is why
            # the service stores the print instead and passes `reference`.
            reference_print = voiceprint_from(
                build_context(
                    reference_audio, reference_sample_rate or sample_rate, self._grid
                )
            )

        results: dict[FactorName, DetectorResult] = {
            FactorName.VOICE_MATCH: _guard(
                "voice_match", lambda: analyse_voice_match(ctx, reference_print)
            ),
            FactorName.SYNTHETIC_SPEECH: _guard(
                "synthetic_speech", lambda: analyse_synthesis(ctx)
            ),
            FactorName.REPLAY: _guard("replay", lambda: analyse_replay(ctx)),
            FactorName.AGENT_BEHAVIOUR: _guard(
                "agent_behaviour", lambda: analyse_behaviour(metadata)
            ),
            FactorName.LIVENESS: _guard("liveness", lambda: analyse_liveness(ctx)),
        }

        factors: list[Factor] = []
        measurements: dict[str, dict[str, float]] = {}
        for name in FACTOR_ORDER:
            result = results[name]
            detail = result.detail
            if ctx.notes and result.verdict is not FactorVerdict.NOT_RUN:
                detail = _append_notes(detail, ctx.notes)
            factors.append(
                Factor(
                    name=name,
                    score=max(0.0, min(1.0, result.score)),
                    weight=self._weights[name],
                    verdict=result.verdict,
                    detail=detail[:500],
                )
            )
            measurements[name.value] = dict(result.measurements)

        return AnalysisReport(
            factors=factors,
            measurements=measurements,
            notes=ctx.notes,
            duration_s=ctx.block.duration_s,
            sample_rate=ctx.block.sample_rate,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            # Never raises; degenerate audio yields a print that says so.
            voice=voiceprint_from(ctx),
        )


def _append_notes(detail: str, notes: tuple[str, ...]) -> str:
    """Attach the audio-conditioning caveats to a factor's detail.

    A reviewer looking at one factor should not have to go and find out that
    the recording was 300 ms long or clipped on 40% of its samples.
    """

    joined = "; ".join(notes)
    return f"{detail} Audio: {joined}."


def _guard(label: str, call: object) -> DetectorResult:
    """Run one detector, converting any failure into a `not_run` factor."""

    try:
        return call()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - fail open to not_run, never 500
        return not_run(
            f"The {label} detector failed ({type(exc).__name__}: {exc}), so this factor "
            "carries no weight in the decision."
        )


def analyse_call(
    audio: object,
    sample_rate: int,
    *,
    metadata: CallMetadata | None = None,
    reference: VoicePrint | None = None,
    reference_audio: object | None = None,
    reference_sample_rate: int | None = None,
) -> list[Factor]:
    """One-shot convenience wrapper around a default `AnalysisEngine`."""

    return AnalysisEngine().analyse(
        audio,
        sample_rate,
        metadata=metadata,
        reference=reference,
        reference_audio=reference_audio,
        reference_sample_rate=reference_sample_rate,
    )
