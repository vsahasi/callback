"""Classical audio analysis for the Callback detector.

A real, measurement-driven replacement for the scripted verdicts in
`detector.mock`, built entirely from signal processing so that it needs no
training corpus and no model checkpoint.

Wiring it in
------------

    from detector.analysis import AnalysisEngine, CallMetadata

    engine = AnalysisEngine()                      # stateless, thread-safe
    factors = engine.analyse(pcm, 8000, metadata=CallMetadata(...))

`factors` is a `list[shared.contract.Factor]` in the same shape
`detector.mock` produces, so it feeds `detector.scoring.blend_risk`,
`fatal_factor` and `decide` unchanged. Use `engine.analyse_report(...)` when
the raw measurements are wanted too, for logging or for the calibration
harness.

Dependency note: this package needs `numpy`, which is currently declared in
the `train` extra rather than the base dependency set. Whoever wires the
engine into `detector.app` has to make numpy available in the detector image.
Nothing here imports torch, scipy or librosa.

Calibration
-----------

    uv run python -m detector.analysis.calibration

prints every detector's score across the synthetic conditions in
`detector.analysis.synthetic`, which is how the thresholds in each module
were chosen and how they should be re-chosen when real audio arrives.
"""

from __future__ import annotations

from detector.analysis.behaviour import CallMetadata, analyse_behaviour
from detector.analysis.cues import Cue, DetectorResult
from detector.analysis.dsp import AudioContext, build_context
from detector.analysis.engine import (
    DEFAULT_WEIGHTS,
    AnalysisEngine,
    AnalysisReport,
    analyse_call,
)
from detector.analysis.liveness import analyse_liveness
from detector.analysis.replay import analyse_replay
from detector.analysis.synthesis import analyse_synthesis
from detector.analysis.voice import analyse_voice_match
from detector.analysis.voiceprint import VoicePrint, voiceprint_from

__all__ = [
    "DEFAULT_WEIGHTS",
    "AnalysisEngine",
    "AnalysisReport",
    "AudioContext",
    "CallMetadata",
    "Cue",
    "DetectorResult",
    "VoicePrint",
    "analyse_behaviour",
    "analyse_call",
    "analyse_liveness",
    "analyse_replay",
    "analyse_synthesis",
    "analyse_voice_match",
    "build_context",
    "voiceprint_from",
]
