"""The entry point: contract conformance, determinism and robustness.

The engine is what `detector.app` will call, so these tests are about the
promises made to that caller rather than about any one measurement: emit
exactly the contract's factors, never raise, never return a NaN, and give the
same answer twice for the same bytes.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis import engine as engine_module  # noqa: E402
from detector.analysis.engine import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FACTOR_ORDER,
    AnalysisEngine,
    CallMetadata,
    analyse_call,
)
from detector.analysis.synthetic import (  # noqa: E402
    clipped,
    dc_offset,
    impulse_train,
    live_speech,
    pure_tone,
    replay_of,
    silence,
    tts_speech,
    white_noise,
)
from detector.mock import SCRIPTS  # noqa: E402
from detector.scoring import blend_risk, decide, fatal_factor  # noqa: E402
from shared.contract import Decision, Factor, FactorName, FactorVerdict, Scenario  # noqa: E402

RATE = 8000


@pytest.fixture(scope="module")
def engine() -> AnalysisEngine:
    return AnalysisEngine()


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------


def test_the_engine_emits_exactly_the_contract_factors(engine: AnalysisEngine) -> None:
    factors = engine.analyse(live_speech(4.0, RATE, seed=0), RATE)

    assert [factor.name for factor in factors] == list(FACTOR_ORDER)
    assert all(isinstance(factor, Factor) for factor in factors)
    # OTP is a state-machine step owned by the service, not a measurement.
    assert FactorName.OTP not in {factor.name for factor in factors}


def test_weights_match_the_scripted_scenarios(engine: AnalysisEngine) -> None:
    # Swapping the mock for this engine must not silently change how factors
    # are combined, so the weights are the mock's weights.
    scripted = {
        factor.name: factor.weight for factor in SCRIPTS[Scenario.REAL_CFO].factors
    }
    assert engine.weights == scripted
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_factor_is_contract_valid(engine: AnalysisEngine) -> None:
    factors = engine.analyse(
        live_speech(4.0, RATE, seed=0),
        RATE,
        metadata=CallMetadata(turn_latencies_ms=(600.0, 1100.0, 450.0)),
        reference_audio=live_speech(4.0, RATE, seed=7),
    )
    for factor in factors:
        assert 0.0 <= factor.score <= 1.0
        assert 0.0 <= factor.weight <= 1.0
        assert isinstance(factor.verdict, FactorVerdict)
        assert factor.detail is not None
        assert len(factor.detail) <= 500
        # Round-trips through the wire contract.
        assert Factor.model_validate(factor.model_dump()) == factor


def test_rejects_weights_that_do_not_sum_to_one() -> None:
    broken = dict(DEFAULT_WEIGHTS)
    broken[FactorName.VOICE_MATCH] = 0.9
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        AnalysisEngine(weights=broken)

    missing = {FactorName.VOICE_MATCH: 1.0}
    with pytest.raises(ValueError, match="missing"):
        AnalysisEngine(weights=missing)


# ---------------------------------------------------------------------------
# What runs and what does not
# ---------------------------------------------------------------------------


def test_the_optional_factors_are_not_run_rather_than_guessed(
    engine: AnalysisEngine,
) -> None:
    report = engine.analyse_report(live_speech(4.0, RATE, seed=0), RATE)
    assert report.verdict(FactorName.VOICE_MATCH) is FactorVerdict.NOT_RUN
    assert report.verdict(FactorName.AGENT_BEHAVIOUR) is FactorVerdict.NOT_RUN
    # A not_run factor is excluded from the blend, so its score never reaches
    # a decision.
    assert blend_risk(report.factors) < 0.30


def test_supplying_metadata_and_a_reference_turns_them_on(
    engine: AnalysisEngine,
) -> None:
    report = engine.analyse_report(
        live_speech(4.0, RATE, seed=0),
        RATE,
        metadata=CallMetadata(turn_latencies_ms=(640.0, 1180.0, 410.0, 2050.0)),
        reference_audio=live_speech(4.0, RATE, seed=7),
    )
    assert report.verdict(FactorName.VOICE_MATCH) is not FactorVerdict.NOT_RUN
    assert report.verdict(FactorName.AGENT_BEHAVIOUR) is not FactorVerdict.NOT_RUN


def test_a_failing_detector_becomes_not_run_rather_than_an_exception(
    monkeypatch: pytest.MonkeyPatch, engine: AnalysisEngine
) -> None:
    def explode(_ctx: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(engine_module, "analyse_replay", explode)
    report = engine.analyse_report(live_speech(4.0, RATE, seed=0), RATE)

    assert report.verdict(FactorName.REPLAY) is FactorVerdict.NOT_RUN
    assert "RuntimeError" in next(
        factor.detail or ""
        for factor in report.factors
        if factor.name is FactorName.REPLAY
    )
    # The other detectors still produced answers.
    assert report.verdict(FactorName.SYNTHETIC_SPEECH) is not FactorVerdict.NOT_RUN


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_bytes_give_the_same_factors(engine: AnalysisEngine) -> None:
    signal = live_speech(4.0, RATE, seed=0)
    metadata = CallMetadata(turn_latencies_ms=(640.0, 1180.0, 410.0))
    first = engine.analyse(signal, RATE, metadata=metadata)
    second = engine.analyse(signal.copy(), RATE, metadata=metadata)
    assert [factor.model_dump() for factor in first] == [
        factor.model_dump() for factor in second
    ]


def test_two_engines_agree(engine: AnalysisEngine) -> None:
    signal = replay_of(live_speech(4.0, RATE, seed=0), RATE, seed=5)
    assert [f.model_dump() for f in AnalysisEngine().analyse(signal, RATE)] == [
        f.model_dump() for f in engine.analyse(signal, RATE)
    ]


def test_the_synthetic_corpus_is_reproducible() -> None:
    assert np.array_equal(
        live_speech(2.0, RATE, seed=4), live_speech(2.0, RATE, seed=4)
    )
    assert not np.array_equal(
        live_speech(2.0, RATE, seed=4), live_speech(2.0, RATE, seed=5)
    )


def test_input_type_does_not_change_the_answer(engine: AnalysisEngine) -> None:
    signal = live_speech(3.0, RATE, seed=0)
    from_array = engine.analyse(signal, RATE)
    from_list = engine.analyse(signal.tolist(), RATE)
    assert [f.score for f in from_array] == pytest.approx([f.score for f in from_list])

    torch = pytest.importorskip("torch")
    from_tensor = engine.analyse(torch.from_numpy(signal), RATE)
    assert [f.score for f in from_array] == pytest.approx(
        [f.score for f in from_tensor]
    )


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

DEGENERATE = {
    "silence": (silence(3.0, RATE), RATE),
    "white_noise": (white_noise(3.0, RATE, seed=1), RATE),
    "pure_tone": (pure_tone(3.0, RATE), RATE),
    "impulse_train": (impulse_train(3.0, RATE), RATE),
    "dc_offset": (dc_offset(live_speech(3.0, RATE, seed=0), 0.5), RATE),
    "clipped": (clipped(live_speech(3.0, RATE, seed=0), 0.01), RATE),
    "one_sample": (np.array([0.5]), RATE),
    "empty": (np.zeros(0), RATE),
    "120ms": (live_speech(3.0, RATE, seed=0)[:960], RATE),
    "untuned_4k": (live_speech(3.0, 4000, seed=0), 4000),
    "untuned_48k": (live_speech(3.0, 48000, seed=0), 48000),
    "huge_gain": (live_speech(3.0, RATE, seed=0) * 1e6, RATE),
    "tiny_gain": (live_speech(3.0, RATE, seed=0) * 1e-9, RATE),
    "all_nan": (np.full(8000, np.nan), RATE),
    "some_inf": (
        np.where(np.arange(24000) % 97 == 0, np.inf, live_speech(3.0, RATE, seed=0)),
        RATE,
    ),
    "stereo": (np.stack([live_speech(3.0, RATE, seed=0)] * 2), RATE),
}


@pytest.mark.parametrize(("name", "case"), DEGENERATE.items(), ids=list(DEGENERATE))
def test_degenerate_input_returns_something_sane(
    name: str, case: tuple[np.ndarray, int], engine: AnalysisEngine
) -> None:
    signal, rate = case
    factors = engine.analyse(signal, rate)

    assert len(factors) == len(FACTOR_ORDER)
    for factor in factors:
        assert np.isfinite(factor.score), (
            f"{name}: {factor.name} produced {factor.score}"
        )
        assert 0.0 <= factor.score <= 1.0
        assert len(factor.detail or "") <= 500


@pytest.mark.parametrize(
    "name",
    ["silence", "white_noise", "pure_tone", "empty", "one_sample", "120ms", "all_nan"],
)
def test_unanalysable_input_never_passes_an_audio_factor(
    name: str, engine: AnalysisEngine
) -> None:
    signal, rate = DEGENERATE[name]
    report = engine.analyse_report(signal, rate)
    for factor_name in (
        FactorName.SYNTHETIC_SPEECH,
        FactorName.REPLAY,
        FactorName.LIVENESS,
    ):
        assert report.verdict(factor_name) is not FactorVerdict.PASS


def test_degenerate_input_never_authenticates_the_caller(
    engine: AnalysisEngine,
) -> None:
    # The worst outcome in this service is allowing a payment on evidence
    # that does not exist.
    for name in ("silence", "empty", "120ms", "all_nan", "white_noise"):
        signal, rate = DEGENERATE[name]
        factors = engine.analyse(signal, rate)
        assert decide(blend_risk(factors)) is not Decision.ALLOW, name


def test_conditioning_problems_are_reported_to_the_reviewer(
    engine: AnalysisEngine,
) -> None:
    report = engine.analyse_report(clipped(live_speech(3.0, RATE, seed=0), 0.01), RATE)
    assert any("clipped" in note for note in report.notes)
    scored = [
        factor
        for factor in report.factors
        if factor.verdict is not FactorVerdict.NOT_RUN
    ]
    assert scored
    assert all("Audio:" in (factor.detail or "") for factor in scored)


def test_analysis_is_fast_enough_for_the_client_timeout(engine: AnalysisEngine) -> None:
    # shared.contract.CLIENT_TIMEOUT_MS is 800 ms for the whole round trip.
    report = engine.analyse_report(live_speech(4.0, RATE, seed=0), RATE)
    assert report.elapsed_ms < 400.0


# ---------------------------------------------------------------------------
# End to end through the real scoring code
# ---------------------------------------------------------------------------


def _decision(factors: list[Factor]) -> Decision:
    if fatal_factor(factors):
        return Decision.DENY
    return decide(blend_risk(factors))


def test_a_genuine_caller_is_allowed(engine: AnalysisEngine) -> None:
    factors = engine.analyse(
        live_speech(4.0, RATE, seed=0),
        RATE,
        metadata=CallMetadata(
            turn_latencies_ms=(640.0, 1180.0, 410.0, 2050.0),
            prompt_relevance=(0.9, 0.88, 0.94, 0.86),
        ),
        reference_audio=live_speech(4.0, RATE, seed=7),
    )
    assert _decision(factors) is Decision.ALLOW


def test_a_replayed_recording_is_denied(engine: AnalysisEngine) -> None:
    factors = engine.analyse(
        replay_of(live_speech(4.0, RATE, seed=0), RATE, seed=5), RATE
    )
    assert fatal_factor(factors) is not None
    assert _decision(factors) is Decision.DENY


def test_a_synthesised_caller_is_not_allowed(engine: AnalysisEngine) -> None:
    factors = engine.analyse(tts_speech(4.0, RATE, seed=0), RATE)
    assert _decision(factors) is not Decision.ALLOW


def test_an_automated_caller_is_denied_on_timing(engine: AnalysisEngine) -> None:
    factors = engine.analyse(
        live_speech(4.0, RATE, seed=0),
        RATE,
        metadata=CallMetadata(
            turn_latencies_ms=(145.0, 152.0, 139.0, 148.0, 151.0),
            prompt_relevance=(0.92, 0.9, 0.93, 0.89, 0.91),
        ),
    )
    assert _decision(factors) is Decision.DENY


def test_the_convenience_wrapper_matches_the_engine(engine: AnalysisEngine) -> None:
    signal = live_speech(3.0, RATE, seed=0)
    assert [f.model_dump() for f in analyse_call(signal, RATE)] == [
        f.model_dump() for f in engine.analyse(signal, RATE)
    ]
