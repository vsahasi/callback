"""Automated-caller detection from turn timing.

No audio here. The point of this module is that it refuses to produce a
number without metadata, so most of these tests are about `not_run`.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.behaviour import (  # noqa: E402
    MIN_TURNS_FOR_LATENCY,
    CallMetadata,
    analyse_behaviour,
)
from shared.contract import FactorVerdict  # noqa: E402

HUMAN = CallMetadata(
    turn_latencies_ms=(640.0, 1180.0, 410.0, 2050.0, 830.0),
    prompt_relevance=(0.91, 0.88, 0.95, 0.83, 0.9),
)
BOT = CallMetadata(
    turn_latencies_ms=(145.0, 152.0, 139.0, 148.0, 151.0),
    prompt_relevance=(0.92, 0.9, 0.93, 0.89, 0.91),
)
OFF_SCRIPT = CallMetadata(
    turn_latencies_ms=(910.0, 1420.0, 600.0, 1750.0),
    prompt_relevance=(0.22, 0.14, 0.31, 0.18),
)


# ---------------------------------------------------------------------------
# No metadata, no opinion
# ---------------------------------------------------------------------------


def test_without_metadata_the_factor_does_not_run() -> None:
    result = analyse_behaviour(None)
    assert result.verdict is FactorVerdict.NOT_RUN
    assert "turn_latencies_ms" in result.detail
    assert result.measurements == {}


def test_empty_metadata_does_not_run_either() -> None:
    assert analyse_behaviour(CallMetadata()).verdict is FactorVerdict.NOT_RUN


def test_too_few_turns_does_not_run() -> None:
    result = analyse_behaviour(CallMetadata(turn_latencies_ms=(500.0, 600.0)))
    assert result.verdict is FactorVerdict.NOT_RUN
    assert str(MIN_TURNS_FOR_LATENCY) in result.detail


# ---------------------------------------------------------------------------
# The headline direction
# ---------------------------------------------------------------------------


def test_a_bot_scores_materially_below_a_human() -> None:
    human = analyse_behaviour(HUMAN)
    bot = analyse_behaviour(BOT)

    assert human.verdict is FactorVerdict.PASS
    assert bot.verdict is FactorVerdict.FAIL
    assert human.score - bot.score >= 0.6


def test_metronomic_timing_is_allowed_to_reach_the_fatal_band() -> None:
    # Five replies in 150 ms with 3% variance is about as clear as
    # behavioural evidence gets, and detector.scoring denies outright at or
    # below 0.10.
    bot = analyse_behaviour(BOT)
    assert bot.score <= 0.10
    assert bot.measurements["score_floor"] <= 0.02


def test_one_odd_latency_alone_cannot_reach_the_fatal_band() -> None:
    # Fast but variable: a person who happened to answer quickly. The floor
    # keeps a single cue out of the fatal band.
    quick_human = CallMetadata(turn_latencies_ms=(180.0, 620.0, 240.0, 1500.0))
    result = analyse_behaviour(quick_human)
    assert result.score > 0.10


def test_replies_that_ignore_the_prompts_are_penalised_on_their_own() -> None:
    # Human-looking timing, content that does not track the questions: the
    # signature of a recording being played at the agent.
    result = analyse_behaviour(OFF_SCRIPT)
    assert result.score < 0.5
    assert result.verdict is not FactorVerdict.PASS
    assert "not tracking" in result.detail


def test_answering_correctly_is_not_evidence_of_being_human() -> None:
    # prompt_tracking is guilt-only. A bot that answers the question must not
    # earn credit for it and dilute the latency cues.
    with_relevance = analyse_behaviour(BOT)
    without_relevance = analyse_behaviour(
        CallMetadata(turn_latencies_ms=BOT.turn_latencies_ms)
    )
    assert with_relevance.score == pytest.approx(without_relevance.score, abs=0.02)


# ---------------------------------------------------------------------------
# Parsing the agent's free-form metadata blob
# ---------------------------------------------------------------------------


def test_from_mapping_reads_the_documented_keys() -> None:
    parsed = CallMetadata.from_mapping(
        {
            "turn_latencies_ms": [640, 1180.5, 410],
            "response_durations_ms": [2000, 1500],
            "prompt_relevance": [0.9, 0.8],
            "barge_ins": 2,
        }
    )
    assert parsed.turn_latencies_ms == (640.0, 1180.5, 410.0)
    assert parsed.response_durations_ms == (2000.0, 1500.0)
    assert parsed.prompt_relevance == (0.9, 0.8)
    assert parsed.barge_ins == 2
    assert parsed.has_timing


def test_from_mapping_drops_junk_rather_than_raising() -> None:
    # A malformed metadata blob must degrade this factor to not_run, not
    # fail the request.
    parsed = CallMetadata.from_mapping(
        {
            "turn_latencies_ms": [640, "soon", None, float("nan"), True, 410, 500],
            "prompt_relevance": "very",
            "barge_ins": "lots",
        }
    )
    assert parsed.turn_latencies_ms == (640.0, 410.0, 500.0)
    assert parsed.prompt_relevance == ()
    assert parsed.barge_ins is None
    assert analyse_behaviour(parsed).verdict is not FactorVerdict.NOT_RUN


@pytest.mark.parametrize(
    "raw", [None, {}, {"unrelated": 1}, {"turn_latencies_ms": None}]
)
def test_from_mapping_on_nothing_useful_is_empty(raw: object) -> None:
    parsed = CallMetadata.from_mapping(raw)  # type: ignore[arg-type]
    assert parsed.is_empty
    assert analyse_behaviour(parsed).verdict is FactorVerdict.NOT_RUN


def test_negative_and_non_finite_latencies_are_discarded() -> None:
    result = analyse_behaviour(
        CallMetadata(turn_latencies_ms=(-100.0, 700.0, 900.0, 1200.0))
    )
    assert result.measurements["turn_latency_count"] == 3.0
    assert np.isfinite(result.score)


def test_detail_reports_the_measurements_and_stays_in_the_contract_limit() -> None:
    for metadata in (HUMAN, BOT, OFF_SCRIPT):
        detail = analyse_behaviour(metadata).detail
        assert "turns" in detail
        assert len(detail) <= 500
