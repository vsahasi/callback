"""The calibration harness, and the decisions it produces.

`detector.analysis.calibration` claims to be the table the thresholds were
chosen from and the one to re-read when real audio arrives. This file holds
it to that: every condition in `CONDITIONS` has to run, and the decision each
one produces through the real `detector.scoring` code has to be the one its
`expectation` string describes.

That makes the harness and the tests a single source of truth. Adding a
condition without deciding what it should do will fail here.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from detector.analysis.calibration import (  # noqa: E402
    CONDITIONS,
    Row,
    format_measurements,
    format_table,
    main,
    run,
)
from shared.contract import FactorName  # noqa: E402

# What the service should do for each synthetic condition. A set, not a
# single value, wherever two outcomes are both defensible: the point is to
# catch a detector that silently starts allowing an attack, not to freeze a
# risk score to three decimal places.
EXPECTED_DECISIONS: dict[str, set[str]] = {
    "live_4s": {"allow"},
    "live_3s": {"allow"},
    "live_1p5s": {"allow", "challenge"},
    "live_16k": {"allow"},
    "replay_of_live": {"deny"},
    "replay_of_live_pcm": {"deny"},
    "replay_dry_room": {"deny"},
    "tts_4s": {"deny"},
    "tts_3s": {"deny", "challenge"},
    "replay_of_tts": {"deny"},
    # Unanalysable audio must never authenticate a caller. Failing closed to
    # challenge or deny is the contract's own policy.
    "silence": {"challenge", "deny"},
    "white_noise": {"challenge", "deny"},
    "pure_tone": {"challenge", "deny"},
    "impulse_train": {"challenge", "deny"},
    "clipped_live": {"allow", "challenge"},
    "dc_offset_live": {"allow"},
    "live_4khz": {"challenge", "allow"},
    "live_48khz": {"allow"},
    "live_120ms": {"challenge", "deny"},
    "live_human_turns": {"allow"},
    "live_bot_turns": {"deny"},
    "replay_replay_turns": {"deny"},
    "live_matched_reference": {"allow"},
    "live_mismatched_reference": {"challenge", "deny"},
}


@pytest.fixture(scope="module")
def rows() -> list[Row]:
    return run()


@pytest.fixture(scope="module")
def by_name(rows: list[Row]) -> dict[str, Row]:
    return {row.name: row for row in rows}


def test_every_condition_is_covered_by_an_expectation() -> None:
    assert {condition.name for condition in CONDITIONS} == set(EXPECTED_DECISIONS)


def test_every_condition_has_a_stated_expectation_in_prose() -> None:
    for condition in CONDITIONS:
        assert condition.expectation, condition.name


@pytest.mark.parametrize("name", sorted(EXPECTED_DECISIONS))
def test_each_condition_produces_the_decision_it_should(
    name: str, by_name: dict[str, Row]
) -> None:
    row = by_name[name]
    assert row.decision in EXPECTED_DECISIONS[name], (
        f"{name}: decided {row.decision} at risk {row.risk:.2f} (scores {row.scores})"
    )


def test_no_condition_produces_a_non_finite_number(rows: list[Row]) -> None:
    for row in rows:
        assert 0.0 <= row.risk <= 1.0
        for name, score in row.scores.items():
            assert np.isfinite(score), f"{row.name}/{name}"
            assert 0.0 <= score <= 1.0


def test_the_harness_is_deterministic() -> None:
    first = {row.name: dict(row.scores) for row in run(CONDITIONS[:4])}
    second = {row.name: dict(row.scores) for row in run(CONDITIONS[:4])}
    assert first == second


def test_the_margins_the_thresholds_rest_on(by_name: dict[str, Row]) -> None:
    # The three separations the whole package is built on, asserted as
    # orderings with a stated margin rather than as absolute scores.
    live = by_name["live_4s"].scores
    replayed = by_name["replay_of_live"].scores
    rendered = by_name["tts_4s"].scores

    assert live[FactorName.REPLAY.value] - replayed[FactorName.REPLAY.value] >= 0.55
    assert (
        live[FactorName.SYNTHETIC_SPEECH.value]
        - rendered[FactorName.SYNTHETIC_SPEECH.value]
        >= 0.50
    )
    assert live[FactorName.LIVENESS.value] - rendered[FactorName.LIVENESS.value] >= 0.45

    # And the two attacks stay distinguishable from each other, so the
    # reason shown to a reviewer is the right one.
    assert replayed[FactorName.SYNTHETIC_SPEECH.value] >= 0.70
    assert rendered[FactorName.REPLAY.value] >= 0.70


def test_the_wider_sample_rate_does_not_change_the_verdicts(
    by_name: dict[str, Row],
) -> None:
    narrow = by_name["live_4s"]
    for wider in ("live_16k", "live_48khz"):
        for factor, verdict in narrow.verdicts.items():
            assert by_name[wider].verdicts[factor] == verdict, (wider, factor)


def test_removing_the_dc_offset_leaves_the_analysis_unchanged(
    by_name: dict[str, Row],
) -> None:
    clean = by_name["live_4s"].scores
    offset = by_name["dc_offset_live"].scores
    for factor, score in clean.items():
        assert offset[factor] == pytest.approx(score, abs=0.02), factor


def test_the_table_prints_every_condition_with_its_decision(rows: list[Row]) -> None:
    table = format_table(rows)
    for row in rows:
        assert row.name in table
    assert "decision" in table
    assert "not_run" in table


def test_the_measurement_dump_names_the_numbers_behind_the_scores(
    rows: list[Row],
) -> None:
    dump = format_measurements(rows)
    for key in ("comb_peak_ratio", "f0_perturbation_pct", "rhythm_cv"):
        assert key in dump


def test_the_harness_runs_as_a_command() -> None:
    assert main(["--only", "live_4s"]) == 0
    assert main(["--only", "live_4s", "--measurements"]) == 0
    assert main(["--only", "nothing-matches-this"]) == 2
