"""Known-answer tests for EER and min t-DCF.

No torch, no numpy, no dataset: `training.metrics` is pure standard library
precisely so that the project's scoreboard can be verified in the default dev
environment.

Every expected number below is either an analytic invariant or was computed by
hand from the reference algorithm; the hand derivations are written out in the
test docstrings so a future reader can check them without re-deriving.
"""

from __future__ import annotations

import math
import random

import pytest

from training.metrics import (
    ASV_COST_MODEL_2019,
    ASV_COST_MODEL_2021,
    AsvErrorRates,
    compute_det_curve,
    compute_eer,
    compute_tdcf_2019,
    compute_tdcf_2021,
    evaluate_cm_scores,
    obtain_asv_error_rates,
    per_attack_eer,
    read_score_file,
    split_scores_by_label,
    write_score_file,
)

# A cost model that reduces the 2019 t-DCF to Pmiss_cm + Pfa_cm, so min t-DCF
# becomes a quantity that can be minimised by hand off the DET curve.
TRIVIAL_COST_MODEL = {
    "Pspoof": 1.0,
    "Ptar": 1.0,
    "Pnon": 0.0,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 1.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 1.0,
}
PERFECT_ASV = AsvErrorRates(p_miss=0.0, p_fa=0.0, p_miss_spoof=0.0)

# A plausible, non-degenerate ASV operating point for the realistic cases.
REAL_ASV = AsvErrorRates(p_miss=0.05, p_fa=0.01, p_miss_spoof=0.9)


# ---------------------------------------------------------------------------
# DET curve
# ---------------------------------------------------------------------------


class TestDetCurve:
    def test_curve_has_one_point_per_score_plus_a_leading_sentinel(self):
        curve = compute_det_curve([1.0, 2.0, 3.0], [-1.0, 0.0])
        assert len(curve) == 6
        assert len(curve.p_miss) == len(curve.p_fa) == len(curve.thresholds) == 6

    def test_curve_starts_accepting_everything_and_ends_rejecting_everything(self):
        curve = compute_det_curve([1.0, 2.0, 3.0], [-1.0, 0.0])
        assert curve.p_miss[0] == 0.0
        assert curve.p_fa[0] == 1.0
        assert curve.p_miss[-1] == 1.0
        assert curve.p_fa[-1] == 0.0

    def test_leading_threshold_is_below_the_lowest_score(self):
        curve = compute_det_curve([1.0], [-5.0])
        assert curve.thresholds[0] == pytest.approx(-5.001)

    def test_p_miss_is_monotonically_non_decreasing_and_p_fa_non_increasing(self):
        rng = random.Random(7)
        targets = [rng.gauss(1.0, 1.0) for _ in range(50)]
        nontargets = [rng.gauss(-1.0, 1.0) for _ in range(50)]
        curve = compute_det_curve(targets, nontargets)
        assert all(
            curve.p_miss[i] <= curve.p_miss[i + 1] + 1e-12
            for i in range(len(curve) - 1)
        )
        assert all(
            curve.p_fa[i] >= curve.p_fa[i + 1] - 1e-12 for i in range(len(curve) - 1)
        )

    @pytest.mark.parametrize(
        ("targets", "nontargets"),
        [([], [1.0]), ([1.0], []), ([], [])],
    )
    def test_empty_class_is_rejected(self, targets, nontargets):
        with pytest.raises(ValueError, match="empty"):
            compute_det_curve(targets, nontargets)

    def test_nan_is_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            compute_det_curve([1.0, float("nan")], [0.0])


# ---------------------------------------------------------------------------
# EER known answers
# ---------------------------------------------------------------------------


class TestEerKnownAnswers:
    def test_perfectly_separable_scores_give_zero(self):
        """Every bonafide above every spoof: there is a threshold with no
        errors either way, so the DET curve passes through (0, 0)."""

        eer, threshold = compute_eer([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0])
        assert eer == 0.0
        # The returned threshold is the highest REJECTED score, so the
        # operating rule is "accept score > threshold". Here that is -1.0, the
        # top spoof score: accept above it and every bonafide passes while
        # every spoof is refused.
        assert threshold == pytest.approx(-1.0)

    def test_fully_inverted_scores_give_one(self):
        """Every spoof above every bonafide. This metric is not folded at 0.5,
        so a value above 0.5 means the score polarity is flipped."""

        eer, _ = compute_eer([-3.0, -2.0, -1.0], [1.0, 2.0, 3.0])
        assert eer == 1.0

    def test_interleaved_identical_distributions_give_exactly_one_half(self):
        """Bonafide 0,2,4,6 and spoof 1,3,5,7 interleave perfectly.

        Sorted labels are 1,0,1,0,1,0,1,0. Working the reference algorithm by
        hand gives Pmiss = [0, .25, .25, .5, .5, .75, .75, 1, 1] and
        Pfa = [1, 1, .75, .75, .5, .5, .25, .25, 0]. The two meet exactly at
        index 4, where both are 0.5.
        """

        eer, _ = compute_eer([0.0, 2.0, 4.0, 6.0], [1.0, 3.0, 5.0, 7.0])
        assert eer == pytest.approx(0.5)

    def test_hand_computed_one_third(self):
        """Bonafide [3, 4, 5], spoof [1, 2, 3.5].

        Sorted: 1(spoof) 2(spoof) 3(bona) 3.5(spoof) 4(bona) 5(bona), so
        labels are 0,0,1,0,1,1 and

            Pmiss = [0, 0,   0,   1/3, 1/3, 2/3, 1]
            Pfa   = [1, 2/3, 1/3, 1/3, 0,   0,   0]

        They are equal at index 3 at 1/3 each, with threshold 3.0 (accepting
        score >= 3 rejects one of three bonafide and accepts one of three
        spoof).
        """

        eer, threshold = compute_eer([3.0, 4.0, 5.0], [1.0, 2.0, 3.5])
        assert eer == pytest.approx(1.0 / 3.0)
        assert threshold == pytest.approx(3.0)

    def test_second_hand_computed_case_with_its_threshold(self):
        """Bonafide [0.5, 0.6], spoof [0.4, 0.55].

        Sorted labels 0,1,0,1 give Pmiss = [0, 0, .5, .5, 1] and
        Pfa = [1, .5, .5, 0, 0], equal at index 2 at 0.5, threshold 0.5.
        """

        eer, threshold = compute_eer([0.5, 0.6], [0.4, 0.55])
        assert eer == pytest.approx(0.5)
        assert threshold == pytest.approx(0.5)

    def test_large_samples_from_one_distribution_approach_one_half(self):
        rng = random.Random(20240515)
        targets = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        nontargets = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        eer, _ = compute_eer(targets, nontargets)
        assert eer == pytest.approx(0.5, abs=0.03)

    def test_well_separated_gaussians_give_the_analytic_eer(self):
        """For two unit-variance Gaussians separated by d, the analytic EER is
        Phi(-d / 2). With d = 2 that is Phi(-1) = 0.1587."""

        rng = random.Random(99)
        targets = [rng.gauss(1.0, 1.0) for _ in range(20_000)]
        nontargets = [rng.gauss(-1.0, 1.0) for _ in range(20_000)]
        eer, _ = compute_eer(targets, nontargets)
        analytic = 0.5 * math.erfc(1.0 / math.sqrt(2.0))  # Phi(-1)
        assert eer == pytest.approx(analytic, abs=0.01)

    def test_eer_is_invariant_under_a_positive_affine_rescaling(self):
        rng = random.Random(3)
        targets = [rng.gauss(0.5, 1.0) for _ in range(200)]
        nontargets = [rng.gauss(-0.5, 1.0) for _ in range(200)]
        base, _ = compute_eer(targets, nontargets)
        scaled, _ = compute_eer(
            [7.0 * s + 3.0 for s in targets], [7.0 * s + 3.0 for s in nontargets]
        )
        assert scaled == pytest.approx(base)

    def test_negating_scores_maps_eer_to_its_complement(self):
        targets = [3.0, 4.0, 5.0]
        nontargets = [1.0, 2.0, 3.5]
        forward, _ = compute_eer(targets, nontargets)
        backward, _ = compute_eer([-s for s in targets], [-s for s in nontargets])
        assert forward + backward == pytest.approx(1.0)

    def test_all_scores_tied_is_the_degenerate_worst_case(self):
        """Every score identical. The reference algorithm's stable sort puts
        the targets first, so the closest Pmiss/Pfa pair is (1, 1). This is
        documented rather than desirable: it is what the official
        implementation returns, and a run that produces it has a dead model,
        not a metric bug."""

        eer, _ = compute_eer([0.0] * 4, [0.0] * 4)
        assert eer == pytest.approx(1.0)

    def test_accepts_tuples_and_objects_exposing_tolist(self):
        class FakeArray:
            def __init__(self, values):
                self._values = values

            def tolist(self):
                return list(self._values)

        plain, _ = compute_eer((3.0, 4.0, 5.0), (1.0, 2.0, 3.5))
        wrapped, _ = compute_eer(FakeArray([3.0, 4.0, 5.0]), FakeArray([1.0, 2.0, 3.5]))
        assert plain == pytest.approx(wrapped)

    def test_scalar_input_is_rejected(self):
        with pytest.raises(TypeError, match="scalar"):
            compute_eer(1.0, [0.0])


# ---------------------------------------------------------------------------
# ASV error rates
# ---------------------------------------------------------------------------


class TestAsvErrorRates:
    def test_rates_outside_the_unit_interval_are_rejected(self):
        with pytest.raises(ValueError, match="p_miss"):
            AsvErrorRates(p_miss=1.5, p_fa=0.0, p_miss_spoof=0.0)
        with pytest.raises(ValueError, match="p_fa"):
            AsvErrorRates(p_miss=0.0, p_fa=-0.1, p_miss_spoof=0.0)

    def test_p_fa_spoof_is_the_complement_of_p_miss_spoof(self):
        rates = AsvErrorRates(p_miss=0.1, p_fa=0.2, p_miss_spoof=0.75)
        assert rates.p_fa_spoof == pytest.approx(0.25)

    def test_hand_computed_rates_at_an_explicit_threshold(self):
        """targets [3, 4, 5], nontargets [1, 2], spoofs [2, 4], threshold 2.5.

        Acceptance is score >= threshold, so: no target falls below 2.5
        (p_miss 0), no nontarget reaches it (p_fa 0), and one of the two
        spoofs (2.0) falls below it (p_miss_spoof 0.5).
        """

        rates, threshold = obtain_asv_error_rates(
            [3.0, 4.0, 5.0], [1.0, 2.0], [2.0, 4.0], 2.5
        )
        assert threshold == 2.5
        assert rates.p_miss == pytest.approx(0.0)
        assert rates.p_fa == pytest.approx(0.0)
        assert rates.p_miss_spoof == pytest.approx(0.5)

    def test_default_threshold_is_the_asv_eer_threshold(self):
        rates, threshold = obtain_asv_error_rates(
            [3.0, 4.0, 5.0], [1.0, 2.0], [2.0, 4.0]
        )
        expected_threshold = compute_eer([3.0, 4.0, 5.0], [1.0, 2.0])[1]
        assert threshold == pytest.approx(expected_threshold)
        assert 0.0 <= rates.p_miss <= 1.0

    def test_missing_spoof_scores_are_rejected(self):
        with pytest.raises(ValueError, match="spoof_asv is empty"):
            obtain_asv_error_rates([3.0, 4.0], [1.0, 2.0], [])


# ---------------------------------------------------------------------------
# t-DCF, 2019 form
# ---------------------------------------------------------------------------


class TestTdcf2019:
    def test_trivial_cost_model_reduces_to_the_hand_computed_minimum(self):
        """With Ptar = Pspoof = all costs = 1, Pnon = 0 and a perfect ASV,
        C1 = C2 = 1 and the normalised t-DCF is just min(Pmiss + Pfa).

        For bonafide [3, 4, 5] and spoof [1, 2, 3.5] the DET curve (derived in
        TestEerKnownAnswers.test_hand_computed_one_third) gives the sums
        [1, 2/3, 1/3, 2/3, 1/3, 2/3, 1], whose minimum is 1/3.

        That minimum is reached twice, at index 2 and index 4. Like numpy's
        argmin in the reference implementation, the first one wins, so the
        threshold is index 2's, which is 2.0.
        """

        value, threshold = compute_tdcf_2019(
            [3.0, 4.0, 5.0], [1.0, 2.0, 3.5], PERFECT_ASV, TRIVIAL_COST_MODEL
        )
        assert value == pytest.approx(1.0 / 3.0)
        assert threshold == pytest.approx(2.0)

    def test_perfectly_separable_scores_give_zero(self):
        value, _ = compute_tdcf_2019([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0], REAL_ASV)
        assert value == pytest.approx(0.0)

    def test_fully_inverted_scores_give_exactly_one(self):
        """The curve always contains (Pmiss 0, Pfa 1) and (Pmiss 1, Pfa 0), so
        the cost never exceeds min(C1, C2) and the normalised value is capped
        at 1.0. A useless countermeasure hits that cap."""

        value, _ = compute_tdcf_2019([-3.0, -2.0, -1.0], [1.0, 2.0, 3.0], REAL_ASV)
        assert value == pytest.approx(1.0)

    def test_normalised_value_is_always_within_the_unit_interval(self):
        rng = random.Random(11)
        for separation in (0.0, 0.5, 1.0, 2.0, 4.0):
            targets = [rng.gauss(separation, 1.0) for _ in range(300)]
            nontargets = [rng.gauss(0.0, 1.0) for _ in range(300)]
            value, _ = compute_tdcf_2019(targets, nontargets, REAL_ASV)
            assert 0.0 <= value <= 1.0

    def test_a_better_separated_system_scores_no_worse(self):
        rng = random.Random(13)
        nontargets = [rng.gauss(0.0, 1.0) for _ in range(500)]
        weak = [rng.gauss(0.5, 1.0) for _ in range(500)]
        strong = [value + 3.0 for value in weak]
        weak_value, _ = compute_tdcf_2019(weak, nontargets, REAL_ASV)
        strong_value, _ = compute_tdcf_2019(strong, nontargets, REAL_ASV)
        assert strong_value <= weak_value

    def test_matches_the_formula_written_out_independently(self):
        """Recompute C1, C2 and the curve minimum straight from the evaluation
        plan, with no shared helper, and compare."""

        targets = [3.0, 4.0, 5.0]
        nontargets = [1.0, 2.0, 3.5]
        model = ASV_COST_MODEL_2019
        c1 = (
            model["Ptar"] * (model["Cmiss_cm"] - model["Cmiss_asv"] * REAL_ASV.p_miss)
            - model["Pnon"] * model["Cfa_asv"] * REAL_ASV.p_fa
        )
        c2 = model["Cfa_cm"] * model["Pspoof"] * (1.0 - REAL_ASV.p_miss_spoof)
        curve = compute_det_curve(targets, nontargets)
        expected = min(
            c1 * curve.p_miss[i] + c2 * curve.p_fa[i] for i in range(len(curve))
        ) / min(c1, c2)

        value, _ = compute_tdcf_2019(targets, nontargets, REAL_ASV)
        assert value == pytest.approx(expected)

    def test_an_asv_that_rejects_every_spoof_makes_the_metric_undefined(self):
        """p_miss_spoof == 1 drives C2 to zero: there is nothing left for the
        countermeasure to catch, so there is no cost to minimise. The official
        implementation exits here; this raises."""

        asv = AsvErrorRates(p_miss=0.0, p_fa=0.0, p_miss_spoof=1.0)
        with pytest.raises(ValueError, match="undefined"):
            compute_tdcf_2019([1.0, 2.0], [-1.0, 0.0], asv)

    def test_negative_weights_are_rejected(self):
        """A huge Pnon * Cfa_asv * Pfa_asv term can drive C1 negative, which
        means the cost model and the ASV operating point disagree."""

        model = dict(ASV_COST_MODEL_2019)
        model["Pnon"] = 0.9
        model["Cfa_asv"] = 1000.0
        asv = AsvErrorRates(p_miss=0.0, p_fa=1.0, p_miss_spoof=0.5)
        with pytest.raises(ValueError, match="non-negative"):
            compute_tdcf_2019([1.0, 2.0], [-1.0, 0.0], asv, model)


# ---------------------------------------------------------------------------
# t-DCF, 2021 form
# ---------------------------------------------------------------------------


class TestTdcf2021:
    def test_perfect_countermeasure_bottoms_out_at_the_asv_cost_floor(self):
        """The 2021 form adds C0, the ASV's own cost, which the countermeasure
        cannot reduce. A perfect countermeasure therefore scores
        C0 / (C0 + min(C1, C2)), not 0."""

        model = ASV_COST_MODEL_2021
        c0 = (
            model["Ptar"] * model["Cmiss"] * REAL_ASV.p_miss
            + model["Pnon"] * model["Cfa"] * REAL_ASV.p_fa
        )
        c1 = model["Ptar"] * model["Cmiss"] - c0
        c2 = model["Pspoof"] * model["Cfa_spoof"] * REAL_ASV.p_fa_spoof
        expected = c0 / (c0 + min(c1, c2))

        value, _ = compute_tdcf_2021([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0], REAL_ASV)
        assert value == pytest.approx(expected)

    def test_a_perfect_asv_makes_the_floor_vanish(self):
        """With p_miss = p_fa = 0 the floor C0 is zero, so the 2021 form
        behaves like the 2019 one and a perfect countermeasure scores 0."""

        asv = AsvErrorRates(p_miss=0.0, p_fa=0.0, p_miss_spoof=0.5)
        value, _ = compute_tdcf_2021([1.0, 2.0, 3.0], [-3.0, -2.0, -1.0], asv)
        assert value == pytest.approx(0.0)

    def test_useless_countermeasure_gives_exactly_one(self):
        value, _ = compute_tdcf_2021([-3.0, -2.0, -1.0], [1.0, 2.0, 3.0], REAL_ASV)
        assert value == pytest.approx(1.0)

    def test_normalised_value_is_always_within_the_unit_interval(self):
        rng = random.Random(17)
        for separation in (0.0, 1.0, 3.0):
            targets = [rng.gauss(separation, 1.0) for _ in range(200)]
            nontargets = [rng.gauss(0.0, 1.0) for _ in range(200)]
            value, _ = compute_tdcf_2021(targets, nontargets, REAL_ASV)
            assert 0.0 <= value <= 1.0

    def test_an_asv_that_rejects_every_spoof_still_leaves_a_defined_metric(self):
        """Unlike the 2019 form, C2 == 0 is survivable here as long as C0 > 0,
        because the denominator is C0 + min(C1, C2)."""

        asv = AsvErrorRates(p_miss=0.05, p_fa=0.01, p_miss_spoof=1.0)
        value, _ = compute_tdcf_2021([1.0, 2.0], [-1.0, 0.0], asv)
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Aggregation and I/O
# ---------------------------------------------------------------------------


class TestEvaluateCmScores:
    def test_reports_eer_only_without_an_asv_operating_point(self):
        result = evaluate_cm_scores([3.0, 4.0, 5.0], [1.0, 2.0, 3.5])
        assert result.eer == pytest.approx(1.0 / 3.0)
        assert result.min_tdcf is None
        assert result.tdcf_version is None
        assert result.n_bonafide == 3
        assert result.n_spoof == 3
        assert "EER" in result.summary()
        assert "t-DCF" not in result.summary()

    def test_reports_tdcf_when_given_asv_rates(self):
        result = evaluate_cm_scores([3.0, 4.0, 5.0], [1.0, 2.0, 3.5], asv=REAL_ASV)
        assert result.min_tdcf is not None
        assert result.tdcf_version == "2019"
        assert "t-DCF" in result.summary()

    def test_selects_the_2021_cost_form_on_request(self):
        nineteen = evaluate_cm_scores(
            [3.0, 4.0, 5.0], [1.0, 2.0, 3.5], asv=REAL_ASV, tdcf_version="2019"
        )
        twentyone = evaluate_cm_scores(
            [3.0, 4.0, 5.0], [1.0, 2.0, 3.5], asv=REAL_ASV, tdcf_version="2021"
        )
        assert nineteen.min_tdcf != pytest.approx(twentyone.min_tdcf)
        assert twentyone.tdcf_version == "2021"

    def test_unknown_tdcf_version_is_rejected(self):
        with pytest.raises(ValueError, match="tdcf_version"):
            evaluate_cm_scores(
                [1.0, 2.0], [-1.0, 0.0], asv=REAL_ASV, tdcf_version="2025"
            )

    def test_per_attack_breakdown(self):
        result = evaluate_cm_scores(
            [3.0, 4.0, 5.0],
            [1.0, 2.0, 3.5, 9.0, 9.5],
            spoof_scores_by_attack={
                "A01": [1.0, 2.0, 3.5],
                "A02": [9.0, 9.5],
            },
        )
        assert result.per_attack_eer is not None
        assert result.per_attack_eer["A01"] == pytest.approx(1.0 / 3.0)
        # A02 scores above every bonafide, so it defeats the system entirely.
        assert result.per_attack_eer["A02"] == pytest.approx(1.0)


class TestPerAttackEer:
    def test_shares_the_bonafide_pool_across_attacks(self):
        results = per_attack_eer(
            [3.0, 4.0, 5.0], {"A01": [1.0, 2.0, 3.5], "A02": [-5.0]}
        )
        assert results["A01"] == pytest.approx(1.0 / 3.0)
        assert results["A02"] == pytest.approx(0.0)

    def test_empty_attacks_are_skipped_not_reported_as_zero(self):
        results = per_attack_eer([3.0, 4.0], {"A01": [], "A02": [1.0]})
        assert "A01" not in results
        assert "A02" in results


class TestScoreFileIo:
    def test_round_trips(self, tmp_path):
        path = tmp_path / "scores.txt"
        write_score_file(path, ["LA_E_1", "LA_E_2"], [0.25, -1.5])
        assert read_score_file(path) == {"LA_E_1": 0.25, "LA_E_2": -1.5}

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "scores.txt"
        write_score_file(path, ["LA_E_1"], [1.0])
        assert path.exists()

    def test_mismatched_lengths_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="2 utterance ids but 1 scores"):
            write_score_file(tmp_path / "s.txt", ["a", "b"], [1.0])

    def test_reader_skips_blanks_and_comments_and_takes_the_last_column(self, tmp_path):
        path = tmp_path / "scores.txt"
        path.write_text(
            "# a comment\n\nLA_E_1 extra 0.5\nLA_E_2 -0.25\n", encoding="utf-8"
        )
        assert read_score_file(path) == {"LA_E_1": 0.5, "LA_E_2": -0.25}

    def test_malformed_line_is_rejected_with_its_line_number(self, tmp_path):
        path = tmp_path / "scores.txt"
        path.write_text("LA_E_1 0.5\nbroken\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":2:"):
            read_score_file(path)


class TestSplitScoresByLabel:
    def test_splits_and_reports_the_gap(self):
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        labels = {"a": "bonafide", "b": "spoof", "c": "spoof", "d": "bonafide"}
        bonafide, spoof, missing = split_scores_by_label(scores, labels)
        assert bonafide == [1.0]
        assert sorted(spoof) == [2.0, 3.0]
        assert missing == ["d"]

    def test_a_complete_score_file_reports_no_gap(self):
        bonafide, spoof, missing = split_scores_by_label(
            {"a": 1.0, "b": 2.0}, {"a": "bonafide", "b": "spoof"}
        )
        assert missing == []
        assert bonafide == [1.0] and spoof == [2.0]
