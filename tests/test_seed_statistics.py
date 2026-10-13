"""Independent analytic/numerical fixtures; no learned-model evidence."""

from copy import deepcopy
from fractions import Fraction
import json
import math
import unittest

import numpy as np

from representation_collapse.seed_statistics import (
    FAMILY_ORDER, paired_behavior_effects, reduce_seed_family, seed_t_summary,
    student_t_survival, student_t_upper_quantile,
)


def density_integral_upper_tail(t, df):
    """Independent Gauss-Legendre density integral, not an incomplete beta."""
    nodes, weights = np.polynomial.legendre.leggauss(192)
    points = (nodes + 1) * t / 2
    coefficient = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi * df)
    density = coefficient * (1 + points * points / df) ** (-(df + 1) / 2)
    return 0.5 - float(np.dot(weights, density)) * t / 2


def outcome(task, delta16, delta2=0.0):
    # Manufacture raw arm losses independently of the production aggregator.
    # The fixture is not an executed/eligible checkpoint.
    scale = 1 if task == "categorical" else 1 / 3
    return {"loss_units": "error_indicator" if task == "categorical" else "raw_squared_error",
            "paired_mean_raw_loss_differences": {"R2": scale * delta2, "R16": scale * delta16},
            "arm_mean_losses": {
                "R2": {"periodic": 0.25 + scale * delta2, "matched_control": 0.25},
                "R16": {"periodic": 0.25 + scale * delta16, "matched_control": 0.25}}}


def records_for_cell(model, task, values, seeds=None, delta2=None):
    if seeds is None:
        seeds = list(range(3, 3 + len(values)))
    if delta2 is None:
        delta2 = [0.0] * len(values)
    return [{"model": model, "task": task, "trained_seed": seed,
             "validation_eligible": True, "unavailable_reason": None,
             "outcome": outcome(task, value, short)}
            for seed, value, short in zip(seeds, values, delta2)]


class StudentCalibrationTests(unittest.TestCase):
    def test_analytic_cauchy_and_df2_calibration(self):
        # df=1 CDF=1/2+atan(t)/pi; df=2 CDF=1/2+t/(2sqrt(t²+2)).
        for statistic in (-100, -10, -2, -0.01, -1e-10, 0, 1e-10, 0.01, 1, 2, 10, 100):
            self.assertAlmostEqual(student_t_survival(statistic, 1),
                                   0.5 - math.atan(statistic) / math.pi, delta=3e-14)
            self.assertAlmostEqual(student_t_survival(statistic, 2),
                                   0.5 - statistic / (2 * math.sqrt(statistic ** 2 + 2)), delta=3e-14)

    def test_df9_independent_density_quadrature_and_documented_df3(self):
        for df in (3, 9, 29):
            for statistic in (0.01, 0.5, 1, 2, 3.5465084980786, 5):
                self.assertAlmostEqual(student_t_survival(statistic, df),
                                       density_integral_upper_tail(statistic, df), delta=5e-14)
        # Independent SciPy stdtr official documentation example, df=3,t=1.
        self.assertAlmostEqual(student_t_survival(1, 3), 1 - 0.8044988905221148, delta=2e-15)

    def test_df9_interval_coefficients_from_reviewed_proposal(self):
        self.assertAlmostEqual(student_t_upper_quantile(9, 0.003125) / math.sqrt(10),
                               1.1215044595072001, delta=2e-12)
        self.assertAlmostEqual(student_t_upper_quantile(9, 0.025) / math.sqrt(10),
                               0.7153569059706628, delta=2e-12)
        # Also check an analytic inverse, rather than only inverting our SF.
        for probability in (0.025, 0.003125):
            self.assertAlmostEqual(student_t_upper_quantile(1, probability),
                                   1 / math.tan(math.pi * probability), delta=2e-11)

    def test_invalid_numeric_domain_is_rejected(self):
        for t, df in ((math.nan, 9), (math.inf, 9), (True, 9), (1, 0), (1, 9.0), (1, True)):
            with self.assertRaises(ValueError):
                student_t_survival(t, df)
        for tail in (0, 0.5, 1, math.nan, True):
            with self.assertRaises(ValueError):
                student_t_upper_quantile(9, tail)

    def test_extreme_cauchy_inverse_tails_are_correct_or_explicitly_unsupported(self):
        # Independent analytic inverse. At p=1e-200 an unguarded squared-beta
        # calculation returned approximately 6.36e161 instead of 3.18e199.
        for probability in (1e-5, 1e-20, 1e-50, 1e-100):
            expected = 1 / math.tan(math.pi * probability)
            actual = student_t_upper_quantile(1, probability)
            self.assertAlmostEqual(actual / expected, 1, delta=2e-12)
        for probability in (1e-101, 1e-150, 1e-160, 1e-200, 1e-300):
            with self.assertRaisesRegex(ArithmeticError, "supported numerical limit"):
                student_t_upper_quantile(1, probability)

    def test_high_df_upper_tail_is_not_lost_by_central_mass_subtraction(self):
        # Independent reviewer: 80-digit mpmath direct beta evaluation.
        reference = 8.335351479300033e-23
        probability = student_t_survival(10, 1000)
        self.assertGreater(probability, 0)
        self.assertAlmostEqual(probability / reference, 1, delta=2e-12)


class PairAndUnitTests(unittest.TestCase):
    def setUp(self):
        self.ids = {h: np.array([[f"{h}:q{q}:i{i}" for i in range(4)] for q in range(8)])
                    for h in ("R2", "R16")}

    def aggregate(self, periodic, control, task="categorical", **kwargs):
        return paired_behavior_effects(periodic, control, task=task,
            input_units="error_indicator" if task == "categorical" else "raw_squared_error",
            periodic_pair_ids=self.ids, control_pair_ids=kwargs.pop("control_pair_ids", self.ids),
            episodes_per_coordinate=4, **kwargs)

    def test_binary_loss_pairing_and_DID_do_not_count_coordinates_as_seeds(self):
        periodic = {"R2": np.tile([1, 0, 0, 0], (8, 1)), "R16": np.tile([1, 1, 0, 0], (8, 1))}
        control = {h: np.tile([0, 1, 0, 0], (8, 1)) for h in ("R2", "R16")}
        result = self.aggregate(periodic, control)
        self.assertEqual(result["effects"], {"delta2": 0, "delta16": 0.25, "DID": 0.25})
        self.assertEqual(result["episode_pairs_per_horizon"], 32)
        self.assertEqual(result["training_seed_replications"], 1)
        self.assertFalse(result["matches_proposed_episode_count"])
        self.assertFalse(result["cross_horizon_episode_pairing_assumed"])

    def test_continuous_fixed_population_variance_and_equal_coordinate_weights(self):
        periodic = {"R2": np.full((8, 4), 0.003), "R16": np.tile([.01, .02, .03, .04], (8, 1))}
        control = {"R2": np.full((8, 4), 0.001), "R16": np.full((8, 4), .02)}
        result = self.aggregate(periodic, control, "continuous")
        self.assertAlmostEqual(result["arm_mean_losses"]["R16"]["periodic"], float(Fraction(1, 40)))
        self.assertAlmostEqual(result["effects"]["delta16"], float(Fraction(3, 200)))
        self.assertAlmostEqual(result["effects"]["delta2"], float(Fraction(3, 500)))
        self.assertAlmostEqual(result["effects"]["DID"], float(Fraction(9, 1000)))
        self.assertEqual(result["effect_units"], "population_normalized_MSE")
        self.assertEqual(result["continuous_population_variance"], 1 / 3)
        record = {"model": "gru", "task": "continuous", "trained_seed": 3,
                  "validation_eligible": True, "outcome": result}
        family = reduce_seed_family([record], declared_seeds=[3, 4])
        effects = family["cells"][1]["seed_ledger"][0]["effects"]
        for name in effects:
            self.assertAlmostEqual(effects[name], result["effects"][name])

    def test_misaligned_duplicate_or_missing_pairs_rejected_before_aggregation(self):
        losses = {h: np.zeros((8, 4)) for h in ("R2", "R16")}
        for mutation in ("permute", "duplicate", "missing"):
            ids = deepcopy(self.ids)
            if mutation == "permute":
                ids["R16"] = ids["R16"][:, ::-1]
            elif mutation == "duplicate":
                ids["R2"][0, 1] = ids["R2"][0, 0]
            else:
                ids["R2"] = ids["R2"][:, :3]
            with self.assertRaises(ValueError):
                self.aggregate(losses, losses, control_pair_ids=ids)

    def test_small_paired_continuous_contrast_survives_large_arm_means(self):
        control = {h: np.full((8, 4), 1e16) for h in ("R2", "R16")}
        periodic = deepcopy(control)
        periodic["R16"][:, :2] += 2
        result = self.aggregate(periodic, control, "continuous")
        # Each R16 pair difference is [2,2,0,0], whose exact mean is one.
        # Rounding of each separately aggregated arm mean erases that one.
        means = result["arm_mean_losses"]["R16"]
        self.assertEqual(means["periodic"] - means["matched_control"], 0)
        self.assertEqual(result["paired_mean_raw_loss_differences"]["R16"], 1)
        self.assertEqual(result["effects"]["delta16"], 3)
        record = {"model": "gru", "task": "continuous", "trained_seed": 3,
                  "validation_eligible": True, "outcome": result}
        family = reduce_seed_family([record], declared_seeds=[3, 4])
        self.assertEqual(family["cells"][1]["seed_ledger"][0]["effects"]["delta16"], 3)

    def test_nonbinary_categorical_nonfinite_negative_or_wrong_units_rejected(self):
        base = {h: np.zeros((8, 4)) for h in ("R2", "R16")}
        for invalid in (.2, math.nan, math.inf, -1):
            changed = deepcopy(base)
            changed["R2"][0, 0] = invalid
            with self.assertRaises(ValueError):
                self.aggregate(changed, base)
        with self.assertRaises(ValueError):
            paired_behavior_effects(base, base, task="continuous", input_units="population_normalized_MSE",
                periodic_pair_ids=self.ids, control_pair_ids=self.ids, episodes_per_coordinate=4)
        with self.assertRaises(ValueError):
            self.aggregate({h: np.zeros((7, 4)) for h in base}, base)


class SeedFamilyTests(unittest.TestCase):
    def test_independent_finite_fixture_mean_SD_test_and_intervals(self):
        summary = seed_t_summary([i / 100 for i in range(1, 11)])
        expected_mean = float(Fraction(11, 200))
        expected_sd = math.sqrt(float(Fraction(11, 12000)))
        expected_se = math.sqrt(float(Fraction(11, 120000)))
        self.assertAlmostEqual(summary["mean"], expected_mean)
        self.assertAlmostEqual(summary["sample_sd"], expected_sd)
        self.assertAlmostEqual(summary["standard_error"], expected_se)
        expected_t = (expected_mean - .02) / expected_se
        self.assertAlmostEqual(summary["raw_approximate_p"], density_integral_upper_tail(expected_t, 9), delta=5e-14)
        self.assertAlmostEqual(summary["bonferroni_99_375_t_interval"]["lower"],
                               expected_mean - 1.1215044595072001 * expected_sd, delta=1e-13)
        self.assertFalse(summary["finite_sample_exactness_claimed"])
        self.assertFalse(summary["calibration_assumptions_verified"])

    def test_DID_is_computed_within_seed_and_keeps_cross_horizon_covariance(self):
        values = [i / 32 for i in range(1, 11)]
        records = records_for_cell("gru", "categorical", values,
                                   delta2=[value - 1 / 32 for value in values])
        result = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
        delta, did = result["slots"][:2]
        self.assertGreater(delta["t_summary"]["sample_sd"], 0)
        self.assertEqual(did["t_summary"]["mean"], 1 / 32)
        self.assertEqual(did["t_summary"]["sample_sd"], 0)
        self.assertIsNone(did["t_summary"]["raw_approximate_p"])
        self.assertEqual(did["decision_p"], 1)
        self.assertFalse(did["approximate_holm_rejects"])

    def test_missing_other_cells_do_not_reduce_eight_slot_multiplicity(self):
        # This fixture's p lies between .05/8 and .05/2. Dropping six absent
        # slots would yield a harmful false reduction of multiplicity.
        records = records_for_cell("gru", "categorical", [.004 + i / 100 for i in range(10)])
        result = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
        first = result["slots"][0]
        p = first["decision_p"]
        self.assertGreater(p, .05 / 8)
        self.assertLess(p, .05 / 2)
        self.assertAlmostEqual(first["approximate_holm_adjusted_p"], 8 * p)
        self.assertFalse(any(slot["approximate_holm_rejects"] for slot in result["slots"]))
        self.assertEqual(result["family_order"], list(FAMILY_ORDER))
        self.assertEqual(result["family_size"], 8)
        for slot in result["slots"][2:]:
            self.assertEqual(slot["decision_p"], 1)
            self.assertIsNone(slot["t_summary"])

    def test_missing_ineligible_or_failed_seed_blocks_cell_without_available_case_mean(self):
        original = records_for_cell("lstm", "categorical", [.12 + i / 100 for i in range(10)])
        for failure in ("missing", "ineligible", "numerical_failure", "missing_outcome"):
            records = deepcopy(original)
            if failure == "missing":
                records.pop()
            elif failure == "ineligible":
                records[-1].update(validation_eligible=False, unavailable_reason="failed_coordinate_7", outcome=None)
            elif failure == "numerical_failure":
                records[-1].update(unavailable_reason="nonfinite_prediction", outcome=None)
            else:
                records[-1].update(outcome=None)
            result = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
            cell = result["cells"][2]
            self.assertEqual(len(cell["seed_ledger"]), 10)
            self.assertEqual(cell["available_training_seeds"], 9)
            self.assertFalse(cell["primary_available_case_mean_reported"])
            for slot in result["slots"][4:6]:
                self.assertEqual(slot["decision_p"], 1)
                self.assertIsNone(slot["t_summary"])
                self.assertEqual(slot["scientific_decision"], "inconclusive")
            json.dumps(result, allow_nan=False)

    def test_inconclusive_cell_does_not_erase_another_complete_cell(self):
        values = [.12 + i / 100 for i in range(10)]
        records = records_for_cell("gru", "categorical", values)
        records += records_for_cell("lstm", "categorical", values)[:-1]
        result = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
        self.assertTrue(result["slots"][0]["approximate_holm_rejects"])
        self.assertEqual(result["slots"][0]["holm_threshold"], .05 / 8)
        self.assertEqual(result["slots"][4]["decision_p"], 1)
        self.assertFalse(result["accepted_harm_claim"])
        self.assertFalse(result["bootstrap_interval_implemented"])

    def test_ties_follow_fixed_family_order_and_outputs_never_accept_science(self):
        values = [.12 + i / 100 for i in range(10)]
        records = records_for_cell("lstm", "categorical", values)
        records += records_for_cell("gru", "categorical", values)
        result = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
        self.assertEqual([result["slots"][i]["holm_rank"] for i in (0, 1, 4, 5)], [1, 2, 3, 4])
        self.assertFalse(result["protocol_frozen"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse(result["equivalence_enabled"])
        self.assertFalse(result["sign_flip_performed"])
        self.assertFalse(any(slot["accepted_harm_claim"] for slot in result["slots"]))
        json.dumps(result, allow_nan=False)

    def test_unlisted_replacement_duplicate_or_bool_seed_identity_rejected(self):
        records = records_for_cell("gru", "categorical", [.1, .2], seeds=[3, 4])
        for bad in (records + [deepcopy(records[0])],
                    records + records_for_cell("gru", "categorical", [.15], seeds=[5]),
                    [dict(records[0], trained_seed=True)]):
            with self.assertRaises(ValueError):
                reduce_seed_family(bad, declared_seeds=[3, 4])
        for declared in ([3, 3], [True, 4], [3], [3, -4]):
            with self.assertRaises(ValueError):
                reduce_seed_family(records, declared_seeds=declared)

    def test_malformed_units_supplied_effects_or_eligibility_not_silently_accepted(self):
        base = records_for_cell("gru", "continuous", [.1, .2], seeds=[3, 4])
        mutations = [lambda r: r.update(validation_eligible="true"),
                     lambda r: r["outcome"].update(loss_units="population_normalized_MSE"),
                     lambda r: r["outcome"]["arm_mean_losses"]["R2"].update(periodic=math.nan),
                     lambda r: r["outcome"].update(effects={"delta2": 0, "delta16": 100, "DID": 100}),
                     lambda r: r.update(validation_eligible=False)]
        for mutate in mutations:
            changed = deepcopy(base)
            mutate(changed[0])
            with self.assertRaises(ValueError):
                reduce_seed_family(changed, declared_seeds=[3, 4])

    def test_available_failed_seed_outcomes_are_preserved_without_primary_rescue(self):
        records = records_for_cell("gru", "categorical", [.12 + i / 100 for i in range(10)])
        records[-1].update(validation_eligible=False, unavailable_reason="failed_coordinate_7")
        family = reduce_seed_family(records, declared_seeds=list(range(3, 13)))
        failed = family["cells"][0]["seed_ledger"][-1]
        self.assertAlmostEqual(failed["observed_outcome_effects"]["delta16"], .21)
        self.assertIsNone(failed["effects"])
        self.assertFalse(failed["available"])
        self.assertEqual(family["slots"][0]["decision_p"], 1)
        self.assertIsNone(family["slots"][0]["t_summary"])

    def test_zero_SD_and_numerical_underflow_never_produce_silent_zero_p(self):
        summary = seed_t_summary([.125] * 10)
        self.assertEqual(summary["mean"], .125)
        self.assertEqual(summary["sample_sd"], 0)
        self.assertIsNone(summary["raw_approximate_p"])
        self.assertIsNone(summary["ordinary_95_t_interval"])
        underflow = seed_t_summary([.1 + i * 1e-16 for i in range(30)])
        self.assertFalse(underflow["numerically_available"])
        self.assertIsNone(underflow["raw_approximate_p"])
        self.assertEqual(underflow["unavailable_reason"], "t_upper_tail_underflow")


if __name__ == "__main__":
    unittest.main()
