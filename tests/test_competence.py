"""Gate-contract and explicit-reference tests; no learned-model evidence."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

import numpy as np

from representation_collapse.competence import (
    IMMEDIATE, R0, R1, R2, NATIVE_PHASE, PROTOCOL, FrozenCompetenceProtocol,
    assess_stage_competence, evaluate_competence, make_competence_batch,
    validate_competence_batch,
)
from representation_collapse.diagnostics import streaming_register_oracle
from representation_collapse.vector_task import QUERY, WRITE


class CompetenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batches = {(assay, task): make_competence_batch(assay, task, 123)
                       for assay in (IMMEDIATE, R0, R1, R2)
                       for task in ("categorical", "continuous")}

    def score(self, batch, predictions=None, **kwargs):
        if predictions is None:
            predictions = streaming_register_oracle(batch.inputs, batch.task)
        return evaluate_competence(batch, predictions, trained_seed=0,
                                   checkpoint_id="reference-fixture-not-trained",
                                   measurement_phase=NATIVE_PHASE, **kwargs)

    def zero_targets(self, assay=R2):
        batch = self.batches[assay, "continuous"]
        x = batch.inputs.copy()
        x[..., 37][x[..., WRITE] == 1] = 0
        return replace(batch, inputs=x, targets=np.zeros(len(x), dtype=np.float64))

    def subset_pairs(self, batch, indexes):
        n = len(batch.targets) // 2
        selection = np.concatenate((indexes, indexes + n))
        return replace(batch, inputs=batch.inputs[selection], targets=batch.targets[selection],
                       queries=batch.queries[selection], pair_ids=np.tile(np.arange(len(indexes)), 2))

    def test_oracle_passes_every_stage_with_exact_strata_and_seed_units(self):
        for (assay, task), batch in self.batches.items():
            with self.subTest(assay=assay, task=task):
                result = self.score(batch, split="fixture")
                self.assertTrue(result["coordinate_gate_passed"])
                self.assertFalse(result["validation_eligibility_met"])
                self.assertFalse(result["scientific_acceptance"])
                self.assertEqual(result["trained_seed_units"], 0)
                self.assertTrue(result["fixture_only"])
                self.assertEqual(result["unique_episode_units"], 1024)
                self.assertEqual(result["paired_episode_units"], 0 if assay == IMMEDIATE else 1024)
                self.assertEqual(result["sampling_seed"], 123)
                self.assertEqual(result["measurement_phase"], NATIVE_PHASE)
                for arm in result["arms"].values():
                    self.assertEqual([r["episodes"] for r in arm["coordinates"]], [128] * 8)
                    self.assertEqual(arm["failed_coordinates"], [])
                json.dumps(result, allow_nan=False)

    def test_manifest_is_deterministic_and_immutable(self):
        batch = self.batches[R2, "categorical"]
        repeated = make_competence_batch(R2, "categorical", np.int64(123))
        first = self.score(batch)
        self.assertEqual(first["batch_sha256"], self.score(repeated)["batch_sha256"])
        self.assertNotEqual(first["batch_sha256"], self.score(make_competence_batch(R2, "categorical", 124))["batch_sha256"])
        with self.assertRaises(ValueError):
            batch.inputs[0, 0, 0] = 0
        with self.assertRaises(FrozenInstanceError):
            PROTOCOL.categorical_min_accuracy = 0.25
        with self.assertRaises(TypeError):
            FrozenCompetenceProtocol(categorical_min_accuracy=0.25)

    def test_immediate_and_r0_have_distinct_chronology_and_requirements(self):
        immediate = self.batches[IMMEDIATE, "categorical"]
        r0 = self.batches[R0, "categorical"]
        self.assertEqual(immediate.inputs.shape, (1024, 2, 38))
        self.assertEqual(r0.inputs.shape, (2048, 17, 38))
        result = self.score(immediate)
        self.assertIsNone(result["paired_coordinate_error_differences"])
        self.assertEqual(set(result["arms"]), {"immediate"})
        report = assess_stage_competence([result], 2)
        self.assertFalse(report["coordinate_requirement_met"])
        self.assertEqual(report["missing_assays"], [R0])
        self.assertIn("later required/nuisance WRITE", self.score(r0)["interpretation"])
        self.assertIn("fixed-batch overfit and untouched IID generalization",
                      assess_stage_competence([result], 1)["remaining_training_requirements"])
        for batch, assay in ((immediate, R0), (r0, IMMEDIATE), (r0, R1)):
            with self.assertRaises(ValueError):
                validate_competence_batch(replace(batch, assay=assay))

    def test_r0_and_r1_anchor_arms_share_exact_inputs(self):
        for assay in (R0, R1):
            batch = self.batches[assay, "continuous"]
            n = len(batch.targets) // 2
            np.testing.assert_array_equal(batch.inputs[:n], batch.inputs[n:])
            result = self.score(batch)
            self.assertEqual(result["paired_coordinate_error_differences"], [0.0] * 8)

    def test_categorical_aggregate_pass_cannot_hide_failed_coordinate(self):
        batch = self.batches[R2, "categorical"]
        prediction = batch.targets.copy()
        indices = np.flatnonzero(batch.queries[:1024] == 7)[:7]
        prediction[indices] = (prediction[indices] + 1) % 16
        result = self.score(batch, prediction)
        self.assertGreater(result["arms"]["periodic"]["aggregate_metric"], 0.95)
        self.assertFalse(result["coordinate_gate_passed"])
        self.assertFalse(result["validation_eligibility_met"])
        self.assertEqual(result["arms"]["periodic"]["failed_coordinates"], [7])
        self.assertTrue(result["arms"]["control"]["passed"])
        self.assertEqual(result["arms"]["periodic"]["coordinates"][7]["failure_reasons"], ["below_accuracy_threshold"])
        self.assertAlmostEqual(result["paired_coordinate_error_differences"][7], 7 / 128)

    def test_categorical_inclusive_gate_128_sample_boundary(self):
        batch = self.batches[IMMEDIATE, "categorical"]
        indices = np.flatnonzero(batch.queries == 0)
        for errors, passed in ((6, True), (7, False)):
            prediction = batch.targets.copy()
            prediction[indices[:errors]] = (prediction[indices[:errors]] + 1) % 16
            self.assertEqual(self.score(batch, prediction)["coordinate_gate_passed"], passed)

    def test_continuous_aggregate_pass_cannot_hide_failed_coordinate(self):
        batch = self.zero_targets()
        prediction = np.zeros(len(batch.targets))
        prediction[:1024][batch.queries[:1024] == 0] = np.sqrt(0.08 / 3)
        result = self.score(batch, prediction)
        self.assertLess(result["arms"]["periodic"]["aggregate_metric"], 0.05)
        self.assertFalse(result["coordinate_gate_passed"])
        self.assertEqual(result["arms"]["periodic"]["failed_coordinates"], [0])
        self.assertTrue(result["arms"]["control"]["passed"])
        self.assertAlmostEqual(result["arms"]["periodic"]["coordinates"][0]["metric"], 0.08)
        self.assertAlmostEqual(result["paired_coordinate_error_differences"][0], 0.08 / 3)
        self.assertIn("raw squared error; not normalized MSE", result["paired_difference_units"])

    def test_normalization_uses_population_variance_even_for_constant_targets(self):
        batch = self.zero_targets(R0)
        result = self.score(batch, np.full(len(batch.targets), 0.1))
        self.assertTrue(result["coordinate_gate_passed"])
        row = result["arms"]["periodic"]["coordinates"][0]
        self.assertAlmostEqual(row["raw_mse"], 0.01)
        self.assertAlmostEqual(row["metric"], 0.03)
        self.assertEqual(row["sample_zero_predictor_mse"], 0)
        self.assertEqual(result["protocol"]["population_value_variance"], 1 / 3)

    def test_continuous_inclusive_boundary_and_fixed_rounding_allowance(self):
        batch = self.zero_targets(R0)
        prediction = np.full(len(batch.targets), np.sqrt(1 / 60))
        boundary = self.score(batch, prediction)
        self.assertTrue(boundary["coordinate_gate_passed"])
        self.assertEqual(boundary["comparison_allowance"], 4 * np.spacing(0.05))
        self.assertFalse(self.score(batch, prediction * (1 + 1e-10))["coordinate_gate_passed"])

    def test_baseline_predictions_fail_and_have_expected_population_scale(self):
        categorical = self.batches[R0, "categorical"]
        self.assertFalse(self.score(categorical, np.zeros(len(categorical.targets), dtype=int))["coordinate_gate_passed"])
        continuous = self.batches[R0, "continuous"]
        result = self.score(continuous, np.zeros(len(continuous.targets)))
        self.assertFalse(result["coordinate_gate_passed"])
        for row in result["arms"]["periodic"]["coordinates"]:
            self.assertEqual(row["raw_mse"], row["sample_zero_predictor_mse"])
            self.assertAlmostEqual(row["metric"], row["raw_mse"] * 3)

    def test_missing_and_undersampled_coordinates_fail_without_nan(self):
        original = self.batches[R0, "categorical"]
        for indexes in (np.flatnonzero(original.queries[:1024] != 7), np.arange(1023),
                        np.arange(0, 1024, 128), np.array([0])):
            batch = self.subset_pairs(original, indexes)
            result = self.score(batch)
            self.assertFalse(result["coordinate_gate_passed"])
            row = result["arms"]["periodic"]["coordinates"][7]
            self.assertIn("sample_count_mismatch", row["failure_reasons"])
            if not row["episodes"]:
                self.assertIsNone(row["metric"])
                self.assertIn("missing_coordinate", row["failure_reasons"])
                self.assertIsNone(result["paired_coordinate_error_differences"][7])
            if len(indexes) <= 8:
                self.assertEqual(result["arms"]["periodic"]["failed_coordinates"], list(range(8)))
            json.dumps(result, allow_nan=False)
        # Empty episode collections violate the renderer's existing nonempty
        # contract; they cannot reach an aggregate mean of an empty array.
        empty = self.subset_pairs(original, np.array([], dtype=int))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.score(empty)

    def test_strict_task_phase_seed_and_prediction_validation(self):
        batch = self.batches[R0, "categorical"]
        for prediction in (batch.targets.astype(float), np.zeros((len(batch.targets), 16)), np.full(len(batch.targets), 16), np.ones(len(batch.targets), dtype=bool), batch.targets[:-1]):
            with self.assertRaises(ValueError):
                self.score(batch, prediction)
        for phase in ("pre_query", "post_write", "after_query", None):
            with self.assertRaises(ValueError):
                evaluate_competence(batch, batch.targets, trained_seed=0, checkpoint_id="id", measurement_phase=phase)
        for seed in (True, -1, 0.0, "0"):
            with self.assertRaises(ValueError):
                make_competence_batch(R0, "categorical", seed)
            with self.assertRaises(ValueError):
                evaluate_competence(batch, batch.targets, trained_seed=seed, checkpoint_id="id", measurement_phase=NATIVE_PHASE)
        for task in ("wrong", None):
            with self.assertRaises(ValueError):
                make_competence_batch(R0, task, 1)
        continuous = self.batches[R0, "continuous"]
        for value in (np.nan, np.inf, 1e308, 1j):
            with self.assertRaises(ValueError):
                self.score(continuous, np.full(len(continuous.targets), value))
        with self.assertRaises(ValueError):
            self.score(batch, split="training")

    def test_finite_squared_errors_with_reduction_or_normalization_overflow_rejected(self):
        batch = self.zero_targets(R0)
        # Coordinate sums overflow at 1e154; at 1e153 the coordinate scores
        # remain finite but the 1,024-episode aggregate reduction overflows.
        for value in (1e154, 1e153):
            prediction = np.full(len(batch.targets), value)
            self.assertTrue(np.isfinite(np.square(prediction)).all())
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "overflowed or is nonfinite"):
                self.score(batch, prediction)
        # With one episode per coordinate, mean MSE remains finite but dividing
        # 6.4e307 by the population variance overflows. Counts cannot hide this.
        small = self.subset_pairs(batch, np.arange(0, 1024, 128))
        prediction = np.full(len(small.targets), 8e153)
        self.assertTrue(np.isfinite(np.square(prediction)).all())
        with self.assertRaisesRegex(ValueError, "coordinate metric overflowed"):
            self.score(small, prediction)
        # Large finite errors with finite reductions remain reportable failures.
        result = self.score(batch, np.full(len(batch.targets), 1e150))
        self.assertFalse(result["coordinate_gate_passed"])
        json.dumps(result, allow_nan=False)

    def test_malformed_chronology_target_pairing_and_motif_control_rejected(self):
        batch = self.batches[R2, "categorical"]
        cases = []
        x = batch.inputs.copy()
        x[:, -1, 21] = 1
        cases.append(replace(batch, inputs=x))  # leaked answer
        x = batch.inputs.copy()
        x[:, [0, 1]] = x[:, [1, 0]]
        cases.append(replace(batch, inputs=x))  # register oracle works, chronology is wrong
        x = batch.inputs.copy()
        x[1024:, 16:-1] = x[:1024, 16:-1]
        cases.append(replace(batch, inputs=x))  # control is also periodic
        x = batch.inputs.copy()
        x[1024, 0, 21:37] = np.roll(x[1024, 0, 21:37], 1)
        cases.append(replace(batch, inputs=x, targets=streaming_register_oracle(x, "categorical")))
        cases.append(replace(batch, targets=(batch.targets + 1) % 16))
        cases.append(replace(batch, queries=np.roll(batch.queries, 1)))
        cases.append(replace(batch, pair_ids=np.zeros(len(batch.targets), dtype=int)))
        cases.append(replace(batch, task="continuous"))
        for corrupted in cases:
            with self.assertRaises(ValueError):
                validate_competence_batch(corrupted)
        immediate = self.batches[IMMEDIATE, "categorical"]
        x = immediate.inputs.copy()
        x[:, 0, :5] = 0
        x[:, 0, QUERY] = 1
        with self.assertRaises(ValueError):
            validate_competence_batch(replace(immediate, inputs=x))

    def test_stage_panel_requires_same_checkpoint_task_seed_and_validation(self):
        records = [self.score(self.batches[assay, "continuous"]) for assay in (R0, R1, R2)]
        for stage in (2, 3, 4, 5):
            self.assertTrue(assess_stage_competence(records, stage)["coordinate_requirement_met"])
        self.assertEqual(assess_stage_competence(records[:2], 4)["missing_assays"], [R2])
        for key, value in (("trained_seed", 1), ("task", "categorical"), ("checkpoint_id", "different"), ("split", "test"), ("measurement_phase", "pre_query")):
            corrupt = deepcopy(records)
            corrupt[-1][key] = value
            with self.assertRaises(ValueError):
                assess_stage_competence(corrupt, 4)
        with self.assertRaises(ValueError):
            assess_stage_competence(records + [records[0]], 4)
        self.assertFalse(assess_stage_competence([], 4)["coordinate_requirement_met"])
        test = self.score(self.batches[R0, "continuous"], split="test")
        self.assertTrue(test["coordinate_gate_passed"])
        self.assertFalse(test["validation_eligibility_met"])

    def test_stage_validator_rechecks_coordinates_instead_of_trusting_flags(self):
        batch = self.batches[R0, "categorical"]
        prediction = batch.targets.copy()
        prediction[batch.queries == 0] = (prediction[batch.queries == 0] + 1) % 16
        failed = self.score(batch, prediction)
        self.assertEqual(assess_stage_competence([failed], 2)["failed_assays"], [R0])
        corrupt = deepcopy(failed)
        corrupt["coordinate_gate_passed"] = True
        with self.assertRaises(ValueError):
            assess_stage_competence([corrupt], 2)
        good = self.score(batch)
        for mutate in (
            lambda r: r["arms"]["periodic"]["coordinates"].pop(),
            lambda r: r["arms"]["periodic"]["coordinates"][0].update(metric=0.2),
            lambda r: r["protocol"].update(categorical_min_accuracy=0.25),
            lambda r: r["arms"]["periodic"]["coordinates"][0].update(episodes=256),
            lambda r: r["arms"]["periodic"].update(passed=False),
        ):
            corrupt = deepcopy(good)
            mutate(corrupt)
            with self.assertRaises(ValueError):
                assess_stage_competence([corrupt], 2)


if __name__ == "__main__":
    unittest.main()
