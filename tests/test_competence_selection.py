"""Selection-contract fixtures; manufactured scores are not learned evidence."""

from copy import deepcopy
import hashlib
import json
import unittest

from representation_collapse.competence import R0, R1, R2, NATIVE_PHASE, evaluate_competence, make_competence_batch
from representation_collapse.competence_selection import HORIZONS, select_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {}
        cls.validation_manifests = {}
        for task in ("categorical", "continuous"):
            cls.records[task] = []
            for assay in (R0, R1, R2):
                batch = make_competence_batch(assay, task, 173)
                # Exact synthetic predictions exercise the validation schema.
                # No fit, trained competence or scientific acceptance follows.
                record = evaluate_competence(batch, batch.targets.copy(), trained_seed=0,
                    checkpoint_id="test-fixture", measurement_phase=NATIVE_PHASE, split="validation")
                record["model"] = "gru"
                cls.records[task].append(record)
            cls.validation_manifests[task] = {
                horizon: {"batch_sha256": record["batch_sha256"], "sampling_seed": record["sampling_seed"]}
                for horizon, record in zip(("R0", "R1", "R2"), cls.records[task])}
            # Long-horizon synthetic binding identities exercise selection's
            # contract only; these tests report no actual task-loss artifacts.
            for index, horizon in enumerate(("R4", "R8", "R16")):
                cls.validation_manifests[task][horizon] = {
                    "batch_sha256": hashlib.sha256((task + "-binding-fixture-" + horizon).encode()).hexdigest(),
                    "sampling_seed": 180 + index}

    def candidate(self, checkpoint_id="a", *, task="categorical", loss=1.0, step=2100, lr=0.001):
        records = deepcopy(self.records[task])
        for record in records:
            record["checkpoint_id"] = checkpoint_id
        return {"model": "gru", "task": task, "trained_seed": 0, "learning_rate": lr,
                "step": step, "stage": 5, "checkpoint_id": checkpoint_id,
                "split": "validation", "native_validation": True,
                "short_horizon_records": records,
                "validation_manifest": deepcopy(self.validation_manifests[task]),
                "validation_loss_by_horizon": {h: {"periodic": loss, "control": loss} for h in HORIZONS}}

    def select(self, candidates):
        return select_checkpoint(candidates, [0.0005, 0.001, 0.003],
                                 validation_manifest=self.validation_manifests[candidates[0]["task"]])

    def test_balanced_task_loss_is_only_objective_for_both_tasks(self):
        for task in ("categorical", "continuous"):
            a, b = self.candidate("a", task=task, loss=0.5), self.candidate("b", task=task, loss=0.6)
            a["validation_loss_by_horizon"]["R16"]["control"] = 2.5
            a["validation_task_loss"] = 0  # Supplied aggregation is diagnostic.
            result = self.select([a, b])
            self.assertEqual(result["selected_checkpoint_id"], "b")
            self.assertAlmostEqual(result["candidate_reviews"][0]["balanced_validation_loss"], 0.7)
            self.assertFalse(result["scientific_acceptance"])
            self.assertFalse(result["native_execution_independently_verified"])
            self.assertEqual(result["loss_units"], "cross_entropy" if task == "categorical" else "raw_squared_error")
            json.dumps(result, allow_nan=False)

    def test_exact_ties_prefer_earlier_step_then_frozen_lr_order(self):
        candidates = [self.candidate("later", step=2500, lr=0.0005),
                      self.candidate("earlier-third-lr", step=2200, lr=0.003),
                      self.candidate("earlier-first-lr", step=2200, lr=0.0005)]
        self.assertEqual(self.select(candidates)["selected_index"], 2)
        self.assertEqual(select_checkpoint(candidates, [0.003, 0.001, 0.0005],
                         validation_manifest=self.validation_manifests["categorical"])["selected_index"], 1)
        candidates[0]["validation_loss_by_horizon"]["R1"]["periodic"] = 0.999999999
        self.assertEqual(self.select(candidates)["selected_index"], 0)

    def test_failed_coordinate_in_either_arm_blocks_even_lowest_loss(self):
        a, b = self.candidate("bad", loss=0), self.candidate("good", loss=1)
        record = a["short_horizon_records"][2]
        arm = record["arms"]["control"]
        row = arm["coordinates"][7]
        row.update(metric=0.5, metric_passed=False, passed=False, failure_reasons=["below_accuracy_threshold"])
        arm.update(passed=False, failed_coordinates=[7])
        record.update(coordinate_gate_passed=False, validation_eligibility_met=False)
        result = self.select([a, b])
        self.assertEqual(result["selected_checkpoint_id"], "good")
        self.assertIn("short_horizon_gate_failed:full_prefix_r2", result["candidate_reviews"][0]["ineligibility_reasons"])

    def test_held_out_fixture_phase_and_other_checkpoint_cannot_qualify(self):
        mutations = [lambda c: c.update(split="test"), lambda c: c.update(native_validation=False),
                     lambda c: c.update(stage=4), lambda c: c.update(step=True),
                     lambda c: c["short_horizon_records"][0].update(split="test"),
                     lambda c: c["short_horizon_records"][0].update(split="fixture"),
                     lambda c: c["short_horizon_records"][0].update(measurement_phase="before_query"),
                     lambda c: c["short_horizon_records"][0].update(checkpoint_id="other"),
                     lambda c: c["short_horizon_records"][0].update(model="lstm"),
                     lambda c: c["short_horizon_records"][0].update(trained_seed=1),
                     lambda c: c["short_horizon_records"][0].update(fixture_only=True),
                     lambda c: c["short_horizon_records"][0].update(validation_eligibility_met=False)]
        for mutate in mutations:
            candidate = self.candidate()
            mutate(candidate)
            result = self.select([candidate])
            self.assertIsNone(result["selected"])
            self.assertTrue(result["candidate_reviews"][0]["ineligibility_reasons"])
            self.assertEqual(result["status"], "acquisition_or_optimization_unresolved")

    def test_missing_assay_and_forged_pass_flags_cannot_qualify(self):
        candidate = self.candidate()
        candidate["short_horizon_records"].pop()
        self.assertIsNone(self.select([candidate])["selected"])
        candidate = self.candidate()
        candidate["short_horizon_records"][1]["arms"]["periodic"]["coordinates"][0]["metric"] = 0.2
        self.assertIsNone(self.select([candidate])["selected"])

    def test_continuous_coordinate_failure_and_long_horizon_non_gate(self):
        candidate = self.candidate(task="continuous", loss=1e3)
        # Long-horizon failure does not itself disqualify a short-competent
        # checkpoint; selection is not conditioned on a desired R16 outcome.
        self.assertEqual(self.select([candidate])["selected_checkpoint_id"], "a")
        record = candidate["short_horizon_records"][0]
        arm = record["arms"]["periodic"]
        arm["coordinates"][0].update(metric=0.051, passed=False, metric_passed=False)
        arm.update(passed=False, failed_coordinates=[0])
        record.update(coordinate_gate_passed=False, validation_eligibility_met=False)
        self.assertIsNone(self.select([candidate])["selected"])

    def test_geometry_arm_contrasts_and_held_out_score_fields_are_rejected(self):
        for name in ("geometry", "DID", "delta16", "test_loss", "selection_metric"):
            candidate = self.candidate()
            candidate[name] = 0
            reasons = self.select([candidate])["candidate_reviews"][0]["ineligibility_reasons"]
            self.assertIn("unexpected_selection_fields:" + name, reasons)

    def test_missing_unbalanced_nonfinite_or_negative_losses_rejected(self):
        for bad in (None, {}, {h: {"periodic": 0} for h in HORIZONS}):
            candidate = self.candidate()
            candidate["validation_loss_by_horizon"] = bad
            self.assertIsNone(self.select([candidate])["selected"])
        for bad in (float("nan"), float("inf"), -1.0, True, "0.5"):
            candidate = self.candidate()
            candidate["validation_loss_by_horizon"]["R8"]["periodic"] = bad
            self.assertIsNone(self.select([candidate])["selected"])
        candidate = self.candidate(loss=1e308)
        self.assertEqual(self.select([candidate])["selected"]["balanced_validation_loss"], 1e308)

    def test_mixed_cell_unknown_rate_duplicate_id_and_empty_grid_rejected(self):
        for field, value in (("model", "lstm"), ("task", "continuous"), ("trained_seed", 1), ("learning_rate", 0.1)):
            a, b = self.candidate("a"), self.candidate("b")
            b[field] = value
            with self.assertRaises(ValueError):
                self.select([a, b])
        for candidates, grid in (([], [0.001]), ([self.candidate(), self.candidate()], [0.001]),
                                 ([self.candidate()], []), ([self.candidate()], [0.001, 0.001]),
                                 ([self.candidate()], [True]), ([self.candidate()], [float("nan")])):
            with self.assertRaises(ValueError):
                select_checkpoint(candidates, grid, validation_manifest=self.validation_manifests["categorical"])

    def test_changed_short_batches_cannot_win_lower_loss_selection(self):
        for task in ("categorical", "continuous"):
            a, b = self.candidate("fixed", task=task, loss=1), self.candidate("changed", task=task, loss=0)
            for index, record in enumerate(b["short_horizon_records"]):
                record["batch_sha256"] = hashlib.sha256(f"replacement-{index}".encode()).hexdigest()
                record["sampling_seed"] += 100
                b["validation_manifest"][f"R{(0, 1, 2)[index]}"] = {
                    "batch_sha256": record["batch_sha256"], "sampling_seed": record["sampling_seed"]}
            result = self.select([a, b])
            self.assertEqual(result["selected_checkpoint_id"], "fixed")
            reasons = result["candidate_reviews"][1]["ineligibility_reasons"]
            self.assertTrue(any("validation_manifest_mismatch:R0" in reason for reason in reasons))
            self.assertTrue(any("record/frozen validation batch hash mismatch:R0" in reason for reason in reasons))

    def test_each_horizon_manifest_binding_is_fixed_and_missing_bindings_fail(self):
        for horizon in ("R0", "R1", "R2", "R4", "R8", "R16"):
            for name, replacement in (("batch_sha256", "a" * 64), ("sampling_seed", 9891)):
                candidate = self.candidate()
                candidate["validation_manifest"][horizon][name] = replacement
                result = self.select([candidate])
                self.assertIsNone(result["selected"])
                self.assertIn(f"validation_manifest_mismatch:{horizon}:{name}", result["candidate_reviews"][0]["ineligibility_reasons"])
        candidate = self.candidate()
        del candidate["validation_manifest"]
        self.assertIsNone(self.select([candidate])["selected"])
        candidate = self.candidate()
        candidate["short_horizon_records"][1]["sampling_seed"] += 1
        self.assertIsNone(self.select([candidate])["selected"])

    def test_caller_freeze_is_required_canonical_and_never_inferred(self):
        candidate = self.candidate()
        with self.assertRaises(ValueError):
            select_checkpoint([candidate], [0.001])
        for bad in (None, {}, {"R0": candidate["validation_manifest"]["R0"]}):
            with self.assertRaises(ValueError):
                select_checkpoint([candidate], [0.001], validation_manifest=bad)
        for name, replacement in (("batch_sha256", "A" * 64), ("batch_sha256", "a" * 63),
                                  ("batch_sha256", "g" * 64), ("sampling_seed", True), ("sampling_seed", -1)):
            bad = deepcopy(self.validation_manifests["categorical"])
            bad["R16"][name] = replacement
            with self.assertRaises(ValueError):
                select_checkpoint([candidate], [0.001], validation_manifest=bad)
        fixed = deepcopy(self.validation_manifests["categorical"])
        result = select_checkpoint([candidate], [0.001], validation_manifest=fixed)
        fixed["R0"]["sampling_seed"] += 1
        self.assertEqual(result["validation_manifest"], self.validation_manifests["categorical"])


if __name__ == "__main__":
    unittest.main()
