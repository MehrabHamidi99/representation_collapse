"""Tiny bounded-engine checks, not full-protocol learned competence evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from representation_collapse.competence import IMMEDIATE, R0, NATIVE_PHASE, evaluate_competence, make_competence_batch
from representation_collapse.diagnostics import streaming_register_oracle


@unittest.skipIf(torch is None, "Torch requires experiments dependencies")
class CompetenceTrainingTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "configs/experiments/competence-recovery-v2.json").read_text())

    def overrides(self):
        return {"stage_updates": [2, 1, 1, 1, 1], "hidden_size": 4,
                "input_bottleneck": 4, "batch_size": 8, "validation_interval": 1,
                "fixed_batch_updates": 1, "fixed_batch_examples_per_coordinate": 1}

    def test_full_protocol_cannot_change_gates_schedule_or_precision(self):
        from representation_collapse.competence_training import validate_config, _effective
        config = self.config()
        validate_config(config)
        for section, key, value in (("validation", "categorical_min_accuracy", 0.25),
                                    ("precision", "cudnn_allow_tf32", True),
                                    ("dimensions", "hidden_size", 8),
                                    ("diagnostics", "fixed_batch_updates", 500)):
            changed = deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError): validate_config(changed)
        for bad in ({"hidden_size": 256}, {"stage_updates": [501, 500, 500, 500, 1500]},
                    {"categorical_min_accuracy": 0.25}, {"stage_updates": [1] * 5}):
            with self.assertRaises(ValueError): _effective(config, bad)
        self.assertEqual(_effective(config, self.overrides())["stages"][0]["updates"], 2)

    def test_stage4_exposure_counts_are_prespecified_one_update_imbalance(self):
        from representation_collapse.competence_training import TRAIN_ASSAYS, STAGE_UPDATES
        cycle = TRAIN_ASSAYS[3]
        counts = {name: sum(cycle[i % len(cycle)] == name for i in range(STAGE_UPDATES[3])) for name in cycle}
        self.assertEqual(list(counts.values()), [167, 167, 166])
        self.assertEqual(STAGE_UPDATES, [500, 500, 500, 500, 1500])

    def test_boundary_uses_current_scores_and_never_advances_early(self):
        from representation_collapse.competence_training import boundary_decision
        batch = make_competence_batch(R0, "categorical", 1)
        arguments = dict(trained_seed=0, checkpoint_id="current", measurement_phase=NATIVE_PHASE)
        good = evaluate_competence(batch, streaming_register_oracle(batch.inputs, "categorical"), **arguments)
        bad = evaluate_competence(batch, np.zeros(len(batch.targets), dtype=int), **arguments)
        self.assertFalse(boundary_decision(2, 499, 500, [good])["advance"])
        self.assertTrue(boundary_decision(2, 500, 500, [good])["advance"])
        self.assertFalse(boundary_decision(2, 500, 500, [bad])["advance"])
        self.assertFalse(boundary_decision(2, 501, 500, [good])["advance"])

    def test_actual_tiny_fit_records_optimization_native_chronology_and_failed_stop(self):
        from representation_collapse.competence_training import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fit"
            original_step = torch.optim.AdamW.step
            before_optimization = []
            def inspected_step(optimizer, *args, **kwargs):
                config = json.loads((output / "config.json").read_text())
                provenance = json.loads((output / "provenance.json").read_text())
                before_optimization.append(config["validation_manifest"])
                self.assertEqual(config["validation_manifest"], provenance["validation_manifest"])
                self.assertEqual(config["validation_manifest_frozen_at_step"], 0)
                self.assertEqual(len(list((output / "validation_manifests").glob("*.npz"))), 7)
                return original_step(optimizer, *args, **kwargs)
            with patch("torch.optim.AdamW.step", new=inspected_step):
                result = run(self.config(), output, "gru", "categorical", 0, 0.001,
                             engineering_overrides=self.overrides())
            self.assertEqual(result["steps"], 1)
            self.assertEqual(result["outcome"], "stopped_fixed_batch_failure")
            self.assertFalse(result["scientific_acceptance"])
            self.assertIsNone(result["selection"]["selected"])
            self.assertTrue(result["engineering_override"])
            self.assertEqual(len(before_optimization), 1)
            self.assertEqual(set(before_optimization[0]), {"R0", "R1", "R2", "R4", "R8", "R16"})
            history = json.loads((output / "history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertTrue(all(v > 0 for v in history[0]["module_parameter_change_norms"].values()))
            self.assertTrue(all(np.isfinite(v) for v in history[0]["module_gradient_norms"].values()))
            self.assertGreater(history[0]["queried_write_input_gradient_norm"], 0)
            self.assertTrue(history[0]["native_chronology"][IMMEDIATE]["passed"])
            self.assertEqual(history[0]["validation_records"][0]["trained_seed_units"], 1)
            for arm in history[0]["validation_records"][0]["arms"].values():
                self.assertEqual([r["episodes"] for r in arm["coordinates"]], [128] * 8)
            provenance = json.loads((output / "provenance.json").read_text())
            self.assertFalse(provenance["cuda_matmul_allow_tf32"])
            self.assertFalse(provenance["cudnn_allow_tf32"])
            self.assertTrue(provenance["deterministic_algorithms"])
            self.assertEqual(provenance["float32_matmul_precision"], "highest")
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "completed")
            with np.load(output / "raw_validation/step-000001.npz") as saved:
                self.assertEqual(saved[f"{IMMEDIATE}_post_write"].shape, (1024, 4))
                self.assertEqual(saved[f"{IMMEDIATE}_pre_query"].shape, (1024, 4))
                self.assertEqual(saved[f"{IMMEDIATE}_post_query"].shape, (1024, 4))
                np.testing.assert_array_equal(saved[f"{IMMEDIATE}_post_write"], saved[f"{IMMEDIATE}_pre_query"])
            manifest = json.loads((output / "manifest.json").read_text())["sha256"]
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            checkpoint = next((output / "checkpoints").glob("*.pt"))
            packet = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(packet["step"], 1)
            self.assertFalse(packet["scientific_acceptance"])
            with self.assertRaises(FileExistsError):
                run(self.config(), output, "gru", "categorical", 0, 0.001,
                    engineering_overrides=self.overrides())

    def test_numerical_failure_is_retained_without_step_extension(self):
        from representation_collapse.competence_training import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failed"
            with patch("representation_collapse.competence_training.VectorMemoryModel.forward", return_value=torch.full((8, 1), float("nan"))):
                with self.assertRaises(FloatingPointError):
                    run(self.config(), output, "lstm", "continuous", 0, 0.001,
                        engineering_overrides=self.overrides())
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["step"], 0)
            self.assertFalse(status["scientific_acceptance"])
            self.assertEqual(json.loads((output / "history.json").read_text()), [])
            self.assertTrue((output / "manifest.json").exists())

    def test_tiny_lstm_fit_preserves_complete_hidden_and_cell_phases(self):
        from representation_collapse.competence_training import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "lstm"
            result = run(self.config(), output, "lstm", "continuous", 0, 0.001,
                         engineering_overrides=self.overrides())
            self.assertEqual(result["steps"], 1)
            self.assertEqual(result["outcome"], "stopped_fixed_batch_failure")
            self.assertFalse(result["scientific_acceptance"])
            with np.load(output / "raw_validation/step-000001.npz") as saved:
                for phase in ("post_write", "pre_query", "post_query"):
                    self.assertEqual(saved[f"{IMMEDIATE}_{phase}"].shape, (1024, 8))
                    self.assertTrue(np.isfinite(saved[f"{IMMEDIATE}_{phase}"]).all())

    def test_mocked_validation_exercises_stage5_candidate_schema_without_performance_claim(self):
        from representation_collapse.competence_training import run
        # Six real tiny updates; oracle validation is deliberately mocked to
        # exercise bookkeeping/selection integration, never learned competence.
        def fixture_scores(model, manifests, assays, device, seed, checkpoint_id, model_kind):
            records, losses = [], {}
            for assay in assays:
                losses[assay] = {"immediate": 0.2} if assay == IMMEDIATE else {"periodic": 0.2, "control": 0.3}
                if assay in (IMMEDIATE, "full_prefix_r0", "full_prefix_r1", "full_prefix_r2"):
                    batch = manifests[assay]
                    record = evaluate_competence(batch, streaming_register_oracle(batch.inputs, model.task),
                                                 trained_seed=seed, checkpoint_id=checkpoint_id, measurement_phase=NATIVE_PHASE)
                    record["model"] = model_kind
                    records.append(record)
            return records, losses, {}, {"scope": "mocked oracle scores; no native performance evidence"}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            with patch("representation_collapse.competence_training.evaluate_assays", side_effect=fixture_scores), \
                 patch("representation_collapse.competence_training._fixed_overfit", return_value={"passed": True, "scope": "mocked diagnostic"}):
                result = run(self.config(), output, "gru", "continuous", 0, 0.001, engineering_overrides=self.overrides())
            self.assertEqual(result["steps"], 6)
            self.assertEqual(result["outcome"], "completed_schedule")
            self.assertFalse(result["scientific_acceptance"])
            self.assertEqual(len(result["stage_results"]), 5)
            candidates = json.loads((output / "candidates.json").read_text())
            self.assertEqual(len(candidates), 1)
            self.assertEqual(set(candidates[0]["validation_loss_by_horizon"]), {"R1", "R2", "R4", "R8", "R16"})
            self.assertEqual(result["selection"]["selected"]["step"], 6)
            self.assertAlmostEqual(result["selection"]["selected"]["balanced_validation_loss"], 0.25)
            self.assertTrue((output / "validation_manifests/full_prefix_r16.npz").exists())
            freeze = json.loads((output / "config.json").read_text())["validation_manifest"]
            self.assertEqual(candidates[0]["validation_manifest"], freeze)
            for record in candidates[0]["short_horizon_records"]:
                horizon = "R" + record["assay"].rsplit("r", 1)[1]
                self.assertEqual(record["batch_sha256"], freeze[horizon]["batch_sha256"])
                self.assertEqual(record["sampling_seed"], freeze[horizon]["sampling_seed"])

    def test_replacement_validation_batch_cannot_pass_the_caller_freeze(self):
        from representation_collapse.competence_training import _validation_identity, _verify_validation_freeze
        first = make_competence_batch(R0, "categorical", 1)
        replacement = make_competence_batch(R0, "categorical", 2)
        freeze = {"R0": _validation_identity(R0, "categorical", first)}
        _verify_validation_freeze({R0: first}, [R0], "categorical", freeze, {})
        with self.assertRaisesRegex(ValueError, "pre-optimization caller freeze"):
            _verify_validation_freeze({R0: replacement}, [R0], "categorical", freeze, {})


if __name__ == "__main__":
    unittest.main()
