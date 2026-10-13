"""Prospective runner controls; reduced-budget fixtures are infrastructure only."""
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

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "Torch requires experiments dependencies")
class PrefixAcquisitionTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/experiments/prefix-acquisition-discovery-v3.json").read_text())

    def fixture_config(self, project_root):
        from representation_collapse.pilot import VectorMemoryModel
        config = self.config()
        for index, pin in enumerate(config["warm_start_checkpoints"]):
            torch.manual_seed(23 + index)
            model = VectorMemoryModel(pin["model"], pin["task"], 128, 64)
            path = project_root / f"warm-{pin['model']}-{pin['task']}.pt"
            torch.save({"model_state_dict": model.state_dict(), "step": 500, "stage": 1,
                        "config": {"model": pin["model"], "task": pin["task"], "seed": 0,
                                   "learning_rate": 0.001, "dimensions": config["dimensions"]},
                        "optimizer_state_dict": {"must_not_be_imported": True},
                        "fixture_only": True, "scientific_acceptance": False}, path)
            pin["checkpoint_relative_to_project"] = path.name
            pin["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return config

    def test_scientific_dimensions_seeds_precision_and_bounds_cannot_change(self):
        from representation_collapse.prefix_acquisition import validate_config
        config = self.config(); validate_config(config)
        for key, value in (("training_stream_seed", 1), ("validation_sampling_seed", 10001),
                           ("updates_per_fit", 5001), ("maximum_fits", 9),
                           ("device", "cuda"), ("precision", "mixed"), ("seed", False),
                           ("maximum_fits", 8.0), ("population_variance", 1.0)):
            bad = deepcopy(config); bad[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): validate_config(bad)
        bad = deepcopy(config); bad["dimensions"]["batch_size"] = 64
        with self.assertRaises(ValueError): validate_config(bad)

    def test_fixture_only_reduces_budget_not_scientific_dimensions(self):
        from representation_collapse.prefix_acquisition import FixtureBudget
        for budget in (FixtureBudget(updates_per_fit=5001), FixtureBudget(evaluate_every=501),
                       FixtureBudget(fit_indices=(0, 0)), FixtureBudget(wall_seconds=1201),
                       FixtureBudget(updates_per_fit=True),
                       FixtureBudget(updates_per_fit=5000, evaluate_every=500, fit_indices=tuple(range(8)), wall_seconds=1200)):
            with self.assertRaises(ValueError): budget.validate()
        FixtureBudget(updates_per_fit=2, evaluate_every=1, fit_indices=(4, 5)).validate()

    def test_stream_is_prefit_pinned_matches_regeneration_and_rejects_changed_batch(self):
        from representation_collapse.prefix_acquisition import prehash_stream, verify_stream_batch, _rolling_start
        from representation_collapse.diagnostics import paired_batch_with_zero
        import time
        expected = prehash_stream("categorical", 3, time.monotonic() + 60)
        self.assertEqual(expected, prehash_stream("categorical", 3, time.monotonic() + 60))
        rolling = _rolling_start("categorical", 3)
        rng = np.random.default_rng(2026101301)
        for step in range(1, 4):
            verify_stream_batch(paired_batch_with_zero(rng, 16, 0, "categorical"), step, expected, rolling)
        self.assertEqual(rolling.hexdigest(), expected["stream_sha256"])
        changed = paired_batch_with_zero(np.random.default_rng(2026101301), 16, 0, "categorical")
        changed.inputs[0, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            verify_stream_batch(changed, 1, expected, _rolling_start("categorical", 3))

    def test_input_target_oracle_rejects_wrong_labels_and_invalid_chronology(self):
        from representation_collapse.prefix_acquisition import target_oracle
        from representation_collapse.diagnostics import paired_batch_with_zero
        for task in ("categorical", "continuous"):
            batch = paired_batch_with_zero(np.random.default_rng(1), 16, 0, task)
            q = np.tile(batch.queries, 2)
            self.assertEqual(target_oracle(batch.inputs, batch.targets, q, task), 32)
            changed = batch.targets.copy(); changed[0] += 1
            with self.assertRaises(ValueError): target_oracle(batch.inputs, changed, q, task)
            changed_inputs = batch.inputs.copy(); changed_inputs[:, -1, :5] = [0, 1, 0, 0, 0]
            with self.assertRaises(ValueError): target_oracle(changed_inputs, batch.targets, q, task)
            changed_inputs = batch.inputs.copy(); changed_inputs[:, -1, 37] = 1
            with self.assertRaises(ValueError): target_oracle(changed_inputs, batch.targets, q, task)

    def test_missing_freeze_cannot_create_production_outputs(self):
        from representation_collapse.prefix_acquisition import run
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "forbidden"
            with self.assertRaises(FileNotFoundError):
                run(ROOT / "configs/experiments/prefix-acquisition-discovery-v3.json", output,
                    Path(temporary) / "missing-freeze.json")
            self.assertFalse(output.exists())

    def test_hash_changed_warm_start_blocks_before_any_optimization(self):
        from representation_collapse.prefix_acquisition import FixtureBudget, run_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root)
            warm = root / config["warm_start_checkpoints"][0]["checkpoint_relative_to_project"]
            warm.write_bytes(warm.read_bytes() + b"changed")
            with patch("torch.optim.AdamW.step") as step:
                with self.assertRaises(ValueError): run_fixture(config, root / "fit", root, FixtureBudget(updates_per_fit=1))
                step.assert_not_called()
            status = json.loads((root / "fit/status.json").read_text())
            self.assertEqual(status["state"], "failed_execution_identity_or_numerics")
            self.assertEqual(status["total_optimizer_updates"], 0)

    def test_rehashed_wrong_step_checkpoint_is_rejected_semantically(self):
        from representation_collapse.prefix_acquisition import FixtureBudget, run_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root)
            pin = config["warm_start_checkpoints"][0]; path = root / pin["checkpoint_relative_to_project"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["step"] = 501; torch.save(checkpoint, path)
            pin["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch("torch.optim.AdamW.step") as step:
                with self.assertRaises(ValueError): run_fixture(config, root / "fit", root, FixtureBudget(updates_per_fit=1))
                step.assert_not_called()

    def test_actual_matched_fixture_resets_adamw_and_preserves_complete_lstm_states(self):
        from representation_collapse.prefix_acquisition import FixtureBudget, run_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root); output = root / "fit"
            before_steps = []
            original = torch.optim.AdamW.step
            def inspected(optimizer, *args, **kwargs):
                frozen = json.loads((output / "prefit-freeze.json").read_text())
                before_steps.append((len(optimizer.state), frozen["frozen_before_first_optimization"]))
                return original(optimizer, *args, **kwargs)
            with patch("torch.optim.AdamW.step", new=inspected), patch("torch.optim.AdamW.load_state_dict", side_effect=AssertionError("optimizer inheritance forbidden")):
                result = run_fixture(config, output, root, FixtureBudget(updates_per_fit=1, fit_indices=(4, 5)))
            self.assertEqual(result["state"], "completed")
            self.assertTrue(result["fixture_only"])
            self.assertFalse(result["scientific_acceptance"])
            self.assertEqual(result["total_optimizer_updates"], 2)
            self.assertEqual(before_steps, [(0, True), (0, True)])
            paths = [output / f"lstm-categorical-{condition}" for condition in ("fresh_seed0", "immediate_step500_transfer_reset_optimizer")]
            provenances = [json.loads((path / "provenance.json").read_text()) for path in paths]
            self.assertEqual(provenances[0]["training_stream_sha256"], provenances[1]["training_stream_sha256"])
            self.assertNotEqual(provenances[0]["initial_weight_sha256"], provenances[1]["initial_weight_sha256"])
            self.assertFalse(provenances[1]["warm_start"]["inherited_optimizer_state"])
            pin = next(p for p in config["warm_start_checkpoints"] if p["model"] == "lstm" and p["task"] == "categorical")
            warm = torch.load(root / pin["checkpoint_relative_to_project"], map_location="cpu", weights_only=True)
            initial = torch.load(paths[1] / "initial-checkpoint.pt", map_location="cpu", weights_only=True)
            for name in warm["model_state_dict"]:
                self.assertTrue(torch.equal(warm["model_state_dict"][name], initial["model_state_dict"][name]))
            for path in paths:
                history = json.loads((path / "history.json").read_text())
                self.assertEqual(len(history), 1)
                self.assertTrue(history[0]["native_chronology"]["passed"])
                self.assertTrue(all(c["packing_order"] == "hidden_then_cell" and c["state_dimension"] == 256 for c in history[0]["native_chronology"]["chunks"]))
                with np.load(path / "validation-step-000001.npz") as raw:
                    for phase in ("post_write", "pre_query", "post_query"):
                        self.assertEqual(raw[phase].shape, (2048, 256))
                    np.testing.assert_array_equal(raw["post_write"], raw["pre_query"])
            for name, digest in json.loads((output / "manifest.json").read_text())["sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError): run_fixture(config, output, root, FixtureBudget(updates_per_fit=1, fit_indices=(4, 5)))

    def test_early_passing_validation_does_not_replace_the_fixed_endpoint(self):
        from representation_collapse import prefix_acquisition as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root)
            original = module._evaluate
            def passing(*args, **kwargs):
                record, raw, chronology = original(*args, **kwargs)
                record["coordinate_gate_passed"] = True  # Mocked fixture score, not empirical evidence.
                return record, raw, chronology
            with patch.object(module, "_evaluate", side_effect=passing):
                result = module.run_fixture(config, root / "fit", root,
                    module.FixtureBudget(updates_per_fit=2, fit_indices=(0,)))
            fit = result["fits"][0]
            self.assertEqual(fit["updates_completed"], 2)
            self.assertEqual(fit["first_passing_validation_step"], 1)
            self.assertEqual(fit["endpoint_step"], 2)
            self.assertIsNone(fit["selected_checkpoint"])

    def test_deadline_preserves_partial_updates_without_fabricating_gate_failure(self):
        from representation_collapse import prefix_acquisition as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root)
            with patch.object(module, "_evaluate", side_effect=module.DeadlineReached("fixture deadline")):
                result = module.run_fixture(config, root / "fit", root,
                    module.FixtureBudget(updates_per_fit=2, fit_indices=(0,)))
            self.assertEqual(result["state"], "incomplete_censored_overall_deadline")
            self.assertEqual(result["total_optimizer_updates"], 1)
            self.assertIsNone(result["fits"][0]["endpoint_coordinate_gate_passed"])
            history = json.loads((root / "fit/gru-categorical-fresh_seed0/history.json").read_text())
            self.assertEqual(len(history), 1)
            self.assertNotIn("validation", history[0])

    def test_between_evaluations_deadline_preserves_current_censored_weights(self):
        from representation_collapse import prefix_acquisition as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root); output = root / "fit"
            clock = {"now": 0.0}; updated = []
            original = torch.optim.AdamW.step
            def advancing_step(optimizer, *args, **kwargs):
                result = original(optimizer, *args, **kwargs)
                updated.extend(parameter.detach().clone() for parameter in optimizer.param_groups[0]["params"])
                clock["now"] = 2.0
                return result
            with patch.object(module.time, "monotonic", side_effect=lambda: clock["now"]), patch("torch.optim.AdamW.step", new=advancing_step):
                result = module.run_fixture(config, output, root,
                    module.FixtureBudget(updates_per_fit=2, evaluate_every=2, fit_indices=(0,), wall_seconds=1))
            self.assertEqual(result["state"], "incomplete_censored_overall_deadline")
            self.assertEqual(result["total_optimizer_updates"], 1)
            fit = result["fits"][0]; path = output / "gru-categorical-fresh_seed0"
            self.assertIsNone(fit["endpoint_coordinate_gate_passed"])
            self.assertEqual(fit["checkpoint_realized_step"], 1)
            self.assertFalse(list(path.glob("checkpoint-step-*.pt")))
            partial = torch.load(path / fit["censored_current_checkpoint_id"], map_location="cpu", weights_only=True)
            self.assertEqual(partial["step"], 1)
            self.assertTrue(partial["censored"])
            self.assertFalse(partial["endpoint_competence_known"])
            self.assertFalse(partial["optimizer_state_saved"])
            self.assertFalse(partial["resume_authorized"])
            self.assertIsNone(partial["selected_checkpoint"])
            initial = torch.load(path / "initial-checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertTrue(any(not torch.equal(initial["model_state_dict"][name], tensor) for name, tensor in partial["model_state_dict"].items()))
            self.assertEqual(len(partial["model_state_dict"]), len(updated))
            for actual, expected in zip(partial["model_state_dict"].values(), updated):
                self.assertTrue(torch.equal(actual, expected))
            manifest = json.loads((output / "manifest.json").read_text())["sha256"]
            checkpoint_relative = str((path / fit["censored_current_checkpoint_id"]).relative_to(output))
            self.assertEqual(manifest[checkpoint_relative], hashlib.sha256((path / fit["censored_current_checkpoint_id"]).read_bytes()).hexdigest())

    def test_completed_nonfinite_optimizer_update_keeps_realized_step_and_unusable_weights(self):
        from representation_collapse import prefix_acquisition as module
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.fixture_config(root); output = root / "fit"
            original = torch.optim.AdamW.step
            def nonfinite_after_step(optimizer, *args, **kwargs):
                result = original(optimizer, *args, **kwargs)
                with torch.no_grad():
                    optimizer.param_groups[0]["params"][0].flatten()[0] = float("nan")
                return result
            with patch("torch.optim.AdamW.step", new=nonfinite_after_step):
                with self.assertRaises(FloatingPointError):
                    module.run_fixture(config, output, root,
                        module.FixtureBudget(updates_per_fit=2, evaluate_every=2, fit_indices=(0,)))
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "failed_execution_identity_or_numerics")
            self.assertEqual(status["total_optimizer_updates"], 1)
            fit = status["fits"][0]; path = output / "gru-categorical-fresh_seed0"
            self.assertEqual(fit["checkpoint_realized_step"], 1)
            self.assertTrue(fit["nonfinite_parameters_after_completed_update"])
            self.assertFalse(fit["preserved_checkpoint_parameters_finite"])
            self.assertFalse(fit["preserved_checkpoint_usable_for_evaluation"])
            self.assertIsNone(fit["endpoint_coordinate_gate_passed"])
            history = json.loads((path / "history.json").read_text())
            self.assertEqual([row["step"] for row in history], [1])
            self.assertFalse(history[0]["post_update_parameters_finite"])
            self.assertNotIn("validation", history[0])
            checkpoint = torch.load(path / fit["censored_current_checkpoint_id"], map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["step"], 1)
            self.assertFalse(checkpoint["parameters_finite"])
            self.assertFalse(checkpoint["usable_for_evaluation"])
            self.assertTrue(checkpoint["diagnostic_preservation_only"])
            self.assertTrue(any(not torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values()))
            self.assertIsNone(checkpoint["selected_checkpoint"])

    def test_stale_review_or_dirty_git_cannot_pass_freeze(self):
        from representation_collapse.prefix_acquisition import verify_freeze
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            folder = Path(temporary); review_path = folder / "review.json"
            review_path.write_text('{"accepted":true}\n')
            module_path = ROOT / "src/representation_collapse/prefix_acquisition.py"
            config_path = ROOT / "configs/experiments/prefix-acquisition-discovery-v3.json"
            packet = {"kind": "prefix_acquisition_v3_freeze", "protocol_id": "prefix-acquisition-discovery-v3",
                      "local_execution_authorized": True, "scientific_acceptance": False,
                      "source_commit": "a" * 40, "config_relative_to_project": str(config_path.relative_to(ROOT)),
                      "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                      "independent_review": {"accepted": True,
                          "receipt_relative_to_project": str(review_path.relative_to(ROOT)),
                          "receipt_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                          "implementation_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                          "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}}
            receipt = folder / "freeze.json"; receipt.write_text(json.dumps(packet))
            with patch("subprocess.check_output", side_effect=["a" * 40 + "\n", " M tracked.py\n"]):
                with self.assertRaises(ValueError): verify_freeze(config_path, receipt, ROOT)
            packet["independent_review"]["implementation_sha256"] = "0" * 64
            receipt.write_text(json.dumps(packet))
            with patch("subprocess.check_output") as git:
                with self.assertRaises(ValueError): verify_freeze(config_path, receipt, ROOT)
                git.assert_not_called()


if __name__ == "__main__":
    unittest.main()
