from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse import multibit_memory as mm  # noqa: E402


class FakeReporter:
    instances = []

    def __init__(self, output, *, project, run_id, mode, config, identities):
        self.output = Path(output)
        self.run_id = run_id
        self.logs = []
        self.manifests = []
        assert project == "representation-collapse" and mode == "offline"
        assert all(type(value) in (str, int, float, bool, type(None)) for value in config.values())
        assert all(type(value) is str and len(value) == 64 for value in identities.values())
        self.instances.append(self)

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log(self, values, *, step):
        self.logs.append((step, dict(values)))

    def log_manifest(self, name, *, metadata, identities):
        self.manifests.append((name, dict(metadata), dict(identities)))


class MultibitMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        mm._configure_torch(torch)

    def test_committed_config_matches_v1_and_shortened_schedule_rejects(self):
        config_path = ROOT / "configs/experiments/multibit-memory-v1.json"
        config = mm.validate_config(json.loads(config_path.read_text()))
        self.assertEqual(config["schema"], "multibit-memory-v1")
        self.assertEqual(config["stages"][-1]["validate_repeats"], [0, 1, 2, 4, 8])
        changed = dict(config)
        changed["stages"] = [dict(x) for x in config["stages"]]
        changed["stages"][-1]["validate_repeats"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            mm.validate_config(changed)

    def test_balanced_payload_and_no_label_leakage_for_every_dimension(self):
        for dimension in (1, 4, 16, 64):
            rng = np.random.default_rng(100 + dimension)
            bits = mm.balanced_bits(rng, 256, dimension)
            np.testing.assert_array_equal(np.sum(bits < 0, axis=0), np.full(dimension, 128))
            pairs = mm.body_pairs(rng, 256, 2)
            x = mm.render_pairs(bits, pairs, dimension)
            self.assertEqual(x.shape, (512, 26, 69))
            for cue in range(256):
                periodic, control = x[2 * cue], x[2 * cue + 1]
                np.testing.assert_array_equal(periodic[0, 5:5 + dimension], bits[cue])
                np.testing.assert_array_equal(periodic[-1], control[-1])
                self.assertEqual(periodic[0, mm.WRITE], 1)
                self.assertEqual(periodic[-1, mm.QUERY], 1)
                self.assertTrue(np.all(periodic[1:, 5:] == 0))
                self.assertTrue(np.all(periodic[-1, :5] == np.eye(5, dtype=np.float32)[mm.QUERY]))

    def test_schedule_pairs_preserve_marginals_and_reject_periodic_control(self):
        from collections import Counter
        from representation_collapse.vector_task import cyclic_bigrams
        for repeats, dwell in ((1, 1), (2, 1), (8, 1), (32, 4)):
            pairs = mm.body_pairs(np.random.default_rng(55 + repeats), 8, repeats, dwell)
            for periodic, control in pairs:
                self.assertEqual(len(periodic), len(control))
                self.assertEqual(Counter(periodic), Counter(control))
                if repeats > 1:
                    self.assertEqual(cyclic_bigrams(periodic), cyclic_bigrams(control))
                    self.assertEqual(periodic[-12 * dwell:], control[-12 * dwell:])
                    self.assertIsNone(mm.proper_period(control))
                if repeats == 1:
                    self.assertEqual(periodic, control)

    def test_balanced_polarity_competence_gate_rejects_one_bad_polarity(self):
        rng = np.random.default_rng(91)
        labels = mm.balanced_bits(rng, 256, 4)
        logits = np.broadcast_to((labels[None, :, None, :] * 10).astype(np.float32),
                                 (1, 256, 2, 4)).copy()
        summary = mm._accuracy(logits, labels)
        self.assertTrue(mm.competence_pass(summary))
        # Aggregate accuracy remains >= .95, but one polarity-specific slice is below gate.
        negative_indices = np.flatnonzero(labels[:, 0] < 0)[:8]
        logits[0, negative_indices, 0, 0] *= -1
        summary = mm._accuracy(logits, labels)
        self.assertGreaterEqual(summary["per_bit_accuracy_by_arm"]["periodic"][0], 0.95)
        self.assertLess(summary["per_bit_polarity_accuracy_by_arm"]["periodic"]["0"]["negative"], 0.95)
        self.assertFalse(mm.competence_pass(summary))

    def test_first_zero_can_select_strict_tie_retains_and_paired_delta_signed(self):
        labels = np.array([[-1], [1]], dtype=np.int8)
        logits = np.zeros((2, 2, 2, 1), dtype=np.float32)
        logits[:, :, 0, 0] = -labels[None, :, 0]  # periodic arm is always wrong
        logits[:, :, 1, 0] = labels[None, :, 0]   # control arm is always correct
        summary = mm._accuracy(logits, labels)
        self.assertAlmostEqual(summary["paired_error_difference_periodic_minus_control"], 1.0)
        self.assertEqual(summary["mean_bit_error_by_arm"], {"periodic": 1.0, "control": 0.0})
        self.assertTrue(mm.competence_pass(mm._accuracy(
            np.broadcast_to(labels[None, :, None, :] * 4, logits.shape), labels)))
        self.assertFalse(mm.competence_pass({"per_bit_accuracy_by_arm":
                                             {"periodic": [0.5], "control": [0.5]}}))

    def test_conditional_covariance_centers_within_forcing_not_across(self):
        cues = np.arange(256, dtype=np.float32)
        base = np.zeros((2, 256, 2, 128), dtype=np.float32)
        base[0, :, :, 0] = cues[:, None]
        base[1, :, :, 0] = cues[:, None] + 10000
        base[0, :, :, 64] = cues[:, None] * 2
        base[1, :, :, 64] = cues[:, None] * 2 + 30000
        summary = mm.conditional_geometry(base, dimension=1)
        spectra = summary["conditional_spectra"]
        self.assertEqual(len(spectra), 4)
        self.assertEqual(summary["theoretical_rank_ceiling"], 1)
        np.testing.assert_allclose(spectra[0]["eigenvalues_ascending"],
                                   spectra[2]["eigenvalues_ascending"], rtol=1e-6, atol=1e-6)
        self.assertEqual(spectra[0]["rank_ceiling"], 1)

    def test_native_lstm_query_resume_matches_segmented_full_forward(self):
        import torch
        rng = np.random.default_rng(7)
        bits = mm.balanced_bits(rng, 8, 4)
        pairs = mm.body_pairs(rng, 8, 2)
        x = torch.from_numpy(mm.render_pairs(bits, pairs, 4))
        model = mm._make_model(torch).eval()
        with torch.no_grad():
            logits, _after, before = model.states_and_logits(x)
            full = model.full_forward(x)
            resumed = model.query_from_state(x[:, -1], before[:, :64], before[:, 64:])
        torch.testing.assert_close(logits, full, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logits, resumed, rtol=0, atol=0)

    def test_chunked_evaluation_writes_four_drivers_correct_labels_and_signed_pairs(self):
        import torch

        class PredictByCue(torch.nn.Module):
            def eval(self):
                return self

            def states_and_logits(self, x):
                n = x.shape[0]
                outputs = torch.zeros((n, 64))
                cue = x[:, 0, 5]
                # Even row is periodic: invert cue. Odd row is control: recall cue.
                arm_sign = torch.where(torch.arange(n) % 2 == 0, -1.0, 1.0)
                outputs[:, 0] = cue * arm_sign
                state = torch.zeros((n, 128))
                state[:, 0] = cue
                return outputs, state, state

            def query_from_state(self, query_input, hidden, cell):
                return torch.zeros((len(query_input), 64))

        bits = np.array([[-1], [1], [-1], [1]], dtype=np.int8)
        forcing = [("AX", "BX"), ("BX", "AX"), ("AB", "BA"), ("BA", "AB")]
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "condition.partial.npz"
            summary = mm._evaluate_paired_design(
                PredictByCue(), torch, bits, forcing, 1,
                SimpleNamespace(stop_file=Path(td) / "STOP", deadline_unix=time.time() + 60),
                condition="test", chunk_rows=4, partial_path=raw)
            self.assertEqual(summary["forcing_units"], 4)
            self.assertEqual(summary["oracle_exact_accuracy"], 1.0)
            self.assertEqual(summary["paired_error_difference_periodic_minus_control"], 1.0)
            self.assertEqual(summary["mean_bit_error_by_arm"], {"periodic": 1.0, "control": 0.0})
            self.assertEqual(len(summary["conditional_complete_state_geometry"]["conditional_spectra"]), 8)
            with np.load(Path(td) / "condition.npz") as data:
                self.assertEqual(data["logits"].shape, (4, 4, 2, 1))
                self.assertEqual(data["labels"].shape, (4, 4, 2, 1))
                self.assertEqual(data["before_query"].shape, (4, 4, 2, 128))
                self.assertEqual(data["oracle_predictions"].shape, (4, 4, 1))
                self.assertTrue(bool(data["complete"]))

    def test_stop_during_evaluation_flushes_incomplete_chunk_artifact(self):
        import torch
        class ZeroModel(torch.nn.Module):
            def eval(self): return self
            def states_and_logits(self, x):
                zeros = torch.zeros((len(x), 128))
                return torch.zeros((len(x), 64)), zeros, zeros
            def query_from_state(self, query_input, hidden, cell):
                return torch.zeros((len(query_input), 64))

        bits = np.tile(np.array([[-1], [1]], dtype=np.int8), (2, 1))
        forcing = [("AX", "BX"), ("BX", "AX")]
        with tempfile.TemporaryDirectory() as td:
            stop = Path(td) / "STOP"; stop.touch()
            partial = Path(td) / "R0.partial.npz"
            with self.assertRaises(mm.StopRequested):
                mm._evaluate_paired_design(
                    ZeroModel(), torch, bits, forcing, 1,
                    SimpleNamespace(stop_file=stop, deadline_unix=time.time() + 60),
                    condition="R0-q1", chunk_rows=4, partial_path=partial)
            with np.load(partial) as data:
                self.assertFalse(bool(data["complete"]))
                np.testing.assert_array_equal(data["completed_cues"], [0, 0])

    def test_two_tiny_fit_runs_initialize_reporters_and_save_failed_checkpoints(self):
        import torch
        stages = [{"updates": 1, "train_repeats": [0], "validate_repeats": [0]}]
        fake_config = {"schema": mm.SCHEMA, "stages": stages,
                       "validation_cue_vectors": 256, "test_cue_vectors": 256,
                       "test_forcing_units": 4, "test_conditions": [[0, 1]], "test_chunk": 64}
        forced_failure = {
            "repeats": 0,
            "per_bit_accuracy_by_arm": {"periodic": [0.5], "control": [0.5]},
            "per_bit_polarity_accuracy_by_arm": {
                arm: {"0": {"negative": 0.5, "positive": 0.5}} for arm in mm.ARM_NAMES},
            "all_active_bits_polarities_arms_min_accuracy": 0.5,
            "all_active_bits_all_arms_min_accuracy": 0.5,
        }
        args = SimpleNamespace(output=None, stop_file=Path("/definitely/not/a/STOP"),
                                deadline_unix=time.time() + 120)
        with tempfile.TemporaryDirectory() as td:
            args.output = Path(td)
            FakeReporter.instances.clear()
            with mock.patch.object(mm, "ExperimentReporter", FakeReporter), \
                    mock.patch.object(mm, "_validate_model", return_value=forced_failure):
                outputs = []
                for dimension in (1, 4):
                    result = mm._fit_one(fake_config, dimension, 0, args,
                                         "a" * 64, "b" * 64, "c" * 64, "multi-smoke")
                    outputs.append(result)
            self.assertEqual([item["status"] for item in outputs],
                             ["competence_failed", "competence_failed"])
            self.assertEqual(len(FakeReporter.instances), 2)
            self.assertEqual([r.run_id for r in FakeReporter.instances],
                             ["multi-smoke-D1-s0", "multi-smoke-D4-s0"])
            for dimension in (1, 4):
                fit_dir = Path(td) / f"D{dimension}-seed0"
                self.assertTrue((fit_dir / "checkpoint.pt").is_file())
                self.assertTrue((fit_dir / "stage-1-validation.json").is_file())
                self.assertFalse((fit_dir / "evaluation-design.json").exists())

    def test_heldout_deadline_returns_partial_result_and_preserves_checkpoint(self):
        import torch
        stages = [{"updates": 1, "train_repeats": [0], "validate_repeats": [0]}]
        config = {"schema": mm.SCHEMA, "stages": stages,
                  "validation_cue_vectors": 256, "test_cue_vectors": 256,
                  "test_forcing_units": 4, "test_conditions": [[0, 1]], "test_chunk": 64}
        correct = np.ones((1, 256, 2, 1), dtype=np.float32)
        labels = np.where(np.arange(256)[:, None] % 2, 1, -1).astype(np.int8)
        summary = mm._accuracy(correct * (labels[None, :, None, :] > 0) * 2 - 1, labels)
        summary["repeats"] = 0
        args = SimpleNamespace(output=None, stop_file=Path("/definitely/not/a/STOP"),
                               deadline_unix=time.time() + 120)
        with tempfile.TemporaryDirectory() as td:
            args.output = Path(td)
            FakeReporter.instances.clear()
            with mock.patch.object(mm, "ExperimentReporter", FakeReporter), \
                    mock.patch.object(mm, "_validate_model", return_value=summary), \
                    mock.patch.object(mm, "_evaluate_paired_design", side_effect=mm.StopRequested("test stop")):
                result = mm._fit_one(config, 1, 0, args, "a" * 64, "b" * 64,
                                     "c" * 64, "partial-smoke")
            self.assertEqual(result["status"], "stopped_partial")
            self.assertTrue(result["heldout_opened"])
            self.assertEqual(result["evaluation"], [])
            fit_dir = Path(td) / "D1-seed0"
            self.assertTrue((fit_dir / "checkpoint.pt").is_file())
            self.assertTrue((fit_dir / "evaluation-design.json").is_file())
            self.assertEqual(json.loads((fit_dir / "status.json").read_text())["status"],
                             "stopped_partial")


if __name__ == "__main__":
    unittest.main()
