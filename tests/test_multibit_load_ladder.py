from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from representation_collapse import multibit_load_ladder as ladder
from representation_collapse import multibit_memory as v1

ROOT = Path(__file__).resolve().parents[1]


class LocalReporter:
    """Keep lifecycle tests local; reporting integration is checked separately."""
    def __init__(self, output, **kwargs):
        self.output = output

    def __enter__(self):
        self.output.mkdir(parents=True)
        return self

    def __exit__(self, *args):
        return False

    def log(self, *args, **kwargs):
        pass

    def log_manifest(self, *args, **kwargs):
        pass


class LoadLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        v1._configure_torch(torch)
        cls.torch = torch

    def test_config_and_long_horizon_design_are_frozen(self):
        config = json.loads((ROOT / "configs/experiments/multibit-load-ladder-v2.json").read_text())
        self.assertEqual(ladder.validate_config(config), ladder.default_config())
        self.assertEqual([c for c in config["test_conditions"] if ladder._condition_horizon(*c) == 6144],
                         [[512, 1], [128, 4], [256, 2]])
        altered = dict(config)
        altered["test_conditions"] = config["test_conditions"][:-1]
        with self.assertRaises(ValueError):
            ladder.validate_config(altered)

    def test_independent_rng_streams_have_v2_namespace(self):
        values = {ladder.derive_seed(16, 1, name) for name in ("train", "validation", "test")}
        self.assertEqual(len(values), 3)
        self.assertNotIn(v1.derived_seed(16, 1, "validation"), values)
        self.assertEqual(ladder.derive_seed(16, 1, "test"),
                         ladder.derive_seed(16, 1, "test"))

    def test_validation_raw_count_axes_and_polarity_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits, designs = v1.validation_design(np.random.default_rng(19), 8, 4, [0])
            model = v1._make_model(self.torch).cpu().float()
            args = v1._Args(root / "STOP", 10**12, root)
            raw = root / "stage.npz"
            report = ladder._validation_stage_save(model, self.torch, bits, designs, 4, [0], args,
                                                   raw, root / "partial.npz")
            check = report["checks"][0]
            with np.load(raw, allow_pickle=False) as payload:
                self.assertEqual(payload["logits_R0"].shape, (8, 2, 4))
                self.assertEqual(payload["polarity_correct_counts_R0"].shape, (2, 4, 2))
                np.testing.assert_array_equal(payload["bit_correct_counts_R0"],
                                              payload["correct_R0"].sum(axis=0))
                self.assertEqual(payload["polarity_denominators_R0"].tolist(), [[4, 4]] * 4)
                self.assertTrue(bool(payload["complete"].item()))
            minimum = min(v for arms in check["per_bit_polarity_accuracy_by_arm"].values()
                          for polarities in arms.values() for v in polarities.values())
            self.assertEqual(check["all_active_bits_polarities_arms_min_accuracy"], minimum)
            self.assertEqual(v1.competence_pass(check, 0.95), minimum >= 0.95)

    def test_stop_during_validation_persists_partial_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits, designs = v1.validation_design(np.random.default_rng(3), 8, 4, [0])
            stop = root / "STOP"
            stop.touch()
            args = v1._Args(stop, 10**12, root)
            args.validation_partial_path = root / "partial.npz"
            with self.assertRaises(v1.StopRequested):
                ladder._validation_stage(v1._make_model(self.torch), self.torch, bits, designs,
                                         4, [0], args)
            with np.load(args.validation_partial_path, allow_pickle=False) as payload:
                self.assertFalse(bool(payload["complete_R0"].item()))
                self.assertEqual(payload["completed_cues_R0"].item(), 0)

    def test_stop_during_endpoint_replay_is_partial_not_engineering_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bits, designs = v1.validation_design(np.random.default_rng(4), 8, 4, [0])
            stop = root / "STOP"
            stop.touch()
            args = v1._Args(stop, 10**12, root)
            args.validation_partial_path = root / "endpoint-replay.partial.npz"
            validation, arrays = ladder._endpoint_validation_replay(
                v1._make_model(self.torch), self.torch, bits, designs, 4, [0], args)
            self.assertIsNone(validation)
            self.assertEqual(arrays, {})
            self.assertTrue(args.validation_partial_path.exists())
            with np.load(args.validation_partial_path, allow_pickle=False) as payload:
                self.assertFalse(bool(payload["complete_R0"].item()))

    def test_failed_competence_gate_runs_no_heldout_evaluation(self):
        # Exercise the real fit lifecycle with a zero-update synthetic stage. This is
        # an infrastructure test only; the random model is expected to fail competence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ladder.default_config()
            config.update(dimensions=[4], seeds=[0], validation_cue_vectors=8,
                          test_cue_vectors=8, test_forcing_units=1,
                          test_conditions=[[0, 1]], test_chunk=8)
            config["stages"] = [{"updates": 0, "train_repeats": [0],
                                "validate_repeats": [0]}]
            args = v1._Args(root / "STOP", 10**12, root)
            args.output = root
            with mock.patch.object(ladder, "ExperimentReporter", LocalReporter):
                result = ladder._fit_one(
                    config, 4, 0, args, run_id="unit-test", source_sha="a" * 64,
                    primitive_sha="b" * 64, protocol_sha="c" * 64, config_sha="d" * 64)
            fit = root / "D4"
            self.assertEqual(result["status"], "competence_failed")
            self.assertFalse(result["heldout_opened"])
            self.assertEqual(result["evaluation"], [])
            self.assertFalse((fit / "evaluation-design.json").exists())
            self.assertTrue((fit / "stage-1-validation.npz").exists())

    def test_native_checkpoint_replay_and_heldout_branch(self):
        # Gate mocking exercises the successful lifecycle, not learned competence.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ladder.default_config()
            config.update(validation_cue_vectors=8, test_cue_vectors=8,
                          test_forcing_units=1, test_conditions=[[0, 1]], test_chunk=8)
            config["stages"] = [{"updates": 0, "train_repeats": [0],
                                 "validate_repeats": [0]} for _ in range(3)]
            args = v1._Args(root / "STOP", 10**12, root)
            with mock.patch.object(ladder, "ExperimentReporter", LocalReporter), \
                 mock.patch.object(v1, "competence_pass", return_value=True):
                result = ladder._fit_one(
                    config, 4, 0, args, run_id="unit-roundtrip", source_sha="a" * 64,
                    primitive_sha="b" * 64, protocol_sha="c" * 64, config_sha="d" * 64)
            self.assertTrue(result["endpoint_replay"]["passed"])
            self.assertTrue(result["heldout_opened"])
            self.assertEqual(len(result["evaluation"]), 1)
            self.assertEqual(len(list((root / "D4").glob("stage-*-checkpoint.pt"))), 3)
            with np.load(root / "D4" / "stage-3-validation.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["polarity_names"].tolist(), ["positive", "negative"])

    def test_campaign_gate_does_not_accept_missing_or_failed_stage(self):
        self.assertTrue(ladder._all_gates_passed([{"passed": True}] * 3, 3))
        self.assertFalse(ladder._all_gates_passed([{"passed": True}] * 2, 3))
        self.assertFalse(ladder._all_gates_passed(
            [{"passed": True}, {"passed": False}, {"passed": True}], 3))


if __name__ == "__main__":
    unittest.main()
