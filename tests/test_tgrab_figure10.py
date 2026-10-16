import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from representation_collapse import graph_reconstruction as gr
from representation_collapse.tgrab_figure10 import (
    EXPECTED_SPLITS, FilePin, Figure10Blocked, per_target_f1_with_predecessor,
    test_reduction, validate_primary_contract, run,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def snap(t, edges):
    return gr.Snapshot(t, 100, tuple(edges), tuple(range(100)))


class Figure10ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/graph/tgrab-reconstruction-v1.json").read_text())
        self.pattern_a = ((0, 1), (1, 0))
        self.pattern_b = ((0, 2), (2, 0))
        self.manifest = {
            "kind": "tgrab_primary_converted_v1",
            "config_sha256": "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f",
            "generation_performed": False,
            "spec": {"nodes": 100, "block": 32, "train_periods": 40,
                     "validation_periods": 4, "test_periods": 4},
            "splits": {k: {"interval": v} for k, v in EXPECTED_SPLITS.items()},
            "snapshot_semantic_sha256": ["0" * 64] * 3072,
            "pattern_pairs": [self.pattern_a, self.pattern_b],
        }

    def test_frozen_identity_and_exact_split_are_required(self):
        validate_primary_contract(self.config, self.manifest)
        changed = json.loads(json.dumps(self.manifest))
        changed["splits"]["test"]["interval"] = [2815, 3072]
        with self.assertRaisesRegex(Figure10Blocked, "expected source intervals"):
            validate_primary_contract(self.config, changed)
        altered_config = json.loads(json.dumps(self.config))
        altered_config["candidate"]["model_seed"] = 7
        with self.assertRaises(gr.ContractError):
            validate_primary_contract(altered_config, self.manifest)

    def test_per_target_f1_uses_fixed_exhaustive_masks_and_threshold(self):
        predecessor = snap(2815, self.pattern_b)
        targets = [snap(t, self.pattern_a if (t // 32) % 2 == 0 else self.pattern_b)
                   for t in range(2816, 3072)]
        scores = np.zeros((256, 9900), dtype=np.float32)
        pairs = gr.ordered_offdiagonal_pairs(100)
        # At t=2816, classify exactly the two positive directed edges at p=.5.
        for pair in self.pattern_a:
            scores[0, pairs.index(pair)] = 0.5
        rows = per_target_f1_with_predecessor(predecessor, targets, scores)
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0]["tp"], 2)
        self.assertEqual(rows[0]["fp"], 0)
        self.assertEqual(rows[0]["fn"], 0)
        self.assertEqual(rows[0]["f1"], 1.0)
        summary = test_reduction(rows)
        self.assertEqual(summary["targets"], 256)
        self.assertEqual(len(summary["per_target_f1"]), 256)
        self.assertEqual(sum(r["phase_transition_expected"] for r in rows), 8)
        self.assertEqual(sum(r["change_point"] for r in rows), 8)
        self.assertEqual(sum(r["pairs"] for r in rows), 256 * 9900)

    def test_metric_rejects_wrong_shape_chronology_and_nonfinite(self):
        predecessor = snap(2815, self.pattern_b)
        targets = [snap(t, self.pattern_a if (t // 32) % 2 == 0 else self.pattern_b)
                   for t in range(2816, 3072)]
        with self.assertRaises(Figure10Blocked):
            per_target_f1_with_predecessor(predecessor, targets[:-1], np.zeros((255, 9900)))
        targets[1] = snap(2818, self.pattern_a)
        with self.assertRaisesRegex(Figure10Blocked, "contiguous"):
            per_target_f1_with_predecessor(predecessor, targets, np.zeros((256, 9900)))
        targets[1] = snap(2817, self.pattern_b)
        scores = np.zeros((256, 9900)); scores[4, 0] = np.nan
        with self.assertRaisesRegex(Figure10Blocked, "finite"):
            per_target_f1_with_predecessor(predecessor, targets, scores)

    def test_reduction_requires_phase_boundary_to_match_actual_graph_change(self):
        predecessor = snap(2815, self.pattern_b)
        targets = [snap(t, self.pattern_a if (t // 32) % 2 == 0 else self.pattern_b)
                   for t in range(2816, 3072)]
        rows = per_target_f1_with_predecessor(
            predecessor, targets, np.zeros((256, 9900)))
        rows[0] = dict(rows[0], change_point=False)
        with self.assertRaisesRegex(Figure10Blocked, "disagrees with actual"):
            test_reduction(rows)

    def test_reduction_rejects_nonboolean_transition_flags(self):
        predecessor = snap(2815, self.pattern_b)
        targets = [snap(t, self.pattern_a if (t // 32) % 2 == 0 else self.pattern_b)
                   for t in range(2816, 3072)]
        rows = per_target_f1_with_predecessor(
            predecessor, targets, np.zeros((256, 9900)))
        rows[0] = dict(rows[0], phase_transition_expected=1)
        with self.assertRaisesRegex(Figure10Blocked, "must be booleans"):
            test_reduction(rows)

    def test_safe_runner_fails_closed_without_opening_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config_path = ROOT / "configs/graph/tgrab-reconstruction-v1.json"
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text("{}\n")
            stop = tmp / "STOP"
            pin = lambda path: FilePin(path, hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(Figure10Blocked, "prospective reconstruction blocked"):
                run(config_pin=pin(config_path), data_manifest_pin=pin(manifest_path),
                    stop_file=stop, deadline_unix=4_000_000_000,
                    source_root=ROOT / "runs/graph/upstream-pin-20261012/source",
                    readiness=gr.Readiness())

    def test_cli_publishes_only_an_atomic_blocked_readiness_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config_path = ROOT / "configs/graph/tgrab-reconstruction-v1.json"
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text("{}\n")
            report = tmp / "new" / "readiness.json"
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            code = main([
                "--config", str(config_path), "--config-sha256", digest(config_path),
                "--data-manifest", str(manifest_path), "--data-manifest-sha256", digest(manifest_path),
                "--source-root", str(ROOT / "runs/graph/upstream-pin-20261012/source"),
                "--stop-file", str(tmp / "STOP"), "--deadline-unix", "4000000000",
                "--report", str(report),
            ])
            self.assertEqual(code, 2)
            receipt = json.loads(report.read_text())
            self.assertEqual(receipt["status"], "blocked")
            self.assertFalse(receipt["training_started"])
            self.assertFalse(receipt["test_payload_opened"])
            self.assertFalse(receipt["model_constructed"])
            self.assertFalse(receipt["job_submitted"])
            self.assertIn("launch_disabled_by_frozen_configuration", receipt["blocker"])
            with self.assertRaises(FileExistsError):
                main([
                    "--config", str(config_path), "--config-sha256", digest(config_path),
                    "--data-manifest", str(manifest_path), "--data-manifest-sha256", digest(manifest_path),
                    "--source-root", str(ROOT / "runs/graph/upstream-pin-20261012/source"),
                    "--stop-file", str(tmp / "STOP"), "--deadline-unix", "4000000000",
                    "--report", str(report),
                ])


if __name__ == "__main__":
    unittest.main()
