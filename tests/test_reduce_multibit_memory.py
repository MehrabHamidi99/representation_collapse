from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "analysis"))

from reduce_multibit_memory import Reducer  # noqa: E402


class IndependentReducerCorruptionTests(unittest.TestCase):
    def test_empty_campaign_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            report = Reducer(Path(td)).run()
            self.assertEqual(report["failed_checks"][0]["check"],
                             "all_four_seed0_fits_present")

    def test_wrong_inventory_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            fitdir = Path(td) / "D1-seed0"
            fitdir.mkdir()
            (fitdir / "result.json").write_text("{}\n")
            (fitdir / "raw.npz").write_bytes(b"synthetic artifact")
            result_sha = hashlib.sha256((fitdir / "result.json").read_bytes()).hexdigest()
            inventory = {
                "result.json": {"sha256": result_sha,
                                "bytes": (fitdir / "result.json").stat().st_size},
                "raw.npz": {"sha256": "0" * 64,
                            "bytes": (fitdir / "raw.npz").stat().st_size},
            }
            (fitdir / "artifact-inventory.json").write_text(
                json.dumps(inventory, sort_keys=True) + "\n")
            reducer = Reducer(Path(td))
            reducer.inventory(fitdir, fitdir.name)
            failures = {item["check"] for item in reducer.checks
                        if item["status"] == "fail"}
            self.assertIn("inventory:raw.npz", failures)
            self.assertNotIn("inventory:result.json", failures)

    def test_misaligned_raw_shape_is_rejected_before_summary_use(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fitdir = root / "D1-seed0"
            fitdir.mkdir()
            arrays = {
                "logits": np.zeros((4, 256, 2, 2), dtype=np.float32),
                "predictions": np.zeros((4, 256, 2, 2), dtype=np.uint8),
                "labels": np.zeros((4, 256, 2, 2), dtype=np.bool_),
                "bit_errors": np.zeros((4, 256, 2, 2), dtype=np.uint8),
                "oracle_predictions": np.zeros((4, 256, 2), dtype=np.uint8),
                "before_query": np.zeros((4, 256, 2, 128), dtype=np.float32),
                "after_write": np.zeros((4, 256, 2, 128), dtype=np.float32),
                "cue_bits": np.ones((256, 1), dtype=np.int8),
                "forcing_unit_ids": np.arange(4, dtype=np.int32),
                "arm_names": np.asarray(("periodic", "control")),
                "completed_cues": np.full(4, 256, dtype=np.int32),
                "complete": np.asarray(True),
                "state_layout": np.asarray("hidden_then_cell"),
            }
            np.savez(fitdir / "bad.npz", **arrays)
            reducer = Reducer(root)
            summary = reducer.reduce_eval(
                fitdir.name, fitdir,
                {"condition": "R32-q1", "raw_artifact": "bad.npz"}, 1, None)
            self.assertIsNone(summary)
            failures = [item for item in reducer.checks if item["status"] == "fail"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["check"], "R32-q1:logit_shape")


if __name__ == "__main__":
    unittest.main()
