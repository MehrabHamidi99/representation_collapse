"""Analytic synthetic controls; no project data or split payload I/O."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import sys

MODULE = Path(__file__).resolve().parents[1] / "code/figure10_deterministic_baselines.py"
spec = importlib.util.spec_from_file_location("figure10_baselines", MODULE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class DeterministicBaselineTests(unittest.TestCase):
    def test_cli_uses_only_manifest_metadata_and_refuses_output_reuse(self):
        sys.path.insert(0, str(MODULE.parents[1] / "src"))
        from representation_collapse import graph_data as gd
        from representation_collapse import reporting
        metadata = {"kind": "tgrab_primary_converted_v1", "spec": gd.PRIMARY_SPEC.as_dict(),
                    "converter": {"sha256": baseline.CONVERTER_SHA256, "revision": "a" * 40},
                    "pattern_pairs": [[[0, 1], [1, 0]], [[2, 3], [3, 2]]],
                    "raw": {"sha256": gd.RAW_SHA256}, "config_sha256": gd.CONFIG_SHA256}
        events, metrics, artifacts, settings = [], [], [], []
        class FakeReporter:
            def __init__(self, output, **kwargs):
                settings.append((output, kwargs))
            def __enter__(self):
                events.append("entered")
                return self
            def __exit__(self, *args):
                events.append("finished")
            def log(self, values, step):
                metrics.append((step, values))
            def log_manifest(self, name, **kwargs):
                artifacts.append((name, kwargs))
        def read(*args):
            self.assertEqual(events, ["entered"])
            events.append("read_manifest")
            return SimpleNamespace(metadata=metadata)
        with tempfile.TemporaryDirectory() as td, patch.object(gd, "read_manifest", side_effect=read) as reader, patch.object(gd, "_load_split", side_effect=AssertionError("split access prohibited")) as split, patch.object(reporting, "ExperimentReporter", FakeReporter):
            output = Path(td) / "controls.json"
            argv = ["--manifest", "/unused/manifest.json", "--manifest-sha256", "b" * 64, "--output", str(output)]
            with patch("builtins.print"):
                self.assertEqual(baseline.main(argv), 0)
            self.assertTrue(output.exists())
            reader.assert_called_once()
            split.assert_not_called()
            self.assertEqual(events, ["entered", "read_manifest", "finished"])
            self.assertEqual(settings[0][1]["mode"], "offline")
            self.assertEqual(settings[0][1]["project"], "representation-collapse")
            self.assertEqual(settings[0][0], output.with_name("controls.wandb"))
            self.assertEqual([step for step, _ in metrics], list(range(771)))
            self.assertEqual(sum("control/f1" in values for _, values in metrics), 768)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0][1]["identities"]["result_sha256"], baseline.hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                baseline.main(argv)

    def test_reporting_failure_publishes_no_numerical_result(self):
        sys.path.insert(0, str(MODULE.parents[1] / "src"))
        from representation_collapse import reporting, graph_data as gd
        with tempfile.TemporaryDirectory() as td, patch.object(reporting, "ExperimentReporter") as reporter, patch.object(gd, "read_manifest") as reader:
            reporter.return_value.__enter__.side_effect = RuntimeError("reporter failed")
            output = Path(td) / "failed.json"
            with self.assertRaisesRegex(RuntimeError, "reporter"):
                baseline.main(["--manifest", "/unused/manifest.json", "--manifest-sha256", "b" * 64, "--output", str(output)])
            self.assertFalse(output.exists())
            reader.assert_not_called()

    def test_finish_failure_publishes_no_completed_result(self):
        sys.path.insert(0, str(MODULE.parents[1] / "src"))
        from representation_collapse import reporting, graph_data as gd
        metadata = {"kind": "tgrab_primary_converted_v1", "spec": gd.PRIMARY_SPEC.as_dict(),
                    "converter": {"sha256": baseline.CONVERTER_SHA256, "revision": "a" * 40},
                    "pattern_pairs": [[[0, 1], [1, 0]], [[2, 3], [3, 2]]],
                    "raw": {"sha256": gd.RAW_SHA256}, "config_sha256": gd.CONFIG_SHA256}
        with tempfile.TemporaryDirectory() as td, patch.object(reporting, "ExperimentReporter") as reporter, patch.object(gd, "read_manifest", return_value=SimpleNamespace(metadata=metadata)):
            reporter.return_value.__exit__.side_effect = RuntimeError("finish failed")
            output = Path(td) / "failed.json"
            with self.assertRaisesRegex(RuntimeError, "finish"):
                baseline.main(["--manifest", "/unused/manifest.json", "--manifest-sha256", "b" * 64, "--output", str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(reporter.return_value.__enter__.return_value.log.call_count, 771)
    def test_disjoint_patterns_persistence_transition_and_block_means(self):
        result = baseline.calculate([[[0, 1], [1, 0]], [[1, 2], [2, 1]]], nodes=3)
        persistence = result["controls"]["previous_graph_persistence"]
        rows = persistence["rows"]
        self.assertEqual([r["target_timestamp"] for r in rows], list(range(2816, 3072)))
        self.assertEqual([r["target_timestamp"] for r in rows if r["change_point"]], list(range(2816, 3072, 32)))
        self.assertEqual(rows[0]["source_timestamp"], 2815)
        self.assertEqual((rows[0]["tp"], rows[0]["fp"], rows[0]["fn"], rows[0]["tn"]), (0, 2, 2, 2))
        self.assertEqual(persistence["reduction"]["primary_time_mean_f1"], 31 / 32)
        self.assertEqual(persistence["reduction"]["transition_time_mean_f1"], 0.0)
        self.assertEqual(persistence["reduction"]["nontransition_time_mean_f1"], 1.0)
        self.assertEqual([b["time_mean_f1"] for b in persistence["reduction"]["blocks"]], [31 / 32] * 8)
        self.assertEqual(result["controls"]["known_generator_phase_counter"]["reduction"]["primary_time_mean_f1"], 1.0)
        self.assertEqual(result["controls"]["always_negative"]["reduction"]["primary_time_mean_f1"], 0.0)

    def test_overlapping_unequal_patterns_correct_directional_confusion(self):
        result = baseline.calculate([[[0, 1], [1, 0]], [[0, 1], [1, 0], [1, 2], [2, 1]]], nodes=3)
        rows = result["controls"]["previous_graph_persistence"]["rows"]
        # B -> A has extra predicted edges; A -> B has missed edges.
        self.assertEqual((rows[0]["tp"], rows[0]["fp"], rows[0]["fn"]), (2, 2, 0))
        self.assertEqual((rows[32]["tp"], rows[32]["fp"], rows[32]["fn"]), (2, 0, 2))
        self.assertAlmostEqual(result["controls"]["previous_graph_persistence"]["reduction"]["transition_time_mean_f1"], 2 / 3)
        self.assertTrue(all(r["pairs"] == 6 for r in rows))

    def test_bad_patterns_reject(self):
        good = [[0, 1], [1, 0]]
        for bad in ([[0, 0]], [[0, 1], [1, 0], [0, 1]], [[0, 3], [3, 0]], [[0, 1]], [[False, 1], [1, False]], []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                baseline.calculate([good, bad], nodes=3)
        with self.assertRaises(ValueError):
            baseline.calculate([good, list(reversed(good))], nodes=3)

    def test_default_100_nodes_uses_all_9900_ordered_offdiagonal_pairs(self):
        result = baseline.calculate([[[0, 1], [1, 0]], [[2, 3], [3, 2]]])
        self.assertEqual(result["ordered_offdiagonal_pairs"], 9900)
        for control in result["controls"].values():
            for row in control["rows"]:
                self.assertEqual(sum(row[k] for k in ("tp", "fp", "fn", "tn")), 9900)


if __name__ == "__main__":
    unittest.main()
