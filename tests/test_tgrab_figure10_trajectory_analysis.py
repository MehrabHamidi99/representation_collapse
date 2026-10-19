"""Synthetic time-cloud calibration only; no empirical model evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from representation_collapse import tgrab_figure10_trajectory_analysis as analysis


def metadata(sha):
    return {"rows": [{"target_timestamp": t} for t in range(2816, 3072)],
        "state_trajectory": {"filename": "test-state-trajectory.f32le", "bytes": analysis.PAYLOAD_BYTES,
            "sha256": sha, "shape": list(analysis.SHAPE), "dtype": "float32_le", "order": "C",
            "components": ["hidden", "cell"], "node_ids": list(range(100)), "target_interval": [2816, 3072],
            "state_timing": "s[t]_before_target_score", "observations": 256, "centered_time_cloud_rank_ceiling": 255}}


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_known_rank_two_gram_and_absolute_variance(self):
        cloud = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)
        result = analysis.covariance_cloud(cloud)
        self.assertEqual(result["numerical_rank"], 2)
        self.assertEqual(result["time_rank_ceiling"], 2)
        self.assertAlmostEqual(result["mean_squared_centered_norm"], 1)
        self.assertAlmostEqual(result["sample_covariance_trace"], 4 / 3)
        self.assertAlmostEqual(result["participation_ratio"], 2)
        self.assertAlmostEqual(result["entropy_rank_natural_log"], 2)
        np.testing.assert_allclose(result["normalized_spectrum"], [.5, .5, 0, 0], atol=1e-15)

    def test_gram_spectrum_matches_independent_dense_svd_reference(self):
        rng = np.random.default_rng(73)
        cloud = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 5))
        centered = cloud - cloud.mean(axis=0)
        reference = np.linalg.svd(centered, compute_uv=False) ** 2 / 7
        result = analysis.covariance_cloud(cloud)
        self.assertEqual(result["numerical_rank"], 3)
        np.testing.assert_allclose(result["cleaned_eigenvalues_descending"][:3], reference[:3], rtol=1e-12, atol=1e-14)
        self.assertAlmostEqual(result["sample_covariance_trace"], float(reference.sum()))
        p = reference[:3] / reference[:3].sum()
        self.assertAlmostEqual(result["participation_ratio"], float(1 / np.sum(p * p)))
        self.assertAlmostEqual(result["entropy_rank_natural_log"], float(np.exp(-np.sum(p * np.log(p)))))

    def test_uniform_scaling_preserves_dimension_and_retains_tiny_absolute_variance(self):
        cloud = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float64)
        original = analysis.covariance_cloud(cloud)
        for scale in (1e-20, 1e-10, 1e10, 1e20):
            with self.subTest(scale=scale):
                result = analysis.covariance_cloud(cloud * scale)
                self.assertTrue(result["normalized_dimension_defined"])
                self.assertEqual(result["numerical_rank"], 2)
                self.assertAlmostEqual(result["participation_ratio"], original["participation_ratio"])
                self.assertAlmostEqual(result["entropy_rank_natural_log"], original["entropy_rank_natural_log"])
                self.assertTrue(math.isclose(result["sample_covariance_trace"], original["sample_covariance_trace"] * scale**2,
                                             rel_tol=1e-12, abs_tol=0))

    def test_constant_nonzero_state_has_undefined_dimension_not_zero_memory(self):
        result = analysis.covariance_cloud(np.full((32, 17), 3.25))
        self.assertFalse(result["normalized_dimension_defined"])
        self.assertIsNone(result["participation_ratio"])
        self.assertIsNone(result["entropy_rank_natural_log"])
        self.assertIsNone(result["normalized_spectrum"])
        self.assertEqual(result["numerical_rank"], 0)
        self.assertEqual(result["sample_covariance_trace"], 0)
        self.assertGreater(result["mean_state_squared_norm"], 0)

    def test_material_negative_eigenvalue_fails_instead_of_silent_clipping(self):
        with patch("numpy.linalg.eigvalsh", return_value=np.asarray([-0.01, 1.0])):
            with self.assertRaisesRegex(analysis.TrajectoryError, "negative tolerance"):
                analysis.covariance_cloud(np.asarray([[0.], [1.]]))
        with self.assertRaises(analysis.TrajectoryError):
            analysis.covariance_cloud(np.asarray([[float("nan")], [1.]]))

    def test_cell_only_information_and_matched_window_counts(self):
        states = np.zeros(analysis.SHAPE, dtype="<f4")
        states[:, 0] = 3.0
        states[:, 1, 0, 0, 0] = np.arange(256, dtype=np.float32) % 32
        result = analysis.analyze_trajectory(states)
        all_views = result["clouds"]["all_targets"]["views"]
        self.assertFalse(all_views["hidden"]["normalized_dimension_defined"])
        self.assertTrue(all_views["cell"]["normalized_dimension_defined"])
        self.assertEqual(all_views["cell"]["numerical_rank"], 1)
        self.assertEqual(all_views["hidden_and_cell"]["numerical_rank"], 1)
        self.assertAlmostEqual(all_views["hidden_and_cell"]["sample_covariance_trace"], all_views["cell"]["sample_covariance_trace"])
        self.assertEqual(all_views["hidden_and_cell"]["time_rank_ceiling"], 255)
        for motif in ("A", "B", "pooled"):
            for window, first, last in (("early", 1, 8), ("late", 24, 31)):
                record = result["clouds"][motif + "_" + window]
                count = 64 if motif == "pooled" else 32
                self.assertEqual(len(record["target_timestamps"]), count)
                self.assertTrue(all(first <= t % 32 <= last for t in record["target_timestamps"]))
                for view in record["views"].values():
                    self.assertEqual(view["samples"], count)
                    self.assertEqual(view["time_rank_ceiling"], count - 1)
        self.assertTrue(result["period64_complete_state_closure"]["all_complete_states_bitwise_equal"])
        self.assertFalse(result["scientific_collapse_claim"])

    def test_closure_order_complete_cell_state_and_signed_zero_bits(self):
        states = np.zeros(analysis.SHAPE, dtype="<f4")
        states[64, 1, 0, 0, 0] = 2.0
        closure = analysis.period_closure(states)
        first = closure["comparisons"][0]
        self.assertEqual(first["reference_target_timestamp"], 2816)
        self.assertEqual(first["next_target_timestamp"], 2880)
        self.assertEqual(first["absolute_l2_residual"], 2.0)
        self.assertEqual(first["relative_l2_residual"], 1.0)
        self.assertFalse(first["exact_bitwise_equal"])
        self.assertEqual(closure["comparison_count"], 192)
        self.assertFalse(closure["floquet_or_jacobian_periodicity_certified"])
        states.fill(0)
        states[64, 1, 0, 0, 0] = -0.0
        signed = analysis.period_closure(states)["comparisons"][0]
        self.assertEqual(signed["absolute_l2_residual"], 0.0)
        self.assertEqual(signed["relative_l2_residual"], 0.0)
        self.assertFalse(signed["exact_bitwise_equal"])
        with self.assertRaises(analysis.TrajectoryError):
            analysis.period_closure(states.astype(">f4"))

    def test_descriptor_requires_exact_shape_node_component_order_timing_and_targets(self):
        sha = hashlib.sha256(b"synthetic").hexdigest()
        baseline = metadata(sha)
        analysis.validate_descriptor(baseline, "test-state-trajectory.f32le", sha)
        for field, wrong in (("shape", [256, 1, 2, 100, 128]), ("components", ["cell", "hidden"]),
                             ("node_ids", list(reversed(range(100)))), ("bytes", analysis.PAYLOAD_BYTES - 4),
                             ("state_timing", "after_target_consume"), ("observations", True)):
            changed = copy.deepcopy(baseline); changed["state_trajectory"][field] = wrong
            with self.subTest(field=field), self.assertRaises(analysis.TrajectoryError):
                analysis.validate_descriptor(changed, "test-state-trajectory.f32le", sha)
        changed = copy.deepcopy(baseline); changed["rows"][0]["target_timestamp"] = 2817
        with self.assertRaises(analysis.TrajectoryError):
            analysis.validate_descriptor(changed, "test-state-trajectory.f32le", sha)

    def test_binary_loading_checks_hash_length_layout_and_finiteness(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test-state-trajectory.f32le"
            states = np.zeros(analysis.SHAPE, dtype="<f4")
            states[0, 0, 0, 0, 0] = 1.0; states[-1, 1, 0, 99, 127] = 2.0
            path.write_bytes(states.tobytes())
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = analysis.load_trajectory(path, sha)
            self.assertEqual(loaded[0, 0, 0, 0, 0], 1.0)
            self.assertEqual(loaded[-1, 1, 0, 99, 127], 2.0)
            with self.assertRaises(analysis.TrajectoryError):
                analysis.load_trajectory(path, hashlib.sha256(b"wrong").hexdigest())
            path.write_bytes(b"short")
            with self.assertRaises(analysis.TrajectoryError):
                analysis.load_trajectory(path, hashlib.sha256(b"short").hexdigest())
            states[0, 0, 0, 0, 0] = np.inf; raw = states.tobytes(); path.write_bytes(raw)
            with self.assertRaisesRegex(analysis.TrajectoryError, "nonfinite"):
                analysis.load_trajectory(path, hashlib.sha256(raw).hexdigest())

    def test_cli_never_reads_binary_when_reporter_initialization_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha = hashlib.sha256(b"unopened").hexdigest()
            path = root / "evaluation-result.json"; path.write_text(json.dumps(metadata(sha)))
            metadata_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            output = root / "analysis"
            with patch("representation_collapse.reporting.ExperimentReporter") as reporter, patch.object(analysis, "load_trajectory") as loader, patch("builtins.print"):
                reporter.return_value.__enter__.side_effect = RuntimeError("offline initialization failed")
                status = analysis.main(["--evaluation-result", str(path), "--evaluation-result-sha256", metadata_sha,
                    "--trajectory", str(root / "test-state-trajectory.f32le"), "--trajectory-sha256", sha,
                    "--output", str(output), "--run-id", "synthetic-test-only"])
                self.assertEqual(status, 2)
                loader.assert_not_called()
            self.assertEqual(json.loads((output / "analysis-manifest.json").read_text())["status"], "failed")


    def test_cli_reports_analysis_hash_and_rejects_failed_finish(self):
        for finish_failure in (False, True):
            with self.subTest(finish_failure=finish_failure), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                sha = hashlib.sha256(b"unopened-synthetic").hexdigest()
                path = root / "evaluation-result.json"; path.write_text(json.dumps(metadata(sha)))
                mh = hashlib.sha256(path.read_bytes()).hexdigest()
                output = root / "analysis"
                events = []
                class Reporter:
                    active = False
                    def __init__(self, *args, **kwargs):
                        pass
                    def __enter__(self):
                        self.active = True
                        return self
                    def __exit__(self, *args):
                        self.active = False
                        if finish_failure:
                            raise RuntimeError("finish failed")
                    def log(self, values, *, step):
                        events.append(("log", step))
                    def log_manifest(self, name, *, metadata, identities):
                        self_test.assertTrue(self.active)
                        self_test.assertEqual(identities["analysis_sha256"], hashlib.sha256((output / "trajectory-analysis.json").read_bytes()).hexdigest())
                        self_test.assertFalse(metadata["binary_state_uploaded"])
                        events.append(("manifest", name))
                self_test = self
                result = {"clouds": {}, "period64_complete_state_closure": {"maximum_absolute_l2_residual": 0.,
                    "maximum_relative_l2_residual": 0., "all_complete_states_bitwise_equal": True}}
                with patch("representation_collapse.reporting.ExperimentReporter", Reporter), patch.object(analysis, "load_trajectory", return_value=object()), patch.object(analysis, "analyze_trajectory", return_value=result), patch("builtins.print"):
                    status = analysis.main(["--evaluation-result", str(path), "--evaluation-result-sha256", mh,
                        "--trajectory", str(root / "test-state-trajectory.f32le"), "--trajectory-sha256", sha,
                        "--output", str(output), "--run-id", "synthetic-analysis-test"])
                self.assertEqual(status, 2 if finish_failure else 0)
                terminal = json.loads((output / "analysis-manifest.json").read_text())
                self.assertEqual(terminal["run_id"], "synthetic-analysis-test")
                self.assertEqual(terminal["source_shape"], list(analysis.SHAPE))
                self.assertEqual(terminal["status"], "failed" if finish_failure else "completed")
                self.assertEqual(events[-1], ("manifest", "trajectory-analysis-result"))


if __name__ == "__main__":
    unittest.main()
