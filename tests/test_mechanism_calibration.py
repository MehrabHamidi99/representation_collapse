import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from representation_collapse.mechanism_calibration import (
    calibration,
    conditioned_dimension,
    linear_recovery_risk,
    propagate_affine,
    propagated_noise_covariance,
)
from representation_collapse.metrics import covariance_dimension


class MechanismCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics, cls.checks, cls.arrays = calibration(seed=11)

    def test_affine_body_order_changes_offset_not_conditioned_cue(self):
        cues = np.array([[-2.0, 0], [1, 2], [0.3, -1]])
        encoded = cues**3
        steps = np.array([np.diag([0.5, 0.8])] * 2)
        first = propagate_affine(encoded, steps, np.array([[1, 0], [0, 2]]))
        second = propagate_affine(encoded, steps, np.array([[0, 2], [1, 0]]))
        expected_cue = encoded * np.array([0.25, 0.64])
        np.testing.assert_allclose(first - [0.5, 2], expected_cue, atol=1e-12, rtol=0)
        np.testing.assert_allclose(second - [1, 1.6], expected_cue, atol=1e-12, rtol=0)
        np.testing.assert_allclose(first - first.mean(0), second - second.mean(0), atol=1e-12, rtol=0)
        m = self.metrics["affine_ordering_null"]
        self.assertGreater(abs(m["native_linear_mse"][0] - m["native_linear_mse"][1]), 1e-3)
        self.assertGreater(abs(m["native_nonlinear_mse"][0] - m["native_nonlinear_mse"][1]), 1e-3)
        self.assertLess(m["offset_corrected_linear_mse"], 1e-24)

    def test_wrong_conditioning_is_rejected_and_creates_spurious_variance(self):
        first = np.array([[-1.0, 0], [1, 0]])
        second = first + [0, 10]
        pooled = np.concatenate([first, second])
        self.assertEqual(conditioned_dimension(first, np.zeros(2))["numerical_rank"], 1)
        self.assertEqual(covariance_dimension(pooled)["numerical_rank"], 2)
        with self.assertRaisesRegex(ValueError, "pooled-body"):
            conditioned_dimension(pooled, np.array([0, 0, 1, 1]))
        with self.assertRaises(ValueError):
            conditioned_dimension(first, np.array([0]))

    def test_commuting_product_null_does_not_imply_process_noise_null(self):
        a, b = np.diag([0.5, 0.8]), np.diag([0.9, 0.4])
        ab = propagated_noise_covariance(np.array([a, b]), np.eye(2))
        ba = propagated_noise_covariance(np.array([b, a]), np.eye(2))
        np.testing.assert_allclose(ab, np.diag([1.81, 1.16]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(ba, np.diag([1.25, 1.64]), atol=1e-12, rtol=0)
        np.testing.assert_allclose(self.arrays["commuting_products"][0],
                                   self.arrays["commuting_products"][1], atol=1e-12, rtol=0)
        self.assertGreater(np.linalg.norm(ab - ba), 0.7)

    def test_noncommuting_positive_construction_has_order_specific_risk(self):
        a, b = self.arrays["noncommuting_symbols"]
        products = self.arrays["noncommuting_products"]
        np.testing.assert_allclose(products[0], b @ b @ a @ a, atol=1e-12, rtol=0)
        np.testing.assert_allclose(products[1], b @ a @ b @ a, atol=1e-12, rtol=0)
        self.assertGreater(np.linalg.norm(products[0] - products[1]), 0.01)
        # Terminal-noise covariance is identical; only propagation order differs.
        self.assertGreater(abs(linear_recovery_risk(products[0], 0.04 * np.eye(2)) -
                               linear_recovery_risk(products[1], 0.04 * np.eye(2))), 0.001)

    def test_noise_location_and_whitening_in_linear_channel(self):
        channel = np.diag([2.0, 0.5])
        noise = np.diag([4.0, 0.25])
        # Both squared signal/noise ratios are one, hence two residual variances 1/2.
        self.assertAlmostEqual(linear_recovery_risk(channel, noise), 1.0)
        scale = np.diag([10.0, 0.1])
        self.assertAlmostEqual(linear_recovery_risk(scale @ channel, scale @ noise @ scale.T), 1.0)
        self.assertAlmostEqual(linear_recovery_risk(np.zeros((2, 3)), np.eye(2)), 3.0)

    def test_isotropic_attenuation_changes_amplitude_not_normalized_dimension(self):
        m = self.metrics["amplitude_only"]
        for horizon, dimension in zip(m["horizons"], m["dimensions"]):
            self.assertAlmostEqual(dimension["participation_ratio"], 4.0)
            self.assertAlmostEqual(dimension["total_variance"], 4.0 * 0.8**(2 * horizon))
        self.assertGreater(m["terminal_noise_linear_mse"][-1], m["terminal_noise_linear_mse"][0])

    def test_local_contracting_forced_basins_preserve_finite_memory(self):
        m = self.metrics["categorical_forced_basins"]
        a = self.arrays
        labels = a["categorical_labels"]
        phases = a["categorical_phase_offsets"]
        cycles = a["categorical_reference_cycles"]
        # Verify the actual forced map closes each of the two distinct references.
        for phase in range(2):
            relative = cycles[:, phase] - phases[phase]
            following = 1.5 * relative - 0.5 * relative**3 + phases[(phase + 1) % 2]
            np.testing.assert_array_equal(following, cycles[:, (phase + 1) % 2])
        np.testing.assert_array_equal(np.sign(a["categorical_trajectory"][-1] - phases[0]), labels)
        self.assertLess(m["neighborhood_absolute_derivative_bound"], 0.16)
        self.assertLess(m["neighborhood_next_error_bound"], 0.05)
        self.assertEqual(m["classification_accuracy"], 1.0)

    def test_continuous_required_memory_survives_nuisance_compression(self):
        a, m = self.arrays, self.metrics["continuous_neutral_memory"]
        np.testing.assert_array_equal(a["continuous_states"][:, :2], a["cues"][:, :2])
        np.testing.assert_allclose(a["continuous_states"][:, 2:], a["cues"][:, 2:] / 65536,
                                   atol=0, rtol=0)
        self.assertEqual(m["required_mse"], 0)
        self.assertAlmostEqual(m["after_dimension"]["participation_ratio"], 2.0, places=8)

    def test_fresh_noise_keeps_geometry_without_old_linear_memory(self):
        a, m = self.arrays, self.metrics["fresh_noise_not_old_memory"]
        n = len(a["noise_cues"])
        k = (n - 1) / n
        gains = np.array(m["cue_gains"])
        v = 1 - gains**2
        expected_gain = k * gains / (k * gains**2 + v)
        np.testing.assert_allclose(a["noise_decoder_gains"], expected_gain, atol=1e-14, rtol=0)
        np.testing.assert_allclose(m["analytic_linear_decoder_mse"], 4 * k * v / (k * gains**2 + v),
                                   atol=1e-14, rtol=0)
        state = a["noise_cues"].copy()
        recorded = {h: i for i, h in enumerate(m["horizons"])}
        for time, innovation in enumerate(a["noise_innovations"], start=1):
            state = 0.8 * state + innovation
            if time in recorded:
                np.testing.assert_array_equal(state, a["noise_states"][recorded[time]])
        self.assertGreater(m["empirical_population_decoder_mse"][-1], 3.8)
        self.assertLess(abs(m["dimensions"][-1]["total_variance"] - 4), 0.2)
        self.assertGreater(m["dimensions"][-1]["participation_ratio"], 3.9)

    def test_declared_checks_and_distinct_cue_design(self):
        self.assertTrue(all(check["passed"] for check in self.checks.values()), self.checks)
        self.assertEqual(len(np.unique(self.arrays["cues"], axis=0)), 128)
        self.assertEqual(set(self.metrics), {
            "affine_ordering_null", "commuting_ordering_null", "noncommuting_construction",
            "amplitude_only", "categorical_forced_basins", "continuous_neutral_memory",
            "fresh_noise_not_old_memory"})

    def test_invalid_dynamics_and_noise_are_rejected(self):
        with self.assertRaises(ValueError):
            propagate_affine(np.zeros((2, 2)), np.eye(2)[None], np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            propagated_noise_covariance(np.eye(2)[None], np.diag([-1.0, 1.0]))
        with self.assertRaises(ValueError):
            linear_recovery_risk(np.eye(2), np.array([[1, 1], [0, 1]]))
        with self.assertRaises(np.linalg.LinAlgError):
            linear_recovery_risk(np.eye(2), np.zeros((2, 2)))

    def test_cli_artifacts_recalculate_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            command = [sys.executable, "-m", "representation_collapse.mechanism_calibration",
                       "--output", str(output), "--seed", "11"]
            subprocess.run(command, check=True, capture_output=True)
            verification = json.loads((output / "verification.json").read_text())
            self.assertTrue(verification["passed"])
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "verified")
            metrics = json.loads((output / "metrics.json").read_text())
            with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(covariance_dimension(arrays["continuous_states"]),
                                 metrics["continuous_neutral_memory"]["after_dimension"])
                for arm in range(2):
                    np.testing.assert_allclose(propagate_affine(arrays["affine_encoded_cues"],
                        arrays["affine_matrices"], arrays["affine_forcing"][arm]), arrays["affine_states"][arm],
                        atol=1e-12, rtol=0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(set(manifest["sha256"]), {
                "config.json", "arrays.npz", "metrics.json", "verification.json", "status.json"})
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            before = (output / "manifest.json").read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_invalid_seed_leaves_a_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "invalid-seed"
            result = subprocess.run([sys.executable, "-m", "representation_collapse.mechanism_calibration",
                "--output", str(output), "--seed", "-1"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((output / "status.json").read_text())["state"], "failed")
            self.assertFalse(json.loads((output / "verification.json").read_text())["passed"])
            self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
