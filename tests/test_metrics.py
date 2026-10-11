import unittest

import numpy as np

from representation_collapse.calibrate import calibration
from representation_collapse.metrics import covariance_dimension, ordered_product, period_spectrum


class MeasurementsTest(unittest.TestCase):
    def test_known_spectrum(self):
        samples = np.array([[1, 0], [-1, 0], [0, 2], [0, -2]], dtype=float)
        result = covariance_dimension(samples)
        self.assertAlmostEqual(result["participation_ratio"], 25 / 17)
        self.assertAlmostEqual(result["entropy_effective_rank"], np.exp(-0.8 * np.log(0.8) - 0.2 * np.log(0.2)))
        self.assertEqual(result["numerical_rank"], 2)

    def test_dimension_translation_rotation_and_scale_invariance(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(30, 3)) * [1, 2, 3]
        orthogonal, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        a = covariance_dimension(x)
        b = covariance_dimension((x @ orthogonal) * 17 + 100)
        for key in ["participation_ratio", "entropy_effective_rank"]:
            self.assertAlmostEqual(a[key], b[key])

    def test_noncommuting_product_and_perturbation(self):
        j = np.array([[[1, 3], [0, 1]], [[2, 0], [1, 2]]], dtype=float)
        state = np.array([0.4, -1.0])
        self.assertTrue(np.allclose(ordered_product(j) @ state, j[1] @ (j[0] @ state)))
        self.assertFalse(np.allclose(ordered_product(j), j[0] @ j[1]))

    def test_nonnormal_gain_is_distinct_from_eigenvalues(self):
        spectrum = period_spectrum(np.array([[[0.8, 4], [0, 0.8]]]))
        self.assertLess(spectrum["spectral_radius"], 1)
        self.assertGreater(spectrum["largest_finite_time_gain"], 1)

    def test_invalid_inputs(self):
        for samples in [np.zeros((1, 3)), np.zeros(3), np.array([[0], [np.nan]])]:
            with self.assertRaises(ValueError):
                covariance_dimension(samples)
        for steps in [np.zeros((0, 2, 2)), np.zeros((3, 2, 3)), np.full((1, 2, 2), np.inf)]:
            with self.assertRaises(ValueError):
                ordered_product(steps)

    def test_zero_variance_is_marked_degenerate(self):
        result = covariance_dimension(np.ones((8, 3)))
        self.assertEqual(result["total_variance"], 0)
        self.assertFalse(result["normalized_dimension_defined"])
        self.assertEqual(result["participation_ratio"], 0)

    def test_overflow_cannot_be_reported_as_dimension(self):
        samples = np.array([[1e154, 0], [-1e154, 0], [0, 1e154], [0, -1e154]])
        with self.assertRaisesRegex(ValueError, "overflow"):
            covariance_dimension(samples)

    def test_calibration_controls(self):
        _, checks, _ = calibration(seed=42)
        self.assertTrue(all(checks.values()), checks)


if __name__ == "__main__":
    unittest.main()
