"""Independent literal arithmetic fixtures for descriptive input statistics."""
import cmath
from collections import Counter
from fractions import Fraction
import math
import unittest

import numpy as np

from representation_collapse.input_diagnostics import (
    body_statistics, matched_body_audit, sequence_statistics,
)
from representation_collapse.vector_task import MOTIFS


class IndependentInputDiagnosticTests(unittest.TestCase):
    def test_direct_complex_dft_matches_every_reported_bin_for_both_parities(self):
        for rows in ([[1, 2], [3, -1], [0, 4], [-2, 0], [2, 3]],
                     [[1, 2], [3, -1], [0, 4], [-2, 0], [2, 3], [4, -2]]):
            with self.subTest(length=len(rows)):
                count = len(rows); width = len(rows[0])
                means = [sum(Fraction(row[j]) for row in rows) / count for j in range(width)]
                centered = [[float(Fraction(row[j]) - means[j]) for j in range(width)] for row in rows]
                expected = []
                for k in range(count // 2 + 1):
                    coefficients = [sum(centered[t][j] * cmath.exp(-2j * math.pi * k * t / count)
                                        for t in range(count)) for j in range(width)]
                    weight = 1 if k == 0 or (count % 2 == 0 and k == count // 2) else 2
                    expected.append(weight * sum(abs(z) ** 2 for z in coefficients) / count ** 2)
                result = sequence_statistics(np.asarray(rows, dtype=np.int16), max_lag=0)
                self.assertEqual(result["frequency"], [k / count for k in range(count // 2 + 1)])
                for actual, target in zip(result["power"], expected):
                    self.assertAlmostEqual(actual, target, places=12)
                energy = sum(sum(value * value for value in row) for row in centered) / count
                self.assertAlmostEqual(sum(expected), energy, places=12)
                self.assertAlmostEqual(result["centered_variance_energy"], energy, places=12)
                self.assertAlmostEqual(sum(result["normalized_power"]), 1, places=13)

    def test_fraction_oracle_has_forward_cyclic_lag_and_exact_T_denominator(self):
        rows = [[1, 4], [0, 2], [-2, -1], [3, 0]]
        count = len(rows)
        mean = [sum(Fraction(row[j]) for row in rows) / count for j in range(2)]
        expected = [[[sum((Fraction(rows[t][i]) - mean[i]) *
                          (Fraction(rows[(t + lag) % count][j]) - mean[j])
                          for t in range(count)) / count for j in range(2)] for i in range(2)]
                    for lag in range(count)]
        result = sequence_statistics(np.asarray(rows, dtype=np.int64), max_lag=128)
        self.assertEqual(result["lags"], [0, 1, 2, 3])
        self.assertNotEqual(expected[1][0][1], expected[1][1][0])
        for lag in range(count):
            for i in range(2):
                for j in range(2):
                    self.assertEqual(result["cyclic_cross_covariance"][lag][i][j], float(expected[lag][i][j]))
        self.assertEqual(result["independent_training_replicates"], 0)

    def test_singleton_empty_and_zero_energy_are_explicitly_undefined(self):
        for values in (np.empty((0, 2)), np.asarray([[3.0, -7.0]]), np.full((4, 2), 9.0)):
            result = sequence_statistics(values)
            self.assertFalse(result["normalized_power_defined"])
            self.assertIsNone(result["normalized_power"])
            self.assertEqual(result["centered_variance_energy"], 0)
        empty = sequence_statistics(np.empty((0, 2)))
        self.assertTrue(empty["empty_sequence"])
        self.assertIsNone(empty["mean"])
        self.assertEqual((empty["frequency"], empty["power"], empty["lags"]), ([], [], []))
        singleton = sequence_statistics(np.asarray([[3.0, -7.0]]))
        self.assertEqual(singleton["lags"], [0])
        self.assertEqual(singleton["power"], [0])

    def test_typed_array_dtypes_are_reported_and_readonly_input_is_unchanged(self):
        values = np.asarray([[1, 2], [-1, 3], [5, 0]], dtype=np.float16)
        expected = values.copy(); values.flags.writeable = False
        result = sequence_statistics(values, max_lag=0)
        self.assertEqual(result["input_dtype"], values.dtype.str)
        self.assertEqual(result["computation_dtype"], "float64")
        self.assertEqual(values.tolist(), expected.tolist())
        self.assertFalse(values.flags.writeable)

    def test_mixed_untyped_sequence_does_not_silently_round_integer_precision(self):
        # asarray before validation used to round these Python integers inside
        # the helper itself, hiding the loss behind a float64 dtype label.
        for values in ([[2**53 + 1, 0.0], [2**53 + 1, 1.0]],
                       ((2**53 + 1, 0.0), (2**53 + 1, 1.0))):
            with self.subTest(container=type(values).__name__), self.assertRaises(ValueError):
                sequence_statistics(values, max_lag=0)

    def test_precision_limits_and_descriptive_work_caps_fail_before_computation(self):
        for values in (np.asarray([[2**53 + 1]], dtype=np.int64),
                       np.asarray([[-2**53 - 1]], dtype=np.int64),
                       np.asarray([[2**53 + 1]], dtype=np.uint64),
                       np.zeros((1, 129)), np.zeros((4097, 1)), np.zeros((513, 128)),
                       np.zeros((1, 1), dtype=bool), np.zeros((1, 1), dtype=complex)):
            with self.subTest(shape=values.shape, dtype=str(values.dtype)), self.assertRaises(ValueError):
                sequence_statistics(values)
        for shape in ((256, 64), (100, 128)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "work exceeds cap"):
                sequence_statistics(np.zeros(shape), max_lag=128)
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaises(ValueError):
                sequence_statistics(np.zeros((1, 1), dtype=np.longdouble))
        for boundary in (-2**53, 2**53):
            self.assertEqual(sequence_statistics(np.asarray([[boundary]], dtype=np.int64))["mean"], [float(boundary)])

    def test_R3_local_motif_repetition_is_allowed_with_realized_controls_retained(self):
        for dwell in (1, 3):
            expand = lambda text: "".join(c * dwell for c in text)
            periodic = expand(MOTIFS[0] * 3)
            control = expand(MOTIFS[0] + MOTIFS[1] + MOTIFS[0])
            result = matched_body_audit(periodic, control, repeats=3, q=dwell, max_lag=0)
            self.assertTrue(all(result["matched"].values()))
            self.assertEqual(result["final_motif_exposures"], {"periodic": 3, "control": 2})
            self.assertEqual(result["periodic"]["proper_whole_body_tiling_period"], 12 * dwell)
            self.assertIsNone(result["control"]["proper_whole_body_tiling_period"])
            self.assertEqual(result["control"]["distinct_motif_count"], 2)
            for label, text in (("periodic", periodic), ("control", control)):
                for n, key in ((2, "cyclic_bigrams"), (3, "cyclic_trigrams")):
                    literal = Counter("".join(text[(start + j) % len(text)] for j in range(n))
                                      for start in range(len(text)))
                    self.assertEqual(result[label][key], dict(literal))


if __name__ == "__main__":
    unittest.main()
