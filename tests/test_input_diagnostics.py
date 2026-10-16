"""Synthetic descriptive controls; no datasets, models, SDK or experiment run."""
import unittest
from collections import Counter

import numpy as np

from representation_collapse.input_diagnostics import (
    body_statistics, matched_body_audit, sequence_statistics,
)
from representation_collapse.vector_task import MOTIFS


class InputDiagnosticTests(unittest.TestCase):
    def test_twenty_distinct_two_motif_controls_retain_declared_matches(self):
        for i, final in enumerate(MOTIFS):
            for j, other in enumerate(MOTIFS):
                if i == j:
                    continue
                result = matched_body_audit(final * 2, other + final, repeats=2)
                self.assertTrue(all(result['matched'].values()))
                self.assertEqual(result['final_motif_exposures'], {'periodic': 2, 'control': 1})

    def test_realized_trigrams_can_match_or_differ(self):
        equal = matched_body_audit(MOTIFS[1] * 2, MOTIFS[2] + MOTIFS[1], repeats=2)
        different = matched_body_audit(MOTIFS[0] * 2, MOTIFS[1] + MOTIFS[0], repeats=2)
        self.assertTrue(equal['cyclic_trigrams_equal'])
        self.assertFalse(different['cyclic_trigrams_equal'])
        word = MOTIFS[1] * 2
        literal = Counter(''.join(word[(i + j) % len(word)] for j in range(3)) for i in range(len(word)))
        self.assertEqual(equal['periodic']['cyclic_trigrams'], dict(literal))

    def test_empty_and_single_motif_anchors(self):
        empty = matched_body_audit('', '', repeats=0)
        self.assertEqual(empty['periodic']['features']['mean'], None)
        self.assertEqual(empty['periodic']['distinct_motif_count'], 0)
        self.assertIsNone(empty['periodic']['features']['normalized_power'])
        self.assertEqual(empty['final_motif_exposures'], {'periodic': 0, 'control': 0})
        shared = matched_body_audit(MOTIFS[0], MOTIFS[0], repeats=1)
        self.assertTrue(shared['cyclic_trigrams_equal'])
        with self.assertRaises(ValueError):
            matched_body_audit(MOTIFS[0], MOTIFS[1], repeats=1)

    def test_dwell_expansion_is_measured_in_update_units(self):
        def dwell(word):
            return ''.join(c * 2 for c in word)
        result = matched_body_audit(dwell(MOTIFS[0] * 2), dwell(MOTIFS[1] + MOTIFS[0]), repeats=2, q=2)
        self.assertEqual(result['periodic']['updates'], 48)
        self.assertEqual(result['periodic']['linear_run_length_histogram'], {'2': 24})
        self.assertEqual(result['periodic']['symbol_counts'], {'A': 12, 'B': 12, 'X': 24})

    def test_one_sided_power_parseval_for_odd_and_even_sequences(self):
        for x in (np.array([[1., 0.], [0., 1.], [-1., 0.]]),
                  np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])):
            result = sequence_statistics(x)
            direct = sum(sum((float(row[j]) - sum(x[:, j]) / len(x)) ** 2 for j in range(x.shape[1])) for row in x) / len(x)
            self.assertAlmostEqual(sum(result['power']), direct, places=14)
            self.assertAlmostEqual(sum(result['normalized_power']), 1., places=14)
        nyquist = sequence_statistics(np.array([[1.], [-1.], [1.], [-1.]]))
        self.assertEqual(nyquist['frequency'][-1], .5)
        self.assertAlmostEqual(nyquist['power'][-1], 1.)

    def test_cyclic_covariance_matches_literal_wrap_and_scaling(self):
        x = np.array([[1., 2.], [3., -1.], [-2., 4.]])
        result = sequence_statistics(x, max_lag=10)
        means = [sum(x[:, i]) / 3 for i in range(2)]
        expected = [[[sum((x[t, i] - means[i]) * (x[(t + lag) % 3, j] - means[j]) for t in range(3)) / 3
                      for j in range(2)] for i in range(2)] for lag in range(3)]
        np.testing.assert_allclose(result['cyclic_cross_covariance'], expected, rtol=0, atol=1e-14)
        scaled = sequence_statistics(x * 7 + 10, max_lag=10)
        np.testing.assert_allclose(scaled['cyclic_cross_covariance'], np.asarray(expected) * 49, atol=1e-12)
        np.testing.assert_allclose(result['normalized_power'], scaled['normalized_power'], atol=1e-14)
        np.testing.assert_array_equal(x, [[1., 2.], [3., -1.], [-2., 4.]])

    def test_constant_features_are_not_given_normalized_power(self):
        result = sequence_statistics(np.ones((3, 2)))
        self.assertEqual(result['centered_variance_energy'], 0.)
        self.assertFalse(result['normalized_power_defined'])
        self.assertIsNone(result['normalized_power'])

    def test_invalid_unmatched_or_unbounded_requests_fail(self):
        for args in ((MOTIFS[0] * 2, MOTIFS[0] * 2, 2),
                     (MOTIFS[0] * 2, MOTIFS[0] + MOTIFS[1], 2),
                     (MOTIFS[0], MOTIFS[0], 2)):
            with self.assertRaises(ValueError):
                matched_body_audit(args[0], args[1], repeats=args[2])
        for values in (np.array([[np.nan]]), np.array([[1j]]), np.ones(3), np.ones((4097, 1)),
                       np.array([[2 ** 63]], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                sequence_statistics(values)
        for kwargs in ({'q': True}, {'q': 1000000000}, {'max_lag': -1}, {'max_lag': 129}):
            with self.assertRaises(ValueError):
                body_statistics(MOTIFS[0], **kwargs)
        with self.assertRaises(ValueError):
            sequence_statistics(np.array([[1e308], [-1e308]]))
        with self.assertRaisesRegex(ValueError, "explicitly typed"):
            sequence_statistics([[2 ** 53 + 1, 0.0], [2 ** 53 + 1, 1.0]])


if __name__ == '__main__':
    unittest.main()
