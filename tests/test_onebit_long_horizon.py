import unittest
from collections import Counter

import numpy as np

from representation_collapse.onebit_long_horizon import FORCING_DRAWS, GRID, schedules
from representation_collapse.vector_task import proper_period


class LongHorizonDesignTest(unittest.TestCase):
    def test_frozen_grid_has_equal_horizon_controls(self):
        self.assertEqual(GRID, ((64, 1), (128, 1), (256, 1), (512, 1), (256, 2), (128, 4)))
        self.assertEqual(FORCING_DRAWS, 512)
        self.assertEqual([12 * r * q for r, q in GRID[-3:]], [6144] * 3)

    def test_long_pair_preserves_marginals_without_exact_tiling(self):
        for r, q in ((64, 1), (512, 1), (128, 4)):
            periodic, control = schedules(np.random.default_rng(4000), 2, r, q)[0]
            self.assertEqual(len(periodic), 12 * r * q)
            self.assertEqual(len(control), len(periodic))
            self.assertEqual(Counter(periodic), Counter(control))
            self.assertEqual(Counter(zip(periodic, periodic[1:])), Counter(zip(control, control[1:])))
            self.assertEqual(periodic[-12 * q:], control[-12 * q:])
            self.assertIsNone(proper_period(control))


if __name__ == '__main__':
    unittest.main()
