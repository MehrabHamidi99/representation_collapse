import unittest
from collections import Counter

import numpy as np

from representation_collapse.vector_task import MOTIFS, QUERY, WRITE, body_from_ids, cyclic_bigrams, paired_batch, paired_body_ids, proper_period, render


class VectorTaskTests(unittest.TestCase):
    def test_library_and_paired_body_statistics(self):
        reference = cyclic_bigrams(MOTIFS[0])
        for motif in MOTIFS:
            self.assertIsNone(proper_period(motif))
            self.assertEqual(Counter(motif), Counter(A=3, B=3, X=6))
            self.assertEqual(cyclic_bigrams(motif), reference)
        rng = np.random.default_rng(17)
        for repeats in [1, 2, 4, 8, 16, 32, 64]:
            for _ in range(10):
                periodic, control = paired_body_ids(rng, repeats)
                self.assertEqual(periodic[-1], control[-1])
                for q in [1, 2]:
                    left, right = body_from_ids(periodic, q), body_from_ids(control, q)
                    self.assertEqual(len(left), 12 * q * repeats)
                    self.assertEqual(Counter(left), Counter(right))
                    self.assertEqual(cyclic_bigrams(left), cyclic_bigrams(right))
                    self.assertEqual(left[-12*q:], right[-12*q:])
                if repeats > 1:
                    self.assertIsNone(proper_period(body_from_ids(control)))
                else:
                    np.testing.assert_array_equal(periodic, control)

    def test_paired_targets_and_no_suffix_value_leak(self):
        for task in ["categorical", "continuous"]:
            batch = paired_batch(np.random.default_rng(5), 7, 2, task)
            np.testing.assert_array_equal(batch.inputs[:7, :16], batch.inputs[7:, :16])
            np.testing.assert_array_equal(batch.inputs[:7, -1], batch.inputs[7:, -1])
            np.testing.assert_array_equal(batch.targets[:7], batch.targets[7:])
            np.testing.assert_array_equal(batch.targets[:7], batch.memory[np.arange(7), batch.queries])
            self.assertTrue((batch.inputs[:, 16:, 21:] == 0).all())
            self.assertTrue((batch.inputs[:, 16:-1, 5:] == 0).all())
            self.assertTrue((batch.inputs[:, :16, QUERY] == 0).all())
            self.assertTrue((batch.inputs[:, :16, WRITE] == 1).all())
            bodies = [body_from_ids(ids) for ids in batch.periodic_ids]
            changed = (batch.memory + 1) % 16 if task == "categorical" else -batch.memory
            rendered = render(changed, batch.queries, bodies, task)
            np.testing.assert_array_equal(rendered[:, 16:], batch.inputs[:7, 16:])

    def test_shared_anchor_and_reproducibility(self):
        left = paired_batch(np.random.default_rng(42), 4, 1, "categorical")
        right = paired_batch(np.random.default_rng(42), 4, 1, "categorical")
        np.testing.assert_array_equal(left.inputs[:4], left.inputs[4:])
        np.testing.assert_array_equal(left.inputs, right.inputs)


if __name__ == "__main__":
    unittest.main()
