import unittest
from collections import Counter
import numpy as np
from representation_collapse.onebit_memory import render, schedules, training_batch, evaluation_batch
from representation_collapse.vector_task import proper_period, cyclic_bigrams

class OneBitTests(unittest.TestCase):
    def test_no_label_leak(self):
        x = render(['AXBX', 'AXBX'], [0, 1])
        np.testing.assert_array_equal(x[0, 1:], x[1, 1:])
        self.assertEqual(x[0, 0, 5], -1); self.assertEqual(x[1, 0, 5], 1)
        self.assertTrue(np.all(x[:, 1:, 5] == 0)); self.assertTrue(np.all(x[:, -1, 4] == 1))
    def test_order_pairing(self):
        x, y = evaluation_batch([('AX', 'BX'), ('BX', 'AX')])
        np.testing.assert_array_equal(y, [0, 0, 1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(x[0, 1:], x[2, 1:]); np.testing.assert_array_equal(x[1, 1:], x[3, 1:])
    def test_matched_bodies(self):
        for q in [1, 2, 4]:
            for p, c in schedules(np.random.default_rng(7), 20, 8, q):
                self.assertEqual(Counter(p), Counter(c)); self.assertEqual(cyclic_bigrams(p), cyclic_bigrams(c))
                self.assertEqual(p[-12*q:], c[-12*q:]); self.assertIsNone(proper_period(c))
                self.assertEqual(len(p), 96*q)
    def test_identical_short(self):
        for r in [0, 1]:
            for p, c in schedules(np.random.default_rng(1), 20, r): self.assertEqual(p, c)
    def test_training_balance(self):
        x, y = training_batch(np.random.default_rng(2), 2)
        self.assertEqual(x.shape, (64, 26, 6)); self.assertEqual(sum(y), 32)
        np.testing.assert_array_equal(y[::2], y[1::2])
    def test_invalid(self):
        for bodies, labels in [([], []), (['X','XX'], [0,1]), (['X'], [2])]:
            with self.assertRaises(ValueError): render(bodies, labels)
    def test_native_chronology(self):
        import torch
        from representation_collapse.onebit_memory import make_model
        torch.manual_seed(1); model = make_model(torch); x = torch.from_numpy(render(['AXBX', 'AXBX'], [0,1]))
        h, c = torch.zeros(2,32), torch.zeros(2,32)
        for t in range(x.shape[1]): h, c = model.cell(x[:,t], (h,c))
        self.assertTrue(torch.equal(model(x), model.head(h)))
        h0,c0,_ = model.before_query(x)
        self.assertTrue(torch.equal(model(x), model.query(x,h0.clone(),c0.clone())))
if __name__ == '__main__': unittest.main()
