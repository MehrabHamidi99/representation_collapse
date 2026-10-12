"""Renderer/reference controls only; these do not certify trained competence."""

from dataclasses import fields
import unittest

import numpy as np

from representation_collapse.diagnostics import (
    POPULATION_VALUE_VARIANCE,
    infer_task,
    normalized_mse,
    paired_batch_with_zero,
    paired_body_ids_with_zero,
    shuffle_memories,
    shuffle_queries,
    single_write_episode,
    streaming_register_oracle,
    uniform_categorical_cross_entropy,
    zero_predictor_mse,
)
from representation_collapse.vector_task import A, QUERY, WRITE, paired_batch, paired_body_ids, render, sample_memory


class DiagnosticsTests(unittest.TestCase):
    def test_r0_geometry_targets_and_rng_draws(self):
        for task in ("categorical", "continuous"):
            with self.subTest(task=task):
                rng, reference_rng = np.random.default_rng(42), np.random.default_rng(42)
                batch = paired_batch_with_zero(rng, 7, 0, task, q=2)
                memory = sample_memory(reference_rng, 7, task)
                queries = reference_rng.integers(8, size=7, dtype=np.int64)
                np.testing.assert_array_equal(batch.memory, memory)
                np.testing.assert_array_equal(batch.queries, queries)
                self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
                self.assertEqual(batch.inputs.shape, (14, 17, 38))
                self.assertEqual((batch.repeats, batch.q), (0, 2))
                for ids in (batch.periodic_ids, batch.control_ids):
                    self.assertEqual(ids.shape, (7, 0))
                    self.assertEqual(ids.dtype, np.dtype("int64"))
                np.testing.assert_array_equal(batch.inputs[:7], batch.inputs[7:])
                np.testing.assert_array_equal(batch.targets[:7], memory[np.arange(7), queries])
                np.testing.assert_array_equal(streaming_register_oracle(batch.inputs, task), batch.targets)

    def test_zero_body_ids_are_independent_and_consume_no_rng(self):
        rng = np.random.default_rng(6)
        before = rng.bit_generator.state
        left, right = paired_body_ids_with_zero(rng, np.int64(0))
        self.assertEqual(left.shape, (0,))
        self.assertFalse(np.shares_memory(left, right))
        self.assertEqual(before, rng.bit_generator.state)

    def test_positive_repeats_exact_v1_delegation_and_rng(self):
        for task in ("categorical", "continuous"):
            for repeats in (1, 2, 16):
                for q in (1, 2):
                    with self.subTest(task=task, repeats=repeats, q=q):
                        rng, ref = np.random.default_rng(7), np.random.default_rng(7)
                        actual = paired_batch_with_zero(rng, 3, repeats, task, q)
                        expected = paired_batch(ref, 3, repeats, task, q)
                        for field in fields(actual):
                            np.testing.assert_array_equal(getattr(actual, field.name), getattr(expected, field.name))
                        self.assertEqual(rng.bit_generator.state, ref.bit_generator.state)
                        np.testing.assert_array_equal(streaming_register_oracle(actual.inputs, task), actual.targets)
                        rng, ref = np.random.default_rng(9), np.random.default_rng(9)
                        a, b = paired_body_ids_with_zero(rng, repeats), paired_body_ids(ref, repeats)
                        for left, right in zip(a, b):
                            np.testing.assert_array_equal(left, right)
                        self.assertEqual(rng.bit_generator.state, ref.bit_generator.state)

    def test_single_write_all_required_coordinates_matches_renderer(self):
        for task in ("categorical", "continuous"):
            for coordinate in range(8):
                with self.subTest(task=task, coordinate=coordinate):
                    batch = single_write_episode(np.random.default_rng(5), 4, task, np.int64(coordinate))
                    memory = sample_memory(np.random.default_rng(5), 4, task)
                    reference = render(memory, np.full(4, coordinate), [""] * 4, task)
                    np.testing.assert_array_equal(batch.inputs, reference[:, [coordinate, 16]])
                    np.testing.assert_array_equal(batch.targets, memory[:, coordinate])
                    np.testing.assert_array_equal(batch.queries, np.full(4, coordinate))
                    np.testing.assert_array_equal(streaming_register_oracle(batch.inputs, task), batch.targets)
                    self.assertTrue(np.all(batch.inputs[:, -1, 21:] == 0))
                    self.assertEqual(infer_task(batch.inputs), task)
        for task in ("categorical", "continuous"):
            batch = single_write_episode(np.random.default_rng(11), 64, task)
            self.assertEqual(set(batch.queries), set(range(8)))
            np.testing.assert_array_equal(streaming_register_oracle(batch.inputs, task), batch.targets)

    def test_counts_require_integers_before_any_rng_draw(self):
        rng = np.random.default_rng(42)
        before = rng.bit_generator.state
        for bad in (True, np.bool_(False), 1.0, 1.5, "1", None, -1):
            for call in (
                lambda: paired_body_ids_with_zero(rng, bad),
                lambda: paired_batch_with_zero(rng, bad, 0, "continuous"),
                lambda: paired_batch_with_zero(rng, 2, bad, "continuous"),
                lambda: paired_batch_with_zero(rng, 2, 0, "continuous", bad),
                lambda: single_write_episode(rng, bad, "continuous"),
                lambda: uniform_categorical_cross_entropy(bad),
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    call()
        for bad in (True, np.bool_(False), 1.0, "1", -1, 8):
            with self.subTest(coordinate=bad), self.assertRaises(ValueError):
                single_write_episode(rng, 2, "continuous", bad)
        for call in (
            lambda: paired_batch_with_zero(rng, 0, 0, "continuous"),
            lambda: paired_batch_with_zero(rng, 1, 0, "continuous", 0),
            lambda: single_write_episode(rng, 0, "continuous"),
            lambda: single_write_episode(rng, 1, "unknown"),
            lambda: paired_batch_with_zero(rng, 1, 0, "unknown"),
        ):
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(before, rng.bit_generator.state)

    def test_streaming_last_write_wins_and_nuisance_is_ignored(self):
        for task in ("categorical", "continuous"):
            memory = np.arange(16)[None, :] if task == "categorical" else np.linspace(-1, 1, 16)[None, :]
            base = render(memory, np.array([0]), ["ABX"], task)
            # WRITE 0, body A, nuisance WRITE 8, updated WRITE 0, body B/X, QUERY.
            update = base[:, 1:2].copy()
            update[..., 5:21] = 0
            update[..., 5] = 1
            x = np.concatenate((base[:, 0:1], base[:, 16:17], base[:, 8:9], update, base[:, 17:]), axis=1)
            expected = memory[0, 1] if task == "categorical" else np.float32(memory[0, 1])
            np.testing.assert_array_equal(streaming_register_oracle(x, task), [expected])
            # Nuisance can overwrite its own register repeatedly without affecting required memory.
            x = np.concatenate((x[:, :-1], base[:, 8:9], x[:, -1:]), axis=1)
            np.testing.assert_array_equal(streaming_register_oracle(x, task), [expected])

    def test_unwritten_register_rejected(self):
        x = single_write_episode(np.random.default_rng(1), 2, "continuous", 0).inputs
        x[:, -1, 5] = 0
        x[:, -1, 6] = 1
        with self.assertRaisesRegex(ValueError, "unwritten"):
            streaming_register_oracle(x, "continuous")

    def test_malformed_layout_rejected_by_oracle_inference_and_controls(self):
        base = paired_batch_with_zero(np.random.default_rng(7), 1, 1, "categorical").inputs[:1]
        corruptions = {
            "missing type": (0, slice(0, 5), 0),
            "double type": (0, A, 1),
            "fractional type": (0, WRITE, 0.5),
            "missing coordinate": (0, slice(5, 21), 0),
            "double coordinate": (0, 6, 1),
            "fractional coordinate": (0, 5, 0.5),
            "body coordinate": (16, 5, 1),
            "body payload": (16, 37, 0.5),
            "query payload": (-1, 21, 1),
            "missing categorical payload": (0, slice(21, 37), 0),
            "fractional categorical payload": (0, slice(21, 37), 0.5),
            "mixed payload": (0, 37, 0.5),
            "nonfinite": (0, 37, np.nan),
        }
        cases = {}
        for name, (step, channels, value) in corruptions.items():
            x = base.copy()
            x[:, step, channels] = value
            cases[name] = x
        cases["early duplicate query"] = np.concatenate((base[:, -1:], base), axis=1)
        cases["nonterminal query"] = np.concatenate((base, base[:, 16:17]), axis=1)
        cases["missing query"] = base[:, :-1]
        x = base.copy()
        x[:, -1, 5:21] = 0
        x[:, -1, 13] = 1
        cases["nuisance query"] = x
        cases.update({"2d": base[0], "wrong width": base[..., :-1], "empty episodes": base[:0], "empty time": base[:, :0], "complex": base.astype(complex), "text": base.astype(str)})
        for name, x in cases.items():
            for function in (infer_task, lambda v: streaming_register_oracle(v, "categorical"), lambda v: shuffle_memories(v, np.random.default_rng(1)), lambda v: shuffle_queries(v, np.random.default_rng(1))):
                with self.subTest(case=name), self.assertRaises(ValueError):
                    function(x)

    def test_continuous_payload_rules_and_all_zero_inference(self):
        x = render(np.zeros((1, 16)), np.array([0]), [""], "continuous")
        self.assertEqual(infer_task(x), "continuous")
        np.testing.assert_array_equal(streaming_register_oracle(x, "continuous"), [0.0])
        for value in (1.0001, -1.0001, np.inf):
            bad = x.copy()
            bad[:, 0, 37] = value
            with self.assertRaises(ValueError):
                infer_task(bad)
        mixed = x.copy()
        mixed[:, 0, 21] = 1
        with self.assertRaises(ValueError):
            infer_task(mixed)
        with self.assertRaises(ValueError):
            streaming_register_oracle(x, "categorical")
        with self.assertRaises(ValueError):
            infer_task(x[:, -1:])

    def test_memory_shuffle_exact_seeded_permutation_dtype_and_immutability(self):
        for task in ("categorical", "continuous"):
            for dtype in (np.float32, np.float64):
                x = paired_batch_with_zero(np.random.default_rng(1), 8, 2, task).inputs.astype(dtype)
                if task == "continuous":
                    x[:, 0, 37] = np.float64(0.123456789012345)  # Preserve FP64 values.
                original = x.copy()
                write = x[..., WRITE] == 1
                rng, reference_rng = np.random.default_rng(23), np.random.default_rng(23)
                expected_payload = x[..., 21:][write][reference_rng.permutation(np.count_nonzero(write))]
                out, targets = shuffle_memories(x, rng)
                self.assertEqual(out.dtype, x.dtype)
                self.assertFalse(np.shares_memory(out, x))
                np.testing.assert_array_equal(x, original)
                np.testing.assert_array_equal(out[..., :21], x[..., :21])
                np.testing.assert_array_equal(out[~write], x[~write])
                np.testing.assert_array_equal(out[..., 21:][write], expected_payload)
                np.testing.assert_array_equal(targets, streaming_register_oracle(out, task))
                self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
                self.assertTrue(np.any(targets != streaming_register_oracle(x, task)))

    def test_memory_shuffle_identity_is_not_forced_away(self):
        x = single_write_episode(np.random.default_rng(1), 2, "continuous", 0).inputs
        # Choose a seed yielding identity for two payloads; this is a legitimate draw.
        seed = next(s for s in range(20) if np.array_equal(np.random.default_rng(s).permutation(2), [0, 1]))
        out, targets = shuffle_memories(x, np.random.default_rng(seed))
        np.testing.assert_array_equal(out, x)
        np.testing.assert_array_equal(targets, streaming_register_oracle(x, "continuous"))

    def test_query_shuffle_selects_new_register_preserves_dtype_and_inputs(self):
        for task in ("categorical", "continuous"):
            memory = np.tile(np.arange(16), (8, 1)) if task == "categorical" else np.tile(np.linspace(-1, 1, 16), (8, 1))
            x = render(memory, np.arange(8), ["ABX"] * 8, task).astype(np.float64)
            if task == "continuous":
                x[:, 0, 37] = 0.123456789012345
            original = x.copy()
            out, targets = shuffle_queries(x, np.random.default_rng(19))
            self.assertEqual(out.dtype, x.dtype)
            np.testing.assert_array_equal(x, original)
            np.testing.assert_array_equal(out[:, :-1], x[:, :-1])
            np.testing.assert_array_equal(out[:, -1, :5], x[:, -1, :5])
            np.testing.assert_array_equal(out[:, -1, 21:], x[:, -1, 21:])
            new = out[:, -1, 5:21].argmax(axis=-1)
            self.assertTrue(np.all(new != np.arange(8)))
            np.testing.assert_array_equal(targets, streaming_register_oracle(out, task))
            self.assertTrue(np.all(targets != streaming_register_oracle(x, task)))

    def test_query_shuffle_requires_all_registers_before_rng_draw(self):
        x = single_write_episode(np.random.default_rng(1), 2, "continuous", 0).inputs
        rng = np.random.default_rng(10)
        before = rng.bit_generator.state
        with self.assertRaisesRegex(ValueError, "all eight"):
            shuffle_queries(x, rng)
        self.assertEqual(rng.bit_generator.state, before)

    def test_normalization_and_reference_values(self):
        self.assertEqual(POPULATION_VALUE_VARIANCE, 1 / 3)
        self.assertEqual(uniform_categorical_cross_entropy(), np.log(16))
        self.assertEqual(uniform_categorical_cross_entropy(np.int64(4)), np.log(4))
        self.assertEqual(zero_predictor_mse(np.array([-1, 0, 1], dtype=np.int64)), 2 / 3)
        self.assertEqual(zero_predictor_mse(np.full(10, 0.5)), 0.25)
        self.assertEqual(normalized_mse(1 / 3), 1)
        self.assertAlmostEqual(normalized_mse(1 / 60), 0.05)
        self.assertEqual(normalized_mse(np.array(0.25)), 0.75)
        for bad in ([], [np.nan], [np.inf], [1j], ["1"]):
            with self.subTest(targets=bad), self.assertRaises(ValueError):
                zero_predictor_mse(bad)
        for bad in (-1, np.nan, np.inf, [0.1], 1j, True, "1"):
            with self.subTest(mse=bad), self.assertRaises(ValueError):
                normalized_mse(bad)


if __name__ == "__main__":
    unittest.main()
