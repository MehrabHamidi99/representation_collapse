"""Actual native GRU/LSTM chronology fixtures, with no acquisition claim."""

import json
import unittest

import numpy as np
import torch

from representation_collapse.diagnostics import paired_batch_with_zero, single_write_episode
from representation_collapse.native_state import native_phase_capture
from representation_collapse.pilot import VectorMemoryModel


class NativeStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def model_inputs(self, kind, task, repeats=2, dtype=torch.float32, immediate=False):
        torch.manual_seed(751)
        model = VectorMemoryModel(kind, task, hidden_size=5, bottleneck=7).to(dtype=dtype)
        rng = np.random.default_rng(157)
        batch = single_write_episode(rng, 4, task, 0) if immediate else paired_batch_with_zero(rng, 2, repeats, task)
        return model, torch.tensor(batch.inputs.copy(), dtype=dtype)

    def test_all_native_model_task_precision_horizon_combinations(self):
        for kind in ("gru", "lstm"):
            for task in ("categorical", "continuous"):
                for dtype in (torch.float32, torch.float64):
                    for repeats in (0, 2, 16):
                        with self.subTest(kind=kind, task=task, dtype=dtype, repeats=repeats):
                            model, inputs = self.model_inputs(kind, task, repeats, dtype)
                            result = native_phase_capture(model, inputs, 16)
                            chronology = result["chronology"]
                            self.assertTrue(chronology["passed"])
                            self.assertFalse(chronology["scientific_acceptance"])
                            self.assertFalse(chronology["floquet_certified"])
                            self.assertEqual(chronology["body_updates"], repeats * 12)
                            self.assertEqual(chronology["phase_after_update_indices_zero_based"],
                                             {"post_write": 15, "pre_query": 15 + repeats * 12, "post_query": 16 + repeats * 12})
                            for phase in ("post_write", "pre_query", "post_query"):
                                self.assertEqual(tuple(result[phase].shape), (4, 10 if kind == "lstm" else 5))
                                self.assertFalse(result[phase].requires_grad)
                                self.assertIsNone(result[phase].grad_fn)
                                self.assertTrue(torch.isfinite(result[phase]).all())
                                hidden = result["component_states"][phase]["hidden"]
                                torch.testing.assert_close(result[phase][:, :5], hidden, atol=0, rtol=0)
                                if kind == "lstm":
                                    torch.testing.assert_close(result[phase][:, 5:], result["component_states"][phase]["cell"], atol=0, rtol=0)
                            torch.testing.assert_close(result["native_logits"], model(inputs))
                            self.assertTrue(all(check["passed"] for check in chronology["checks"].values()))
                            json.dumps(chronology, allow_nan=False)

    def test_immediate_query_is_included_and_full_prefix_r0_is_distinct(self):
        for kind in ("gru", "lstm"):
            for task in ("categorical", "continuous"):
                model, inputs = self.model_inputs(kind, task, immediate=True)
                result = native_phase_capture(model, inputs, 1)
                self.assertEqual(result["chronology"]["sequence_updates"], 2)
                torch.testing.assert_close(result["post_write"], result["pre_query"], atol=0, rtol=0)
                self.assertFalse(torch.equal(result["pre_query"], result["post_query"]))
                with self.assertRaises(ValueError):
                    native_phase_capture(model, inputs, 16)

    def test_immediate_query_must_match_written_coordinate_both_tasks_and_models(self):
        for kind in ("gru", "lstm"):
            for task in ("categorical", "continuous"):
                model, inputs = self.model_inputs(kind, task, immediate=True)
                changed = inputs.clone()
                changed[:, -1, 5:21] = 0
                changed[:, -1, 5 + 1] = 1
                with self.assertRaisesRegex(ValueError, "address the single written coordinate"):
                    native_phase_capture(model, changed, 1)
                # The complete-prefix task permits any required coordinate:
                # matching the final WRITE's nuisance address is not required.
                full_model, full = self.model_inputs(kind, task, repeats=0)
                full[:, -1, 5:21] = 0
                full[:, -1, 5 + 1] = 1
                self.assertTrue(native_phase_capture(full_model, full, 16)["chronology"]["passed"])

    def test_changed_query_cannot_change_any_pre_query_component(self):
        for kind in ("gru", "lstm"):
            for task in ("categorical", "continuous"):
                model, inputs = self.model_inputs(kind, task)
                original = native_phase_capture(model, inputs, 16)
                changed = inputs.clone()
                new_queries = (inputs[:, -1, 5:21].argmax(dim=-1) + 1) % 8
                changed[:, -1, 5:21] = 0
                changed[torch.arange(len(changed)), -1, 5 + new_queries] = 1
                second = native_phase_capture(model, changed, 16)
                for phase in ("post_write", "pre_query"):
                    torch.testing.assert_close(original[phase], second[phase], atol=0, rtol=0)
                self.assertFalse(torch.equal(original["post_query"], second["post_query"]))
                self.assertNotEqual(original["chronology"]["inputs_sha256"], second["chronology"]["inputs_sha256"])
                self.assertEqual(original["chronology"]["weights_sha256"], second["chronology"]["weights_sha256"])

    def test_saved_states_are_detached_independent_copies_and_modes_restored(self):
        model, inputs = self.model_inputs("lstm", "continuous", repeats=0)
        model.train()
        model.encoder.eval()
        before = [module.training for module in model.modules()]
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        result = native_phase_capture(model, inputs, 16)
        self.assertEqual(before, [module.training for module in model.modules()])
        result["post_write"].zero_()
        self.assertFalse(torch.equal(result["post_write"], result["pre_query"]))
        self.assertFalse(torch.equal(result["post_write"][:, :5], result["component_states"]["post_write"]["hidden"]))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

    def test_chronology_dtype_payload_and_recurrence_validation(self):
        model, inputs = self.model_inputs("gru", "categorical")
        invalid = []
        invalid.append(inputs.to(torch.float64))
        invalid.append(inputs.to(torch.float16))
        invalid.append(inputs[:, :0])
        invalid.append(inputs[:0])
        bad = inputs.clone(); bad[:, 0, 0] = 0; invalid.append(bad)
        bad = inputs.clone(); bad[:, -1, 21] = 1; invalid.append(bad)
        bad = inputs.clone(); bad[:, 16, 5] = 1; invalid.append(bad)
        bad = inputs.clone(); bad[:, 1, 5:21] = 0; bad[:, 1, 5] = 1; invalid.append(bad)
        bad = inputs.clone(); bad[:, 2, 21:37] = 0; invalid.append(bad)
        bad = inputs.clone(); bad[:, 0, 37] = float("nan"); invalid.append(bad)
        for bad in invalid:
            with self.assertRaises(ValueError):
                native_phase_capture(model, bad, 16)
        for bad_count in (0, 2, True):
            with self.assertRaises(ValueError):
                native_phase_capture(model, inputs, bad_count)
        model.core = torch.nn.GRU(7, 5, batch_first=True, num_layers=2)
        with self.assertRaises(ValueError):
            native_phase_capture(model, inputs, 16)

    def test_forward_head_shortcut_or_lost_cell_is_detected_and_modes_restored(self):
        model, inputs = self.model_inputs("lstm", "continuous")
        before = [module.training for module in model.modules()]
        original_forward = model.forward
        model.forward = lambda x: original_forward(x) + 0.25
        with self.assertRaises(FloatingPointError):
            native_phase_capture(model, inputs, 16)
        self.assertEqual(before, [module.training for module in model.modules()])
        model.forward = original_forward
        model.pack_state = lambda state: state[0][0]
        with self.assertRaises(ValueError):
            native_phase_capture(model, inputs, 16)


if __name__ == "__main__":
    unittest.main()
