"""Dependency-free preparation checks; no Torch/NumPy/PyG/SDK/native calls."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from representation_collapse.native_graph_calibration import (
    CalibrationError, Ledger, analytic_order_reference, chronological_trace,
    coordinate_permutation, extend_real_operator, frozen_config, kernel_memory_limit, ordered_pairs,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/native-graph-calibration-v1.json"
MODULE = ROOT / "src/representation_collapse/native_graph_calibration.py"


class NativeGraphPreparationTests(unittest.TestCase):
    def test_import_is_dependency_free(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import representation_collapse.native_graph_calibration; "
            "assert not any(k in ('torch','numpy','wandb') or k.startswith('torch_geometric') for k in sys.modules)"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_protocol_and_all_numerical_fields_are_bound(self):
        config = frozen_config(CONFIG)
        self.assertEqual(config["architecture"]["joint_state_dimension"], 12)
        self.assertFalse(config["execution_authorized"])
        self.assertFalse(config["training_authorized"])
        modified = deepcopy(config); modified["tolerances"]["derivative_atol"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "config.json"; file.write_text(json.dumps(modified))
            with self.assertRaisesRegex(CalibrationError, "config changed"):
                frozen_config(file)

    def test_ledger_counts_before_attempt_and_refuses_cap(self):
        events = []
        ledger = Ledger({"native_spy": 1}, lambda: events.append("guard"))
        ledger.charge("native_spy")
        self.assertEqual(ledger.counts, {"native_spy": 1})
        self.assertEqual(events, ["guard"])
        with self.assertRaisesRegex(CalibrationError, "finite call cap"):
            ledger.charge("native_spy")
        self.assertEqual(ledger.counts["native_spy"], 1)
        ledger.complete()

    def test_ledger_partial_and_expired_work_cannot_complete(self):
        ledger = Ledger({"api": 2}, lambda: None)
        ledger.charge("api")
        with self.assertRaisesRegex(CalibrationError, "counts differ"):
            ledger.complete()
        def stopped():
            raise CalibrationError("STOP before call")
        ledger.guard = stopped
        with self.assertRaisesRegex(CalibrationError, "STOP"):
            ledger.charge("api")
        self.assertEqual(ledger.counts["api"], 1)

    def test_successful_schedule_is_finite_and_matches_native_hook_accounting(self):
        config = frozen_config(CONFIG)
        # Count from the declared stages, not a native execution or a model.
        cell_stages = {"cell_wrapper": 2, "dense_step_A": 1, "state_A": 1,
            "dense_step_B": 1, "step_jvp": 2, "step_vjp": 2,
            "composed_dense": 2, "composed_jvp": 2, "composed_vjp": 2,
            "composed_stepwise_jvp": 2, "composed_stepwise_vjp": 2,
            "finite_difference_step": 6, "finite_difference_composed": 12,
            "complex_real_imaginary_extension": 8,
            "permutation_forward": 1, "permutation_jacobian": 1, "chronology": 3}
        self.assertEqual(sum(cell_stages.values()), 50)
        self.assertEqual(config["finite_counts"]["cell_forward_calls"], 50)
        self.assertEqual(config["finite_counts"]["head_forward_calls"], 5)
        self.assertEqual(config["finite_counts"]["head_sequence_forward_calls"], 6)
        self.assertEqual(config["finite_counts"]["dense_jacobian_api_calls"], 4)
        self.assertEqual(config["finite_counts"]["jvp_api_calls"], 9)
        self.assertEqual(config["finite_counts"]["vjp_api_calls"], 5)

    def test_noncommuting_exact_order_control_has_real_distinct_products(self):
        values = analytic_order_reference()
        self.assertEqual(values["chronological"], [[1, 1], [1, 2]])
        self.assertEqual(values["reversed"], [[2, 1], [1, 1]])
        self.assertNotEqual(values["chronological"], values["reversed"])
        self.assertEqual(values["chronological_direction"], [3, 5])
        self.assertEqual(values["reversed_direction"], [5, 3])

    def test_complex_extension_retains_imaginary_part_with_real_operator_calls(self):
        called = []
        def real_operator(value):
            self.assertIsInstance(value, float)
            called.append(value)
            return 3 * value
        value = 1 + 2j
        result = extend_real_operator(real_operator, value, combine=complex)
        imaginary = extend_real_operator(real_operator, 1j * value, combine=complex)
        self.assertEqual(called, [1.0, 2.0, -2.0, 1.0])
        self.assertEqual(result, 3 + 6j)
        self.assertEqual(imaginary, 1j * result)

    def test_complete_hidden_then_cell_coordinate_permutation(self):
        inverse, coordinates = coordinate_permutation([2, 0, 1])
        self.assertEqual(inverse, [1, 2, 0])
        self.assertEqual(coordinates, [4, 5, 0, 1, 2, 3, 10, 11, 6, 7, 8, 9])
        self.assertEqual(sorted(coordinates), list(range(12)))
        with self.assertRaises(CalibrationError):
            coordinate_permutation([0, 0, 2])

    def test_chronology_scores_from_only_preceding_inputs(self):
        events = []
        def consume(state, graph):
            events.append(("consume", graph)); return state + (graph,)
        def score(state, pairs):
            events.append(("score", state)); return state
        state, predictions, trace = chronological_trace(["A", "B", "A"], (), consume=consume, score=score, guard=lambda: None)
        self.assertEqual(trace, ["consume_0", "score_1", "consume_1", "score_2", "consume_2"])
        self.assertEqual(predictions, [("A",), ("A", "B")])
        self.assertEqual(state, ("A", "B", "A"))
        self.assertEqual(events[1], ("score", ("A",)))
        self.assertEqual(ordered_pairs(), ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))

    def test_guard_stops_before_any_chronology_native_spy(self):
        called = []
        def stop(): raise CalibrationError("deadline")
        with self.assertRaisesRegex(CalibrationError, "deadline"):
            chronological_trace(["A"], (), consume=lambda *a: called.append("consume"), score=lambda *a: called.append("score"), guard=stop)
        self.assertEqual(called, [])

    def test_kernel_cgroup_cap_uses_physical_ancestor_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cgroup = root / "cgroup"; child = cgroup / "job" / "step"
            child.mkdir(parents=True)
            (cgroup / "memory.max").write_text("max")
            (child.parent / "memory.max").write_text("2147483648")
            (child / "memory.max").write_text("max")
            proc = root / "proc-cgroup"; proc.write_text("0::/job/step\n")
            value = kernel_memory_limit(proc, cgroup)
            self.assertEqual(value["physical_memory_limit_bytes"], 2147483648)
            self.assertEqual(value["path"], str(child.parent / "memory.max"))

    def test_unbounded_kernel_memory_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "memory.max").write_text("max")
            proc = root / "proc-cgroup"; proc.write_text("0::/\n")
            with self.assertRaisesRegex(CalibrationError, "cap unavailable"):
                kernel_memory_limit(proc, root)

    def test_actual_upstream_wrapper_source_bytes_match_contract(self):
        config = frozen_config(CONFIG)
        source = ROOT / "runs/graph/upstream-pin-20261012/source"
        for relative, expected in config["sources"]["wrapper_files"].items():
            with self.subTest(relative=relative):
                self.assertEqual(sha256(source / relative), expected)
        self.assertEqual(sha256(ROOT / "runs/graph/upstream-pin-20261012/gclstm-v0.54.0.py"), config["sources"]["cell_sha256"])

    def test_critical_import_and_reporting_order_is_explicit_in_worker_source(self):
        source = MODULE.read_text(); tree = ast.parse(source)
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker")
        lines = source.splitlines()
        text = "\n".join(lines[worker.lineno - 1:worker.end_lineno])
        self.assertLess(text.index("signal.setitimer"), text.index("from .reporting"))
        self.assertLess(text.index("install_spawn_journal"), text.index("with ExperimentReporter"))
        self.assertLess(text.index("with ExperimentReporter"), text.index('importlib.import_module("torch")'))
        self.assertLess(text.index('importlib.import_module("torch")'), text.index("_numerics("))
        self.assertNotIn("RLIMIT_AS", text)


if __name__ == "__main__":
    unittest.main()
