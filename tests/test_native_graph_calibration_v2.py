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
from types import ModuleType, SimpleNamespace
from unittest import mock

from representation_collapse import native_graph_calibration_v2 as engine

from representation_collapse.native_graph_calibration_v2 import (
    CalibrationError, Ledger, analytic_order_reference, chronological_trace,
    coordinate_permutation, extend_real_operator, frozen_config, kernel_memory_limit, ordered_pairs,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/native-graph-calibration-v2.json"
MODULE = ROOT / "src/representation_collapse/native_graph_calibration_v2.py"


class NativeGraphV2PreparationTests(unittest.TestCase):
    def test_actual_worker_reporter_constructor_accepts_hashes_and_public_paths_without_sdk(self):
        from representation_collapse.reporting import ExperimentReporter
        tree = ast.parse(MODULE.read_text(), feature_version=(3, 10))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker")
        identity_assignment = next(node for node in ast.walk(worker) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "identities" for target in node.targets))
        constructor = next(node for node in ast.walk(worker) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "ExperimentReporter")
        # Execute the actual worker's metadata expressions and actual reporting
        # constructor, excluding its with-body and __enter__/SDK lifecycle.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "worker"
            namespace = dict(vars(engine), args=SimpleNamespace(config=CONFIG, root=ROOT),
                admission={"source_commit": "a" * 40, "environment": {"sha256": "b" * 64}},
                config=frozen_config(CONFIG), output=output, ExperimentReporter=ExperimentReporter)
            assignment = ast.Module(body=[deepcopy(identity_assignment)], type_ignores=[])
            exec(compile(assignment, str(MODULE), "exec"), namespace)
            with mock.patch.object(engine.importlib, "import_module", side_effect=AssertionError("SDK import forbidden")) as imports:
                reporter = eval(compile(ast.Expression(body=deepcopy(constructor)), str(MODULE), "eval"), namespace)
                imports.assert_not_called()
            expected_paths = {"module_path": engine.MODULE_PATH, "config_path": engine.CONFIG_PATH,
                              "protocol_path": engine.PROTOCOL_PATH}
            self.assertIsInstance(reporter, ExperimentReporter)
            self.assertEqual({key: reporter.config[key] for key in expected_paths}, expected_paths)
            self.assertTrue(all(isinstance(value, str) and len(value) < 256 for value in expected_paths.values()))
            self.assertTrue(set(expected_paths).isdisjoint(reporter.identities))
            self.assertEqual(len(reporter.identities), 8)
            for value in reporter.identities.values():
                self.assertRegex(value, r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
            self.assertIsNone(reporter._run)
            self.assertEqual(reporter._state, "new")
            self.assertFalse(output.exists())
            # The old placement fails the real scalar contract. This catches
            # an accidental path-to-identities regression without a fake reporter.
            with self.assertRaisesRegex(ValueError, "identities must be lowercase"):
                ExperimentReporter(output / "bad", project=reporter.project, config=reporter.config,
                    identities={**reporter.identities, "module_path": engine.MODULE_PATH})

    def test_import_is_dependency_free(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import representation_collapse.native_graph_calibration_v2; "
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


    def test_v2_preserves_every_v1_numeric_environment_and_resource_field(self):
        previous = json.loads((ROOT / "configs/graph/native-graph-calibration-v1.json").read_text())
        current = frozen_config(CONFIG)
        for field in ("architecture", "state_order", "fixtures", "tolerances", "finite_counts",
                      "resources", "environment_policy", "reporting", "required_distributions",
                      "complex_tangent_control", "forbidden"):
            with self.subTest(field=field):
                self.assertEqual(current[field], previous[field])
        self.assertEqual(current["protocol_id"], "native_graph_calibration_v2")
        self.assertFalse(current["execution_authorized"])
        self.assertFalse(current["import_policy"]["full_Temporal_package_API_verified"])
        previous_tree = ast.parse((ROOT / "src/representation_collapse/native_graph_calibration.py").read_text())
        current_tree = ast.parse(MODULE.read_text(), feature_version=(3, 10))
        for name in ("_numerics", "_runtime_preflight", "kernel_memory_limit", "chronological_trace"):
            first = next(node for node in previous_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            second = next(node for node in current_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            with self.subTest(function=name):
                self.assertEqual(ast.dump(first), ast.dump(second))

    def test_v2_helper_is_exactly_pinned(self):
        config = frozen_config(CONFIG)
        record = config["sources"]["graph_import_helper"]
        self.assertEqual(record["path"], "src/representation_collapse/pinned_graph_import.py")
        self.assertEqual(sha256(ROOT / record["path"]), record["sha256"])
        self.assertEqual(record["sha256"], "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344")
        ast.parse((ROOT / record["path"]).read_text(), feature_version=(3, 10))

    def _spy_bundle(self, *, release_match=True):
        source_root = ROOT / "runs/graph/upstream-pin-20261012/source"
        cell_path = ROOT / "runs/graph/upstream-pin-20261012/gclstm-v0.54.0.py"
        cell_type, wrapper_type, parameter_type, head_type = (type(name, (), {}) for name in
            ("CellSpy", "WrapperSpy", "ParameterSpy", "HeadSpy"))
        recurrent, head, cell = ModuleType("recurrent_spy"), ModuleType("head_spy"), ModuleType("cell_spy")
        recurrent.GCLSTM = cell.GCLSTM = cell_type
        recurrent.MultiLayerGCLSTM = wrapper_type
        recurrent.GCLSTMParam = parameter_type
        head.LinkPredictor = head_type
        provenance = {"source_release_hashes_match": release_match,
                      "Temporal_package_initializers_executed": False,
                      "full_Temporal_package_API_verified": False,
                      "native_math_or_source_modified": False}
        bundle = SimpleNamespace(recurrent=recurrent, head=head, cell=cell, node=ModuleType("node_spy"),
            cell_type=cell_type, wrapper_type=wrapper_type, parameter_type=parameter_type, head_type=head_type,
            provenance=provenance, provenance_dict=lambda: dict(provenance))
        paths = {cell_type: cell_path, wrapper_type: source_root / "model/gclstm.py",
                 parameter_type: source_root / "model/gclstm.py", head_type: source_root / "model/link_predictor.py"}
        return bundle, paths, source_root, cell_path

    def test_v2_loader_passes_installed_path_pins_guard_and_records_limited_scope(self):
        config = frozen_config(CONFIG); bundle, paths, source_root, cell_path = self._spy_bundle()
        calls, guarded = [], []
        guard = lambda: guarded.append("boundary")
        def load(**kwargs):
            calls.append(kwargs); kwargs["guard"](); return bundle
        helper_path = ROOT / config["sources"]["graph_import_helper"]["path"]
        helper = SimpleNamespace(__file__=str(helper_path), load_pinned_graph_modules=load)
        details = {}
        with mock.patch.dict(sys.modules, {"torch": ModuleType("torch")}), \
                mock.patch.object(engine, "_execute_pinned_helper", return_value=helper) as execute, \
                mock.patch("inspect.getfile", side_effect=lambda value: str(paths[value])):
            recurrent, head = engine._load_wrappers(source_root, config, root=ROOT,
                installed_cell_path=cell_path, guard=guard, provenance=details)
        self.assertIs(recurrent, bundle.recurrent); self.assertIs(head, bundle.head)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["installed_cell_path"], cell_path)
        self.assertEqual(calls[0]["cell_sha256"], config["sources"]["cell_sha256"])
        self.assertEqual(calls[0]["wrapper_sha256"], config["sources"]["wrapper_files"])
        self.assertIs(calls[0]["guard"], guard)
        self.assertGreater(len(guarded), 4)
        self.assertEqual(execute.call_args.args[1], helper_path.read_bytes())
        self.assertIs(details["source_release_hashes_match"], True)
        self.assertIs(details["full_Temporal_package_API_verified"], False)
        self.assertEqual(details["helper"], config["sources"]["graph_import_helper"])

    def test_v2_loader_refuses_missing_torch_before_helper_execution(self):
        config = frozen_config(CONFIG); _, _, source_root, cell_path = self._spy_bundle()
        saved = sys.modules.pop("torch", None)
        try:
            with mock.patch.object(engine, "_execute_pinned_helper") as execute:
                with self.assertRaisesRegex(CalibrationError, "Torch must already"):
                    engine._load_wrappers(source_root, config, root=ROOT, installed_cell_path=cell_path,
                        guard=lambda: None, provenance={})
                execute.assert_not_called()
        finally:
            if saved is not None: sys.modules["torch"] = saved

    def test_v2_loader_refuses_bad_helper_source_before_execution(self):
        config = frozen_config(CONFIG); config["sources"]["graph_import_helper"]["sha256"] = "0" * 64
        _, _, source_root, cell_path = self._spy_bundle()
        with mock.patch.dict(sys.modules, {"torch": ModuleType("torch")}), \
                mock.patch.object(engine, "_execute_pinned_helper") as execute:
            with self.assertRaisesRegex(CalibrationError, "source pin mismatch before execution"):
                engine._load_wrappers(source_root, config, root=ROOT, installed_cell_path=cell_path,
                    guard=lambda: None, provenance={})
            execute.assert_not_called()

    def test_v2_loader_rejects_fixture_hashes_or_substituted_classes(self):
        config = frozen_config(CONFIG)
        for mismatch in ("fixture_hashes", "substitute_class", "wrong_class_path", "full_package_claim"):
            with self.subTest(mismatch=mismatch):
                bundle, paths, source_root, cell_path = self._spy_bundle(release_match=mismatch != "fixture_hashes")
                if mismatch == "substitute_class": bundle.recurrent.GCLSTM = type("Wrong", (), {})
                if mismatch == "wrong_class_path": paths[bundle.cell_type] = source_root / "model/gclstm.py"
                if mismatch == "full_package_claim": bundle.provenance["full_Temporal_package_API_verified"] = True
                helper = SimpleNamespace(__file__=str(ROOT / config["sources"]["graph_import_helper"]["path"]),
                    load_pinned_graph_modules=lambda **kwargs: bundle)
                with mock.patch.dict(sys.modules, {"torch": ModuleType("torch")}), \
                        mock.patch.object(engine, "_execute_pinned_helper", return_value=helper), \
                        mock.patch("inspect.getfile", side_effect=lambda value: str(paths[value])):
                    with self.assertRaises(CalibrationError):
                        engine._load_wrappers(source_root, config, root=ROOT, installed_cell_path=cell_path,
                            guard=lambda: None, provenance={})

    def test_v2_helper_execution_uses_verified_bytes_and_fresh_owned_namespace(self):
        namespace = "_native_graph_calibration_v2_import_helper"
        self.assertNotIn(namespace, sys.modules)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "helper.py"
            path.write_text("raise RuntimeError('unverified disk bytes')")
            module = engine._execute_pinned_helper(path, b"value = 'verified bytes'\n")
            try:
                self.assertEqual(module.value, "verified bytes")
                with self.assertRaisesRegex(CalibrationError, "fresh worker"):
                    engine._execute_pinned_helper(path, b"value = 'replacement'\n")
                self.assertIs(sys.modules[namespace], module)
            finally:
                if sys.modules.get(namespace) is module: del sys.modules[namespace]
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                engine._execute_pinned_helper(path, b"raise ValueError('fixture failure')\n")
            self.assertNotIn(namespace, sys.modules)

    def test_v2_missing_helper_admission_pin_fails_before_any_git_subprocess(self):
        config = frozen_config(CONFIG)
        source_root = ROOT / "runs/graph/upstream-pin-20261012/source"
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            args = SimpleNamespace(root=ROOT, config=CONFIG, source_root=source_root,
                output=temporary / "output", admission=temporary / "admission.json", started_monotonic=1.0)
            records = [engine.MODULE_PATH, engine.PROTOCOL_PATH, engine.CONFIG_PATH,
                config["sources"]["ownership_helper"]["path"], config["sources"]["reporter"]["path"]]
            admission = {"kind": "native_graph_calibration_root_admission", "protocol_id": engine.PROTOCOL_ID,
                "owner": "root coordinator", "execution_authorized": True, "training_authorized": False,
                "scientific_acceptance": False, "retries": 0, "config_sha256": sha256(CONFIG),
                "finite_counts": config["finite_counts"], "resources": config["resources"],
                "output": str(args.output.resolve()), "source_root": str(source_root.resolve()),
                "stop_path": str(temporary / "STOP"), "source_files": {name: sha256(ROOT / name) for name in records}}
            args.admission.write_text(json.dumps(admission))
            with mock.patch.object(engine.subprocess, "check_output") as subprocess_spy:
                with self.assertRaisesRegex(CalibrationError, "published source identity mismatch.*pinned_graph_import"):
                    engine.verify_admission(args, config, worker=True)
                subprocess_spy.assert_not_called()

    def test_v2_worker_reporting_order_provenance_and_argv_are_version_bound(self):
        source = MODULE.read_text(); tree = ast.parse(source, feature_version=(3, 10))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_worker")
        text = "\n".join(source.splitlines()[worker.lineno - 1:worker.end_lineno])
        self.assertLess(text.index("with ExperimentReporter"), text.index('importlib.import_module("torch")'))
        self.assertLess(text.index('importlib.import_module("torch")'), text.index("_load_wrappers("))
        self.assertLess(text.index("_load_wrappers("), text.index("_numerics("))
        self.assertIn('runtime["installed_cell_path"]', text)
        self.assertIn('payload["isolated_import_provenance"]', text)
        self.assertEqual(source.count('"representation_collapse.native_graph_calibration_v2"'), 2)
        self.assertNotIn('"representation_collapse.native_graph_calibration"', source)


if __name__ == "__main__":
    unittest.main()
