"""Independent reducer checks, including realistic tiny production artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("independent_periodicity_reducer", ROOT / "code/analysis/reduce_periodicity.py")
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)
try:
    import torch
except ImportError:
    torch = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refresh_manifest(path):
    write_json(path / "manifest.json", {"schema_version": 1, "sha256": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in sorted(reducer.REQUIRED_FILES)}})


class ReductionNumericsTests(unittest.TestCase):
    def test_noncommuting_chronology_and_dense_lstm_shape(self):
        a = np.array([[.8, .3], [0, .7]])
        b = np.array([[.7, 0], [.3, .8]])
        product = reducer.ordered_product(np.stack([a, b]))
        np.testing.assert_array_equal(product, b @ a)
        self.assertGreater(np.linalg.norm(product - a @ b), .01)
        with self.assertRaises(reducer.InvalidArtifact):
            reducer.ordered_product(np.zeros((12, 2, 3)))

    def test_strict_global_closure_is_not_elementwise_allclose(self):
        a = np.ones((13, 4))
        b = a + 3e-8
        value = reducer.closure(a, b, 1e-7, 1e-5, 1e-12)
        self.assertTrue(value["coordinatewise_passed"])
        self.assertFalse(value["passed"])
        with self.assertRaises(reducer.InvalidArtifact):
            reducer.closure(a, b[:1], 1e-7, 1e-5, 1e-12)

    def test_ulp_zero_signed_zero_and_one_spacing(self):
        for dtype in (np.float32, np.float64):
            a = np.array([1, 0, -0.0], dtype=dtype)
            b = np.array([np.nextafter(dtype(1), dtype(2)), 0, 0], dtype=dtype)
            summary = reducer.ulp_diagnostics(a, b)
            self.assertEqual(summary["maximum_coordinate_spacing_ratio"], 1)
            self.assertEqual(summary["numerically_changed_coordinates"], 1)
            self.assertEqual(summary["bitwise_changed_coordinates"], 2)
            self.assertFalse(summary["bitwise_equal"])
            self.assertTrue(reducer.ulp_diagnostics(a, a)["bitwise_equal"])

    def test_eigen_matching_preserves_multiplicity_and_conjugates(self):
        with self.assertRaisesRegex(reducer.InvalidArtifact, "multiplicity"):
            reducer.eigenvalue_matching(np.array([0, 0, 1]), np.array([0, 1, 1]))
        result = reducer.eigenvalue_matching(np.array([1j, -1j, 0]), np.array([0, -1j, 1j]))
        self.assertTrue(result["one_to_one_matching_passed"])
        self.assertEqual(result["maximum_matched_absolute_residual"], 0)

    def test_scalar_audit_does_not_erase_subulp_residual(self):
        with self.assertRaisesRegex(reducer.InvalidArtifact, "disagrees"):
            reducer.compare(0.0, 3e-19, "tiny residual")
        reducer.compare(3e-19 * (1 + 1e-10), 3e-19, "tiny residual")
        with self.assertRaises(reducer.InvalidArtifact):
            reducer.compare(1e-30, 0.0, "exact zero")

    def test_nonfinite_values_rejected_without_json_nan(self):
        with self.assertRaises(reducer.InvalidArtifact):
            reducer.closure(np.array([np.nan]), np.array([0.0]), 1e-7, 1e-5, 1e-12)
        with self.assertRaises(reducer.InvalidArtifact):
            reducer.ordered_product(np.array([[[np.inf]]]))


@unittest.skipIf(torch is None, "Torch required only to create tiny fixture")
class ReductionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from representation_collapse.pilot import VectorMemoryModel, atomic_json, _manifest
        from representation_collapse.periodicity import run
        torch.set_num_threads(1)
        torch.manual_seed(4)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        parent = root / "original"
        parent.mkdir()
        frozen = json.loads((ROOT / "configs/experiments/periodicity-analysis-v1.json").read_text())
        frozen.update(driver_panels=1, hidden_size=4, input_bottleneck=3)
        frozen.pop("cells")
        saved = {"model": "lstm", "task": "continuous", "seed": 0, "hidden_size": 4,
                 "input_bottleneck": 3, "effective_training_steps": 1500, "max_steps_override": False}
        checkpoint = parent / "checkpoint.pt"
        model = VectorMemoryModel("lstm", "continuous", 4, 3)
        torch.save({"model_state_dict": model.state_dict(), "config": saved, "selected_step": 100}, checkpoint)
        atomic_json(parent / "config.json", saved)
        atomic_json(parent / "status.json", {"state": "completed"})
        atomic_json(parent / "provenance.json", {"git_revision": frozen["training_source_commit"], "dirty_checkout": False})
        _manifest(parent)
        digest = reducer.sha256(checkpoint)
        cls.original = root / "diagnostic"
        # The trusted-identity fixture is synthesized for parser/reducer testing;
        # it never purports to be a clean immutable scientific run.
        run(frozen, checkpoint, cls.original, "lstm", "continuous", 0, reducer.DEFAULT_SOURCE,
            cluster="fixture", verify_source=False, checkpoint_sha256=digest)
        frozen["cells"] = [{"model": "lstm", "task": "continuous", "seed": 0, "cluster": "fixture",
                             "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "baseline_competence_passed": False}]
        cls.frozen = frozen
        cls.frozen_path = root / "periodicity-analysis-v1.json"
        write_json(cls.frozen_path, frozen)
        origin = reducer.read_json(cls.original / "provenance.json")
        origin.update(git_revision=reducer.DEFAULT_SOURCE, diagnostic_source_commit=reducer.DEFAULT_SOURCE,
                      source_verified=True, dirty_checkout=False,
                      diagnostic_module_sha256=reducer.DEFAULT_MODULE_SHA256,
                      source_sha256={str(cls.frozen_path): reducer.sha256(cls.frozen_path)})
        write_json(cls.original / "provenance.json", origin)
        refresh_manifest(cls.original)
        # Cover the other architecture and categorical output/input schema.
        gru_parent = root / "original-gru"
        gru_parent.mkdir()
        gru_saved = {**saved, "model": "gru", "task": "categorical"}
        gru_checkpoint = gru_parent / "checkpoint.pt"
        gru_model = VectorMemoryModel("gru", "categorical", 4, 3)
        torch.save({"model_state_dict": gru_model.state_dict(), "config": gru_saved, "selected_step": 100}, gru_checkpoint)
        atomic_json(gru_parent / "config.json", gru_saved)
        atomic_json(gru_parent / "status.json", {"state": "completed"})
        atomic_json(gru_parent / "provenance.json", {"git_revision": frozen["training_source_commit"], "dirty_checkout": False})
        _manifest(gru_parent)
        cls.gru_frozen = json.loads(json.dumps(frozen))
        cls.gru_frozen["cells"] = [{**frozen["cells"][0], "model": "gru", "task": "categorical",
                                    "checkpoint": str(gru_checkpoint), "checkpoint_sha256": reducer.sha256(gru_checkpoint)}]
        cls.gru_frozen_path = root / "gru-periodicity-analysis-v1.json"
        write_json(cls.gru_frozen_path, cls.gru_frozen)
        cls.gru_artifact = root / "diagnostic-gru"
        run(cls.gru_frozen, gru_checkpoint, cls.gru_artifact, "gru", "categorical", 0, reducer.DEFAULT_SOURCE,
            cluster="fixture", verify_source=False, checkpoint_sha256=reducer.sha256(gru_checkpoint))
        gru_origin = reducer.read_json(cls.gru_artifact / "provenance.json")
        gru_origin.update(git_revision=reducer.DEFAULT_SOURCE, diagnostic_source_commit=reducer.DEFAULT_SOURCE,
                          source_verified=True, dirty_checkout=False, diagnostic_module_sha256=reducer.DEFAULT_MODULE_SHA256,
                          source_sha256={str(cls.gru_frozen_path): reducer.sha256(cls.gru_frozen_path)})
        write_json(cls.gru_artifact / "provenance.json", gru_origin)
        refresh_manifest(cls.gru_artifact)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = Path(tempfile.mkdtemp(dir=self.temporary.name)) / "artifact"
        shutil.copytree(self.original, self.case)

    def tearDown(self):
        shutil.rmtree(self.case.parent)

    def reduce(self):
        return reducer.reduce_directory(self.case, self.frozen, frozen_sha256=reducer.sha256(self.frozen_path))

    def alter_array(self, name, mutate):
        path = self.case / name
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        mutate(arrays)
        np.savez_compressed(path, **arrays)
        refresh_manifest(self.case)

    def test_all_saved_arrays_reduced_and_r1_unavailable(self):
        report = self.reduce()
        self.assertTrue(report["independent_raw_reduction_passed"])
        self.assertEqual(report["dense_windows"], 8)
        self.assertEqual(report["step_jacobian_evaluations"], 192)
        self.assertEqual(len(report["assays"]), 16)
        self.assertEqual(len(report["precision_comparisons"]), 8)
        self.assertEqual(report["probe_floor_sensitivity"]["status"], "pending")
        self.assertEqual(report["query_adapter_validation"]["status"], "partial")
        self.assertFalse(report["floquet_acceptance"])
        for key, assay in report["assays"].items():
            for cue in assay["cues"]:
                self.assertEqual(len(cue["manual_cycles"]), assay["repeats"] - 1)
                if assay["repeats"] == 1:
                    self.assertEqual(cue["native_cycles"], [])
                    self.assertIsNone(assay["paired_cue_separation"]["last_manual_closure_to_terminal_cue_separation"])
                if cue["tangent"]:
                    self.assertEqual(len(cue["tangent"]["singular_values"]), 8)
                    self.assertEqual(cue["tangent"]["interpretation"], "finite_time_one_motif_propagator")
                    self.assertFalse(cue["tangent"]["exact_periodic_orbit_verified"])
        json.dumps(report, allow_nan=False)

    def test_gru_categorical_complete_artifact_schema(self):
        report = reducer.reduce_directory(self.gru_artifact, self.gru_frozen,
                                          frozen_sha256=reducer.sha256(self.gru_frozen_path))
        self.assertTrue(report["independent_raw_reduction_passed"])
        self.assertEqual(report["cell"]["model"], "gru")
        assay = report["assays"]["float64_panel0_R16_periodic"]
        self.assertEqual(len(assay["cues"][0]["native_query_prediction"]), 16)
        self.assertEqual(len(assay["cues"][0]["tangent"]["singular_values"]), 4)
        self.assertEqual(assay["paired_cue_separation"]["component_separations"], {})

    def test_manifest_mutation_rejected(self):
        with (self.case / "inputs.npz").open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(reducer.InvalidArtifact, "hash mismatch"):
            self.reduce()

    def test_rehashed_wrong_jacobian_order_rejected(self):
        def mutate(arrays):
            key = "float64_panel0_R2_periodic_cue0_step_jacobians"
            arrays[key] = arrays[key][::-1].copy()
        self.alter_array("tangents.npz", mutate)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "ordered product"):
            self.reduce()

    def test_rehashed_nonfinite_raw_array_rejected(self):
        def mutate(arrays):
            arrays["float32_panel0_R32_control_cue1_native_core_states"][20, 0] = np.nan
        self.alter_array("trajectories.npz", mutate)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "nonfinite"):
            self.reduce()

    def test_missing_required_raw_array_reports_exact_field(self):
        missing = "float64_panel0_R16_periodic_cue0_previous_step_jacobians"
        self.alter_array("tangents.npz", lambda arrays: arrays.pop(missing))
        with self.assertRaisesRegex(reducer.InvalidArtifact, missing):
            self.reduce()

    def test_unsupported_floquet_promotion_rejected(self):
        metrics = reducer.read_json(self.case / "metrics.json")
        metrics["assays"]["float64_panel0_R16_periodic"]["cues"][0]["tangent"]["floquet_acceptance"] = True
        write_json(self.case / "metrics.json", metrics)
        refresh_manifest(self.case)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "unsupported tangent"):
            self.reduce()

    def test_reported_closure_metric_mutation_rejected(self):
        metrics = reducer.read_json(self.case / "metrics.json")
        metrics["assays"]["float64_panel0_R16_periodic"]["cues"][0]["tangent"]["state_closure"]["absolute"] += .01
        write_json(self.case / "metrics.json", metrics)
        refresh_manifest(self.case)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "state_closure:absolute"):
            self.reduce()

    def test_source_and_checkpoint_registry_fail_closed(self):
        origin = reducer.read_json(self.case / "provenance.json")
        origin["checkpoint_sha256"] = "0" * 64
        write_json(self.case / "provenance.json", origin)
        refresh_manifest(self.case)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "checkpoint hash"):
            self.reduce()
        shutil.copyfile(self.original / "provenance.json", self.case / "provenance.json")
        origin = reducer.read_json(self.case / "provenance.json")
        origin["source_verified"] = False
        write_json(self.case / "provenance.json", origin)
        refresh_manifest(self.case)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "unverified"):
            self.reduce()

    def test_complete_lstm_state_required(self):
        def mutate(arrays):
            key = "float32_panel0_R1_periodic_cue0_states"
            arrays[key] = arrays[key][:, :4]
        self.alter_array("trajectories.npz", mutate)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "expected shape"):
            self.reduce()

    def test_query_or_body_input_leak_rejected(self):
        def mutate(arrays):
            arrays["float32_panel0_R1_periodic_inputs"][0, 20, 37] = .125
        self.alter_array("inputs.npz", mutate)
        with self.assertRaisesRegex(reducer.InvalidArtifact, "literal memory/motifs/query"):
            self.reduce()

    def test_cli_records_failure_and_refuses_overwrite(self):
        output = self.case.parent / "report.json"
        exit_code = reducer.main(["--artifact-dir", str(self.case), "--frozen-config", str(self.frozen_path), "--output", str(output)])
        self.assertEqual(exit_code, 0)
        report = reducer.read_json(output)
        self.assertEqual(report["total_dense_windows"], 8)
        self.assertTrue(report["all_frozen_cells_present"])
        with self.assertRaisesRegex(reducer.InvalidArtifact, "overwrite"):
            reducer.main(["--artifact-dir", str(self.case), "--frozen-config", str(self.frozen_path), "--output", str(output)])
        output.unlink()
        (self.case / "status.json").write_text('{"state":"running"}')
        exit_code = reducer.main(["--artifact-dir", str(self.case), "--frozen-config", str(self.frozen_path), "--output", str(output)])
        self.assertEqual(exit_code, 1)
        self.assertEqual(len(reducer.read_json(output)["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
