"""Tiny independent verifier checks; no real arrays, SDK, models or network."""
import hashlib
import math
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "code/analysis/verify_legacy_floquet_audit.py"))


class LegacyReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()

    def save(self, name, array):
        path = self.root / name
        np.save(path, array, allow_pickle=False)
        return {"path": name, "bytes": path.stat().st_size, "dtype": array.dtype.str,
                "shape": list(array.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_complete_tiny_complex_fixture_and_schema_exclusions(self):
        entries = []
        for name, (dtype, shape) in VERIFY["schemas"](8).items():
            if name == "eigvals.npy":
                array = np.array([1j, -1j, -1, 1, 0, 3 + 4j], dtype=dtype)
            elif name.startswith("norm_env"):
                array = np.full(shape, math.sqrt(40), dtype=dtype)
            else:
                array = np.asfortranarray(np.full(shape, 1 + 2j, dtype=dtype))
            entries.append(self.save(name, array))
        manifest = {"schema_version": 1, "kind": "legacy_floquet_array_manifest", "files": entries}
        result = VERIFY["reduce_inputs"](self.root, manifest, fixture_width=8)
        self.assertEqual(len(result["file_stats"]), 40)
        self.assertEqual(len(result["prefix_differences"]), 36)
        self.assertTrue(result["all_envelopes_have_byte_identical_time_rows"])
        self.assertEqual(result["norm_comparison_count"], 2280)
        self.assertEqual(result["norm_failure_count"], 0)
        np.testing.assert_allclose(result["stored_eigenvectors"]["column_norms"], [math.sqrt(40)] * 6)
        np.testing.assert_allclose(result["stored_eigenvectors"]["gram_real"], np.full((6, 6), 40))
        self.assertEqual(result["stored_eigenvalues"]["magnitude"], [1, 1, 1, 1, 0, 5])
        self.assertFalse(result["scientific_acceptance"])
        self.assertFalse(result["numerical_integrity_accepted"])
        with self.assertRaises(VERIFY["ReconciliationError"]):
            VERIFY["manifest_entries"](manifest)
        for bad_path in ("../eigvals.npy", "jacobian_0.pt", "model.npy"):
            bad = dict(manifest, files=[dict(entries[0], path=bad_path), *entries[1:]])
            with self.assertRaises(VERIFY["ReconciliationError"]):
                VERIFY["manifest_entries"](bad, fixture_width=8)

    def test_signed_zero_and_complex_state_axis_with_failed_saved_reference(self):
        envelope = np.zeros((20, 2, 6), dtype=np.complex64)
        envelope[:, 0, 0] = 3 + 4j
        envelope[3, 1, 2] = complex(0, -0.0)
        saved = np.zeros((20, 6), dtype=np.float32)
        result = VERIFY["envelope_metrics"](envelope, saved)
        self.assertEqual(result["recomputed_norms"][0], [5, 0, 0, 0, 0, 0])
        self.assertFalse(result["time_rows_exactly_equal_to_row0"][3])
        self.assertEqual(result["max_absolute_time_difference_from_row0"][3], 0)
        self.assertEqual(result["norm_failure_indices"], [[row, 0] for row in range(20)])
        self.assertEqual(result["norm_tolerances"][0][0], 1e-6)

    def test_adjacent_byte_counts_retain_zero_sign_and_complex_difference(self):
        first = np.zeros((20, 2, 6), dtype=np.complex64)
        second = first.copy()
        second[0, 0, 0] = 3 + 4j
        second[1, 0, 0] = complex(-0.0, 0)
        result = VERIFY["adjacent_metrics"](first, second)
        self.assertEqual(result["scalar_entries"], 240)
        self.assertEqual(result["numeric_changed_scalar_entries"], 1)
        self.assertEqual(result["bitwise_changed_scalar_entries"], 2)
        self.assertEqual(result["l2_difference"], 5)
        self.assertEqual(result["max_absolute_difference"], 5)

    def test_claim_corruption_is_retained_and_discrete_types_are_exact(self):
        checks = VERIFY["Comparisons"]()
        checks.compare("norms", [1.0, 2.0], [1.0 + 5e-13, 2.001], numeric=True)
        checks.compare("flags", [True, False], [1, False])
        self.assertEqual([m["path"] for m in checks.mismatches], ["norms/1", "flags/0"])
        self.assertEqual(checks.count, 4)
        self.assertGreater(checks.mismatches[0]["absolute_error"], checks.mismatches[0]["tolerance"])

    def test_header_hash_length_and_mapping_mutation_fail_closed(self):
        entry = self.save("eigvals.npy", np.arange(6, dtype=np.complex64))
        reader, error = VERIFY["numeric_mmap"], VERIFY["ReconciliationError"]
        with self.assertRaises(error):
            with reader(self.root, dict(entry, sha256="0" * 64)):
                pass
        with self.assertRaises(error):
            with reader(self.root, entry):
                with (self.root / "eigvals.npy").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\x01")
        path = self.root / "eigvals.npy"
        path.write_bytes(path.read_bytes() + b"extra")
        bad = dict(entry, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(error):
            with reader(self.root, bad):
                pass

    def test_object_symlink_and_nonfinite_refusal(self):
        reader, error = VERIFY["numeric_mmap"], VERIFY["ReconciliationError"]
        path = self.root / "eigvals.npy"
        np.save(path, np.array([{"unexpected": True}] * 6, dtype=object), allow_pickle=True)
        bad = {"path": "eigvals.npy", "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "dtype": "<c8", "shape": [6]}
        with self.assertRaises(error):
            with reader(self.root, bad):
                pass
        entry = self.save("eigvals.npy", np.zeros(6, dtype=np.complex64))
        path.rename(self.root / "original.npy")
        path.symlink_to(self.root / "original.npy")
        with self.assertRaises(error):
            with reader(self.root, entry):
                pass
        for value in (np.nan, np.inf, complex(0, np.inf)):
            with self.assertRaises(error):
                VERIFY["require_finite"](np.array([value], dtype=np.complex64))


if __name__ == "__main__":
    unittest.main()
