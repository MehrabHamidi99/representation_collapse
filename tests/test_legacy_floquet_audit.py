"""Tiny NumPy fixtures; no real archive, model, checkpoint, SDK or network."""
import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

from representation_collapse.legacy_floquet_audit import (
    AuditError, audit_arrays, eigenvalue_summary, envelope_summary,
    expected_files, finite_check, prefix_difference, validate_manifest,
    verified_array,
)


def pin(path):
    array = np.load(path, allow_pickle=False)
    return {"path": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dtype": array.dtype.str, "shape": list(array.shape)}


class LegacyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def save(self, name="eigvals.npy", value=None):
        value = np.arange(6, dtype=np.complex64) if value is None else value
        path = self.root / name
        np.save(path, value, allow_pickle=False)
        return pin(path)

    def test_valid_mmap_and_frozen_stats(self):
        entry = self.save()
        with verified_array(self.root, entry) as (array, snapshots):
            self.assertIsInstance(array, np.memmap)
            self.assertFalse(array.flags.writeable)
            finite_check(array)
        self.assertEqual(snapshots["before"], snapshots["after"])
        self.assertEqual(snapshots["before"], snapshots["path_after"])

    def test_missing_array(self):
        entry = self.save()
        (self.root / entry["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            with verified_array(self.root, entry):
                pass

    def test_changed_hash(self):
        entry = self.save()
        with (self.root / entry["path"]).open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\x7f")
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass

    def test_wrong_pin(self):
        entry = self.save()
        entry["sha256"] = "0" * 64
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass

    def test_object_header_rejected_before_pickle(self):
        path = self.root / "eigvals.npy"
        np.save(path, np.array([{"unsafe": True}] * 6, dtype=object), allow_pickle=True)
        entry = {"path": path.name, "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dtype": "|O", "shape": [6]}
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass

    def test_truncation_with_updated_size_and_hash(self):
        entry = self.save()
        path = self.root / entry["path"]
        path.write_bytes(path.read_bytes()[:-8])
        entry.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass

    def test_shape_pin_mismatch(self):
        entry = self.save()
        entry["shape"] = [3, 2]
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass

    def test_unsafe_path_and_exclusions(self):
        entry = self.save()
        for name in ("../eigvals.npy", str(self.root / "eigvals.npy"), "nested/eigvals.npy",
                     "jacobian_0.npy", "model.pt", "dataset.npy", "norm_env_19.npy"):
            with self.subTest(name=name), self.assertRaises(AuditError):
                with verified_array(self.root, dict(entry, path=name)):
                    pass

    def test_symlink_refusal(self):
        entry = self.save()
        original = self.root / "original.npy"
        (self.root / "eigvals.npy").rename(original)
        (self.root / "eigvals.npy").symlink_to(original)
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                pass
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(AuditError):
            with verified_array(alias, entry):
                pass

    def test_mutation_while_mapped(self):
        entry = self.save()
        with self.assertRaises(AuditError):
            with verified_array(self.root, entry):
                with (self.root / "eigvals.npy").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\x01")

    def test_nonfinite(self):
        for value in (np.nan, np.inf, complex(1, np.inf)):
            with self.subTest(value=value), self.assertRaises(AuditError):
                finite_check(np.array([value], dtype=np.complex64))

    def test_complex_phase_has_no_period_normalization(self):
        values = np.array([1j, -1j, -1, 1, 0, 3 + 4j], dtype=np.complex64)
        result = eigenvalue_summary(values)
        np.testing.assert_allclose(result["magnitude"], [1, 1, 1, 1, 0, 5])
        np.testing.assert_allclose(result["phase_radians_per_stored_product"],
                                   [np.pi / 2, -np.pi / 2, np.pi, 0, 0, np.arctan2(4, 3)])

    def test_repeated_time_and_directed_norm_failure(self):
        base = np.ones((8, 6), dtype=np.complex64) * (3 + 4j)
        array = np.repeat(base[None], 20, axis=0)
        reference = np.full((20, 6), np.sqrt(8) * 5, dtype=np.float32)
        result = envelope_summary(array, reference)
        self.assertTrue(result["all_time_rows_exactly_equal_to_row0"])
        self.assertTrue(result["norm_comparison_passed"])
        reference[3, 2] = 0
        result = envelope_summary(array, reference)
        self.assertEqual(result["norm_failure_indices"], [[3, 2]])
        self.assertEqual(result["norm_tolerances"][3][2], 1e-6)
        array[2, 0, 1] += 1j
        result = envelope_summary(array, reference)
        self.assertFalse(result["time_rows_exactly_equal_to_row0"][2])

    def test_prefix_difference(self):
        left = np.zeros((20, 8, 6), dtype=np.complex64)
        right = left.copy()
        right[1, 2, 3] = 3 + 4j
        result = prefix_difference(left, right)
        self.assertEqual(result["changed_scalar_entries"], 1)
        self.assertEqual(result["max_absolute_difference"], 5)
        self.assertEqual(result["l2_difference"], 5)

    def test_full_tiny_manifest_and_audit_preserves_failures(self):
        files = []
        for name, (dtype, shape) in expected_files(8).items():
            if name.startswith("norm_env"):
                value = np.zeros(shape, dtype=dtype)
            elif name == "eigvecs.npy":
                value = np.asfortranarray(np.ones(shape, dtype=dtype))
            else:
                value = np.ones(shape, dtype=dtype)
            files.append(self.save(name, value))
        manifest = {"schema_version": 1, "kind": "legacy_floquet_array_manifest", "files": files}
        self.assertEqual(len(validate_manifest(manifest, fixture_width=8)), 40)
        with self.assertRaises(AuditError):
            validate_manifest(manifest)
        result = audit_arrays(self.root, manifest, fixture_width=8)
        self.assertTrue(result["integrity_completed"])
        self.assertFalse(result["all_norm_comparisons_passed"])
        self.assertFalse(result["scientific_acceptance"])
        self.assertEqual(len(result["file_stats"]), 40)
        np.testing.assert_allclose(result["stored_eigenvectors"]["column_norms"], [np.sqrt(8)] * 6)
        np.testing.assert_allclose(result["stored_eigenvectors"]["gram_real"], np.full((6, 6), 8))
        self.assertEqual(len(result["prefix_differences"]), 36)
        self.assertTrue(all(value["exactly_equal"] for value in result["prefix_differences"].values()))
        for key in ("../eigvals.npy", "jacobian_0.npy"):
            bad = dict(manifest, files=[dict(files[0], path=key), *files[1:]])
            with self.assertRaises(AuditError):
                validate_manifest(bad, fixture_width=8)


class LegacyAuditRepairTests(unittest.TestCase):
    def test_signed_zero_time_row_and_prefix_byte_identity(self):
        array = np.zeros((20, 8, 6), dtype=np.complex64)
        array[3, 2, 4] = complex(-0.0, 0.0)
        array[5, 1, 2] = complex(0.0, -0.0)
        reference = np.zeros((20, 6), dtype=np.float32)
        self.assertTrue(np.array_equal(array[0], array[3]))
        result = envelope_summary(array, reference)
        self.assertFalse(result["time_rows_exactly_equal_to_row0"][3])
        self.assertFalse(result["time_rows_exactly_equal_to_row0"][5])
        self.assertTrue(result["norm_comparison_passed"])
        self.assertEqual(result["max_absolute_time_difference_from_row0"][3], 0)
        baseline = np.zeros_like(array)
        difference = prefix_difference(baseline, array)
        self.assertFalse(difference["exactly_equal"])
        self.assertEqual(difference["numeric_changed_scalar_entries"], 0)
        self.assertEqual(difference["bitwise_changed_scalar_entries"], 2)
        self.assertEqual(difference["changed_scalar_entries"], 2)
        self.assertEqual(difference["l2_difference"], 0)

    def test_detached_orphan_journal_cleanup_preserves_unrelated_process(self):
        import json
        import os
        import runpy
        import signal
        import subprocess
        import sys
        import time
        launcher = Path(__file__).resolve().parents[1] / "code/ops/legacy_floquet_audit.py"
        helpers = runpy.run_path(str(launcher))
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory).resolve() / "ownership.json"
            script = ("import runpy,subprocess,sys;from pathlib import Path;"
                      "h=runpy.run_path(sys.argv[1]);h['install_spawn_journal'](Path(sys.argv[2]));"
                      "subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)")
            unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True)
            worker = subprocess.Popen([sys.executable, "-c", script, str(launcher), str(journal)], start_new_session=True)
            tracker = helpers["OwnedProcesses"](worker.pid, journal)
            child_identity = None
            try:
                worker.wait(timeout=5)
                record = json.loads(journal.read_text())
                child_identity = record["processes"][0]
                self.assertEqual(child_identity["pgid"], child_identity["pid"])
                tracker.refresh()
                self.assertIn(child_identity["pid"], tracker.known)
                self.assertEqual(tracker.cleanup(), [])
                until = time.monotonic() + 2
                while time.monotonic() < until:
                    current = helpers["process_identity"](child_identity["pid"])
                    if current is None or current["state"] == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail("owned detached descendant survived cleanup")
                self.assertIsNone(unrelated.poll())
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=2)
                if child_identity is not None:
                    try:
                        helpers["signal_owned"](child_identity, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                unrelated.kill()
                unrelated.wait(timeout=2)

    def test_changed_start_identity_refuses_signal(self):
        import os
        import runpy
        from unittest.mock import patch
        launcher = Path(__file__).resolve().parents[1] / "code/ops/legacy_floquet_audit.py"
        helpers = runpy.run_path(str(launcher))
        identity = helpers["process_identity"](os.getpid())
        self.assertIsNotNone(identity)
        mismatched = dict(identity, start="different-process-incarnation")
        function = helpers["signal_owned"]
        with patch.dict(function.__globals__, process_identity=lambda pid: mismatched):
            with patch("os.kill") as kill:
                self.assertFalse(function(identity, 9))
                kill.assert_not_called()

    def test_cleanup_retains_snapshot_failure_and_unverified_survivor(self):
        import os
        import runpy
        from unittest.mock import patch
        launcher = Path(__file__).resolve().parents[1] / "code/ops/legacy_floquet_audit.py"
        helpers = runpy.run_path(str(launcher))
        tracker = helpers["OwnedProcesses"](os.getpid(), Path("does-not-exist-owned-journal"))
        identity = helpers["process_identity"](os.getpid())
        namespace = tracker.cleanup.__func__.__globals__
        with patch.object(tracker, "refresh", side_effect=RuntimeError("injected snapshot failure")):
            with patch.dict(namespace, signal_owned=lambda *args, **kwargs: True,
                            process_identity=lambda pid: identity):
                errors = tracker.cleanup()
        self.assertIn("RuntimeError", errors)
        self.assertIn("OwnedDescendantUnverified", errors)
        self.assertEqual(tracker.cleanup_reconciliation[str(os.getpid())],
                         "live_or_unverified_at_cleanup_deadline")


if __name__ == "__main__":
    unittest.main()
