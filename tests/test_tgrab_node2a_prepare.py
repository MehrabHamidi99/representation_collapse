import importlib.util
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tgrab_node2a_prepare", ROOT / "code/ops/tgrab_node2a_prepare.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SUPERVISOR_SPEC = importlib.util.spec_from_file_location("tgrab_node2a_supervisor", ROOT / "code/ops/tgrab_node2a_supervisor.py")
SUPERVISOR = importlib.util.module_from_spec(SUPERVISOR_SPEC)
SUPERVISOR_SPEC.loader.exec_module(SUPERVISOR)


class Node2APreparationTests(unittest.TestCase):
    def manifest(self):
        return {
            "source_commit": "a" * 40, "source_tree": "b" * 40,
            "pinned_graph_import_sha256": MODULE.PINNED_IMPORT_SHA256,
            "bootstrap_sha256": "c" * 64, "job_sha256": "d" * 64,
            "supervisor_sha256": "e" * 64, "environment_receipt_sha256": "f" * 64,
            "environment_receipt_local_path": "/tmp/environment-receipt.json",
            "environment_inventory_sha256": "1" * 64,
            "environment_inventory_local_path": "/tmp/environment-identity.json",
            "independent_review_receipt": "/missing/review.json",
            "independent_review_sha256": "0" * 64,
            "run_id": "node2a-20261016T120000Z-1234abcd",
        }

    def test_closed_schema_rejects_extra_or_missing_fields(self):
        repo = ROOT
        with self.assertRaisesRegex(MODULE.PreparationError, "exactly match"):
            MODULE.validate(repo, {**self.manifest(), "submit": True})
        data = self.manifest(); data.pop("source_tree")
        with self.assertRaisesRegex(MODULE.PreparationError, "exactly match"):
            MODULE.validate(repo, data)

    def test_invalid_commit_and_run_id_fail_before_git_or_submission(self):
        with mock.patch.object(MODULE, "git", side_effect=AssertionError("git should not be called")):
            data = self.manifest(); data["source_commit"] = "not-a-commit"
            with self.assertRaisesRegex(MODULE.PreparationError, "full immutable Git SHA"):
                MODULE.validate(ROOT, data)

    def test_importer_sha_is_fixed_by_reviewed_source_pin(self):
        data = self.manifest(); data["pinned_graph_import_sha256"] = "0" * 64
        with mock.patch.object(MODULE, "git", side_effect=AssertionError("git should not be called")):
            with self.assertRaisesRegex(MODULE.PreparationError, "pinned graph importer"):
                MODULE.validate(ROOT, data)
            data = self.manifest(); data["run_id"] = "../run"
            with self.assertRaisesRegex(MODULE.PreparationError, "unique UTC"):
                MODULE.validate(ROOT, data)

    def test_missing_independent_review_fails_closed(self):
        data = self.manifest()
        with self.assertRaisesRegex(MODULE.PreparationError, "review receipt missing"):
            MODULE.validate(ROOT, data)

    def test_staged_environment_receipt_hash_and_schema_are_checked(self):
        receipt = {
            "state": "installed_metadata_verified_native_smoke_pending",
            "installed_metadata_verified": True,
            "requirements_sha256": "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06",
            "artifact_manifest_sha256": "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842",
            "installed_cell_sha256": "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af",
            "venv": str(MODULE.INTERPRETER.parent.parent),
        }
        raw = json.dumps(receipt, sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"; path.write_bytes(raw)
            MODULE.validate_staged_environment_receipt(path, hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(MODULE.PreparationError, "digest mismatch"):
                MODULE.validate_staged_environment_receipt(path, "0" * 64)
            receipt["state"] = "failed"; path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(MODULE.PreparationError, "exact pinned install"):
                MODULE.validate_staged_environment_receipt(path, MODULE.digest(path))

    def test_staged_environment_identity_uses_bootstrap_closed_schema(self):
        modules = {
            name: {"path": str(MODULE.INTERPRETER.parent.parent / "lib/module.py"),
                   "sha256": "a" * 64, "distribution": distribution,
                   "record_sha256": "b" * 64}
            for name, distribution in MODULE.EXPECTED_MODULE_DISTRIBUTIONS.items()
        }
        identity = {"python": "3.10.19 (main, Jan 1 2026)", "python_version": [3, 10, 19],
            "platform": "Linux-6.x", "machine": "x86_64",
            "prefix": str(MODULE.INTERPRETER.parent.parent),
            "lock_sha256": "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06",
            "wheel_manifest_sha256": "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842",
            "installed_versions": {"torch": "2.0.0+cpu"}, "installed_module_files": modules,
            "source_identity": {
                "wrappers": dict(MODULE.PINNED_WRAPPER_SHA256),
                "cell_sha256": MODULE.PINNED_CELL_SHA256,
                "graph_reconstruction_sha256": MODULE.PINNED_GRAPH_RECONSTRUCTION_SHA256,
                "pinned_graph_import_sha256": MODULE.PINNED_IMPORT_SHA256,
            }}
        raw = json.dumps({"schema": "tgrab-node2a-environment-identity-v1", "identity": identity},
                         sort_keys=True).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identity.json"; path.write_bytes(raw)
            MODULE.validate_staged_environment_inventory(path, hashlib.sha256(raw).hexdigest())
            with self.assertRaisesRegex(MODULE.PreparationError, "digest mismatch"):
                MODULE.validate_staged_environment_inventory(path, "0" * 64)
            identity["source_identity"]["wrappers"]["model/gclstm.py"] = "0" * 64
            path.write_text(json.dumps({"schema": "tgrab-node2a-environment-identity-v1", "identity": identity},
                                       sort_keys=True))
            with self.assertRaisesRegex(MODULE.PreparationError, "source identity"):
                MODULE.validate_staged_environment_inventory(path, hashlib.sha256(path.read_bytes()).hexdigest())
            identity["source_identity"]["wrappers"] = dict(MODULE.PINNED_WRAPPER_SHA256)
            identity["installed_module_files"].pop("torch_geometric_temporal.nn.recurrent.gc_lstm")
            path.write_text(json.dumps({"schema": "tgrab-node2a-environment-identity-v1", "identity": identity},
                                       sort_keys=True))
            with self.assertRaisesRegex(MODULE.PreparationError, "module coverage"):
                MODULE.validate_staged_environment_inventory(path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rehashed_plan_cannot_weaken_stop_or_deadline(self):
        run_id = "node2a-20261016T120000Z-1234abcd"
        valid = {"run_id": run_id,
            "output": str(SUPERVISOR.SCRATCH / "runs/figure10-node2a" / run_id),
            "stop_file": str(SUPERVISOR.SCRATCH / "runs/figure10-node2a" / run_id / "STOP"),
            "deadline_seconds": 570, "allocation": dict(SUPERVISOR.EXPECTED_PLAN_ALLOCATION),
            "fixture": "synthetic-node2a-zero-update-only", "scientific_acceptance": False}
        SUPERVISOR.validate_plan_contract(valid)
        for field, value in (("stop_file", "/tmp/STOP"), ("deadline_seconds", 571)):
            altered = dict(valid); altered[field] = value
            # A caller can recompute the plan digest; fixed policy must still reject it.
            self.assertEqual(len(hashlib.sha256(json.dumps(altered, sort_keys=True).encode()).hexdigest()), 64)
            with self.assertRaises(SystemExit):
                SUPERVISOR.validate_plan_contract(altered)

    def test_main_only_reports_blocked_dry_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"bad": True}))
            output = base / "prepared-plan.json"
            with mock.patch("sys.stderr"):
                rc = MODULE.main(["--repo", str(ROOT), "--manifest", str(manifest), "--output", str(output)])
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
