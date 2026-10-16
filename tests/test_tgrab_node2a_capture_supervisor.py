"""Dry contract tests for the no-data node2a identity capture wrapper."""
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "code/ops/tgrab_node2a_capture_supervisor.py"
SPEC = importlib.util.spec_from_file_location("capture_supervisor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plan():
    run_id = "node2a-capture-20261016T120000Z-a1b2c3d4"
    run_dir = str(MODULE.RUNS / run_id)
    return {
        "schema": "tgrab-node2a-capture-plan-v1", "run_id": run_id,
        "source_commit": "a" * 40, "source_tree": "b" * 40,
        "bootstrap_sha256": MODULE.BOOTSTRAP_SHA256,
        "pinned_graph_import_sha256": MODULE.IMPORTER_SHA256,
        "job_sha256": "c" * 64, "supervisor_sha256": "d" * 64,
        "environment_receipt_sha256": MODULE.ENV_RECEIPT_SHA256,
        "review_receipt_path": str(MODULE.REVIEW_RECEIPT), "review_receipt_sha256": "e" * 64,
        "inventory_path": str(MODULE.INVENTORY), "run_dir": run_dir,
        "stop_file": run_dir + "/STOP", "deadline_seconds": 570,
        "allocation": dict(MODULE.ALLOCATION),
    }


class CaptureSupervisorContractTests(unittest.TestCase):
    def test_fixed_plan_contract(self):
        MODULE.validate_plan_contract(plan())

    def test_rehashed_plan_cannot_change_stop_deadline_or_pins(self):
        original = plan()
        for field, value in (("stop_file", "/tmp/STOP"), ("deadline_seconds", 571),
                             ("pinned_graph_import_sha256", "f" * 64),
                             ("inventory_path", "/tmp/inventory.json")):
            changed = dict(original); changed[field] = value
            # A recomputed digest cannot override the fixed contract.
            json.dumps(changed, sort_keys=True)
            with self.assertRaises(SystemExit, msg=field):
                MODULE.validate_plan_contract(changed)

    def test_run_directory_parent_is_created_private_and_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs" / "capture"
            old_runs = MODULE.RUNS
            try:
                MODULE.RUNS = root
                run = root / "node2a-capture-20261016T120000Z-a1b2c3d4"
                MODULE.ensure_private_run_dir(run)
                self.assertTrue(run.is_dir())
                self.assertEqual(root.stat().st_mode & 0o777, 0o700)
                with self.assertRaises(SystemExit):
                    MODULE.ensure_private_run_dir(run)
            finally:
                MODULE.RUNS = old_runs

    def test_inventory_must_be_unique_absent_fixed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_inventory = MODULE.INVENTORY
            try:
                MODULE.INVENTORY = Path(tmp) / "identity.json"
                MODULE.require_absent_inventory(MODULE.INVENTORY)
                MODULE.INVENTORY.write_text("occupied")
                with self.assertRaises(SystemExit):
                    MODULE.require_absent_inventory(MODULE.INVENTORY)
                MODULE.INVENTORY.unlink()
                target = Path(tmp) / "target"
                target.write_text("x")
                MODULE.INVENTORY.symlink_to(target)
                with self.assertRaises(SystemExit):
                    MODULE.require_absent_inventory(MODULE.INVENTORY)
            finally:
                MODULE.INVENTORY = old_inventory

    def test_child_interpreter_must_be_pinned_symlink_and_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / "python3.10"; target.write_bytes(b"pinned")
            launcher = root / "venv-python"; launcher.symlink_to(target)
            digest = hashlib.sha256(b"pinned").hexdigest()
            with mock.patch.object(MODULE, "INTERPRETER_SHA256", digest):
                result = MODULE.verify_child_interpreter(launcher, target)
                self.assertEqual(result["resolved_path"], str(target.resolve()))
                self.assertEqual(result["sha256"], digest)
                other = root / "other"; other.write_bytes(b"different")
                with self.assertRaisesRegex(SystemExit, "does not resolve"):
                    MODULE.verify_child_interpreter(launcher, other)
                regular = root / "regular"; regular.write_bytes(b"pinned")
                with self.assertRaisesRegex(SystemExit, "must be a symlink"):
                    MODULE.verify_child_interpreter(regular, target)

    def test_bootstrap_mutation_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bootstrap.py"
            path.write_bytes(b"reviewed bytes")
            digest = hashlib.sha256(b"reviewed bytes").hexdigest()
            with mock.patch.object(MODULE, "BOOTSTRAP_SHA256", digest):
                self.assertEqual(MODULE.read_pinned_bootstrap(path), b"reviewed bytes")
                path.write_bytes(b"mutated bytes")
                with self.assertRaisesRegex(SystemExit, "differ from pinned"):
                    MODULE.read_pinned_bootstrap(path)

    def test_post_capture_source_closure_rejects_mutated_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old_file = MODULE.__file__
            files = {
                "src/representation_collapse/tgrab_figure10_bootstrap.py": b"bootstrap",
                "src/representation_collapse/pinned_graph_import.py": b"importer",
                "code/ops/tgrab_node2a_capture_job.sh": b"job",
                "code/ops/tgrab_node2a_capture_supervisor.py": b"supervisor",
            }
            digests = {}
            for rel, content in files.items():
                path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                digests[rel] = hashlib.sha256(content).hexdigest()
            old_bootstrap, old_importer = MODULE.BOOTSTRAP_SHA256, MODULE.IMPORTER_SHA256
            try:
                MODULE.BOOTSTRAP_SHA256 = digests["src/representation_collapse/tgrab_figure10_bootstrap.py"]
                MODULE.IMPORTER_SHA256 = digests["src/representation_collapse/pinned_graph_import.py"]
                candidate = plan(); candidate["job_sha256"] = digests["code/ops/tgrab_node2a_capture_job.sh"]
                candidate["supervisor_sha256"] = digests["code/ops/tgrab_node2a_capture_supervisor.py"]
                completed = [subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 0, candidate["source_commit"] + "\n", ""),
                    subprocess.CompletedProcess([], 0, candidate["source_tree"] + "\n", "")]
                with mock.patch.object(MODULE, "__file__", str(root / "code/ops/tgrab_node2a_capture_supervisor.py")), \
                     mock.patch.object(MODULE.subprocess, "run", side_effect=completed):
                    baseline = MODULE.verify_source_closure(root, candidate)
                self.assertEqual(baseline["bootstrap_sha256"], MODULE.BOOTSTRAP_SHA256)
                (root / "src/representation_collapse/tgrab_figure10_bootstrap.py").write_bytes(b"mutated")
                completed = [subprocess.CompletedProcess([], 0, "", ""),
                    subprocess.CompletedProcess([], 0, candidate["source_commit"] + "\n", ""),
                    subprocess.CompletedProcess([], 0, candidate["source_tree"] + "\n", "")]
                with mock.patch.object(MODULE, "__file__", str(root / "code/ops/tgrab_node2a_capture_supervisor.py")), \
                     mock.patch.object(MODULE.subprocess, "run", side_effect=completed):
                    with self.assertRaisesRegex(SystemExit, "bootstrap_sha256"):
                        MODULE.verify_source_closure(root, candidate)
            finally:
                MODULE.BOOTSTRAP_SHA256, MODULE.IMPORTER_SHA256 = old_bootstrap, old_importer

    def test_review_receipt_exact_path_schema_and_digest(self):
        candidate = plan()
        expected = {"schema": "tgrab-node2a-capture-job-review-v1", "status": "accepted",
            "source_commit": candidate["source_commit"], "source_tree": candidate["source_tree"],
            "bootstrap_sha256": MODULE.BOOTSTRAP_SHA256,
            "pinned_graph_import_sha256": MODULE.IMPORTER_SHA256,
            "job_sha256": candidate["job_sha256"], "supervisor_sha256": candidate["supervisor_sha256"],
            "environment_receipt_sha256": MODULE.ENV_RECEIPT_SHA256,
            "allocation": MODULE.ALLOCATION}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            old_path = MODULE.REVIEW_RECEIPT
            try:
                MODULE.REVIEW_RECEIPT = path
                candidate["review_receipt_path"] = str(path)
                path.write_text(json.dumps(expected, sort_keys=True))
                candidate["review_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                MODULE.verify_review(candidate)
                candidate["review_receipt_sha256"] = "0" * 64
                with self.assertRaisesRegex(SystemExit, "hash mismatch"):
                    MODULE.verify_review(candidate)
                candidate["review_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                altered = dict(expected); altered["status"] = "pending"
                path.write_text(json.dumps(altered))
                candidate["review_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaisesRegex(SystemExit, "does not bind"):
                    MODULE.verify_review(candidate)
                candidate["review_receipt_path"] = str(Path(tmp) / "elsewhere.json")
                with self.assertRaisesRegex(SystemExit, "fixed approved path"):
                    MODULE.verify_review(candidate)
            finally:
                MODULE.REVIEW_RECEIPT = old_path

    def test_cleanup_kills_term_ignoring_child_and_outer_grace_exceeds_it(self):
        child = subprocess.Popen(["python3", "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        MODULE.terminate_owned_group(child, graceful_seconds=0.1)
        self.assertEqual(child.returncode, -signal.SIGKILL)
        script = (MODULE_PATH.parent / "tgrab_node2a_capture_job.sh").read_text()
        self.assertIn("--kill-after=15s 575s", script)


if __name__ == "__main__":
    unittest.main()
