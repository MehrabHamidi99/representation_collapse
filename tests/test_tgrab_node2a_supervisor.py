"""Focused restart-metadata contract for the native node2a runner."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "code/ops/tgrab_node2a_supervisor.py"
SPEC = importlib.util.spec_from_file_location("node2a_supervisor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RestartMetadataTests(unittest.TestCase):
    def test_fixture_publishes_under_separate_child_path(self):
        run = Path("/scratch/runs/figure10-node2a/node2a-test")
        self.assertEqual(MODULE.fixture_output_path(run), run / "fixture")

    def test_mila_may_omit_environment_restart_count(self):
        MODULE.verify_environment_restart_count(None)

    def test_pinned_upstream_wrappers_are_hashed_and_revalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = {"model_source_root": str(root),
                    "model_source_revision": MODULE.MODEL_SOURCE_REVISION,
                    "model_wrapper_sha256": dict(MODULE.MODEL_WRAPPERS)}
            for relative in MODULE.MODEL_WRAPPERS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture source\n")
            with mock.patch.object(MODULE, "MODEL_SOURCE", root):
                with mock.patch.object(MODULE, "sha", side_effect=lambda path: MODULE.MODEL_WRAPPERS[path.relative_to(root).as_posix()]):
                    self.assertEqual(MODULE.verify_model_source(plan), plan["model_wrapper_sha256"])
                with mock.patch.object(MODULE, "sha", side_effect=lambda path: "0" * 64 if path.name == "node_emb.py" else MODULE.MODEL_WRAPPERS[path.relative_to(root).as_posix()]):
                    with self.assertRaisesRegex(SystemExit, "wrapper digest"):
                        MODULE.verify_model_source(plan)

    def test_pinned_upstream_source_plan_rejects_revision_and_wrapper_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"run_id": "node2a-20261016T120000Z-1234abcd",
                "output": str(MODULE.SCRATCH / "runs/figure10-node2a/node2a-20261016T120000Z-1234abcd"),
                "stop_file": str(MODULE.SCRATCH / "runs/figure10-node2a/node2a-20261016T120000Z-1234abcd/STOP"),
                "deadline_seconds": 570, "allocation": dict(MODULE.EXPECTED_PLAN_ALLOCATION),
                "model_source_root": str(MODULE.MODEL_SOURCE),
                "model_source_revision": MODULE.MODEL_SOURCE_REVISION,
                "model_wrapper_sha256": dict(MODULE.MODEL_WRAPPERS),
                "fixture": "synthetic-node2a-zero-update-only", "scientific_acceptance": False}
            for field, replacement in (("model_source_revision", "0" * 40),
                                       ("model_wrapper_sha256", {"model/node_emb.py": "0" * 64})):
                altered = dict(plan); altered[field] = replacement
                with self.assertRaisesRegex(SystemExit, "upstream model source"):
                    MODULE.validate_plan_contract(altered)

    def test_explicit_zero_is_accepted(self):
        MODULE.verify_environment_restart_count("0")

    def test_nonzero_empty_and_noncanonical_counts_are_rejected(self):
        for value in ("", "1", "00", "0\n"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                MODULE.verify_environment_restart_count(value)

    @staticmethod
    def slurm_record(**changes):
        fields = {"JobState": "RUNNING", "Account": "mila", "QOS": "normal",
            "Partition": "main-cpu", "NumNodes": "1", "NumCPUs": "1",
            "CPUs/Task": "1", "TimeLimit": "00:10:00", "Requeue": "0",
            "Restarts": "0", "ReqTRES": "cpu=1,mem=2G,node=1",
            "AllocTRES": "cpu=1,mem=2G,node=1"}
        fields.update(changes)
        return "JobId=123 " + " ".join(f"{key}={value}" for key, value in fields.items())

    def verify_record(self, record):
        completed = subprocess.CompletedProcess([], 0, record, "")
        with mock.patch.object(MODULE.subprocess, "run", return_value=completed):
            MODULE.verify_scheduler_record("123")

    def test_authoritative_scheduler_zero_restarts_and_requeue_are_required(self):
        self.verify_record(self.slurm_record())
        for field in ("Restarts", "Requeue"):
            with self.subTest(field=field), self.assertRaisesRegex(SystemExit, "authoritative Slurm"):
                self.verify_record(self.slurm_record(**{field: "1"}))

    def test_cgroup_memory_uses_available_ancestor_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            parent = root / "user.slice"
            leaf = parent / "job.scope"
            leaf.mkdir(parents=True)
            (parent / "memory.max").write_text(str(4 * 1024**3))
            (leaf / "memory.max").write_text(str(2 * 1024**3))
            (leaf / "memory.current").write_text("1024")
            (leaf / "memory.peak").write_text("2048")
            result = MODULE.read_cgroup_memory_files([leaf, parent, root], leaf)
        self.assertEqual(result["effective_limit_bytes"], 2 * 1024**3)
        self.assertEqual(result["memory_current_bytes"], 1024)
        self.assertEqual(result["memory_peak_bytes"], 2048)


if __name__ == "__main__":
    unittest.main()
