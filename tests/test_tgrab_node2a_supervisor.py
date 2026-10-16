"""Focused restart-metadata contract for the native node2a runner."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "code/ops/tgrab_node2a_supervisor.py"
SPEC = importlib.util.spec_from_file_location("node2a_supervisor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RestartMetadataTests(unittest.TestCase):
    def test_mila_may_omit_environment_restart_count(self):
        MODULE.verify_environment_restart_count(None)

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


if __name__ == "__main__":
    unittest.main()
