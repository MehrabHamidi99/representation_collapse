"""Allocation guards with real local Bash and a stub sbatch, never a scheduler.

The cluv API fixture mirrors the relevant0.1.5 command shape. Tests neither
load native cluv configuration nor touch its cache, SSH or provider services.
Synthetic approvals/target responses in the execution test are unit fixtures.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "competence_submission_guard_under_test", ROOT / "code/ops/submit_competence_recovery.py")
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


@dataclass(frozen=True)
class SubmissionFixture:
    cluster: str
    remote: object
    job_script: Path
    sbatch_args: dict
    program_args: list[str]
    sbatch_command: str
    n_chunks: None
    git_commit: str


@dataclass(frozen=True)
class PartialConfigFixture:
    env: dict = field(default_factory=dict)
    project_dir: str = ""
    job_script_path: str = ops.WRAPPER_PATH
    sbatch_args: dict = field(default_factory=dict)
    results_path: str = ""


def job_environment_fixture(cluster, git_commit, config):
    return config.env | {"GIT_COMMIT": git_commit, "CLUV_CLUSTER": cluster}


def sbatch_command_fixture(job_script, sbatch_args, program_args, env_vars, project_dir=None):
    # Preserve native0.1.5's environment prefix, login-shell shape and literal
    # program-argument quoting. Local fixtures contain no variable expansions.
    environment = " ".join(name + "=" + value for name, value in env_vars.items())
    prefix = environment + "; " if environment else ""
    cd = "cd " + str(project_dir) + " && " if project_dir else ""
    flags = ["--" + name + ("" if value is True else "=" + str(value))
             for name, value in sbatch_args.items()]
    body = prefix + cd + "sbatch --parsable " + " ".join(flags) + " " + str(job_script) + " " + shlex.join(program_args)
    return "bash --login -c " + shlex.quote(body)


class CompetenceSubmissionGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rc-comp-guard-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.marker = self.root / "stub-sbatch-invocations"
        stub = self.root / "sbatch"
        stub.write_text("#!/bin/sh\nprintf 'called\\n' >> \"$RC_COMP_TEST_MARKER\"\nprintf '123\\n'\n")
        stub.chmod(0o700)
        self.config = SimpleNamespace(
            env={"PATH": str(self.root) + ":/usr/bin:/bin", "UV_OFFLINE": "1"},
            job_script_path=Path(ops.WRAPPER_PATH), project_dir=self.source)
        self.submit_api = ModuleType("cluv.cli.submit")
        self.submit_api.get_job_env_vars = job_environment_fixture
        self.submit_api.get_sbatch_command = sbatch_command_fixture
        cluv = ModuleType("cluv"); cluv.__path__ = []
        cli = ModuleType("cluv.cli"); cli.__path__ = []
        cluv.cli = cli; cli.submit = self.submit_api
        self.modules = {"cluv": cluv, "cluv.cli": cli, "cluv.cli.submit": self.submit_api}
        self.module_patch = patch.dict(sys.modules, self.modules)
        self.module_patch.start(); self.addCleanup(self.module_patch.stop)
        self.version_patch = patch.object(ops, "version", return_value="0.1.5")
        self.version_patch.start(); self.addCleanup(self.version_patch.stop)

    def submission(self, *, config=None, remote=None, resources=None, program=None):
        config = config or self.config
        resources = resources or {"nodes": "1", "ntasks": "1", "cpus-per-task": "4",
                                  "mem": "16G", "time": "00:30:00", "gres": "gpu:test:1",
                                  "no-requeue": True, "job-name": "cluv-rc-comp-v2-unit-0"}
        program = program or ["--source-commit", "a" * 40]
        command = sbatch_command_fixture(config.job_script_path, resources, program,
                    job_environment_fixture("fir", "a" * 40, config), config.project_dir)
        return SubmissionFixture("fir", remote, Path(ops.WRAPPER_PATH), resources,
                                 program, command, None, "a" * 40)

    def run_local_stub(self, guarded, additions=None):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("SBATCH_") and key != "SLURM_CLUSTERS"}
        environment["RC_COMP_TEST_MARKER"] = str(self.marker)
        environment.update(additions or {})
        # Execute the guarded command's actual bash --login -c shape. The only
        # sbatch available in its explicit PATH is our local harmless fixture.
        return subprocess.run(shlex.split(guarded.sbatch_command), env=environment,
                              capture_output=True, text=True, timeout=10)

    def test_clean_login_command_invokes_local_stub_exactly_once(self):
        original = self.submission()
        guarded = ops.guard_native_submission(original, self.config)
        result = self.run_local_stub(guarded)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "123\n")
        self.assertEqual(self.marker.read_text(), "called\n")
        words = shlex.split(guarded.sbatch_command)
        self.assertEqual(words[:3], ["bash", "--login", "-c"])
        self.assertTrue(words[3].startswith(ops.SBATCH_ENVIRONMENT_GUARD))
        self.assertEqual(words[3].count("sbatch --parsable"), 1)
        self.assertEqual(guarded.sbatch_args, original.sbatch_args)
        self.assertEqual(guarded.program_args, original.program_args)
        self.assertEqual(guarded.git_commit, original.git_commit)

    def test_inherited_array_and_federation_flags_prevent_all_invocation(self):
        guarded = ops.guard_native_submission(self.submission(), self.config)
        for key in ("SBATCH_ARRAY_INX", "SBATCH_CLUSTERS", "SLURM_CLUSTERS", "SBATCH_CPUS_PER_TASK"):
            for value in ("private-fixture-value", ""):
                with self.subTest(key=key, empty=value == ""):
                    result = self.run_local_stub(guarded, {key: value})
                    self.assertEqual(result.returncode, 2)
                    self.assertFalse(self.marker.exists())
                    self.assertEqual(result.stdout, "")
                    self.assertNotIn("private-fixture-value", result.stderr)
                    self.assertNotIn(key, result.stderr)

    def test_configured_scheduler_keys_are_rejected_without_value_logging(self):
        ops.validate_cluv_environment({"UV_OFFLINE": "1", "WANDB_MODE": "offline"})
        for key in ("SBATCH_ARRAY_INX", "SBATCH_CLUSTERS", "SBATCH_GRES", "SLURM_CLUSTERS"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as error:
                    ops.validate_cluv_environment({key: "private-fixture-value"})
                self.assertNotIn("private-fixture-value", str(error.exception))
                bad = SimpleNamespace(env={key: "private-fixture-value"},
                    job_script_path=self.config.job_script_path, project_dir=self.source)
                with self.assertRaises(ValueError):
                    ops.guard_native_submission(self.submission(config=bad), bad)

    def test_malformed_or_changed_native_commands_fail_before_local_invocation(self):
        original = self.submission()
        for command in ("sh -c true", original.sbatch_command + "; true",
                        shlex.join(["bash", "-c", shlex.split(original.sbatch_command)[3]]),
                        "bash --login -c 'sbatch --parsable altered-wrapper.sh'"):
            with self.subTest(command_kind=command.split()[0]):
                with self.assertRaises(ValueError):
                    ops.guard_native_submission(replace(original, sbatch_command=command), self.config)
                self.assertFalse(self.marker.exists())

    def test_other_native_version_or_wrapper_fails_closed(self):
        original = self.submission()
        with patch.object(ops, "version", return_value="0.2.0"):
            with self.assertRaises(ValueError):
                ops.guard_native_submission(original, self.config)
        bad = SimpleNamespace(env=self.config.env, job_script_path=Path("other.sh"), project_dir=self.source)
        with self.assertRaises(ValueError):
            ops.guard_native_submission(original, bad)
        self.assertFalse(self.marker.exists())

    def test_generated_python_bodies_compile_without_execution(self):
        self.assertEqual(len(ops.validate_embedded_python()), 5)
        for name in ("PREFLIGHT", "STAGE_RECEIPT"):
            with self.subTest(program=name), patch.object(ops, name, " if invalid:\n"):
                with self.assertRaises(ValueError):
                    ops.validate_embedded_python()

    def test_mocked_native_submit_receives_exact_journaled_guarded_command(self):
        # Exercise coordinator integration with pure cluv/remote fixtures. No
        # owner approval, cluster assertion or provider access is created.
        project = self.root / "project"
        project.mkdir()
        args = SimpleNamespace(journal=project / "runs/ops/unit-journal.json", cell=0,
                               source_commit="a" * 40)
        target = {"scratch_root": str(self.root), "source_directory": str(self.source),
                  "account": "test-account", "partition": "test-gpu", "qos": "normal", "gres": "gpu:test:1"}
        plan = {"targets": {"fir": target, "nibi": dict(target)},
                "cells": [{"cell": 0, "cluster": "fir", "predecessor_cell": None,
                           "model": "gru", "task": "categorical", "seed": 0}]}
        hashes = {ops.PROTOCOL_PATH: "b" * 64, ops.CANARY_PATH: "c" * 64}
        approval = {"approval_binding": {"fixture_only": True}}
        preview = {"run_prefix": "unit-fixture", "run_id": "unit-fixture-0-gru-categorical-s0-lr001",
                   "job_name_prefix": "cluv-rc-comp-v2-unit-"}
        remote = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps({"queue": [], "sacct": []}), stderr="")))
        class NativeConfigFixture:
            env = {}
            sbatch_args = {}
            clusters = {host: PartialConfigFixture() for host in ("fir", "nibi")}
            def get_cluster_config(self, host):
                partial = self.clusters[host]
                return SimpleNamespace(env=self.env | partial.env,
                    job_script_path=Path(partial.job_script_path), project_dir=Path(partial.project_dir))
        native = NativeConfigFixture()
        async def submissions(cluster, native_remote, **kwargs):
            resources = {}
            for option in kwargs["sbatch_args"]:
                key, separator, value = option[2:].partition("=")
                resources[key] = value if separator else True
            config = native.get_cluster_config(cluster)
            return [self.submission(config=config, remote=native_remote,
                                    resources=resources, program=kwargs["program_args"])]
        cache = ModuleType("cluv.cache"); cache.save_job = Mock()
        login = ModuleType("cluv.cli.login"); login.get_remote_without_2fa_prompt = AsyncMock(return_value=remote)
        config_api = ModuleType("cluv.config"); config_api.get_cluv_config = Mock(return_value=native)
        self.submit_api.get_submissions = AsyncMock(side_effect=submissions)
        self.submit_api.submit_job = AsyncMock(return_value=SimpleNamespace(job_id=123))
        self.submit_api.display_commands = SimpleNamespace(set=Mock())
        with patch.dict(sys.modules, {"cluv.cache": cache, "cluv.cli.login": login, "cluv.config": config_api}), \
             patch.object(ops, "ignored_path"), patch.object(ops, "target_verified", return_value=True):
            result = asyncio.run(ops.execute(args, plan, hashes, approval, preview))
        submitted = self.submit_api.submit_job.await_args.args[0]
        journal = json.loads(args.journal.read_text())
        self.assertEqual(result["state"], "submitted")
        self.assertEqual(journal["cells"]["0"]["command"], submitted.sbatch_command)
        self.assertTrue(shlex.split(submitted.sbatch_command)[3].startswith(ops.SBATCH_ENVIRONMENT_GUARD))
        self.assertEqual(journal["cells"]["0"]["inherited_scheduler_input_guard"],
                         "same_login_shell_before_native_sbatch")
        self.assertEqual(self.submit_api.submit_job.await_count, 1)
        self.assertEqual(cache.save_job.call_count, 1)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
