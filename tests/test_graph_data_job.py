"""Preparation fixtures only: no primary bytes, SDK, numerical imports or Slurm."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "code/ops/graph_data_conversion_job.py"
CONFIG = ROOT / "configs/graph/primary-data-conversion-v1.json"
spec = importlib.util.spec_from_file_location("graph_data_conversion_job_fixture", MODULE)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


class ConversionPreparationTests(unittest.TestCase):
    def test_module_is_stdlib_only_and_clock_precedes_other_imports(self):
        tree = ast.parse(MODULE.read_text(), feature_version=(3, 10))
        self.assertEqual(tree.body[2].names[0].name, "time")
        self.assertEqual(tree.body[3].targets[0].id, "PROCESS_STARTED")
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(any(a.name in ("numpy", "torch", "wandb") for a in n.names) for n in imports))

    def test_pinned_candidate_requires_external_plan_and_all_policy_fields_frozen(self):
        config = job.load_config(CONFIG)
        self.assertTrue(config["execution_authorized"])
        for key in job.CONFIG_PIN_FIELDS:
            record = config["sources"][key]
            self.assertEqual(record["sha256"], job.sha256(ROOT / record["path"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "config.json"
            for mutation in (lambda c: c["limits"].update(raw_rows=999999),
                             lambda c: c["resources"].update(wall_seconds=601),
                             lambda c: c["allocation"].update(retries=1)):
                modified = deepcopy(config); mutation(modified); path.write_text(json.dumps(modified))
                with self.assertRaisesRegex(job.ConversionJobError, "policy changed"):
                    job.load_config(path)
            changed = deepcopy(config); changed["execution_authorized"] = True
            changed["sources"]["converter"]["sha256"] = "a" * 64
            changed["sources"]["conversion_protocol"]["sha256"] = "b" * 64
            path.write_text(json.dumps(changed))
            self.assertTrue(job.load_config(path)["execution_authorized"])

    def test_real_reporter_constructor_accepts_actual_metadata_without_sdk(self):
        from representation_collapse.reporting import ExperimentReporter
        config = job.load_config(CONFIG)
        for key in job.CONFIG_PIN_FIELDS:
            config["sources"][key]["sha256"] = "a" * 64
        plan = {"source_commit": "c" * 40, "config_sha256": "d" * 64, "environment": {"sha256": "e" * 64}}
        scalar_config, identities = job.reporter_metadata(plan, config)
        with tempfile.TemporaryDirectory() as temporary, mock.patch("importlib.import_module", side_effect=AssertionError("SDK forbidden")) as sdk:
            path = Path(temporary) / "reporting"
            reporter = ExperimentReporter(path, project=config["reporting"]["project"], mode="offline", config=scalar_config, identities=identities)
            sdk.assert_not_called()
            self.assertEqual(reporter._state, "new")
            self.assertFalse(path.exists())
            self.assertTrue(all(len(value) in (40, 64) for value in reporter.identities.values()))
            self.assertEqual(reporter.config["module_path"], job.MODULE_PATH)
            with self.assertRaises(ValueError):
                ExperimentReporter(path, project="representation-collapse", identities={**identities, "path": job.MODULE_PATH}, config=scalar_config)

    def test_bad_or_missing_admission_is_rejected_before_subprocess_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "plan.json"; path.write_text('{}')
            args = SimpleNamespace(plan=path, plan_sha256="0" * 64, worker=False)
            with mock.patch.object(job.subprocess, "check_output") as git:
                with self.assertRaisesRegex(job.ConversionJobError, "plan pin"):
                    job.verify_plan(args)
                args.plan_sha256 = job.sha256(path)
                with self.assertRaisesRegex(job.ConversionJobError, "externally approved"):
                    job.verify_plan(args)
                git.assert_not_called()
            self.assertEqual([p.name for p in Path(temporary).iterdir()], ["plan.json"])

    def test_receipt_pin_requires_concrete_matching_sha_and_unique_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "receipt.json"; path.write_text('{"status":"verified"}')
            for pin in (None, "0" * 64):
                with self.assertRaises(job.ConversionJobError):
                    job.pinned_record({"path": str(path), "sha256": pin})
            self.assertEqual(job.pinned_record({"path": str(path), "sha256": job.sha256(path)})["status"], "verified")
            path.write_text('{"status":"verified","status":"failed"}')
            with self.assertRaisesRegex(job.ConversionJobError, "duplicate"):
                job.read_json(path)

    def test_output_inventory_hashes_regular_files_and_rejects_bounds_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve(); (path / "file.json").write_text('{}')
            receipt = job.inventory(path, max_files=1, max_bytes=2, hashes=True)
            self.assertEqual(receipt["bytes"], 2)
            self.assertEqual(receipt["files"][0]["sha256"], job.sha256(path / "file.json"))
            with self.assertRaisesRegex(job.ConversionJobError, "bound"):
                job.inventory(path, max_files=0, max_bytes=2)
            with self.assertRaisesRegex(job.ConversionJobError, "bound"):
                job.inventory(path, max_files=1, max_bytes=1)
            (path / "alias").symlink_to(path / "file.json")
            with self.assertRaisesRegex(job.ConversionJobError, "symlink"):
                job.inventory(path, max_files=3, max_bytes=10)

    def test_cgroup_takes_physical_ancestor_cap_and_unknown_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary); root = path / "cg"; child = root / "job"; child.mkdir(parents=True)
            (root / "memory.max").write_text("2147483648")
            (child / "memory.max").write_text("max")
            proc = path / "proc"; proc.write_text("0::/job\n")
            self.assertEqual(job.kernel_memory_limit(proc, root)["physical_memory_limit_bytes"], 2147483648)
            (root / "memory.max").write_text("max")
            with self.assertRaisesRegex(job.ConversionJobError, "cap unavailable"):
                job.kernel_memory_limit(proc, root)

    def test_worker_source_orders_reporting_before_primary_and_converter(self):
        tree = ast.parse(MODULE.read_text())
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
        source = ast.get_source_segment(MODULE.read_text(), worker)
        self.assertLess(source.index("signal.setitimer"), source.index("install_spawn_journal"))
        self.assertLess(source.index("install_spawn_journal"), source.index("with ExperimentReporter"))
        self.assertLess(source.index("with ExperimentReporter"), source.index('regular(config["raw"]["path"])'))
        self.assertLess(source.index("with ExperimentReporter"), source.index('importlib.import_module("representation_collapse.graph_data")'))
        self.assertLess(source.index("conversion_attempts=1"), source.index("converter.convert_primary_csv"))
        self.assertNotIn("open_train_validation", source)
        self.assertNotIn("RLIMIT_AS", source)
        self.assertNotIn("fixture_only=True", source)

    def test_runtime_rejects_platform_before_distribution_or_cgroup_reads(self):
        with mock.patch.object(job.sys, "platform", "darwin"), mock.patch.object(job.importlib.metadata, "version") as version, mock.patch.object(job, "kernel_memory_limit") as memory:
            with self.assertRaisesRegex(job.ConversionJobError, "Linux"):
                job.runtime(job.load_config(CONFIG))
            version.assert_not_called(); memory.assert_not_called()

    def test_scheduler_query_only_actual_job_and_rejects_missing_duplicate_mismatched_fields(self):
        text = "JobId=999 Account=mila QOS=normal Partition=main-cpu NumCPUs=1 NumNodes=1 NumTasks=1 TimeLimit=00:10:00 Requeue=0 ReqTRES=cpu=1,mem=2G,node=1 AllocTRES=cpu=1,mem=2048M,node=1 JobName=private-uncollected"
        with mock.patch.object(job.subprocess, "run", return_value=SimpleNamespace(stdout=text)) as query:
            receipt = job.scheduler_receipt("999")
            self.assertEqual(set(receipt["fields"]), {"JobId", "Account", "QOS", "Partition", "NumCPUs", "NumNodes", "NumTasks", "TimeLimit", "Requeue", "ReqTRES", "AllocTRES"})
            query.assert_called_once_with(["scontrol", "show", "job", "-o", "999"], capture_output=True, text=True, timeout=5, check=True)
        for invalid in (text.replace("NumTasks=1", ""), text + " QOS=normal", text.replace("NumCPUs=1", "NumCPUs=2"), text.replace("Requeue=0", "Requeue=1"), text.replace("TimeLimit=00:10:00", "TimeLimit=00:30:00"), text.replace("mem=2G", "mem=4G"), text.replace("AllocTRES=cpu=1", "AllocTRES=gres/gpu=1,cpu=1")):
            with mock.patch.object(job.subprocess, "run", return_value=SimpleNamespace(stdout=invalid)):
                with self.assertRaises(job.ConversionJobError):
                    job.scheduler_receipt("999")

    def test_complete_fixture_plan_binds_sources_pins_argv_limits_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve(); source = base / "source"; source.mkdir()
            config = job.load_config(CONFIG); config["execution_authorized"] = True
            for name, record in config["sources"].items():
                path = source / record["path"]; path.parent.mkdir(parents=True, exist_ok=True)
                original = ROOT / record["path"]
                path.write_bytes(original.read_bytes() if original.is_file() else b"fixture protocol only\n")
                record["sha256"] = job.sha256(path)
            # Accepted fixed helper/reporter pins remain exact.
            for relative in (job.MODULE_PATH, job.SHELL_PATH, job.REQUEST_PATH, "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]):
                path = source / relative; path.parent.mkdir(parents=True, exist_ok=True)
                original = ROOT / relative
                path.write_bytes(original.read_bytes() if original.is_file() else b"fixture request only\n")
            config_path = source / job.CONFIG_PATH; config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config)); config_sha = job.sha256(config_path)
            plan_path = base / "plan.json"; output = base / "attempt001"
            plan = {"kind": "primary_data_conversion_single_submission_plan", "owner": "root coordinator", "protocol_id": job.PROTOCOL_ID,
                "execution_authorized": True, "training_authorized": False, "scientific_acceptance": False,
                "max_submissions": 1, "retries": 0, "source_root": str(source), "source_commit": "c" * 40,
                "config_sha256": config_sha, "resources": config["resources"], "allocation": config["allocation"], "limits": config["limits"],
                "output": str(output), "stop_path": str(base / "STOP")}
            required = {job.MODULE_PATH, job.SHELL_PATH, job.CONFIG_PATH, job.REQUEST_PATH, "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]}
            required.update(record["path"] for record in config["sources"].values())
            plan["source_files"] = {relative: job.sha256(source / relative) for relative in required}
            for name in ("implementation_review", "reporting_readiness", "environment", "storage_gate", "owner_approval", "submission_claim"):
                value = {"status": "verified", "config_sha256": config_sha}
                if name == "environment":
                    value.update(python="3.10.19", wandb="0.30.0", python_executable=config["python"])
                if name in ("owner_approval", "submission_claim"):
                    value.update(allocation=config["allocation"], resources=config["resources"])
                    value.update({key: plan[key] for key in ("source_commit", "source_files", "output", "stop_path", "limits", "max_submissions", "retries")})
                if name == "submission_claim":
                    value.update(kind="root_exclusive_submission_claim", claim_id="fixture-only", claimed_submissions=1, exclusive=True)
                if name == "storage_gate":
                    value.update(used_bytes=1, reserved_bytes=config["limits"]["conversion_bytes"] + config["limits"]["evidence_bytes"], source_root=str(source), output=str(output))
                path = base / (name + ".json"); path.write_text(json.dumps(value))
                plan[name] = {"path": str(path), "sha256": job.sha256(path)}
            args = SimpleNamespace(plan=plan_path, plan_sha256="0" * 64, worker=False)
            def freeze():
                plan["supervisor_argv"] = [config["python"], "-B", str(source / job.MODULE_PATH), "--plan", str(plan_path), "--plan-sha256", "{plan_sha256}"]
                plan_path.write_text(json.dumps(plan))
                args.plan_sha256 = job.sha256(plan_path)
            freeze()
            with mock.patch.object(job, "SCRATCH", base), mock.patch.object(job, "__file__", str(source / job.MODULE_PATH)), mock.patch.object(job.Path, "cwd", return_value=source), mock.patch.object(job.sys, "executable", config["python"]), mock.patch.object(job.subprocess, "check_output", return_value="c" * 40), mock.patch.dict(job.os.environ, {"SLURM_JOB_ID": "999", "SLURM_CPUS_PER_TASK": "1"}, clear=True):
                with mock.patch.object(job.sys, "argv", [str(source / job.MODULE_PATH), "--plan", str(plan_path), "--plan-sha256", args.plan_sha256]):
                    self.assertEqual(job.verify_plan(args)[3], output)
                plan["source_files"].pop(config["sources"]["converter"]["path"]); freeze()
                with self.assertRaisesRegex(job.ConversionJobError, "source closure"):
                    job.verify_plan(args)

    def test_fake_supervisor_failure_keeps_partial_output_and_only_owned_cleanup(self):
        config = job.load_config(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve(); output = base / "attempt001"; source = base / "source"
            (source / job.CONFIG_PATH).parent.mkdir(parents=True)
            (source / job.CONFIG_PATH).write_text(CONFIG.read_text())
            plan_path = base / "plan.json"; plan_path.write_text('{}')
            args = SimpleNamespace(plan=plan_path, plan_sha256=job.sha256(plan_path))
            plan = {"source_commit": "c" * 40, "stop_path": str(base / "STOP")}
            child = mock.Mock(pid=123, returncode=1); child.poll.return_value = 1
            owned = mock.Mock(known={123: {"pid": 123, "start": "fixture"}}, signals=[], cleanup_reconciliation={"123": "original_process_exited"})
            owned.cleanup.return_value = []
            helpers = {"process_identity": lambda pid: {"pid": pid, "start": "fixture"}, "OwnedProcesses": mock.Mock(return_value=owned)}
            with mock.patch.object(job, "PROCESS_STARTED", job.time.monotonic()), mock.patch.object(job.sys, "platform", "linux"), mock.patch.object(job.os, "sched_getaffinity", return_value={2}, create=True), mock.patch.object(job.os, "sched_setaffinity", create=True), mock.patch.object(job, "runtime", return_value={"fixture": True}), mock.patch.object(job, "scheduler_receipt", return_value={"fixture": True}), mock.patch.object(job.runpy, "run_path", return_value=helpers), mock.patch.object(job.subprocess, "Popen", return_value=child) as spawn, mock.patch.dict(job.os.environ, {"SLURM_JOB_ID": "999"}):
                self.assertEqual(job.supervise(args, plan, config, source, output), 1)
            owned.cleanup.assert_called_once()
            receipt = job.read_json(output / "supervisor-receipt.json")
            self.assertEqual(receipt["state"], "failed")
            self.assertEqual(receipt["worker_exit_code"], 1)
            self.assertEqual(spawn.call_args.kwargs["env"]["GOMAXPROCS"], "1")
            self.assertTrue((output / "worker.log").exists())
            self.assertTrue((output / "evidence-inventory.json").exists())
            admission = job.read_json(output / "worker-admission.json")
            self.assertEqual(spawn.call_args.args[0][:-2], admission["worker_argv"])
            self.assertEqual(spawn.call_args.args[0][-1], job.sha256(output / "worker-admission.json"))

    def test_reporting_failure_blocks_raw_access_and_conversion_failure_retains_attempt(self):
        import resource
        for reporting_fails in (True, False):
            with self.subTest(reporting_fails=reporting_fails), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary).resolve(); output = base / "attempt001"; output.mkdir()
                config = job.load_config(CONFIG)
                raw = base / "fixture.csv"; raw.write_text("{}")
                config["raw"]["path"] = str(raw); config["raw"]["bytes"] = 2
                config["sources"]["converter"]["sha256"] = job.sha256(ROOT / config["sources"]["converter"]["path"])
                config["sources"]["conversion_protocol"]["sha256"] = "b" * 64
                plan = {"source_commit": "c" * 40, "config_sha256": "d" * 64, "environment": {"sha256": "e" * 64}, "stop_path": str(base / "STOP")}
                scheduler = base / "scheduler.json"
                scheduler.write_text(json.dumps({"kind": "primary_conversion_scheduler_v1", "status": "verified", "fields": {
                    "JobId": "999", "Account": "mila", "QOS": "normal", "Partition": "main-cpu", "NumCPUs": "1", "NumNodes": "1", "NumTasks": "1", "TimeLimit": "00:10:00", "Requeue": "0", "ReqTRES": {"cpu": "1", "mem": "2147483648", "node": "1"}, "AllocTRES": {"cpu": "1", "mem": "2147483648", "node": "1"}}}))
                args = SimpleNamespace(worker_admission=output / "worker-admission.json", plan_sha256="f" * 64)
                prefix = [str(Path(sys.executable).absolute()), "-B", str(ROOT / job.MODULE_PATH), "--worker"]
                admission = {"started_monotonic": job.time.monotonic(), "plan_sha256": args.plan_sha256, "slurm_job_id": "999",
                    "worker_argv": prefix, "supervisor_identity": {"pid": 42}, "scheduler": {"path": str(scheduler), "sha256": job.sha256(scheduler)}}
                args.worker_admission.write_text(json.dumps(admission)); args.worker_admission_sha256 = job.sha256(args.worker_admission)
                helpers = {"same_process": lambda a, b: True, "process_identity": lambda pid: {"pid": pid}, "install_spawn_journal": mock.Mock()}
                reporter = mock.MagicMock()
                if reporting_fails:
                    reporter.__enter__.side_effect = RuntimeError("fixture reporting initialization failure")
                converter = SimpleNamespace(__file__=str(ROOT / config["sources"]["converter"]["path"]), convert_primary_csv=mock.Mock(side_effect=ValueError("fixture lexical failure")))
                with mock.patch.object(job.sys, "argv", [str(ROOT / job.MODULE_PATH), "--worker", "--worker-admission-sha256", args.worker_admission_sha256]), mock.patch.dict(job.os.environ, {"SLURM_JOB_ID": "999"}), mock.patch.object(job, "runtime", return_value={}), mock.patch.object(job.runpy, "run_path", return_value=helpers), mock.patch.object(job.signal, "signal"), mock.patch.object(job.signal, "setitimer"), mock.patch.object(resource, "setrlimit"), mock.patch("representation_collapse.reporting.ExperimentReporter", return_value=reporter), mock.patch.object(job.importlib, "import_module", return_value=converter) as conversion_import, mock.patch.object(job, "regular", wraps=job.regular) as files:
                    self.assertEqual(job.worker(args, plan, config, ROOT, output), 1)
                status = job.read_json(output / "worker/status.json")
                self.assertEqual(status["state"], "failed")
                self.assertEqual(status["conversion_attempts"], 0 if reporting_fails else 1)
                if reporting_fails:
                    conversion_import.assert_not_called(); converter.convert_primary_csv.assert_not_called()
                    self.assertNotIn(mock.call(str(raw)), files.call_args_list)
                else:
                    converter.convert_primary_csv.assert_called_once()
                    self.assertEqual(converter.convert_primary_csv.call_args.kwargs["expected_raw_bytes"], 2)
                    self.assertNotIn("fixture_only", converter.convert_primary_csv.call_args.kwargs)
                    reporter.__exit__.assert_called_once()

    def test_only_exact_sdk_debug_core_symlink_is_censused_without_following(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            logs = base / "worker/reporting/wandb/offline-run-20261015_102718-fixture/logs"; logs.mkdir(parents=True)
            (base / "worker/reporting/reporting-receipt.json").write_text(json.dumps({"mode": "offline", "sdk_version_required": "0.30.0", "run_id": "fixture"}))
            link = logs / "debug-core.log"; link.symlink_to("/does/not/exist/private-target")
            with self.assertRaises(job.ConversionJobError):
                job.inventory(base, max_files=20, max_bytes=1024)
            receipt = job.inventory(base, max_files=20, max_bytes=1024, sdk_metadata=True, hashes=True)
            self.assertIn({"path": str(link.relative_to(base)), "bytes": link.lstat().st_size, "skipped_sdk_symlink": True}, receipt["files"])
            (logs / "arbitrary.log").symlink_to("/does/not/exist")
            with self.assertRaises(job.ConversionJobError):
                job.inventory(base, max_files=20, max_bytes=1024, sdk_metadata=True)

    def test_live_inventory_tolerates_only_bounded_vanishing_sdk_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve(); sdk = base / "worker/reporting/wandb"; sdk.mkdir(parents=True)
            file = sdk / "temporary.log"; file.write_text("fixture")
            original = Path.lstat
            def disappear(path, *args, **kwargs):
                if path == file:
                    raise FileNotFoundError("fixture disappearance")
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "lstat", disappear):
                receipt = job.inventory(base, max_files=20, max_bytes=1024, sdk_metadata=True, live=True)
                self.assertEqual(receipt["vanished_sdk_files"], [str(file.relative_to(base))])
                with self.assertRaises(FileNotFoundError):
                    job.inventory(base, max_files=20, max_bytes=1024, sdk_metadata=True, live=False)

    def test_exact_nested_comparison_rejects_boolean_integer_confusion(self):
        self.assertFalse(job.exact({"cpus": 1, "gpus": 0}, {"cpus": True, "gpus": False}))


if __name__ == "__main__":
    unittest.main()
