"""Independent mocked launcher transactions; no SDK, parsers or cluster work."""
import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "code/ops/graph_data_conversion_job.py"
spec = importlib.util.spec_from_file_location("independent_verified_conversion_job", MODULE)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


@contextmanager
def transaction(*, mutate=None, finish_failure=False, verify_failure=False):
    """Exercise actual worker with independently constructed fake dependencies."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
        base = Path(temporary)
        output = base / "attempt001"; output.mkdir()
        config = job.load_config(ROOT / job.CONFIG_PATH)
        raw_path = base / "nonexistent-synthetic-input.csv"
        config["raw"]["path"] = str(raw_path)
        plan = {"source_commit": "c" * 40, "config_sha256": "d" * 64,
                "environment": {"sha256": "e" * 64}, "stop_path": str(base / "STOP")}
        events = []; logs = []
        metadata = {"kind": "tgrab_primary_converted_v1",
                    "converter": {"revision": plan["source_commit"], "sha256": config["sources"]["converter"]["sha256"]},
                    "counts": {"raw_rows": 6144, "identical_duplicate_rows": 0,
                               "source_marker_rows": 0, "unique_offdiagonal_edges": 6144},
                    "snapshot_semantic_sha256": ["a" * 64] * 3072}
        manifest = output / "worker/converted/manifest.json"
        parser_calls = {"converter": [], "verifier": []}

        class Reporter:
            def __init__(self, path, **kwargs):
                self.path = path
            def __enter__(self):
                events.append("sdk_enter"); self.path.mkdir(parents=True); return self
            def log(self, values, *, step):
                logs.append((step, dict(values)))
            def __exit__(self, exc_type, exc, traceback):
                events.append("sdk_exit")
                job.atomic_json(self.path / "reporting-receipt.json", {
                    "state": "failed" if finish_failure or exc_type else "completed",
                    "finish_returned": not finish_failure, "mode": "offline", "last_completed_step": logs[-1][0]})
                if finish_failure:
                    raise RuntimeError("synthetic SDK finish failed")
                return False

        def convert(raw, directory, **kwargs):
            events.append("convert")
            assert events[0] == "sdk_enter"
            charged = job.read_json(output / "worker/status.json")
            assert charged["conversion_attempts"] == 1
            parser_calls["converter"].append((raw, directory, kwargs))
            directory.mkdir()
            for name in ("train", "validation", "test"):
                (directory / (name + ".json")).write_bytes(b'{"synthetic_metadata_only":true}\n')
            manifest.write_text(json.dumps(metadata))
            return manifest

        def verify(raw, path, **kwargs):
            events.append("verify")
            charged = job.read_json(output / "worker/status.json")
            assert charged["verification_attempts"] == 1 and charged["state"] == "verifying"
            parser_calls["verifier"].append((raw, path, kwargs))
            if verify_failure:
                raise ValueError("synthetic independent rejection")
            receipt = {"kind": "tgrab_primary_conversion_independent_verification_v1", "status": "verified",
                "primary_eligible": True, "raw_sha256": config["raw"]["sha256"], "raw_bytes": config["raw"]["bytes"],
                "manifest_sha256": job.sha256(manifest), "converter_sha256": config["sources"]["converter"]["sha256"],
                "converter_revision": plan["source_commit"], "config_sha256": config["reconstruction_config"]["sha256"],
                "verifier_source_sha256": config["sources"]["verifier"]["sha256"],
                "spec": {"nodes": 100, "block": 32, "train_periods": 40, "validation_periods": 4, "test_periods": 4},
                "snapshot_count": 3072, "split_count": 3, **metadata["counts"],
                "converted_snapshot_comparisons": 3072, "semantic_snapshot_comparisons": 3072,
                "split_byte_comparisons": 3, "complete_manifest_byte_comparisons": 1, "stable_input_files_checked": 7,
                "split_hashes": {name: {"filename": name + ".json", "bytes": (manifest.parent / (name + ".json")).stat().st_size,
                    "sha256": job.sha256(manifest.parent / (name + ".json")), "semantic_sha256": "b" * 64, "interval": interval}
                    for name, interval in (("train", [0, 2560]), ("validation", [2560, 2816]), ("test", [2816, 3072]))},
                "independent_of_producer": True, "scientific_acceptance": False}
            if mutate:
                mutate(receipt)
            return receipt

        converter = SimpleNamespace(__file__=str(ROOT / config["sources"]["converter"]["path"]),
            convert_primary_csv=convert, read_manifest=lambda *args: SimpleNamespace(metadata=deepcopy(metadata)))
        verifier = SimpleNamespace(__file__=str(ROOT / config["sources"]["verifier"]["path"]), verify_primary_conversion=verify)
        scheduler = {"kind": "primary_conversion_scheduler_v1", "status": "verified", "fields": {
            "JobId": "999", "Account": "mila", "QOS": "normal", "Partition": "main-cpu", "NumCPUs": "1",
            "NumNodes": "1", "NumTasks": "1", "TimeLimit": "00:10:00", "Requeue": "0",
            "ReqTRES": {"cpu": "1", "mem": "2147483648", "node": "1"},
            "AllocTRES": {"cpu": "1", "mem": "2147483648", "node": "1"}}}
        admission_path = output / "worker-admission.json"
        args = SimpleNamespace(worker_admission=admission_path, worker_admission_sha256="f" * 64,
                               plan_sha256="a" * 64)
        argv = [str(MODULE), "--worker", "--worker-admission-sha256", args.worker_admission_sha256]
        admission = {"started_monotonic": job.time.monotonic(), "plan_sha256": args.plan_sha256,
            "slurm_job_id": "999", "worker_argv": [str(Path(sys.executable).absolute()), "-B", str(MODULE), "--worker"],
            "supervisor_identity": {"pid": 99}, "scheduler": {"fixture_scheduler": True}}
        def pinned(record):
            return scheduler if "fixture_scheduler" in record else admission
        real_regular = job.regular
        def regular(path):
            if Path(path) == raw_path:
                assert events == ["sdk_enter"]
                return SimpleNamespace(st_size=config["raw"]["bytes"])
            return real_regular(path)
        helpers = {"same_process": lambda *args: True, "process_identity": lambda pid: {"pid": pid},
                   "install_spawn_journal": lambda *args: None}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(job, "pinned_record", side_effect=pinned))
            stack.enter_context(mock.patch.object(job, "runtime", return_value={}))
            stack.enter_context(mock.patch.object(job, "regular", side_effect=regular))
            stack.enter_context(mock.patch.object(job.sys, "argv", argv))
            stack.enter_context(mock.patch.dict(job.os.environ, {"SLURM_JOB_ID": "999"}))
            stack.enter_context(mock.patch.object(job.runpy, "run_path", return_value=helpers))
            stack.enter_context(mock.patch.object(job.signal, "signal"))
            stack.enter_context(mock.patch.object(job.signal, "setitimer"))
            stack.enter_context(mock.patch.object(resource, "setrlimit"))
            stack.enter_context(mock.patch("representation_collapse.reporting.ExperimentReporter", Reporter))
            modules = {"representation_collapse.graph_data": converter, "representation_collapse.graph_data_verifier": verifier}
            stack.enter_context(mock.patch.object(job.importlib, "import_module", side_effect=lambda name: modules[name]))
            result = job.worker(args, plan, config, ROOT, output)
        yield SimpleNamespace(result=result, output=output, config=config, plan=plan, events=events,
                              calls=parser_calls, logs=logs, status=job.read_json(output / "worker/status.json"))


class IndependentConversionVerificationJobTests(unittest.TestCase):
    def test_actual_worker_calls_exact_production_api_after_sdk_and_charges_before_entry(self):
        with transaction() as case:
            self.assertEqual(case.result, 0)
            self.assertEqual(case.events, ["sdk_enter", "convert", "verify", "sdk_exit"])
            self.assertEqual([step for step, _ in case.logs], [0, 1, 2])
            self.assertEqual((len(case.calls["converter"]), len(case.calls["verifier"])), (1, 1))
            raw, manifest, kwargs = case.calls["verifier"][0]
            self.assertEqual(set(kwargs), {"expected_raw_sha256", "expected_raw_bytes", "expected_manifest_sha256",
                "expected_converter_sha256", "expected_converter_revision", "config_path", "guard"})
            self.assertEqual(kwargs["expected_manifest_sha256"], job.sha256(manifest))
            self.assertEqual(kwargs["expected_raw_sha256"], case.config["raw"]["sha256"])
            self.assertEqual(kwargs["expected_converter_revision"], case.plan["source_commit"])
            self.assertEqual(kwargs["expected_converter_sha256"], case.config["sources"]["converter"]["sha256"])
            self.assertEqual(kwargs["config_path"], ROOT / case.config["reconstruction_config"]["path"])
            self.assertTrue(callable(kwargs["guard"]))

    def test_missing_or_tampered_actual_count_spec_and_split_fields_cannot_succeed(self):
        mutations = [lambda r: r.pop("raw_rows"), lambda r: r.update(source_marker_rows=1),
                     lambda r: r.update(identical_duplicate_rows=True),
                     lambda r: r.update(raw_rows=6145, identical_duplicate_rows=1),
                     lambda r: r["spec"].update(nodes=True),
                     lambda r: r["split_hashes"]["test"].update(interval=[2815, 3072]),
                     lambda r: r["split_hashes"]["test"].pop("semantic_sha256"),
                     lambda r: r["split_hashes"]["train"].update(sha256="BAD"),
                     lambda r: r.update(unexpected_field=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), transaction(mutate=mutation) as case:
                self.assertEqual(case.result, 1)
                self.assertEqual(case.status["state"], "failed")
                self.assertEqual(case.status["verification_attempts"], 1)
                self.assertEqual(len(case.calls["verifier"]), 1)
                self.assertFalse((case.output / "worker/independent-verification.json").exists())

    def test_sdk_finish_failure_is_terminal_despite_both_parsers_returning(self):
        with transaction(finish_failure=True) as case:
            self.assertEqual(case.result, 1)
            self.assertEqual(case.status["state"], "failed")
            self.assertEqual(case.status["error_type"], "RuntimeError")
            self.assertEqual((len(case.calls["converter"]), len(case.calls["verifier"])), (1, 1))
            self.assertTrue((case.output / "worker/independent-verification.json").is_file())
            self.assertNotIn("manifest_sha256", case.status)

    def test_verifier_failure_cannot_retry_or_emit_success_step(self):
        with transaction(verify_failure=True) as case:
            self.assertEqual(case.result, 1)
            self.assertEqual(case.status["verification_attempts"], 1)
            self.assertEqual(len(case.calls["verifier"]), 1)
            self.assertEqual([step for step, _ in case.logs], [0, 1])
            self.assertTrue((case.output / "worker/conversion-receipt.json").is_file())

    def test_final_current_files_and_verification_receipt_are_bound(self):
        for target in ("worker/independent-verification.json", "worker/converted/manifest.json",
                       "worker/converted/test.json"):
            with self.subTest(target=target), transaction() as case:
                self.assertEqual(case.result, 0)
                records = job.inventory(case.output, max_files=260, max_bytes=469762048, hashes=True)
                job.validate_terminal_artifacts(records, case.output, case.plan, case.config, lambda: None)
                path = case.output / target
                path.write_bytes(path.read_bytes() + b" ")
                changed = job.inventory(case.output, max_files=260, max_bytes=469762048, hashes=True)
                with self.assertRaises(job.ConversionJobError):
                    job.validate_terminal_artifacts(changed, case.output, case.plan, case.config, lambda: None)

    def test_source_limits_clock_and_source_pins_are_not_weakened(self):
        source = MODULE.read_text(); tree = ast.parse(source, feature_version=(3, 10))
        self.assertEqual(tree.body[2].names[0].name, "time")
        self.assertEqual(tree.body[3].targets[0].id, "PROCESS_STARTED")
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        worker = ast.get_source_segment(source, functions["worker"])
        supervisor = ast.get_source_segment(source, functions["supervise"])
        self.assertIn("deadline = started + 590", worker)
        self.assertIn("started = PROCESS_STARTED; deadline = started + 600", supervisor)
        self.assertEqual(worker.count("verify_primary_conversion("), 1)
        self.assertNotIn("fixture_only", worker); self.assertNotIn("spec=", worker)
        self.assertLess(worker.index("with ExperimentReporter"), worker.index('regular(config["raw"]["path"])'))
        self.assertLess(worker.index('verification_attempts=1'), worker.index("verifier.verify_primary_conversion"))
        config = job.load_config(ROOT / job.CONFIG_PATH)
        self.assertEqual(config["resources"], {"cpus": 1, "memory_bytes": 2147483648, "gpus": 0, "wall_seconds": 600, "work_seconds": 590})
        self.assertEqual(set(config["sources"]), {"converter", "conversion_protocol", "verifier", "verification_protocol",
                         "ownership_helper", "reporter", "submission_controller", "submission_protocol"})
        for name in config["sources"]:
            record = config["sources"][name]
            self.assertEqual(hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
