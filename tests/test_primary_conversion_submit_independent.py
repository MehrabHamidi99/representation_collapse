"""Independent local fake-admission fixtures; no native cluv API is called."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "code/ops/submit_primary_conversion.py"
spec = importlib.util.spec_from_file_location("independent_primary_submit", PATH)
controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
FIELDS = ("source_commit", "source_files", "output", "stop_path", "limits", "max_submissions", "retries")


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim(plan):
    return {"kind": "root_exclusive_submission_claim", "status": "verified", "exclusive": True,
            "claim_id": plan["claim_id"], "claimed_submissions": 1, **{key: plan[key] for key in FIELDS}}


@dataclass(frozen=True)
class Row:
    cluster: str
    remote: object
    job_script: Path
    sbatch_args: dict
    program_args: list
    sbatch_command: str
    n_chunks: object
    git_commit: str


def row(context):
    p = context["plan"]
    resources = {"account": "mila", "qos": "normal", "partition": "main-cpu", "nodes": "1", "ntasks": "1",
                 "cpus-per-task": "1", "mem": "2G", "time": "00:10:00", "no-requeue": True,
                 "job-name": "cluv-rc-primary-conversion-v1", "output": p["logs"] + "/mila_%j/slurm-%j.out",
                 "chdir": str(context["root"]), "export": "ALL"}
    arguments = ["--plan", str(context["plan_path"]), "--plan-sha256", context["plan_sha256"]]
    flags = ["--" + key if value is True else "--" + key + "=" + value for key, value in resources.items()]
    command = f"GIT_COMMIT={p['source_commit']} CLUV_CLUSTER=mila; cd {context['root']} && sbatch --parsable " + shlex.join([*flags, controller.SHELL, *arguments])
    return Row("mila", None, Path(controller.SHELL), resources, arguments, shlex.join(["bash", "--login", "-c", command]), None, p["source_commit"])


@contextmanager
def packet():
    with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
        base = Path(temporary); source = base / "sources/source001"; source.mkdir(parents=True)
        run = base / "runs/review-fixture001"; inputs = run / "inputs"; inputs.mkdir(parents=True)
        config = json.loads((ROOT / controller.CONFIG).read_bytes())
        config["sources"].update(controller={"path": controller.CONTROLLER, "sha256": None},
                                 controller_protocol={"path": controller.PROTOCOL, "sha256": None})
        required = {controller.WORKER, controller.SHELL, controller.CONFIG, controller.REQUEST,
                    "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]}
        required.update(value["path"] for value in config["sources"].values())
        for relative in required - {controller.CONFIG}:
            target = source / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        for value in config["sources"].values():
            value["sha256"] = sha(source / value["path"])
        target = source / controller.CONFIG; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(encode(config))
        plan_path = inputs / "plan.json"
        plan = {"kind": "primary_data_conversion_single_submission_plan", "owner": "root coordinator",
                "protocol_id": "primary_data_conversion_v1", "execution_authorized": True, "training_authorized": False,
                "scientific_acceptance": False, "max_submissions": 1, "retries": 0, "source_root": str(source),
                "source_commit": "c" * 40, "config_sha256": sha(source / controller.CONFIG),
                "source_files": {relative: sha(source / relative) for relative in required},
                "allocation": deepcopy(config["allocation"]), "resources": deepcopy(config["resources"]), "limits": deepcopy(config["limits"]),
                "output": str(run / "attempt001"), "stop_path": str(run / "STOP"),
                "controller_cache": str(inputs / "controller-cache"), "preview_receipt": str(inputs / "native-submission-preview.json"),
                "submission_receipt": str(inputs / "native-submission-receipt.json"), "logs": str(base / "logs" / run.name),
                "claim_id": "independent-fixture-001", "native_cluv_source_sha256": deepcopy(controller.LIBRARY_PINS),
                "supervisor_argv": [config["python"], "-B", str(source / controller.WORKER), "--plan", str(plan_path), "--plan-sha256", "{plan_sha256}"]}
        receipts = {}
        for name in ("owner_approval", "implementation_review", "reporting_readiness", "environment", "storage_gate"):
            value = {"status": "verified", "config_sha256": plan["config_sha256"]}
            if name == "owner_approval":
                value.update(kind="primary_data_conversion_owner_allocation_approval", approved_by="owner", approved=True,
                             **{key: deepcopy(plan[key]) for key in (*FIELDS, "allocation", "resources")})
            if name == "environment":
                value.update(python="3.10.19", wandb="0.30.0", python_executable=config["python"])
            if name == "storage_gate":
                value.update(used_bytes=0, reserved_bytes=1073741824, source_root=str(source), output=plan["output"])
            path = inputs / ("owner-allocation-approval.json" if name == "owner_approval" else name + ".json")
            receipts[name] = [path, value]
            plan[name] = {"path": str(path), "sha256": "0" * 64}
        context = {"plan": plan, "config": config, "root": source, "plan_path": plan_path, "plan_sha256": "0" * 64}
        def freeze():
            for name, (path, value) in receipts.items():
                path.write_bytes(encode(value)); plan[name]["sha256"] = sha(path)
            plan["submission_claim"] = {"path": str(inputs / "native-submission-claim.json"), "sha256": hashlib.sha256(encode(claim(plan))).hexdigest()}
            plan_path.write_bytes(encode(plan)); context["plan_sha256"] = sha(plan_path)
        freeze()
        def git(arguments, **kwargs):
            return "" if arguments[1] == "status" else "c" * 40
        with mock.patch.object(controller, "SCRATCH", base), mock.patch.object(controller, "__file__", str(source / controller.CONTROLLER)), mock.patch.object(controller.Path, "cwd", return_value=source), mock.patch.object(controller.socket, "gethostname", return_value="login-1.server.mila.quebec"), mock.patch.object(controller.subprocess, "check_output", side_effect=git) as check_git, mock.patch.dict(controller.os.environ, {}, clear=True):
            yield SimpleNamespace(context=context, plan=plan, config=config, base=base, source=source,
                                  run=run, inputs=inputs, receipts=receipts, freeze=freeze, git=check_git)


class IndependentControllerTests(unittest.TestCase):
    def verify(self, fixture):
        ctx = fixture.context
        return controller.preflight(ctx["plan_path"], ctx["plan_sha256"])

    def test_complete_fake_owner_packet_passes_preflight_without_import_or_claim(self):
        with packet() as f, mock.patch.object(controller.importlib, "import_module", side_effect=AssertionError("no native import")):
            self.assertEqual(self.verify(f)["plan"], f.plan)
            self.assertFalse(Path(f.plan["submission_claim"]["path"]).exists())
            self.assertEqual(f.git.call_count, 3)

    def test_owner_allocation_bindings_cannot_be_rebound_by_new_receipt_hash(self):
        for key, value in (("approved", False), ("source_commit", "d" * 40), ("output", "/wrong"), ("max_submissions", True), ("resources", {"cpus": 8})):
            with self.subTest(key=key), packet() as f:
                f.receipts["owner_approval"][1][key] = value; f.freeze()
                with self.assertRaises(controller.SubmissionError):
                    self.verify(f)
                self.assertFalse(Path(f.plan["submission_claim"]["path"]).exists())

    def test_dirty_unpublished_or_wrong_head_refuses_even_with_valid_source_hashes(self):
        for variant in ("dirty", "origin", "head"):
            with self.subTest(variant=variant), packet() as f:
                def git(args, **kwargs):
                    if args[1] == "status":
                        return " M changed" if variant == "dirty" else ""
                    return "d" * 40 if args[2] == ("HEAD" if variant == "head" else "refs/remotes/origin/main") and variant != "dirty" else "c" * 40
                with mock.patch.object(controller.subprocess, "check_output", side_effect=git), self.assertRaises(controller.SubmissionError):
                    self.verify(f)

    def test_bad_resources_libraries_paths_or_existing_evidence_fail_closed(self):
        for kind in ("resources", "library", "logs", "cache", "claim", "stop", "source"):
            with self.subTest(kind=kind), packet() as f:
                if kind == "resources": f.plan["resources"]["cpus"] = True
                elif kind == "library": f.plan["native_cluv_source_sha256"]["cluv/config.py"] = "0" * 64
                elif kind == "logs": f.plan["logs"] = str(f.source)
                elif kind == "cache": Path(f.plan["controller_cache"]).mkdir()
                elif kind == "claim": Path(f.plan["submission_claim"]["path"]).write_bytes(b"existing")
                elif kind == "stop": Path(f.plan["stop_path"]).write_bytes(b"STOP")
                else: (f.source / controller.SHELL).write_bytes(b"changed")
                f.freeze()
                with self.assertRaises(controller.SubmissionError): self.verify(f)

    def test_metadata_rejects_links_fifo_growth_and_duplicate_or_nonfinite_json(self):
        with packet() as f:
            path = f.inputs / "metadata.json"; path.write_bytes(b"{}")
            for data in (b'{"x":1,"x":2}', b'{"x":NaN}'):
                path.write_bytes(data)
                with self.assertRaises(controller.SubmissionError): controller.read_json(path)
            path.unlink(); path.symlink_to(f.context["plan_path"])
            with self.assertRaises(controller.SubmissionError): controller.read_bytes(path)
            path.unlink()
            import os
            os.mkfifo(path)
            with self.assertRaises(controller.SubmissionError): controller.read_bytes(path)
            path.unlink(); path.write_bytes(b"x" * 70000)
            original_read = os.read
            grown = False
            def grow(descriptor, size):
                nonlocal grown
                data = original_read(descriptor, size)
                if data and not grown:
                    with path.open("ab") as stream: stream.write(b"x")
                    grown = True
                return data
            with mock.patch.object(os, "read", side_effect=grow), self.assertRaisesRegex(controller.SubmissionError, "changed while reading"):
                controller.read_bytes(path)

    def test_single_claim_has_independent_bytes_no_hash_cycle_and_precedes_fake_api(self):
        with packet() as f:
            ctx = self.verify(f); original = mock.AsyncMock(return_value=[row(ctx)])
            native = SimpleNamespace(get_submissions=original)
            async def submit(*args, **kwargs):
                self.assertEqual(Path(f.plan["submission_claim"]["path"]).read_bytes(), encode(claim(f.plan)))
                self.assertNotIn("plan_sha256", claim(f.plan))
                self.assertEqual(sha(f.plan["submission_claim"]["path"]), f.plan["submission_claim"]["sha256"])
                self.assertTrue((f.inputs / "native-submission-intent.json").is_file())
                self.assertEqual(args, ("mila", Path(controller.SHELL), [], ["--plan", str(ctx["plan_path"]), "--plan-sha256", ctx["plan_sha256"]]))
                self.assertEqual(kwargs, {"autocommit": False, "chunking": None, "vram": None, "_skip_sync": True, "sync_datasets": False, "parsable": True})
                await native.get_submissions()
                return SimpleNamespace(cluster="mila", job_id=12345)
            native.submit = mock.AsyncMock(side_effect=submit)
            result = asyncio.run(controller.submit_once(ctx, native))
            self.assertEqual(result["state"], "native_returned_job_requires_scheduler_reconciliation")
            native.submit.assert_awaited_once(); original.assert_awaited_once()
            self.assertIs(native.get_submissions, original)
            self.assertFalse(result["retry_authorized"])

    def test_fake_api_exception_none_and_invalid_job_preserve_ambiguous_no_retry(self):
        for kind in ("exception", "none", "wrong_cluster", "invalid_id"):
            with self.subTest(kind=kind), packet() as f:
                ctx = self.verify(f); native = SimpleNamespace(get_submissions=mock.AsyncMock(return_value=[row(ctx)]))
                async def submit(*args, **kwargs):
                    await native.get_submissions()
                    if kind == "exception": raise TimeoutError("synthetic ambiguous response")
                    if kind == "none": return None
                    return SimpleNamespace(cluster="tamia" if kind == "wrong_cluster" else "mila", job_id=0 if kind == "invalid_id" else 123)
                native.submit = mock.AsyncMock(side_effect=submit)
                result = asyncio.run(controller.submit_once(ctx, native))
                self.assertEqual(result["state"], "ambiguous_requires_reconciliation")
                self.assertIsNone(result["job_id"]); native.submit.assert_awaited_once()
                self.assertTrue(Path(f.plan["submission_claim"]["path"]).exists())
                self.assertTrue(Path(f.plan["submission_receipt"]).exists())

    def test_second_call_does_not_retry_or_replace_prior_claim_and_receipt(self):
        with packet() as f:
            ctx = self.verify(f)
            original = mock.AsyncMock(return_value=[row(ctx)])
            native = SimpleNamespace(get_submissions=original)
            async def submit(*args, **kwargs):
                await native.get_submissions()
                return SimpleNamespace(cluster="mila", job_id=12345)
            native.submit = mock.AsyncMock(side_effect=submit)
            asyncio.run(controller.submit_once(ctx, native))
            prior = {key: Path(f.plan[key]["path"] if key == "submission_claim" else f.plan[key]).read_bytes() for key in ("submission_claim", "submission_receipt")}
            second = SimpleNamespace(get_submissions=mock.AsyncMock(), submit=mock.AsyncMock())
            with self.assertRaises(FileExistsError): asyncio.run(controller.submit_once(ctx, second))
            second.submit.assert_not_awaited()
            for key, data in prior.items():
                self.assertEqual(data, Path(f.plan[key]["path"] if key == "submission_claim" else f.plan[key]).read_bytes())

    def test_changed_plan_after_preflight_or_late_stop_blocks_fake_api_and_claim(self):
        for kind in ("plan_bytes", "context", "stop", "owner"):
            with self.subTest(kind=kind), packet() as f:
                ctx = self.verify(f)
                if kind == "plan_bytes": ctx["plan_path"].write_bytes(b"{}")
                elif kind == "context": ctx["plan"]["claim_id"] = "changed"
                elif kind == "stop": Path(f.plan["stop_path"]).write_bytes(b"STOP")
                else: f.receipts["owner_approval"][0].write_bytes(b"{}")
                native = SimpleNamespace(get_submissions=mock.AsyncMock(), submit=mock.AsyncMock())
                result = asyncio.run(controller.submit_once(ctx, native))
                native.submit.assert_not_awaited(); self.assertIsNone(result["job_id"])
                self.assertFalse(Path(f.plan["submission_claim"]["path"]).exists())

    def test_actual_native_shell_with_correct_metadata_cannot_hide_extra_sbatch(self):
        with packet() as f:
            good = row(f.context)
            for replacement in ("--mem=4G", "--mem=2G --mem=2G", "--mem=2G --gpus=1", "--mem=2G; evil-command", "--mem=2G --array=1-2"):
                altered = replace(good, sbatch_command=good.sbatch_command.replace("--mem=2G", replacement))
                with self.subTest(replacement=replacement), self.assertRaises(controller.SubmissionError): controller.guard_row([altered], f.context)
            for rows in ([], [good, good], [replace(good, remote=object())], [replace(good, n_chunks=1)], [replace(good, git_commit="d" * 40)]):
                with self.assertRaises(controller.SubmissionError): controller.guard_row(rows, f.context)

    def test_multiple_fake_expansions_are_not_returned_to_scheduler(self):
        with packet() as f:
            ctx = self.verify(f); original = mock.AsyncMock(return_value=[row(ctx)])
            native = SimpleNamespace(get_submissions=original)
            async def submit(*args, **kwargs):
                await native.get_submissions()
                await native.get_submissions()
                raise AssertionError("second guarded expansion must not return")
            native.submit = mock.AsyncMock(side_effect=submit)
            result = asyncio.run(controller.submit_once(ctx, native))
            self.assertEqual(result["state"], "ambiguous_requires_reconciliation")
            self.assertEqual(result["guarded_expansion_attempts"], 2)
            original.assert_awaited_once(); native.submit.assert_awaited_once()

    def test_controller_and_login_environment_guards_cover_empty_scheduler_values(self):
        for env in ({"SBATCH_ACCOUNT": ""}, {"SLURM_CLUSTERS": ""}, {"PYTHONOPTIMIZE": "1"}):
            with mock.patch.dict(controller.os.environ, env, clear=True), self.assertRaises(controller.SubmissionError): controller.reject_environment()
        self.assertIn("${!SBATCH_@}", controller.LOGIN_GUARD)
        self.assertIn("${SLURM_CLUSTERS+x}", controller.LOGIN_GUARD)
        self.assertIn("PYTHONOPTIMIZE", controller.LOGIN_GUARD)
        with packet() as f:
            guarded = controller.guard_row([row(f.context)], f.context)
            body = shlex.split(guarded.sbatch_command)[3]
            self.assertTrue(body.startswith(controller.LOGIN_GUARD))
            self.assertLess(body.index("inherited scheduler input rejected"), body.index("sbatch --parsable"))

    def test_actual_installed_native_metadata_and_source_pins_without_import(self):
        with mock.patch.object(controller.importlib, "import_module", side_effect=AssertionError("native import forbidden")):
            paths = controller.verify_native_library()
        self.assertEqual(set(paths), set(controller.LIBRARY_PINS))
        for member, path in paths.items(): self.assertEqual(sha(path), controller.LIBRARY_PINS[member])

    def test_fake_imported_library_origin_checked_before_config_api_and_cache_isolated(self):
        @dataclass(frozen=True)
        class Target:
            env: dict
            project_dir: str
            results_path: str
            job_script_path: str
            sbatch_args: object
        paths = controller.verify_native_library()
        for wrong_origin in (False, True):
            with self.subTest(wrong_origin=wrong_origin), packet() as f, mock.patch.dict(sys.modules, {}, clear=False):
                ctx = self.verify(f)
                config = SimpleNamespace(env={"UNWANTED": "value"}, sbatch_args={"gpus": 8},
                    clusters={"mila": Target({}, "old", "old", "old", [{"gpus": 8}]), "tamia": Target({}, "old", "old", "old", [])})
                get_config = mock.Mock(return_value=config)
                modules = {member[:-3].replace("/", "."): SimpleNamespace(__file__=str(path)) for member, path in paths.items()}
                modules["cluv.config"].get_cluv_config = get_config
                if wrong_origin:
                    modules["cluv.config"].__file__ = "/different/source.py"
                def fake_import(name):
                    self.assertEqual(controller.os.environ["XDG_CACHE_HOME"], f.plan["controller_cache"])
                    self.assertTrue(Path(f.plan["controller_cache"]).is_dir())
                    sys.modules.update(modules)
                    return modules[name]
                with mock.patch.object(controller, "verify_native_library", return_value=paths), mock.patch.object(controller.importlib, "import_module", side_effect=fake_import):
                    if wrong_origin:
                        with self.assertRaisesRegex(controller.SubmissionError, "actual imported"):
                            controller.prepare_native(ctx)
                        get_config.assert_not_called()
                    else:
                        self.assertIs(controller.prepare_native(ctx), modules["cluv.cli.submit"])
                        get_config.assert_called_once()
                        self.assertEqual(set(config.clusters), {"mila"})
                        self.assertEqual(config.env, {}); self.assertEqual(config.sbatch_args, {})
                        self.assertEqual(config.clusters["mila"].sbatch_args, [controller.SBATCH])


if __name__ == "__main__":
    unittest.main()
