"""Mocked admission/submission checks only; never call native cluv or Slurm."""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
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
MODULE = ROOT / "code/ops/submit_primary_conversion.py"
spec = importlib.util.spec_from_file_location("conversion_submit_fixture", MODULE)
controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)


@dataclass
class Row:
    cluster: str
    remote: object
    n_chunks: object
    sbatch_args: dict
    git_commit: str
    job_script: Path
    program_args: list
    sbatch_command: str


def fixture(base):
    source = base / "sources/source001"; source.mkdir(parents=True)
    run = base / "runs/run001"; inputs = run / "inputs"; inputs.mkdir(parents=True)
    config = json.loads((ROOT / controller.CONFIG).read_text())
    config["sources"].update(controller={"path": controller.CONTROLLER, "sha256": None}, controller_protocol={"path": controller.PROTOCOL, "sha256": None})
    required = {controller.WORKER, controller.SHELL, controller.CONFIG, controller.REQUEST, "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]}
    required.update(record["path"] for record in config["sources"].values())
    for relative in required - {controller.CONFIG}:
        destination = source / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        original = ROOT / relative
        destination.write_bytes(original.read_bytes() if original.is_file() else b"fixture protocol\n")
    for record in config["sources"].values():
        record["sha256"] = controller.sha256(source / record["path"])
    (source / controller.CONFIG).parent.mkdir(parents=True, exist_ok=True)
    (source / controller.CONFIG).write_bytes(controller.encoded(config))
    plan_path = inputs / "plan.json"
    plan = {"kind": "primary_data_conversion_single_submission_plan", "owner": "root coordinator", "protocol_id": "primary_data_conversion_v1",
        "execution_authorized": True, "training_authorized": False, "scientific_acceptance": False, "max_submissions": 1, "retries": 0,
        "source_root": str(source), "source_commit": "c" * 40, "config_sha256": controller.sha256(source / controller.CONFIG),
        "source_files": {relative: controller.sha256(source / relative) for relative in required}, "allocation": config["allocation"],
        "resources": config["resources"], "limits": config["limits"], "output": str(run / "attempt001"), "stop_path": str(run / "STOP"),
        "controller_cache": str(inputs / "controller-cache"), "preview_receipt": str(inputs / "native-submission-preview.json"),
        "submission_receipt": str(inputs / "native-submission-receipt.json"), "logs": str(base / "logs/run001"), "claim_id": "fixture-claim001",
        "native_cluv_source_sha256": dict(controller.LIBRARY_PINS),
        "supervisor_argv": [config["python"], "-B", str(source / controller.WORKER), "--plan", str(plan_path), "--plan-sha256", "{plan_sha256}"]}
    for key in ("owner_approval", "implementation_review", "reporting_readiness", "environment", "storage_gate"):
        value = {"status": "verified", "config_sha256": plan["config_sha256"]}
        if key == "owner_approval":
            value.update(kind="primary_data_conversion_owner_allocation_approval", approved_by="owner", approved=True)
            value.update({name: plan[name] for name in (*controller.BINDINGS, "allocation", "resources")})
        if key == "environment":
            value.update(python="3.10.19", wandb="0.30.0", python_executable=config["python"])
        if key == "storage_gate":
            value.update(used_bytes=1, reserved_bytes=1073741824, source_root=str(source), output=plan["output"])
        path = inputs / ("owner-allocation-approval.json" if key == "owner_approval" else key + ".json")
        path.write_bytes(controller.encoded(value)); plan[key] = {"path": str(path), "sha256": controller.sha256(path)}
    plan["submission_claim"] = {"path": str(inputs / "native-submission-claim.json"), "sha256": controller.hashlib.sha256(controller.encoded(controller.claim_payload(plan))).hexdigest()}
    plan_path.write_bytes(controller.encoded(plan))
    return {"plan": plan, "config": config, "root": source, "plan_path": plan_path, "plan_sha256": controller.sha256(plan_path)}


def row_for(context):
    plan = context["plan"]
    resources = {key: str(value) if type(value) is int else value for key, value in controller.SBATCH.items()}
    resources.update({"job-name": "cluv-rc-primary-conversion-v1", "output": plan["logs"] + "/mila_%j/slurm-%j.out", "chdir": str(context["root"]), "export": "ALL"})
    args = ["--plan", str(context["plan_path"]), "--plan-sha256", context["plan_sha256"]]
    flags = ["--" + key if value is True else f"--{key}={value}" for key, value in resources.items()]
    inner = f"GIT_COMMIT={plan['source_commit']} CLUV_CLUSTER=mila; cd {context['root']} && sbatch --parsable " + shlex.join([*flags, controller.SHELL, *args])
    return Row("mila", None, None, resources, plan["source_commit"], Path(controller.SHELL), args, shlex.join(["bash", "--login", "-c", inner]))


class ControllerTests(unittest.TestCase):
    def test_admitted_fixture_passes_without_importing_or_calling_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve(); context = fixture(base)
            def git(args, **kwargs): return "" if args[1] == "status" else "c" * 40
            with mock.patch.object(controller, "SCRATCH", base), mock.patch.object(controller, "__file__", str(context["root"] / controller.CONTROLLER)), mock.patch.object(Path, "cwd", return_value=context["root"]), mock.patch.object(controller.socket, "gethostname", return_value="login-1.server.mila.quebec"), mock.patch.object(controller.subprocess, "check_output", side_effect=git), mock.patch.dict(controller.os.environ, {}, clear=True), mock.patch.object(controller.importlib, "import_module", side_effect=AssertionError("no native import")):
                actual = controller.preflight(context["plan_path"], context["plan_sha256"])
            self.assertEqual(actual["root"], context["root"])
            self.assertFalse(Path(context["plan"]["submission_claim"]["path"]).exists())

    def test_normal_python_and_inherited_scheduler_inputs_fail_before_main_import(self):
        for environment in ({"PYTHONOPTIMIZE": "1"}, {"SBATCH_ACCOUNT": "mila"}, {"SLURM_CLUSTERS": ""}):
            with mock.patch.dict(controller.os.environ, environment, clear=True):
                with self.assertRaises(controller.SubmissionError): controller.reject_environment()
        with mock.patch.object(controller.sys, "flags", SimpleNamespace(optimize=1)), mock.patch.dict(controller.os.environ, {}, clear=True):
            with self.assertRaises(controller.SubmissionError): controller.reject_environment()
        with mock.patch.object(controller, "preflight", side_effect=controller.SubmissionError("no approval")), mock.patch.object(controller, "prepare_native") as native:
            with self.assertRaises(controller.SubmissionError): controller.main(["--plan", "/absent", "--plan-sha256", "a" * 64])
            native.assert_not_called()

    def test_owner_mutation_is_rejected_before_any_claim_or_native_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve(); context = fixture(base); plan = context["plan"]
            path = Path(plan["owner_approval"]["path"]); value = controller.read_json(path); value["approved"] = False
            path.write_bytes(controller.encoded(value)); plan["owner_approval"]["sha256"] = controller.sha256(path)
            context["plan_path"].write_bytes(controller.encoded(plan))
            with mock.patch.object(controller, "SCRATCH", base), mock.patch.object(controller, "__file__", str(context["root"] / controller.CONTROLLER)), mock.patch.object(Path, "cwd", return_value=context["root"]), mock.patch.object(controller.socket, "gethostname", return_value="login-1.server.mila.quebec"), mock.patch.object(controller.subprocess, "check_output", side_effect=lambda args, **kwargs: "" if args[1] == "status" else "c" * 40), mock.patch.dict(controller.os.environ, {}, clear=True):
                with self.assertRaisesRegex(controller.SubmissionError, "owner-approved"):
                    controller.preflight(context["plan_path"], controller.sha256(context["plan_path"]))
            self.assertFalse(Path(plan["submission_claim"]["path"]).exists())

    def test_exact_row_and_shell_guard_reject_expansions_and_hidden_command_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = fixture(Path(temporary).resolve()); row = row_for(context)
            guarded = controller.guard_row([row], context)
            self.assertTrue(shlex.split(guarded.sbatch_command)[3].startswith(controller.LOGIN_GUARD))
            for invalid in ([], [row, row]):
                with self.assertRaises(controller.SubmissionError): controller.guard_row(invalid, context)
            mutations = [lambda r: setattr(r, "cluster", "tamia"), lambda r: setattr(r, "n_chunks", 1),
                lambda r: r.sbatch_args.update(gpus="1"), lambda r: r.sbatch_args.update(array="1-2"),
                lambda r: r.sbatch_args.update(mem="4G"), lambda r: r.sbatch_args.update(nodes=True),
                lambda r: r.program_args.append("extra"), lambda r: setattr(r, "git_commit", "d" * 40),
                lambda r: setattr(r, "sbatch_command", r.sbatch_command.replace("--mem=2G", "--mem=4G"))]
            for mutate in mutations:
                candidate = deepcopy(row); mutate(candidate)
                with self.assertRaises(controller.SubmissionError): controller.guard_row([candidate], context)

    def test_claim_bytes_are_exclusive_precomputed_and_omit_plan_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = fixture(Path(temporary).resolve()); plan = context["plan"]
            payload = controller.encoded(controller.claim_payload(plan))
            self.assertEqual(controller.hashlib.sha256(payload).hexdigest(), plan["submission_claim"]["sha256"])
            self.assertNotIn("plan_sha256", controller.claim_payload(plan))
            controller.write_new(plan["submission_claim"]["path"], payload)
            with self.assertRaises(FileExistsError): controller.write_new(plan["submission_claim"]["path"], payload)

    def test_mocked_submit_once_retains_preview_claim_and_pending_job_without_running_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = fixture(Path(temporary).resolve()); original = mock.AsyncMock(return_value=[row_for(context)])
            native = SimpleNamespace(get_submissions=original)
            async def submit(*args, **kwargs):
                self.assertTrue(Path(context["plan"]["submission_claim"]["path"]).is_file())
                self.assertTrue(kwargs["_skip_sync"]); self.assertFalse(kwargs["sync_datasets"])
                self.assertIsNone(kwargs["chunking"]); self.assertIsNone(kwargs["vram"])
                await native.get_submissions()
                return SimpleNamespace(cluster="mila", job_id=123)
            native.submit = mock.AsyncMock(side_effect=submit)
            with mock.patch.dict(controller.os.environ, {}, clear=True): outcome = asyncio.run(controller.submit_once(context, native))
            native.submit.assert_awaited_once(); original.assert_awaited_once()
            self.assertEqual(outcome["state"], "native_returned_job_requires_scheduler_reconciliation")
            self.assertEqual(outcome["job_id"], "123")
            self.assertIs(native.get_submissions, original)
            self.assertTrue(Path(context["plan"]["preview_receipt"]).is_file())

    def test_ambiguous_native_exception_never_retries_and_keeps_claim_intent_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = fixture(Path(temporary).resolve())
            native = SimpleNamespace(get_submissions=mock.AsyncMock(return_value=[row_for(context)]))
            async def ambiguous(*args, **kwargs):
                await native.get_submissions()
                raise TimeoutError("fixture ambiguous scheduler response")
            native.submit = mock.AsyncMock(side_effect=ambiguous)
            with mock.patch.dict(controller.os.environ, {}, clear=True): outcome = asyncio.run(controller.submit_once(context, native))
            native.submit.assert_awaited_once()
            self.assertEqual(outcome["state"], "ambiguous_requires_reconciliation")
            self.assertIsNone(outcome["job_id"])
            self.assertTrue(Path(context["plan"]["submission_claim"]["path"]).is_file())
            self.assertTrue((context["plan_path"].parent / "native-submission-intent.json").is_file())
            self.assertEqual(controller.read_json(context["plan"]["submission_receipt"])["error_type"], "TimeoutError")

    def test_pinned_native_source_identity_rejected_without_native_import(self):
        distribution = SimpleNamespace(version="0.1.5", locate_file=lambda member: Path("/fixture") / member)
        with mock.patch.object(controller.importlib.metadata, "distribution", return_value=distribution), mock.patch.object(controller, "sha256", return_value="0" * 64), mock.patch.object(controller.importlib, "import_module") as native:
            with self.assertRaisesRegex(controller.SubmissionError, "source identity"):
                controller.verify_native_library()
            native.assert_not_called()

    def test_late_stop_or_changed_context_blocks_claim_and_submit(self):
        for stopped in (True, False):
            with tempfile.TemporaryDirectory() as temporary:
                context = fixture(Path(temporary).resolve())
                if stopped:
                    Path(context["plan"]["stop_path"]).write_text("STOP")
                else:
                    context["plan"]["claim_id"] = "changed-after-preflight"
                native = SimpleNamespace(get_submissions=mock.AsyncMock(), submit=mock.AsyncMock())
                with mock.patch.dict(controller.os.environ, {}, clear=True): outcome = asyncio.run(controller.submit_once(context, native))
                native.submit.assert_not_awaited()
                self.assertFalse(Path(context["plan"]["submission_claim"]["path"]).exists())
                self.assertEqual(outcome["guarded_expansion_attempts"], 0)


if __name__ == "__main__": unittest.main()
