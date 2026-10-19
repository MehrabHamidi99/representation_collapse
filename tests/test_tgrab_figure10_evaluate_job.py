"""Admission/reference tests; no native execution and no project data reads."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from representation_collapse import tgrab_figure10_evaluate_job as job


class EvaluationJobTests(unittest.TestCase):
    def test_supervisor_receipt_binds_actual_run_arguments(self):
        args = SimpleNamespace(run_id="evaluation01", deadline_unix=123.0)
        receipt = {"schema":"tgrab-figure10-evaluation-supervisor-receipt-v2",
                   "protocol_id":job.PROTOCOL,"run_id":args.run_id,"job_id":"123",
                   "source_sha256":"a"*64,"canary_args":["--run-id","evaluation01"],
                   "sys_executable":job.sys.executable,"deadline_unix":123.0}
        env = {"RC_SUPERVISOR_RECEIPT":"/receipt", "RC_SUPERVISOR_RECEIPT_SHA256":"b"*64,"SLURM_JOB_ID":"123"}
        with patch.dict(job.os.environ,env), patch.object(job,"read_json",return_value=receipt), patch.object(job.common,"sha256_file",return_value="a"*64):
            self.assertEqual(job.verify_supervisor(args,receipt["canary_args"]),"b"*64)
            with self.assertRaisesRegex(job.EvaluationJobError,"mismatch"):
                job.verify_supervisor(args,["--run-id","another"])

    def test_unadmitted_evaluation_rejected(self):
        with patch.dict(job.os.environ,{},clear=True):
            with self.assertRaisesRegex(job.EvaluationJobError,"supervisor required"):
                job.verify_supervisor(SimpleNamespace(),[])

    def test_fit_bound_claim_forbids_another_output_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            job.charge_fit_evaluation(root, {"final_head_sha256": "a" * 64, "evaluation_run_id": "first"})
            with self.assertRaises(FileExistsError):
                job.charge_fit_evaluation(root, {"final_head_sha256": "a" * 64, "evaluation_run_id": "second"})
            self.assertEqual(json.loads((root / "selected-evaluation-claim.json").read_text())["evaluation_run_id"], "first")
    def test_external_reference_exact_role_bytes_hash_and_safe_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            raw = b'{"seed":1}'
            (root / "negative.json").write_bytes(raw)
            reference = {"relative_path": "negative.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                         "role": "negative_generator_initialization", "encoding": "primitive_json"}
            self.assertEqual(job.verify_external_reference(root, reference, reference["role"]), {"seed": 1})
            for field, value in (("relative_path", "../negative.json"), ("bytes", 0), ("sha256", "a" * 64), ("role", "other")):
                bad = {**reference, field: value}
                with self.assertRaises(Exception):
                    job.verify_external_reference(root, bad, reference["role"])

    def test_external_reference_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            raw = b"{}"
            (root / "real").write_bytes(raw)
            (root / "alias").symlink_to(root / "real")
            reference = {"relative_path": "alias", "bytes": 2, "sha256": hashlib.sha256(raw).hexdigest(),
                         "role": "seeding_before_models", "encoding": "primitive_json"}
            with self.assertRaises(Exception):
                job.verify_external_reference(root, reference, reference["role"])

    def test_frozen_config_mismatch_stops_before_reporter_native_or_owner(self):
        names = [action for action in job.make_parser()._actions if action.required]
        argv = []
        for action in names:
            argv.extend([action.option_strings[0], "123" if action.dest == "deadline_unix" else "/unused"])
        with patch.object(job, "run_native") as native, patch.object(job.common.ProcessOwner, "acquire") as owner:
            with self.assertRaisesRegex(job.EvaluationJobError, "frozen"):
                job.main(argv)
            native.assert_not_called()
            owner.assert_not_called()

    def test_external_reporting_identity_mismatch_rejects(self):
        initialization = {key: {} for key in ("seeding_record", "post_model_rng_record", "negative_generator_record")}
        verified = {"manifest": {"identities": {"pin": 1}}, "provenance": {
            "initialization": initialization, "reporting_identity_sha256": "a" * 64,
            "runtime_policy": {"module_mode_receipt_sha256": "b" * 64}, "run_id": "fit"}}
        with patch.object(job, "verify_external_reference", return_value={"seed": 1}), patch.object(job, "read_json", side_effect=[
                {"run_id": "wrong", "mode": "offline", "project": "representation-collapse"}, {}]):
            with self.assertRaisesRegex(job.EvaluationJobError, "content mismatch"):
                job.verify_selected_external_records(Path("unused"), verified, expected_identities={"pin": 1})


if __name__ == "__main__":
    unittest.main()
