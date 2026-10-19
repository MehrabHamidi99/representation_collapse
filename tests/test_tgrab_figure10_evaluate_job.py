"""Admission/reference tests; no native execution and no project data reads."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from representation_collapse import tgrab_figure10_evaluate_job as job


class EvaluationJobTests(unittest.TestCase):
    def test_result_artifact_binds_actual_retained_json_without_payload_upload(self):
        calls = []
        class Reporter:
            def log_manifest(self, name, *, metadata, identities):
                self_test.assertTrue((root / "evaluation-result.json").exists())
                calls.append((name, metadata, identities))
        self_test = self
        result = {"checkpoint_sha256": "a" * 64, "state_trajectory": {"sha256": "b" * 64}, "rows": []}
        args = SimpleNamespace(final_head_sha256="c" * 64, config_sha256="d" * 64, data_manifest_sha256="e" * 64)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            reference = job.publish_evaluation_result(root, result, args, Reporter())
            actual = hashlib.sha256((root / "evaluation-result.json").read_bytes()).hexdigest()
            self.assertEqual(reference["sha256"], actual)
            self.assertEqual(calls[0][2]["evaluation_result_sha256"], actual)
            self.assertEqual(calls[0][2]["checkpoint_sha256"], "a" * 64)
            self.assertEqual(calls[0][2]["trajectory_sha256"], "b" * 64)
            self.assertEqual(calls[0][2]["final_head_sha256"], "c" * 64)
            self.assertFalse(calls[0][1]["payload_upload"])
            self.assertFalse(calls[0][1]["model_upload"])
            self.assertEqual(calls[0][1]["trajectory_bytes"], 26214400)

    def test_trajectory_full_hc_bytes_order_signedzero_and_no_mutation(self):
        from tests.test_tgrab_figure10_evaluate import Tensor
        from representation_collapse.graph_reconstruction import CompleteState
        hidden = Tensor(np.zeros((1, 100, 128), dtype=np.float32))
        hidden.data.flat[0] = -0.0
        cell = Tensor(np.full((1, 100, 128), 2, dtype=np.float32))
        expected = hidden.data.tobytes() + cell.data.tobytes()
        with tempfile.TemporaryDirectory() as td:
            writer = job.TrajectoryWriter(Path(td))
            for t in range(2816, 3072):
                writer(CompleteState(hidden, cell, t, tuple(range(100))), t)
            descriptor = writer.finish()
            payload = writer.final.read_bytes()
            self.assertEqual(len(payload), 26214400)
            self.assertEqual(payload, expected * 256)
            self.assertEqual(payload[:4], bytes.fromhex("00000080"))
            self.assertEqual(descriptor["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(descriptor["shape"], [256, 2, 1, 100, 128])
            self.assertEqual(hidden.data.tobytes() + cell.data.tobytes(), expected)
            self.assertFalse(writer.partial.exists())

    def test_trajectory_order_failure_preserves_partial_and_is_terminal(self):
        from tests.test_tgrab_figure10_evaluate import Backend
        from representation_collapse.graph_reconstruction import CompleteState
        zero = Backend().reset()
        with tempfile.TemporaryDirectory() as td:
            writer = job.TrajectoryWriter(Path(td))
            state = CompleteState(zero.hidden, zero.cell, 2816, zero.node_ids)
            writer(state, 2816)
            with self.assertRaises(job.EvaluationJobError):
                writer(state, 2816)
            with self.assertRaises(job.EvaluationJobError):
                writer.finish()
            writer.close()
            self.assertEqual(writer.partial.stat().st_size, 102400)
            self.assertFalse(writer.final.exists())

    def test_trajectory_rejects_wrong_cell_shape_or_node_order(self):
        from tests.test_tgrab_figure10_evaluate import Backend, Tensor
        from representation_collapse.graph_reconstruction import CompleteState
        zero = Backend().reset()
        for cell, nodes in ((Tensor(np.zeros((100, 128), dtype=np.float32)), zero.node_ids),
                            (zero.cell, tuple(reversed(zero.node_ids)))):
            with tempfile.TemporaryDirectory() as td:
                writer = job.TrajectoryWriter(Path(td))
                try:
                    with self.assertRaises(Exception):
                        writer(CompleteState(zero.hidden, cell, 2816, nodes), 2816)
                    self.assertTrue(writer.poisoned)
                finally:
                    writer.close()

    def test_trajectory_replaced_or_corrupted_partial_cannot_publish(self):
        from tests.test_tgrab_figure10_evaluate import Backend
        from representation_collapse.graph_reconstruction import CompleteState
        zero = Backend().reset()
        for replaced in (False, True):
            with self.subTest(replaced=replaced), tempfile.TemporaryDirectory() as td:
                writer = job.TrajectoryWriter(Path(td))
                for t in range(2816, 3072):
                    writer(CompleteState(zero.hidden, zero.cell, t, zero.node_ids), t)
                writer.stream.flush()
                if replaced:
                    writer.partial.unlink()
                    writer.partial.write_bytes(bytes(26214400))
                else:
                    with writer.partial.open("r+b") as handle:
                        handle.write(b"\x01")
                with self.assertRaises(job.EvaluationJobError):
                    writer.finish()
                self.assertTrue(writer.poisoned)
                self.assertFalse(writer.final.exists())
                self.assertTrue(writer.partial.exists())
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
