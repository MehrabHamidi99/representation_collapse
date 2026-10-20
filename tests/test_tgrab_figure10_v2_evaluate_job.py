"""Synthetic primitive-v2 selection/codec/gating tests; no native/data execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from representation_collapse import graph_reconstruction as gr
from representation_collapse import tgrab_figure10_evaluate as numerical
from representation_collapse import tgrab_figure10_v2_evaluate_job as job
from representation_collapse import tgrab_figure10_long_fit as writer
from representation_collapse import tgrab_figure10_v2_codec as codec
from tests.test_tgrab_figure10_evaluate import Backend
from tests.test_tgrab_figure10_selection_sink import _evidence
from tests.test_tgrab_figure10_v2_codec import fixture as codec_fixture


def selection_fixture(root):
    root.mkdir()
    selection = writer.PrimitiveSelectionV2(root / "selection", run_id="long-fit", admission_hash="a" * 64)
    for epoch, score in ((1, 0.0), (2, 0.5), (3, 0.5)):
        candidate = {"epoch": epoch, "manifest_relative_path": f"candidates/epoch-{epoch:03d}/manifest.json",
                     **{k: f"{epoch:064x}" for k in ("manifest_sha256", "recurrent_sha256", "link_head_sha256", "terminal_state_sha256")}}
        selection.consider(candidate, score, selected_flag=epoch != 3)
    accounting = {"attempted_updates": 7677, "completed_updates": 7677, "optimizer_steps_confirmed": 7677, "pending": None}
    selection.finalize("complete_epoch_cap", accounting)
    acceptance = {"fit_run_id": "long-fit", "fit_attempt_id": "longfit01", "fit_admission_sha256": "a" * 64,
        "selected_pointer_sha256": selection.pointer_hash, "selected": selection.selected,
        "selected_validation_f1": selection.best_f1,
        "collected_metadata_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*.json")}}
    summary = {"selected_pointer_sha256": selection.pointer_hash, "selected": selection.selected,
        "selected_validation_f1": selection.best_f1, "finalized": True, "test_payload_opened": False,
        "stop_reason": "complete_epoch_cap", "journal_accounting": accounting}
    return acceptance, summary


class V2EvaluationTests(unittest.TestCase):
    def test_event_schema_and_identity_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(Exception):
                job.EvaluationEvents(root / "bad", {"kind": "foreign"})
            self.assertFalse((root / "bad").exists())
            events = job.EvaluationEvents(root / "events", {"fit_run_id": "original"})
            for field in ("kind", "sequence", "previous_event_sha256", "fit_run_id"):
                with self.subTest(field=field), self.assertRaises(Exception):
                    events.emit("charged", **{field: "foreign"})
            self.assertEqual(events.sequence, 0)
            self.assertEqual(list(events.root.iterdir()), [])

    def test_real_v2_finalized_selection_earliest_tie_and_authoritative_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve() / "fit"
            acceptance, summary = selection_fixture(root)
            pointer = job.verify_selection(root, acceptance=acceptance, summary=summary,
                                           pointer_sha256=acceptance["selected_pointer_sha256"])
            self.assertEqual(pointer["selected"]["epoch"], 2)
            with self.assertRaises(Exception):
                job.verify_selection(root, acceptance=acceptance, summary=summary, pointer_sha256="f" * 64)
            (root / "selection/intents/epoch-002.json").write_bytes(b"{}")
            with self.assertRaises(Exception):
                job.verify_selection(root, acceptance=acceptance, summary=summary,
                                     pointer_sha256=acceptance["selected_pointer_sha256"])

    def _flow(self, root, *, bad_replay=False, bad_load=False, observer_error=False, uncharged=False, v1=False):
        expected, provenance = codec_fixture()
        evidence, _ = _evidence(1, 0.0)
        selected = writer.write_candidate_v2(root, evidence, provenance)
        verified = codec.read_candidate_v2(root / "epoch-001", selected["manifest_sha256"], expected=expected)
        self.assertEqual(verified["manifest"]["schema_version"], 2)
        if bad_replay:
            verified["terminal_state"] = b"\x01" + bytes(102399)
        if v1:
            verified["manifest"]["kind"] = "tgrab_selected_inference_v1"
        backend = Backend()
        events = job.EvaluationEvents(root / "events", {"fit_run_id": "synthetic-v2"})
        if not uncharged:
            events.emit("charged", selected_load_attempts=1, test_payload_read_attempts=0)
        snapshots = [gr.Snapshot(t, 100, ((0, 1), (1, 0)) if (t // 32) % 2 else ((2, 3), (3, 2)), tuple(range(100))) for t in range(3072)]
        opened, observed, reports = [], [], []
        def opener():
            self.assertEqual(backend.calls, list(range(2816)))
            self.assertTrue((root / "events/003-test_read.json").exists())
            opened.append(True)
            return iter(snapshots[2816:])
        def observer(state, t):
            self.assertEqual(backend.calls[-1], t - 1)
            observed.append(t)
            state.hidden.data.fill(5); state.cell.data.fill(6)
            if observer_error:
                raise RuntimeError("observer failed")
        def loader(backend, supplied):
            self.assertIs(supplied, verified)
            self.assertEqual(supplied["manifest"]["schema_version"], 2)
            if bad_load:
                raise RuntimeError("native load failed")
        with patch.object(numerical, "load_native_weights", side_effect=loader) as load:
            kwargs = dict(verified=verified, selected=selected, pointer_sha256="a" * 64, backend=backend,
                train=snapshots[:2560], validation=snapshots[2560:2816], test_opener=opener,
                events=events, report=reports.append, observe_state=observer)
            if bad_replay or bad_load or observer_error or uncharged or v1:
                with self.assertRaises(Exception):
                    job.run_verified_candidate(**kwargs)
                self.assertEqual(len(opened), 1 if observer_error else 0)
                if uncharged or v1:
                    load.assert_not_called()
            else:
                result = job.run_verified_candidate(**kwargs)
                self.assertEqual(result["candidate_manifest_sha256"], selected["manifest_sha256"])
                self.assertNotIn("final_head_sha256", result)
                self.assertEqual(result["reduction"]["targets"], 256)
                self.assertEqual(backend.calls, list(range(3072)))
                self.assertEqual(observed, list(range(2816, 3072)))
                self.assertEqual(len(opened), 1)
                metrics_event = json.loads((root / "events/004-metrics_complete.json").read_bytes())
                self.assertEqual(metrics_event["kind"], "figure10_primitive_v2_evaluation_event_v3")
                self.assertEqual(metrics_event["event"], "metrics_complete")
                self.assertEqual(metrics_event["fit_run_id"], "synthetic-v2")
                self.assertEqual(metrics_event["result"], result)
                self.assertEqual(metrics_event["result"]["kind"], job.PROTOCOL)
                with self.assertRaises(Exception):
                    job.run_verified_candidate(**kwargs)
                self.assertEqual(len(opened), 1)

    def test_real_v2_codec_then_single_replay_test_and_cloned_observer(self):
        with tempfile.TemporaryDirectory() as td:
            self._flow(Path(td).resolve())

    def test_failed_load_replay_or_observer_never_reopens(self):
        for flag in ("bad_load", "bad_replay", "observer_error", "uncharged", "v1"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as td:
                self._flow(Path(td).resolve(), **{flag: True})

    def test_reporter_entered_before_metadata_or_native_work(self):
        from representation_collapse import reporting
        actions = []
        class Reporter:
            def __init__(self, *args, **kwargs):
                self_test.assertEqual(kwargs["mode"], "offline")
            def __enter__(self):
                actions.append("enter"); return self
            def __exit__(self, *args):
                actions.append("finish")
        self_test = self
        args = job.make_parser().parse_args([value for action in job.make_parser()._actions if action.required
            for value in (action.option_strings[0], "1000000000000" if action.dest == "deadline_unix" else
                          job.common.CONFIG_SHA256 if action.dest == "config_sha256" else
                          job.CONTRACT_SHA256 if action.dest == "long_contract_sha256" else "/unused")])
        def metadata(_):
            self.assertEqual(actions, ["enter"]); actions.append("metadata"); return ()
        def native(*a):
            self.assertEqual(actions, ["enter", "metadata"]); actions.append("native")
            return {"reduction": {}, "candidate_manifest_sha256": "a" * 64, "state_trajectory": {}, "evaluation_result_record": {}}
        with tempfile.TemporaryDirectory() as td, patch.object(job, "make_parser") as parser, patch.object(job, "pinned", return_value=({"input_pins": {"source_files_sha256": {}}}, b"{}")), patch.object(job.common, "verify_pin"), patch.object(job.common, "validate_environment_receipt"), patch.object(job.common, "validate_node2a_receipt"), patch.object(job, "verify_supervisor", return_value="b" * 64), patch.object(job.common.ProcessOwner, "acquire", return_value=SimpleNamespace(run_dir=Path(td))), patch.object(job.common, "publish_terminal_manifest") as terminal, patch.object(reporting, "ExperimentReporter", Reporter), patch.object(job, "metadata_inputs", side_effect=metadata), patch.object(job, "run_native", side_effect=native), patch.dict(os.environ, {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}):
            parser.return_value.parse_args.return_value = args
            self.assertEqual(job.main([]), 0)
            self.assertEqual(actions, ["enter", "metadata", "native", "finish"])
            self.assertEqual(terminal.call_args.args[1]["status"], "evaluation_complete")


if __name__ == "__main__":
    unittest.main()
