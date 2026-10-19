"""Synthetic v2 storage/selection/parity orchestration; no native fit or graphs."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from representation_collapse import tgrab_figure10_long_fit as long
from representation_collapse.tgrab_figure10_fit import EpochRecord, FitResult
from tests.test_tgrab_figure10_selection_sink import _evidence

ROOT = Path(__file__).resolve().parents[1]
ADMISSION = hashlib.sha256(b"synthetic-execution-receipt").hexdigest()


def candidate(epoch):
    return {"epoch": epoch, "manifest_relative_path": f"candidates/epoch-{epoch:03d}/manifest.json",
        **{key: hashlib.sha256(f"{epoch}-{key}".encode()).hexdigest()
           for key in ("manifest_sha256", "recurrent_sha256", "link_head_sha256", "terminal_state_sha256")}}


def provenance(epoch):
    return {"kind": "tgrab_long_fit_candidate_provenance_v2", "schema_version": 2,
        "executed_long_contract_sha256": long.CONTRACT_SHA256, "numerical_template_sha256": long.common.CONFIG_SHA256,
        "complete_epoch": epoch, "actual_supervisor_receipt_sha256": ADMISSION,
        "fixture_only": True}


class LongFitTests(unittest.TestCase):
    def test_contract_is_separate_and_v1_template_remains_unchanged(self):
        path = ROOT / "configs/graph/tgrab-long-fit-v2.json"
        contract = long.load_contract(path, long.CONTRACT_SHA256)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), long.CONTRACT_SHA256)
        old = ROOT / contract["numerical_template"]["path"]
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), long.common.CONFIG_SHA256)
        self.assertEqual(json.loads(old.read_bytes())["candidate"]["stopping"]["hard_cap_entire_pilot_elapsed_seconds"], 3600)
        self.assertEqual(contract["stopping"]["work_seconds"], 172680)
        self.assertEqual(contract["stopping"]["slurm_seconds"], 172800)
        self.assertEqual(contract["stopping"]["max_complete_epochs"], 200)
        self.assertEqual(contract["stopping"]["loss_patience"], 100)
        self.assertEqual(contract["prefix_parity"]["accepted_reference_sha256"], long.PREFIX_SHA256)
        for name, expected in contract["input_pins"]["source_files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src/representation_collapse" / name).read_bytes()).hexdigest(), expected)
        with self.assertRaises(long.LongFitError):
            long.load_contract(path, hashlib.sha256(b"wrong").hexdigest())

    def test_v2_bundle_serializes_both_components_terminal_and_validation_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            evidence, _ = _evidence(1, 0.0)
            ref = long.write_candidate_v2(root, evidence, provenance(1))
            final = root / "epoch-001"
            manifest = long.read_candidate_v2(final, ref["manifest_sha256"])
            self.assertEqual(set(p.name for p in final.iterdir()), long.FILES)
            self.assertEqual(manifest["kind"], "tgrab_complete_epoch_candidate_v2")
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["executed_long_contract_sha256"], long.CONTRACT_SHA256)
            self.assertEqual(manifest["numerical_template_sha256"], long.common.CONFIG_SHA256)
            self.assertFalse(manifest["optimizer_state_included"])
            self.assertEqual((final / "validation-terminal.f32le").stat().st_size, 102400)
            rows = json.loads((final / "validation.json").read_bytes())
            self.assertEqual(rows["kind"], "tgrab_complete_validation_rows_v2")
            self.assertEqual(len(rows["rows"]), 256)
            self.assertEqual(rows["rows"][0]["target_timestamp"], 2560)
            self.assertEqual(ref["recurrent_sha256"], hashlib.sha256((final / "recurrent.f32le").read_bytes()).hexdigest())
            with self.assertRaises(long.LongFitError):
                long.write_candidate_v2(root, evidence, provenance(1))

    def test_bundle_readback_rejects_truncated_component_or_extra_member(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            evidence, _ = _evidence(1, 0.0)
            ref = long.write_candidate_v2(root, evidence, provenance(1))
            final = root / "epoch-001"
            (final / "extra.json").write_text("{}")
            with self.assertRaises(long.LongFitError):
                long.read_candidate_v2(final, ref["manifest_sha256"])
            (final / "extra.json").unlink()
            (final / "link-head.f32le").write_bytes(b"truncated")
            with self.assertRaises(long.common.CanaryError):
                long.read_candidate_v2(final, ref["manifest_sha256"])

    def test_partial_epoch_or_false_executed_config_cannot_publish(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            evidence, _ = _evidence(1, 0.0)
            evidence.record.completed_updates = 2558
            with self.assertRaises(long.LongFitError):
                long.write_candidate_v2(root, evidence, provenance(1))
            self.assertEqual(list(root.iterdir()), [])
            evidence.record.completed_updates = 2559
            bad = provenance(1); bad["executed_long_contract_sha256"] = long.common.CONFIG_SHA256
            with self.assertRaises(long.LongFitError):
                long.write_candidate_v2(root, evidence, bad)
            self.assertEqual(list(root.iterdir()), [])

    def test_selection_zero_first_score_strict_tie_then_improvement(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = long.PrimitiveSelectionV2(Path(td).resolve() / "selection", run_id="synthetic-long", admission_hash=ADMISSION)
            ledger.consider(candidate(1), 0.0, selected_flag=True)
            self.assertEqual(ledger.selected["epoch"], 1)
            ledger.consider(candidate(2), 0.0, selected_flag=False)
            self.assertEqual(ledger.selected["epoch"], 1)
            ledger.consider(candidate(3), 0.5, selected_flag=True)
            self.assertEqual(ledger.selected, candidate(3))
            audit = ledger.finalize("deadline", {"attempted_updates": 3*2559+25, "completed_updates": 3*2559+25,
                "optimizer_steps_confirmed": 3*2559+25, "pending": None})
            self.assertEqual(audit["complete_epochs"], 3)
            self.assertEqual(audit["pointer"]["selected"], candidate(3))
            self.assertTrue(audit["pointer"]["finalized"])
            self.assertEqual(len(list((ledger.root / "receipts").iterdir())), 4)
            receipt = json.loads((ledger.root / "receipts/final-003.json").read_bytes())
            self.assertEqual(receipt["accounting"]["attempted_updates"], 3*2559+25)
            self.assertFalse(receipt["test_authorized"])
            with self.assertRaises(long.LongFitError):
                ledger.consider(candidate(4), .7, selected_flag=True)

    def test_selection_mismatched_native_flag_or_epoch_is_terminal(self):
        for reference, score, selected in ((candidate(1), .5, False), (candidate(2), .5, True)):
            with self.subTest(reference=reference), tempfile.TemporaryDirectory() as td:
                ledger = long.PrimitiveSelectionV2(Path(td).resolve() / "selection", run_id="synthetic", admission_hash=ADMISSION)
                with self.assertRaises(long.LongFitError):
                    ledger.consider(reference, score, selected_flag=selected)
                self.assertTrue(ledger.poisoned)
                with self.assertRaises(long.LongFitError):
                    ledger.consider(candidate(1), .5, selected_flag=True)

    def test_selection_detects_mutated_durable_history_and_pointer(self):
        for target in ("receipts/epoch-001.json", "selected.json"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as td:
                ledger = long.PrimitiveSelectionV2(Path(td).resolve() / "selection", run_id="synthetic", admission_hash=ADMISSION)
                ledger.consider(candidate(1), .5, selected_flag=True)
                path = ledger.root / target
                value = json.loads(path.read_bytes()); value["best_f1"] = .9
                path.write_bytes(long.canonical(value))
                with self.assertRaises(long.common.CanaryError):
                    ledger.audit()

    def test_candidate_publication_failure_never_advances_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            candidates = root / "candidates"; candidates.mkdir()
            selection = long.PrimitiveSelectionV2(root / "selection", run_id="synthetic", admission_hash=ADMISSION)
            evidence, _ = _evidence(1, 0.0)
            with patch.object(long.os, "rename", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    long.write_candidate_v2(candidates, evidence, provenance(1))
            self.assertIsNone(selection.selected)
            self.assertIsNone(selection.audit())
            self.assertFalse((candidates / "epoch-001").exists())

    def test_exact_prefix_gate_receipt_only_after_all256_matches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            rows = [{"loss": i / 1024} for i in range(256)]
            gate = long.PrefixParity(rows, root, long.PREFIX_SHA256)
            for index in range(256):
                self.assertEqual((root / "prefix-parity.json").exists(), False)
                gate.on_update({"attempt": index+1, "target_timestamp": index+1, "completed": True, "loss": rows[index]["loss"]})
            self.assertEqual(gate.compared, 256)
            receipt = json.loads((root / "prefix-parity.json").read_bytes())
            self.assertEqual(receipt["exact_loss_matches"], 256)
            self.assertEqual(receipt["status"], "pass")
            gate.on_update({"attempt": 257, "loss": 999})
            self.assertEqual(gate.compared, 256)

    def test_prefix_gate_retains_first_disagreement_and_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            gate = long.PrefixParity([{"loss": 1.0}] * 256, root, long.PREFIX_SHA256)
            gate.on_update({"attempt": 1, "target_timestamp": 1, "completed": True, "loss": 1.0})
            with self.assertRaises(long.LongFitError):
                gate.on_update({"attempt": 2, "target_timestamp": 2, "completed": True, "loss": 1.01})
            self.assertFalse((root / "prefix-parity.json").exists())
            failure = json.loads((root / "prefix-parity-failure.json").read_bytes())
            self.assertEqual(failure["prior_exact_matches"], 1)
            self.assertEqual(failure["observed_loss"], 1.01)
            self.assertTrue(gate.poisoned)
            with self.assertRaisesRegex(long.LongFitError, "terminal"):
                gate.on_update({"attempt": 2, "target_timestamp": 2, "completed": True, "loss": 1.0})
            self.assertEqual(gate.compared, 1)
            self.assertFalse((root / "prefix-parity.json").exists())

    def test_actual_native_tuple_normal_stop_variants_and_return_integrity(self):
        for native_reason, expected in (("training_loss_patience", "loss_patience"), ("complete_epoch_cap", "complete_epoch_cap")):
            result = FitResult(stop_reason=native_reason, epochs_completed=2, attempted_updates=5118,
                completed_updates=5118, optimizer_steps_confirmed=5118, best_loss_epoch=2,
                best_training_loss=.3, selected_validation_epoch=1, selected_validation_mean_f1=.5, history=())
            reason, summary = long.summarize_native_result((result, ({},) * 512), 2)
            self.assertEqual(reason, expected)
            self.assertEqual(summary["completed_updates"], 5118)
            with self.assertRaises(long.LongFitError):
                long.summarize_native_result((result, ({},) * 511), 2)
            with self.assertRaises(long.LongFitError):
                long.summarize_native_result((result, ({},) * 512), 1)

    def test_supervisor_48hour_contract_and_bound_inputs(self):
        import os, sys
        args = SimpleNamespace(run_id="synthetic-long", source_commit="a"*40, runtime_seconds=long.WORK_SECONDS,
            deadline_unix=172681.)
        for key in ("long_contract", "prefix_updates", "environment_receipt", "node2a_receipt", "conversion_receipt"):
            setattr(args, key, Path("/synthetic/"+key)); setattr(args, key+"_sha256", hashlib.sha256(key.encode()).hexdigest())
        argv = ["--synthetic-test-only"]
        child = [sys.executable, "-B", "-c", "synthetic bootstrap", *argv]
        receipt = {"schema":"tgrab-figure10-supervisor-receipt-v2", "protocol_id":long.PROTOCOL_ID,
            "run_id":args.run_id,"source_commit":args.source_commit,"job_id":"123", "source_sha256":long.common.sha256_file(Path(long.__file__)),
            "canary_args":argv,"wall_started_epoch_ns":1000000000,"work_seconds":long.WORK_SECONDS,"outer_wall_seconds":long.OUTER_SECONDS,
            "deadline_unix":args.deadline_unix,"sys_executable":sys.executable,"child_argv":child,
            "argv_sha256":long.digest(json.dumps(child,separators=(",",":")).encode()),
            "job":{"job_id":"123","account":"mila","partition":"main-cpu","qos":"normal","nodes":1,"tasks":1,
                "cpus_on_node":"1","cpus_per_task":"1","gpu_allocation":"0","memory_max_bytes":2147483648,"time_limit":"2-00:00:00"},
            **{key:{"path":str(getattr(args,key)),"sha256":getattr(args,key+"_sha256")}
               for key in ("long_contract","prefix_updates","environment_receipt","node2a_receipt","conversion_receipt")}}
        with patch.dict(os.environ, {"SLURM_JOB_ID":"123","SLURM_JOB_ACCOUNT":"mila","SLURM_JOB_PARTITION":"main-cpu","SLURM_CPUS_ON_NODE":"1"}):
            long.validate_supervisor(receipt,args,argv)
            for field, wrong in (("time_limit","01:00:00"),("gpu_allocation","1"),("memory_max_bytes",4294967296)):
                altered=copy.deepcopy(receipt);altered["job"][field]=wrong
                with self.subTest(field=field),self.assertRaises(long.LongFitError):long.validate_supervisor(altered,args,argv)
            altered=copy.deepcopy(receipt);altered["prefix_updates"]["sha256"]=hashlib.sha256(b"wrong").hexdigest()
            with self.assertRaises(long.LongFitError):long.validate_supervisor(altered,args,argv)

    def test_no_heldout_opener_or_optimizer_resume_or_v1_candidate_encoder(self):
        source = Path(long.__file__).read_text()
        self.assertNotIn("sealed_test_opener", source)
        self.assertNotIn("encode_epoch_candidate(", source)
        self.assertNotIn("torch.load", source)
        self.assertIn("journal.seal_epoch(epoch)", source)
        self.assertIn("reporter.log_manifest", source)


if __name__ == "__main__":
    unittest.main()
