from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from representation_collapse import tgrab_figure10_full_fit as full
from representation_collapse import tgrab_selected_checkpoint as codec
from tests.test_tgrab_figure10_selection_sink import _new_sink, _evidence

ROOT = Path(__file__).resolve().parents[1]


class Reporter:
    def __init__(self, path, fail=False):
        self.path, self.fail = path, fail
        self.calls = []

    def log(self, metrics, *, step):
        # The row must already exist before any reporter side effect.
        rows = self.path.read_text().splitlines()
        assert json.loads(rows[-1])["event"] == step
        self.calls.append((step, metrics))
        if self.fail:
            raise RuntimeError("offline reporting failed")


class FullFitTests(unittest.TestCase):
    def test_native_adapter_tuple_at_both_normal_stopping_conditions(self):
        from representation_collapse.tgrab_figure10_fit import FitResult
        for native_reason, expected in (("training_loss_patience", "loss_patience"),
                                        ("complete_epoch_cap", "complete_epoch_cap")):
            result = FitResult(native_reason, 2, 5118, 5118, 5118, 1, 0.5, 2, 0.7)
            reason, summary = full.summarize_completed_fit((result, ({"epoch": 2},)))
            self.assertEqual(reason, expected)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual(summary["selected_validation_mean_f1"], 0.7)

    def test_event_sequence_is_independent_of_optimizer_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reporter = Reporter(root / "fit-events.jsonl")
            events = full.DurableEvents(root, reporter)
            try:
                events.emit("optimizer_update", {"attempt": 2559}, {"train/loss": 0.3})
                events.emit("complete_epoch", {"epoch": 1}, {"validation/f1": 0.2})
                events.emit("optimizer_update", {"attempt": 2560}, {"train/loss": 0.2})
            finally:
                events.close()
            self.assertEqual([step for step, _ in reporter.calls], [0, 1, 2])
            self.assertEqual([json.loads(row)["record"] for row in reporter.path.read_text().splitlines()],
                             [{"attempt": 2559}, {"epoch": 1}, {"attempt": 2560}])

    def test_reporter_failure_retains_raw_row_and_does_not_advance_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reporter = Reporter(root / "fit-events.jsonl", fail=True)
            events = full.DurableEvents(root, reporter)
            try:
                with self.assertRaisesRegex(RuntimeError, "reporting failed"):
                    events.emit("optimizer_update", {"attempt": 1}, {"train/loss": 1.0})
                self.assertEqual(events.sequence, 0)
                self.assertEqual(json.loads(reporter.path.read_text())["record"], {"attempt": 1})
            finally:
                events.close()

    def test_observation_reference_hashes_actual_retained_bytes_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            observation = {"state": [1, 2, 3], "seed": 0}
            ref = full.write_record(root, "rng.json", observation, "seeding_before_models")
            self.assertEqual(ref["sha256"], full.digest((root / "rng.json").read_bytes()))
            self.assertEqual(ref["bytes"], (root / "rng.json").stat().st_size)
            with self.assertRaises(FileExistsError):
                full.write_record(root, "rng.json", {"state": [4]})
            self.assertEqual(json.loads((root / "rng.json").read_bytes()), observation)

    def test_real_ledger_finalization_preserves_selected_complete_epoch(self):
        for reason in ("deadline", "owner_stop", "loss_patience", "complete_epoch_cap"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as td:
                sink, ledger, selection = _new_sink(Path(td) / "ledger")
                try:
                    self.assertIsNone(full.finalize_selection(ledger, selection, reason))
                    self.assertIsNone(ledger.verify_committed_state())
                    evidence, _ = _evidence(1, 1.0)
                    evidence.record.validation_selected = True
                    sink.on_epoch_complete(evidence)
                    sink.on_selection(evidence.record)
                    selected = selection.selected
                    full.finalize_selection(ledger, selection, reason)
                    head = ledger.verify_committed_state()
                    self.assertTrue(head["finalized"])
                    self.assertTrue(selection.finalized)
                    self.assertEqual(selection.selected, selected)
                    self.assertEqual(selection.last_epoch, 1)
                    pointer, _ = ledger._read_reference(head["selected_pointer"], "immutable_selected_pointer")
                    self.assertEqual(pointer["selected"], asdict(selected))
                    receipt, _ = ledger._read_reference(head["decision_receipt"], "immutable_selection_receipt")
                    self.assertEqual(receipt["run_stop_reason"], reason)
                    self.assertFalse(receipt["test_authorized_by_this_receipt"])
                finally:
                    ledger.close()

    def test_manifest_identity_assembly_reads_metadata_without_test_opener(self):
        schema, _ = codec._schema_and_catalog()
        contract = schema["$defs"]["manifest"]["properties"]["identities"]["properties"]
        hashes = {name: full.digest(name.encode()) for name in ("train", "validation", "test")}
        md = {"features": "float32_identity", "weights": "float32_unit_aligned_with_pairs",
              "node_order": "ascending_integer_ids", "raw": {"sha256": contract["raw_sha256"]["const"]},
              "converter": {"sha256": contract["converter_sha256"]["const"]},
              "splits": {name: {"sha256": value, "semantic_sha256": full.digest((name + "semantic").encode())}
                         for name, value in hashes.items()}}
        actual = lambda name: full.digest((ROOT / name).read_bytes())
        ids = full.assemble_identities(metadata=md, config_hash=full.common.CONFIG_SHA256,
            manifest_hash=full.digest(full.canonical(md)), conversion_hash=full.digest(b"conversion receipt"),
            reporting_hash=full.digest(b"report identity"), admission_hash=full.digest(b"admission"),
            codec_hash=actual("src/representation_collapse/tgrab_selected_checkpoint.py"),
            trainer_hash=actual("src/representation_collapse/tgrab_figure10_native_fit.py"),
            environment_hash=full.digest(b"environment receipt"), schema=schema,
            catalog_hash=codec.CATALOG_SHA256, schema_hash=codec.SCHEMA_SHA256,
            conversion_protocol_hash=actual("doc/auto/tgrab-primary-data-conversion-v1.md"))
        codec._schema_validate(ids, schema["$defs"]["manifest"]["properties"]["identities"], schema)
        self.assertEqual(ids["test_file_sha256"], hashes["test"])
        self.assertEqual(ids["trainer_source_sha256"], actual("src/representation_collapse/tgrab_figure10_native_fit.py"))
        md["features"] = "other"
        with self.assertRaisesRegex(full.FullFitError, "encoding"):
            full.assemble_identities(metadata=md, config_hash="", manifest_hash="", conversion_hash="",
                reporting_hash="", admission_hash="", codec_hash="", trainer_hash="", environment_hash="",
                schema=schema, catalog_hash="", schema_hash="", conversion_protocol_hash="")

    def test_conversion_gate_rejects_unverified_or_different_dataset(self):
        md = {"raw": {"sha256": full.digest(b"raw")}, "converter": {"sha256": full.common.GRAPH_DATA_SHA256}}
        receipt = {"kind": "tgrab_primary_conversion_independent_verification_v1", "status": "verified",
            "primary_eligible": True, "independent_of_producer": True, "manifest_sha256": full.digest(b"manifest"),
            "raw_sha256": md["raw"]["sha256"], "converter_sha256": md["converter"]["sha256"],
            "config_sha256": full.common.CONFIG_SHA256, "snapshot_count": 3072}
        full.check_conversion(receipt, md, receipt["manifest_sha256"])
        for field, wrong in (("independent_of_producer", False), ("manifest_sha256", full.digest(b"wrong")),
                             ("status", "pending"), ("snapshot_count", 0)):
            with self.subTest(field=field), self.assertRaises(full.FullFitError):
                full.check_conversion({**receipt, field: wrong}, md, receipt["manifest_sha256"])

    def test_failed_durability_barrier_keeps_visible_final_outcome_ambiguous(self):
        from representation_collapse.tgrab_figure10_attempt_journal import AttemptJournalError
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            intent = full.canonical({"sequence": 1})
            (root / "00000001.intent.json").write_bytes(intent)
            (root / "00000001.outcome.json").write_bytes(full.canonical({"sequence": 1,
                "intent_sha256": full.digest(intent), "optimizer_step_status": "completed", "completed_return": True}))
            ordinary = full.reconcile_failure(root, RuntimeError("other failure"))
            self.assertEqual(ordinary["optimizer_steps_confirmed"], 1)
            wrapped = RuntimeError("callback failed")
            wrapped.__cause__ = AttemptJournalError("fsync failed")
            failed = full.reconcile_failure(root, wrapped)
            self.assertEqual(failed["attempted_updates"], 1)
            self.assertEqual(failed["optimizer_steps_confirmed"], 0)
            self.assertEqual(failed["optimizer_mutation_unknown_attempts"], 1)
            self.assertIsNone(failed["committed_updates"])

    def test_no_heldout_opener_or_payload_loader_in_worker(self):
        # A direct source audit guards the cooperative isolation promise: only
        # open_train_validation is allowed, despite storing test metadata pins.
        source = Path(full.__file__).read_text()
        self.assertNotIn("sealed_test_opener", source)
        self.assertNotIn("_load_split", source)
        self.assertNotIn("schema_sample", source)
        self.assertIn("open_train_validation", source)


if __name__ == "__main__":
    unittest.main()
