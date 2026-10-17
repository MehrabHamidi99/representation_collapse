from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse import graph_reconstruction as gr  # noqa: E402
from representation_collapse import tgrab_selected_checkpoint as codec  # noqa: E402
from representation_collapse import tgrab_selection_transaction as tx  # noqa: E402
from representation_collapse.tgrab_figure10_candidate_encoder import encode_epoch_candidate  # noqa: E402
from representation_collapse.tgrab_figure10_selection_sink import (  # noqa: E402
    Figure10SelectionSink,
    SelectionSinkError,
)
from tests.test_tgrab_figure10_candidate_encoder import _inputs  # noqa: E402


RUN_ID = "sink-run"
ATTEMPT_ID = "sink-attempt"
ADMISSION = "a" * 64


def _evidence(epoch: int, score: float):
    evidence, _, manifest = _inputs()
    rows = evidence.validation_rows
    if score == 1.0:
        counts = (2, 0, 0, 9898)
    elif score == 0.0:
        counts = (0, 0, 2, 9898)
    else:
        raise AssertionError("fixture supports exact zero or one F1 only")
    tp, fp, fn, tn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / 9900
    adjusted = []
    for row in rows:
        row = dict(row)
        row.update(epoch=epoch, tp=tp, fp=fp, fn=fn, tn=tn,
                   precision=precision, recall=recall, f1=f1, accuracy=accuracy)
        adjusted.append(row)
    evidence.validation_rows = tuple(adjusted)
    evidence.record = SimpleNamespace(
        epoch=epoch, attempted_updates=2559, completed_updates=2559,
        optimizer_steps_confirmed=2559, validation_mean_f1=score,
        validation_selected=False)
    manifest["identities"]["admission_sha256"] = ADMISSION
    return evidence, manifest["identities"]


def _context(expected_head=None):
    return tx.SelectionContext(RUN_ID, ATTEMPT_ID, ADMISSION,
                               codec.SCHEMA_SHA256, expected_head)


def _new_sink(root: Path, *, failure_hook=None, provenance_mismatch=False,
              provenance_drift=False):
    root = root.resolve()
    root.mkdir()
    ledger = tx.SelectionLedger(root, _context(), failure_hook=failure_hook)
    selection = gr.ValidationOnlySelection()
    _, identity_map = _evidence(1, 0.0)
    identity_map["schema_sha256"] = codec.SCHEMA_SHA256

    def provenance_factory(evidence):
        provenance = copy.deepcopy(_inputs()[1])
        epoch = evidence.record.epoch
        provenance.update(run_id=RUN_ID, attempt_id=ATTEMPT_ID,
                          admission_sha256=("b" * 64 if provenance_mismatch else ADMISSION))
        provenance["source_sha256"]["schema"] = codec.SCHEMA_SHA256
        if provenance_drift and epoch > 1:
            provenance["source_sha256"]["trainer"] = "9" * 64
        provenance["counters"].update(
            complete_epochs=epoch, validation_epochs=epoch,
            attempted_updates=epoch * 2559, completed_updates=epoch * 2559,
            selected_load_attempts=0, test_payload_read_attempts=0)
        return provenance

    sink = Figure10SelectionSink(
        ledger=ledger, selection=selection, identity_map=identity_map,
        provenance_factory=provenance_factory,
        transaction_id_factory=lambda epoch: f"sink-tx-{epoch:03d}",
        timestamp_factory=lambda epoch: f"2026-10-17T12:00:{epoch:02d}+00:00")
    return sink, ledger, selection


class Figure10SelectionSinkTests(unittest.TestCase):
    def test_encoder_writer_reader_and_validation_selection_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            sink, ledger, selection = _new_sink(root)
            try:
                for epoch, score, expected in (
                        (1, 0.0, True),   # first zero-F1 candidate is selectable
                        (2, 1.0, True),   # strict improvement replaces it
                        (3, 1.0, False)): # exact tie retains earliest winner
                    evidence, _ = _evidence(epoch, score)
                    evidence.record.validation_selected = expected
                    sink.on_epoch_complete(evidence)
                    receipt = ledger.verify_committed_state()
                    self.assertEqual(receipt["sequence"], epoch)
                    decision, _ = ledger._read_reference(
                        receipt["decision_receipt"], "immutable_selection_receipt")
                    self.assertEqual(decision["decision"], "selected" if expected else "retained")
                    self.assertEqual(selection.last_epoch, epoch)
                    self.assertEqual(selection.selected.epoch, 1 if epoch == 1 else 2)
                    self.assertEqual(selection.best_f1, 0.0 if epoch == 1 else 1.0)
                    if expected:
                        sink.on_selection(evidence.record)
                self.assertEqual(len(list((root / "candidates").iterdir())), 3)
                for candidate in (root / "candidates").iterdir():
                    parsed = codec.read_candidate(candidate.resolve())
                    self.assertEqual(len(parsed["validation"]["rows"]), 256)
            finally:
                ledger.close()

    def test_malformed_provenance_is_terminal_before_publication(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run", provenance_mismatch=True)
            try:
                evidence, _ = _evidence(1, 0.0)
                evidence.record.validation_selected = True
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
                self.assertTrue(sink.poisoned)
                self.assertEqual(list((Path(td) / "run" / "candidates").iterdir()), [])
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
            finally:
                ledger.close()

    def test_stable_provenance_and_identity_pins_cannot_drift_across_epochs(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run", provenance_drift=True)
            try:
                self.assertNotIsInstance(sink.identity_map, dict)
                with self.assertRaises(TypeError):
                    sink.identity_map["admission_sha256"] = "0" * 64
                evidence, _ = _evidence(1, 0.0)
                evidence.record.validation_selected = True
                sink.on_epoch_complete(evidence)
                sink.on_selection(evidence.record)

                evidence, _ = _evidence(2, 1.0)
                evidence.record.validation_selected = True
                with self.assertRaisesRegex(SelectionSinkError, "not ledger-bound"):
                    sink.on_epoch_complete(evidence)
                self.assertTrue(sink.poisoned)
                self.assertEqual(ledger.verify_committed_state()["sequence"], 1)
                self.assertEqual(len(list((Path(td) / "run" / "candidates").iterdir())), 1)
            finally:
                ledger.close()

    def test_failed_publication_is_terminal_and_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run")
            evidence, _ = _evidence(1, 0.0)
            evidence.record.validation_selected = True
            original = codec.write_candidate
            try:
                codec.write_candidate = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected publication failure"))
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
                self.assertTrue(sink.poisoned)
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
            finally:
                codec.write_candidate = original
                ledger.close()

    def test_failed_ledger_transaction_after_gr_apply_is_terminal(self):
        def fail_after_apply(stage):
            if stage == "selection_apply:after":
                raise OSError("injected transaction failure")

        with tempfile.TemporaryDirectory() as td:
            sink, ledger, selection = _new_sink(Path(td) / "run", failure_hook=fail_after_apply)
            evidence, _ = _evidence(1, 0.0)
            evidence.record.validation_selected = True
            try:
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
                self.assertTrue(sink.poisoned)
                self.assertTrue(ledger.poisoned)
                self.assertEqual(selection.last_epoch, 1)
                with self.assertRaises(SelectionSinkError):
                    sink.on_epoch_complete(evidence)
            finally:
                ledger.close()

    def test_rejects_unexpected_fit_selection_callback(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run")
            try:
                with self.assertRaises(SelectionSinkError):
                    sink.on_selection(SimpleNamespace(epoch=1))
                self.assertTrue(sink.poisoned)
            finally:
                ledger.close()

    def test_transaction_identity_factories_must_be_unique(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run")
            try:
                sink._identifiers(1)
                sink.transaction_id_factory = lambda _epoch: "sink-tx-001"
                with self.assertRaises(SelectionSinkError):
                    sink._identifiers(2)
                self.assertTrue(sink.poisoned)
            finally:
                ledger.close()

    def test_utc_z_timestamp_is_normalized_to_frozen_schema_offset(self):
        with tempfile.TemporaryDirectory() as td:
            sink, ledger, _ = _new_sink(Path(td) / "run")
            try:
                sink.timestamp_factory = lambda _epoch: "2026-10-17T12:00:01Z"
                _, timestamp = sink._identifiers(1)
                self.assertEqual(timestamp, "2026-10-17T12:00:01+00:00")
            finally:
                ledger.close()

    def test_nonpristine_gr_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = (Path(td) / "run").resolve()
            root.mkdir()
            ledger = tx.SelectionLedger(root, _context())
            _, identities = _evidence(1, 0.0)
            identities["schema_sha256"] = codec.SCHEMA_SHA256
            state = gr.ValidationOnlySelection(last_epoch=1)
            try:
                with self.assertRaises(SelectionSinkError):
                    Figure10SelectionSink(
                        ledger=ledger, selection=state, identity_map=identities,
                        provenance_factory=lambda _: {},
                        transaction_id_factory=lambda epoch: f"tx-{epoch}",
                        timestamp_factory=lambda epoch: f"2026-10-17T12:00:{epoch:02d}+00:00")
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
