import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from representation_collapse import tgrab_selected_checkpoint as codec
from representation_collapse import tgrab_selection_transaction as tx
from tests.test_tgrab_selected_checkpoint import synthetic_valid_bundle


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "fixture-run"
ATTEMPT_ID = "fixture-attempt"
ADMISSION_SHA = "a" * 64
STAMP = "2026-10-16T12:00:00+00:00"


def candidate_files(epoch: int, f1: float) -> dict[str, bytes]:
    files = synthetic_valid_bundle()
    validation = json.loads(files["validation.json"])
    validation["epoch"] = epoch
    if f1 == 0.0:
        counts = (0, 0, 2, 9898)
    elif f1 == 1.0:
        counts = (2, 0, 0, 9898)
    else:
        raise AssertionError("fixture uses only exact scores zero or one")
    tp, fp, fn, tn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    accuracy = (tp + tn) / 9900
    for row in validation["rows"]:
        row.update(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision,
                   recall=recall, f1=score, accuracy=accuracy)
    validation["time_mean_f1"] = score
    files["validation.json"] = codec._canonical(validation)

    provenance = json.loads(files["provenance.json"])
    provenance["run_id"] = RUN_ID
    provenance["attempt_id"] = ATTEMPT_ID
    provenance["admission_sha256"] = ADMISSION_SHA
    provenance["counters"].update(complete_epochs=epoch, validation_epochs=epoch,
                                  attempted_updates=epoch * 2559, completed_updates=epoch * 2559)
    files["provenance.json"] = codec._canonical(provenance)

    manifest = json.loads(files["manifest.json"])
    manifest["epoch"] = epoch
    manifest["terminal_state"]["epoch"] = epoch
    manifest["validation"].update(bytes=len(files["validation.json"]),
                                  sha256=hashlib.sha256(files["validation.json"]).hexdigest(),
                                  time_mean_f1=score)
    manifest["provenance"].update(bytes=len(files["provenance.json"]),
                                  sha256=hashlib.sha256(files["provenance.json"]).hexdigest())
    files["manifest.json"] = codec._canonical(manifest)
    return files


def stage_candidate(root: Path, epoch: int, score: float) -> str:
    relative = f"candidate-{epoch:03d}"
    codec.write_candidate(root / relative, candidate_files(epoch, score))
    return relative


def context(expected_head: str | None = None) -> tx.SelectionContext:
    return tx.SelectionContext(RUN_ID, ATTEMPT_ID, ADMISSION_SHA,
                               codec.SCHEMA_SHA256, expected_head)


class FixtureGRSelection:
    """Test double only; no native or project selection object is imported."""
    def __init__(self, selected=None, best_f1=None, last_epoch=0, finalized=False):
        self.selected, self.best_f1 = selected, best_f1
        self.last_epoch, self.finalized = last_epoch, finalized

    def apply(self, record):
        if record["kind"] == "tgrab_validation_selection_decision_v1":
            if self.finalized or record["epoch"] != self.last_epoch + 1:
                raise RuntimeError("fixture GR progression rejected")
            self.last_epoch = record["epoch"]
            if record["decision"] == "selected":
                self.selected, self.best_f1 = record["candidate"], record["candidate_f1"]
        else:
            if self.finalized or self.selected is None:
                raise RuntimeError("fixture GR finalization rejected")
            self.finalized = True
        return self.snapshot()

    def snapshot(self):
        return {"selected": self.selected, "best_f1": self.best_f1,
                "last_epoch": self.last_epoch, "finalized": self.finalized}

    @staticmethod
    def verify(observed, record):
        expected = {"selected": record["selected"], "best_f1": record["best_f1"],
                    "last_epoch": record["last_epoch"], "finalized": record["finalized"]}
        return observed == expected


def callbacks(state):
    return {"apply_selection": state.apply, "verify_selection": state.verify}


class PureDecisionTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"epoch": 1, "checkpoint_sha256": "b" * 64,
                          "recurrent_sha256": "c" * 64, "link_head_sha256": "d" * 64}

    def make(self, sequence, score, previous_receipt=None, previous_selected=None, previous_best=None):
        candidate = {**self.candidate, "epoch": sequence}
        return tx.decide(sequence=sequence, candidate=candidate, candidate_f1=score,
                         candidate_manifest_relative_path=f"candidate-{sequence:03d}/manifest.json",
                         candidate_validation_sha256="e" * 64,
                         previous_receipt=previous_receipt, previous_selected=previous_selected,
                         previous_best_f1=previous_best)

    def test_first_zero_selects_and_strict_ties_retain_earliest(self):
        first = self.make(1, 0.0)
        self.assertEqual(first["decision"], "selected")
        self.assertEqual(first["selected"], self.candidate)
        receipt = {"role": "immutable_selection_receipt", "relative_path": "selection/receipts/r1.json",
                   "bytes": 100, "sha256": "f" * 64}
        tied = self.make(2, 0.0, receipt, first["selected"], 0.0)
        self.assertEqual(tied["decision"], "retained")
        self.assertEqual(tied["selected"], first["selected"])
        self.assertFalse(tied["test_authorized_by_this_receipt"])

    def test_strict_improvement_and_invalid_progression_reject(self):
        first = self.make(1, 0.2)
        receipt = {"role": "immutable_selection_receipt", "relative_path": "selection/receipts/r1.json",
                   "bytes": 100, "sha256": "f" * 64}
        improved = self.make(2, 0.3, receipt, first["selected"], 0.2)
        self.assertEqual(improved["decision"], "selected")
        self.assertEqual(improved["best_f1"], 0.3)
        lower = self.make(2, 0.1, receipt, first["selected"], 0.2)
        self.assertEqual(lower["decision"], "retained")
        self.assertEqual(lower["selected"], first["selected"])
        with self.assertRaises(tx.SelectionError):
            self.make(True, 0.1)
        with self.assertRaises(tx.SelectionError):
            self.make(201, 0.1)

    def test_intent_identity_must_match_receipt_transaction_sequence_time(self):
        receipt = {"transaction_id": "tx-one", "sequence": 1,
                   "created_at_utc": STAMP}
        intent = dict(receipt)
        tx._require_intent_receipt_match(intent, receipt, 1)
        for key, value in (("transaction_id", "tx-other"), ("sequence", 2),
                           ("created_at_utc", "2026-10-16T12:00:01+00:00")):
            mutated = dict(intent)
            mutated[key] = value
            with self.subTest(key=key), self.assertRaises(tx.SelectionError):
                tx._require_intent_receipt_match(mutated, receipt, 1)


class DurableTransactionTests(unittest.TestCase):
    def test_no_head_rejects_stale_selected_alias_before_transaction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            (root / "selected.json").write_bytes(b"{}\n")
            with self.assertRaisesRegex(tx.SelectionError, "alias exists without a committed head"):
                tx.SelectionLedger(root, context())
            self.assertFalse((root / "selection-head.json").exists())
            self.assertEqual((root / "selected.json").read_bytes(), b"{}\n")

    def test_select_retain_improve_reopen_and_finalize(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for epoch, score in ((1, 0.0), (2, 0.0), (3, 1.0), (4, 1.0)):
                stage_candidate(root, epoch, score)
            ledger = tx.SelectionLedger(root, context())
            state = FixtureGRSelection()
            cb = callbacks(state)
            try:
                first = ledger.commit_candidate("candidate-001", transaction_id="tx-one", created_at_utc=STAMP, **cb)
                p1 = first.pointer_reference
                alias_after_first = (root / "selected.json").read_bytes()
                second = ledger.commit_candidate("candidate-002", transaction_id="tx-two", created_at_utc=STAMP, **cb)
                self.assertEqual(second.decision, "retained")
                self.assertEqual(second.pointer_reference, p1)
                self.assertEqual((root / "selected.json").read_bytes(), alias_after_first)
                third = ledger.commit_candidate("candidate-003", transaction_id="tx-three", created_at_utc=STAMP, **cb)
                self.assertEqual(third.decision, "selected")
                self.assertNotEqual(third.pointer_reference, p1)
                fourth = ledger.commit_candidate("candidate-004", transaction_id="tx-four", created_at_utc=STAMP, **cb)
                self.assertEqual(fourth.decision, "retained")
                self.assertEqual(fourth.pointer_reference, third.pointer_reference)
                expected = fourth.head_sha256
            finally:
                ledger.close()
            reopened = tx.SelectionLedger(root, context(expected))
            state2 = FixtureGRSelection(fourth.selected, fourth.best_f1, fourth.sequence)
            cb2 = callbacks(state2)
            try:
                head = reopened.verify_committed_state()
                self.assertEqual(head["sequence"], 4)
                final = reopened.finalize(transaction_id="tx-final", run_stop_reason="complete_epoch_cap",
                                          created_at_utc=STAMP, **cb2)
                self.assertTrue(final.finalized)
                with self.assertRaises(tx.SelectionError):
                    reopened.commit_candidate("candidate-004", transaction_id="tx-after-final", created_at_utc=STAMP, **cb2)
            finally:
                reopened.close()
            finalized_sha = json.loads((root / "selection-head.json").read_text())
            self.assertTrue(finalized_sha["finalized"])
            with tx.SelectionLedger(root, context(final.head_sha256)) as final_reopen:
                self.assertTrue(final_reopen.verify_committed_state()["finalized"])

    def test_candidate_mismatch_fails_closed_before_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            relative = stage_candidate(root, 1, 0.0)
            bundle = root / relative
            files = {name: (bundle / name).read_bytes() for name in codec.FILES}
            provenance = json.loads(files["provenance.json"])
            provenance["run_id"] = "other-run"
            files["provenance.json"] = codec._canonical(provenance)
            manifest = json.loads(files["manifest.json"])
            manifest["provenance"].update(bytes=len(files["provenance.json"]),
                                         sha256=hashlib.sha256(files["provenance.json"]).hexdigest())
            files["manifest.json"] = codec._canonical(manifest)
            # This is a synthetic rewrite only; create-only semantics require a fresh path.
            import shutil
            shutil.rmtree(bundle)
            codec.write_candidate(bundle, files)
            ledger = tx.SelectionLedger(root, context())
            state = FixtureGRSelection()
            try:
                with self.assertRaises(tx.SelectionError):
                    ledger.commit_candidate("candidate-001", transaction_id="valid-tx", created_at_utc=STAMP, **callbacks(state))
                self.assertEqual(ledger._head_bytes(), None)
                failure = codec.validate_metadata((root / "selection/failures/failure-001-valid-tx.json").read_bytes(),
                                                  expected_kind="tgrab_selection_transaction_failure_v1")
                self.assertEqual(failure["failed_stage"], "payload")
            finally:
                ledger.close()

    def test_reopen_refuses_orphan_transaction_objects(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            stage_candidate(root, 1, 0.0)
            def crash_at(name):
                if name == "receipt:before":
                    raise OSError("synthetic crash")
            ledger = tx.SelectionLedger(root, context(), failure_hook=crash_at)
            state = FixtureGRSelection()
            try:
                with self.assertRaises(tx.SelectionError):
                    ledger.commit_candidate("candidate-001", transaction_id="orphan", created_at_utc=STAMP, **callbacks(state))
            finally:
                ledger.close()
            with self.assertRaises(tx.SelectionError):
                tx.SelectionLedger(root, context())

    def test_failure_injection_each_stage_is_terminal_and_records_stage(self):
        stages = ("transaction_intent", "receipt", "pointer_object", "selection_apply",
                  "selection_verify", "selected_alias", "head_replace", "head_fsync")
        cases = [(stage, "after" if stage == "head_fsync" else "before") for stage in stages]
        cases.extend((("transaction_intent", "after"), ("selection_apply", "after"), ("head_replace", "after")))
        for stage, when in cases:
            with self.subTest(stage=stage, when=when), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                stage_candidate(root, 1, 0.0)
                def crash_at(name, expected=f"{stage}:{when}"):
                    if name == expected:
                        raise OSError("injected")
                ledger = tx.SelectionLedger(root, context(), failure_hook=crash_at)
                state = FixtureGRSelection()
                try:
                    with self.assertRaises(tx.SelectionError):
                        ledger.commit_candidate("candidate-001", transaction_id=f"fault-{stage}", created_at_utc=STAMP, **callbacks(state))
                    names = os.listdir(root / "selection/failures")
                    self.assertEqual(len(names), 1)
                    failure = codec.validate_metadata((root / "selection/failures" / names[0]).read_bytes(),
                                                      expected_kind="tgrab_selection_transaction_failure_v1")
                    self.assertEqual(failure["failed_stage"], stage)
                    self.assertTrue(failure["terminal"])
                    self.assertFalse(failure["automatic_retry"])
                    if stage == "transaction_intent" and when == "before":
                        self.assertIsNone(failure["intent"])
                    if stage == "transaction_intent" and when == "after":
                        self.assertIsNotNone(failure["intent"])
                    if stage == "head_fsync" or (stage == "head_replace" and when == "after"):
                        self.assertEqual(failure["commit_state"], "uncertain_after_head_replace")
                    elif stage == "selected_alias" or stage == "head_replace":
                        self.assertEqual(failure["commit_state"], "not_committed")
                finally:
                    ledger.close()
                with self.assertRaises(tx.SelectionError):
                    tx.SelectionLedger(root, context())

    def test_mutated_committed_receipt_and_alias_reject_on_reopen(self):
        for target in ("receipt", "intent", "pointer", "alias"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                stage_candidate(root, 1, 0.0)
                with tx.SelectionLedger(root, context()) as ledger:
                    result = ledger.commit_candidate("candidate-001", transaction_id="tx-one", created_at_utc=STAMP, **callbacks(FixtureGRSelection()))
                if target == "alias":
                    (root / "selected.json").write_bytes(b"{}\n")
                else:
                    ref = {"receipt": result.receipt_reference,
                           "intent": json.loads((root / "selection-head.json").read_text())["transaction_intent"],
                           "pointer": result.pointer_reference}[target]
                    artifact = root / ref["relative_path"]
                    artifact.write_bytes(artifact.read_bytes() + b" ")
                with self.assertRaises(tx.SelectionError):
                    tx.SelectionLedger(root, context(result.head_sha256))

    def test_wrong_external_head_sha_and_second_owner_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            stage_candidate(root, 1, 0.0)
            owner = tx.SelectionLedger(root, context())
            with self.assertRaises(tx.SelectionError):
                tx.SelectionLedger(root, context())
            result = owner.commit_candidate("candidate-001", transaction_id="tx-one", created_at_utc=STAMP, **callbacks(FixtureGRSelection()))
            owner.close()
            with self.assertRaises(tx.SelectionError):
                tx.SelectionLedger(root, context("0" * 64))

    def test_payload_failure_is_terminal_and_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            stage_candidate(root, 1, 0.0)
            payload = root / "candidate-001/recurrent.f32le"
            payload.write_bytes(payload.read_bytes()[:-1])
            ledger = tx.SelectionLedger(root, context())
            state = FixtureGRSelection()
            try:
                with self.assertRaises(tx.SelectionError):
                    ledger.commit_candidate("candidate-001", transaction_id="bad-payload", created_at_utc=STAMP, **callbacks(state))
                failure = codec.validate_metadata((root / "selection/failures/failure-001-bad-payload.json").read_bytes(),
                                                  expected_kind="tgrab_selection_transaction_failure_v1")
                self.assertEqual(failure["failed_stage"], "payload")
            finally:
                ledger.close()
            with self.assertRaises(tx.SelectionError):
                tx.SelectionLedger(root, context())


if __name__ == "__main__":
    unittest.main()
