from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse import tgrab_selected_checkpoint as codec  # noqa: E402
from representation_collapse import tgrab_selection_transaction as tx  # noqa: E402
from representation_collapse.tgrab_figure10_attempt_journal import (  # noqa: E402
    AttemptJournalError,
    Figure10AttemptJournal,
)


RUN_ID = "journal-run"
ATTEMPT_ID = "journal-attempt"
ADMISSION = "a" * 64


def _context():
    return tx.SelectionContext(RUN_ID, ATTEMPT_ID, ADMISSION,
                               codec.SCHEMA_SHA256, None)


def _attempt(sequence: int, *, epoch: int = 1, target: int | None = None):
    target = sequence if target is None else target
    return {
        "kind": "tgrab-figure10-fit-attempt-v1",
        "sequence": sequence,
        "epoch": epoch,
        "training_order_index": target - 1,
        "target_timestamp": target,
        "consumed_timestamp": target - 1,
    }


def _update(sequence: int, *, outcome="completed", completed_return=True,
            error_type=None, epoch=1, target=None):
    target = sequence if target is None else target
    return {
        "epoch": epoch,
        "target_timestamp": target,
        "consumed_timestamp": target - 1,
        "attempt": sequence,
        "attempted_updates": sequence,
        "completed_updates": sequence if completed_return else sequence - 1,
        "epoch_attempted_updates": target,
        "epoch_completed_updates": target if completed_return else target - 1,
        "optimizer_steps_confirmed": sequence if outcome == "completed" else sequence - 1,
        "completed": completed_return,
        "optimizer_step_status": outcome,
        "error_type": error_type,
    }


def _read(root: Path, seq: int, suffix: str):
    path = root / f"{seq:08d}.{suffix}.json"
    return json.loads(path.read_text())


class Figure10AttemptJournalTests(unittest.TestCase):
    def test_cadenced_audit_preserves_pending_integrity_and_forced_prior_audit(self):
        with tempfile.TemporaryDirectory() as td:
            with Figure10AttemptJournal(td, _context(), namespace_audit_interval=256) as journal:
                journal.on_attempt(_attempt(1))
                journal.on_update(_update(1))
                prior = Path(td) / "00000001.outcome.json"
                prior.write_bytes(prior.read_bytes().replace(b'"completed_return":true', b'"completed_return":false'))
                # A pending intent is still protected immediately off cadence.
                journal.on_attempt(_attempt(2))
                with self.assertRaises(AttemptJournalError):
                    journal.audit()
                self.assertTrue(journal.poisoned)

    def test_off_cadence_pending_intent_mutation_prevents_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            with Figure10AttemptJournal(td, _context(), namespace_audit_interval=256) as journal:
                journal.on_attempt(_attempt(1))
                path = Path(td) / "00000001.intent.json"
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(AttemptJournalError):
                    journal.on_update(_update(1))
                self.assertFalse((Path(td) / "00000001.outcome.json").exists())

    def test_full_namespace_work_is_amortized_and_forced_audit_catches_extra_file(self):
        with tempfile.TemporaryDirectory() as td:
            with Figure10AttemptJournal(td, _context(), namespace_audit_interval=256) as journal:
                with mock.patch.object(journal, "_verify_namespace", wraps=journal._verify_namespace) as scans, mock.patch.object(journal, "_read_file", wraps=journal._read_file) as reads:
                    for sequence in range(1, 513):
                        journal.on_attempt(_attempt(sequence))
                        journal.on_update(_update(sequence))
                    journal.audit()
                    # 1024 callbacks require four cadence scans + boundary.
                    self.assertLessEqual(scans.call_count, 5)
                    # Bound total historical+pending reads, rather than timing
                    # flaky filesystem performance or mirroring name generation.
                    self.assertLess(reads.call_count, 5000)
                (Path(td) / "unexpected").write_text("tampered")
                with self.assertRaises(AttemptJournalError):
                    journal.audit()

    def test_attempt_is_fsynced_before_callback_returns_and_pairs_with_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            fsyncs = []
            real_fsync = os.fsync

            def observed_fsync(fd):
                fsyncs.append(fd)
                return real_fsync(fd)

            try:
                with mock.patch("representation_collapse.tgrab_figure10_attempt_journal.os.fsync",
                                side_effect=observed_fsync):
                    before = len(fsyncs)
                    journal.on_attempt(_attempt(1))
                    # The caller reaches its optimizer only after this returns.
                    self.assertGreaterEqual(len(fsyncs) - before, 2)
                    self.assertEqual(_read(root, 1, "intent")["sequence"], 1)
                    journal.on_update(_update(1))
                intent = _read(root, 1, "intent")
                outcome = _read(root, 1, "outcome")
                self.assertEqual(intent["run_id"], RUN_ID)
                self.assertEqual(intent["attempt_id"], ATTEMPT_ID)
                self.assertEqual(intent["admission_sha256"], ADMISSION)
                self.assertEqual(outcome["outcome"], "completed")
                self.assertEqual(outcome["sequence"], intent["sequence"])
                self.assertEqual(outcome["intent_sha256"], hashlib.sha256(
                    (root / "00000001.intent.json").read_bytes()).hexdigest())
                self.assertFalse(journal.poisoned)
            finally:
                journal.close()

    def test_fsync_failure_in_intent_callback_prevents_optimizer_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            optimizer_calls = []
            try:
                with mock.patch("representation_collapse.tgrab_figure10_attempt_journal.os.fsync",
                                side_effect=OSError("disk failure")):
                    with self.assertRaises(AttemptJournalError):
                        journal.on_attempt(_attempt(1))
                    # Mirrors the fit core: train_update is after successful return.
                    if not journal.poisoned:
                        optimizer_calls.append(1)
                self.assertEqual(optimizer_calls, [])
                self.assertTrue(journal.poisoned)
            finally:
                journal.close()

    def test_truthful_completed_not_started_and_unknown_outcomes(self):
        cases = (
            ("completed", True, None),
            ("not_started", False, "OptimizerAttemptError"),
            ("unknown", False, "OptimizerAttemptError"),
        )
        for status, completed, error in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "attempts"
                root.mkdir()
                journal = Figure10AttemptJournal(root, _context())
                try:
                    seq = 1
                    journal.on_attempt(_attempt(seq))
                    journal.on_update(_update(seq, outcome=status,
                                              completed_return=completed,
                                              error_type=error))
                    saved = _read(root, seq, "outcome")
                    self.assertEqual(saved["outcome"], status)
                    self.assertEqual(saved["completed_return"], completed)
                    self.assertEqual(saved["error_type"], error)
                    self.assertEqual(saved["epoch_completed_updates"],
                                     seq if completed else seq - 1)
                    self.assertEqual(saved["optimizer_steps_confirmed"],
                                     seq if status == "completed" else seq - 1)
                finally:
                    journal.close()

    def test_failed_callback_outcome_is_terminal_and_preserved(self):
        # The fit core re-raises after all backend exceptions, regardless of
        # whether a step was confirmed complete, definitely not started, or
        # remains unknown. No second intent may follow any such outcome.
        for status in ("completed", "not_started", "unknown"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "attempts"
                root.mkdir()
                journal = Figure10AttemptJournal(root, _context())
                try:
                    journal.on_attempt(_attempt(1))
                    journal.on_update(_update(1, outcome=status,
                                              completed_return=False,
                                              error_type="OptimizerAttemptError"))
                    saved = _read(root, 1, "outcome")
                    self.assertEqual(saved["outcome"], status)
                    self.assertFalse(saved["completed_return"])
                    with self.assertRaisesRegex(AttemptJournalError, "journal is terminal"):
                        journal.on_attempt(_attempt(2))
                    self.assertFalse((root / "00000002.intent.json").exists())
                    self.assertEqual(_read(root, 1, "outcome"), saved)
                finally:
                    journal.close()

    def test_raised_after_step_is_completed_but_not_a_completed_return(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                journal.on_attempt(_attempt(1))
                journal.on_update(_update(1, outcome="completed",
                                          completed_return=False,
                                          error_type="OptimizerAttemptError"))
                saved = _read(root, 1, "outcome")
                self.assertEqual(saved["optimizer_step_status"], "completed")
                self.assertFalse(saved["completed_return"])
                self.assertEqual(saved["completed_updates"], 0)
                self.assertEqual(saved["epoch_completed_updates"], 0)
                self.assertEqual(saved["optimizer_steps_confirmed"], 1)
            finally:
                journal.close()

    def test_inconsistent_epoch_or_optimizer_counters_are_rejected(self):
        invalid = (
            {"epoch_completed_updates": 0},  # Successful return must count in epoch.
            {"optimizer_steps_confirmed": 0},  # Successful call confirms a step.
            {"epoch_completed_updates": 1,
             "optimizer_steps_confirmed": 1},  # not_started did neither.
            {"optimizer_steps_confirmed": 1},  # unknown is not a confirmed step.
        )
        statuses = ("completed", "completed", "not_started", "unknown")
        completed_flags = (True, True, False, False)
        for index, (changes, status, completed) in enumerate(
                zip(invalid, statuses, completed_flags), start=1):
            with self.subTest(changes=changes, status=status), tempfile.TemporaryDirectory() as td:
                root = Path(td) / "attempts"
                root.mkdir()
                journal = Figure10AttemptJournal(root, _context())
                try:
                    journal.on_attempt(_attempt(1))
                    update = _update(1, outcome=status, completed_return=completed,
                                     error_type=None if completed else "OptimizerAttemptError")
                    update.update(changes)
                    with self.assertRaises(AttemptJournalError):
                        journal.on_update(update)
                    self.assertTrue(journal.poisoned)
                    self.assertFalse((root / "00000001.outcome.json").exists())
                finally:
                    journal.close()

    def test_incomplete_intent_is_charged_ambiguous_and_cannot_be_reopened(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            journal.on_attempt(_attempt(1))
            journal.close()  # Simulated process interruption before on_update.
            self.assertTrue((root / "00000001.intent.json").exists())
            self.assertFalse((root / "00000001.outcome.json").exists())
            with self.assertRaisesRegex(AttemptJournalError, "fresh attempts directory"):
                Figure10AttemptJournal(root, _context())

    def test_duplicate_gap_and_mismatched_update_poison_without_retry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                with self.assertRaises(AttemptJournalError):
                    journal.on_attempt(_attempt(2))
                self.assertTrue(journal.poisoned)
            finally:
                journal.close()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                journal.on_attempt(_attempt(1))
                with self.assertRaises(AttemptJournalError):
                    journal.on_attempt(_attempt(1))
                self.assertTrue(journal.poisoned)
            finally:
                journal.close()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                journal.on_attempt(_attempt(1))
                with self.assertRaises(AttemptJournalError):
                    journal.on_update(_update(2))
                self.assertFalse((root / "00000001.outcome.json").exists())
            finally:
                journal.close()

    def test_partial_nonempty_and_symlink_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            partial = base / "partial"
            partial.mkdir()
            (partial / "00000001.intent.json").write_bytes(b"{")
            with self.assertRaises(AttemptJournalError):
                Figure10AttemptJournal(partial, _context())

            real = base / "real"
            real.mkdir()
            link = base / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(AttemptJournalError):
                Figure10AttemptJournal(link, _context())

    def test_symlink_or_partial_namespace_tampering_is_detected_before_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                journal.on_attempt(_attempt(1))
                (root / "extra").symlink_to(root / "00000001.intent.json")
                with self.assertRaises(AttemptJournalError):
                    journal.on_update(_update(1))
                self.assertTrue(journal.poisoned)
                self.assertFalse((root / "00000001.outcome.json").exists())
            finally:
                journal.close()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "attempts"
            root.mkdir()
            journal = Figure10AttemptJournal(root, _context())
            try:
                journal.on_attempt(_attempt(1))
                (root / "00000001.intent.json").write_bytes(b"partial")
                with self.assertRaises(AttemptJournalError):
                    journal.on_update(_update(1))
            finally:
                journal.close()


if __name__ == "__main__":
    unittest.main()
