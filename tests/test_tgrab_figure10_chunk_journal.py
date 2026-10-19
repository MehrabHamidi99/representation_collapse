"""Synthetic callback/durability tests; no model/data execution."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from representation_collapse.tgrab_figure10_chunk_journal import (
    ChunkJournalContext, ChunkJournalError, Figure10ChunkJournal, UPDATES,
)
from tests.test_tgrab_figure10_attempt_journal import _attempt, _update


def context():
    return ChunkJournalContext("chunk-run", "fit02", "a" * 64, "b" * 64)


def callbacks(epoch, target):
    sequence = (epoch - 1) * UPDATES + target
    return _attempt(sequence, epoch=epoch, target=target), _update(sequence, epoch=epoch, target=target)


class ChunkJournalTests(unittest.TestCase):
    def test_intent_fsync_precedes_return_and_pending_stop_audits(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            with patch("representation_collapse.tgrab_figure10_chunk_journal.os.fsync", wraps=os.fsync) as sync:
                journal.on_attempt(_attempt(1))
                self.assertGreaterEqual(sync.call_count, 2)
            self.assertEqual(journal.audit()["pending"]["sequence"], 1)
            raw = (Path(td) / "epoch-001.jsonl").read_bytes()
            self.assertEqual(json.loads(raw)["kind"], "figure10_longfit_optimizer_intent_v2")
            self.assertEqual(json.loads(raw)["admission_sha256"], "b" * 64)
            journal.on_update(_update(1))
            audit = journal.audit()
            self.assertEqual((audit["attempted_updates"], audit["completed_updates"], audit["pending"]), (1, 1, None))

    def test_fsync_failure_leaves_charged_or_ambiguous_bytes_terminal(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            journal.on_attempt(_attempt(1))
            with patch("representation_collapse.tgrab_figure10_chunk_journal.os.fsync", side_effect=OSError("disk")):
                with self.assertRaises(ChunkJournalError):
                    journal.on_update(_update(1))
            self.assertTrue(journal.poisoned)
            self.assertEqual(journal.pending["sequence"], 1)
            with self.assertRaises(ChunkJournalError):
                journal.on_update(_update(1))
            self.assertTrue((Path(td) / "epoch-001.jsonl").exists())

    def test_short_writes_are_completed_and_zero_write_poisoned(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            real_write = os.write
            with patch("representation_collapse.tgrab_figure10_chunk_journal.os.write", side_effect=lambda fd, data: real_write(fd, data[:17])):
                journal.on_attempt(_attempt(1))
            self.assertEqual(journal.audit()["pending"]["sequence"], 1)
            with patch("representation_collapse.tgrab_figure10_chunk_journal.os.write", return_value=0):
                with self.assertRaises(ChunkJournalError):
                    journal.on_update(_update(1))
            self.assertTrue(journal.poisoned)

    def test_active_tamper_or_torn_prefix_fails_before_next_callback(self):
        for torn in (False, True):
            with self.subTest(torn=torn), tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
                journal.on_attempt(_attempt(1))
                path = Path(td) / "epoch-001.jsonl"
                payload = path.read_bytes()
                path.write_bytes(payload[:-1] if torn else payload.replace(b'"epoch":1', b'"epoch":2'))
                with self.assertRaises(ChunkJournalError):
                    journal.on_update(_update(1))
                self.assertTrue(journal.poisoned)

    def test_backend_failure_outcome_terminal_but_honestly_auditable(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            journal.on_attempt(_attempt(1))
            journal.on_update(_update(1, outcome="unknown", completed_return=False, error_type="RuntimeError"))
            audit = journal.audit()
            self.assertEqual((audit["attempted_updates"], audit["completed_updates"], audit["optimizer_steps_confirmed"]), (1, 0, 0))
            self.assertTrue(audit["terminal"])
            with self.assertRaises(ChunkJournalError):
                journal.on_attempt(_attempt(2))

    def test_confirmed_step_with_reported_numerical_error_is_terminal(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            journal.on_attempt(_attempt(1))
            journal.on_update(_update(1, error_type="FitProtocolError"))
            audit = journal.audit()
            self.assertEqual((audit["attempted_updates"], audit["completed_updates"], audit["optimizer_steps_confirmed"]), (1, 1, 1))
            self.assertTrue(audit["terminal"])
            with self.assertRaises(ChunkJournalError):
                journal.on_attempt(_attempt(2))

    def test_bad_chronology_and_incomplete_seal_terminal(self):
        for action in (lambda j: j.on_attempt(_attempt(2)), lambda j: j.seal_epoch(1)):
            with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
                with self.assertRaises(ChunkJournalError):
                    action(journal)
                self.assertTrue(journal.poisoned)

    def test_two_epochs_linear_files_and_no_repeated_historical_reads(self):
        with tempfile.TemporaryDirectory() as td, Figure10ChunkJournal(Path(td).resolve(), context()) as journal:
            with patch.object(journal, "_read", wraps=journal._read) as reads:
                for epoch in (1, 2):
                    for target in range(1, UPDATES + 1):
                        attempt, update = callbacks(epoch, target)
                        journal.on_attempt(attempt); journal.on_update(update)
                    receipt = journal.seal_epoch(epoch)
                    self.assertEqual(receipt["records"], 5118)
                    self.assertEqual(reads.call_count, epoch)
                audit = journal.audit()
                self.assertEqual(reads.call_count, 6)  # two seals + two chunks/two receipts
            self.assertEqual(len(list(Path(td).iterdir())), 5)  # two chunks, two seals, owner
            self.assertEqual(audit["completed_updates"], 5118)
            first = json.loads((Path(td) / "epoch-001.seal.json").read_bytes())
            second = json.loads((Path(td) / "epoch-002.seal.json").read_bytes())
            first_second = json.loads((Path(td) / "epoch-002.jsonl").read_bytes().splitlines()[0])
            self.assertEqual(second["previous_chunk_sha256"], first["chunk_sha256"])
            self.assertEqual(first_second["previous_record_sha256"], first["last_record_sha256"])
            self.assertEqual(first_second["previous_chunk_sha256"], first["chunk_sha256"])
            self.assertEqual(second["chunk_sha256"], hashlib.sha256((Path(td) / "epoch-002.jsonl").read_bytes()).hexdigest())
            # Earlier sealed tampering is detected by the final single audit.
            path = Path(td) / "epoch-001.jsonl"
            path.chmod(0o600)
            with path.open("r+b") as handle:
                handle.write(b"!")
            with self.assertRaises(ChunkJournalError):
                journal.audit()

    def test_fresh_only_no_reopen_and_no_unsealed_epoch_advance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            with Figure10ChunkJournal(root, context()) as journal:
                with self.assertRaises(ChunkJournalError):
                    Figure10ChunkJournal(root, context())
            with self.assertRaises(ChunkJournalError):
                Figure10ChunkJournal(root, context())


if __name__ == "__main__":
    unittest.main()
