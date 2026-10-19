from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import Mock

from representation_collapse import tgrab_selected_checkpoint as codec
from representation_collapse.tgrab_figure10_attempt_journal import Figure10AttemptJournal
from representation_collapse.tgrab_figure10_journaled_fit import (
    JournaledFitError,
    run_journaled_figure10_fit,
)
from representation_collapse.tgrab_figure10_native_fit import Figure10NativeFitAdapter
from representation_collapse.tgrab_figure10_selection_sink import Figure10SelectionSink
from representation_collapse.tgrab_selection_transaction import SelectionContext, SelectionLedger


def _context(*, attempt_id="attempt-a"):
    return SelectionContext(
        run_id="run-a", attempt_id=attempt_id, run_admission_sha256="a" * 64,
        schema_sha256="b" * 64, expected_head_sha256=None)


def _components(context=None):
    context = _context() if context is None else context
    journal = object.__new__(Figure10AttemptJournal)
    journal.context = context
    journal.on_attempt = Mock(name="on_attempt")
    journal.on_update = Mock(name="on_update")
    journal.close = Mock(name="close")

    sink = object.__new__(Figure10SelectionSink)
    sink.ledger = object.__new__(SelectionLedger)
    sink.ledger.context = context
    sink.on_epoch_complete = Mock(name="on_epoch_complete")
    sink.on_selection = Mock(name="on_selection")

    adapter = object.__new__(Figure10NativeFitAdapter)
    adapter.fit = Mock(name="fit")
    return adapter, journal, sink


class JournaledFigure10FitTests(unittest.TestCase):
    def test_wires_all_callbacks_passes_guard_and_preserves_fit_result(self):
        adapter, journal, sink = _components()
        guard = Mock(name="guard")
        result = (object(), ("validation",))
        adapter.fit.return_value = result

        returned = run_journaled_figure10_fit(
            adapter, attempt_journal=journal, selection_sink=sink, guard=guard)

        self.assertIs(returned, result)
        adapter.fit.assert_called_once()
        kwargs = adapter.fit.call_args.kwargs
        self.assertIs(kwargs["guard"], guard)
        self.assertIs(kwargs["on_attempt"], journal.on_attempt)
        self.assertIs(kwargs["on_update"], journal.on_update)
        self.assertIs(kwargs["on_epoch_complete"], sink.on_epoch_complete)
        self.assertIs(kwargs["on_selection"], sink.on_selection)
        journal.close.assert_called_once_with()

    def test_context_mismatch_rejects_before_fit_and_closes_journal(self):
        adapter, journal, sink = _components()
        sink.ledger.context = _context(attempt_id="attempt-b")

        with self.assertRaisesRegex(JournaledFitError, "context"):
            run_journaled_figure10_fit(
                adapter, attempt_journal=journal, selection_sink=sink)

        adapter.fit.assert_not_called()
        journal.close.assert_called_once_with()

    def test_fit_exception_closes_without_retry(self):
        adapter, journal, sink = _components()
        failure = RuntimeError("fit failed")
        adapter.fit.side_effect = failure

        with self.assertRaisesRegex(RuntimeError, "fit failed") as caught:
            run_journaled_figure10_fit(
                adapter, attempt_journal=journal, selection_sink=sink)

        self.assertIs(caught.exception, failure)
        adapter.fit.assert_called_once()
        journal.close.assert_called_once_with()

    def test_wrong_component_type_rejects_before_fit_and_closes_journal(self):
        adapter, journal, _ = _components()

        with self.assertRaisesRegex(JournaledFitError, "SelectionSink"):
            run_journaled_figure10_fit(
                adapter, attempt_journal=journal, selection_sink=object())

        adapter.fit.assert_not_called()
        journal.close.assert_called_once_with()

    def test_subclassed_sink_is_rejected_and_real_journal_closes(self):
        class OverrideSink(Figure10SelectionSink):
            def on_selection(self, record):
                raise AssertionError("overridden callback must never be dispatched")

        with tempfile.TemporaryDirectory() as td:
            attempts_dir = Path(td) / "attempts"
            attempts_dir.mkdir()
            context = SelectionContext(
                "run-a", "attempt-a", "a" * 64, codec.SCHEMA_SHA256, None)
            journal = Figure10AttemptJournal(attempts_dir, context)
            sink = object.__new__(OverrideSink)
            adapter = object.__new__(Figure10NativeFitAdapter)
            adapter.fit = Mock(name="fit")

            with self.assertRaisesRegex(JournaledFitError, "SelectionSink"):
                run_journaled_figure10_fit(
                    adapter, attempt_journal=journal, selection_sink=sink)

            adapter.fit.assert_not_called()
            self.assertTrue(journal._closed)
            self.assertIsNone(journal._dir_fd)

    def test_subclassed_journal_is_rejected_and_base_descriptors_close(self):
        class OverrideJournal(Figure10AttemptJournal):
            def close(self):
                raise AssertionError("overridden close must never be dispatched")

        with tempfile.TemporaryDirectory() as td:
            attempts_dir = Path(td) / "attempts"
            attempts_dir.mkdir()
            journal = OverrideJournal(attempts_dir, _context())
            adapter = object.__new__(Figure10NativeFitAdapter)
            adapter.fit = Mock(name="fit")
            sink = object.__new__(Figure10SelectionSink)

            with self.assertRaisesRegex(JournaledFitError, "AttemptJournal"):
                run_journaled_figure10_fit(
                    adapter, attempt_journal=journal, selection_sink=sink)

            adapter.fit.assert_not_called()
            self.assertTrue(journal._closed)
            self.assertIsNone(journal._dir_fd)
            self.assertIsNone(journal._lock_fd)


if __name__ == "__main__":
    unittest.main()
