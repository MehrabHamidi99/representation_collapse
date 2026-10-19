"""Fail-closed orchestration of Figure 10 fit accounting and selection callbacks.

This helper only joins already-created single-use components. It does not
construct, reopen, recover, or retry a journal, ledger, sink, or fit adapter.
"""
from __future__ import annotations

from typing import Any, Callable

from .tgrab_figure10_attempt_journal import Figure10AttemptJournal
from .tgrab_figure10_native_fit import Figure10NativeFitAdapter
from .tgrab_figure10_selection_sink import Figure10SelectionSink
from .tgrab_selection_transaction import SelectionContext, SelectionLedger


class JournaledFitError(RuntimeError):
    """Invalid orchestration components or mismatched accounting context."""


def run_journaled_figure10_fit(
        adapter: Figure10NativeFitAdapter, *,
        attempt_journal: Figure10AttemptJournal,
        selection_sink: Figure10SelectionSink,
        guard: Callable[[], None] = lambda: None,
        policy: Any = None):
    """Run exactly one fit with durable attempt and selection callbacks.

    The journal must have been created for the exact immutable context held by
    the sink's existing ledger. Validation happens before entering ``fit``.
    Once the journal component is accepted, its ownership is always closed in
    ``finally``, including validation failures and fit/callback exceptions.
    The adapter's return value is passed through unchanged.
    """
    if type(attempt_journal) is not Figure10AttemptJournal:
        if isinstance(attempt_journal, Figure10AttemptJournal):
            # A subclass is untrusted for callback dispatch, but may already
            # own descriptors initialized by the base constructor. Close only
            # through the base implementation; arbitrary objects are untouched.
            Figure10AttemptJournal.close(attempt_journal)
        raise JournaledFitError("an existing Figure10AttemptJournal is required")
    try:
        if type(adapter) is not Figure10NativeFitAdapter:
            raise JournaledFitError("an existing Figure10NativeFitAdapter is required")
        if type(selection_sink) is not Figure10SelectionSink:
            raise JournaledFitError("an existing Figure10SelectionSink is required")
        ledger = getattr(selection_sink, "ledger", None)
        ledger_context = getattr(ledger, "context", None)
        journal_context = getattr(attempt_journal, "context", None)
        if (type(ledger) is not SelectionLedger
                or type(ledger_context) is not SelectionContext
                or type(journal_context) is not SelectionContext
                or journal_context != ledger_context):
            raise JournaledFitError(
                "attempt journal context must exactly match the selection ledger context")
        if not callable(guard):
            raise JournaledFitError("caller-provided guard must be callable")

        fit_kwargs = {
            "guard": guard,
            "on_attempt": attempt_journal.on_attempt,
            "on_update": attempt_journal.on_update,
            "on_epoch_complete": selection_sink.on_epoch_complete,
            "on_selection": selection_sink.on_selection,
        }
        if policy is not None:
            fit_kwargs["policy"] = policy
        return adapter.fit(**fit_kwargs)
    finally:
        attempt_journal.close()
