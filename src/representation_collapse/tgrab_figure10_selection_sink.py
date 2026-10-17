"""Production callback joining Figure 10 epoch evidence to durable selection.

The sink serializes each complete validation epoch, verifies the published
candidate, and advances the existing GR ``ValidationOnlySelection`` only
through the S07/S08 ledger transaction. It does not create trust in caller
supplied admission, provenance, or identity values.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping

from . import graph_reconstruction as gr
from . import tgrab_selected_checkpoint as codec
from . import tgrab_selection_transaction as tx
from .tgrab_figure10_candidate_encoder import encode_epoch_candidate


class SelectionSinkError(RuntimeError):
    """Selection callback failed or this single-use sink is terminal."""


_TX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


class Figure10SelectionSink:
    """Single-fit callback adapter for candidate publication and selection.

    A fresh sink requires an empty, admission-bound ledger and a pristine
    ``ValidationOnlySelection``. Existing histories need an independently
    designed recovery path and are rejected here. Every callback failure is
    terminal; the object never retries or resumes.

    ``identity_map`` and ``provenance_factory`` are externally supplied pins,
    not authenticated evidence. The factory is called once per complete epoch.
    Transaction IDs and timestamps must be supplied by deterministic factories
    and are checked for validity and uniqueness within this fit.
    """

    def __init__(self, *, ledger: tx.SelectionLedger,
                 selection: gr.ValidationOnlySelection,
                 identity_map: Mapping[str, Any],
                 provenance_factory: Callable[[Any], Mapping[str, Any]],
                 transaction_id_factory: Callable[[int], str],
                 timestamp_factory: Callable[[int], str]):
        if not isinstance(ledger, tx.SelectionLedger):
            raise SelectionSinkError("an explicit admission-bound SelectionLedger is required")
        if type(selection) is not gr.ValidationOnlySelection:
            raise SelectionSinkError("the pinned graph_reconstruction.ValidationOnlySelection is required")
        if not isinstance(identity_map, Mapping) or not callable(provenance_factory):
            raise SelectionSinkError("explicit identity map and provenance factory are required")
        if not callable(transaction_id_factory) or not callable(timestamp_factory):
            raise SelectionSinkError("unique transaction ID and timestamp factories are required")
        context = ledger.context
        if context.expected_head_sha256 is not None:
            raise SelectionSinkError("nonempty/recovery selection contexts are not supported by this sink")
        try:
            committed = ledger.verify_committed_state()
        except BaseException as exc:
            raise SelectionSinkError("ledger state is not safely verifiable") from exc
        if committed is not None:
            raise SelectionSinkError("nonempty selection history requires explicit recovery support")
        if (selection.selected is not None or selection.best_f1 is not None
                or selection.last_epoch != 0 or selection.finalized):
            raise SelectionSinkError("selection object must be pristine for a new ledger")
        try:
            identity_bytes = codec._canonical(dict(identity_map))
            frozen_identities = codec.parse_canonical_json(identity_bytes, max_bytes=16384)
        except BaseException as exc:
            raise SelectionSinkError("identity map must be bounded canonical JSON data") from exc
        root = ledger.root
        candidates_dir = root / "candidates"
        try:
            os.mkdir(candidates_dir, 0o700)
            st = os.lstat(candidates_dir)
            if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
                raise OSError("candidate namespace is not a regular directory")
        except BaseException as exc:
            raise SelectionSinkError("candidate namespace must be new and create-only") from exc

        self.ledger = ledger
        self.selection = selection
        self._identity_bytes = identity_bytes
        self._stable_provenance_bytes: bytes | None = None
        self.provenance_factory = provenance_factory
        self.transaction_id_factory = transaction_id_factory
        self.timestamp_factory = timestamp_factory
        self._transaction_ids: set[str] = set()
        self._timestamps: set[str] = set()
        self._last_epoch = 0
        self._pending_selection_callback: Any | None = None
        self._poisoned = False

    @property
    def identity_map(self) -> Mapping[str, Any]:
        parsed = codec.parse_canonical_json(self._identity_bytes, max_bytes=16384)
        return _freeze(parsed)

    @property
    def poisoned(self) -> bool:
        return self._poisoned or self.ledger.poisoned

    def _fail(self, message: str, exc: BaseException | None = None):
        self._poisoned = True
        if exc is None:
            raise SelectionSinkError(message)
        raise SelectionSinkError(message) from exc

    def _ensure_live(self) -> None:
        if self.poisoned:
            raise SelectionSinkError("selection sink is terminal after failure; never retry")

    def _identifiers(self, epoch: int) -> tuple[str, str]:
        try:
            transaction_id = self.transaction_id_factory(epoch)
            timestamp = self.timestamp_factory(epoch)
            if type(transaction_id) is not str or _TX_ID.fullmatch(transaction_id) is None:
                raise ValueError("invalid transaction ID")
            if transaction_id in self._transaction_ids:
                raise ValueError("duplicate transaction ID")
            if type(timestamp) is not str:
                raise ValueError("timestamp must be an ISO-8601 string")
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
                raise ValueError("timestamp must have UTC offset")
            timestamp = parsed.astimezone(timezone.utc).isoformat()
            timestamp_identity = timestamp
            if timestamp_identity in self._timestamps:
                raise ValueError("duplicate transaction timestamp")
            self._transaction_ids.add(transaction_id)
            self._timestamps.add(timestamp_identity)
            return transaction_id, timestamp
        except BaseException as exc:
            self._fail("could not reserve unique transaction identity", exc)

    def _provenance(self, evidence: Any, epoch: int) -> Mapping[str, Any]:
        try:
            provenance = self.provenance_factory(evidence)
            if not isinstance(provenance, Mapping):
                raise ValueError("provenance factory must return a mapping")
            provenance = dict(provenance)
            context = self.ledger.context
            if (provenance.get("run_id") != context.run_id
                    or provenance.get("attempt_id") != context.attempt_id
                    or provenance.get("admission_sha256") != context.run_admission_sha256):
                raise ValueError("provenance does not match ledger run/attempt/admission")
            if self.identity_map.get("admission_sha256") != context.run_admission_sha256:
                raise ValueError("identity map admission differs from ledger context")
            if self.identity_map.get("schema_sha256") != context.schema_sha256:
                raise ValueError("identity map schema differs from ledger context")
            counters = provenance.get("counters")
            if not isinstance(counters, Mapping):
                raise ValueError("provenance counters are required")
            expected = {"complete_epochs": epoch, "validation_epochs": epoch,
                        "attempted_updates": epoch * 2559,
                        "completed_updates": epoch * 2559,
                        "selected_load_attempts": 0,
                        "test_payload_read_attempts": 0}
            if dict(counters) != expected:
                raise ValueError("provenance counters differ from complete-epoch totals")
            stable = {key: value for key, value in provenance.items()
                      if key not in {"created_at_utc", "counters"}}
            stable_bytes = codec._canonical(stable)
            if self._stable_provenance_bytes is None:
                self._stable_provenance_bytes = stable_bytes
            elif stable_bytes != self._stable_provenance_bytes:
                raise ValueError("stable source/runtime/initialization provenance changed across epochs")
            return provenance
        except BaseException as exc:
            self._fail("epoch provenance is malformed or not ledger-bound", exc)

    def _check_predecessor(self, epoch: int) -> None:
        """Bind the in-memory GR state to the verified durable predecessor."""
        try:
            head = self.ledger.verify_committed_state()
            if epoch == 1:
                if head is not None or self.selection.selected is not None or self.selection.best_f1 is not None:
                    raise ValueError("first epoch requires the empty selection predecessor")
                return
            if head is None or head["sequence"] != epoch - 1 or head["finalized"]:
                raise ValueError("durable selection predecessor is not the prior complete epoch")
            pointer, _ = self.ledger._read_reference(
                head["selected_pointer"], "immutable_selected_pointer")
            reference = {
                "epoch": self.selection.selected.epoch,
                "checkpoint_sha256": self.selection.selected.checkpoint_sha256,
                "recurrent_sha256": self.selection.selected.recurrent_sha256,
                "link_head_sha256": self.selection.selected.link_head_sha256,
            }
            if (pointer["selected"] != reference
                    or pointer["best_f1"] != self.selection.best_f1
                    or pointer["last_epoch"] != self.selection.last_epoch):
                raise ValueError("in-memory GR selection differs from durable predecessor pointer")
        except BaseException as exc:
            self._fail("selection predecessor is not safely reconciled", exc)

    def on_epoch_complete(self, evidence: Any) -> None:
        """Publish and durably apply one complete epoch, including retained ties."""
        self._ensure_live()
        record = getattr(evidence, "record", None)
        epoch = getattr(record, "epoch", None)
        if type(epoch) is not int or epoch != self._last_epoch + 1:
            self._fail("complete epochs must arrive once in strict sequence")
        if self._pending_selection_callback is not None:
            self._fail("prior selected epoch callback was not observed")
        if type(getattr(record, "validation_selected", None)) is not bool:
            self._fail("epoch record must contain an exact boolean selection decision")
        score = getattr(record, "validation_mean_f1", None)
        if (type(score) not in (int, float) or not math.isfinite(score)
                or not 0 <= score <= 1):
            self._fail("epoch record validation mean F1 must be finite and in [0,1]")

        # Compare before any candidate publication or durable decision.
        current_best = self.selection.best_f1
        expected_selected = current_best is None or score > current_best
        if record.validation_selected is not expected_selected:
            self._fail("fit record selection flag differs from strict-improvement rule")
        if self.selection.last_epoch != epoch - 1:
            self._fail("GR selection chronology differs from candidate epoch")
        self._check_predecessor(epoch)

        transaction_id, timestamp = self._identifiers(epoch)
        provenance = self._provenance(evidence, epoch)
        try:
            files = encode_epoch_candidate(
                evidence, provenance=provenance,
                manifest={"identities": dict(self.identity_map)})
            relative = f"candidates/epoch-{epoch:03d}-{transaction_id}"
            codec.write_candidate(self.ledger.root / relative, files)
            verified = codec.read_candidate(self.ledger.root / relative)
            manifest = verified["manifest"]
            validation = verified["validation"]
            if (manifest["epoch"] != epoch
                    or validation["time_mean_f1"] != score):
                raise SelectionSinkError("read-back candidate epoch or score differs from fit evidence")
            reference = gr.CheckpointReference(
                epoch=epoch,
                checkpoint_sha256=verified["files_sha256"]["manifest.json"],
                recurrent_sha256=verified["files_sha256"]["recurrent.f32le"],
                link_head_sha256=verified["files_sha256"]["link-head.f32le"],
            )

            def apply_selection(transaction: dict[str, Any]):
                if (transaction.get("candidate") != {
                        "epoch": reference.epoch,
                        "checkpoint_sha256": reference.checkpoint_sha256,
                        "recurrent_sha256": reference.recurrent_sha256,
                        "link_head_sha256": reference.link_head_sha256}
                        or transaction.get("candidate_f1") != score):
                    raise SelectionSinkError("ledger transaction candidate differs from verified read-back")
                selected_now = self.selection.consider(
                    reference, score,
                    split="validation", epoch_complete=True)
                if selected_now is not record.validation_selected:
                    raise SelectionSinkError("GR selection application differs from prechecked fit decision")
                return self.selection.selected

            def verify_selection(applied: Any, transaction: dict[str, Any]) -> bool:
                chosen = transaction["selected"]
                expected = gr.CheckpointReference(**chosen)
                return (applied == expected
                        and self.selection.selected == expected
                        and self.selection.best_f1 == transaction["best_f1"]
                        and self.selection.last_epoch == epoch
                        and self.selection.finalized is False
                        and transaction["last_epoch"] == epoch
                        and transaction["decision"] == ("selected" if record.validation_selected else "retained"))

            result = self.ledger.commit_candidate(
                relative, transaction_id=transaction_id,
                created_at_utc=timestamp,
                apply_selection=apply_selection,
                verify_selection=verify_selection)
            selected_ref = gr.CheckpointReference(**result.selected)
            if (result.sequence != epoch
                    or result.decision != ("selected" if record.validation_selected else "retained")
                    or self.selection.selected != selected_ref
                    or self.selection.best_f1 != result.best_f1
                    or self.selection.last_epoch != epoch):
                raise SelectionSinkError("durable ledger result and applied GR selection disagree")
            self._last_epoch = epoch
            self._pending_selection_callback = record if record.validation_selected else None
        except BaseException as exc:
            self._fail("candidate publication or selection transaction failed; run is terminal", exc)

    def on_selection(self, record: Any) -> None:
        """Fit callback barrier; selection was already applied inside the ledger."""
        self._ensure_live()
        if (self._pending_selection_callback is None
                or record is not self._pending_selection_callback):
            self._fail("unexpected or duplicate fit selection callback")
        if (self.selection.selected is None
                or self.selection.selected.epoch != record.epoch
                or self.selection.last_epoch != record.epoch):
            self._fail("fit selection callback does not match the committed GR selection")
        self._pending_selection_callback = None

