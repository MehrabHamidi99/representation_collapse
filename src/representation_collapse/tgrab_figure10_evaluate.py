"""Single-use selected-checkpoint load, prefix replay gate, and held-out scoring.

The caller supplies independently admitted context, final head and backend. This
module verifies consistency; it cannot authenticate those external authorities.
It never restores optimizer, RNG or saved terminal state into inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import graph_reconstruction as gr
from . import tgrab_selected_checkpoint as codec
from .tgrab_figure10 import test_reduction
from .tgrab_figure10_candidate_encoder import _component_bytes, _cpu_f32_bytes
from .tgrab_selection_transaction import SelectionContext, SelectionLedger


class EvaluationError(RuntimeError):
    """Terminal evaluation failure; an existing output directory cannot resume."""


def load_native_weights(backend: Any, verified: dict) -> None:
    """Strictly load both FP32 CPU components and verify their exact bytes."""
    _, catalog = codec._schema_and_catalog()
    if backend.device != "cpu" or set(backend.models) != {"node_emb", "link_pred"}:
        raise EvaluationError("CPU backend with exactly both native models required")
    pointers = set()
    # Validate both components before mutating either component.
    for component, spec in catalog["components"].items():
        model = backend.models[spec["backend_key"]]
        current = model.state_dict()
        # Also verifies the complete native names/order/shape/dtype before load.
        _component_bytes(component, current, catalog)
        for value in current.values():
            storage = value.untyped_storage()
            pointer = storage.data_ptr()
            if pointer in pointers or value.storage_offset() != 0 or storage.nbytes() != value.numel() * 4:
                raise EvaluationError("aliased/overlapping/noncompact parameter storage")
            pointers.add(pointer)
    for component, spec in catalog["components"].items():
        model = backend.models[spec["backend_key"]]
        descriptors = {d["name"]: d for d in spec["tensors"]}
        values = {}
        payload = verified[component]
        for name in spec["native_state_dict_order"]:
            d = descriptors[name]
            raw = payload[d["payload_offset"]:d["payload_offset"] + d["nbytes"]]
            array = np.frombuffer(raw, dtype="<f4").reshape(d["shape"]).copy()
            values[name] = backend.torch.from_numpy(array)
        result = model.load_state_dict(values, strict=True)
        if getattr(result, "missing_keys", ()) or getattr(result, "unexpected_keys", ()):
            raise EvaluationError("strict native load returned incompatible keys")
        actual, _ = _component_bytes(component, model.state_dict(), catalog)
        if actual != payload:
            raise EvaluationError("native weights differ after strict load")
        model.eval()
        if any(module.training is not False for module in model.modules()):
            raise EvaluationError("native module did not enter evaluation mode")


def verify_terminal(state: Any, verified: dict) -> None:
    """Compare every H/C bit and complete state identity at s[2816]."""
    if state.next_target_t != 2816 or state.node_ids != tuple(range(100)):
        raise EvaluationError("replay terminal timestamp/node ordering mismatch")
    payload = b"".join(_cpu_f32_bytes(getattr(state, name), name=name, shape=(1, 100, 128))
                       for name in ("hidden", "cell"))
    if payload != verified["terminal_state"]:
        raise EvaluationError("node2b complete H/C replay mismatch")


class _Events:
    def __init__(self, root: Path, identity: dict):
        self.root = root
        # O_EXCL directory creation is the durable single-use charge boundary.
        os.mkdir(root, 0o700)
        self.fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        parent = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        self.sequence = 0
        self.identity = dict(identity)
        self.previous_sha256 = None

    def write(self, event: str, **fields):
        raw = (json.dumps({"kind": "figure10_evaluation_diagnostic_event_v1",
                          **self.identity, "event": event, "sequence": self.sequence,
                          "previous_event_sha256": self.previous_sha256, **fields}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
        name = f"{self.sequence:03d}-{event}.json"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(self.fd)
            self.previous_sha256 = hashlib.sha256(raw).hexdigest()
        finally:
            self.sequence += 1

    def close(self):
        os.close(self.fd)


def run_selected_evaluation(*, ledger: SelectionLedger, expected_context: SelectionContext,
                            expected_final_head_sha256: str, backend: Any,
                            train, validation, test_opener: Callable,
                            output_dir: Path, report: Callable,
                            guard: Callable = lambda: None,
                            observe_state: Callable | None = None) -> dict:
    """Consume one finalized selection and one lazy graph-data test capability.

    ``report`` must durably log reporting identity before this call returns from
    its first invocation. No data/numerical work occurs before that callback.
    Pass ``store.sealed_test_opener(selection)`` from graph_data as test_opener;
    constructing that capability does not read test bytes. Existing output_dir
    means the attempt is spent, including a previous crash or failed load.
    """
    report({"event": "evaluation_reporting_initialized"})
    events = _Events(Path(output_dir), {
        "run_id": expected_context.run_id, "attempt_id": expected_context.attempt_id,
        "admission_sha256": expected_context.run_admission_sha256,
        "final_head_sha256": expected_final_head_sha256})
    try:
        events.write("charged", selected_load_attempts=1, test_payload_read_attempts=0,
                     expected_final_head_sha256=expected_final_head_sha256)
        guard()
        if (type(ledger) is not SelectionLedger or type(expected_context) is not SelectionContext
                or ledger.context != expected_context
                or expected_context.expected_head_sha256 != expected_final_head_sha256
                or len(expected_final_head_sha256) != 64):
            raise EvaluationError("trusted final head/context mismatch")
        head = ledger.verify_committed_state()
        if not head or head["finalized"] is not True:
            raise EvaluationError("finalized selection required")
        pointer, _ = ledger._read_reference(head["selected_pointer"], "immutable_selected_pointer")
        relative = pointer["manifest_relative_path"]
        if not relative.endswith("/manifest.json"):
            raise EvaluationError("selected manifest path required")
        bundle_relative = relative[:-len("/manifest.json")]
        candidate, _, _, _ = ledger._candidate(bundle_relative)
        if candidate != pointer["selected"]:
            raise EvaluationError("selected candidate differs from finalized pointer")
        verified = codec.read_candidate(ledger.root / bundle_relative)
        if verified["files_sha256"]["manifest.json"] != candidate["checkpoint_sha256"]:
            raise EvaluationError("candidate changed after selection verification")
        load_native_weights(backend, verified)
        events.write("loaded", checkpoint_sha256=candidate["checkpoint_sha256"])
        # Materialize only the already-admitted train/validation collections.
        prefix = tuple(train) + tuple(validation)
        if len(prefix) != 2816 or [s.timestamp for s in prefix] != list(range(2816)):
            raise EvaluationError("exact chronological train/validation prefix required")
        with backend.evaluation_mode():
            state = backend.reset()
            if state.next_target_t != 0 or state.node_ids != tuple(range(100)):
                raise EvaluationError("fresh zero-state identity mismatch")
            for name in ("hidden", "cell"):
                if _cpu_f32_bytes(getattr(state, name), name=name, shape=(1, 100, 128)) != bytes(51200):
                    raise EvaluationError("prefix replay must start at fresh positive-zero H/C")
            for snapshot in prefix:
                guard()
                state = backend.detach(backend.consume(state, snapshot))
            verify_terminal(state, verified)
            events.write("replay_verified", next_target_t=2816,
                         terminal_sha256=hashlib.sha256(verified["terminal_state"]).hexdigest())
            guard()
            # Durable read charge precedes even calling the lazy opener.
            events.write("test_read", test_payload_read_attempts=1)
            iterator = iter(test_opener())
            rows = []
            previous = prefix[-1]
            for t in range(2816, 3072):
                guard()
                target = next(iterator)
                if target.timestamp != t or state.next_target_t != t:
                    raise EvaluationError("held-out score-before-consume chronology mismatch")
                if observe_state is not None:
                    # Observers receive independent data, never writable live
                    # recurrent storage. Cloning uses no RNG or model calls.
                    observed = gr.CompleteState(state.hidden.detach().clone(), state.cell.detach().clone(),
                                                state.next_target_t, state.node_ids)
                    observe_state(observed, t)
                probabilities = backend.score_pairs(state, gr.ordered_offdiagonal_pairs(100))
                row = gr.snapshot_metrics(target, probabilities, previous)
                rows.append(row)
                state = backend.detach(backend.consume(state, target))
                previous = target
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise EvaluationError("extra held-out target")
        reduction = test_reduction(rows)
        result = {"rows": rows, "reduction": reduction,
                  "checkpoint_sha256": candidate["checkpoint_sha256"],
                  "final_head_sha256": expected_final_head_sha256,
                  "claim": "behavioral reconstruction only; no dimension-collapse inference"}
        report({"event": "evaluation_complete", **reduction})
        events.write("complete", **result)
        return result
    except BaseException as exc:
        events.write("failed", error_type=type(exc).__name__, error=str(exc)[:1000])
        raise
    finally:
        events.close()
