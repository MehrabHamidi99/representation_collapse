"""Pure-data validation-selection history and durable transaction records.

This layer checks record consistency and filesystem publication only. It never
creates trust: callers must supply run/admission and prior-head identities from
an independently trusted root admission. Synthetic tests use synthetic context
values and establish no authorization, selection authority, or model result.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable

from . import tgrab_selected_checkpoint as codec


SCHEMA_SHA256 = codec.SCHEMA_SHA256
_HEX = re.compile(r"^[0-9a-f]{64}$")
_RUNID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_MAX_OBJECTS = {"receipt": 16384, "intent": 16384, "failure": 16384, "pointer": 4096, "head": 4096}


class SelectionError(RuntimeError):
    """Invalid history, unsafe path, untrusted predecessor, or failed commit."""


@dataclass(frozen=True)
class SelectionContext:
    """Externally supplied identity pins; values are never self-authenticating.

    `expected_head_sha256=None` means the caller independently established a
    new run has no committed selection head. On process restart, the caller
    must provide the externally recorded current head SHA. The module verifies
    that exact state but cannot verify who supplied these values.
    """

    run_id: str
    attempt_id: str
    run_admission_sha256: str
    schema_sha256: str
    expected_head_sha256: str | None


@dataclass(frozen=True)
class CommitResult:
    sequence: int
    decision: str
    selected: dict[str, Any]
    best_f1: float
    head_sha256: str
    receipt_reference: dict[str, Any]
    pointer_reference: dict[str, Any]
    finalized: bool


def _canonical(value: Any) -> bytes:
    return codec._canonical(value)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(value: str | None, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise SelectionError(f"{name} must be lowercase SHA-256")


def _safe_rel(value: str) -> PurePosixPath:
    if type(value) is not str:
        raise SelectionError("relative path must be a string")
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise SelectionError("unsafe relative path")
    if str(p) != value or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in p.parts):
        raise SelectionError("relative path does not match frozen path alphabet")
    if len(value) > 256:
        raise SelectionError("relative path too long")
    return p


def _role_for_path(relative: str) -> str:
    if relative.startswith("selection/receipts/"):
        return "immutable_selection_receipt"
    if relative.startswith("selection/pointers/"):
        return "immutable_selected_pointer"
    if relative.startswith("selection/intents/"):
        return "immutable_transaction_intent"
    if relative.startswith("selection/failures/"):
        return "immutable_transaction_failure"
    raise SelectionError("immutable record path is outside the fixed selection namespace")


def _validate_record(value: dict[str, Any], expected_kind: str) -> bytes:
    raw = _canonical(value)
    try:
        parsed = codec.validate_metadata(raw, expected_kind=expected_kind, max_bytes=_MAX_OBJECTS[
            {"tgrab_validation_selection_decision_v1": "receipt",
             "tgrab_validation_selection_finalized_v1": "receipt",
             "tgrab_selected_pointer_v1": "pointer",
             "tgrab_selection_head_v1": "head",
             "tgrab_selection_transaction_intent_v1": "intent",
             "tgrab_selection_transaction_failure_v1": "failure"}[expected_kind]])
    except Exception as exc:
        raise SelectionError(f"invalid {expected_kind} record: {exc}") from exc
    if parsed != value:
        raise SelectionError("record roundtrip changed representation")
    return raw


def decide(*, sequence: int, candidate: dict[str, Any], candidate_f1: float,
           candidate_manifest_relative_path: str, candidate_validation_sha256: str,
           previous_receipt: dict[str, Any] | None,
           previous_selected: dict[str, Any] | None,
           previous_best_f1: float | None) -> dict[str, Any]:
    """Return the frozen strict-improvement decision without claiming trust."""
    if type(sequence) is not int or not 1 <= sequence <= 200:
        raise SelectionError("decision sequence must be an exact integer in 1..200")
    if type(candidate_f1) not in (int, float) or not 0 <= candidate_f1 <= 1:
        raise SelectionError("candidate F1 must be finite and in [0,1]")
    if not isinstance(candidate, dict):
        raise SelectionError("candidate CheckpointReference required")
    _safe_rel(candidate_manifest_relative_path)
    _digest(candidate_validation_sha256, "candidate validation SHA")
    if sequence == 1:
        if previous_receipt is not None or previous_selected is not None or previous_best_f1 is not None:
            raise SelectionError("first decision must have null predecessor state")
    elif previous_receipt is None or previous_selected is None or previous_best_f1 is None:
        raise SelectionError("later decision requires complete predecessor state")
    if sequence > 1 and (type(previous_best_f1) not in (int, float) or not 0 <= previous_best_f1 <= 1):
        raise SelectionError("previous best F1 outside [0,1]")
    improve = sequence == 1 or candidate_f1 > previous_best_f1  # strict: ties retain earliest
    if (set(candidate) != {"epoch", "checkpoint_sha256", "recurrent_sha256", "link_head_sha256"} or
            candidate["epoch"] != sequence or any(type(candidate[k]) is not str or _HEX.fullmatch(candidate[k]) is None
            for k in ("checkpoint_sha256", "recurrent_sha256", "link_head_sha256"))):
        raise SelectionError("candidate CheckpointReference differs from frozen schema/epoch")
    selected = candidate if improve else previous_selected
    best = candidate_f1 if improve else previous_best_f1
    return {
        "kind": "tgrab_validation_selection_decision_v1", "schema_version": 1,
        "sequence": sequence, "epoch": sequence, "last_epoch": sequence,
        "candidate": candidate, "candidate_f1": candidate_f1,
        "candidate_manifest_relative_path": candidate_manifest_relative_path,
        "candidate_validation_sha256": candidate_validation_sha256,
        "previous_receipt": previous_receipt, "previous_selected": previous_selected,
        "previous_best_f1": previous_best_f1,
        "decision": "selected" if improve else "retained", "selected": selected,
        "best_f1": best, "finalized": False, "test_authorized_by_this_receipt": False,
    }


def _ensure_common(record: dict[str, Any], context: SelectionContext, transaction_id: str,
                   timestamp: str) -> dict[str, Any]:
    return {**record, "run_id": context.run_id, "attempt_id": context.attempt_id,
            "created_at_utc": timestamp, "schema_sha256": context.schema_sha256,
            "run_admission_sha256": context.run_admission_sha256,
            "transaction_id": transaction_id}


def _verify_context(context: SelectionContext) -> None:
    if not isinstance(context, SelectionContext) or not _RUNID.fullmatch(context.run_id) or not _RUNID.fullmatch(context.attempt_id):
        raise SelectionError("explicit selection context with valid run/attempt IDs required")
    _digest(context.run_admission_sha256, "run admission SHA")
    _digest(context.schema_sha256, "schema SHA")
    _digest(context.expected_head_sha256, "expected prior head SHA", nullable=True)
    if context.schema_sha256 != SCHEMA_SHA256:
        raise SelectionError("selection context schema SHA differs from frozen v1")


def _require_intent_receipt_match(intent: dict[str, Any], receipt: dict[str, Any], sequence: int) -> None:
    """Bind each S08 intent to the exact immutable record it advances."""
    if (intent.get("transaction_id") != receipt.get("transaction_id") or
            intent.get("sequence") != sequence or receipt.get("sequence") != sequence or
            intent.get("created_at_utc") != receipt.get("created_at_utc")):
        raise SelectionError("transaction intent ID/sequence/timestamp differs from receipt")


def _open_dir(path: Path) -> int:
    return codec._open_bundle_directory(path)


class SelectionLedger:
    """Single-owner, create-only S07/S08 transaction ledger.

    Root admission and predecessor SHA are mandatory caller inputs. This class
    verifies record and byte consistency; it is not an authority verifier.
    `failure_hook` exists only for deterministic failure-injection tests. Never
    provide it in a production runner.
    """

    def __init__(self, root: str | os.PathLike[str], context: SelectionContext,
                 *, failure_hook: Callable[[str], None] | None = None):
        _verify_context(context)
        self.root = Path(root)
        if not self.root.is_absolute():
            self.root = Path.cwd() / self.root
        self.root_fd = _open_dir(self.root)
        self.context = context
        self.failure_hook = failure_hook
        self.poisoned = False
        self._lock_fd = None
        try:
            self._create_layout()
            self._lock_fd = self._open_lock()
            lock_st = os.fstat(self._lock_fd)
            if not stat.S_ISREG(lock_st.st_mode) or lock_st.st_nlink != 1:
                raise SelectionError("selection lock must be a regular single-link file")
            import fcntl
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError, OSError) as exc:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            os.close(self.root_fd)
            raise SelectionError("selection ownership lock unavailable") from exc
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            os.close(self.root_fd)
            raise
        try:
            self.verify_committed_state()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_lock_fd", None) is not None:
            try:
                import fcntl
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        if getattr(self, "root_fd", None) is not None:
            os.close(self.root_fd)
            self.root_fd = None

    def __enter__(self) -> "SelectionLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_layout(self) -> None:
        for dirname in ("selection", "selection/receipts", "selection/pointers", "selection/intents", "selection/failures"):
            path = PurePosixPath(dirname)
            parent = self.root_fd
            owned = []
            try:
                for part in path.parts:
                    try:
                        os.mkdir(part, 0o700, dir_fd=parent)
                    except FileExistsError:
                        pass
                    child = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                    getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
                    owned.append(child)
                    parent = child
            except OSError as exc:
                raise SelectionError("unsafe selection metadata directory") from exc
            finally:
                for fd in reversed(owned):
                    os.close(fd)

    def _open_lock(self) -> int:
        select_fd = os.open("selection", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                            getattr(os, "O_NOFOLLOW", 0), dir_fd=self.root_fd)
        try:
            return os.open("selection.lock", os.O_RDWR | os.O_CREAT |
                           getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=select_fd)
        finally:
            os.close(select_fd)

    def _hook(self, stage: str, when: str) -> None:
        if self.failure_hook is not None:
            self.failure_hook(f"{stage}:{when}")

    def _write_immutable(self, relative: str, data: bytes, limit: int) -> dict[str, Any]:
        p = _safe_rel(relative)
        parent = self.root_fd
        opened = []
        try:
            for part in p.parts[:-1]:
                parent = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                 getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
                opened.append(parent)
            fd = os.open(p.parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent)
            try:
                if len(data) > limit:
                    raise SelectionError("immutable record exceeds schema bound")
                view = memoryview(data)
                while view:
                    n = os.write(fd, view)
                    if n <= 0:
                        raise SelectionError("short immutable record write")
                    view = view[n:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(parent)
            actual = self._read_relative(relative, limit)
            if actual != data:
                raise SelectionError("immutable record readback differs")
            return {"role": _role_for_path(relative), "relative_path": relative,
                    "bytes": len(data), "sha256": _sha(data)}
        finally:
            for fd in reversed(opened):
                os.close(fd)

    def _read_relative(self, relative: str, limit: int) -> bytes:
        p = _safe_rel(relative)
        parent = self.root_fd
        opened = []
        try:
            for part in p.parts[:-1]:
                parent = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                 getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
                opened.append(parent)
            return codec._read_stable_at(parent, p.parts[-1], limit)
        finally:
            for fd in reversed(opened):
                os.close(fd)

    def _read_reference(self, ref: dict[str, Any], role: str) -> tuple[dict[str, Any], bytes]:
        if not isinstance(ref, dict) or ref.get("role") != role:
            raise SelectionError("reference role mismatch")
        limit = 4096 if role == "immutable_selected_pointer" else 16384
        data = self._read_relative(ref["relative_path"], limit)
        if len(data) != ref["bytes"] or _sha(data) != ref["sha256"]:
            raise SelectionError("immutable reference size/SHA mismatch")
        kind = {"immutable_selection_receipt": None,
                "immutable_selected_pointer": "tgrab_selected_pointer_v1",
                "immutable_transaction_intent": "tgrab_selection_transaction_intent_v1"}[role]
        if kind is None:
            value = self._parse_decision_or_finalization(data)
        else:
            try:
                value = codec.validate_metadata(data, expected_kind=kind, max_bytes=limit)
            except Exception as exc:
                raise SelectionError("referenced immutable record fails frozen schema") from exc
        if value["run_id"] != self.context.run_id or value["attempt_id"] != self.context.attempt_id:
            raise SelectionError("record run/attempt identity differs from external context")
        if value["schema_sha256"] != self.context.schema_sha256 or value["run_admission_sha256"] != self.context.run_admission_sha256:
            raise SelectionError("record schema/admission identity differs from external context")
        return value, data

    def _parse_decision_or_finalization(self, data: bytes) -> dict[str, Any]:
        value = codec.parse_canonical_json(data, max_bytes=16384)
        kind = value.get("kind") if isinstance(value, dict) else None
        if kind not in ("tgrab_validation_selection_decision_v1", "tgrab_validation_selection_finalized_v1"):
            raise SelectionError("receipt has unsupported record kind")
        try:
            parsed = codec.validate_metadata(data, expected_kind=kind, max_bytes=16384)
        except Exception as exc:
            raise SelectionError("receipt fails frozen closed schema") from exc
        return parsed

    def _atomic_replace(self, filename: str, data: bytes, transaction_id: str, stage: str) -> None:
        tmp = f".{filename}.{transaction_id}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=self.root_fd)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise SelectionError("short temporary pointer/head write")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self._hook(stage, "before")
        os.replace(tmp, filename, src_dir_fd=self.root_fd, dst_dir_fd=self.root_fd)
        self._hook(stage, "after")
        if stage == "head_fsync":
            self._hook(stage, "before")
            os.fsync(self.root_fd)
            self._hook(stage, "after")
        else:
            os.fsync(self.root_fd)

    def _head_bytes(self) -> bytes | None:
        try:
            return self._read_relative("selection-head.json", 4096)
        except FileNotFoundError:
            return None

    def verify_committed_state(self) -> dict[str, Any] | None:
        """Reopen and verify the externally pinned head and its full S07/S08 chain."""
        if self._listdir("selection/failures"):
            raise SelectionError("terminal transaction failure exists; independent reconciliation required")
        root_entries = set(os.listdir(self.root_fd))
        if any(name.startswith((".selected.json.", ".selection-head.")) and name.endswith(".tmp") for name in root_entries):
            raise SelectionError("orphaned atomic replacement temporary exists")
        raw = self._head_bytes()
        expected = self.context.expected_head_sha256
        if raw is None:
            if expected is not None:
                raise SelectionError("externally expected committed head is missing")
            try:
                os.stat("selected.json", dir_fd=self.root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise SelectionError("selected alias exists without a committed head; stale/orphan alias")
            if any(self._listdir(f"selection/{d}") for d in ("receipts", "pointers", "intents", "failures")):
                raise SelectionError("orphaned transaction objects exist without a head; recovery is required")
            return None
        if expected is None or _sha(raw) != expected:
            raise SelectionError("selection head differs from externally supplied expected SHA")
        head = codec.validate_metadata(raw, expected_kind="tgrab_selection_head_v1", max_bytes=4096)
        self._check_common(head)
        self._verify_history(head)
        return head

    def _listdir(self, rel: str) -> list[str]:
        p = _safe_rel(rel)
        parent = self.root_fd
        opened = []
        try:
            for part in p.parts:
                parent = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                                 getattr(os, "O_NOFOLLOW", 0), dir_fd=parent)
                opened.append(parent)
            return os.listdir(parent)
        finally:
            for fd in reversed(opened):
                os.close(fd)

    def _check_common(self, record: dict[str, Any]) -> None:
        if record["run_id"] != self.context.run_id or record["attempt_id"] != self.context.attempt_id:
            raise SelectionError("run/attempt mismatch")
        if record["schema_sha256"] != self.context.schema_sha256 or record["run_admission_sha256"] != self.context.run_admission_sha256:
            raise SelectionError("schema/admission mismatch")

    def _verify_history(self, head: dict[str, Any]) -> None:
        seq = head["sequence"]
        head_receipt, _ = self._read_reference(head["decision_receipt"], "immutable_selection_receipt")
        if head_receipt["sequence"] != seq:
            raise SelectionError("head and current receipt sequence differ")
        final = head_receipt if head["finalized"] else None
        if head["finalized"]:
            if final["kind"] != "tgrab_validation_selection_finalized_v1":
                raise SelectionError("finalized head must reference a finalization receipt")
            current_ref = final["previous_receipt"]
            current, _ = self._read_reference(current_ref, "immutable_selection_receipt")
            decision_count = seq - 1
        else:
            current_ref, current = head["decision_receipt"], head_receipt
            decision_count = seq
        decisions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        while current["kind"] == "tgrab_validation_selection_decision_v1":
            decisions.append((current, current_ref))
            if current["sequence"] == 1:
                if current["previous_receipt"] is not None:
                    raise SelectionError("first decision has previous receipt")
                break
            current_ref = current["previous_receipt"]
            current, _ = self._read_reference(current_ref, "immutable_selection_receipt")
        if current["kind"] != "tgrab_validation_selection_decision_v1":
            raise SelectionError("decision chain is interrupted by unsupported record")
        decisions.reverse()
        if len(decisions) != decision_count or not 1 <= decision_count <= 200:
            raise SelectionError("decision chain length differs from committed sequence")

        expected_receipts: set[str] = set()
        expected_intents: set[str] = set()
        expected_pointers: set[str] = set()
        prior_selected = prior_best = prior_receipt_ref = prior_pointer_ref = prior_pointer_value = None
        prior_head_sha = None
        for index, (item, item_ref) in enumerate(decisions, start=1):
            if item["sequence"] != index or item["epoch"] != index or item["last_epoch"] != index:
                raise SelectionError("decision chain contains a sequence gap")
            improve = index == 1 or item["candidate_f1"] > prior_best
            expected_decision = "selected" if improve else "retained"
            expected_selected = item["candidate"] if improve else prior_selected
            expected_best = item["candidate_f1"] if improve else prior_best
            if (item["decision"], item["selected"], item["best_f1"]) != (expected_decision, expected_selected, expected_best):
                raise SelectionError("decision violates strict validation-F1 improvement/earliest tie rule")
            if (item["previous_receipt"], item["previous_selected"], item["previous_best_f1"]) != (prior_receipt_ref, prior_selected, prior_best):
                raise SelectionError("decision predecessor fields differ from the immutable chain")
            self._check_common(item)
            manifest_rel = item["candidate_manifest_relative_path"]
            if not manifest_rel.endswith("/manifest.json"):
                raise SelectionError("candidate manifest reference must end in fixed manifest basename")
            _safe_rel(manifest_rel)
            if item["candidate"]["epoch"] != index or _HEX.fullmatch(item["candidate_validation_sha256"]) is None:
                raise SelectionError("candidate reference epoch or validation digest invalid")
            receipt_rel = _safe_rel(item_ref["relative_path"])
            expected_receipts.add(str(receipt_rel))
            txid = item["transaction_id"]
            intent_rel = f"selection/intents/intent-{index:03d}-{txid}.json"
            intent_bytes = self._read_relative(intent_rel, 16384)
            try:
                intent_i = codec.validate_metadata(intent_bytes, expected_kind="tgrab_selection_transaction_intent_v1", max_bytes=16384)
            except Exception as exc:
                raise SelectionError("immutable intent fails frozen schema") from exc
            self._check_common(intent_i)
            _require_intent_receipt_match(intent_i, item, index)
            intent_ref = {"role": "immutable_transaction_intent", "relative_path": intent_rel,
                          "bytes": len(intent_bytes), "sha256": _sha(intent_bytes)}
            expected_intents.add(intent_rel)
            if intent_i["next_decision_receipt"] != item_ref or intent_i["operation"] != ("select" if improve else "retain"):
                raise SelectionError("decision and intent disagree")
            if intent_i["pointer_alias_will_change"] is not improve:
                raise SelectionError("intent alias-change flag differs from selection operation")
            if intent_i["previous_head_sha256"] != prior_head_sha:
                raise SelectionError("intent does not bind exact prior committed head")
            expected_previous_pointer = prior_pointer_ref["sha256"] if prior_pointer_ref else None
            if intent_i["previous_pointer_sha256"] != expected_previous_pointer:
                raise SelectionError("intent does not bind exact prior pointer")
            next_pointer_ref = intent_i["next_selected_pointer"]
            if improve:
                pvalue, _ = self._read_reference(next_pointer_ref, "immutable_selected_pointer")
                expected_p = {"selected": item["candidate"], "selection_receipt": item_ref,
                              "best_f1": item["candidate_f1"], "last_epoch": index,
                              "manifest_relative_path": item["candidate_manifest_relative_path"],
                              "stage": "selected"}
                if any(pvalue[k] != v for k, v in expected_p.items()):
                    raise SelectionError("new pointer does not match its improvement decision")
                prior_pointer_ref = next_pointer_ref
                prior_pointer_value = pvalue
                expected_pointers.add(str(_safe_rel(next_pointer_ref["relative_path"])))
            elif next_pointer_ref != prior_pointer_ref:
                raise SelectionError("retained decision changed pointer reference")
            head_i = {"kind": "tgrab_selection_head_v1", "schema_version": 1,
                      "run_id": item["run_id"], "attempt_id": item["attempt_id"],
                      "created_at_utc": item["created_at_utc"], "schema_sha256": item["schema_sha256"],
                      "run_admission_sha256": item["run_admission_sha256"], "sequence": index,
                      "decision_receipt": item_ref, "selected_pointer": next_pointer_ref,
                      "transaction_intent": intent_ref, "finalized": False}
            prior_head_sha = _sha(_validate_record(head_i, "tgrab_selection_head_v1"))
            prior_selected, prior_best, prior_receipt_ref = item["selected"], item["best_f1"], item_ref

        latest_intent_ref = head["transaction_intent"]
        latest_intent, latest_intent_bytes = self._read_reference(latest_intent_ref, "immutable_transaction_intent")
        if latest_intent["sequence"] != seq or head["created_at_utc"] != latest_intent["created_at_utc"]:
            raise SelectionError("head and latest intent identity mismatch")
        _require_intent_receipt_match(latest_intent, head_receipt, seq)

        if final is not None:
            final_ref = head["decision_receipt"]
            expected_receipts.add(str(_safe_rel(final_ref["relative_path"])))
            self._check_common(final)
            if final["sequence"] != decision_count + 1 or final["previous_receipt"] != prior_receipt_ref:
                raise SelectionError("finalization sequence/receipt chain mismatch")
            if final["selected"] != prior_selected or final["best_f1"] != prior_best or final["last_epoch"] != decision_count:
                raise SelectionError("finalization differs from committed selection history")
            if final["created_at_utc"] != latest_intent["created_at_utc"]:
                raise SelectionError("finalization and intent timestamps differ")
            if latest_intent["operation"] != "finalize" or latest_intent["pointer_alias_will_change"] is not True:
                raise SelectionError("finalization intent operation/alias flag mismatch")
            if latest_intent["next_decision_receipt"] != head["decision_receipt"] or latest_intent["next_selected_pointer"] != head["selected_pointer"]:
                raise SelectionError("finalization intent references differ from head")
            if latest_intent["previous_head_sha256"] != prior_head_sha:
                raise SelectionError("finalization intent prior head mismatch")
            if latest_intent["previous_pointer_sha256"] != prior_pointer_ref["sha256"]:
                raise SelectionError("finalization intent prior pointer mismatch")
            pointer_ref = latest_intent["next_selected_pointer"]
            pvalue, pointer_bytes = self._read_reference(pointer_ref, "immutable_selected_pointer")
            if pvalue["stage"] != "finalized" or pvalue["selection_receipt"] != final_ref or pvalue["selected"] != prior_selected or pvalue["best_f1"] != prior_best or pvalue["last_epoch"] != decision_count:
                raise SelectionError("finalized pointer differs from finalization receipt/history")
            if prior_pointer_value is None or pvalue["manifest_relative_path"] != prior_pointer_value["manifest_relative_path"]:
                raise SelectionError("finalized pointer changed the selected manifest path")
            expected_pointers.add(str(_safe_rel(pointer_ref["relative_path"])))
            if pointer_ref != head["selected_pointer"]:
                raise SelectionError("final head pointer reference mismatch")
            expected_intents.add(str(_safe_rel(latest_intent_ref["relative_path"])))
        else:
            if latest_intent_ref["relative_path"] != f"selection/intents/intent-{seq:03d}-{head_receipt['transaction_id']}.json":
                raise SelectionError("latest head intent path differs from current decision")
            if latest_intent_bytes != self._read_relative(latest_intent_ref["relative_path"], 16384):
                raise SelectionError("latest intent bytes differ")
            if prior_pointer_ref != head["selected_pointer"]:
                raise SelectionError("nonfinal head pointer differs from decision chain")
            expected_intents.add(str(_safe_rel(latest_intent_ref["relative_path"])))
            expected_pointers.add(str(_safe_rel(prior_pointer_ref["relative_path"])))

        pointer, pointer_bytes = self._read_reference(head["selected_pointer"], "immutable_selected_pointer")
        if pointer["selected"] != prior_selected or pointer["best_f1"] != prior_best:
            raise SelectionError("current pointer differs from final selected history")
        if pointer["stage"] != ("finalized" if head["finalized"] else "selected"):
            raise SelectionError("pointer stage differs from head finalization state")
        if latest_intent["next_decision_receipt"] != head["decision_receipt"] or latest_intent["next_selected_pointer"] != head["selected_pointer"]:
            raise SelectionError("head and current intent references differ")
        if head["created_at_utc"] != head_receipt["created_at_utc"]:
            raise SelectionError("head and current receipt timestamps differ")
        if self._read_relative("selected.json", 4096) != pointer_bytes:
            raise SelectionError("selected alias differs from immutable pointer")

        actual_receipts = {"selection/receipts/" + name for name in self._listdir("selection/receipts")}
        actual_intents = {"selection/intents/" + name for name in self._listdir("selection/intents")}
        actual_pointers = {"selection/pointers/" + name for name in self._listdir("selection/pointers")}
        if (actual_receipts, actual_intents, actual_pointers) != (expected_receipts, expected_intents, expected_pointers):
            raise SelectionError("uncommitted/orphaned immutable transaction objects exist")

    def _candidate(self, relative_bundle: str) -> tuple[dict[str, Any], str, float, str]:
        rel = _safe_rel(relative_bundle)
        full = self.root.joinpath(*rel.parts)
        try:
            verified = codec.read_candidate(full)
        except Exception as exc:
            raise SelectionError(f"candidate bundle failed data-only verification: {exc}") from exc
        manifest = verified["manifest"]
        validation = verified["validation"]
        provenance = verified["provenance"]
        if (provenance["run_id"] != self.context.run_id or provenance["attempt_id"] != self.context.attempt_id or
                provenance["admission_sha256"] != self.context.run_admission_sha256 or
                manifest["identities"]["schema_sha256"] != self.context.schema_sha256):
            raise SelectionError("candidate provenance differs from externally supplied run context")
        ref = {"epoch": manifest["epoch"], "checkpoint_sha256": verified["files_sha256"]["manifest.json"],
               "recurrent_sha256": verified["files_sha256"]["recurrent.f32le"],
               "link_head_sha256": verified["files_sha256"]["link-head.f32le"]}
        if manifest["artifact_role"] != "selected_inference_candidate":
            raise SelectionError("candidate bundle artifact role mismatch")
        return ref, f"{relative_bundle}/manifest.json", validation["time_mean_f1"], verified["files_sha256"]["validation.json"]

    def commit_candidate(self, relative_bundle: str, *, transaction_id: str,
                         apply_selection: Callable[[dict[str, Any]], Any],
                         verify_selection: Callable[[Any, dict[str, Any]], bool],
                         created_at_utc: str | None = None) -> CommitResult:
        return self._commit("decision", relative_bundle=relative_bundle,
                            transaction_id=transaction_id, timestamp=created_at_utc or _now(),
                            apply_selection=apply_selection, verify_selection=verify_selection)

    def finalize(self, *, transaction_id: str, run_stop_reason: str,
                 apply_selection: Callable[[dict[str, Any]], Any],
                 verify_selection: Callable[[Any, dict[str, Any]], bool],
                 created_at_utc: str | None = None) -> CommitResult:
        return self._commit("finalize", transaction_id=transaction_id,
                            timestamp=created_at_utc or _now(), run_stop_reason=run_stop_reason,
                            apply_selection=apply_selection, verify_selection=verify_selection)

    def _commit(self, operation_kind: str, *, transaction_id: str, timestamp: str,
                relative_bundle: str | None = None, run_stop_reason: str | None = None,
                apply_selection: Callable[[dict[str, Any]], Any] | None = None,
                verify_selection: Callable[[Any, dict[str, Any]], bool] | None = None) -> CommitResult:
        if self.poisoned:
            raise SelectionError("ledger is terminal after a transaction failure; never retry")
        if not callable(apply_selection) or not callable(verify_selection):
            raise SelectionError("explicit GR selection apply/verify callbacks are required")
        if not _RUNID.fullmatch(transaction_id):
            raise SelectionError("invalid transaction ID")
        if not isinstance(timestamp, str):
            raise SelectionError("timestamp must be supplied as UTC ISO-8601")
        current = self.verify_committed_state()
        if operation_kind == "decision" and current is not None and current["finalized"]:
            raise SelectionError("cannot append decision after finalization")
        if operation_kind == "finalize" and current is not None and current["finalized"]:
            raise SelectionError("selection history is already finalized")
        stage = "payload"
        candidate = None
        intent_ref = None
        intent_reference = None
        new_head = None
        new_head_bytes = None
        new_head_sha = None
        sequence = (current["sequence"] + 1) if current else 1
        previous_head_sha = self.context.expected_head_sha256
        try:
            if operation_kind == "decision":
                if current and current["finalized"]:
                    raise SelectionError("cannot append decision after finalization")
                candidate, manifest_rel, score, validation_sha = self._candidate(relative_bundle or "")
                if candidate["epoch"] != sequence:
                    raise SelectionError("candidate epoch must equal next complete decision sequence")
                prev_receipt = current["decision_receipt"] if current else None
                prev_pointer_value = None
                if current:
                    prev_pointer_value, _ = self._read_reference(current["selected_pointer"], "immutable_selected_pointer")
                prev_selected = prev_pointer_value["selected"] if prev_pointer_value else None
                prev_best = prev_pointer_value["best_f1"] if prev_pointer_value else None
                decision = decide(sequence=sequence, candidate=candidate, candidate_f1=score,
                                  candidate_manifest_relative_path=manifest_rel,
                                  candidate_validation_sha256=validation_sha,
                                  previous_receipt=prev_receipt, previous_selected=prev_selected,
                                  previous_best_f1=prev_best)
                improve = decision["decision"] == "selected"
                pointer_changes = improve
                record = _ensure_common(decision, self.context, transaction_id, timestamp)
                receipt_kind = "tgrab_validation_selection_decision_v1"
            elif operation_kind == "finalize":
                if not current or current["finalized"]:
                    raise SelectionError("finalization requires one nonfinalized committed history")
                if run_stop_reason not in ("loss_patience", "complete_epoch_cap", "deadline", "owner_stop", "incomplete_epoch", "error"):
                    raise SelectionError("invalid frozen finalization stop reason")
                prev_pointer_value, _ = self._read_reference(current["selected_pointer"], "immutable_selected_pointer")
                prev_receipt = current["decision_receipt"]
                selected = prev_pointer_value["selected"]
                best = prev_pointer_value["best_f1"]
                # Finalization records the last completed validation epoch, not
                # the (possibly earlier) epoch of the winning checkpoint.
                epoch_last = current["sequence"]
                record = {"kind": "tgrab_validation_selection_finalized_v1", "schema_version": 1,
                          "sequence": sequence, "previous_receipt": prev_receipt,
                          "selected": selected, "best_f1": best, "last_epoch": epoch_last,
                          "finalized": True, "run_stop_reason": run_stop_reason,
                          "test_authorized_by_this_receipt": False}
                record = _ensure_common(record, self.context, transaction_id, timestamp)
                candidate = selected
                improve = False
                pointer_changes = True
                receipt_kind = "tgrab_validation_selection_finalized_v1"
                manifest_rel = prev_pointer_value["manifest_relative_path"]
                score = best
            else:
                raise SelectionError("unsupported transaction operation")

            record_bytes = _validate_record(record, receipt_kind)
            receipt_rel = f"selection/receipts/receipt-{sequence:03d}-{transaction_id}.json"
            receipt_reference = {"role": "immutable_selection_receipt", "relative_path": receipt_rel,
                                 "bytes": len(record_bytes), "sha256": _sha(record_bytes)}
            if pointer_changes:
                pointer = {"kind": "tgrab_selected_pointer_v1", "schema_version": 1,
                           "stage": "finalized" if operation_kind == "finalize" else "selected",
                           "selected": record["selected"], "manifest_relative_path": manifest_rel,
                           "selection_receipt": receipt_reference, "best_f1": record["best_f1"],
                           "last_epoch": record["last_epoch"]}
                pointer = _ensure_common(pointer, self.context, transaction_id, timestamp)
                pointer_bytes = _validate_record(pointer, "tgrab_selected_pointer_v1")
                pointer_rel = f"selection/pointers/pointer-{sequence:03d}-{transaction_id}.json"
                pointer_ref = {"role": "immutable_selected_pointer", "relative_path": pointer_rel,
                               "bytes": len(pointer_bytes), "sha256": _sha(pointer_bytes)}
            else:
                if current is None:
                    raise SelectionError("first decision cannot retain a pointer")
                pointer_ref = current["selected_pointer"]
                pointer_value, pointer_bytes = self._read_reference(pointer_ref, "immutable_selected_pointer")
                pointer = pointer_value

            previous_pointer_sha = current["selected_pointer"]["sha256"] if current else None
            intent = {"kind": "tgrab_selection_transaction_intent_v1", "schema_version": 1,
                      "sequence": sequence, "operation": "finalize" if operation_kind == "finalize" else ("select" if improve else "retain"),
                      "previous_head_sha256": previous_head_sha, "previous_pointer_sha256": previous_pointer_sha,
                      "next_decision_receipt": receipt_reference, "next_selected_pointer": pointer_ref,
                      "pointer_alias_will_change": pointer_changes, "automatic_retry": False}
            intent = _ensure_common(intent, self.context, transaction_id, timestamp)
            intent_bytes = _validate_record(intent, "tgrab_selection_transaction_intent_v1")
            intent_rel = f"selection/intents/intent-{sequence:03d}-{transaction_id}.json"
            intent_reference = {"role": "immutable_transaction_intent", "relative_path": intent_rel,
                                "bytes": len(intent_bytes), "sha256": _sha(intent_bytes)}
            head = {"kind": "tgrab_selection_head_v1", "schema_version": 1,
                    "sequence": sequence, "decision_receipt": receipt_reference,
                    "selected_pointer": pointer_ref, "transaction_intent": intent_reference,
                    "finalized": operation_kind == "finalize"}
            head.update({"run_id": self.context.run_id, "attempt_id": self.context.attempt_id,
                         "created_at_utc": timestamp, "schema_sha256": self.context.schema_sha256,
                         "run_admission_sha256": self.context.run_admission_sha256})
            new_head_bytes = _validate_record(head, "tgrab_selection_head_v1")
            new_head_sha = _sha(new_head_bytes)
            new_head = head

            stage = "transaction_intent"
            self._hook(stage, "before")
            intent_ref = self._write_immutable(intent_rel, intent_bytes, 16384)
            self._hook(stage, "after")

            stage = "receipt"
            self._hook(stage, "before")
            stored_receipt_ref = self._write_immutable(receipt_rel, record_bytes, 16384)
            if stored_receipt_ref != receipt_reference:
                raise SelectionError("receipt publication reference mismatch")
            self._hook(stage, "after")

            if pointer_changes:
                stage = "pointer_object"
                self._hook(stage, "before")
                stored_pointer_ref = self._write_immutable(pointer_rel, pointer_bytes, 4096)
                if stored_pointer_ref != pointer_ref:
                    raise SelectionError("pointer object publication reference mismatch")
                self._hook(stage, "after")

            stage = "selection_apply"
            self._hook(stage, "before")
            # The callback is supplied by the separately controlled caller. It
            # runs only after immutable intent/receipt/pointer prerequisites are
            # durable. Its exceptions are terminal; state is never rolled back.
            applied = apply_selection(record)
            self._hook(stage, "after")

            stage = "selection_verify"
            self._hook(stage, "before")
            if verify_selection(applied, record) is not True:
                raise SelectionError("selection callback result differs from prepared immutable decision")
            self._hook(stage, "after")

            if pointer_changes:
                stage = "selected_alias"
                self._atomic_replace("selected.json", pointer_bytes, transaction_id, stage)

            stage = "head_replace"
            self._hook(stage, "before")
            temp = f".selection-head.{transaction_id}.tmp"
            tfd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=self.root_fd)
            try:
                view = memoryview(new_head_bytes)
                while view:
                    n = os.write(tfd, view)
                    if n <= 0:
                        raise SelectionError("short selection-head write")
                    view = view[n:]
                os.fsync(tfd)
            finally:
                os.close(tfd)
            os.replace(temp, "selection-head.json", src_dir_fd=self.root_fd, dst_dir_fd=self.root_fd)
            self._hook(stage, "after")

            stage = "head_fsync"
            self._hook(stage, "before")
            os.fsync(self.root_fd)
            self._hook(stage, "after")
            self.context = SelectionContext(self.context.run_id, self.context.attempt_id,
                                            self.context.run_admission_sha256, self.context.schema_sha256,
                                            new_head_sha)
            self._verify_history(new_head)
            return CommitResult(sequence, record.get("decision", "finalized"), record["selected"],
                                record["best_f1"], new_head_sha, receipt_reference, pointer_ref,
                                operation_kind == "finalize")
        except BaseException as exc:
            self.poisoned = True
            # Any error in/after atomic head replacement is reconciled by reading
            # its bytes; differing or unreadable state is uncertain, never retried.
            commit_state = "not_committed"
            if stage in ("head_replace", "head_fsync"):
                try:
                    observed = self._head_bytes()
                    if stage == "head_fsync":
                        commit_state = "uncertain_after_head_replace"
                    else:
                        observed_sha = _sha(observed) if observed is not None else None
                        if observed_sha == new_head_sha or observed_sha != previous_head_sha:
                            commit_state = "uncertain_after_head_replace"
                except Exception:
                    commit_state = "uncertain_after_head_replace"
            try:
                if intent_ref is None and intent_reference is not None:
                    try:
                        attempted = self._read_relative(intent_reference["relative_path"], 16384)
                        if len(attempted) == intent_reference["bytes"] and _sha(attempted) == intent_reference["sha256"]:
                            intent_ref = intent_reference
                    except Exception:
                        pass
                failure = {"kind": "tgrab_selection_transaction_failure_v1", "schema_version": 1,
                           "sequence": sequence, "intent": intent_ref,
                           "previous_committed_head_sha256": previous_head_sha,
                           "candidate": candidate, "failed_stage": stage,
                           "commit_state": commit_state,
                           "error_type": type(exc).__name__[:128], "terminal": True,
                           "automatic_retry": False}
                failure = _ensure_common(failure, self.context, transaction_id, timestamp)
                raw = _validate_record(failure, "tgrab_selection_transaction_failure_v1")
                self._write_immutable(f"selection/failures/failure-{sequence:03d}-{transaction_id}.json", raw, 16384)
            except Exception:
                pass
            raise SelectionError(f"terminal transaction failure at {stage} ({commit_state}); no retry: {exc}") from exc

