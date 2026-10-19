"""Create-only durable accounting for Figure 10 optimizer attempts.

This journal is a single-process callback sink, not a checkpoint, recovery
mechanism, or source of trust. An intent is fsynced before ``on_attempt``
returns, so the fit core cannot enter the optimizer first. A process that dies
after an intent leaves a charged, ambiguous attempt; reopening is prohibited.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from . import tgrab_selected_checkpoint as codec
from .tgrab_selection_transaction import SelectionContext


class AttemptJournalError(RuntimeError):
    """The journal is invalid, unsafe, or terminal after an I/O failure."""


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_UPDATES_PER_EPOCH = 2559
_MAX_SEQUENCE = 200 * _UPDATES_PER_EPOCH
_LOCK = ".owner-lock"


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return codec._canonical(dict(value))
    except Exception as exc:
        raise AttemptJournalError("journal record is not bounded canonical JSON") from exc


def _parse(raw: bytes) -> dict[str, Any]:
    try:
        value = codec.parse_canonical_json(raw, max_bytes=4096)
    except Exception as exc:
        raise AttemptJournalError("journal record is partial or corrupt") from exc
    if type(value) is not dict:
        raise AttemptJournalError("journal record must be an object")
    return value


class Figure10AttemptJournal:
    """Durably pair fit-core ``on_attempt`` and ``on_update`` callbacks.

    ``attempts_dir`` must already be a newly created, empty directory. The
    journal takes exclusive ownership with a create-only lock. It never opens
    or resumes a prior journal. Each intent and outcome is an immutable
    canonical JSON file, fsynced along with its directory entry before the
    corresponding callback returns.
    """

    def __init__(self, attempts_dir: str | os.PathLike[str],
                 context: SelectionContext):
        if not isinstance(context, SelectionContext):
            raise AttemptJournalError("explicit SelectionContext is required")
        if (type(context.run_id) is not str or _ID.fullmatch(context.run_id) is None
                or type(context.attempt_id) is not str
                or _ID.fullmatch(context.attempt_id) is None
                or type(context.run_admission_sha256) is not str
                or _SHA.fullmatch(context.run_admission_sha256) is None):
            raise AttemptJournalError("valid run, attempt and admission identities are required")

        self.path = Path(attempts_dir)
        if not self.path.is_absolute():
            self.path = Path.cwd() / self.path
        try:
            st = os.lstat(self.path)
            if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
                raise OSError("attempts path is not a real directory")
            self._dir_fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                   | getattr(os, "O_NOFOLLOW", 0))
            dst = os.fstat(self._dir_fd)
            if not stat.S_ISDIR(dst.st_mode) or (dst.st_dev, dst.st_ino) != (st.st_dev, st.st_ino):
                raise OSError("attempts directory changed during open")
            if os.listdir(self._dir_fd):
                raise OSError("attempts directory must be empty and fresh")
            self._lock_fd = os.open(_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                    | getattr(os, "O_NOFOLLOW", 0), 0o600,
                                    dir_fd=self._dir_fd)
            lock_st = os.fstat(self._lock_fd)
            if not stat.S_ISREG(lock_st.st_mode) or lock_st.st_nlink != 1:
                raise OSError("owner lock must be a regular single-link file")
            os.fsync(self._lock_fd)
            os.fsync(self._dir_fd)
        except BaseException as exc:
            for fd_name in ("_lock_fd", "_dir_fd"):
                fd = getattr(self, fd_name, None)
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    setattr(self, fd_name, None)
            raise AttemptJournalError("could not take exclusive ownership of fresh attempts directory") from exc

        self.context = context
        self._pid = os.getpid()
        self._last_sequence = 0
        self._pending: dict[str, Any] | None = None
        self._pending_sha: str | None = None
        self._poisoned = False
        self._terminal = False
        self._closed = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd_name in ("_lock_fd", "_dir_fd"):
            fd = getattr(self, fd_name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, fd_name, None)

    def __enter__(self) -> "Figure10AttemptJournal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _live(self) -> None:
        if self._closed or self._poisoned or os.getpid() != self._pid:
            raise AttemptJournalError("journal is closed, poisoned, or used outside its owning process")
        if self._terminal:
            raise AttemptJournalError("fit attempt ended after an optimizer exception; journal is terminal")

    def _fail(self, message: str, exc: BaseException | None = None):
        self._poisoned = True
        if exc is None:
            raise AttemptJournalError(message)
        raise AttemptJournalError(message) from exc

    def _expected_names(self, *, include_pending_outcome: bool = False) -> set[str]:
        names = {_LOCK}
        for sequence in range(1, self._last_sequence + 1):
            names.add(f"{sequence:08d}.intent.json")
            # With a pending intent, its outcome has not yet been observed.
            # Otherwise every charged intent must already have a final outcome.
            if self._pending is None or sequence < self._last_sequence or include_pending_outcome:
                names.add(f"{sequence:08d}.outcome.json")
        return names

    def _verify_namespace(self, *, outcome_expected: bool = False) -> None:
        try:
            names = set(os.listdir(self._dir_fd))
            expected = self._expected_names(include_pending_outcome=outcome_expected)
            if names != expected:
                raise OSError("journal namespace differs from its expected create-only contents")
            for name in names:
                fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._dir_fd)
                try:
                    st = os.fstat(fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                        raise OSError("journal entry is not a regular single-link file")
                finally:
                    os.close(fd)
            if self._pending is not None:
                raw = self._read_file(f"{self._last_sequence:08d}.intent.json")
                if hashlib.sha256(raw).hexdigest() != self._pending_sha or _parse(raw) != self._pending:
                    raise OSError("pending attempt intent changed after durable commit")
        except BaseException as exc:
            self._fail("journal namespace or prior intent integrity check failed", exc)

    def _read_file(self, name: str) -> bytes:
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._dir_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size > 4096:
                raise OSError("journal file is not a bounded regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _write(self, name: str, value: Mapping[str, Any]) -> bytes:
        raw = _canonical(value)
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=self._dir_fd)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise OSError("new journal record is not regular and single-link")
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short write while persisting journal record")
                view = view[count:]
            os.fsync(fd)
            os.fsync(self._dir_fd)
            return raw
        finally:
            os.close(fd)

    def on_attempt(self, record: Mapping[str, Any]) -> None:
        """Persist intent before returning control to the fit core."""
        self._live()
        try:
            if not isinstance(record, Mapping):
                raise ValueError("attempt callback must be a mapping")
            sequence = record.get("sequence")
            epoch = record.get("epoch")
            target_t = record.get("target_timestamp")
            if (type(sequence) is not int or type(epoch) is not int or type(target_t) is not int
                    or not 1 <= sequence <= _MAX_SEQUENCE
                    or not 1 <= epoch <= 200 or not 1 <= target_t <= _UPDATES_PER_EPOCH
                    or sequence != (epoch - 1) * _UPDATES_PER_EPOCH + target_t
                    or self._pending is not None or sequence != self._last_sequence + 1):
                raise ValueError("attempt sequence/epoch/timestamp is duplicate, gapped, or out of order")
            if (record.get("kind") != "tgrab-figure10-fit-attempt-v1"
                    or type(record.get("training_order_index")) is not int
                    or type(record.get("consumed_timestamp")) is not int
                    or record.get("training_order_index") != target_t - 1
                    or record.get("consumed_timestamp") != target_t - 1):
                raise ValueError("attempt chronology differs from the frozen fit callback")
            self._verify_namespace()
            intent = {
                "kind": "tgrab_figure10_optimizer_intent_v1",
                "run_id": self.context.run_id,
                "attempt_id": self.context.attempt_id,
                "admission_sha256": self.context.run_admission_sha256,
                "sequence": sequence,
                "epoch": epoch,
                "training_order_index": target_t - 1,
                "target_timestamp": target_t,
                "consumed_timestamp": target_t - 1,
            }
            raw = self._write(f"{sequence:08d}.intent.json", intent)
            # In-memory charging happens only after file and directory fsync.
            self._last_sequence = sequence
            self._pending = intent
            self._pending_sha = hashlib.sha256(raw).hexdigest()
        except BaseException as exc:
            self._fail("could not durably record optimizer intent; journal is terminal", exc)

    def on_update(self, record: Mapping[str, Any]) -> None:
        """Persist the final observed outcome paired to the pending intent."""
        self._live()
        try:
            if not isinstance(record, Mapping) or self._pending is None:
                raise ValueError("post-update callback has no pending durable intent")
            pending = self._pending
            seq = pending["sequence"]
            epoch = pending["epoch"]
            target_t = pending["target_timestamp"]
            counters = ("attempt", "attempted_updates", "completed_updates",
                        "epoch_attempted_updates", "epoch_completed_updates",
                        "optimizer_steps_confirmed")
            if any(type(record.get(key)) is not int for key in counters):
                raise ValueError("post-update counters must be exact integers")
            if any(type(record.get(key)) is not int for key in
                   ("epoch", "target_timestamp", "consumed_timestamp")):
                raise ValueError("post-update chronology must use exact integers")
            if (record.get("attempt") != seq or record.get("epoch") != epoch
                    or record.get("target_timestamp") != target_t
                    or record.get("consumed_timestamp") != target_t - 1
                    or record.get("attempted_updates") != seq
                    or record.get("epoch_attempted_updates") != target_t):
                raise ValueError("post-update chronology/counters do not match pending intent")
            status = record.get("optimizer_step_status")
            completed_flag = record.get("completed")
            if type(completed_flag) is not bool or status not in {"completed", "not_started", "unknown"}:
                raise ValueError("post-update optimizer outcome is invalid")
            if completed_flag and status != "completed":
                raise ValueError("fit core marked update completed with a non-completed optimizer status")
            if record.get("completed_updates") not in (seq, seq - 1):
                raise ValueError("completed update counter is inconsistent")
            if completed_flag and record["completed_updates"] != seq:
                raise ValueError("successful backend return must increment completed update count")
            if not completed_flag and record["completed_updates"] != seq - 1:
                raise ValueError("raised backend call cannot increment completed update count")
            expected_epoch_completed = target_t if completed_flag else target_t - 1
            if record["epoch_completed_updates"] != expected_epoch_completed:
                raise ValueError("epoch completed counter differs from exact callback chronology")
            # A completed backend call, or an exception carrying an explicit
            # completed-step witness, confirms this update. Unknown and
            # not-started exceptions do not. All earlier attempts in this fit
            # must have completed before the current sequential callback.
            expected_steps = seq if status == "completed" else seq - 1
            if record["optimizer_steps_confirmed"] != expected_steps:
                raise ValueError("optimizer step counter differs from exact outcome semantics")
            self._verify_namespace()
            outcome = {
                "kind": "tgrab_figure10_optimizer_outcome_v1",
                "run_id": self.context.run_id,
                "attempt_id": self.context.attempt_id,
                "admission_sha256": self.context.run_admission_sha256,
                "sequence": seq,
                "intent_sha256": self._pending_sha,
                "epoch": epoch,
                "target_timestamp": target_t,
                "consumed_timestamp": target_t - 1,
                "attempted_updates": seq,
                "completed_updates": record["completed_updates"],
                "epoch_completed_updates": record["epoch_completed_updates"],
                "optimizer_steps_confirmed": record["optimizer_steps_confirmed"],
                "optimizer_step_status": status,
                "outcome": status,
                "completed_return": completed_flag,
                "error_type": record.get("error_type"),
            }
            # If a write/fsync fails, leave the intent as charged and ambiguous;
            # poison the object and never retry the outcome write.
            self._write(f"{seq:08d}.outcome.json", outcome)
            self._pending = None
            self._pending_sha = None
            if not completed_flag:
                # The fit core re-raises after every failed backend call. Even
                # when an optimizer step is confirmed complete, it cannot
                # advance to another attempt in this fit invocation.
                self._terminal = True
        except BaseException as exc:
            self._fail("could not durably record optimizer outcome; attempt remains charged/ambiguous", exc)
