"""Fresh-only long-fit v2 append journal; no v1 journal/receipt compatibility.

One fsynced intent precedes each optimizer entry. One fsynced outcome follows
its observed return/exception. A crash can leave an honestly charged pending
intent. Nothing here permits reopening, repairing, retrying or resuming.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat

from . import tgrab_selected_checkpoint as codec

UPDATES = 2559
MAX_EPOCHS = 200
MAX_CHUNK_BYTES = 32 * 1024 * 1024


class ChunkJournalError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkJournalContext:
    run_id: str
    attempt_id: str
    contract_sha256: str
    admission_sha256: str


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class Figure10ChunkJournal:
    def __init__(self, root, context):
        if (type(context) is not ChunkJournalContext
                or any(type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value)
                       for value in (context.run_id, context.attempt_id))
                or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in (context.contract_sha256, context.admission_sha256))):
            raise ChunkJournalError("explicit valid chunk context required")
        self.root, self.context = Path(root), context
        self.fd = self.lock = self.chunk = None
        self.closed = self.poisoned = self.terminal = False
        self.pid = os.getpid()
        self.sequence = self.completed = self.confirmed = 0
        self.pending = None
        self.epoch = None
        self.chunk_bytes = self.chunk_records = 0
        self.chunk_hasher = hashlib.sha256()
        self.last_record_sha256 = None
        self.sealed = []
        try:
            # Caller creates the fresh empty directory, matching the v1 API.
            self.fd = codec._open_bundle_directory(self.root.absolute())
            if os.listdir(self.fd):
                raise ChunkJournalError("fresh empty chunk directory required")
            self.directory_identity = _identity(os.fstat(self.fd))[:2]
            self.lock = os.open(".owner-lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o600, dir_fd=self.fd)
            os.fsync(self.lock); os.fsync(self.fd)
            self.lock_identity = _identity(os.fstat(self.lock))
        except BaseException:
            self.close()
            raise

    def _fail(self, message, exc=None):
        self.poisoned = True
        raise ChunkJournalError(message) from exc

    def _live(self, *, audit=False):
        if self.closed or self.poisoned or os.getpid() != self.pid or (self.terminal and not audit):
            raise ChunkJournalError("journal closed/poisoned/terminal or wrong owner process")
        directory = os.lstat(self.root)
        lock = os.stat(".owner-lock", dir_fd=self.fd, follow_symlinks=False)
        if (not stat.S_ISDIR(directory.st_mode) or _identity(directory)[:2] != self.directory_identity
                or not stat.S_ISREG(lock.st_mode) or lock.st_nlink != 1 or _identity(lock) != self.lock_identity):
            self._fail("journal ownership changed")
        if self.chunk is not None:
            opened = os.fstat(self.chunk)
            linked = os.stat(self._chunk_name(), dir_fd=self.fd, follow_symlinks=False)
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or _identity(opened) != self.chunk_identity or _identity(linked) != self.chunk_identity):
                self._fail("active chunk identity/size changed")

    def _chunk_name(self):
        return f"epoch-{self.epoch:03d}.jsonl"

    def _open_epoch(self, epoch):
        if self.chunk is not None or epoch != len(self.sealed) + 1:
            raise ChunkJournalError("previous epoch must be sealed before advancing")
        self.epoch = epoch
        self.chunk = os.open(self._chunk_name(), os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=self.fd)
        os.fsync(self.chunk); os.fsync(self.fd)
        self.chunk_identity = _identity(os.fstat(self.chunk))
        self.chunk_bytes = self.chunk_records = 0
        self.chunk_hasher = hashlib.sha256()

    def _append(self, kind, record):
        self._live()
        value = {"kind": kind, "run_id": self.context.run_id, "attempt_id": self.context.attempt_id,
                 "contract_sha256": self.context.contract_sha256,
                 "admission_sha256": self.context.admission_sha256,
                 "previous_record_sha256": self.last_record_sha256,
                 "previous_chunk_sha256": self.sealed[-1]["chunk_sha256"] if self.sealed else None,
                 "record": dict(record)}
        raw = codec._canonical(value)
        if len(raw) > 4096 or self.chunk_bytes + len(raw) > MAX_CHUNK_BYTES:
            raise ChunkJournalError("bounded journal record/chunk exceeded")
        view = memoryview(raw)
        while view:
            count = os.write(self.chunk, view)
            if count <= 0:
                raise OSError("short/zero journal write")
            view = view[count:]
        os.fsync(self.chunk)
        # The epoch directory entry is durable before any callback returns.
        os.fsync(self.fd)
        self.chunk_hasher.update(raw)
        self.last_record_sha256 = _sha(raw)
        self.chunk_bytes += len(raw)
        self.chunk_records += 1
        self.chunk_identity = _identity(os.fstat(self.chunk))

    def on_attempt(self, record):
        try:
            self._live()
            sequence, epoch, target = (record.get(key) for key in ("sequence", "epoch", "target_timestamp"))
            if (any(type(x) is not int for x in (sequence, epoch, target))
                    or not 1 <= epoch <= MAX_EPOCHS or not 1 <= target <= UPDATES
                    or sequence != (epoch - 1) * UPDATES + target or sequence != self.sequence + 1
                    or self.pending is not None or record.get("kind") != "tgrab-figure10-fit-attempt-v1"
                    or type(record.get("training_order_index")) is not int
                    or type(record.get("consumed_timestamp")) is not int
                    or record["training_order_index"] != target - 1 or record["consumed_timestamp"] != target - 1):
                raise ChunkJournalError("invalid attempt chronology")
            if self.chunk is None:
                self._open_epoch(epoch)
            if epoch != self.epoch:
                raise ChunkJournalError("epoch changed before sealing")
            self._append("figure10_longfit_optimizer_intent_v2", record)
            self.sequence = sequence
            self.pending = {"sequence": sequence, "epoch": epoch, "target_timestamp": target,
                            "intent_record_sha256": self.last_record_sha256}
        except BaseException as exc:
            self._fail("intent failed; attempt may be charged and journal cannot resume", exc)

    def on_update(self, record):
        try:
            self._live()
            if self.pending is None:
                raise ChunkJournalError("no pending intent")
            seq, epoch, target = (self.pending[key] for key in ("sequence", "epoch", "target_timestamp"))
            keys = ("attempt", "attempted_updates", "completed_updates", "epoch_attempted_updates",
                    "epoch_completed_updates", "optimizer_steps_confirmed", "epoch", "target_timestamp", "consumed_timestamp")
            if any(type(record.get(key)) is not int for key in keys):
                raise ChunkJournalError("exact integer outcome counters required")
            completed, status = record.get("completed"), record.get("optimizer_step_status")
            if (type(completed) is not bool or status not in {"completed", "not_started", "unknown"}
                    or (completed and status != "completed")
                    or (record["attempt"], record["attempted_updates"], record["epoch"], record["target_timestamp"],
                        record["consumed_timestamp"], record["epoch_attempted_updates"])
                       != (seq, seq, epoch, target, target - 1, target)
                    or record["completed_updates"] != seq - (not completed)
                    or record["epoch_completed_updates"] != target - (not completed)
                    or record["optimizer_steps_confirmed"] != seq - (status != "completed")):
                raise ChunkJournalError("outcome disagrees with charged chronology/status")
            self._append("figure10_longfit_optimizer_outcome_v2", {**dict(record),
                "intent_record_sha256": self.pending["intent_record_sha256"]})
            self.completed, self.confirmed = record["completed_updates"], record["optimizer_steps_confirmed"]
            self.pending = None
            if not completed or record.get("error_type") is not None:
                self.terminal = True
        except BaseException as exc:
            self._fail("outcome failed; intent remains charged/ambiguous and cannot resume", exc)

    def _read(self, name, expected_sha, expected_bytes, *, chunk=False):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != expected_bytes:
                raise ChunkJournalError("journal file identity/type/size mismatch")
            sha, count, lines = hashlib.sha256(), 0, 0
            previous = None
            with os.fdopen(os.dup(fd), "rb") as stream:
                for raw in stream:
                    if len(raw) > 4096 or not raw.endswith(b"\n"):
                        raise ChunkJournalError("torn/oversized JSONL record")
                    if chunk:
                        value = codec.parse_canonical_json(raw, max_bytes=4096)
                        if codec._canonical(value) != raw:
                            raise ChunkJournalError("noncanonical journal record")
                        if lines and value["previous_record_sha256"] != previous:
                            raise ChunkJournalError("broken record hashchain")
                        previous = _sha(raw)
                    sha.update(raw); count += len(raw); lines += 1
                    if count > expected_bytes:
                        raise ChunkJournalError("journal read grew beyond bound")
            after = os.fstat(fd)
            linked = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            if (_identity(before) != _identity(after) or _identity(after) != _identity(linked)
                    or sha.hexdigest() != expected_sha or count != expected_bytes):
                raise ChunkJournalError("journal file changed or hash mismatch")
            return lines
        finally:
            os.close(fd)

    def seal_epoch(self, epoch):
        try:
            self._live()
            if (type(epoch) is not int or epoch != self.epoch or self.pending is not None
                    or self.sequence != epoch * UPDATES or self.completed != self.sequence
                    or self.confirmed != self.sequence or self.chunk_records != UPDATES * 2):
                raise ChunkJournalError("seal requires exactly 2559 successful epoch updates")
            self._read(self._chunk_name(), self.chunk_hasher.hexdigest(), self.chunk_bytes, chunk=True)
            receipt = {"kind": "figure10_longfit_epoch_chunk_seal_v2", "epoch": epoch,
                       "run_id": self.context.run_id, "attempt_id": self.context.attempt_id,
                       "contract_sha256": self.context.contract_sha256,
                       "admission_sha256": self.context.admission_sha256,
                       "filename": self._chunk_name(), "chunk_sha256": self.chunk_hasher.hexdigest(),
                       "bytes": self.chunk_bytes, "records": self.chunk_records,
                       "previous_chunk_sha256": self.sealed[-1]["chunk_sha256"] if self.sealed else None,
                       "last_record_sha256": self.last_record_sha256,
                       "attempted_updates": self.sequence, "completed_updates": self.completed,
                       "optimizer_steps_confirmed": self.confirmed}
            raw = codec._canonical(receipt)
            name = f"epoch-{epoch:03d}.seal.json"
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400, dir_fd=self.fd)
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short seal write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fchmod(self.chunk, 0o400)
            os.fsync(self.chunk); os.fsync(self.fd)
            os.close(self.chunk); self.chunk = None
            stored = {**receipt, "receipt_filename": name, "receipt_sha256": _sha(raw), "receipt_bytes": len(raw)}
            self.sealed.append(stored)
            return dict(stored)
        except BaseException as exc:
            self._fail("epoch sealing failed; no candidate may be accepted", exc)

    def audit(self):
        """Verify all immutable chunks and current prefix once; allow pending STOP."""
        try:
            self._live(audit=True)
            names = {".owner-lock"}
            for item in self.sealed:
                self._read(item["filename"], item["chunk_sha256"], item["bytes"], chunk=True)
                self._read(item["receipt_filename"], item["receipt_sha256"], item["receipt_bytes"])
                names.update((item["filename"], item["receipt_filename"]))
            if self.chunk is not None:
                self._read(self._chunk_name(), self.chunk_hasher.hexdigest(), self.chunk_bytes, chunk=True)
                names.add(self._chunk_name())
            if set(os.listdir(self.fd)) != names:
                raise ChunkJournalError("unexpected/missing journal namespace entries")
            return {"attempted_updates": self.sequence, "completed_updates": self.completed,
                    "optimizer_steps_confirmed": self.confirmed,
                    "pending": dict(self.pending) if self.pending else None,
                    "sealed_epochs": len(self.sealed), "terminal": self.terminal,
                    "current_epoch": self.epoch if self.chunk is not None else None,
                    "current_chunk_bytes": self.chunk_bytes if self.chunk is not None else 0,
                    "current_chunk_sha256": self.chunk_hasher.hexdigest() if self.chunk is not None else None}
        except BaseException as exc:
            self._fail("full chunk audit failed; no retry or acceptance", exc)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for name in ("chunk", "lock", "fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
