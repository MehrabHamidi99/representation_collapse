#!/usr/bin/env python3
"""Supervise one admitted legacy array audit; never submit or retry work."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def process_identity(pid):
    """Public PID/parent/start identity only; no command line or environment."""
    if pid <= 0:
        return None  # Kernel table sentinel, never an owned child PID.
    try:
        if sys.platform.startswith("linux"):
            proc = Path("/proc") / str(pid)
            with (proc / "stat").open() as stream:
                fields = stream.read(16384).rsplit(") ", 1)[1].split()
            return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                    "uid": proc.stat().st_uid, "start": "linux-ticks:" + fields[19], "state": fields[0]}
        if sys.platform == "darwin":
            class BSDInfo(ctypes.Structure):
                _fields_ = [(key, ctypes.c_uint32) for key in
                    ("flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid", "svuid", "svgid", "reserved")]
                _fields_ += [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
                _fields_ += [(key, ctypes.c_uint32) for key in ("nfiles", "pgid", "pjobc", "tdev", "tpgid")]
                _fields_ += [("nice", ctypes.c_int32), ("start_seconds", ctypes.c_uint64), ("start_microseconds", ctypes.c_uint64)]
            info = BSDInfo()
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            library.proc_pidinfo.restype = ctypes.c_int
            ctypes.set_errno(0)
            size = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if size != ctypes.sizeof(info):
                if ctypes.get_errno() in (2, 3):
                    return None
                raise RuntimeError("process identity could not be verified")
            return {"pid": pid, "ppid": info.ppid, "pgid": info.pgid, "uid": info.uid,
                    "start": f"darwin-time:{info.start_seconds}:{info.start_microseconds}",
                    "state": "Z" if info.status == 5 else "live"}
        raise RuntimeError("unsupported process identity platform")
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(left, right):
    return bool(left and right and all(left[key] == right[key] for key in ("pid", "uid", "start")))


def process_table(*, timeout=0.5, parents=None):
    """Bounded kernel process metadata, without invoking ps or reading argv."""
    if sys.platform == "darwin" and parents is not None:
        # Inspect the owned parent chain only. Sandbox-denied unrelated
        # processes are outside this snapshot, never treated as absent owners.
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        library.proc_listpids.restype = ctypes.c_int
        queue, seen, identities = list(parents), set(), []
        until = time.monotonic() + timeout
        while queue:
            if time.monotonic() >= until:
                raise TimeoutError("bounded owned process snapshot expired")
            parent = queue.pop()
            if parent in seen:
                continue
            seen.add(parent)
            size = library.proc_listpids(6, parent, None, 0)
            if not 0 <= size <= 262144:
                raise RuntimeError("owned child table unavailable")
            buffer = (ctypes.c_int * (size // 4 + 32))()
            returned = library.proc_listpids(6, parent, buffer, ctypes.sizeof(buffer))
            if not 0 <= returned < ctypes.sizeof(buffer) or returned % 4:
                raise RuntimeError("owned child table unavailable")
            for pid in buffer[:returned // 4]:
                identity = process_identity(pid)
                if identity is not None and identity["ppid"] == parent and identity["uid"] == os.getuid():
                    identities.append(identity)
                    queue.append(pid)
            if len(identities) > 256:
                raise RuntimeError("owned child table exceeds bound")
        return identities
    if sys.platform.startswith("linux"):
        pids = [int(item.name) for item in Path("/proc").iterdir() if item.name.isdigit()]
    elif sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.proc_listallpids.restype = ctypes.c_int
        capacity = library.proc_listallpids(None, 0) + 1024
        if not 0 < capacity <= 65536:
            raise RuntimeError("invalid bounded process table size")
        buffer = (ctypes.c_int * capacity)()
        count = library.proc_listallpids(buffer, ctypes.sizeof(buffer))
        if not 0 <= count < capacity:
            raise RuntimeError("process table unavailable or overflowed")
        pids = list(buffer[:count])
    else:
        raise RuntimeError("unsupported process table platform")
    if len(pids) > 65536:
        raise RuntimeError("process table exceeds bound")
    deadline = time.monotonic() + timeout
    identities = []
    for pid in pids:
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded process snapshot expired")
        identities.append(process_identity(pid))
    return [identity for identity in identities if identity is not None]


def signal_owned(identity, signum, *, deadline=None):
    """Refuse PID reuse. Linux pidfd additionally closes the check/signal race."""
    current = process_identity(identity["pid"])
    if not same_process(identity, current) or current["state"] == "Z":
        return False
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("owned signal deadline expired")
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        descriptor = os.pidfd_open(identity["pid"])
        try:
            if not same_process(identity, process_identity(identity["pid"])):
                return False
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("owned signal deadline expired")
            signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
    else:
        os.kill(identity["pid"], signum)
    return True


def install_spawn_journal(path):
    """Track this worker's Python-spawned SDK core before it can be orphaned."""
    original = subprocess.Popen
    records = []
    class RegisteredPopen(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            identity = process_identity(self.pid)
            if identity is None:
                if self.poll() is None:
                    raise RuntimeError("spawn identity unavailable")
                return
            if identity["ppid"] != os.getpid() or identity["uid"] != os.getuid():
                raise RuntimeError("spawn parent identity mismatch")
            records.append(identity)
            try:
                if len(records) > 256:
                    raise RuntimeError("owned spawn journal exceeds bound")
                atomic_json(path, {"schema_version": 1, "created_by": os.getpid(), "processes": records})
            except BaseException:
                signal_owned(identity, signal.SIGKILL)
                raise
    subprocess.Popen = RegisteredPopen


class OwnedProcesses:
    def __init__(self, pid, journal):
        self.pid, self.journal = pid, journal
        identity = process_identity(pid)
        self.known = {pid: identity} if identity is not None else {}
        self.signals = []

    def refresh(self, *, timeout=0.5):
        # Journal captures a start_new_session SDK child even if its parent
        # fails between polling snapshots. Only the owned worker writes it.
        if self.journal.exists():
            with self.journal.open("rb") as stream:
                payload = stream.read(65537)
            if len(payload) > 65536:
                raise RuntimeError("oversized owned process journal")
            record = json.loads(payload)
            if record.get("created_by") != self.pid or len(record["processes"]) > 256:
                raise RuntimeError("invalid owned process journal")
            for identity in record["processes"]:
                if identity["ppid"] != self.pid or identity["uid"] != os.getuid():
                    raise RuntimeError("invalid journal ownership")
                if same_process(identity, process_identity(identity["pid"])):
                    self.known[identity["pid"]] = identity
        live = {pid for pid, identity in self.known.items() if same_process(identity, process_identity(pid))}
        rows = process_table(timeout=timeout, parents=live)
        while True:
            added = False
            for row in rows:
                pid, parent, uid = row["pid"], row["ppid"], row["uid"]
                if pid not in live and parent in live and uid == os.getuid():
                    identity = process_identity(pid)
                    if identity is not None and identity["ppid"] == parent:
                        self.known[pid] = identity
                        live.add(pid)
                        added = True
            if not added:
                break

    def cleanup(self, *, deadline=None):
        # Freeze known parents before a final graph scan, preventing further
        # worker/core spawns while capturing their already created descendants.
        errors = []
        for identity in list(self.known.values()):
            try:
                signal_owned(identity, signal.SIGSTOP, deadline=deadline)
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(type(error).__name__)
        try:
            self.refresh(timeout=0.25)
        except Exception as error:
            errors.append(type(error).__name__)
        for identity in reversed(list(self.known.values())):
            try:
                if signal_owned(identity, signal.SIGKILL, deadline=deadline):
                    self.signals.append({"pid": identity["pid"], "start": identity["start"], "signal": "SIGKILL"})
            except ProcessLookupError:
                pass
            except Exception as error:
                errors.append(type(error).__name__)
        # A successful kill call is not an observed exit. Reconcile every
        # recorded start identity, bounded by the outer envelope and250ms.
        until = time.monotonic() + 0.25
        if deadline is not None:
            until = min(until, deadline)
        pending = dict(self.known)
        self.cleanup_reconciliation = {}
        while pending and time.monotonic() < until:
            for pid, identity in list(pending.items()):
                if time.monotonic() >= until:
                    break
                try:
                    current = process_identity(pid)
                    if current is None or not same_process(identity, current) or current["state"] == "Z":
                        self.cleanup_reconciliation[str(pid)] = "original_process_exited"
                        del pending[pid]
                except Exception as error:
                    errors.append(type(error).__name__)
                    self.cleanup_reconciliation[str(pid)] = "unknown"
                    del pending[pid]
            if pending and time.monotonic() < until:
                time.sleep(min(0.005, max(0, until - time.monotonic())))
        for pid in pending:
            self.cleanup_reconciliation[str(pid)] = "live_or_unverified_at_cleanup_deadline"
        if pending:
            errors.append("OwnedDescendantUnverified")
        return errors


def atomic_json(path, value):
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive-root", "manifest", "manifest-sha256", "config", "config-sha256", "output",
                "source-sha256", "launcher-sha256", "source-commit"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--wall-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not 30 <= args.wall_seconds <= 600:
        parser.error("wall-seconds must be30..600")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.launcher_sha256:
        parser.error("launcher SHA256 mismatch")
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "state": "initializing", "wall_seconds": args.wall_seconds,
               "work_seconds": args.wall_seconds - 10, "cpu_threads": 1,
               "retry_count": 0, "scientific_acceptance": False}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "supervisor-receipt.json", receipt)
    persist()
    child = None
    ownership = None
    deadline = started + args.wall_seconds
    try:
        command = [sys.executable, "-m", "representation_collapse.legacy_floquet_audit"]
        for key in ("archive_root", "manifest", "manifest_sha256", "config", "config_sha256",
                    "source_sha256", "launcher_sha256", "source_commit"):
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        command.extend(["--output", str(output / "worker"), "--started-monotonic", str(started),
                        "--work-deadline-monotonic", str(started + args.wall_seconds - 10),
                        "--owned-process-journal", str(output / "owned-processes.json")])
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env[key] = "1"
        # No credentials or ambient environment are persisted. Offline remains fixed.
        env["WANDB_MODE"] = "offline"
        repo = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(repo / "src")
        with (output / "worker.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, env=env, cwd=repo, start_new_session=True)
            ownership = OwnedProcesses(child.pid, output / "owned-processes.json")
            ownership.refresh()
            receipt.update(state="running", owned_pid=child.pid)
            persist()
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    raise subprocess.TimeoutExpired(command, args.wall_seconds)
                ownership.refresh(timeout=min(0.5, remaining - 0.5))
                try:
                    child.wait(timeout=min(0.1, remaining - 1))
                except subprocess.TimeoutExpired:
                    pass
        receipt.update(state="completed" if child.returncode == 0 else "failed", exit_code=child.returncode)
    except BaseException as error:
        receipt.update(state="incomplete", error_type=type(error).__name__)
        receipt["outer_deadline_enforced"] = isinstance(error, subprocess.TimeoutExpired)
    finally:
        if child is not None:
            # This runs even when persistence or a post-spawn check fails.
            if ownership is None:
                ownership = OwnedProcesses(child.pid, output / "owned-processes.json")
            errors = ownership.cleanup(deadline=deadline)
            receipt["owned_processes"] = list(ownership.known.values())
            receipt["cleanup_signals"] = ownership.signals
            receipt["cleanup_error_types"] = errors
            receipt["cleanup_reconciliation"] = ownership.cleanup_reconciliation
            if errors:
                receipt["state"] = "incomplete"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    receipt["state"] = "incomplete"
        if time.monotonic() >= started + args.wall_seconds:
            receipt["state"] = "incomplete"
        persist()
    return 0 if receipt["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
