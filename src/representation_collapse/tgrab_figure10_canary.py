"""Prospective CPU-only T-GRAB 256-update prefix runtime diagnostic.

This is not fit sizing, historical replay, complete-epoch training, Figure 10
reproduction, or scientific evidence. The CLI requires an externally pinned
admission receipt from the stdlib supervisor, verified offline reporting,
explicit pins, process ownership, a deadline and STOP file. It has no test-split
opener, checkpoint writer, selection path, retry or resume path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import stat
import sys
import threading
import time
from typing import Callable, Sequence


class CanaryError(RuntimeError):
    """Canary precondition or runtime integrity failure."""


MAX_UPDATES = 256
PROTOCOL_ID = "tgrab_figure10_256_prefix_runtime_diagnostic_v1"
CONFIG_SHA256 = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
WRAPPER_SHA256 = {
    "model/gclstm.py": "86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609",
    "model/link_predictor.py": "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86",
    "model/node_emb.py": "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29",
}
IMPORT_HELPER_SHA256 = "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344"
REPORTER_SHA256 = "443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184"
GRAPH_DATA_SHA256 = "498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e"
GRAPH_RECONSTRUCTION_SHA256 = "1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13"
ENVIRONMENT_LOCK_SHA256 = "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
WHEEL_MANIFEST_SHA256 = "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"
REQUIRED_VERSIONS = {
    "torch": "2.0.0+cpu", "torch-geometric": "2.6.1",
    "torch-geometric-temporal": "0.54.0", "numpy": "1.24.4",
    "torch-cluster": "1.6.3+pt20cpu", "torch-scatter": "2.1.2+pt20cpu",
    "torch-sparse": "0.6.18+pt20cpu", "torch-spline-conv": "1.2.2+pt20cpu",
    "wandb": "0.30.0",
}


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_nlink != 1:
        raise CanaryError(f"single-link regular file required: {p}")
    if max_bytes is not None and p.stat().st_size > max_bytes:
        raise CanaryError(f"bounded file required: {p}")
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pinned_bytes(path: Path, expected: str, *, max_bytes: int) -> bytes:
    p = Path(os.path.abspath(os.fspath(path)))
    for ancestor in (p, *p.parents):
        try:
            if stat.S_ISLNK(ancestor.lstat().st_mode):
                raise CanaryError("symlink path component refused")
        except FileNotFoundError:
            raise CanaryError(f"pinned input is missing: {ancestor}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CanaryError("explicit lowercase SHA-256 pin required")
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise CanaryError(f"cannot open pinned regular file: {p}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes:
            raise CanaryError(f"bounded single-link regular input required: {p}")
        chunks = []
        size = 0
        while True:
            chunk = os.read(fd, min(65536, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise CanaryError(f"input exceeded size bound: {p}")
        after = os.fstat(fd)
        path_stat = p.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        raw = b"".join(chunks)
        if identity(before) != identity(after) or identity(after) != identity(path_stat) or size != before.st_size:
            raise CanaryError(f"pinned input changed while reading: {p}")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise CanaryError(f"SHA-256 identity mismatch: {p}")
        return raw
    finally:
        os.close(fd)


def verify_pin(path: Path, expected: str, *, max_bytes: int | None = None) -> None:
    if max_bytes is None:
        max_bytes = 64 * 1024 * 1024
    read_pinned_bytes(path, expected, max_bytes=max_bytes)


def environment_versions() -> dict[str, str]:
    observed = {}
    for name, expected in REQUIRED_VERSIONS.items():
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CanaryError(f"required distribution missing: {name}") from exc
        if value != expected:
            raise CanaryError(f"distribution mismatch: {name}={value}, expected {expected}")
        observed[name] = value
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise CanaryError("pinned CPython 3.10 / Linux x86_64 runtime required")
    return observed


def validate_environment_receipt(receipt: object) -> None:
    """Validate the actual preparation script's installed-metadata receipt."""
    if type(receipt) is not dict:
        raise CanaryError("environment preparation receipt must be a JSON object")
    if receipt.get("state") != "installed_metadata_verified_native_smoke_pending":
        raise CanaryError("environment receipt is not a completed isolated CPU install")
    expected = {
        "requirements_sha256": ENVIRONMENT_LOCK_SHA256,
        "artifact_manifest_sha256": WHEEL_MANIFEST_SHA256,
        "installed_cell_sha256": CELL_SHA256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise CanaryError(f"environment receipt identity mismatch: {key}")
    if receipt.get("installed_metadata_verified") is not True:
        raise CanaryError("environment distribution metadata was not verified")
    installed = receipt.get("installed_versions")
    if type(installed) is not dict or any(installed.get(name) != version for name, version in REQUIRED_VERSIONS.items()):
        raise CanaryError("environment receipt distribution versions mismatch")


def validate_node2a_receipt(receipt: object) -> None:
    """Require the pre-fit synthetic bootstrap gate before any optimizer attempt."""
    if type(receipt) is not dict or receipt.get("schema") != "node2a-synthetic-bootstrap-pass-v1" or receipt.get("status") != "pass":
        raise CanaryError("accepted node2a synthetic bootstrap receipt required")
    required_true = ("both_model_components_roundtripped", "complete_hc_state_roundtripped",
                     "forward_loss_gradient_parity", "mode_dropout_parity", "transaction_crash_reopen")
    if any(receipt.get(key) is not True for key in required_true):
        raise CanaryError("node2a receipt missing required parity evidence")
    if (type(receipt.get("optimizer_updates")) is not int or receipt.get("optimizer_updates") != 0
            or receipt.get("primary_payload_opened") is not False
            or receipt.get("validation_payload_opened") is not False
            or receipt.get("test_payload_opened") is not False):
        raise CanaryError("node2a receipt violates the synthetic pre-fit no-training/no-data boundary")


@dataclass
class ProcessOwner:
    run_dir: Path
    fd: int
    device: int
    inode: int
    token: str

    @classmethod
    def acquire(cls, run_dir: Path) -> "ProcessOwner":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        path = run_dir / ".owner.lock"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st = os.fstat(fd)
            token = f"{os.getpid()}:{st.st_dev}:{st.st_ino}:{time.time_ns()}"
            os.write(fd, (token + "\n").encode("ascii"))
            os.fsync(fd)
            return cls(run_dir, fd, st.st_dev, st.st_ino, token)
        except BaseException:
            os.close(fd)
            raise

    def check(self) -> None:
        st = os.fstat(self.fd)
        path_st = (self.run_dir / ".owner.lock").lstat()
        if (st.st_dev, st.st_ino) != (self.device, self.inode) or (path_st.st_dev, path_st.st_ino) != (self.device, self.inode):
            raise CanaryError("process ownership lock identity changed")

    def close(self) -> None:
        os.close(self.fd)


@dataclass
class CanaryGuard:
    stop_file: Path
    deadline_unix: float
    owner: ProcessOwner
    interrupted: bool = False

    def __call__(self) -> None:
        self.owner.check()
        if self.interrupted:
            raise CanaryError("termination signal received")
        # lexists semantics: a dangling symlink is still an explicit STOP.
        if os.path.lexists(self.stop_file):
            raise CanaryError("STOP file present")
        if not math.isfinite(self.deadline_unix) or time.time() >= self.deadline_unix:
            raise CanaryError("hard deadline reached")


def _finite_loss(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise CanaryError("numeric update loss required")
    loss = float(value)
    if not math.isfinite(loss) or loss < 0:
        raise CanaryError("finite nonnegative BCE loss required")
    return loss


class OptimizerWitnessError(CanaryError):
    def __init__(self, message: str, *, step_status: str):
        super().__init__(message)
        self.step_status = step_status


def witnessed_native_update(backend, state, target, negatives):
    """Observe exactly one optimizer.step without inferring mutation from loss."""
    optimizer = getattr(backend, "optimizer", None)
    if optimizer is None or not callable(getattr(optimizer, "step", None)):
        raise OptimizerWitnessError("optimizer step unavailable", step_status="not_started")
    sentinel = object()
    prior = getattr(optimizer, "__dict__", {}).get("step", sentinel)
    original = optimizer.step
    witness = {"entered": 0, "returned": 0}
    installed = False
    def status():
        if witness["entered"] == 0:
            return "not_started"
        return "completed" if witness["entered"] == witness["returned"] == 1 else "unknown"
    def step(*args, **kwargs):
        witness["entered"] += 1
        result = original(*args, **kwargs)
        witness["returned"] += 1
        return result
    try:
        optimizer.step = step
        installed = optimizer.step is step
        if not installed:
            raise RuntimeError("could not install optimizer witness")
        loss = backend.loss_and_update(state, target, negatives)
        if witness["entered"] != 1 or witness["returned"] != 1:
            raise RuntimeError("backend must perform exactly one returned optimizer.step")
        return loss
    except BaseException as exc:
        raise OptimizerWitnessError("native update failed", step_status=status()) from exc
    finally:
        if installed:
            if prior is sentinel:
                del optimizer.step
            else:
                optimizer.step = prior


def reconcile_attempts(path: Path, *, terminal_journal_error: bool = False) -> dict:
    """Read durable intents/outcomes; an unmatched/invalid outcome stays charged."""
    attempted = completed = steps = unknown = not_started = 0
    path = Path(path)
    if path.exists():
        intents = sorted(path.glob("????????.intent.json"))
        for index, intent_path in enumerate(intents):
            attempted += 1
            try:
                # A failed journal write/fsync is terminal. Even a complete
                # visible outcome may lack its durability barrier, so do not
                # upgrade that final attempt to a confirmed durable outcome.
                if terminal_journal_error and index == len(intents) - 1:
                    raise ValueError("final journal durability barrier failed")
                raw = intent_path.read_bytes()
                intent = json.loads(raw)
                outcome_path = intent_path.with_name(intent_path.name.replace(".intent.", ".outcome."))
                outcome = json.loads(outcome_path.read_bytes())
                if (outcome.get("intent_sha256") != hashlib.sha256(raw).hexdigest()
                        or outcome.get("sequence") != intent.get("sequence")):
                    raise ValueError("unpaired outcome")
                status = outcome["optimizer_step_status"]
                if status not in {"completed", "not_started", "unknown"}:
                    raise ValueError("invalid outcome")
                completed += outcome.get("completed_return") is True
                steps += status == "completed"
                unknown += status == "unknown"
                not_started += status == "not_started"
            except (OSError, ValueError, KeyError, TypeError):
                unknown += 1
    return {"attempted_updates": attempted, "completed_updates": completed,
            "optimizer_steps_confirmed": steps,
            "optimizer_mutation_unknown_attempts": unknown,
            "optimizer_not_started_attempts": not_started,
            "committed_updates": steps if unknown == 0 else None}


def run_prefix_updates(*, backend, graphs: Sequence, sampler: Callable,
                       guard: Callable[[], None], on_update: Callable[[dict], None],
                       on_attempt: Callable[[dict], None] = lambda _: None,
                       update: Callable | None = None,
                       limit: int = MAX_UPDATES) -> dict:
    """Run the first 256 targets only, with fit-compatible durable callbacks.

    The runtime supplies a fsync-before-return journal for on_attempt. No
    validation, checkpoint, selection, retry or resume is performed.
    """
    if type(limit) is not int or limit != MAX_UPDATES:
        raise CanaryError("canary update limit is frozen at exactly 256")
    if len(graphs) < limit + 1 or [g.timestamp for g in graphs[:limit + 1]] != list(range(limit + 1)):
        raise CanaryError("training prefix must contain contiguous timestamps 0..256")
    if any(g.nodes != 100 or g.node_ids != tuple(range(100)) for g in graphs[:limit + 1]):
        raise CanaryError("N=100 ascending node ordering required")
    guard()
    started = time.monotonic()
    state = backend.reset()
    attempted = completed = steps = 0
    losses = []
    update = update or backend.loss_and_update
    with backend.training_mode():
        for target_t in range(1, limit + 1):
            guard()
            predecessor, target = graphs[target_t - 1], graphs[target_t]
            if predecessor.timestamp != target_t - 1 or target.timestamp != target_t or state.next_target_t != target_t - 1:
                raise CanaryError("state/target chronology mismatch")
            state = backend.consume(backend.detach(state), predecessor)
            if state.next_target_t != target_t:
                raise CanaryError("exactly one predecessor consumption required")
            negatives = sampler(target)
            guard()
            on_attempt({"kind": "tgrab-figure10-fit-attempt-v1", "sequence": target_t,
                        "epoch": 1, "training_order_index": target_t - 1,
                        "target_timestamp": target_t, "consumed_timestamp": target_t - 1})
            attempted += 1
            record = {"attempt": attempted, "attempted_updates": attempted,
                      "epoch": 1, "epoch_attempted_updates": attempted,
                      "target_timestamp": target_t, "consumed_timestamp": target_t - 1}
            try:
                raw_loss = update(state, target, negatives)
            except BaseException as exc:
                status = getattr(exc, "step_status", "unknown")
                if status not in {"completed", "not_started", "unknown"}:
                    status = "unknown"
                steps += status == "completed"
                on_update(dict(record, completed=False, completed_updates=completed,
                    epoch_completed_updates=completed, optimizer_steps_confirmed=steps,
                    optimizer_step_status=status, optimizer_mutation_status=status,
                    error_type=type(exc).__name__))
                raise
            # A returned backend call is counted before loss validation or I/O.
            completed += 1
            steps += 1
            try:
                loss = _finite_loss(raw_loss)
            except BaseException as exc:
                on_update(dict(record, completed=True, completed_updates=completed,
                    epoch_completed_updates=completed, optimizer_steps_confirmed=steps,
                    optimizer_step_status="completed", optimizer_mutation_status="completed",
                    raw_loss_repr=repr(raw_loss), error_type=type(exc).__name__))
                raise
            losses.append(loss)
            on_update(dict(record, completed=True, completed_updates=completed,
                epoch_completed_updates=completed, optimizer_steps_confirmed=steps,
                optimizer_step_status="completed", optimizer_mutation_status="completed",
                raw_loss_repr=repr(raw_loss), loss=loss))
        guard()
    elapsed = time.monotonic() - started
    return {"attempted_updates": attempted, "completed_updates": completed,
            "optimizer_steps_confirmed": steps,
            "optimizer_mutation_unknown_attempts": 0, "optimizer_not_started_attempts": 0,
            "committed_updates": steps, "first_target": 1, "last_target": 256,
            "mean_loss": math.fsum(losses) / len(losses), "elapsed_seconds": elapsed,
            "updates_per_second": completed / max(elapsed, 1e-12),
            "validation_replayed": False, "selection_performed": False,
            "checkpoint_written": False, "test_payload_opened": False}


def _atomic_json_new(path: Path, value: dict) -> None:
    path = Path(path)
    temp = path.with_name(path.name + ".pending")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(temp, path, follow_symlinks=False)
        os.unlink(temp)
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def publish_terminal_manifest(owner: ProcessOwner, value: dict) -> None:
    """Always release the process lock after terminal evidence publication."""
    try:
        _atomic_json_new(owner.run_dir / "manifest.json", value)
    finally:
        owner.close()


def _runtime_train(args, owner: ProcessOwner, guard: CanaryGuard, reporter,
                   config_bytes: bytes):
    """Imports Torch, opens only the train split and performs the 256 updates."""
    guard()
    module_dir = Path(__file__).resolve().parent
    verify_pin(module_dir / "graph_data.py", GRAPH_DATA_SHA256, max_bytes=100_000)
    verify_pin(module_dir / "graph_reconstruction.py", GRAPH_RECONSTRUCTION_SHA256, max_bytes=100_000)
    verify_pin(module_dir / "pinned_graph_import.py", IMPORT_HELPER_SHA256, max_bytes=100_000)
    verify_pin(module_dir / "reporting.py", REPORTER_SHA256, max_bytes=100_000)
    import torch
    import numpy as np
    from . import graph_data as gd
    from . import graph_reconstruction as gr
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256

    config = json.loads(config_bytes.decode("ascii"))
    gr.validate_frozen_config(config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    modules = load_pinned_graph_modules(source_root=args.source_root,
        installed_cell_path=args.cell_path, cell_sha256=PINNED_CELL_SHA256,
        wrapper_sha256=PINNED_WRAPPER_SHA256, guard=guard)
    manifest = gd.read_manifest(args.data_manifest, args.data_manifest_sha256, guard)
    md = manifest.metadata
    if md["kind"] != "tgrab_primary_converted_v1" or md["splits"]["train"]["interval"] != [0, 2560]:
        raise CanaryError("verified primary training split [0,2560) required")
    if md["splits"]["validation"]["interval"] != [2560, 2816] or md["splits"]["test"]["interval"] != [2816, 3072]:
        raise CanaryError("primary chronology metadata mismatch")
    # Deliberately call only the train split loader. No test path is opened,
    # no validation split is read/replayed, and no sealed test capability exists.
    graphs = gd._load_split(manifest, "train", guard, gr.Snapshot)
    if len(graphs) != 2560:
        raise CanaryError("expected 2560 chronological training snapshots")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    guard()
    recurrent = modules.wrapper_type(1, modules.parameter_type(100, 128, 2)).to(device=device, dtype=torch.float32)
    guard()
    head = modules.head_type(128, num_layers=2, p_dropout=0.2).to(device=device, dtype=torch.float32)
    models = {"node_emb": recurrent, "link_pred": head}
    named = [(f"{component}.{name}", parameter)
             for component in ("node_emb", "link_pred")
             for name, parameter in sorted(models[component].named_parameters())]
    if len({id(p) for _, p in named}) != len(named):
        raise CanaryError("aliased optimizer parameters")
    guard()
    optimizer = torch.optim.Adam([p for _, p in named], lr=1e-4,
                                 betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, amsgrad=False)
    backend = gr.NativeGCLSTMBackend(torch, models, optimizer, device="cpu")
    rng = np.random.default_rng(1)

    from .tgrab_figure10_attempt_journal import Figure10AttemptJournal
    from .tgrab_selection_transaction import SelectionContext
    from .tgrab_selected_checkpoint import SCHEMA_SHA256
    attempts_path = owner.run_dir / "optimizer-attempts"
    attempts_path.mkdir(mode=0o700)
    context = SelectionContext(args.run_id, "prefix01",
        os.environ["RC_SUPERVISOR_RECEIPT_SHA256"], SCHEMA_SHA256, None)
    journal = Figure10AttemptJournal(attempts_path, context)
    raw_path = owner.run_dir / "updates.jsonl"
    with raw_path.open("x", encoding="ascii") as raw:
        raw.flush(); os.fsync(raw.fileno())

    def persist_update(record):
        journal.on_update(record)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with raw_path.open("a", encoding="ascii") as stream:
            stream.write(line); stream.flush(); os.fsync(stream.fileno())
        guard()
        if record.get("completed") is not True or "loss" not in record:
            return
        reporter.log({"train/loss": record["loss"],
                      "train/target_timestamp": record["target_timestamp"],
                      "train/completed_updates": record["completed_updates"]}, step=record["attempt"] - 1)

    try:
        result = run_prefix_updates(backend=backend, graphs=graphs,
            sampler=lambda target: gr.sample_negatives(target, rng), guard=guard,
            on_attempt=journal.on_attempt, on_update=persist_update,
            update=lambda state, target, negatives: witnessed_native_update(backend, state, target, negatives))
    finally:
        journal.close()
    data_identity = {"kind": md["kind"], "raw_sha256": md["raw"]["sha256"],
                     "conversion_protocol_sha256": md["converter"]["sha256"],
                     "train_split_sha256": md["splits"]["train"]["sha256"],
                     "train_split_semantic_sha256": md["splits"]["train"]["semantic_sha256"],
                     "train_split_interval": md["splits"]["train"]["interval"]}
    return result, raw_path, manifest.sha256, modules.provenance_dict(), named, data_identity


def validate_supervisor_runtime(receipt: dict, *, runtime_seconds: int,
                                deadline_unix: float, environ: dict) -> None:
    """Use supervisor-attested scheduler fields; never SBATCH request fallbacks."""
    job = receipt.get("job", {})
    mapping = {"job_id": "SLURM_JOB_ID", "account": "SLURM_JOB_ACCOUNT",
               "partition": "SLURM_JOB_PARTITION", "cpus_on_node": "SLURM_CPUS_ON_NODE"}
    for field, variable in mapping.items():
        value = environ.get(variable)
        if not job.get(field) or (field == "job_id" and not value) or (value is not None and str(job.get(field)) != str(value)):
            raise CanaryError("supervisor receipt Slurm identity mismatch: " + field)
    started = receipt.get("wall_started_epoch_ns")
    if (type(runtime_seconds) is not int or runtime_seconds != 540
            or type(started) is not int or started <= 0
            or receipt.get("work_seconds") != runtime_seconds
            or receipt.get("deadline_unix") != deadline_unix
            or deadline_unix != started / 1e9 + runtime_seconds):
        raise CanaryError("supervisor runtime-relative deadline mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--wheel-manifest", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--node2a-receipt", type=Path, required=True)
    parser.add_argument("--node2a-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cell-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--runtime-seconds", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        parser.error("invalid run ID")
    if args.config_sha256 != CONFIG_SHA256:
        raise CanaryError("only the exact frozen reconstruction config is eligible")
    config_bytes = read_pinned_bytes(args.config, args.config_sha256, max_bytes=2_000_000)
    verify_pin(args.environment_lock, ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
    verify_pin(args.wheel_manifest, WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
    environment_receipt_bytes = read_pinned_bytes(args.environment_receipt, args.environment_receipt_sha256, max_bytes=1_000_000)
    node2a_receipt_bytes = read_pinned_bytes(args.node2a_receipt, args.node2a_receipt_sha256, max_bytes=1_000_000)
    validate_node2a_receipt(json.loads(node2a_receipt_bytes.decode("ascii")))
    verify_pin(args.data_manifest, args.data_manifest_sha256, max_bytes=2_000_000)
    for relative, expected in WRAPPER_SHA256.items():
        verify_pin(args.source_root / relative, expected, max_bytes=64_000)
    module_dir = Path(__file__).resolve().parent
    for name, expected in (("graph_data.py", GRAPH_DATA_SHA256),
                           ("graph_reconstruction.py", GRAPH_RECONSTRUCTION_SHA256),
                           ("pinned_graph_import.py", IMPORT_HELPER_SHA256),
                           ("reporting.py", REPORTER_SHA256)):
        verify_pin(module_dir / name, expected, max_bytes=100_000)
    versions = environment_versions()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time():
        raise CanaryError("deadline already expired")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    supervisor_receipt_path = os.environ.get("RC_SUPERVISOR_RECEIPT")
    supervisor_receipt_hash = os.environ.get("RC_SUPERVISOR_RECEIPT_SHA256")
    if not supervisor_receipt_path or not supervisor_receipt_hash:
        raise CanaryError("must be started by the admitted stdlib supervisor")
    supervisor_receipt_bytes = read_pinned_bytes(Path(supervisor_receipt_path), supervisor_receipt_hash, max_bytes=1_000_000)
    supervisor_receipt = json.loads(supervisor_receipt_bytes.decode("ascii"))
    if (supervisor_receipt.get("schema") != "tgrab-figure10-supervisor-receipt-v2"
            or supervisor_receipt.get("protocol_id") != PROTOCOL_ID):
        raise CanaryError("unsupported supervisor receipt schema")
    if supervisor_receipt.get("source_sha256") != source_hash:
        raise CanaryError("executing source differs from admitted immutable source SHA")
    if supervisor_receipt.get("run_id") != args.run_id or supervisor_receipt.get("job_id") != os.environ.get("SLURM_JOB_ID"):
        raise CanaryError("supervisor receipt run/job identity mismatch")
    if supervisor_receipt.get("canary_args") != list(sys.argv[1:]):
        raise CanaryError("executed argv differs from external admission")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise CanaryError(f"single-thread environment must be set before Python start: {key}")
    job = supervisor_receipt.get("job", {})
    validate_supervisor_runtime(supervisor_receipt, runtime_seconds=args.runtime_seconds,
                                deadline_unix=args.deadline_unix, environ=os.environ)
    if supervisor_receipt.get("environment_receipt", {}).get("path") != str(args.environment_receipt):
        raise CanaryError("environment preparation receipt path differs from admission")
    if supervisor_receipt.get("environment_receipt", {}).get("sha256") != args.environment_receipt_sha256:
        raise CanaryError("environment preparation receipt hash differs from admission")
    if supervisor_receipt.get("node2a_receipt", {}).get("path") != str(args.node2a_receipt):
        raise CanaryError("node2a pass receipt path differs from admission")
    if supervisor_receipt.get("node2a_receipt", {}).get("sha256") != args.node2a_receipt_sha256:
        raise CanaryError("node2a pass receipt hash differs from admission")
    if supervisor_receipt.get("sys_executable") != sys.executable:
        raise CanaryError("interpreter executable differs from supervisor receipt")
    child_argv = supervisor_receipt.get("child_argv")
    if (type(child_argv) is not list or not child_argv or child_argv[0] != sys.executable
            or child_argv[1:3] != ["-B", "-c"] or child_argv[-len(sys.argv[1:]):] != list(sys.argv[1:])):
        raise CanaryError("executed bootstrap command differs from admitted argv")
    argv_hash = hashlib.sha256(json.dumps(child_argv, separators=(",", ":")).encode()).hexdigest()
    if supervisor_receipt.get("argv_sha256") != argv_hash:
        raise CanaryError("executing argv differs from admitted command identity")
    owner = ProcessOwner.acquire(args.run_dir)
    worker_started = time.monotonic()
    guard = CanaryGuard(args.stop_file, args.deadline_unix, owner,
                        interrupted=False)
    original_guard = guard
    stop_requested = threading.Event()
    def checked_guard():
        original_guard.interrupted = stop_requested.is_set()
        original_guard()
    watch_stop = threading.Event()
    def stop_watcher():
        while not watch_stop.wait(0.2):
            if os.path.lexists(args.stop_file):
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return
    watcher = threading.Thread(target=stop_watcher, name="tgrab-stop-watch", daemon=True)
    data_hash = args.data_manifest_sha256
    config_hash = args.config_sha256
    report_config = {"protocol": PROTOCOL_ID,
                     "updates": MAX_UPDATES, "target_interval": "[1,257)",
                     "model_seed": 0, "negative_sampling_seed": 1,
                     "device": "cpu", "precision": "float32",
                     "validation_replay": False, "test_access": False}
    identities = {"config_sha256": config_hash, "data_manifest_sha256": data_hash,
                  "environment_receipt_sha256": args.environment_receipt_sha256,
                  "node2a_receipt_sha256": args.node2a_receipt_sha256,
                  "source_sha256": source_hash, "environment_lock_sha256": ENVIRONMENT_LOCK_SHA256,
                  "wheel_manifest_sha256": WHEEL_MANIFEST_SHA256,
                  "graph_data_sha256": GRAPH_DATA_SHA256,
                  "graph_reconstruction_sha256": GRAPH_RECONSTRUCTION_SHA256,
                  "import_helper_sha256": IMPORT_HELPER_SHA256, "reporter_sha256": REPORTER_SHA256}
    final = {"kind": PROTOCOL_ID, "status": "failed",
             "interpretation": "prospective partial-prefix canary; not historical replay or Figure 10 reproduction",
            "scientific_acceptance": False, "max_updates": MAX_UPDATES,
             "complete_epoch_updates": 2559, "competence_fit": False,
             "figure10_reproduction": False,
             "attempted_updates": 0, "completed_updates": 0,
             "optimizer_mutation_unknown_attempts": 0, "committed_updates": 0,
             "validation_replayed": False, "selection_performed": False,
             "checkpoint_written": False, "test_payload_opened": False,
             "process": {"pid": os.getpid(), "host": socket.gethostname(),
                         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "owner_token": owner.token},
             "supervisor_receipt_sha256": supervisor_receipt_hash,
             "execution_scope": "prefix-only runtime diagnostic; no fit-sizing or readiness claim",
             "fit_sizing_use": False, "readiness_claim": False,
             "identities": identities, "versions": versions,
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    def terminate(signum, frame):
        stop_requested.set()
    def deadline_alarm(signum, frame):
        raise CanaryError("hard deadline reached")
    try:
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, args.deadline_unix - time.time()))
        watcher.start()
        checked_guard()
        # Reporter module is stdlib-only; source/control identities were already
        # pinned, but no data arrays, weights, checkpoints or numerical payloads
        # are hashed/opened before entering the mandatory offline reporter.
        # Torch, NumPy and training split payload remain lazy until _runtime_train.
        from .reporting import ExperimentReporter
        reporter = ExperimentReporter(owner.run_dir / "wandb", project="representation-collapse",
            config=report_config, identities=identities, mode="offline", run_id=args.run_id)
        with reporter:
            checked_guard()
            verify_pin(args.data_manifest, data_hash, max_bytes=2_000_000)
            verify_pin(args.environment_lock, ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
            verify_pin(args.wheel_manifest, WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
            receipt = json.loads(environment_receipt_bytes.decode("ascii"))
            validate_environment_receipt(receipt)
            result, raw_path, verified_data_sha, import_provenance, named, data_identity = _runtime_train(
                args, owner, checked_guard, reporter, config_bytes)
            final.update(result)
            final["status"] = "completed_256_updates"
            final["raw_updates"] = {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}
            final["verified_data_manifest_sha256"] = verified_data_sha
            final["data_identity"] = data_identity
            final["import_provenance"] = import_provenance
            final["optimizer_parameter_names"] = [name for name, _ in named]
            reporter.log_manifest("canary-result", metadata={
                "status": final["status"], "attempted_updates": final["attempted_updates"],
                "completed_updates": final["completed_updates"], "mean_loss": final["mean_loss"],
                "validation_replayed": False, "test_payload_opened": False,
            }, identities=identities)
    except BaseException as exc:
        final["status"] = "failed" if not isinstance(exc, (KeyboardInterrupt, SystemExit)) else "interrupted"
        final["error_type"] = type(exc).__name__
        from .tgrab_figure10_attempt_journal import AttemptJournalError
        final.update(reconcile_attempts(owner.run_dir / "optimizer-attempts",
            terminal_journal_error=isinstance(exc, AttemptJournalError)))
        if (owner.run_dir / "updates.jsonl").exists():
            raw_path = owner.run_dir / "updates.jsonl"
            final["raw_updates"] = {"path": raw_path.name, "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            final["status"] = "interrupted"
    finally:
        watch_stop.set()
        if watcher.ident is not None:
            watcher.join(timeout=1)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, previous_handler)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        final["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        final["elapsed_total_seconds"] = time.monotonic() - worker_started
        final["deadline_unix"] = args.deadline_unix
        final["stop_file"] = str(args.stop_file)
        final["environment_receipt_sha256"] = args.environment_receipt_sha256
        reporter_path = owner.run_dir / "wandb" / "reporting-receipt.json"
        if reporter_path.exists():
            final["reporting_receipt"] = {"path": "wandb/reporting-receipt.json",
                                           "sha256": sha256_file(reporter_path)}
        publish_terminal_manifest(owner, final)
    return 0 if final["status"] == "completed_256_updates" else 2


if __name__ == "__main__":
    raise SystemExit(main())
