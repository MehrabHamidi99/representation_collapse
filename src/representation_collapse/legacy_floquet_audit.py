"""Bounded descriptive audit of hash-pinned legacy .npy assets.

No model, checkpoint, Jacobian, dataset, periodicity or task assay is read.
The production entry point requires the SDK-0.30.0 offline reporter before
opening any array. Pure readers/reducers support tiny synthetic unit fixtures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import signal
import stat
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np

ATOL = 1e-6
RTOL = 1e-5
CHUNK_ROWS = 1024
MAX_HEADER = 10000
MAX_MANIFEST = 65536
PROTOCOL_ID = "legacy-floquet-array-audit-v1"
WANDB_ENTITY = "mehrab-hamidi-milaquebec5431"
UNAVAILABLE = ["producer/checkpoint/data identity", "Jacobian eigenpair residuals",
               "trajectory and Jacobian closure", "Floquet certification",
               "per-step exponents/frequencies", "task performance",
               "covariance dimension", "causal memory"]


class AuditError(ValueError):
    """Integrity or schema failure; retain the failure receipt."""


class AuditDeadline(TimeoutError):
    """The numerical work deadline has expired."""


def atomic_json(path: Path, payload: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(stream, guard: Callable[[], None]) -> str:
    result = hashlib.sha256()
    while True:
        guard()
        block = stream.read(1024 * 1024)
        if not block:
            break
        result.update(block)
    return result.hexdigest()


def _noop() -> None:
    pass


def stat_identity(value: os.stat_result) -> dict:
    return {key: int(getattr(value, "st_" + key))
            for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns", "mode", "nlink")}


def _regular(value: os.stat_result) -> None:
    if not stat.S_ISREG(value.st_mode):
        raise AuditError("only regular files are admitted")


def _safe_root(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    for item in [root, *root.parents]:
        if item.is_symlink():
            raise AuditError("symlink archive root or ancestor")
    if not root.is_dir():
        raise AuditError("archive root is not a directory")
    return root


def expected_files(width: int = 25600) -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {"eigvals.npy": ("<c8", (6,)), "eigvecs.npy": ("<c8", (width, 6))}
    for index in range(19):
        result[f"envelope_{index}.npy"] = ("<c8", (20, width, 6))
        result[f"norm_env_{index}.npy"] = ("<f4", (20, 6))
    return result


def validate_manifest(manifest: dict, *, fixture_width: int | None = None) -> dict[str, dict]:
    """Production schema is fixed; fixture mode only admits widths 1..8."""
    width = 25600
    if fixture_width is not None:
        if type(fixture_width) is not int or not 1 <= fixture_width <= 8:
            raise AuditError("fixture width must be an integer in 1..8")
        width = fixture_width
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            manifest.get("kind") != "legacy_floquet_array_manifest"):
        raise AuditError("invalid manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 40:
        raise AuditError("exactly forty admitted arrays are required")
    expected = expected_files(width)
    result = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "dtype", "shape"}:
            raise AuditError("invalid array manifest entry")
        name = entry["path"]
        if (not isinstance(name, str) or PurePosixPath(name).name != name or
                name not in expected or "\\" in name or name in result):
            raise AuditError("unsafe, excluded or duplicate array path")
        dtype, shape = expected[name]
        if (entry["dtype"] != dtype or entry["shape"] != list(shape) or
                not isinstance(entry["shape"], list) or
                any(type(item) is not int for item in entry["shape"])):
            raise AuditError("unexpected dtype or shape")
        if type(entry["bytes"]) is not int or not math.prod(shape) * np.dtype(dtype).itemsize < entry["bytes"] <= math.prod(shape) * np.dtype(dtype).itemsize + MAX_HEADER + 12:
            raise AuditError("unexpected exact byte size")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise AuditError("invalid SHA256")
        result[name] = dict(entry)
    if set(result) != set(expected):
        raise AuditError("incomplete array whitelist")
    return result


@contextmanager
def verified_array(root: Path, entry: dict, *, guard: Callable[[], None] = _noop):
    """Hash, inspect and mmap one fixed numeric file through a stable descriptor.

    On Linux np.load uses allow_pickle=False and a descriptor alias, so an archive-path
    replacement cannot redirect its second open. Stat checks surround all work.
    macOS maps the safe numeric descriptor directly, without a pickle path.
    The /proc alias is internal; symlinks in archive paths are refused.
    """
    root = _safe_root(root)
    name = entry.get("path")
    if not isinstance(name, str) or name not in expected_files() or PurePosixPath(name).name != name:
        raise AuditError("excluded or unsafe array path")
    path = root / name
    guard()
    before = path.lstat()
    _regular(before)
    if before.st_size != entry["bytes"]:
        raise AuditError("exact byte size mismatch")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    array = None
    snapshots = {"before": stat_identity(before)}
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            if stat_identity(os.fstat(stream.fileno())) != snapshots["before"]:
                raise AuditError("file changed during open")
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=MAX_HEADER)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=MAX_HEADER)
            else:
                raise AuditError("only bounded npy version 1/2 headers admitted")
            offset = stream.tell()
            if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or
                    dtype.kind not in "fc" or dtype.str != entry["dtype"] or
                    list(shape) != entry["shape"]):
                raise AuditError("unsafe or mismatched npy header")
            if offset + math.prod(shape) * dtype.itemsize != before.st_size:
                raise AuditError("truncated payload or appended bytes")
            stream.seek(0)
            if _digest(stream, guard) != entry["sha256"]:
                raise AuditError("SHA256 mismatch")
            if stat_identity(os.fstat(stream.fileno())) != snapshots["before"]:
                raise AuditError("file changed during hash")
            guard()
            if Path("/proc/self/fd").is_dir():
                # Linux descriptor aliases independently reopen at offset zero.
                array = np.load(Path("/proc/self/fd") / str(stream.fileno()), allow_pickle=False,
                                mmap_mode="r", max_header_size=MAX_HEADER)
            else:
                # macOS /dev/fd shares offsets, incompatible with np.load's
                # two opens. Map the already validated numeric descriptor;
                # memmap has no pickle/deserialization path at all.
                array = np.memmap(stream, dtype=dtype, mode="r", offset=offset,
                                  shape=shape, order="F" if fortran else "C")
            if array.dtype != dtype or array.shape != shape:
                raise AuditError("mapped header mismatch")
            yield array, snapshots
            guard()
            snapshots["after"] = stat_identity(os.fstat(stream.fileno()))
            snapshots["path_after"] = stat_identity(path.lstat())
            if snapshots["before"] != snapshots["after"] or snapshots["before"] != snapshots["path_after"]:
                raise AuditError("file changed during numerical read")
    finally:
        if array is not None:
            array._mmap.close()
        if descriptor >= 0:
            os.close(descriptor)


def _blocks(array: np.ndarray, guard: Callable[[], None]):
    """Only slices of at most1024 state rows, never flatten a whole envelope."""
    if array.ndim == 3:
        for phase in range(array.shape[0]):
            for start in range(0, array.shape[1], CHUNK_ROWS):
                guard()
                yield array[phase, start:start + CHUNK_ROWS, :]
    else:
        for start in range(0, array.shape[0], CHUNK_ROWS):
            guard()
            yield array[start:start + CHUNK_ROWS]


def finite_check(array: np.ndarray, *, guard: Callable[[], None] = _noop) -> None:
    for block in _blocks(array, guard):
        if not np.isfinite(block).all():
            raise AuditError("nonfinite array values")


def eigenvalue_summary(array: np.ndarray) -> dict:
    values = array.astype(np.complex128)
    return {"real": values.real.tolist(), "imag": values.imag.tolist(),
            "magnitude": np.abs(values).tolist(), "phase_radians_per_stored_product": np.angle(values).tolist(),
            "interpretation": "stored partial spectrum; product chronology and periodicity unverified"}


def eigenvector_summary(array: np.ndarray, *, guard: Callable[[], None] = _noop) -> dict:
    gram = np.zeros((6, 6), dtype=np.complex128)
    for block in _blocks(array, guard):
        selected = block.astype(np.complex128)
        gram += selected.conj().T @ selected
    return {"column_norms": np.sqrt(np.maximum(gram.diagonal().real, 0)).tolist(),
            "gram_real": gram.real.tolist(), "gram_imag": gram.imag.tolist(),
            "interpretation": "descriptive Euclidean norms and unnormalized Hermitian Gram; no residual or orthogonality criterion"}


def envelope_summary(array: np.ndarray, reference: np.ndarray, *, guard: Callable[[], None] = _noop) -> dict:
    """State-axis L2 reduction uses float64; reference directs tolerances."""
    norms = np.zeros((array.shape[0], array.shape[2]), dtype=np.float64)
    equal = np.ones(array.shape[0], dtype=bool)
    max_time_difference = np.zeros(array.shape[0], dtype=np.float64)
    for phase in range(array.shape[0]):
        for start in range(0, array.shape[1], CHUNK_ROWS):
            guard()
            block = array[phase, start:start + CHUNK_ROWS, :].astype(np.complex128)
            base = array[0, start:start + CHUNK_ROWS, :]
            equal[phase] &= array[phase, start:start + CHUNK_ROWS, :].tobytes(order="C") == base.tobytes(order="C")
            max_time_difference[phase] = max(max_time_difference[phase], float(np.max(np.abs(block - base))))
            norms[phase] += np.sum(block.real * block.real + block.imag * block.imag, axis=0)
    np.sqrt(norms, out=norms)
    saved = reference.astype(np.float64)
    error = np.abs(norms - saved)
    tolerance = ATOL + RTOL * np.abs(saved)
    failures = np.argwhere(error > tolerance)
    return {"all_time_rows_exactly_equal_to_row0": bool(equal.all()),
            "exact_equality_convention": "full complex entry byte identity, preserving signed zero",
            "time_rows_exactly_equal_to_row0": equal.tolist(),
            "max_absolute_time_difference_from_row0": max_time_difference.tolist(),
            "norm_axis": 1, "norm_accumulation_dtype": "float64", "norm_atol": ATOL, "norm_rtol": RTOL,
            "norm_reference": "saved norm_env; directed atol + rtol * abs(saved)",
            "recomputed_norms": norms.tolist(), "saved_norms": saved.tolist(),
            "norm_absolute_errors": error.tolist(), "norm_tolerances": tolerance.tolist(),
            "norm_comparison_passed": len(failures) == 0,
            "norm_failure_count": len(failures), "norm_failure_indices": failures.tolist()}


def prefix_difference(left: np.ndarray, right: np.ndarray, *, guard: Callable[[], None] = _noop) -> dict:
    changed = 0
    bitwise_changed = 0
    squared = 0.0
    maximum = 0.0
    count = 0
    for first, second in zip(_blocks(left, guard), _blocks(right, guard)):
        difference = first.astype(np.complex128) - second.astype(np.complex128)
        changed += int(np.count_nonzero(first != second))
        first_bytes = np.frombuffer(first.tobytes(order="C"), dtype=np.uint8).reshape(-1, first.dtype.itemsize)
        second_bytes = np.frombuffer(second.tobytes(order="C"), dtype=np.uint8).reshape(-1, second.dtype.itemsize)
        bitwise_changed += int(np.count_nonzero(np.any(first_bytes != second_bytes, axis=1)))
        absolute = np.abs(difference)
        squared += float(np.sum(absolute * absolute))
        maximum = max(maximum, float(np.max(absolute)))
        count += int(first.size)
    return {"exactly_equal": bitwise_changed == 0, "numeric_changed_scalar_entries": changed,
            "bitwise_changed_scalar_entries": bitwise_changed,
            "changed_scalar_entries": bitwise_changed,
            "exact_equality_convention": "full scalar entry byte identity, preserving signed zero",
            "scalar_entries": count, "max_absolute_difference": maximum,
            "l2_difference": math.sqrt(squared), "comparison": "adjacent stored prefix files; no chronological derivative claim"}


def audit_arrays(root: Path, manifest: dict, *, fixture_width: int | None = None,
                 guard: Callable[[], None] = _noop, progress: Callable[[dict], None] | None = None) -> dict:
    entries = validate_manifest(manifest, fixture_width=fixture_width)
    root = _safe_root(root)
    result = {"schema_version": 1, "kind": "legacy_floquet_descriptive_array_audit",
              "scientific_acceptance": False, "fixture_only": fixture_width is not None,
              "unavailable": UNAVAILABLE, "file_stats": {}, "envelopes": {}, "prefix_differences": {}}
    for name, reducer, key in [("eigvals.npy", eigenvalue_summary, "stored_eigenvalues"),
                               ("eigvecs.npy", eigenvector_summary, "stored_eigenvectors")]:
        with verified_array(root, entries[name], guard=guard) as (array, snapshots):
            finite_check(array, guard=guard)
            result[key] = reducer(array) if name == "eigvals.npy" else reducer(array, guard=guard)
        result["file_stats"][name] = snapshots
    for index in range(19):
        name, saved_name = f"envelope_{index}.npy", f"norm_env_{index}.npy"
        with verified_array(root, entries[name], guard=guard) as (array, snapshots):
            with verified_array(root, entries[saved_name], guard=guard) as (reference, saved_snapshots):
                finite_check(array, guard=guard)
                finite_check(reference, guard=guard)
                result["envelopes"][str(index)] = envelope_summary(array, reference, guard=guard)
                if index > 0:
                    previous = f"envelope_{index - 1}.npy"
                    with verified_array(root, entries[previous], guard=guard) as (prior, prior_stats):
                        result["prefix_differences"][f"{index - 1}:{index}"] = prefix_difference(prior, array, guard=guard)
                    if prior_stats["before"] != result["file_stats"][previous]["before"]:
                        raise AuditError("prefix changed between accesses")
                    previous_norm = f"norm_env_{index - 1}.npy"
                    with verified_array(root, entries[previous_norm], guard=guard) as (prior_norm, prior_norm_stats):
                        result["prefix_differences"][f"norm_{index - 1}:{index}"] = prefix_difference(prior_norm, reference, guard=guard)
                    if prior_norm_stats["before"] != result["file_stats"][previous_norm]["before"]:
                        raise AuditError("saved norm changed between accesses")
        result["file_stats"][name], result["file_stats"][saved_name] = snapshots, saved_snapshots
        if progress is not None:
            progress(result)
    for name, frozen in result["file_stats"].items():
        guard()
        final = stat_identity((root / name).lstat())
        if final != frozen["before"]:
            raise AuditError("file changed before audit completion")
        frozen["final"] = final
    result["all_norm_comparisons_passed"] = all(item["norm_comparison_passed"] for item in result["envelopes"].values())
    result["integrity_completed"] = True
    return result


def _pinned_json(path: Path, digest: str) -> dict:
    with path.open("rb") as stream:
        value = stream.read(MAX_MANIFEST + 1)
    if len(value) > MAX_MANIFEST or hashlib.sha256(value).hexdigest() != digest:
        raise AuditError("metadata pin or size mismatch")
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AuditError("duplicate metadata keys")
            result[key] = value
        return result
    return json.loads(value, object_pairs_hook=no_duplicates)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive-root", "manifest", "manifest-sha256", "config", "config-sha256", "output",
                "source-sha256", "launcher-sha256", "source-commit"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--work-deadline-monotonic", type=float, required=True)
    parser.add_argument("--started-monotonic", type=float, required=True)
    parser.add_argument("--owned-process-journal", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "state": "initializing", "scientific_acceptance": False,
               "started_at_utc": datetime.now(timezone.utc).isoformat(), "array_work_started": False,
               "unavailable": UNAVAILABLE}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - args.started_monotonic
        atomic_json(output / "audit-receipt.json", receipt)
    def guard():
        if time.monotonic() >= args.work_deadline_monotonic:
            raise AuditDeadline("numerical deadline expired")
    def deadline_signal(signum, frame):
        raise AuditDeadline("numerical deadline expired")
    persist()
    try:
        guard()
        remaining = args.work_deadline_monotonic - time.monotonic()
        if not 0 < remaining <= 590 or args.work_deadline_monotonic - args.started_monotonic > 590:
            raise AuditError("invalid shared numerical deadline")
        signal.signal(signal.SIGALRM, deadline_signal)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        from .reporting import ExperimentReporter, file_sha256
        repo = Path(__file__).resolve().parents[2]
        source = file_sha256(Path(__file__))
        launcher = file_sha256(repo / "code/ops/legacy_floquet_audit.py")
        if source != args.source_sha256 or launcher != args.launcher_sha256:
            raise AuditError("source pin mismatch")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True)
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit) or head != args.source_commit or dirty:
            raise AuditError("exact clean tracked source commit required")
        manifest = _pinned_json(Path(args.manifest), args.manifest_sha256)
        config = _pinned_json(Path(args.config), args.config_sha256)
        if config.get("protocol_id") != PROTOCOL_ID or config.get("norm_atol") != ATOL or config.get("norm_rtol") != RTOL:
            raise AuditError("frozen prospective config required")
        validate_manifest(manifest)
        identities = {"source_sha256": source, "launcher_sha256": launcher, "source_commit": head,
                      "reporter_sha256": file_sha256(repo / "src/representation_collapse/reporting.py"),
                      "lock_sha256": file_sha256(repo / "uv.lock"), "config_sha256": args.config_sha256,
                      "manifest_sha256": args.manifest_sha256}
        receipt["identities"] = identities
        receipt["thresholds"] = {"norm_atol": ATOL, "norm_rtol": RTOL}
        persist()
        # Install before importing the SDK. Its detached core's identity is
        # registered immediately after spawn, including failure before a poll.
        launch_helpers = runpy.run_path(str(repo / "code/ops/legacy_floquet_audit.py"))
        launch_helpers["install_spawn_journal"](Path(args.owned_process_journal))
        with ExperimentReporter(output / "reporting", project="representation-collapse", entity=WANDB_ENTITY,
                                mode="offline", config={"protocol_id": PROTOCOL_ID, "cpu_threads": 1,
                                "norm_atol": ATOL, "norm_rtol": RTOL, "array_count": 40}, identities=identities) as report:
            guard()
            receipt.update(state="running", array_work_started=True)
            persist()
            step = 0
            def progress(partial):
                nonlocal step
                guard()
                atomic_json(output / "audit-progress.json", partial)
                report.log({"audit/prefixes_completed": len(partial["envelopes"]),
                            "audit/norm_failures": sum(item["norm_failure_count"] for item in partial["envelopes"].values()),
                            "audit/elapsed_seconds": time.monotonic() - args.started_monotonic}, step=step)
                step += 1
            result = audit_arrays(Path(args.archive_root), manifest, guard=guard, progress=progress)
            guard()
            result["identities"] = identities
            atomic_json(output / "audit.json", result)
            report.log({"audit/integrity_completed": True,
                        "audit/all_norm_comparisons_passed": result["all_norm_comparisons_passed"],
                        "audit/scientific_acceptance": False}, step=step)
            signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(state="completed", array_integrity_completed=True,
                       all_norm_comparisons_passed=result["all_norm_comparisons_passed"])
        persist()
        return 0
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(state="failed", error_type=type(error).__name__)
        persist()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
