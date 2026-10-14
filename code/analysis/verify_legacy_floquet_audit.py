#!/usr/bin/env python3
"""Independent, admitted reconciliation of stored legacy array summaries.

Only the reviewed supervisor/process helpers and reporter are shared with the
producer. No producer reader or numerical reduction is imported. The CLI has
no fixture-width option, retries, model execution, or cluster operations.
"""
from __future__ import annotations

import time

_PROCESS_STARTED_MONOTONIC = time.monotonic()

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import stat
import subprocess
import sys
import tempfile

import numpy as np

ROWS = 257
NORM_ATOL, NORM_RTOL = 1e-6, 1e-5
RECON_ATOL, RECON_RTOL = 1e-12, 1e-11
MAX_HEADER, MAX_JSON = 10000, 2 * 1024 * 1024
PRODUCER_COMMIT = "351fa688f3333f0e44701d15d2acf9a2553a4b98"
HELPER_SHA = "8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09"
REPORTER_SHA = "443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184"
LOCK_SHA = "85304e351feacf6d73326996a202ddc1a7470f3fb5a55b2e212b9747e8ace996"
UNAVAILABLE = ["producer/checkpoint/data linkage", "full-Jacobian eigenpair residuals",
               "orbit/Jacobian closure", "Floquet certification", "per-step exponents/frequencies",
               "task performance", "covariance dimension", "causal memory"]


class ReconciliationError(ValueError):
    pass


def atomic_json(path, payload):
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def no_op():
    pass


def sha_file(path, guard=no_op):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            guard()
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return {field: int(getattr(value, "st_" + field)) for field in
            ("dev", "ino", "size", "mtime_ns", "ctime_ns", "mode", "nlink")}


def safe_root(root):
    root = Path(os.path.abspath(root))
    if any(item.is_symlink() for item in [root, *root.parents]) or not root.is_dir():
        raise ReconciliationError("unsafe input directory")
    return root


def pinned_json(path, digest=None):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON:
        raise ReconciliationError("unsafe compact metadata")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        if identity(os.fstat(stream.fileno())) != identity(before):
            raise ReconciliationError("metadata replaced during open")
        raw = stream.read(MAX_JSON + 1)
        if len(raw) > MAX_JSON or identity(os.fstat(stream.fileno())) != identity(before):
            raise ReconciliationError("metadata changed or exceeded bound")
    if identity(path.lstat()) != identity(before):
        raise ReconciliationError("metadata path changed")
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ReconciliationError("metadata digest mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReconciliationError("duplicate metadata key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ReconciliationError("nonfinite compact metadata")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def schemas(width=25600):
    result = {"eigvals.npy": ("<c8", (6,)), "eigvecs.npy": ("<c8", (width, 6))}
    for prefix in range(19):
        result[f"envelope_{prefix}.npy"] = ("<c8", (20, width, 6))
        result[f"norm_env_{prefix}.npy"] = ("<f4", (20, 6))
    return result


def manifest_entries(manifest, *, fixture_width=None):
    if fixture_width is not None and (type(fixture_width) is not int or not 1 <= fixture_width <= 8):
        raise ReconciliationError("invalid tiny fixture width")
    expected = schemas(25600 if fixture_width is None else fixture_width)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or
            manifest.get("kind") != "legacy_floquet_array_manifest" or
            not isinstance(manifest.get("files"), list) or len(manifest["files"]) != 40):
        raise ReconciliationError("invalid fixed manifest")
    entries = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "dtype", "shape"}:
            raise ReconciliationError("invalid manifest entry")
        name = entry["path"]
        if not isinstance(name, str) or name not in expected or name in entries:
            raise ReconciliationError("excluded, escaping or duplicate filename")
        dtype, shape = expected[name]
        if entry["dtype"] != dtype or entry["shape"] != list(shape) or any(type(v) is not int for v in entry["shape"]):
            raise ReconciliationError("wrong fixed numeric schema")
        payload_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        if type(entry["bytes"]) is not int or not payload_bytes < entry["bytes"] <= payload_bytes + MAX_HEADER + 12:
            raise ReconciliationError("invalid exact file length")
        if not isinstance(entry["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", entry["sha256"]):
            raise ReconciliationError("invalid input digest")
        entries[name] = dict(entry)
    if set(entries) != set(expected):
        raise ReconciliationError("incomplete whitelist")
    if fixture_width is None and sum(v["bytes"] for v in entries.values()) != 468187088:
        raise ReconciliationError("wrong frozen acquisition byte total")
    return entries


@contextmanager
def numeric_mmap(root, entry, guard=no_op):
    """Safe numeric header plus hash/stat stability; no pickle path on macOS."""
    root = safe_root(root)
    if entry["path"] not in schemas():
        raise ReconciliationError("excluded array")
    path = root / entry["path"]
    guard()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != entry["bytes"]:
        raise ReconciliationError("unsafe array or exact length mismatch")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    mapped = None
    evidence = {"before": identity(before), "sha256": entry["sha256"]}
    try:
        with os.fdopen(fd, "rb") as stream:
            if identity(os.fstat(stream.fileno())) != evidence["before"]:
                raise ReconciliationError("array replaced during open")
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=MAX_HEADER)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=MAX_HEADER)
            else:
                raise ReconciliationError("unsupported bounded numeric header")
            offset = stream.tell()
            if (dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None or
                    dtype.str != entry["dtype"] or list(shape) != entry["shape"]):
                raise ReconciliationError("unsafe or mismatched array header")
            if offset + math.prod(shape) * dtype.itemsize != before.st_size:
                raise ReconciliationError("truncated or appended input")
            stream.seek(0)
            digest = hashlib.sha256()
            while True:
                guard()
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != entry["sha256"] or identity(os.fstat(stream.fileno())) != evidence["before"]:
                raise ReconciliationError("array hash or stability mismatch")
            guard()
            if Path("/proc/self/fd").is_dir():
                mapped = np.load(Path("/proc/self/fd") / str(stream.fileno()),
                                 allow_pickle=False, mmap_mode="r", max_header_size=MAX_HEADER)
                evidence["mapping"] = "allow_pickle_false_proc_descriptor"
            else:
                # /dev/fd shares seek offsets on Darwin. Header validation above
                # excludes object/structured types before direct numeric mapping.
                mapped = np.memmap(stream, dtype=dtype, mode="r", offset=offset,
                                   shape=shape, order="F" if fortran else "C")
                evidence["mapping"] = "safe_numeric_descriptor_no_pickle_path"
            if mapped.shape != shape or mapped.dtype != dtype:
                raise ReconciliationError("mapped schema changed")
            yield mapped, evidence
            guard()
            evidence["after"] = identity(os.fstat(stream.fileno()))
            evidence["path_after"] = identity(path.lstat())
            if evidence["after"] != evidence["before"] or evidence["path_after"] != evidence["before"]:
                raise ReconciliationError("array changed during reduction")
    finally:
        if mapped is not None:
            mapped._mmap.close()


def pieces(array, guard=no_op):
    if array.ndim == 3:
        for row in range(array.shape[0]):
            for start in range(0, array.shape[1], ROWS):
                guard()
                yield array[row, start:start + ROWS]
    else:
        for start in range(0, array.shape[0], ROWS):
            guard()
            yield array[start:start + ROWS]


def require_finite(array, guard=no_op):
    for part in pieces(array, guard):
        if not np.isfinite(part).all():
            raise ReconciliationError("nonfinite input")


def spectrum(values):
    real = [float(v.real) for v in values]
    imag = [float(v.imag) for v in values]
    return {"real": real, "imag": imag,
            "magnitude": [math.hypot(r, i) for r, i in zip(real, imag)],
            "phase_radians_per_stored_product": [math.atan2(i, r) for r, i in zip(real, imag)],
            "interpretation": "stored partial spectrum; product chronology and periodicity unverified"}


def gram_summary(vectors, guard=no_op):
    contributions = []
    for part in pieces(vectors, guard):
        selected = np.asarray(part, dtype=np.complex128)
        contributions.append(np.einsum("ni,nj->ij", selected.conj(), selected, optimize=False))
    gram = np.array([[complex(math.fsum(v[i, j].real for v in contributions),
                             math.fsum(v[i, j].imag for v in contributions))
                      for j in range(6)] for i in range(6)], dtype=np.complex128)
    return {"column_norms": [math.sqrt(max(0.0, float(gram[i, i].real))) for i in range(6)],
            "gram_real": gram.real.tolist(), "gram_imag": gram.imag.tolist(),
            "interpretation": "descriptive Euclidean norms and unnormalized Hermitian Gram; no residual or orthogonality criterion"}


def envelope_metrics(envelope, saved, guard=no_op):
    norms, repeated, differences = [], [], []
    for row in range(envelope.shape[0]):
        sums = [[] for _ in range(6)]
        same, maximum = True, 0.0
        for start in range(0, envelope.shape[1], ROWS):
            guard()
            part = envelope[row, start:start + ROWS]
            base = envelope[0, start:start + ROWS]
            same = same and part.tobytes(order="C") == base.tobytes(order="C")
            real, imag = np.asarray(part.real, dtype=np.float64), np.asarray(part.imag, dtype=np.float64)
            base_real, base_imag = np.asarray(base.real, dtype=np.float64), np.asarray(base.imag, dtype=np.float64)
            maximum = max(maximum, float(np.max(np.hypot(real - base_real, imag - base_imag))))
            for mode in range(6):
                sums[mode].append(float(np.dot(real[:, mode], real[:, mode]) + np.dot(imag[:, mode], imag[:, mode])))
        norms.append([math.sqrt(math.fsum(terms)) for terms in sums])
        repeated.append(same)
        differences.append(maximum)
    derived, reference = np.asarray(norms, dtype=np.float64), np.asarray(saved, dtype=np.float64)
    errors = np.abs(derived - reference)
    tolerances = NORM_ATOL + NORM_RTOL * np.abs(reference)
    failed = np.argwhere(errors > tolerances).tolist()
    return {"all_time_rows_exactly_equal_to_row0": all(repeated),
            "exact_equality_convention": "full complex entry byte identity, preserving signed zero",
            "time_rows_exactly_equal_to_row0": repeated,
            "max_absolute_time_difference_from_row0": differences,
            "norm_axis": 1, "norm_accumulation_dtype": "float64", "norm_atol": NORM_ATOL, "norm_rtol": NORM_RTOL,
            "norm_reference": "saved norm_env; directed atol + rtol * abs(saved)",
            "recomputed_norms": derived.tolist(), "saved_norms": reference.tolist(),
            "norm_absolute_errors": errors.tolist(), "norm_tolerances": tolerances.tolist(),
            "norm_comparison_passed": not failed, "norm_failure_count": len(failed), "norm_failure_indices": failed}


def adjacent_metrics(first, second, guard=no_op):
    numeric_changes = bit_changes = count = 0
    maximum, sums = 0.0, []
    for left, right in zip(pieces(first, guard), pieces(second, guard)):
        numeric_changes += int(np.count_nonzero(left != right))
        bits_left = np.ascontiguousarray(left).view(np.dtype(f"V{left.dtype.itemsize}"))
        bits_right = np.ascontiguousarray(right).view(np.dtype(f"V{right.dtype.itemsize}"))
        bit_changes += int(np.count_nonzero(bits_left != bits_right))
        real = np.asarray(left.real, dtype=np.float64) - np.asarray(right.real, dtype=np.float64)
        imag = np.asarray(left.imag, dtype=np.float64) - np.asarray(right.imag, dtype=np.float64)
        distance = np.hypot(real, imag)
        maximum = max(maximum, float(np.max(distance)))
        sums.append(float(np.sum(real * real + imag * imag, dtype=np.float64)))
        count += int(left.size)
    return {"exactly_equal": bit_changes == 0, "numeric_changed_scalar_entries": numeric_changes,
            "bitwise_changed_scalar_entries": bit_changes, "changed_scalar_entries": bit_changes,
            "scalar_entries": count, "max_absolute_difference": maximum, "l2_difference": math.sqrt(math.fsum(sums)),
            "exact_equality_convention": "full scalar entry byte identity, preserving signed zero",
            "comparison": "adjacent stored prefix files; no chronological derivative claim"}


class Comparisons:
    """Every scalar mismatch is retained; integer/boolean/schema fields are exact."""
    def __init__(self):
        self.count = 0
        self.mismatches = []
        self.max_numeric_error = 0.0

    def compare(self, path, observed, claimed, *, numeric=False):
        if isinstance(observed, dict):
            if not isinstance(claimed, dict):
                self._mismatch(path, observed, claimed, "mapping required")
                return
            for key, value in observed.items():
                self.compare(path + "/" + key, value, claimed.get(key), numeric=numeric)
            return
        if isinstance(observed, list):
            if not isinstance(claimed, list) or len(observed) != len(claimed):
                self._mismatch(path, observed, claimed, "list length/shape mismatch")
                return
            for index, value in enumerate(observed):
                self.compare(f"{path}/{index}", value, claimed[index], numeric=numeric)
            return
        self.count += 1
        if numeric:
            if type(observed) not in (int, float) or type(claimed) not in (int, float):
                self._mismatch(path, observed, claimed, "finite numeric scalar required", counted=True)
                return
            if not math.isfinite(observed) or not math.isfinite(claimed):
                self._mismatch(path, observed, claimed, "finite numeric scalar required", counted=True)
                return
            error = abs(observed - claimed)
            tolerance = RECON_ATOL + RECON_RTOL * abs(claimed)
            self.max_numeric_error = max(self.max_numeric_error, error)
            if error > tolerance:
                self._mismatch(path, observed, claimed, "roundoff reconciliation tolerance exceeded", counted=True,
                               absolute_error=error, tolerance=tolerance)
        elif type(observed) is not type(claimed) or observed != claimed:
            self._mismatch(path, observed, claimed, "exact field mismatch", counted=True)

    def _mismatch(self, path, observed, claimed, reason, *, counted=False, **diagnostics):
        if not counted:
            self.count += 1
        self.mismatches.append({"path": path, "independent": observed, "claimed": claimed,
                                "reason": reason, **diagnostics})


NUMERIC_FIELDS = {"real", "imag", "magnitude", "phase_radians_per_stored_product", "column_norms",
                  "gram_real", "gram_imag", "max_absolute_time_difference_from_row0", "recomputed_norms",
                  "saved_norms", "norm_absolute_errors", "norm_tolerances", "max_absolute_difference", "l2_difference"}


def compare_metrics(checks, path, observed, claimed):
    for key, value in observed.items():
        checks.compare(path + "/" + key, value, claimed.get(key), numeric=key in NUMERIC_FIELDS)


def reduce_inputs(root, manifest, claim=None, *, fixture_width=None, guard=no_op, progress=None):
    entries = manifest_entries(manifest, fixture_width=fixture_width)
    root = safe_root(root)
    checks = Comparisons()
    result = {"schema_version": 1, "kind": "independent_legacy_floquet_array_reconciliation",
              "scientific_acceptance": False, "fixture_only": fixture_width is not None,
              "norm_thresholds": {"atol": NORM_ATOL, "rtol": NORM_RTOL, "directed_reference": "saved float32 norms"},
              "reconciliation_thresholds": {"atol": RECON_ATOL, "rtol": RECON_RTOL, "scope": "reported floating summaries only; norm decision unchanged"},
              "chunk_state_rows": ROWS, "unavailable": UNAVAILABLE, "file_stats": {},
              "envelopes": {}, "prefix_differences": {}}
    def note(name, evidence):
        result["file_stats"][name] = evidence
        if claim is not None:
            for key in ("before", "after", "path_after", "final"):
                checks.compare("file_stats/" + name + "/" + key, evidence["before"], claim.get("file_stats", {}).get(name, {}).get(key))
    for name, key, reducer in [("eigvals.npy", "stored_eigenvalues", spectrum), ("eigvecs.npy", "stored_eigenvectors", gram_summary)]:
        with numeric_mmap(root, entries[name], guard) as (array, evidence):
            require_finite(array, guard)
            result[key] = reducer(array) if name == "eigvals.npy" else reducer(array, guard)
        note(name, evidence)
        if claim is not None:
            compare_metrics(checks, key, result[key], claim.get(key, {}))
    for prefix in range(19):
        name, saved_name = f"envelope_{prefix}.npy", f"norm_env_{prefix}.npy"
        with numeric_mmap(root, entries[name], guard) as (envelope, evidence):
            with numeric_mmap(root, entries[saved_name], guard) as (saved, saved_evidence):
                require_finite(envelope, guard)
                require_finite(saved, guard)
                result["envelopes"][str(prefix)] = envelope_metrics(envelope, saved, guard)
                if claim is not None:
                    compare_metrics(checks, f"envelopes/{prefix}", result["envelopes"][str(prefix)], claim.get("envelopes", {}).get(str(prefix), {}))
                if prefix:
                    for previous, current, key in [(f"envelope_{prefix - 1}.npy", envelope, f"{prefix - 1}:{prefix}"),
                                                    (f"norm_env_{prefix - 1}.npy", saved, f"norm_{prefix - 1}:{prefix}")]:
                        with numeric_mmap(root, entries[previous], guard) as (prior, prior_evidence):
                            require_finite(prior, guard)
                            result["prefix_differences"][key] = adjacent_metrics(prior, current, guard)
                        if prior_evidence["before"] != result["file_stats"][previous]["before"]:
                            raise ReconciliationError("prior prefix identity changed")
                        if claim is not None:
                            compare_metrics(checks, "prefix_differences/" + key, result["prefix_differences"][key], claim.get("prefix_differences", {}).get(key, {}))
        note(name, evidence)
        note(saved_name, saved_evidence)
        if progress is not None:
            progress(prefix + 1, checks)
    for name, evidence in result["file_stats"].items():
        guard()
        evidence["final"] = identity((root / name).lstat())
        if evidence["final"] != evidence["before"]:
            raise ReconciliationError("input changed before reconciliation completion")
    result["all_norm_comparisons_passed"] = all(v["norm_comparison_passed"] for v in result["envelopes"].values())
    result["all_envelopes_have_byte_identical_time_rows"] = all(v["all_time_rows_exactly_equal_to_row0"] for v in result["envelopes"].values())
    result["norm_comparison_count"] = 19 * 20 * 6
    result["norm_failure_count"] = sum(v["norm_failure_count"] for v in result["envelopes"].values())
    if claim is not None:
        for key, expected in [("schema_version", 1), ("kind", "legacy_floquet_descriptive_array_audit"),
                              ("scientific_acceptance", False), ("fixture_only", fixture_width is not None),
                              ("integrity_completed", True), ("all_norm_comparisons_passed", result["all_norm_comparisons_passed"])]:
            checks.compare(key, expected, claim.get(key))
        checks.compare("envelope_indices", sorted(result["envelopes"]), sorted(claim.get("envelopes", {})))
        checks.compare("prefix_difference_keys", sorted(result["prefix_differences"]), sorted(claim.get("prefix_differences", {})))
        checks.compare("file_stat_keys", sorted(entries), sorted(claim.get("file_stats", {})))
    result.update(integrity_completed=True, scalar_comparisons=checks.count,
                  reconciliation_passed=not checks.mismatches, mismatches=checks.mismatches,
                  numerical_integrity_accepted=False,
                  mismatch_count=len(checks.mismatches), maximum_numeric_reconciliation_error=checks.max_numeric_error)
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for key in ("inputs", "manifest", "manifest-sha256", "audit", "audit-sha256", "output", "source-sha256", "helper-sha256", "source-commit"):
        value.add_argument("--" + key, required=True)
    value.add_argument("--wall-seconds", type=int, default=600)
    value.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    value.add_argument("--started-monotonic", type=float, help=argparse.SUPPRESS)
    value.add_argument("--work-deadline-monotonic", type=float, help=argparse.SUPPRESS)
    value.add_argument("--owned-process-journal", help=argparse.SUPPRESS)
    return value


def source_identities(args, repo):
    expected = {Path(__file__): args.source_sha256, repo / "code/ops/legacy_floquet_audit.py": args.helper_sha256,
                repo / "src/representation_collapse/reporting.py": REPORTER_SHA, repo / "uv.lock": LOCK_SHA}
    if args.helper_sha256 != HELPER_SHA:
        raise ReconciliationError("unreviewed owning supervisor helper")
    for path, digest in expected.items():
        if sha_file(path) != digest:
            raise ReconciliationError("source or environment pin mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True)
    if not re.fullmatch("[0-9a-f]{40}", args.source_commit) or head != args.source_commit or dirty:
        raise ReconciliationError("exact clean published source required")
    for path, digest in expected.items():
        blob = subprocess.check_output(["git", "show", "HEAD:" + str(path.resolve().relative_to(repo))], cwd=repo)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ReconciliationError("reviewed source differs from committed HEAD")
    return {"source_sha256": args.source_sha256, "helper_sha256": args.helper_sha256,
            "reporter_sha256": REPORTER_SHA, "lock_sha256": LOCK_SHA, "source_commit": head,
            "manifest_sha256": args.manifest_sha256, "producer_audit_sha256": args.audit_sha256}


def check_producer_receipts(audit_path, claim, manifest_sha):
    worker = audit_path.parent
    paths = {"worker": worker / "audit-receipt.json", "supervisor": worker.parent / "supervisor-receipt.json",
             "reporting": worker / "reporting/reporting-receipt.json", "admission": worker.parent.parent / "root-admission.json"}
    receipts = {key: pinned_json(path) for key, path in paths.items()}
    if any(receipts[key].get("state") != "completed" for key in ("worker", "supervisor", "reporting")):
        raise ReconciliationError("producer receipts incomplete")
    supervision = receipts["supervisor"]
    if (supervision.get("exit_code") != 0 or supervision.get("cleanup_error_types") != [] or
            supervision.get("cpu_threads") != 1 or supervision.get("retry_count") != 0 or
            not 0 < supervision.get("elapsed_seconds", 601) < 600 or
            not all(v == "original_process_exited" for v in supervision.get("cleanup_reconciliation", {}).values())):
        raise ReconciliationError("producer supervision not verified")
    owned = supervision.get("owned_processes", [])
    reconciliation = supervision.get("cleanup_reconciliation", {})
    if (not owned or set(reconciliation) != {str(v["pid"]) for v in owned} or
            str(supervision.get("owned_pid")) not in reconciliation):
        raise ReconciliationError("producer owned-process exits not reconciled")
    report = receipts["reporting"]
    if (report.get("mode") != "offline" or report.get("sdk_version") != "0.30.0" or not report.get("finish_returned") or
            report.get("remote_delivery_verified") is not False):
        raise ReconciliationError("producer offline reporting not verified")
    expected = {"source_commit": PRODUCER_COMMIT, "manifest_sha256": manifest_sha,
                "source_sha256": "edfa4336040d5f54856d474d40b8b8eeb06e0089ad0af2c442dba1b7efe3db6a",
                "launcher_sha256": HELPER_SHA, "reporter_sha256": REPORTER_SHA, "lock_sha256": LOCK_SHA,
                "config_sha256": "a078ec7306fb88cf83b21758e307490e5e025005212bd93f28b92272b83c5409"}
    for selected in (claim.get("identities", {}), receipts["worker"].get("identities", {}), report.get("identities", {})):
        if selected != expected:
            raise ReconciliationError("producer lineage pin mismatch")
    if receipts["admission"].get("source_commit") != PRODUCER_COMMIT or receipts["admission"].get("manifest_sha256") != manifest_sha:
        raise ReconciliationError("producer admission mismatch")
    if any(receipts[key].get("scientific_acceptance") is not False for key in ("worker", "supervisor", "admission")):
        raise ReconciliationError("scientific acceptance is outside reconciliation scope")
    return {key: sha_file(path) for key, path in paths.items()}


def worker_main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": 1, "state": "initializing", "scientific_acceptance": False,
               "array_work_started": False, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - args.started_monotonic
        atomic_json(output / "reconciliation-receipt.json", receipt)
    def guard():
        if time.monotonic() >= args.work_deadline_monotonic:
            raise TimeoutError("numerical work deadline")
    def alarm(signum, frame):
        raise TimeoutError("numerical work deadline")
    persist()
    try:
        remaining = args.work_deadline_monotonic - time.monotonic()
        if not 0 < remaining <= 590 or args.work_deadline_monotonic - args.started_monotonic > 590:
            raise ReconciliationError("invalid shared deadline")
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        repo = Path(__file__).resolve().parents[2]
        identities = source_identities(args, repo)
        receipt["identities"] = identities
        persist()
        helpers = runpy.run_path(str(repo / "code/ops/legacy_floquet_audit.py"))
        helpers["install_spawn_journal"](Path(args.owned_process_journal))
        from representation_collapse.reporting import ExperimentReporter
        with ExperimentReporter(output / "reporting", project="representation-collapse", entity="mehrab-hamidi-milaquebec5431",
                                mode="offline", identities=identities, config={"protocol_id": "legacy-array-independent-reconciliation-v1",
                                "cpu_threads": 1, "chunk_state_rows": ROWS, "norm_atol": NORM_ATOL, "norm_rtol": NORM_RTOL,
                                "reconciliation_atol": RECON_ATOL, "reconciliation_rtol": RECON_RTOL}) as report:
            guard()
            receipt.update(state="running", input_hash_work_started=True)
            persist()
            manifest = pinned_json(args.manifest, args.manifest_sha256)
            manifest_entries(manifest)
            claim = pinned_json(args.audit, args.audit_sha256)
            receipt["producer_receipt_sha256"] = check_producer_receipts(Path(args.audit), claim, args.manifest_sha256)
            guard()
            receipt["array_work_started"] = True
            persist()
            def progress(prefix_count, checks):
                guard()
                report.log({"reconciliation/prefixes_completed": prefix_count, "reconciliation/mismatches": len(checks.mismatches),
                            "reconciliation/elapsed_seconds": time.monotonic() - args.started_monotonic}, step=prefix_count - 1)
            result = reduce_inputs(Path(args.inputs), manifest, claim, guard=guard, progress=progress)
            guard()
            result["identities"] = identities
            result["audit_sha256"] = args.audit_sha256
            result["producer_receipt_sha256"] = receipt["producer_receipt_sha256"]
            atomic_json(output / "reconciliation.json", result)
            report.log({"reconciliation/completed": True, "reconciliation/passed": result["reconciliation_passed"],
                        "reconciliation/mismatches": result["mismatch_count"], "reconciliation/scientific_acceptance": False}, step=19)
            signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(state="completed", reconciliation_passed=result["reconciliation_passed"], mismatch_count=result["mismatch_count"],
                       result_sha256=sha_file(output / "reconciliation.json"))
        persist()
        return 0
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(state="failed", error_type=type(error).__name__,
                       failure_reason=str(error) if isinstance(error, ReconciliationError) else None)
        persist()
        return 1


def main(argv=None):
    started = _PROCESS_STARTED_MONOTONIC
    args = parser().parse_args(argv)
    if args.worker:
        if any(v is None for v in (args.started_monotonic, args.work_deadline_monotonic, args.owned_process_journal)):
            raise ReconciliationError("owning supervisor required")
        return worker_main(args)
    if not 30 <= args.wall_seconds <= 600:
        raise ReconciliationError("outer wall limit must be30..600")
    repo = Path(__file__).resolve().parents[2]
    source_identities(args, repo)
    helpers = runpy.run_path(str(repo / "code/ops/legacy_floquet_audit.py"))
    output = Path(args.output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    deadline, child, ownership = started + args.wall_seconds, None, None
    receipt = {"schema_version": 1, "state": "initializing", "scientific_acceptance": False,
               "cpu_threads": 1, "wall_seconds": args.wall_seconds, "work_seconds": args.wall_seconds - 10, "retry_count": 0}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "supervisor-receipt.json", receipt)
    persist()
    try:
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"]
        for key in ("inputs", "manifest", "manifest_sha256", "audit", "audit_sha256", "source_sha256", "helper_sha256", "source_commit"):
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        command += ["--output", str(output / "worker"), "--started-monotonic", str(started),
                    "--work-deadline-monotonic", str(started + args.wall_seconds - 10),
                    "--owned-process-journal", str(output / "owned-processes.json")]
        env = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            env[key] = "1"
        env.update(WANDB_MODE="offline", PYTHONPATH=str(repo / "src"))
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, env=env, cwd=repo, start_new_session=True)
            ownership = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
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
        receipt.update(state="incomplete", error_type=type(error).__name__, outer_deadline_enforced=isinstance(error, subprocess.TimeoutExpired))
    finally:
        if child is not None:
            if ownership is None:
                ownership = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
            errors = ownership.cleanup(deadline=deadline)
            receipt.update(owned_processes=list(ownership.known.values()), cleanup_signals=ownership.signals,
                           cleanup_error_types=errors, cleanup_reconciliation=ownership.cleanup_reconciliation)
            if errors:
                receipt["state"] = "incomplete"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    receipt["state"] = "incomplete"
        if time.monotonic() >= deadline:
            receipt["state"] = "incomplete"
        persist()
    return 0 if receipt["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
