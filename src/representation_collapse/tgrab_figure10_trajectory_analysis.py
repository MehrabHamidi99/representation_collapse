"""Descriptive FP64 time-cloud spectra and closure for pinned Figure 10 states.

No model, checkpoint, graph-data loader, Jacobian or held-out opener is used.
The CLI enters mandatory offline reporting before opening numerical bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat

SHAPE = (256, 2, 1, 100, 128)
PAYLOAD_BYTES = math.prod(SHAPE) * 4
PROTOCOL = "figure10_descriptive_state_trajectory_v1"
PROTOCOL_PATH = Path("doc/auto/figure10-trajectory-diagnostics-2026-10-19.md")


class TrajectoryError(ValueError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise TrajectoryError("nonfinite JSON")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, expected_sha256, *, max_bytes, exact_bytes=None):
    """Bounded no-follow read with identity checks; binary callers run in reporter."""
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise TrajectoryError("explicit SHA-256 required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > max_bytes
                or (exact_bytes is not None and before.st_size != exact_bytes)):
            raise TrajectoryError("regular single-link file of exact bounded size required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        after = os.fstat(fd)
        current = path.lstat()
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_nlink)
        if (identity(before) != identity(after) or identity(after) != identity(current)
                or len(raw) != before.st_size or _sha(raw) != expected_sha256):
            raise TrajectoryError("input identity changed or hash mismatch")
        return raw
    finally:
        os.close(fd)


def validate_descriptor(metadata, trajectory_path, expected_hash):
    descriptor = metadata.get("state_trajectory")
    expected = {"filename": "test-state-trajectory.f32le", "bytes": PAYLOAD_BYTES,
                "shape": list(SHAPE), "dtype": "float32_le", "order": "C",
                "components": ["hidden", "cell"], "node_ids": list(range(100)),
                "target_interval": [2816, 3072], "state_timing": "s[t]_before_target_score",
                "observations": 256, "centered_time_cloud_rank_ceiling": 255}
    if type(descriptor) is not dict or set(descriptor) != set(expected) | {"sha256"}:
        raise TrajectoryError("exact versioned state descriptor required")
    for key, value in expected.items():
        if _canonical(descriptor[key]) != _canonical(value):
            raise TrajectoryError("state descriptor mismatch: " + key)
    if descriptor["sha256"] != expected_hash or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise TrajectoryError("explicit state SHA differs from evaluation descriptor")
    if Path(trajectory_path).name != descriptor["filename"]:
        raise TrajectoryError("state filename differs from evaluation descriptor")
    rows = metadata.get("rows")
    if (type(rows) is not list or len(rows) != 256
            or any(type(row) is not dict or type(row.get("target_timestamp")) is not int for row in rows)
            or [row["target_timestamp"] for row in rows] != list(range(2816, 3072))):
        raise TrajectoryError("chronological 256 target metadata rows required")
    return descriptor


def load_trajectory(path, expected_hash):
    import numpy as np
    raw = read_pinned(path, expected_hash, max_bytes=PAYLOAD_BYTES, exact_bytes=PAYLOAD_BYTES)
    array = np.frombuffer(raw, dtype="<f4").reshape(SHAPE)
    if not np.isfinite(array).all():
        raise TrajectoryError("nonfinite complete states")
    return array


def covariance_cloud(values):
    """FP64 centered-time covariance; no inferred population sampling model."""
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.isfinite(x).all():
        raise TrajectoryError("finite n-by-d cloud with n>=2 required")
    n, d = x.shape
    mean = x.mean(axis=0)
    centered = x - mean
    squared_sum = float(np.sum(centered * centered, dtype=np.float64))
    gram = centered @ centered.T / (n - 1)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    if not np.isfinite(eigenvalues).all():
        raise TrajectoryError("nonfinite Gram eigenvalues")
    largest = max(float(eigenvalues[0]), 0.0)
    tolerance = 64 * np.finfo(np.float64).eps * max(n, d) * largest
    if float(eigenvalues[-1]) < -tolerance:
        raise TrajectoryError("Gram eigenvalue below scale-aware negative tolerance")
    cleaned = np.where(eigenvalues > tolerance, eigenvalues, 0.0)
    rank = int(np.count_nonzero(cleaned))
    if rank > min(n - 1, d):
        raise TrajectoryError("numerical covariance rank exceeds centered-cloud ceiling")
    trace = squared_sum / (n - 1)
    if not math.isclose(float(eigenvalues.sum()), trace, rel_tol=1e-10, abs_tol=tolerance * n):
        raise TrajectoryError("Gram eigenspectrum trace differs from centered variance")
    positive_trace_removed = float(np.sum(np.maximum(eigenvalues, 0.0) - cleaned))
    retained_trace = float(cleaned.sum())
    defined = squared_sum != 0.0
    if defined and retained_trace <= 0:
        raise TrajectoryError("nonzero cloud lost its entire spectrum to numerical threshold")
    if defined:
        p = cleaned / retained_trace
        positive = p[p > 0]
        pr = float(1.0 / np.sum(p * p))
        entropy = float(np.exp(-np.sum(positive * np.log(positive))))
        normalized = p.tolist()
    else:
        pr, entropy, normalized = None, None, None
    return {"samples": n, "features": d, "centering": "own_time_cloud_mean",
            "covariance_denominator": n - 1, "time_rank_ceiling": min(n - 1, d),
            "mean_squared_centered_norm": squared_sum / n, "sample_covariance_trace": trace,
            "mean_state_squared_norm": float(np.sum(mean * mean)),
            "maximum_absolute_state_value": float(np.max(np.abs(x))),
            "raw_gram_eigenvalues_descending": eigenvalues.tolist(),
            "cleaned_eigenvalues_descending": cleaned.tolist(), "spectral_tolerance": tolerance,
            "positive_trace_removed_by_tolerance": positive_trace_removed,
            "numerical_rank": rank, "normalized_dimension_defined": defined,
            "normalized_spectrum": normalized, "participation_ratio": pr,
            "entropy_rank_natural_log": entropy}


def period_closure(states):
    """All 192 adjacent-cycle complete-state comparisons; no Floquet inference."""
    import numpy as np
    array = np.asarray(states)
    if array.shape != SHAPE or array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
        raise TrajectoryError("closure requires complete finite canonical FP32 trajectory")
    flat = array.reshape(256, -1)
    comparisons = []
    for index in range(192):
        a, b = flat[index].astype(np.float64), flat[index + 64].astype(np.float64)
        delta = b - a
        residual = float(np.linalg.norm(delta))
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        bitwise = bool(np.array_equal(flat[index].view("<u4"), flat[index + 64].view("<u4")))
        comparisons.append({"reference_target_timestamp": 2816 + index, "next_target_timestamp": 2880 + index,
            "phase_mod64": index % 64, "reference_cycle": index // 64 + 1,
            "next_cycle": index // 64 + 2, "exact_bitwise_equal": bitwise,
            "absolute_l2_residual": residual, "maximum_absolute_coordinate_difference": float(np.max(np.abs(delta))),
            "relative_l2_residual": residual / denominator if denominator != 0 else 0.0})
    return {"period_steps": 64, "phase_count": 64, "cycle_pairs": 3, "comparisons": comparisons,
            "comparison_count": 192, "all_complete_states_bitwise_equal": all(r["exact_bitwise_equal"] for r in comparisons),
            "maximum_absolute_l2_residual": max(r["absolute_l2_residual"] for r in comparisons),
            "rms_absolute_l2_residual": math.sqrt(math.fsum(r["absolute_l2_residual"] ** 2 for r in comparisons) / 192),
            "maximum_relative_l2_residual": max(r["relative_l2_residual"] for r in comparisons),
            "floquet_or_jacobian_periodicity_certified": False}


def analyze_trajectory(states):
    import numpy as np
    array = np.asarray(states)
    if array.shape != SHAPE or array.dtype != np.dtype("<f4") or not np.isfinite(array).all():
        raise TrajectoryError("canonical finite time,hidden/cell,layer,node,feature FP32 trajectory required")
    views = {"hidden_and_cell": array.reshape(256, -1),
             "hidden": array[:, 0].reshape(256, -1), "cell": array[:, 1].reshape(256, -1)}
    windows = {"all_targets": np.arange(256)}
    for motif in ("A", "B", "pooled"):
        for name, first, last in (("early", 1, 8), ("late", 24, 31)):
            indices = [i for i in range(256) if first <= i % 32 <= last
                       and (motif == "pooled" or ("A" if i % 64 < 32 else "B") == motif)]
            if len(indices) != (64 if motif == "pooled" else 32):
                raise TrajectoryError("fixed equal-count phase-window selection failed")
            windows[motif + "_" + name] = np.asarray(indices)
    clouds = {window: {"target_timestamps": (indices + 2816).tolist(),
                       "views": {name: covariance_cloud(values[indices]) for name, values in views.items()}}
              for window, indices in windows.items()}
    return {"kind": PROTOCOL, "clouds": clouds, "period64_complete_state_closure": period_closure(array),
            "covariance_kind": "centered_time_cloud_not_population", "model_seeds": 1, "graph_realizations": 1,
            "independent_cycle_replicates": False, "inferential_tests": False,
            "scientific_collapse_claim": False, "causal_retention_claim": False,
            "floquet_or_jacobian_measurement": False,
            "early_within_motif_phases_inclusive": [1, 8], "late_within_motif_phases_inclusive": [24, 31]}


def _write_new(path, value):
    raw = _canonical(value)
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return _sha(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-result", "evaluation-result-sha256", "trajectory", "trajectory-sha256", "output", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        raise TrajectoryError("valid single-use analysis run ID required")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None and re.fullmatch(r"[1-9][0-9]{0,19}", job_id) is None:
        raise TrajectoryError("invalid actual Slurm job identity")
    metadata = json.loads(read_pinned(args.evaluation_result, args.evaluation_result_sha256, max_bytes=2_000_000),
                          object_pairs_hook=_unique, parse_constant=_reject_constant)
    validate_descriptor(metadata, args.trajectory, args.trajectory_sha256)
    root = Path(__file__).resolve().parents[2]
    protocol_sha = _sha((root / PROTOCOL_PATH).read_bytes())
    identities = {"evaluation_result_sha256": args.evaluation_result_sha256, "trajectory_sha256": args.trajectory_sha256,
                  "analysis_source_sha256": _sha(Path(__file__).read_bytes()), "protocol_sha256": protocol_sha}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    terminal = {"kind": PROTOCOL, "status": "failed", "scientific_collapse_claim": False, "identities": identities,
                "run_id": args.run_id, "slurm_job_id": job_id, "source_shape": list(SHAPE), "source_dtype": "float32_le"}
    try:
        from .reporting import ExperimentReporter
        with ExperimentReporter(output / "wandb", project="representation-collapse", run_id=args.run_id, mode="offline",
                config={"protocol": PROTOCOL, "source_dtype": "float32_le", "analysis_precision": "float64",
                        "model_seeds": 1, "time_rows": 256, "inferential_tests": False}, identities=identities) as reporter:
            import numpy as np
            environment = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                           "system": platform.system(), "machine": platform.machine(), "numpy": np.__version__}
            terminal["analysis_environment"] = environment
            states = load_trajectory(args.trajectory, args.trajectory_sha256)
            result = analyze_trajectory(states)
            result.update(identities=identities, state_descriptor=metadata["state_trajectory"],
                          run_id=args.run_id, slurm_job_id=job_id, analysis_environment=environment)
            result_hash = _write_new(output / "trajectory-analysis.json", result)
            event = 0
            for window, record in result["clouds"].items():
                for view, values in record["views"].items():
                    metrics = {"cloud/window": window, "cloud/view": view,
                        "cloud/mean_squared_centered_norm": values["mean_squared_centered_norm"],
                        "cloud/sample_covariance_trace": values["sample_covariance_trace"],
                        "cloud/numerical_rank": values["numerical_rank"],
                        "cloud/normalized_dimension_defined": values["normalized_dimension_defined"]}
                    if values["normalized_dimension_defined"]:
                        metrics.update({"cloud/participation_ratio": values["participation_ratio"],
                                        "cloud/entropy_rank": values["entropy_rank_natural_log"]})
                    reporter.log(metrics, step=event); event += 1
            closure = result["period64_complete_state_closure"]
            reporter.log({"closure/max_absolute_l2": closure["maximum_absolute_l2_residual"],
                          "closure/max_relative_l2": closure["maximum_relative_l2_residual"],
                          "closure/exact_bitwise": closure["all_complete_states_bitwise_equal"]}, step=event)
            reporter.log_manifest("trajectory-analysis-result", metadata={"status": "completed", "time_rows": 256,
                "components": 2, "state_features": 25600, "covariance_clouds": 21,
                "scientific_collapse_claim": False, "binary_state_uploaded": False},
                identities={**identities, "analysis_sha256": result_hash})
            terminal.update(status="completed", analysis_sha256=result_hash, reporting_events=event + 1)
    except BaseException as error:
        terminal.update(status="failed", error_type=type(error).__name__)
    finally:
        receipt = output / "wandb/reporting-receipt.json"
        if receipt.is_file():
            terminal["reporting_receipt_sha256"] = _sha(receipt.read_bytes())
        _write_new(output / "analysis-manifest.json", terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0 if terminal["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
