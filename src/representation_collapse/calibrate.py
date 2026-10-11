"""CPU calibration fixtures. These are measurement checks, not model results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from .metrics import covariance_dimension, ordered_product, period_spectrum


def calibration(seed: int = 0) -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(seed)
    # Centered orthogonal columns give exactly isotropic empirical covariance.
    cloud = rng.normal(size=(512, 4))
    cloud -= cloud.mean(axis=0)
    isotropic, _ = np.linalg.qr(cloud)
    rank_one = isotropic * np.array([1.0, 0, 0, 0])
    rotating = np.array([[0.0, -0.8], [0.8, 0.0]])
    # Eigenvalue decay can coexist with finite-time amplification (nonnormality).
    nonnormal = np.array([[0.8, 4.0], [0.0, 0.8]])
    steps = np.array([[[1.0, 2.0], [0.0, 1.0]], [[0.5, 0.0], [1.0, 0.5]]])
    # Stable periodically forced system: oscillation does not require neutral modes.
    state = np.zeros(2)
    trajectory = []
    forcing = np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
    for t in range(240):
        state = 0.5 * state + forcing[t % 4]
        trajectory.append(state.copy())
    trajectory = np.asarray(trajectory)
    metrics = {
        "isotropic": covariance_dimension(isotropic),
        "rank_one": covariance_dimension(rank_one),
        "zero": covariance_dimension(np.zeros((8, 4))),
        "rotation_period_four": period_spectrum(np.repeat(rotating[None], 4, axis=0)),
        "nonnormal": period_spectrum(nonnormal[None]),
        "forced_period_four": period_spectrum(np.repeat((0.5 * np.eye(2))[None], 4, axis=0)),
        "forced_orbit_relative_residual": float(
            np.linalg.norm(trajectory[-4:] - trajectory[-8:-4])
            / max(np.linalg.norm(trajectory[-4:]), np.finfo(float).eps)
        ),
    }
    checks = {
        "isotropic_dimension_four": bool(np.isclose(metrics["isotropic"]["participation_ratio"], 4)),
        "rank_one_dimension_one": bool(np.isclose(metrics["rank_one"]["participation_ratio"], 1)),
        "zero_variance_convention": metrics["zero"]["participation_ratio"] == 0,
        "product_order": bool(np.allclose(ordered_product(steps), steps[1] @ steps[0])),
        "rotation_period_multiplier": bool(np.isclose(metrics["rotation_period_four"]["spectral_radius"], 0.8**4)),
        "nonnormal_stable_eigenvalues": metrics["nonnormal"]["spectral_radius"] < 1,
        "nonnormal_transient_gain": metrics["nonnormal"]["largest_finite_time_gain"] > 1,
        "forced_orbit_periodic": metrics["forced_orbit_relative_residual"] < 1e-10,
        "forced_orbit_nonconstant": bool(np.linalg.norm(trajectory[-1] - trajectory[-2]) > 0.1),
        "forced_orbit_strict_contraction": metrics["forced_period_four"]["spectral_radius"] < 1,
    }
    arrays = {"isotropic": isotropic, "rank_one": rank_one, "forcing": forcing, "trajectory": trajectory}
    return metrics, checks, arrays


def _write_json(path: Path, value: dict) -> None:
    if path.name == "status.json":
        value = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "cluster": os.environ.get("CLUV_CLUSTER_NAME"), **value}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; refuses to overwrite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        revision, dirty = None, True
    root = Path(__file__).resolve().parents[2]
    source_files = [Path(__file__), Path(__file__).with_name("metrics.py"), root / "uv.lock"]
    _write_json(args.output / "config.json", {
        "schema_version": 1, "kind": "synthetic_measurement_calibration", "seed": args.seed,
        "git_revision": revision, "dirty_checkout": dirty,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.exists()},
        "python": platform.python_version(), "numpy": np.__version__, "started_at": started,
        "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output": str(args.output.resolve()),
    })
    _write_json(args.output / "status.json", {"schema_version": 1, "state": "running", "updated_at": started})
    metrics, checks, arrays = calibration(args.seed)
    _write_json(args.output / "metrics.json", metrics)
    _write_json(args.output / "verification.json", {"schema_version": 1, "passed": all(checks.values()), "checks": checks})
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    _write_json(args.output / "status.json", {
        "schema_version": 1, "state": "verified" if all(checks.values()) else "failed",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(args.output / "manifest.json", {
        "schema_version": 1,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file()},
    })
    print(json.dumps({"output": str(args.output), "checks": len(checks), "passed": all(checks.values())}))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
