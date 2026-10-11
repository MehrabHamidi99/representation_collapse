#!/usr/bin/env python3
"""Deterministically verify explicit phase-one artifacts; no SSH or allocations.

The verifier never treats baseline incompetence as a corrupt artifact, or one
trained seed as population replication. Run from the pinned project checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Works even if this machine's editable .pth is hidden/ignored by Python.
sys.path.insert(0, str(ROOT / "src"))
from representation_collapse.metrics import covariance_dimension

ATOL, RTOL = 1e-7, 1e-5


class Blocked(ValueError):
    pass


class Verifier:
    def __init__(self, directory: Path, expected_sha: str):
        self.directory = directory.resolve()
        self.expected_sha = expected_sha
        self.checks = []
        self.reduced = {}

    def require(self, condition, field, detail=""):
        self.checks.append({"field": field, "passed": bool(condition), "detail": detail})
        if not condition:
            raise Blocked(f"{field}: {detail or 'verification failed'}")

    def close(self, actual, expected, field, *, atol=ATOL):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=RTOL), field,
                     "recomputed values must match saved values within declared tolerance")

    def read(self, name):
        data = json.loads((self.directory / name).read_text())
        finite_json(data)
        return data

    def manifest(self, required):
        self.require(self.directory.is_dir(), "directory", "explicit artifact directory missing")
        manifest = self.read("manifest.json")
        entries = manifest["sha256"]
        self.require(set(required).issubset(entries), "manifest.required_artifacts",
                     "required: " + ", ".join(sorted(required)))
        for name, expected in entries.items():
            self.require(Path(name).name == name and name not in {".", "..", "manifest.json"}, "manifest.path", name)
            path = self.directory / name
            self.require(path.is_file() and not path.is_symlink(), "manifest.file", name)
            self.require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "manifest.digest_format", name)
            self.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "manifest.sha256", name)

    def source(self, provenance, hashes, sources):
        self.require(provenance.get("git_revision") == self.expected_sha, "source.git_revision", "must equal expected full SHA")
        self.require(provenance.get("dirty_checkout") is False, "source.clean_checkout", "dirty or unrecorded checkout")
        by_name = {Path(name).name: value for name, value in hashes.items()}
        for relative in sources:
            result = subprocess.run(["git", "show", f"{self.expected_sha}:{relative}"], cwd=ROOT, capture_output=True)
            self.require(result.returncode == 0, "source.expected_object", relative)
            self.require(by_name.get(Path(relative).name) == hashlib.sha256(result.stdout).hexdigest(), "source.sha256", relative)

    def arrays(self, name):
        with np.load(self.directory / name, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        for key, value in arrays.items():
            self.require(value.dtype.kind in "biufc" and np.isfinite(value).all(), "arrays.finite", name + ":" + key)
        return arrays


def finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        raise Blocked("nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    if isinstance(value, list):
        for child in value:
            finite_json(child)


def body(ids, motifs):
    return "".join(motifs[int(i)] for i in ids)


def assert_pair(v, periodic, control, repeats, motifs, field):
    v.require(periodic.shape == control.shape and periodic.shape[-1] == repeats,
              field + ".shape", "matched horizon motif arrays required")
    v.require(periodic.dtype.kind in "iu" and control.dtype.kind in "iu" and
              np.all((periodic >= 0) & (periodic < len(motifs))) and
              np.all((control >= 0) & (control < len(motifs))), field + ".ids")
    lefts = periodic[None] if periodic.ndim == 1 else periodic
    rights = control[None] if control.ndim == 1 else control
    for left, right in zip(lefts, rights):
        v.require(np.all(left == left[0]) and left[-1] == right[-1], field + ".periodic_and_final_matching")
        a, b = body(left, motifs), body(right, motifs)
        v.require(Counter(a) == Counter(b) and Counter(zip(a, a[1:] + a[:1])) == Counter(zip(b, b[1:] + b[:1])), field + ".low_order_matching")
        if repeats == 1:
            v.require(a == b, field + ".shared_anchor")
        else:
            proper = any(len(b) % p == 0 and b == b[:p] * (len(b) // p) for p in range(1, len(b) // 2 + 1))
            v.require(not proper, field + ".varying_arm_not_properly_periodic")


def pilot(v):
    required = {"config.json", "provenance.json", "history.json", "checkpoint.pt", "metrics.json", "status.json", "paired_errors.npz", "cue_clouds.npz"}
    v.manifest(required)
    config, metrics, status, provenance = (v.read(name) for name in ("config.json", "metrics.json", "status.json", "provenance.json"))
    v.require(status.get("state") == "completed", "status.completed", "running/failed attempts are not accepted")
    v.require(config.get("seed") == 0 and metrics.get("seed") == 0, "config.seed0")
    v.require(config.get("training_steps") == config.get("effective_training_steps") == 1500 and config.get("max_steps_override") is False, "config.full_step_budget", "requires1500 steps without engineering truncation")
    v.require(config.get("scientific_acceptance") is False and metrics.get("scientific_acceptance") is False, "scientific_acceptance.remains_false")
    v.source(provenance, provenance["source_sha256"], ["src/representation_collapse/pilot.py", "src/representation_collapse/vector_task.py", "src/representation_collapse/metrics.py", "uv.lock", "pyproject.toml", "configs/experiments/vector-pilot-v1.json"])
    frozen = subprocess.run(["git", "show", f"{v.expected_sha}:configs/experiments/vector-pilot-v1.json"], cwd=ROOT, capture_output=True, check=True)
    base = json.loads(frozen.stdout)
    for key, value in base.items():
        v.require(config.get(key) == value, "config.immutable." + key)
    task, model = config["task"], config["model"]
    v.require(task in {"categorical", "continuous"} and model in {"gru", "lstm"}, "config.model_task")
    v.require(metrics["model"] == model and metrics["task"] == task, "metrics.model_task")
    v.require(config.get("device") == provenance.get("device") == "cuda", "phase.gpu_device", "initial phase pilot cells require recorded CUDA execution")
    v.require(provenance.get("deterministic_algorithms") is True and provenance.get("cublas_workspace_config") in {":4096:8", ":16:8"}, "phase.cuda_determinism_configuration")
    v.require(bool(provenance.get("slurm_job_id")) and bool(provenance.get("accelerator")), "phase.allocation_provenance", "scheduler ID and accelerator description are required")
    history = v.read("history.json")
    v.require(len(history) == 1500 and [row["step"] for row in history] == list(range(1, 1501)), "history.full_optimization_steps")
    v.require(all(row["repeats"] == config["train_repeats"][(row["step"] - 1) % 5] for row in history), "history.horizon_schedule")
    validation = [row for row in history if "validation_loss" in row]
    selected = min(validation, key=lambda row: (row["validation_loss"], row["step"]))
    v.require(metrics["selected_step"] == status["selected_step"] == selected["step"], "selection.validation_only_checkpoint")
    v.close(metrics["selection_validation_loss"], selected["validation_loss"], "selection.loss")
    v.require(metrics["training_episodes"] == 1500 * config["batch_size"], "training.episode_budget")
    state_dim = config["hidden_size"] * (2 if model == "lstm" else 1)
    components = ["hidden", "cell"] if model == "lstm" else ["hidden"]
    v.require(metrics["state"]["dimension"] == state_dim and metrics["state"]["components"] == components and metrics["state"]["persistent_fp32_bytes_per_sequence"] == 4 * state_dim, "state.complete_dimensions_and_bytes")
    h, b, o = config["hidden_size"], config["input_bottleneck"], 16 if task == "categorical" else 1
    gate_count = 3 if model == "gru" else 4
    core_head = gate_count * h * (b + h) + 2 * gate_count * h + o * (h + 1)
    v.require(metrics["parameters"]["core_plus_head"] == core_head and metrics["parameters"]["total"] == core_head + 39 * b and core_head <= 1000000 and 4 * state_dim <= 1048576, "state.resource_accounting")
    raw, clouds = v.arrays("paired_errors.npz"), v.arrays("cue_clouds.npz")
    reduced, competence = {}, {}
    for repeats in config["evaluation_repeats"] + config["secondary_repeats"]:
        prefix, n = f"R{repeats}_", config["evaluation_pairs_per_horizon"]
        memory, queries, targets = (raw[prefix + key] for key in ("memory", "queries", "targets"))
        v.require(memory.shape == (n, 16) and queries.shape == targets.shape == (n,), prefix + "cue_shapes")
        v.require((memory.dtype.kind in "iu" and np.all((memory >= 0) & (memory < 16))) if task == "categorical" else np.all((memory >= -1) & (memory <= 1)), prefix + "cue_domain")
        v.require(queries.dtype.kind in "iu" and np.all((queries >= 0) & (queries < 8)), prefix + "query_range")
        v.close(targets, memory[np.arange(n), queries], prefix + "target_from_independent_write")
        assert_pair(v, raw[prefix + "periodic_motif_ids"], raw[prefix + "control_motif_ids"], repeats, config["motif_library"], prefix + "bodies")
        errors = {}
        for arm in ["periodic", "control"]:
            prediction, saved_error = raw[prefix + arm + "_predictions"], raw[prefix + arm + "_errors"]
            v.require(prediction.shape == saved_error.shape == (n,), prefix + arm + ".shapes")
            if task == "categorical":
                v.require(prediction.dtype.kind in "iu" and np.all((prediction >= 0) & (prediction < 16)), prefix + arm + ".class_range")
                error = (prediction != targets).astype(float)
            else:
                error = np.square(prediction.astype(float) - targets.astype(float))
            v.close(saved_error, error, prefix + arm + ".errors_from_predictions")
            errors[arm] = error
            v.close(metrics["behavior"][str(repeats)][arm + "_error"], error.mean(), prefix + arm + ".mean")
        delta = float(np.mean(errors["periodic"] - errors["control"]))
        v.close(metrics["behavior"][str(repeats)]["delta"], delta, prefix + "delta")
        v.close(metrics["behavior"][str(repeats)]["paired_error_difference_mean"], delta, prefix + "paired_delta")
        reduced[str(repeats)] = {"periodic_error": float(errors["periodic"].mean()), "control_error": float(errors["control"].mean()), "delta": delta}
        if task == "continuous":
            zero_error = float(np.square(targets.astype(float)).mean())
            reduced[str(repeats)]["zero_predictor_mse_from_saved_targets"] = zero_error
            reduced[str(repeats)]["native_to_zero_mse_ratio"] = {
                arm: float(error.mean()) / zero_error if zero_error > 0 else None
                for arm, error in errors.items()}
        else:
            reduced[str(repeats)]["uniform_classifier_expected_error"] = 15 / 16
        if repeats in config["competence"]["horizons"]:
            competence[str(repeats)] = all(1 - error.mean() >= config["competence"]["categorical_min_accuracy"] for error in errors.values()) if task == "categorical" else all(error.mean() <= config["competence"]["continuous_max_mse"] for error in errors.values())
            v.require(bool(metrics["baseline_competence"][str(repeats)]["passed"]) == competence[str(repeats)], prefix + "competence_agreement")
        if repeats == 1:
            v.close(errors["periodic"], errors["control"], "behavior.shared_R1_anchor")
    did = reduced["16"]["delta"] - reduced["2"]["delta"]
    for field, value in [("delta2", reduced["2"]["delta"]), ("delta16", reduced["16"]["delta"]), ("difference_in_differences", did)]:
        v.close(metrics[field], value, "behavior." + field)
    panel_reductions = {}
    for repeats in [2, 16]:
        for panel in range(config["cloud_driver_pairs_per_horizon"]):
            keys = [f"R{repeats}_panel{panel}_{arm}" for arm in ("periodic", "control")]
            assert_pair(v, clouds[keys[0] + "_motif_ids"], clouds[keys[1] + "_motif_ids"], repeats, config["motif_library"], keys[0] + ".paired_bodies")
            for suffix in ["train_memory", "test_memory", "control_permutation"]:
                v.close(clouds[keys[0] + "_" + suffix], clouds[keys[1] + "_" + suffix], keys[0] + ".paired_" + suffix)
            for key in keys:
                train, test = clouds[key + "_train_states"], clouds[key + "_test_states"]
                train_tags, test_tags = clouds[key + "_train_memory"], clouds[key + "_test_memory"]
                v.require(train.shape == (config["cloud_samples"], state_dim) and test.shape == (config["probe_test_samples"], state_dim) and train.dtype == test.dtype == np.float32, key + ".complete_FP32_state_shapes")
                v.require(train_tags.shape == (len(train), 16) and test_tags.shape == (len(test), 16), key + ".cue_shapes")
                v.require((train_tags.dtype.kind in "iu" and test_tags.dtype.kind in "iu" and np.all((train_tags >= 0) & (train_tags < 16)) and np.all((test_tags >= 0) & (test_tags < 16))) if task == "categorical" else (np.all((train_tags >= -1) & (train_tags <= 1)) and np.all((test_tags >= -1) & (test_tags <= 1))), key + ".cue_domain")
                v.close(train_tags[:, 8:], np.broadcast_to(train_tags[0, 8:], train_tags[:, 8:].shape), key + ".fixed_train_nuisance")
                v.close(test_tags[:, 8:], np.broadcast_to(train_tags[0, 8:], test_tags[:, 8:].shape), key + ".fixed_test_nuisance")
                overlap = set(map(tuple, train_tags[:, :8])) & set(map(tuple, test_tags[:, :8]))
                v.require(not overlap, key + ".heldout_cue_histories", "train/test required-tag tuples must not overlap")
                geometry = covariance_dimension(train)
                for field, value in geometry.items():
                    if isinstance(value, (list, float)):
                        # Tiny cue variance must not pass merely because it is
                        # below a dimensionless1e-7 absolute tolerance. Spectrum
                        # tolerance scales with this cloud's recomputed trace.
                        atol = geometry["total_variance"] * 1e-12 if field == "covariance_eigenvalues" else 0.0 if field == "total_variance" else ATOL
                        v.close(metrics["clouds"][key]["geometry"][field], value, key + ".geometry." + field, atol=atol)
                    else:
                        v.require(metrics["clouds"][key]["geometry"][field] == value, key + ".geometry." + field)
                mean = train.astype(float).mean(axis=0)
                scale = np.maximum(train.astype(float).std(axis=0), 1e-8)
                v.close(clouds[key + "_probe_feature_mean"], mean, key + ".probe_centering")
                v.close(clouds[key + "_probe_feature_scale"], scale, key + ".probe_scaling")
                permutation = clouds[key + "_control_permutation"]
                v.require(permutation.shape == (len(train),) and np.array_equal(np.sort(permutation), np.arange(len(train))), key + ".label_control_permutation")
                standardized_train = (train.astype(float) - mean) / scale
                labels = np.eye(16)[train_tags[:, :8].astype(int)].reshape(len(train), -1) if task == "categorical" else train_tags[:, :8].astype(float)
                system = standardized_train.T @ standardized_train + config["probe_ridge"] * len(train) * np.eye(state_dim)
                probe_results = {}
                for name in ["real", "label_control"]:
                    coefficients, intercept = clouds[key + "_" + name + "_coefficients"], clouds[key + "_" + name + "_intercept"]
                    v.require(coefficients.shape == (state_dim, 128 if task == "categorical" else 8), key + ".probe_coefficient_shape")
                    fit_labels = labels if name == "real" else labels[permutation]
                    v.close(intercept, fit_labels.mean(axis=0), key + "." + name + ".training_only_intercept")
                    v.close(coefficients, np.linalg.solve(system, standardized_train.T @ (fit_labels - fit_labels.mean(axis=0))), key + "." + name + ".training_only_ridge_fit")
                    outputs = ((test.astype(float) - mean) / scale) @ coefficients + intercept
                    prediction = outputs.reshape(len(test), 8, 16).argmax(axis=2) if task == "categorical" else outputs
                    v.close(clouds[key + "_" + name + "_predictions"], prediction, key + "." + name + ".predictions_from_coefficients")
                    if task == "categorical":
                        score = (prediction == test_tags[:, :8]).mean(axis=0)
                        mean_field, coordinate_field = "mean_accuracy", "coordinate_accuracy"
                    else:
                        score = np.square(prediction - test_tags[:, :8]).mean(axis=0)
                        mean_field, coordinate_field = "mean_mse", "coordinate_mse"
                    saved = metrics["clouds"][key]["probes"][name]
                    v.close(saved[mean_field], score.mean(), key + "." + name + ".score")
                    v.close(saved[coordinate_field], score, key + "." + name + ".coordinate_scores")
                    probe_results[name] = {mean_field: float(score.mean())}
                panel_reductions[key] = {"participation_ratio": geometry["participation_ratio"], "total_variance": geometry["total_variance"],
                                         "normalized_dimension_defined": geometry["normalized_dimension_defined"],
                                         "numerical_rank": geometry["numerical_rank"], "samples": len(train), "state_dimension": state_dim,
                                         "probes": probe_results}
                # Post hoc numerical-resolution audit of saved FP32 states.
                # ULP ratios describe stored resolution, not a noise model or
                # proof that the cue trace is rounding error.
                centered_rms = float(np.sqrt(np.square(train.astype(float) - mean).mean()))
                ulp_rms = float(np.sqrt(np.square(np.abs(np.spacing(train)).astype(float)).mean()))
                distinct = [len(np.unique(train[:, i])) for i in range(state_dim)]
                panel_reductions[key]["precision_audit"] = {
                    "state_rms": float(np.sqrt(np.square(train.astype(float)).mean())),
                    "centered_state_rms": centered_rms, "stored_FP32_ulp_rms": ulp_rms,
                    "variation_to_ulp_rms_ratio": centered_rms / ulp_rms if ulp_rms > 0 else None,
                    "distinct_state_rows": len(np.unique(train, axis=0)),
                    "varying_coordinates": sum(n > 1 for n in distinct),
                    "coordinate_distinct_value_count_min": min(distinct),
                    "coordinate_distinct_value_count_median": float(np.median(distinct)),
                    "coordinate_distinct_value_count_max": max(distinct),
                    "scope": "post hoc stored-state resolution; not FP64 dynamics or robust memory validation"}
    passed = all(competence.values())
    v.require(metrics["baseline_competence_passed"] is bool(passed), "competence.overall_agreement")
    v.reduced = {"model": model, "task": task, "seed": 0, "selected_step": selected["step"], "behavior": reduced,
                 "delta2": reduced["2"]["delta"], "delta16": reduced["16"]["delta"], "DID": did,
                 "baseline_competence_passed": passed, "scientific_state": "mechanism_inconclusive_baseline_failed" if not passed else "engineering_evidence_only_confirmation_pending",
                 "clouds": panel_reductions, "independent_training_replicates": 1, "scientific_acceptance": False}


def calibration(v):
    v.manifest({"config.json", "metrics.json", "verification.json", "status.json", "arrays.npz"})
    config, verification, status = (v.read(name) for name in ("config.json", "verification.json", "status.json"))
    kind = config["kind"]
    expected = {"synthetic_measurement_calibration": (10, "calibrate.py"), "synthetic_mechanism_calibration": (36, "mechanism_calibration.py")}
    v.require(kind in expected, "calibration.kind")
    count, module = expected[kind]
    v.require(status.get("state") == "verified" and verification.get("passed") is True, "calibration.successful_status")
    v.require(config.get("seed") == 0, "calibration.seed0")
    v.source(config, config["source_hashes"], ["src/representation_collapse/" + module, "src/representation_collapse/metrics.py", "uv.lock"])
    checks = verification["checks"]
    v.require(len(checks) == count and all(value is True if isinstance(value, bool) else value.get("passed") is True for value in checks.values()), "calibration.check_count_and_pass", f"requires exactly{count} passing checks")
    v.read("metrics.json")
    arrays = v.arrays("arrays.npz")
    v.reduced = {"kind": kind, "checks": count, "arrays": len(arrays), "scientific_state": "numerical_infrastructure_only", "scientific_acceptance": False}


def atomic_output(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, action="append", default=[])
    parser.add_argument("--calibration-dir", type=Path, action="append", default=[])
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path, required=True, help="atomic JSON report outside immutable artifact directories")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sha) or not args.pilot_dir + args.calibration_dir:
        parser.error("provide a full expected SHA and at least one explicit artifact directory")
    directories = [(path, "pilot") for path in args.pilot_dir] + [(path, "calibration") for path in args.calibration_dir]
    if len({path.resolve() for path, _ in directories}) != len(directories):
        parser.error("duplicate artifact directories are not independent evidence")
    if any(args.output.resolve().is_relative_to(path.resolve()) for path, _ in directories):
        parser.error("write the report outside immutable input directories")
    frozen_metrics = subprocess.run(["git", "show", f"{args.expected_sha}:src/representation_collapse/metrics.py"], cwd=ROOT, capture_output=True)
    local_metrics_sha = hashlib.sha256((ROOT / "src/representation_collapse/metrics.py").read_bytes()).hexdigest()
    if frozen_metrics.returncode or hashlib.sha256(frozen_metrics.stdout).hexdigest() != local_metrics_sha:
        parser.error("the reducer must use the expected source's metric definitions")
    records = []
    for path, kind in directories:
        verifier = Verifier(path, args.expected_sha)
        blocker = None
        try:
            (pilot if kind == "pilot" else calibration)(verifier)
        except Exception as error:
            blocker = f"{type(error).__name__}: {error}"
        records.append({"directory": str(path.resolve()), "requested_kind": kind, "artifact_verified": blocker is None,
                        "blocked_reason": blocker, "checks": verifier.checks, "reduction": verifier.reduced})
    summary = {"artifact_directories": len(records), "verified": sum(record["artifact_verified"] for record in records),
               "blocked": sum(not record["artifact_verified"] for record in records), "scientific_acceptance": False,
               "single_seed_policy": "report each model/task separately; never pool architectures/tasks as training replication",
               "baseline_inconclusive_pilots": sum(record["artifact_verified"] and record["reduction"].get("baseline_competence_passed") is False for record in records)}
    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "expected_source_sha": args.expected_sha,
              "verifier": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "metrics_sha256": local_metrics_sha, "python": platform.python_version(), "numpy": np.__version__},
              "numeric_tolerance": {"absolute": ATOL, "relative": RTOL,
                                    "total_variance_absolute": 0.0,
                                    "covariance_eigenvalues_absolute": "1e-12 times recomputed cloud total variance"},
              "summary": summary, "records": records,
              "scope": "artifact integrity and independent raw-output reduction; not reproduced training or scientific acceptance",
              "required_external_acceptance": "Reconcile every directory with the immutable submission journal and scheduler terminal state, ExitCode, hardware and resource use; this offline verifier does not establish Slurm completion."}
    atomic_output(args.output, report)
    print(json.dumps(summary, sort_keys=True))
    if summary["blocked"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
