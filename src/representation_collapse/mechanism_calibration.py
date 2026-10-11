"""Analytic memory/geometry controls, not results about trained recurrent models.

Run with ``python -m representation_collapse.mechanism_calibration --output NEWDIR``.
Covariances vary cues at a fixed body and terminal phase. Products below are
finite-time cue propagators; no Floquet interpretation is assigned.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .metrics import covariance_dimension, ordered_product


ATOL = 1e-11
RTOL = 1e-10


def isotropic_cues(rng: np.random.Generator, observations: int, features: int) -> np.ndarray:
    """Distinct centered rows with exactly identity empirical covariance.

    A deterministic orthogonal design given the seed, not iid Gaussian cues.
    """
    if observations <= features or features < 1:
        raise ValueError("more observations than positive feature count required")
    raw = rng.normal(size=(observations, features))
    raw -= raw.mean(axis=0)
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return basis * np.sqrt(observations - 1)


def propagate_affine(cues: np.ndarray, matrices: np.ndarray, forcing: np.ndarray) -> np.ndarray:
    """Column-state chronology, with cue observations represented by rows."""
    cues = np.asarray(cues, dtype=float)
    matrices = np.asarray(matrices, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    if cues.ndim != 2 or matrices.ndim != 3 or matrices.shape[1:] != (cues.shape[1],) * 2:
        raise ValueError("expected cue rows and a sequence of square full-state matrices")
    if forcing.shape != matrices.shape[:2] or len(matrices) < 1:
        raise ValueError("one forcing vector per positive-horizon update required")
    if not all(np.isfinite(x).all() for x in (cues, matrices, forcing)):
        raise ValueError("finite inputs required")
    states = cues.copy()
    for matrix, injection in zip(matrices, forcing):
        states = states @ matrix.T + injection
    if not np.isfinite(states).all():
        raise ValueError("state propagation overflow")
    return states


def conditioned_dimension(states: np.ndarray, body_ids: np.ndarray) -> dict:
    """Reject pooled-body covariance when the requested quantity conditions on body.

    Each row must vary the cue under one identical complete body and phase.
    Labels are a caller-supplied provenance contract, not inferred from states.
    """
    states = np.asarray(states)
    labels = np.asarray(body_ids)
    if states.ndim != 2 or labels.ndim != 1 or len(labels) != len(states):
        raise ValueError("one body/phase identifier per cue observation required")
    if len(np.unique(labels)) != 1:
        raise ValueError("condition on one body and phase; pooled-body covariance is not a cue-channel metric")
    return covariance_dimension(states)


def propagated_noise_covariance(matrices: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Independent additive process noise after every update, starting noiseless."""
    matrices = np.asarray(matrices, dtype=float)
    ordered_product(matrices)  # Validate shape, finiteness and product.
    covariance = np.asarray(covariance, dtype=float)
    size = matrices.shape[1]
    if covariance.shape != (size, size) or not np.isfinite(covariance).all():
        raise ValueError("finite state-noise covariance of matching shape required")
    if not np.allclose(covariance, covariance.T, atol=ATOL, rtol=RTOL):
        raise ValueError("symmetric noise covariance required")
    if np.linalg.eigvalsh(covariance).min() < -ATOL:
        raise ValueError("positive semidefinite noise covariance required")
    result = np.zeros_like(covariance)
    for matrix in matrices:
        result = matrix @ result @ matrix.T + covariance
    return result


def linear_recovery_risk(channel: np.ndarray, noise_covariance: np.ndarray) -> float:
    """Population linear MSE for a centered identity-covariance cue.

    It is also MMSE for an independent additive Gaussian channel with Gaussian
    cue prior. That stronger interpretation is not assigned to our cue designs.
    Noise is at the *terminal observed channel*, not at intermediate states.
    """
    channel = np.asarray(channel, dtype=float)
    covariance = np.asarray(noise_covariance, dtype=float)
    if channel.ndim != 2 or min(channel.shape) < 1:
        raise ValueError("a nonempty finite channel matrix required")
    if covariance.shape != (channel.shape[0],) * 2:
        raise ValueError("noise covariance must match observed-channel coordinates")
    if not np.isfinite(channel).all() or not np.isfinite(covariance).all():
        raise ValueError("finite channel and covariance required")
    if not np.allclose(covariance, covariance.T, atol=ATOL, rtol=RTOL):
        raise ValueError("symmetric positive definite covariance required")
    np.linalg.cholesky(covariance)
    information = channel.T @ np.linalg.solve(covariance, channel)
    return float(np.trace(np.linalg.solve(np.eye(channel.shape[1]) + information,
                                         np.eye(channel.shape[1]))))


def calibration(seed: int = 0) -> tuple[dict, dict, dict]:
    """Return metrics, tolerance-qualified checks, and arrays for recalculation."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    checks: dict[str, dict] = {}

    def close(name: str, observed, expected, *, atol=ATOL, rtol=RTOL) -> None:
        actual, target = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
        error = float(np.max(np.abs(actual - target)))
        scale = float(np.max(np.abs(target)))
        checks[name] = {"passed": bool(np.allclose(actual, target, atol=atol, rtol=rtol)),
                        "maximum_absolute_error": error, "expected_scale": scale,
                        "absolute_tolerance": atol, "relative_tolerance": rtol}

    def above(name: str, observed: float, threshold: float) -> None:
        checks[name] = {"passed": bool(observed > threshold), "observed": float(observed),
                        "strict_lower_threshold": float(threshold)}

    cues = isotropic_cues(rng, 128, 4)
    arrays["cues"] = cues
    close("distinct_cues", len(np.unique(cues, axis=0)), len(cues), atol=0, rtol=0)
    close("cue_covariance_identity", np.cov(cues, rowvar=False), np.eye(4))

    # Fixed A: body order changes only the deterministic offset, even for E nonlinear.
    encoded = cues + 0.1 * cues**3
    matrix = 0.8 * np.eye(4)
    matrix[0, 1] = 0.05
    sequences = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    inject = np.array([[1.0, -0.3, 0.2, 0.0], [-0.5, 0.2, 0.1, 0.4]])
    steps = np.repeat(matrix[None], 4, axis=0)
    product = np.linalg.matrix_power(matrix, 4)
    readout = np.array([1.0, 0.5, -0.2, 0.1])
    target = encoded @ product.T @ readout
    states, offsets, dimensions = [], [], []
    for arm, sequence in enumerate(sequences):
        forcing = inject[sequence] + 0.05
        state = propagate_affine(encoded, steps, forcing)
        offset = propagate_affine(np.zeros((1, 4)), steps, forcing)[0]
        states.append(state)
        offsets.append(offset)
        dimensions.append(conditioned_dimension(state, np.full(len(cues), arm)))
    states, offsets = np.asarray(states), np.asarray(offsets)
    centered = states - states.mean(axis=1, keepdims=True)
    corrected = states - offsets[:, None, :]
    predictions = states @ readout
    nonlinear_predictions = np.tanh(predictions)
    nonlinear_target = np.tanh(target)
    linear_error = np.mean((predictions - target)**2, axis=1)
    nonlinear_error = np.mean((nonlinear_predictions - nonlinear_target)**2, axis=1)
    close("affine_equal_centered_cue_cloud", centered[0], centered[1])
    close("affine_exact_cue_component", corrected, np.repeat((encoded @ product.T)[None], 2, axis=0))
    close("affine_offset_explains_linear_error", linear_error, (offsets @ readout)**2)
    close("affine_offset_corrected_native_error", np.mean((corrected @ readout - target)**2), 0)
    corrected_nonlinear_error = float(np.mean((np.tanh(corrected @ readout) - nonlinear_target)**2))
    close("affine_offset_corrected_nonlinear_error", corrected_nonlinear_error, 0)
    above("affine_native_linear_error_can_differ", abs(linear_error[0] - linear_error[1]), 1e-3)
    above("affine_native_nonlinear_error_can_differ", abs(nonlinear_error[0] - nonlinear_error[1]), 1e-3)
    pooled = covariance_dimension(states.reshape(-1, 4))
    rejected = False
    try:
        conditioned_dimension(states.reshape(-1, 4), np.repeat([0, 1], len(cues)))
    except ValueError:
        rejected = True
    close("pooled_body_conditioning_is_rejected", rejected, True, atol=0, rtol=0)
    above("pooled_body_variance_contains_offset", pooled["total_variance"] - dimensions[0]["total_variance"], 1e-3)
    metrics["affine_ordering_null"] = {"horizon": 4, "fixed_body_dimensions": dimensions,
        "pooled_body_dimension_for_confound_only": pooled,
        "native_linear_mse": linear_error.tolist(), "native_nonlinear_mse": nonlinear_error.tolist(),
        "offset_corrected_linear_mse": float(np.mean((corrected @ readout - target)**2)),
        "offset_corrected_nonlinear_mse": corrected_nonlinear_error,
        "conditioning": "vary distinct cues; fixed complete body, horizon and terminal phase per arm",
        "interpretation": "equal conditioned cue channel; raw readout offsets can change native error"}
    arrays.update(affine_encoded_cues=encoded, affine_matrices=steps, affine_injections=inject,
                  affine_forcing=inject[sequences] + 0.05,
                  affine_sequences=sequences, affine_states=states, affine_offsets=offsets,
                  affine_readout=readout, affine_target=target, affine_product=product)

    # Commuting symbols fix the cue product, but intermediate noise need not agree.
    commuting = np.array([np.diag([0.5, 0.8]), np.diag([0.9, 0.4])])
    commuting_steps = commuting[sequences]
    products = np.array([ordered_product(arm) for arm in commuting_steps])
    process_covariances = np.array([propagated_noise_covariance(arm, np.eye(2))
                                    for arm in commuting_steps])
    commuting_states = np.array([propagate_affine(cues[:, :2], arm, np.zeros((4, 2)))
                                 for arm in commuting_steps])
    close("commuting_matched_counts_equal_product", products[0], products[1])
    close("commuting_equal_cue_cloud", commuting_states[0], commuting_states[1])
    noise_gap = float(np.linalg.norm(process_covariances[0] - process_covariances[1]))
    above("commuting_process_noise_can_depend_on_order", noise_gap, 1e-2)
    metrics["commuting_ordering_null"] = {"horizon": 4, "process_noise_covariance_gap": noise_gap,
        "terminal_noise_linear_risk": [linear_recovery_risk(p, 0.04 * np.eye(2)) for p in products],
        "risk_prior": "standardized identity-population-covariance cue model; analytic risk, not a fitted probe",
        "qualifier": "noiseless cue product or common terminal noise; fresh intermediate noise can differ"}
    arrays.update(commuting_symbols=commuting, commuting_steps=commuting_steps,
                  commuting_products=products, commuting_states=commuting_states,
                  commuting_process_noise_covariances=process_covariances)

    # Noncommutativity permits, but does not universally force, an ordering effect.
    switched = np.array([[[0.8, 0.3], [0.0, 0.7]], [[0.7, 0.0], [0.3, 0.8]]])
    switched_steps = switched[sequences]
    switched_products = np.array([ordered_product(arm) for arm in switched_steps])
    switched_states = np.array([propagate_affine(cues[:, :2], arm, np.zeros((4, 2)))
                                for arm in switched_steps])
    risks = np.array([linear_recovery_risk(p, 0.04 * np.eye(2)) for p in switched_products])
    above("noncommuting_products_differ", np.linalg.norm(switched_products[0] - switched_products[1]), 1e-3)
    above("noncommuting_conditioned_clouds_differ", np.linalg.norm(switched_states[0] - switched_states[1]), 1e-3)
    above("noncommuting_terminal_noise_recovery_differs", abs(risks[0] - risks[1]), 1e-4)
    close("noncommuting_explicit_chronology", switched_products[0], switched[1] @ switched[1] @ switched[0] @ switched[0])
    metrics["noncommuting_construction"] = {"terminal_noise_covariance": (0.04 * np.eye(2)).tolist(),
        "population_linear_mse": risks.tolist(),
        "risk_prior": "standardized identity-population-covariance cue model; analytic risk, not a fitted probe",
        "interpretation": "specific order-sensitive construction; not a universal periodicity effect"}
    arrays.update(noncommuting_symbols=switched, noncommuting_products=switched_products,
                  noncommuting_states=switched_states)

    horizons = np.array([0, 4, 16, 32])
    amplitudes = 0.8**horizons
    amplitude_states = amplitudes[:, None, None] * cues
    amplitude_dimensions = [covariance_dimension(state) for state in amplitude_states]
    close("isotropic_decay_preserves_pr", [d["participation_ratio"] for d in amplitude_dimensions], np.full(4, 4.0))
    close("isotropic_variance_decays_exactly", [d["total_variance"] for d in amplitude_dimensions], 4 * amplitudes**2)
    metrics["amplitude_only"] = {"dimensions": amplitude_dimensions,
        "horizons": horizons.tolist(), "amplitudes": amplitudes.tolist(),
        "risk_prior": "standardized identity-population-covariance cue model; analytic risk, not a fitted probe",
        "terminal_noise_linear_mse": [linear_recovery_risk(g * np.eye(4), 0.01 * np.eye(4)) for g in amplitudes]}
    arrays.update(amplitude_horizons=horizons, amplitude_states=amplitude_states)

    # Smooth forced maps: translate the two attracting fixed points by driver phase.
    phase_offsets = np.array([0.2, -0.2])
    labels = np.repeat([-1.0, 1.0], 64)
    perturbations = np.tile(np.linspace(-0.05, 0.05, 64), 2)
    state = labels + perturbations + phase_offsets[0]
    trajectory = [state.copy()]
    for step in range(16):
        relative = state - phase_offsets[step % 2]
        state = 1.5 * relative - 0.5 * relative**3 + phase_offsets[(step + 1) % 2]
        trajectory.append(state.copy())
    trajectory = np.asarray(trajectory)
    recovered = np.sign(state - phase_offsets[0])
    exact_cycle = labels[:, None] + phase_offsets[None, :]
    local_derivatives = 1.5 - 1.5 * (exact_cycle - phase_offsets[None, :])**2
    tube_derivative_bound = 1.5 * (2 * 0.05 + 0.05**2)
    tube_next_error_bound = 1.5 * 0.05**2 + 0.5 * 0.05**3
    close("categorical_labels_retained", recovered, labels, atol=0, rtol=0)
    close("categorical_within_basin_convergence", state - phase_offsets[0], labels)
    close("categorical_closed_two_step_reference", 1.5 * labels - 0.5 * labels**3, labels)
    close("categorical_reference_local_derivatives_zero", local_derivatives, np.zeros_like(local_derivatives))
    above("categorical_neighborhood_is_contracting", 1 - tube_derivative_bound, 0.8)
    above("categorical_neighborhood_is_invariant", 0.05 - tube_next_error_bound, 0)
    close("categorical_terminal_cycle_closure", trajectory[-1], trajectory[-3])
    above("categorical_basins_remain_separated", float(state[labels > 0].min() - state[labels < 0].max()), 1.9)
    metrics["categorical_forced_basins"] = {"classification_accuracy": float(np.mean(recovered == labels)),
        "terminal_dimension": covariance_dimension(state[:, None]),
        "initial_perturbation_radius": 0.05, "two_step_reference_local_gain": 0.0,
        "neighborhood_absolute_derivative_bound": tube_derivative_bound,
        "neighborhood_next_error_bound": tube_next_error_bound,
        "closed_reference_period_steps": 2,
        "closure_absolute_residual": float(np.max(np.abs(trajectory[-1] - trajectory[-3]))),
        "qualifier": "contraction is local within the two constructed basins, not global synchronization"}
    arrays.update(categorical_labels=labels, categorical_initial_perturbations=perturbations,
                  categorical_phase_offsets=phase_offsets, categorical_trajectory=trajectory,
                  categorical_reference_cycles=exact_cycle, categorical_local_derivatives=local_derivatives)

    # Exact continuously varying required coordinates; nuisance directions contract.
    continuous_product = np.diag([1.0, 1.0, 0.5**16, 0.5**16])
    continuous_states = cues @ continuous_product.T
    close("continuous_required_coordinates_retained", continuous_states[:, :2], cues[:, :2])
    close("continuous_neutral_required_directions", continuous_product[:, :2], np.eye(4)[:, :2])
    close("continuous_nuisance_contraction", continuous_states[:, 2:], 0.5**16 * cues[:, 2:])
    above("continuous_dimension_reduction_is_task_preserving", covariance_dimension(cues)["participation_ratio"] - covariance_dimension(continuous_states)["participation_ratio"], 1.9)
    metrics["continuous_neutral_memory"] = {"required_coordinates": [0, 1], "nuisance_coordinates": [2, 3],
        "required_mse": float(np.mean((continuous_states[:, :2] - cues[:, :2])**2)),
        "before_dimension": covariance_dimension(cues), "after_dimension": covariance_dimension(continuous_states),
        "qualifier": "exact smooth linear memory coordinates; no claim of noise robustness"}
    arrays.update(continuous_states=continuous_states, continuous_product=continuous_product)

    # Fresh independent process noise keeps population covariance I, while old cue
    # gain decays. Empirical moments have a prespecified finite-sample tolerance.
    noise_cues = isotropic_cues(rng, 8192, 4)
    innovations = rng.normal(size=(32, len(noise_cues), 4)) * np.sqrt(1 - 0.8**2)
    noise_states = [noise_cues.copy()]
    state = noise_cues.copy()
    for update in range(32):
        state = 0.8 * state + innovations[update]
        if update + 1 in horizons:
            noise_states.append(state.copy())
    noise_states = np.asarray(noise_states)
    noise_dimensions = [covariance_dimension(state) for state in noise_states]
    noise_variances = 1 - amplitudes**2
    # Uniform sampling over these fixed cue rows has population covariance k I;
    # their unbiased *empirical* covariance is I. Keep the N versus N-1 distinction.
    cue_population_variance = (len(noise_cues) - 1) / len(noise_cues)
    state_population_variances = cue_population_variance * amplitudes**2 + noise_variances
    decoder_gains = cue_population_variance * amplitudes / state_population_variances
    empirical_mse = np.mean(np.sum((noise_states * decoder_gains[:, None, None] - noise_cues)**2, axis=2), axis=1)
    expected_mse = 4 * cue_population_variance * noise_variances / state_population_variances
    close("fresh_noise_covariance_sustained", [np.cov(s, rowvar=False) for s in noise_states], np.repeat(np.eye(4)[None], 4, axis=0), atol=0.08, rtol=0)
    close("fresh_noise_linear_decoder_risk_matches", empirical_mse, expected_mse, atol=0.12, rtol=0)
    above("fresh_noise_linear_cue_recovery_deteriorates", empirical_mse[-1] - empirical_mse[0], 3.8)
    above("fresh_noise_pr_remains_high", min(d["participation_ratio"] for d in noise_dimensions), 3.9)
    metrics["fresh_noise_not_old_memory"] = {"horizons": horizons.tolist(), "dimensions": noise_dimensions,
        "cue_gains": amplitudes.tolist(), "conditional_noise_variances": noise_variances.tolist(),
        "cue_population_variance_per_coordinate": cue_population_variance,
        "state_population_variances": state_population_variances.tolist(),
        "empirical_population_decoder_mse": empirical_mse.tolist(), "analytic_linear_decoder_mse": expected_mse.tolist(),
        "noise_location": "independent Gaussian innovation after every state update; covariance 0.36 I",
        "cue_distribution": "distinct centered orthogonal finite design with identity empirical covariance",
        "qualifier": "linear population-risk calibration; Gaussian-prior nonlinear-MMSE claim is not made"}
    arrays.update(noise_cues=noise_cues, noise_innovations=innovations, noise_states=noise_states,
                  noise_decoder_gains=decoder_gains)
    return metrics, checks, arrays


def _write_json(path: Path, value: dict) -> None:
    if path.name == "status.json":
        value = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "cluster": os.environ.get("CLUV_CLUSTER_NAME"), **value}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _manifest(output: Path) -> None:
    _write_json(output / "manifest.json", {"schema_version": 1, "algorithm": "sha256",
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwrites an attempt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, True
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name("metrics.py"), root / "uv.lock"]
    _write_json(args.output / "config.json", {"schema_version": 1, "kind": "synthetic_mechanism_calibration",
        "seed": args.seed, "argv": sys.argv, "started_at": started,
        "git_revision": revision, "dirty_checkout": dirty,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.exists()},
        "python": platform.python_version(), "numpy": np.__version__, "host": platform.node(),
        "dtype": "float64", "analytic_absolute_tolerance": ATOL, "analytic_relative_tolerance": RTOL,
        "noise_moment_absolute_tolerance": 0.08, "noise_mse_absolute_tolerance": 0.12,
        "noise_observations": 8192, "noise_horizon": 32,
        "sampling_unit": "distinct cue under fixed complete body and terminal phase",
        "scope": "analytic and finite-sample calibration, not trained-model evidence",
        "product_label": "finite_time_cue_propagator"})
    _write_json(args.output / "status.json", {"schema_version": 1, "state": "running", "started_at": started,
        "scientific_decision": "pending", "updated_at": started})
    try:
        metrics, checks, arrays = calibration(args.seed)
        _write_json(args.output / "metrics.json", metrics)
        np.savez_compressed(args.output / "arrays.npz", **arrays)
        passed = all(check["passed"] for check in checks.values())
        _write_json(args.output / "verification.json", {"schema_version": 1, "passed": passed,
            "verifier": "declared analytic expectations and prespecified empirical tolerances",
            "checks": checks})
        _write_json(args.output / "status.json", {"schema_version": 1,
            "state": "verified" if passed else "failed", "exit_code": 0 if passed else 1,
            "scientific_decision": "pending", "updated_at": datetime.now(timezone.utc).isoformat()})
        _manifest(args.output)
    except Exception as error:
        _write_json(args.output / "verification.json", {"schema_version": 1, "passed": False,
            "error": f"{type(error).__name__}: {error}"})
        _write_json(args.output / "status.json", {"schema_version": 1, "state": "failed", "exit_code": 1,
            "failure_category": "calibration_exception", "scientific_decision": "pending",
            "updated_at": datetime.now(timezone.utc).isoformat()})
        _manifest(args.output)
        raise
    print(json.dumps({"output": str(args.output), "checks": len(checks), "passed": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
