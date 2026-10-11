"""Small dense reference measurements; use matrix-free methods for large states.

The input rows to covariance_dimension are independent observations at a
specified phase, not arbitrary samples pooled across time or phase.
"""

from __future__ import annotations

import numpy as np


def covariance_dimension(samples: np.ndarray, *, relative_tolerance: float = 1e-10) -> dict:
    """Centered covariance spectrum and two distinct effective dimensions.

    A zero-variance cloud has undefined normalized dimensions, encoded as zero
    sentinels together with normalized_dimension_defined=False.
    Numerical rank uses eigenvalue > relative_tolerance * maximum eigenvalue.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("samples must have shape (at least 2 observations, at least 1 feature)")
    if not np.isfinite(x).all() or not 0 < relative_tolerance < 1:
        raise ValueError("finite samples and a relative tolerance in (0, 1) required")
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - x.mean(axis=0, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError("centering overflow; rescale samples before covariance measurement")
    with np.errstate(over="ignore", invalid="ignore"):
        eigenvalues = np.linalg.svd(centered, compute_uv=False) ** 2 / (x.shape[0] - 1)
        total = float(eigenvalues.sum())
    if not np.isfinite(eigenvalues).all() or not np.isfinite(total):
        raise ValueError("covariance spectrum overflow; rescale samples before measurement")
    if total == 0:
        pr = entropy_rank = 0.0
        numerical_rank = 0
    else:
        probabilities = eigenvalues / total
        nonzero = probabilities[probabilities > 0]
        pr = float(1 / np.square(probabilities).sum())
        entropy_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
        numerical_rank = int(np.count_nonzero(eigenvalues > relative_tolerance * eigenvalues[0]))
    return {
        "participation_ratio": pr,
        "entropy_effective_rank": entropy_rank,
        "numerical_rank": numerical_rank,
        "total_variance": total,
        "normalized_dimension_defined": total > 0,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "observations": x.shape[0],
        "features": x.shape[1],
        "relative_tolerance": relative_tolerance,
    }


def ordered_product(jacobians: np.ndarray) -> np.ndarray:
    """Return J[T-1] @ ... @ J[0] for column-vector perturbations."""
    j = np.asarray(jacobians, dtype=np.float64)
    if j.ndim != 3 or j.shape[0] < 1 or j.shape[1] < 1 or j.shape[1] != j.shape[2]:
        raise ValueError("jacobians must have shape (positive period, state, state)")
    if not np.isfinite(j).all():
        raise ValueError("jacobians must be finite")
    result = np.eye(j.shape[1])
    for step in j:
        result = step @ result
    if not np.isfinite(result).all():
        raise ValueError("product overflow; use stabilized or matrix-free estimation")
    return result


def period_spectrum(jacobians: np.ndarray) -> dict:
    """Spectrum of a declared periodic variational system.

    This function does not verify periodicity of the input trajectory. The
    caller must establish it before interpreting eigenvalues as Floquet
    multipliers. Singular values measure finite-time perturbation gain.
    """
    product = ordered_product(jacobians)
    multipliers = np.linalg.eigvals(product)
    singular_values = np.linalg.svd(product, compute_uv=False)
    return {
        "period_steps": len(jacobians),
        "multipliers_real": multipliers.real.tolist(),
        "multipliers_imag": multipliers.imag.tolist(),
        "multiplier_moduli": np.abs(multipliers).tolist(),
        "singular_values": singular_values.tolist(),
        "spectral_radius": float(np.abs(multipliers).max()),
        "largest_finite_time_gain": float(singular_values[0]),
    }
