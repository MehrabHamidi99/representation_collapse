"""Realized body statistics for the matched-order experiment.

Descriptive input/encoder measurements only. No model, training, statistical
inference or Floquet certification is performed. Callers own reporting and
source/encoder identities; these helpers do not admit an experiment.
"""
from __future__ import annotations

from collections import Counter
from itertools import groupby
import numpy as np

from .vector_task import MOTIFS, proper_period


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def sequence_statistics(features, *, max_lag=24):
    """Time-centered real features, cyclic lag covariances and FFT power.

    Cov(lag) = centered.T @ roll(centered, -lag) / T; the circular boundary
    is deliberate and is not an observed extra transition. One-sided powers
    sum across features and use doubled interior rFFT bins divided by T**2,
    so their sum equals mean squared centered feature norm (Parseval).
    Empty/constant sequences have undefined normalized power, returned as null.
    No stationarity or independent-sampling assumption is made.
    """
    max_lag = _integer(max_lag, "max_lag")
    if not isinstance(features, np.ndarray):
        raise ValueError("explicitly typed NumPy features required; no implicit mixed-sequence coercion")
    original = np.asarray(features)
    if original.ndim != 2 or original.dtype.kind not in "fiu":
        raise ValueError("real numeric [updates, features] matrix required")
    time, width = original.shape
    if not 1 <= width <= 128 or time > 4096 or original.size > 65536:
        raise ValueError("bounded input/encoder feature matrix required")
    if max_lag > 128:
        raise ValueError("max_lag exceeds the descriptive diagnostic cap")
    if original.dtype.kind == "f" and original.dtype.itemsize > 8:
        raise ValueError("no implicit reduction of extended floating precision")
    if original.size and original.dtype.kind in "iu" and (int(original.min()) < -(2 ** 53) or int(original.max()) > 2 ** 53):
        raise ValueError("integer features outside the exact float64 range")
    lag = min(max_lag, time - 1) if time else -1
    if (lag + 1) * width * width > 1000000 or time * width * width * (lag + 1) > 128000000:
        raise ValueError("requested cyclic cross-covariance work exceeds cap")
    x = np.array(original, dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("finite feature values required")
    base = {"updates": time, "features": width,
            "input_dtype": original.dtype.str, "computation_dtype": "float64",
            "centering": "per-body feature mean over all updates",
            "covariance_denominator": "T, not T-1 or T-lag",
            "lag_boundary": "circular wrap, descriptive only",
            "frequency_unit": "cycles per recurrent update",
            "independent_training_replicates": 0}
    if not time:
        return {**base, "mean": None, "lags": [], "cyclic_cross_covariance": [],
                "centered_variance_energy": 0.0, "frequency": [], "power": [],
                "normalized_power_defined": False, "normalized_power": None,
                "empty_sequence": True}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = x.mean(axis=0)
            centered = x - mean
            covariances = [centered.T @ np.roll(centered, -index, axis=0) / time
                           for index in range(lag + 1)]
            transformed = np.fft.rfft(centered, axis=0)
            power = np.sum(np.abs(transformed) ** 2, axis=1) / (time * time)
            if time % 2:
                power[1:] *= 2
            else:
                power[1:-1] *= 2
            energy = float(np.sum(centered * centered) / time)
            defined = energy > 0
            normalized = power / energy if defined else None
    except FloatingPointError as error:
        raise ValueError("feature arithmetic overflow; no implicit rescaling") from error
    if not np.isfinite(power).all() or not all(np.isfinite(c).all() for c in covariances) or not np.isfinite(energy):
        raise ValueError("nonfinite descriptive feature statistic")
    if normalized is not None and not np.isfinite(normalized).all():
        raise ValueError("nonfinite normalized power")
    return {**base, "mean": mean.tolist(), "lags": list(range(lag + 1)),
            "cyclic_cross_covariance": [c.tolist() for c in covariances],
            "centered_variance_energy": energy, "frequency": np.fft.rfftfreq(time).tolist(),
            "power": power.tolist(), "normalized_power_defined": defined,
            "normalized_power": normalized.tolist() if defined else None,
            "empty_sequence": False}


def _ngrams(text, order):
    if not text:
        return {}
    extended = text + text[:order - 1]
    return dict(sorted(Counter(extended[i:i + order] for i in range(len(text))).items()))


def body_statistics(body, *, q=1, max_lag=24):
    """Analyze a realized A/B/X body, excluding WRITE and QUERY observations."""
    q = _integer(q, "q", 1)
    if q > 64:
        raise ValueError("dwell expansion exceeds the diagnostic cap")
    if not isinstance(body, str) or len(body) > 4096 or set(body) - set("ABX"):
        raise ValueError("bounded A/B/X body string required")
    unit = 12 * q
    library = {"".join(symbol * q for symbol in motif) for motif in MOTIFS}
    if len(body) % unit or any(body[i:i + unit] not in library for i in range(0, len(body), unit)):
        raise ValueError("body must be tiled by registered dwell-expanded motifs")
    runs = Counter(len(tuple(run)) for _, run in groupby(body))
    matrix = np.zeros((len(body), 3), dtype=np.float64)
    for i, symbol in enumerate(body):
        matrix[i, "ABX".index(symbol)] = 1
    return {"updates": len(body), "q": q, "feature_order": ["A", "B", "X"],
            "symbol_counts": {symbol: body.count(symbol) for symbol in "ABX"},
            "cyclic_bigrams": _ngrams(body, 2), "cyclic_trigrams": _ngrams(body, 3),
            "x_skipping_cyclic_letter_bigrams": _ngrams(body.replace("X", ""), 2),
            "linear_run_length_histogram": {str(n): count for n, count in sorted(runs.items())},
            "proper_whole_body_tiling_period": proper_period(body),
            "distinct_motif_count": len({body[i:i + unit] for i in range(0, len(body), unit)}),
            "features": sequence_statistics(matrix, max_lag=max_lag)}


def matched_body_audit(periodic, control, *, repeats, q=1, max_lag=24):
    """Validate declared arm controls and retain realized uncontrolled differences.

    R0 and R1 are identical-arm anchors. A nonempty R>1 control must have no
    proper whole-body tiling period, while local motif repetitions are allowed.
    Matching low-order statistics is not independence from higher-order input
    predictability; these measurements establish no state-memory mechanism.
    """
    repeats = _integer(repeats, "repeats")
    q = _integer(q, "q", 1)
    left = body_statistics(periodic, q=q, max_lag=max_lag)
    right = body_statistics(control, q=q, max_lag=max_lag)
    unit = 12 * q
    if len(periodic) != repeats * unit or len(control) != len(periodic):
        raise ValueError("both arms must have the exact declared equal horizon")
    if repeats <= 1:
        if periodic != control:
            raise ValueError("R0/R1 anchors must have identical arms")
    elif periodic != periodic[:unit] * repeats or proper_period(control) is not None:
        raise ValueError("periodic arm must repeat one motif; control must not tile")
    matched = {key: left[key] == right[key] for key in
               ("updates", "symbol_counts", "cyclic_bigrams", "linear_run_length_histogram")}
    matched["start_context"] = periodic[:2 * q] == control[:2 * q]
    matched["end_context"] = periodic[-2 * q:] == control[-2 * q:]
    matched["final_motif"] = periodic[-unit:] == control[-unit:]
    if not all(matched.values()):
        raise ValueError("declared matched body controls differ")
    final = periodic[-unit:] if repeats else ""
    return {"periodic": left, "control": right, "matched": matched,
            "repeats": repeats, "q": q,
            "cyclic_trigrams_equal": left["cyclic_trigrams"] == right["cyclic_trigrams"],
            "x_skipping_letter_bigrams_equal": left["x_skipping_cyclic_letter_bigrams"] == right["x_skipping_cyclic_letter_bigrams"],
            "final_motif_exposures": {
                "periodic": sum(periodic[i:i + unit] == final for i in range(0, len(periodic), unit)),
                "control": sum(control[i:i + unit] == final for i in range(0, len(control), unit))},
            "interpretation": "realized input-order controls; no trained-memory or Floquet claim"}
