r"""Independently audit saved periodicity-v1 arrays; never load weights or use Torch.

Example (no GPU, network, new trajectories or model calls)::

    OPENBLAS_NUM_THREADS=1 .venv/bin/python code/analysis/reduce_periodicity.py \
        --artifact-dir runs/downloaded/periodicity-cell0 \
        --artifact-dir runs/downloaded/periodicity-cell1 \
        --output runs/ops/periodicity-independent.json

The frozen config/checkpoint/source identities are the trust root. Manifest
integrity is checked before NPZ loading. Products are reconstructed directly
from saved single-step Jacobians, separately in replay dtype and float64.
Approximate closure is never promoted to a certified periodic orbit. Original
pilot probe train/test clouds are outside this schema, so probe-floor robustness
is explicitly pending. Reports use JSON null for undefined ratios/log gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = "458bc9cfce474b8e11e433e33bf132faf5032387"
DEFAULT_MODULE_SHA256 = "5ff03bcddc5b0c3810eec8c02b4efbbd86da370567e876eaa150dd446bcc25e8"
MOTIFS = ("AXAXAXBXBXBX", "AXAXBXAXBXBX", "AXAXBXBXAXBX", "AXBXAXAXBXBX", "AXBXBXAXAXBX")
REQUIRED_FILES = {"config.json", "provenance.json", "status.json", "metrics.json",
                  "inputs.npz", "trajectories.npz", "tangents.npz"}
REPORT_ATOL = 0.0
REPORT_RTOL = 1e-9
SPECTRUM_ATOL = 1e-12
SPECTRUM_RTOL = 1e-8


class InvalidArtifact(ValueError):
    """Integrity, schema, provenance or saved numerical comparison failure."""


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def bad_constant(value):
        raise InvalidArtifact(f"nonfinite JSON constant in {path}: {value}")
    packet = json.loads(Path(path).read_text(), parse_constant=bad_constant)
    def finite_tree(value):
        if isinstance(value, float):
            require(np.isfinite(value), f"nonfinite JSON number in {path}")
        elif isinstance(value, dict):
            for child in value.values(): finite_tree(child)
        elif isinstance(value, list):
            for child in value: finite_tree(child)
    finite_tree(packet)
    return packet


def compare(saved, actual, context, *, atol=REPORT_ATOL, rtol=REPORT_RTOL):
    """Audit reported numbers with an explicit scalar/array comparison contract."""
    a, b = np.asarray(saved), np.asarray(actual)
    require(a.shape == b.shape, f"{context}: reported/reduced shapes differ")
    require(np.isfinite(a).all() and np.isfinite(b).all(), f"{context}: nonfinite comparison")
    require(np.allclose(a, b, atol=atol, rtol=rtol), f"{context}: reported metric disagrees with raw arrays")


def ordered_product(jacobians, dtype=np.float64):
    """Column perturbations: Phi = J[last] @ ... @ J[first]."""
    steps = np.asarray(jacobians)
    require(steps.ndim == 3 and steps.shape[1] == steps.shape[2] and steps.shape[0] > 0,
            "step Jacobians must be a nonempty square-matrix sequence")
    result = np.eye(steps.shape[1], dtype=dtype)
    for step in steps:
        result = step.astype(dtype, copy=False) @ result
    require(np.isfinite(result).all(), "nonfinite chronological product")
    return result


def closure(a, b, atol, rtol, floor):
    """Global L2/Frobenius and coordinate gates, independent of production helper."""
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(left.shape == right.shape and left.size > 0, "closure requires equal nonempty shapes")
    require(np.isfinite(left).all() and np.isfinite(right).all(), "nonfinite closure data")
    delta = np.abs(left - right)
    absolute = float(np.linalg.norm(left - right))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), floor)
    coordinate_pass = bool(np.all(delta <= atol + rtol * np.maximum(np.abs(left), np.abs(right))))
    relative = absolute / scale
    return {"absolute": absolute, "relative": relative, "reference_scale": scale,
            "maximum_coordinate_absolute": float(delta.max()),
            "coordinatewise_passed": coordinate_pass, "absolute_tolerance": atol,
            "relative_tolerance": rtol, "scale_floor": floor,
            "passed": bool(absolute <= atol and relative <= rtol and coordinate_pass)}


def ulp_diagnostics(a, b):
    """Coordinate spacing ratios in the saved assay dtype, including zero states.

    This is a magnitude-to-local-spacing ratio, not integer representable-number
    distance. A relative or ULP summary alone does not imply lost information.
    """
    a, b = np.asarray(a), np.asarray(b)
    require(a.dtype == b.dtype and a.dtype in (np.dtype("float32"), np.dtype("float64")),
            "ULP diagnostics require matching saved float32/float64 dtypes")
    require(a.shape == b.shape and a.size > 0, "ULP diagnostics require equal nonempty shapes")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite ULP data")
    spacing = np.maximum(np.abs(np.spacing(a)).astype(np.float64),
                         np.abs(np.spacing(b)).astype(np.float64))
    require(np.isfinite(spacing).all() and np.all(spacing > 0), "nonfinite/zero reference coordinate spacing")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    changed = delta != 0
    # Finite bounded recurrent states have finite ratios. Huge adversarial inputs
    # can overflow a spacing ratio; reject rather than emit invalid JSON.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratios = delta / spacing
    require(np.isfinite(ratios).all(), "nonfinite coordinate spacing ratio")
    integer_dtype = np.uint32 if a.dtype.itemsize == 4 else np.uint64
    return {"dtype": str(a.dtype), "coordinate_count": a.size,
            "numerically_changed_coordinates": int(changed.sum()),
            "bitwise_changed_coordinates": int(np.count_nonzero(a.view(integer_dtype) != b.view(integer_dtype))),
            "maximum_coordinate_spacing_ratio": float(ratios.max()),
            "median_coordinate_spacing_ratio": float(np.median(ratios)),
            "minimum_reference_spacing": float(spacing.min()),
            "maximum_reference_spacing": float(spacing.max()),
            "bitwise_equal": bool(np.array_equal(a.view(integer_dtype), b.view(integer_dtype))),
            "ratio_convention": "absolute coordinate difference / maximum absolute local np.spacing in saved dtype"}


def eigenvalue_matching(saved, reduced, atol=SPECTRUM_ATOL, rtol=SPECTRUM_RTOL):
    """One-to-one tolerance matching preserves complex-spectrum multiplicity."""
    left, right = np.asarray(saved), np.asarray(reduced)
    require(left.shape == right.shape and left.ndim == 1, "eigenvalue matching shape mismatch")
    residual = np.abs(left[:, None] - right[None, :])
    allowed = residual <= atol + rtol * np.maximum(np.abs(left[:, None]), np.abs(right[None, :]))
    candidates = [np.flatnonzero(row).tolist() for row in allowed]
    match = [-1] * len(right)
    def augment(index, visited):
        for candidate in candidates[index]:
            if candidate in visited: continue
            visited.add(candidate)
            if match[candidate] == -1 or augment(match[candidate], visited):
                match[candidate] = index
                return True
        return False
    for index in range(len(left)):
        # An inexpensive free-edge pass avoids recursive chains for repeated
        # zero/clustered eigenvalues while preserving a full matching guarantee.
        free = next((candidate for candidate in candidates[index] if match[candidate] == -1), None)
        if free is not None:
            match[free] = index
        else:
            require(augment(index, set()), "eigenvalue multiplicity/tolerance matching failed")
    return {"one_to_one_matching_passed": True,
            "maximum_matched_absolute_residual": float(max(residual[original, candidate] for candidate, original in enumerate(match))),
            "absolute_tolerance": atol, "relative_tolerance": rtol}


def audit_distance(saved, actual, context):
    for key, value in actual.items():
        require(key in saved, f"{context}: missing reported distance field {key}")
        if isinstance(value, bool):
            require(saved[key] is value, f"{context}: reported {key} flag disagrees")
        else:
            compare(saved[key], value, context + ":" + key)


class Arrays:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.values = {key: data[key] for key in data.files}
        self.used = set()
        for key, value in self.values.items():
            require(value.dtype.kind in "biuf" and np.isfinite(value).all(),
                    f"{path.name}:{key}: unsupported/nonfinite raw array")

    def get(self, key, shape=None, dtype=None):
        require(key in self.values, f"required raw array missing: {key}")
        value = self.values[key]
        self.used.add(key)
        if shape is not None:
            require(value.shape == shape, f"{key}: expected shape {shape}, got {value.shape}")
        if dtype is not None:
            require(value.dtype == np.dtype(dtype), f"{key}: dtype does not match assay precision")
        return value

    def complete(self, context):
        require(self.used == set(self.values), f"{context}: unexpected/unreduced arrays: {sorted(set(self.values)-self.used)}")


def proper_period(sequence):
    for period in range(1, len(sequence) // 2 + 1):
        if len(sequence) % period == 0 and np.array_equal(sequence, np.tile(sequence[:period], (len(sequence) // period, 1))):
            return period
    return None


def input_checks(inputs, ids, memory, query, task, repeats):
    """Independently reconstruct observed WRITE/body/QUERY vectors from literals."""
    require(ids.dtype.kind in "iu" and np.all((ids >= 0) & (ids < len(MOTIFS))), "invalid motif IDs")
    symbols = "".join(MOTIFS[int(index)] for index in ids)
    require(len(symbols) == 12 * repeats, "motif/input body length mismatch")
    expected = np.zeros((len(memory), 17 + len(symbols), 38), dtype=np.float32)
    for cue in range(len(memory)):
        for coordinate in range(16):
            expected[cue, coordinate, 0] = 1
            expected[cue, coordinate, 5 + coordinate] = 1
            if task == "categorical":
                require(np.all(memory == memory.astype(np.int64)) and np.all((memory >= 0) & (memory < 16)),
                        "invalid categorical memory")
                expected[cue, coordinate, 21 + int(memory[cue, coordinate])] = 1
            else:
                require(np.all(np.abs(memory) <= 1), "continuous memory outside [-1,1]")
                expected[cue, coordinate, 37] = memory[cue, coordinate]
        for time_index, symbol in enumerate(symbols):
            expected[cue, 16 + time_index, {"A": 1, "B": 2, "X": 3}[symbol]] = 1
        expected[cue, -1, 4] = 1
        expected[cue, -1, 5 + int(query[cue])] = 1
    require(np.array_equal(inputs, expected), "saved inputs do not match literal memory/motifs/query")
    body = inputs[0, 16:-1]
    return {"body_steps": len(body), "driver_periodic_over_observed_body":
            bool(repeats >= 2 and np.array_equal(body[:-12], body[12:])),
            "proper_whole_body_period": proper_period(body),
            "last_motif_id": int(ids[-1]),
            "symbol_counts": {symbol: symbols.count(symbol) for symbol in "ABX"}}


def cycles(states, repeats, tol):
    body = states[16:]
    result = []
    for cycle_index in range(repeats - 1):
        left = body[12 * cycle_index:12 * (cycle_index + 1) + 1]
        right = body[12 * (cycle_index + 1):12 * (cycle_index + 2) + 1]
        record = closure(left, right, tol["state_absolute"], tol["state_relative"], tol["scale_floor"])
        record.update(cycle_pair=[cycle_index + 1, cycle_index + 2], ulp=ulp_diagnostics(left, right),
                      phase_l2_residual=np.linalg.norm(left.astype(np.float64)-right.astype(np.float64), axis=1).tolist(),
                      phase_max_coordinate_residual=np.abs(left.astype(np.float64)-right.astype(np.float64)).max(axis=1).tolist())
        result.append(record)
    return result


def reduce_directory(directory, frozen, *, expected_source=DEFAULT_SOURCE,
                     expected_module_sha256=DEFAULT_MODULE_SHA256, frozen_sha256=None):
    """Fail closed on invalid identities/schema/nonfinite arrays or metric drift.

    A numerical-contract failure can be reported even for a complete artifact;
    a disagreement with its own worker summary raises InvalidArtifact instead.
    No checkpoint/model code is loaded: checkpoint identity is checked against
    the trusted frozen registry, and external file verification remains recorded.
    """
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    manifest_digest = sha256(directory / "manifest.json")
    require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    hashes = manifest.get("sha256", {})
    require(REQUIRED_FILES <= set(hashes), "manifest lacks required periodicity artifacts")
    for name, digest in hashes.items():
        require(Path(name).name == name and name not in ("", ".", ".."), "unsafe manifest path")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"missing/nonregular artifact: {name}")
        require(sha256(path) == digest, f"manifest hash mismatch: {name}")
    config, origin, status, metrics = [read_json(directory / name) for name in
                                      ("config.json", "provenance.json", "status.json", "metrics.json")]
    require(status.get("state") == "completed", "diagnostic artifact is not completed")
    require(config.get("source_commit") == expected_source and origin.get("git_revision") == expected_source
            and origin.get("diagnostic_source_commit") == expected_source, "diagnostic source identity mismatch")
    require(origin.get("source_verified") is True and origin.get("dirty_checkout") is False,
            "diagnostic source was unverified or dirty")
    require(origin.get("diagnostic_module_sha256") == expected_module_sha256, "diagnostic module hash mismatch")
    if frozen_sha256 is not None:
        require(frozen_sha256 in origin.get("source_sha256", {}).values(), "frozen config hash absent from provenance")
    training_origin = origin.get("training_origin", {})
    require(training_origin.get("git_revision") == frozen["training_source_commit"]
            and training_origin.get("dirty_checkout") is False, "training source mismatch/dirty")
    model, task, seed = config["model"], config["task"], config["seed"]
    cell = next((row for row in frozen.get("cells", []) if (row["model"], row["task"], row["seed"]) == (model, task, seed)), None)
    require(cell is not None, "cell not in trusted frozen registry")
    for packet in (config, origin):
        require(packet.get("checkpoint_sha256") == cell["checkpoint_sha256"], "checkpoint hash outside frozen registry")
    require(origin.get("checkpoint_path") == cell["checkpoint"], "checkpoint path outside frozen registry")
    require(config.get("cluster") == cell["cluster"] and status.get("cluster") == cell["cluster"], "cluster identity mismatch")
    for name in ("horizons", "derivative_horizons", "cue_histories", "driver_panels", "derivative_cues",
                 "precisions", "dwell", "hidden_size", "input_bottleneck", "tolerances", "training_source_commit"):
        require(config.get(name) == frozen.get(name), f"effective/frozen config mismatch: {name}")
    require(model in ("gru", "lstm") and task in ("categorical", "continuous"), "unknown model/task")
    require(config.get("no_retraining") is True and config.get("allow_tf32") is False,
            "retraining/TF32 contract mismatch")
    require(origin.get("cuda_matmul_allow_tf32") is False and origin.get("cudnn_allow_tf32") is False
            and origin.get("float32_matmul_precision") == "highest"
            and origin.get("precision") == "FP32 native replay and FP64 cast-copy fixed-weight replay; derivatives in assay dtype",
            "recorded precision/TF32 controls missing/changed")
    dimension = config["hidden_size"] * (2 if model == "lstm" else 1)
    require(config.get("state_dimension") == dimension and config.get("state_order") == (["h", "c"] if model == "lstm" else ["h"]),
            "complete recurrent-state packing mismatch")
    require(metrics.get("schema_version") == 1 and metrics.get("kind") == "checkpoint_periodicity_diagnostic",
            "unsupported metrics schema")
    require(metrics.get("scientific_acceptance") is False and metrics.get("native_competence_established") is False,
            "worker summary claims unsupported scientific/native acceptance")
    inputs, trajectories, tangents = [Arrays(directory / name) for name in ("inputs.npz", "trajectories.npz", "tangents.npz")]
    histories = config["cue_histories"]
    query = inputs.get("query_coordinates", (histories,))
    require(query.dtype.kind in "iu" and np.all(query == 0), "v1 diagnostic must query coordinate zero")
    assays = {}
    failures = []
    window_count = 0
    for panel in range(config["driver_panels"]):
        memory = inputs.get(f"panel{panel}_memory", (histories, 16))
        compare(inputs.get(f"panel{panel}_query_targets", (histories,)), memory[:, 0], "query targets", atol=0, rtol=0)
        require(histories == 2 and np.array_equal(memory[0, 8:], memory[1, 8:])
                and not np.array_equal(memory[0, :8], memory[1, :8]), "paired distinct required histories/shared nuisance mismatch")
        for precision in config["precisions"]:
            tol = config["tolerances"][precision]
            for repeats in config["horizons"]:
                for arm in ("periodic", "control"):
                    key = f"{precision}_panel{panel}_R{repeats}_{arm}"
                    require(key in metrics["assays"], f"missing assay metric {key}")
                    reported = metrics["assays"][key]
                    literal = inputs.get(key + "_inputs", (histories, 17 + 12 * repeats, 38), "float32")
                    ids = inputs.get(key + "_motif_ids", (repeats,))
                    driver = input_checks(literal, ids, memory, query, task, repeats)
                    require(arm != "periodic" or np.all(ids == ids[-1]), "periodic motif IDs vary")
                    require(arm != "control" or repeats == 1 or driver["proper_whole_body_period"] is None,
                            "control violates proper whole-string nonperiodicity")
                    for name in ("body_steps", "driver_periodic_over_observed_body"):
                        require(reported.get(name) == driver[name], f"{key}: reported driver mismatch")
                    require(reported.get("primitive_motif_steps") == 12 and reported.get("query_included_in_dynamics") is False
                            and reported.get("observed_adjacent_cycle_comparisons") == repeats - 1, f"{key}: chronology mismatch")
                    states_by_cue, native_by_cue, cue_records = [], [], []
                    for cue in range(histories):
                        ckey = key + f"_cue{cue}"
                        shape = (17 + 12 * repeats, dimension)
                        states = trajectories.get(ckey + "_states", shape, precision)
                        native = trajectories.get(ckey + "_native_core_states", shape, precision)
                        full_state = trajectories.get(ckey + "_native_full_pre_query_state", (dimension,), precision)
                        full_hidden = trajectories.get(ckey + "_native_full_hidden_trajectory", (shape[0] - 1, config["hidden_size"]), precision)
                        prediction = trajectories.get(ckey + "_native_query_prediction", (16 if task == "categorical" else 1,), precision)
                        require(np.all(states[0] == 0) and np.all(native[0] == 0), "nonzero initial state")
                        states_by_cue.append(states)
                        native_by_cue.append(native)
                        manual_cycles, native_cycles = cycles(states, repeats, tol), cycles(native, repeats, tol)
                        compare(trajectories.get(ckey + "_cycle_state_closure_absolute", (repeats - 1,)),
                                [row["absolute"] for row in manual_cycles], ckey + ":cycle absolute")
                        compare(trajectories.get(ckey + "_cycle_state_closure_relative", (repeats - 1,)),
                                [row["relative"] for row in manual_cycles], ckey + ":cycle relative")
                        report_cue = reported["cues"][cue]
                        adapter = report_cue["adapter"]
                        adapter_values = {"manual_vs_native_step_max_absolute": float(np.max(np.abs(states-native))),
                                          "manual_vs_full_native_state_max_absolute": float(np.max(np.abs(states[-1]-full_state))),
                                          "manual_vs_full_native_hidden_trajectory_max_absolute": float(np.max(np.abs(states[1:, :config["hidden_size"]]-full_hidden)))}
                        for name, value in adapter_values.items():
                            compare(adapter[name], value, ckey + ":adapter:" + name)
                        def elementwise_close(a, b):
                            return bool(np.all(np.abs(a-b) <= tol["adapter_absolute"] + tol["adapter_relative"] * np.abs(b)))
                        forward_pass = elementwise_close(states, native) and elementwise_close(states[-1], full_state) and elementwise_close(states[1:, :config["hidden_size"]], full_hidden)
                        # Manual query logits were not saved: its scalar maximum error
                        # is reported-only, never independently reconstructed here.
                        require(adapter["absolute_tolerance"] == tol["adapter_absolute"] and adapter["relative_tolerance"] == tol["adapter_relative"], "adapter tolerance drift")
                        require(adapter["passed"] is True and forward_pass, f"{ckey}: adapter/native check fails")
                        record = {"cue": cue, "manual_cycles": manual_cycles, "native_cycles": native_cycles,
                                  "adapter_saved_forward_checks_passed": forward_pass, "adapter_max_errors": adapter_values,
                                  "manual_query_output_comparison": "unavailable: manual query prediction array was not saved",
                                  "native_query_prediction": prediction.tolist(), "tangent": None}
                        if repeats in config["derivative_horizons"] and cue < config["derivative_cues"]:
                            window_count += 1
                            current = tangents.get(ckey + "_step_jacobians", (12, dimension, dimension), precision)
                            previous = tangents.get(ckey + "_previous_step_jacobians", (12, dimension, dimension), precision)
                            saved_product = tangents.get(ckey + "_product", (dimension, dimension), precision)
                            direct = tangents.get(ckey + "_direct_jacobian", (dimension, dimension), precision)
                            product = ordered_product(current, dtype=np.dtype(precision))
                            product64 = ordered_product(current)
                            dtype_product_audit = closure(saved_product, product, tol["product_absolute"], tol["product_relative"], tol["scale_floor"])
                            require(dtype_product_audit["passed"], ckey + ": saved dtype product disagrees with independently ordered product")
                            compare(tangents.get(ckey + "_product_float64_accumulation", (dimension, dimension), "float64"), product64,
                                    ckey + ":FP64 ordered product", atol=SPECTRUM_ATOL, rtol=SPECTRUM_RTOL)
                            singular = np.linalg.svd(product64, compute_uv=False)
                            eigenvalues = np.linalg.eigvals(product64)
                            compare(tangents.get(ckey + "_singular_values", (dimension,)), singular, ckey + ":singular spectrum", atol=SPECTRUM_ATOL, rtol=SPECTRUM_RTOL)
                            saved_eigen = tangents.get(ckey + "_eigenvalues_real", (dimension,)) + 1j * tangents.get(ckey + "_eigenvalues_imag", (dimension,))
                            eigen_audit = eigenvalue_matching(saved_eigen, eigenvalues)
                            require(report_cue["tangent"]["interpretation"] == "finite_time_one_motif_propagator"
                                    and report_cue["tangent"]["exact_orbit_verified"] is False
                                    and report_cue["tangent"]["floquet_acceptance"] is False
                                    and report_cue["tangent"]["spectra_from"] == "float64 ordered accumulation of saved assay-dtype step Jacobians",
                                    ckey + ": unsupported tangent interpretation/precision claims")
                            radius = float(np.abs(eigenvalues).max())
                            compare(report_cue["tangent"]["spectral_radius"], radius, ckey + ":spectral radius", atol=SPECTRUM_ATOL, rtol=SPECTRUM_RTOL)
                            compare(report_cue["tangent"]["largest_singular_value"], singular[0], ckey + ":largest gain", atol=SPECTRUM_ATOL, rtol=SPECTRUM_RTOL)
                            phase = 12 * (repeats - 1)
                            state_closure = closure(states[16+phase-12:16+phase+1], states[16+phase:16+phase+13],
                                                    tol["state_absolute"], tol["state_relative"], tol["scale_floor"])
                            jacobian_closure = closure(previous, current, tol["jacobian_absolute"], tol["jacobian_relative"], tol["scale_floor"])
                            direct_agreement = closure(saved_product, direct, tol["product_absolute"], tol["product_relative"], tol["scale_floor"])
                            for name, value in (("state_closure", state_closure), ("jacobian_closure", jacobian_closure), ("product_direct_agreement", direct_agreement)):
                                audit_distance(report_cue["tangent"][name], value, ckey + ":" + name)
                            compatible = arm == "periodic" and driver["driver_periodic_over_observed_body"] and state_closure["passed"] and jacobian_closure["passed"]
                            require(report_cue["tangent"]["numerical_closure_compatible"] is compatible, ckey + ":closure compatibility flag drift")
                            require(report_cue["tangent"]["window_body_start"] == phase and report_cue["tangent"]["window_steps"] == 12
                                    and report_cue["tangent"]["step_order"] == "J[11] @ ... @ J[0]", ckey + ":tangent chronology drift")
                            tangent_record = {"interpretation": "finite_time_one_motif_propagator", "window_body_start": phase,
                                              "window_steps": 12, "state_closure": state_closure, "native_state_closure": native_cycles[-1],
                                              "jacobian_closure": jacobian_closure, "dtype_product_recomputation": dtype_product_audit,
                                              "product_direct_agreement": direct_agreement, "approximate_manual_closure_compatible": compatible,
                                              "exact_periodic_orbit_verified": False, "floquet_acceptance": False,
                                              "singular_values": singular.tolist(), "eigenvalues_real": eigenvalues.real.tolist(),
                                              "eigenvalues_imag": eigenvalues.imag.tolist(), "saved_eigenvalue_audit": eigen_audit,
                                              "spectral_radius": radius, "largest_singular_value": float(singular[0]),
                                              "finite_window_log_gain_per_step": float(np.log(singular[0])/12) if singular[0] > 0 else None,
                                              "zero_singular_values": int(np.count_nonzero(singular == 0)), "directional_checks": None}
                            if not direct_agreement["passed"]:
                                failures.append(ckey + ":product_direct")
                            if precision == "float64":
                                epsilon = tangents.get(ckey + "_fd_epsilon", (3,))
                                direction = tangents.get(ckey + "_fd_direction", (dimension,), precision)
                                derivatives = tangents.get(ckey + "_fd_derivatives", (3, dimension), precision)
                                jvp = tangents.get(ckey + "_autograd_jvp", (dimension,), precision)
                                compare(epsilon, [1e-4, 1e-5, 1e-6], ckey + ":FD epsilons", atol=0, rtol=0)
                                expected_direction = np.linspace(-1, 1, dimension) if dimension > 1 else np.ones(1)
                                expected_direction /= np.linalg.norm(expected_direction)
                                compare(direction, expected_direction, ckey + ":FD direction", atol=1e-15, rtol=1e-15)
                                expectation = direct.astype(np.float64) @ direction
                                fd_checks = [closure(row, expectation, 1e-8, 1e-4, tol["scale_floor"]) for row in derivatives]
                                jvp_check = closure(jvp, expectation, tol["product_absolute"], tol["product_relative"], tol["scale_floor"])
                                for index, row in enumerate(fd_checks):
                                    audit_distance(report_cue["tangent"]["directional_finite_difference"]["checks"][index], row, ckey + f":FD{index}")
                                audit_distance(report_cue["tangent"]["directional_finite_difference"]["jvp_agreement"], jvp_check, ckey + ":JVP")
                                passed = all(row["passed"] for row in fd_checks) and jvp_check["passed"]
                                require(report_cue["tangent"]["directional_finite_difference"]["passed"] is passed, "FD passed flag drift")
                                tangent_record["directional_checks"] = {"epsilon": epsilon.tolist(), "fd_checks": fd_checks, "jvp_check": jvp_check, "passed": passed,
                                                                         "scope": "reduction of saved single-direction checks; recurrence is not rerun"}
                                if not passed:
                                    failures.append(ckey + ":finite_difference_jvp")
                            record["tangent"] = tangent_record
                        else:
                            require(report_cue["tangent"] is None, "unexpected tangent summary")
                        cue_records.append(record)
                    pair = states_by_cue[0].astype(np.float64)-states_by_cue[1].astype(np.float64)
                    native_pair = native_by_cue[0].astype(np.float64)-native_by_cue[1].astype(np.float64)
                    separation, native_separation = np.linalg.norm(pair, axis=1), np.linalg.norm(native_pair, axis=1)
                    compare(trajectories.get(key + "_paired_cue_separation_l2", (len(pair),)), separation, key + ":cue separation")
                    compare(trajectories.get(key + "_native_paired_cue_separation_l2", (len(pair),)), native_separation, key + ":native cue separation")
                    components = {}
                    if model == "lstm":
                        for component, section in (("h", slice(0, config["hidden_size"])), ("c", slice(config["hidden_size"], None))):
                            components[component] = {"manual": np.linalg.norm(pair[:, section], axis=1).tolist(),
                                                     "native": np.linalg.norm(native_pair[:, section], axis=1).tolist()}
                            for native_prefix, value in (("", components[component]["manual"]), ("native_", components[component]["native"])):
                                compare(trajectories.get(key + "_" + native_prefix + "paired_cue_separation_" + component, (len(pair),)), value, key + ":component separation")
                    paired = {"after_write": float(separation[16]), "pre_query": float(separation[-1]),
                              "native_after_write": float(native_separation[16]), "native_pre_query": float(native_separation[-1])}
                    for name, value in paired.items():
                        compare(reported["paired_cue_separation"][name], value, key + ":reported separation")
                    last_closure = cue_records[0]["manual_cycles"][-1]["absolute"] if repeats > 1 else None
                    paired.update(query_coordinate=0, literal_query_targets=memory[:, 0].tolist(),
                                  queried_targets_differ=bool(memory[0, 0] != memory[1, 0]),
                                  manual_separation_l2=separation.tolist(), native_separation_l2=native_separation.tolist(),
                                  component_separations=components,
                                  pre_query_ulp=ulp_diagnostics(states_by_cue[0][-1], states_by_cue[1][-1]),
                                  native_pre_query_ulp=ulp_diagnostics(native_by_cue[0][-1], native_by_cue[1][-1]),
                                  last_manual_closure_to_terminal_cue_separation=last_closure/float(separation[-1]) if last_closure is not None and separation[-1] > 0 else None,
                                  ratio_undefined_reason="R1 has no cycle pair or terminal cue separation is zero" if last_closure is None or separation[-1] == 0 else None,
                                  two_point_population_variance_at_query=float(separation[-1]**2/4),
                                  dimension_estimate_available=False)
                    assays[key] = {"precision": precision, "panel": panel, "repeats": repeats, "arm": arm,
                                   "driver": driver, "cues": cue_records, "paired_cue_separation": paired}
    require(set(assays) == set(metrics["assays"]), "unexpected worker assay summaries")
    anchors, precision_checks = [], []
    for panel in range(config["driver_panels"]):
        for repeats in config["horizons"]:
            for precision in config["precisions"]:
                left_key = f"{precision}_panel{panel}_R{repeats}_periodic"
                right_key = f"{precision}_panel{panel}_R{repeats}_control"
                left, right = inputs.values[left_key+"_inputs"], inputs.values[right_key+"_inputs"]
                require(np.array_equal(left[:, :16], right[:, :16]) and np.array_equal(left[:, -1], right[:, -1])
                        and np.array_equal(left[:, -13:-1], right[:, -13:-1]), "paired arm WRITE/query/final motif mismatch")
                require(assays[left_key]["driver"]["symbol_counts"] == assays[right_key]["driver"]["symbol_counts"], "paired arm symbol count mismatch")
                if repeats == 1:
                    require(np.array_equal(left, right), "R1 realized input anchor mismatch")
                    max_manual, max_native = 0.0, 0.0
                    for cue in range(histories):
                        for suffix, prefix in (("states", "manual"), ("native_core_states", "native")):
                            a, b = trajectories.values[f"{left_key}_cue{cue}_{suffix}"], trajectories.values[f"{right_key}_cue{cue}_{suffix}"]
                            require(np.array_equal(a, b), "R1 identical-arm trajectory anchor mismatch")
                            if prefix == "manual": max_manual = max(max_manual, float(np.max(np.abs(a-b))))
                            else: max_native = max(max_native, float(np.max(np.abs(a-b))))
                    anchors.append({"panel": panel, "precision": precision, "input_exact_match": True,
                                    "manual_max_absolute": max_manual, "native_max_absolute": max_native})
            for arm in ("periodic", "control"):
                key32, key64 = f"float32_panel{panel}_R{repeats}_{arm}", f"float64_panel{panel}_R{repeats}_{arm}"
                require(np.array_equal(inputs.values[key32+"_inputs"], inputs.values[key64+"_inputs"]), "precision input sample mismatch")
                precision_record = {"panel": panel, "repeats": repeats, "arm": arm, "identical_float32_literal_inputs": True,
                                    "arithmetic_assays": "separate full recurrence from zero; FP64 product accumulation does not upgrade FP32 dynamics", "cues": []}
                for cue in range(histories):
                    record = {"cue": cue}
                    for suffix, name in (("states", "manual"), ("native_core_states", "native")):
                        a = trajectories.values[f"{key32}_cue{cue}_{suffix}"].astype(np.float64)
                        b = trajectories.values[f"{key64}_cue{cue}_{suffix}"]
                        record[name] = {"maximum_coordinate_absolute": float(np.max(np.abs(a-b))),
                                        "terminal_l2": float(np.linalg.norm(a[-1]-b[-1])),
                                        "after_write_l2": float(np.linalg.norm(a[16]-b[16]))}
                    p32 = trajectories.values[f"{key32}_cue{cue}_native_query_prediction"].astype(np.float64)
                    p64 = trajectories.values[f"{key64}_cue{cue}_native_query_prediction"]
                    record["native_query_prediction_max_absolute"] = float(np.max(np.abs(p32-p64)))
                    record["tangent_comparison"] = None
                    if repeats in config["derivative_horizons"] and cue < config["derivative_cues"]:
                        product32 = tangents.values[f"{key32}_cue{cue}_product_float64_accumulation"]
                        product64 = tangents.values[f"{key64}_cue{cue}_product_float64_accumulation"]
                        difference = float(np.linalg.norm(product32-product64))
                        scale = max(float(np.linalg.norm(product32)), float(np.linalg.norm(product64)))
                        gain32 = assays[key32]["cues"][cue]["tangent"]["largest_singular_value"]
                        gain64 = assays[key64]["cues"][cue]["tangent"]["largest_singular_value"]
                        record["tangent_comparison"] = {"product_frobenius_difference": difference,
                                                         "product_relative_difference": difference/scale if scale > 0 else None,
                                                         "largest_gain_float32_dynamics": gain32,
                                                         "largest_gain_float64_dynamics": gain64,
                                                         "largest_gain_ratio_fp32_over_fp64": gain32/gain64 if gain64 > 0 else None}
                    precision_record["cues"].append(record)
                precision_checks.append(precision_record)
    panel_final_ids = []
    for panel in range(config["driver_panels"]):
        final = int(inputs.values[f"float32_panel{panel}_R1_periodic_motif_ids"][-1])
        panel_final_ids.append(final)
        reference_control = inputs.values[f"float32_panel{panel}_R32_control_motif_ids"]
        for precision in config["precisions"]:
            for repeats in config["horizons"]:
                periodic_ids = inputs.values[f"{precision}_panel{panel}_R{repeats}_periodic_motif_ids"]
                control_ids = inputs.values[f"{precision}_panel{panel}_R{repeats}_control_motif_ids"]
                require(np.all(periodic_ids == final) and int(control_ids[-1]) == final,
                        "panel final motif changes across horizons/precisions")
                require(np.array_equal(control_ids[:-1], reference_control[:repeats-1]),
                        "control source prefix changes across horizons/precisions")
    require(len(set(panel_final_ids)) == len(panel_final_ids), "driver panels share the same declared final motif")
    for arrays, name in ((inputs, "inputs"), (trajectories, "trajectories"), (tangents, "tangents")):
        arrays.complete(name)
    require(metrics.get("dense_windows") == window_count, "dense-window count mismatch")
    expected_step_count = window_count * 24
    require(metrics.get("step_jacobian_evaluations") == expected_step_count, "step-Jacobian count mismatch")
    require(metrics.get("numerical_failures") == failures and metrics.get("numerical_checks_passed") is (not failures), "worker numerical acceptance summary mismatch")
    # Reject a source directory that changed while it was being reduced.
    require(sha256(directory / "manifest.json") == manifest_digest, "manifest changed during reduction")
    for name, digest in hashes.items():
        require(sha256(directory / name) == digest, f"artifact changed during reduction: {name}")
    return {"cell": {"model": model, "task": task, "seed": seed, "cluster": cell["cluster"],
                     "checkpoint_sha256": cell["checkpoint_sha256"], "baseline_competence_passed": False},
            "artifact_directory": str(directory.resolve()), "artifact_hashes": hashes,
            "manifest_sha256": manifest_digest, "training_source_commit": frozen["training_source_commit"],
            "analysis_source_commit": expected_source, "dense_windows": window_count,
            "step_jacobian_evaluations": expected_step_count, "independent_raw_reduction_passed": True,
            "numerical_contract_passed": not failures, "numerical_failures": failures,
            "assays": assays, "r1_anchor_checks": anchors, "precision_comparisons": precision_checks,
            "probe_floor_sensitivity": {"status": "pending", "reason": "periodicity NPZ files omit original pilot probe train/test states, tags and feature scales; two histories cannot supply that assay"},
            "query_adapter_validation": {"status": "partial", "reason": "native query predictions saved; manual query prediction array absent, so saved manual/native query scalar cannot be independently recomputed"},
            "checkpoint_file_rehash": "not performed by reducer; only frozen identity checked, external scheduler/checkpoint integrity requires coordinator evidence",
            "scientific_acceptance": False, "floquet_acceptance": False,
            "limitations": ["All original native competence gates failed.", "One seed and two fixed driver panels per cell; no independent population uncertainty.",
                            "Two histories give at most rank-one conditioned covariance; no dimension-collapse estimate.",
                            "Derivatives are local manual-equation sensitivities at cue zero, not native readout/cue channel derivatives.",
                            "Approximate state/Jacobian repetition is tolerance-qualified finite-window evidence, not an exact periodic orbit.",
                            "One saved finite-difference direction does not validate every derivative direction."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs/experiments/periodicity-analysis-v1.json")
    parser.add_argument("--expected-source-commit", default=DEFAULT_SOURCE)
    parser.add_argument("--expected-module-sha256", default=DEFAULT_MODULE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    require(not args.output.exists(), "refusing to overwrite an existing reduction report")
    frozen = read_json(args.frozen_config)
    report = {"schema_version": 1, "kind": "independent_periodicity_array_reduction", "scientific_acceptance": False,
              "reducer_sha256": sha256(__file__), "frozen_config_sha256": sha256(args.frozen_config),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "blas_threads_requested": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "comparison_tolerances": {"reported_scalar_absolute": REPORT_ATOL, "reported_scalar_relative": REPORT_RTOL,
                                       "spectral_absolute": SPECTRUM_ATOL, "spectral_relative": SPECTRUM_RTOL,
                                       "closure_product_adapter_contract": "unchanged frozen config tolerances"},
              "cells": [], "errors": []}
    for directory in args.artifact_dir:
        try:
            reduced = reduce_directory(directory, frozen, expected_source=args.expected_source_commit,
                                       expected_module_sha256=args.expected_module_sha256, frozen_sha256=report["frozen_config_sha256"])
            identity = (reduced["cell"]["model"], reduced["cell"]["task"], reduced["cell"]["seed"])
            require(identity not in [(cell["cell"]["model"], cell["cell"]["task"], cell["cell"]["seed"]) for cell in report["cells"]], "duplicate cell directory in batch")
            report["cells"].append(reduced)
        except (InvalidArtifact, OSError, KeyError, ValueError, TypeError, IndexError) as error:
            report["errors"].append({"artifact_directory": str(directory), "error_type": type(error).__name__, "message": str(error)})
    report.update(independent_raw_reduction_passed=not report["errors"], runtime_seconds=time.monotonic()-started,
                  total_dense_windows=sum(cell["dense_windows"] for cell in report["cells"]),
                  all_frozen_cells_present=len(report["cells"]) == len(frozen["cells"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "cells": len(report["cells"]), "errors": report["errors"],
                      "dense_windows": report["total_dense_windows"], "runtime_seconds": report["runtime_seconds"],
                      "scientific_acceptance": False}))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
