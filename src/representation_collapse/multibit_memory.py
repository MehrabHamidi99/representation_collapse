"""Competence-gated multibit recurrent-memory experiment.

This CPU-only experiment asks whether repeated paired temporal bodies alter
the retention of a simultaneous D-bit payload. It is an experiment runner,
not evidence of dimension collapse or a Floquet result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .reporting import ExperimentReporter, file_sha256
from .vector_task import BODY_TYPES, body_from_ids, paired_body_ids, proper_period


SCHEMA = "multibit-memory-v1"
INPUT_CHANNELS = 69
HIDDEN_SIZE = 64
OUTPUT_BITS = 64
WRITE, A, B, X, QUERY = range(5)
PAYLOAD_START = 5
ARM_NAMES = ("periodic", "control")
STAGES = (
    {"updates": 2000, "train_repeats": [0], "validate_repeats": [0]},
    {"updates": 4000, "train_repeats": [0, 1, 2], "validate_repeats": [0, 1, 2]},
    {"updates": 4000, "train_repeats": [0, 1, 2, 4, 8], "validate_repeats": [0, 1, 2, 4, 8]},
)
TEST_CONDITIONS = ((0, 1), (1, 1), (2, 1), (4, 1), (8, 1),
                   (16, 1), (32, 1), (64, 1), (128, 1), (32, 4))
_SAFE_RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class StopRequested(RuntimeError):
    """Deadline or stop-file boundary reached; current artifacts are preserved."""


def default_config(*, dimensions=(1, 4, 16, 64), seeds=(0, 1, 2)) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dimensions": list(dimensions),
        "seeds": list(seeds),
        "batch_cues": 32,
        "stages": [dict(stage) for stage in STAGES],
        "competence_threshold": 0.95,
        "validation_cue_vectors": 256,
        "test_cue_vectors": 256,
        "test_forcing_units": 4,
        "test_conditions": [list(condition) for condition in TEST_CONDITIONS],
        "test_chunk": 64,
        "learning_rate": 0.001,
    }


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii") + b"\n"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_config(value: Any) -> dict[str, Any]:
    expected = default_config()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("config must contain exactly the frozen multibit v1 keys")
    if value["schema"] != SCHEMA:
        raise ValueError("unsupported multibit config schema")
    for key, allowed in (("dimensions", (1, 4, 16, 64)), ("seeds", (0, 1, 2))):
        items = value[key]
        if (not isinstance(items, list) or not items
                or any(type(item) is not int or item not in allowed for item in items)
                or len(set(items)) != len(items)):
            raise ValueError(f"{key} must be a nonempty distinct list from its frozen values")
    for key in ("batch_cues", "validation_cue_vectors", "test_cue_vectors",
                "test_forcing_units", "test_chunk"):
        if type(value[key]) is not int or value[key] != expected[key]:
            raise ValueError(f"{key} differs from frozen protocol")
    if value["stages"] != expected["stages"]:
        raise ValueError("stage schedule differs from frozen competence ladder")
    if value["test_conditions"] != expected["test_conditions"]:
        raise ValueError("held-out condition grid differs from frozen protocol")
    if type(value["competence_threshold"]) not in (int, float) or value["competence_threshold"] != 0.95:
        raise ValueError("competence threshold is frozen at 0.95")
    if type(value["learning_rate"]) not in (int, float) or value["learning_rate"] != 0.001:
        raise ValueError("learning rate is frozen at 0.001")
    return json.loads(canonical(value))


def derived_seed(dimension: int, seed: int, stream: str) -> int:
    raw = f"multibit-memory-v1\0{dimension}\0{seed}\0{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def balanced_bits(rng: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    if count < 2 or count % 2 or not 1 <= dimension <= OUTPUT_BITS:
        raise ValueError("balanced cue design requires an even sample and D in 1..64")
    result = np.empty((count, dimension), dtype=np.int8)
    for bit in range(dimension):
        values = np.concatenate((np.full(count // 2, -1, dtype=np.int8),
                                 np.full(count // 2, 1, dtype=np.int8)))
        rng.shuffle(values)
        result[:, bit] = values
    return result


def body_pairs(rng: np.random.Generator, count: int, repeats: int,
               dwell: int = 1) -> list[tuple[str, str]]:
    if count < 1 or repeats < 0 or dwell < 1:
        raise ValueError("invalid body schedule dimensions")
    pairs = []
    for _ in range(count):
        if repeats == 0:
            pair = ("", "")
        else:
            p, c = paired_body_ids(rng, repeats)
            pair = (body_from_ids(p, dwell), body_from_ids(c, dwell))
        if len(pair[0]) != len(pair[1]):
            raise RuntimeError("paired body horizons differ")
        if repeats > 1:
            if proper_period(pair[0]) is None or proper_period(pair[1]) is not None:
                raise RuntimeError("paired body periodicity contract failed")
        pairs.append(pair)
    return pairs


def render_pairs(bits: np.ndarray, pairs: Sequence[tuple[str, str]],
                 dimension: int) -> np.ndarray:
    bits = np.asarray(bits)
    if (bits.ndim != 2 or bits.shape[1] != dimension or len(bits) != len(pairs)
            or len(bits) == 0 or not np.isin(bits, (-1, 1)).all()):
        raise ValueError("cue vectors and body pairs must align and contain active ±1 bits")
    horizons = {len(p) for pair in pairs for p in pair}
    if len(horizons) != 1:
        raise ValueError("paired bodies must have one common horizon")
    body_length = horizons.pop()
    result = np.zeros((len(bits) * 2, body_length + 2, INPUT_CHANNELS), dtype=np.float32)
    for cue, pair in enumerate(pairs):
        for arm, body in enumerate(pair):
            row = cue * 2 + arm
            result[row, 0, WRITE] = 1.0
            result[row, 0, PAYLOAD_START:PAYLOAD_START + dimension] = bits[cue]
            for t, symbol in enumerate(body, start=1):
                result[row, t, BODY_TYPES[symbol]] = 1.0
            result[row, -1, QUERY] = 1.0
    return result


def training_batch(rng: np.random.Generator, dimension: int, repeats: int,
                   count: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = balanced_bits(rng, count, dimension)
    pairs = body_pairs(rng, count, repeats)
    inputs = render_pairs(bits, pairs, dimension)
    targets = np.repeat((bits > 0).astype(np.float32), 2, axis=0)
    return inputs, targets, bits


def validation_design(rng: np.random.Generator, count: int, dimension: int,
                      repeats_list: Sequence[int]) -> tuple[np.ndarray, dict[int, list[tuple[str, str]]]]:
    bits = balanced_bits(rng, count, dimension)
    designs = {repeats: body_pairs(rng, count, repeats) for repeats in repeats_list}
    return bits, designs


def _make_model(torch):
    class MultiBitLSTM(torch.nn.Module):
        """Fixed architecture for every D: input69, H64/C64, 64-logit head."""
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(INPUT_CHANNELS, HIDDEN_SIZE, batch_first=True)
            self.head = torch.nn.Linear(HIDDEN_SIZE, OUTPUT_BITS)

        def states_and_logits(self, inputs):
            _, (h, c) = self.lstm(inputs[:, :1])
            after_write = torch.cat((h[-1], c[-1]), dim=1)
            if inputs.shape[1] > 2:
                _, (h, c) = self.lstm(inputs[:, 1:-1], (h, c))
            before_query = torch.cat((h[-1], c[-1]), dim=1)
            query_out, _ = self.lstm(inputs[:, -1:], (h, c))
            return self.head(query_out[:, -1]), after_write, before_query

        def forward(self, inputs):
            return self.states_and_logits(inputs)[0]

        def full_forward(self, inputs):
            output, _ = self.lstm(inputs)
            return self.head(output[:, -1])

        def query_from_state(self, query_input, hidden, cell):
            output, _ = self.lstm(query_input[:, None, :], (hidden[None], cell[None]))
            return self.head(output[:, -1])

    return MultiBitLSTM()


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    # logits [driver,cue,arm,D], labels [cue,D]
    predicted = logits >= 0.0
    target = labels[None, :, None, :] > 0
    correct = predicted == target
    errors = ~correct
    bit = correct.mean(axis=(0, 1))  # [arm,D]
    whole = correct.all(axis=-1).mean(axis=(0, 1))
    bit_accuracy = {arm: [float(v) for v in bit[i]] for i, arm in enumerate(ARM_NAMES)}
    polarity_accuracy = {}
    for arm_index, arm in enumerate(ARM_NAMES):
        polarity_accuracy[arm] = {}
        for bit_index in range(labels.shape[1]):
            observed = correct[:, :, arm_index, bit_index]
            negative = labels[:, bit_index] < 0
            positive = ~negative
            polarity_accuracy[arm][str(bit_index)] = {
                "negative": float(observed[:, negative].mean()),
                "positive": float(observed[:, positive].mean()),
            }
    polarity_values = [polarity_accuracy[arm][str(bit_index)][polarity]
                       for arm in ARM_NAMES for bit_index in range(labels.shape[1])
                       for polarity in ("negative", "positive")]
    return {
        "per_bit_accuracy_by_arm": bit_accuracy,
        "per_bit_polarity_accuracy_by_arm": polarity_accuracy,
        "whole_vector_exact_accuracy_by_arm": {arm: float(whole[i]) for i, arm in enumerate(ARM_NAMES)},
        "mean_bit_error_by_arm": {arm: float(errors[:, :, i, :].mean())
                                  for i, arm in enumerate(ARM_NAMES)},
        "per_forcing_unit_mean_bit_error_by_arm": errors.mean(axis=(1, 3)).astype(float).tolist(),
        "all_active_bits_polarities_arms_min_accuracy": float(min(polarity_values)),
        "all_active_bits_all_arms_min_accuracy": float(bit.min()),
        "paired_bit_error_difference_periodic_minus_control": [
            float(v) for v in ((~correct[:, :, 0, :]).astype(np.float64)
                               - (~correct[:, :, 1, :]).astype(np.float64)).mean(axis=(0, 1))],
        "paired_error_difference_periodic_minus_control": float(
            ((~correct[:, :, 0, :]).mean(axis=-1)
             - (~correct[:, :, 1, :]).mean(axis=-1)).mean()),
        "paired_error_difference_by_forcing_unit": [float(v) for v in
            ((~correct[:, :, 0, :]).mean(axis=-1)
             - (~correct[:, :, 1, :]).mean(axis=-1)).mean(axis=1)],
    }


def competence_pass(validation: Mapping[str, Any], threshold: float = 0.95) -> bool:
    values = validation.get("per_bit_accuracy_by_arm")
    if not isinstance(values, Mapping) or set(values) != set(ARM_NAMES):
        return False
    arrays = [values[arm] for arm in ARM_NAMES]
    if not arrays or any(not isinstance(v, list) or not v for v in arrays):
        return False
    widths = {len(v) for v in arrays}
    polarity = validation.get("per_bit_polarity_accuracy_by_arm")
    if not isinstance(polarity, Mapping) or set(polarity) != set(ARM_NAMES):
        return False
    expected_bits = {str(i) for i in range(len(arrays[0]))}
    if any(not isinstance(polarity[arm], Mapping)
           or set(polarity[arm]) != expected_bits
           or any(not isinstance(polarity[arm][bit], Mapping)
                  or set(polarity[arm][bit]) != {"negative", "positive"}
                  for bit in expected_bits)
           for arm in ARM_NAMES):
        return False
    flattened = [value for arm in ARM_NAMES for per_bit in polarity[arm].values()
                 for value in per_bit.values()]
    return (len(widths) == 1 and len(flattened) == 4 * len(arrays[0])
            and all(type(x) in (int, float) and math.isfinite(x) and x >= threshold
                    for row in arrays for x in row)
            and all(type(x) in (int, float) and math.isfinite(x) and x >= threshold
                    for x in flattened))


def conditional_geometry(states: np.ndarray, dimension: int) -> dict[str, Any]:
    """Covariance summaries centered separately by forcing draw and arm.

    Input axes are forcing,cue,arm,complete-state( H then C ).
    """
    states = np.asarray(states)
    if (states.ndim != 4 or states.shape[2] != 2 or states.shape[3] != 128
            or states.shape[1] < 2 or not 1 <= dimension <= 64):
        raise ValueError("geometry requires [forcing,cue,2,128] complete-state samples")
    summaries = []
    for forcing in range(states.shape[0]):
        for arm, arm_name in enumerate(ARM_NAMES):
            sample = states[forcing, :, arm, :].astype(np.float64)
            sample -= sample.mean(axis=0, keepdims=True)
            eigenvalues = np.linalg.eigvalsh(sample.T @ sample / (len(sample) - 1))
            scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
            tolerance = 64 * np.finfo(np.float64).eps * max(sample.shape) * scale
            if not np.isfinite(eigenvalues).all() or float(eigenvalues.min()) < -tolerance:
                raise RuntimeError("materially negative/nonfinite conditional covariance spectrum")
            minimum = float(eigenvalues.min())
            eigenvalues = np.maximum(eigenvalues, 0.0)
            trace = float(eigenvalues.sum())
            summaries.append({
                "forcing_unit": forcing, "arm": arm_name, "samples": len(sample),
                "coordinates": "hidden_then_cell", "state_dimension": 128,
                "centering": "within forcing unit and arm across cue vectors",
                "normalization": "sample covariance, n-1; unrescaled FP32 state values",
                "rank_ceiling": min(len(sample) - 1, 128, (1 << dimension) - 1),
                "negative_eigenvalue_tolerance": tolerance,
                "minimum_unclamped_eigenvalue": minimum,
                "trace": trace,
                "participation_ratio": float(trace * trace / np.dot(eigenvalues, eigenvalues)) if trace else None,
                "numerical_rank": int(np.count_nonzero(eigenvalues > tolerance)),
                "eigenvalues_ascending": eigenvalues.tolist(),
            })
    return {"independent_forcing_units": states.shape[0],
            "cue_vectors_per_forcing_unit_arm": states.shape[1],
            "conditional_spectra": summaries,
            "theoretical_rank_ceiling": min(states.shape[1] - 1, 128, (1 << dimension) - 1),
            "inference_note": "forcing units are the independent body draws; cue vectors share each forcing"}


def _make_inputs_for_slice(bits: np.ndarray, pairs: Sequence[tuple[str, str]],
                           start: int, stop: int, dimension: int) -> np.ndarray:
    return render_pairs(bits[start:stop], pairs[start:stop], dimension)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _evaluate_paired_design(model, torch, bits: np.ndarray,
                            forcing_pairs: Sequence[tuple[str, str]],
                            dimension: int, args, *, condition: str,
                            chunk_rows: int, partial_path: Path | None = None) -> dict[str, Any]:
    count, drivers = len(bits), len(forcing_pairs)
    logits = np.zeros((drivers, count, 2, dimension), dtype=np.float32)
    states = np.zeros((drivers, count, 2, 128), dtype=np.float32)
    after_write = np.zeros_like(states)
    oracle_predictions = np.zeros((drivers, count, dimension), dtype=np.uint8)
    completed_cues = np.zeros(drivers, dtype=np.int32)
    model.eval()
    with torch.no_grad():
        try:
            for driver, forcing_pair in enumerate(forcing_pairs):
                pairs = [forcing_pair] * count
                for start in range(0, count, chunk_rows // 2):
                    _guard(args)
                    stop = min(count, start + chunk_rows // 2)
                    x = torch.from_numpy(_make_inputs_for_slice(bits, pairs, start, stop, dimension))
                    x_np = x.numpy()
                    output, postwrite, prequery = model.states_and_logits(x)
                    expected_shape = (2 * (stop - start),)
                    if (output.shape[0] != expected_shape[0]
                            or postwrite.shape != (expected_shape[0], 128)
                            or prequery.shape != (expected_shape[0], 128)
                            or not all(torch.isfinite(v).all() for v in (output, postwrite, prequery))):
                        raise RuntimeError("nonfinite or misaligned native evaluation outputs")
                    output = output[:, :dimension].cpu().numpy().reshape(stop - start, 2, dimension)
                    write = postwrite.cpu().numpy().reshape(stop - start, 2, 128)
                    state = prequery.cpu().numpy().reshape(stop - start, 2, 128)
                    logits[driver, start:stop] = output
                    after_write[driver, start:stop] = write
                    states[driver, start:stop] = state
                    oracle_predictions[driver, start:stop] = (
                        x_np[::2, 0, PAYLOAD_START:PAYLOAD_START + dimension] > 0).astype(np.uint8)
                    completed_cues[driver] = stop
        except StopRequested:
            if partial_path is not None:
                _write_npz_atomic(partial_path, logits=logits, before_query=states,
                                  after_write=after_write, cue_bits=bits,
                                  forcing_unit_ids=np.arange(drivers, dtype=np.int32),
                                  arm_names=np.asarray(ARM_NAMES), completed_cues=completed_cues,
                                  complete=np.asarray(False), state_layout=np.asarray("hidden_then_cell"))
            raise
    summary = _accuracy(logits, bits)
    predicted = logits >= 0
    errors = (predicted != (bits[None, :, None, :] > 0)).astype(np.uint8)
    zero_state_logits = []
    # The zero-state readout receives the actual query flag and no memory state.
    query = np.zeros((1, INPUT_CHANNELS), dtype=np.float32); query[0, QUERY] = 1.0
    with torch.no_grad():
        baseline = model.query_from_state(torch.from_numpy(query),
                                          torch.zeros((1, HIDDEN_SIZE)),
                                          torch.zeros((1, HIDDEN_SIZE)))
        zero_state_logits = baseline[0, :dimension].cpu().numpy()
    summary.update({
        "condition": condition, "cue_vectors": count, "forcing_units": drivers,
        "paired_rows_per_condition": count * drivers * 2,
        "raw_artifact": (partial_path.name.removesuffix(".partial.npz") + ".npz") if partial_path else None,
        "axes": {"logits": ["forcing", "cue", "arm", "active_bit"],
                 "states": ["forcing", "cue", "arm", "hidden_then_cell"]},
        "oracle_exact_accuracy": float(np.mean(np.all(
            oracle_predictions == (bits[None, :, :] > 0), axis=-1))),
        "zero_state_readout_bit_accuracy": [float(v) for v in
            ((zero_state_logits >= 0) == (bits > 0)).mean(axis=0)],
        "zero_state_readout_whole_vector_exact_accuracy": float(np.mean(np.all(
            (zero_state_logits[None, :] >= 0) == (bits > 0), axis=-1))),
        "error_count_by_arm": {arm: int(errors[:, :, i, :].sum()) for i, arm in enumerate(ARM_NAMES)},
        "paired_bit_errors_by_forcing_unit": (
            errors[:, :, 0, :].astype(np.int8) - errors[:, :, 1, :].astype(np.int8)
        ).mean(axis=1).astype(float).tolist(),
        "conditional_complete_state_geometry": conditional_geometry(states, dimension),
    })
    if partial_path is not None:
        complete_path = partial_path.with_name(partial_path.name.removesuffix(".partial.npz") + ".npz")
        _write_npz_atomic(complete_path,
                          logits=logits, predictions=predicted.astype(np.uint8),
                          labels=np.broadcast_to((bits > 0)[None, :, None, :], predicted.shape),
                          oracle_predictions=oracle_predictions,
                          bit_errors=errors, before_query=states, after_write=after_write,
                          cue_bits=bits.astype(np.int8),
                          forcing_unit_ids=np.arange(drivers, dtype=np.int32),
                          arm_names=np.asarray(ARM_NAMES), completed_cues=completed_cues,
                          complete=np.asarray(True), state_layout=np.asarray("hidden_then_cell"))
    return summary


class _Args:
    def __init__(self, stop_file: Path, deadline: float, output: Path):
        self.stop_file, self.deadline_unix = stop_file, deadline
        self.output = output


_SIGNAL_STOP = False
_TORCH_CONFIGURED = False


def _configure_torch(torch) -> None:
    global _TORCH_CONFIGURED
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        _TORCH_CONFIGURED = True


def _signal_stop(_signum, _frame):
    global _SIGNAL_STOP
    _SIGNAL_STOP = True


def _guard(args) -> None:
    if _SIGNAL_STOP or args.stop_file.exists() or time.time() >= args.deadline_unix:
        raise StopRequested("STOP signal/file or wall-clock deadline reached")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _save_checkpoint(path: Path, torch, model, optimizer, metadata: dict[str, Any]) -> str:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    payload = {"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "metadata": metadata}
    with temp.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return file_sha256(path)


def _validate_model(model, torch, bits, designs, dimension, repeats, args, out):
    pairs = designs[repeats]
    logits = np.zeros((1, len(bits), 2, dimension), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bits), 32):
            _guard(args)
            stop = min(len(bits), start + 32)
            x = torch.from_numpy(_make_inputs_for_slice(bits, pairs, start, stop, dimension))
            output = model(x)[:, :dimension]
            if not torch.isfinite(output).all():
                raise RuntimeError("nonfinite competence validation logits")
            logits[0, start:stop] = output.cpu().numpy().reshape(stop - start, 2, dimension)
    summary = _accuracy(logits, bits)
    return {"repeats": repeats, "cue_vectors": len(bits), **summary}


def _fit_one(config: dict[str, Any], dimension: int, seed: int, args,
             protocol_sha: str, config_sha: str, source_sha: str,
             base_run_id: str) -> dict[str, Any]:
    import torch

    out = args.output / f"D{dimension}-seed{seed}"
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    run_id = f"{base_run_id}-D{dimension}-s{seed}"
    model_config = {"schema": SCHEMA, "dimension": dimension, "seed": seed,
                    "input_channels": INPUT_CHANNELS, "hidden_size": HIDDEN_SIZE,
                    "state_dimension": 2 * HIDDEN_SIZE, "head_outputs": OUTPUT_BITS,
                    "active_outputs": dimension, "batch_cues": 32,
                    "stage_count": len(config["stages"]),
                    "total_updates": sum(item["updates"] for item in config["stages"]),
                    "competence_threshold": 0.95,
                    "training_loss": "binary_cross_entropy_with_logits",
                    "optimizer": "Adam(lr=0.001,weight_decay=0)",
                    "gradient_clip_norm": 1.0}
    stage_schedule_sha = hashlib.sha256(canonical(config["stages"])).hexdigest()
    identities = {"source_sha256": source_sha, "protocol_sha256": protocol_sha,
                  "config_sha256": config_sha,
                  "stage_schedule_sha256": stage_schedule_sha}
    reporter = ExperimentReporter(out / "reporting", project="representation-collapse",
                                  run_id=run_id, mode="offline", config=model_config,
                                  identities=identities)
    report_step = 0
    started = time.monotonic()
    stage_results = []
    update = 0
    passed = True
    stopped = False
    checkpoint_meta = {"source_sha256": source_sha, "protocol_sha256": protocol_sha,
                       "config_sha256": config_sha, "dimension": dimension, "seed": seed,
                       "architecture": "torch.nn.LSTM(69,64,batch_first=True)+Linear(64,64)",
                       "optimizer": "Adam(lr=0.001,weight_decay=0)",
                       "train_seed": derived_seed(dimension, seed, "train"),
                       "validation_seed": derived_seed(dimension, seed, "validation"),
                       "test_seed": derived_seed(dimension, seed, "test"),
                       "updates": 0, "hostname": socket.gethostname(),
                       "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                       "torch_version": torch.__version__, "numpy_version": np.__version__}

    with reporter:
        _configure_torch(torch)
        random.seed(7000 + seed)
        np.random.seed(derived_seed(dimension, seed, "legacy_numpy") % (2**32))
        torch.manual_seed(7000 + seed)  # matched initial architecture across D for each seed
        model = _make_model(torch).cpu().float()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite initial model parameter")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        train_rng = np.random.default_rng(derived_seed(dimension, seed, "train"))
        validation_rng = np.random.default_rng(derived_seed(dimension, seed, "validation"))
        validation_bits, validation_designs = validation_design(
            validation_rng, config["validation_cue_vectors"], dimension,
            sorted({r for item in config["stages"] for r in item["validate_repeats"]}))
        atomic_json(out / "validation-design.json", {
            "dimension": dimension, "seed": seed, "seed_stream": "validation",
            "cue_bits": validation_bits.tolist(),
            "body_pairs": {str(r): [[p, c] for p, c in v]
                           for r, v in validation_designs.items()},
        })
        design_sha = file_sha256(out / "validation-design.json")
        checkpoint_meta.update({"validation_design_sha256": design_sha,
                                "initial_model_seed": 7000 + seed})
        atomic_json(out / "identity.json", {
            **checkpoint_meta, "python_version": platform.python_version(),
            "git_commit": _git_commit(), "validation_design_sha256": design_sha,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(out / "status.json", {"status": "training", "dimension": dimension,
                                           "seed": seed, "updates": 0, "stage": 0,
                                           "validation_passed": None, "elapsed_seconds": 0.0})

        with (out / "training.jsonl").open("x", encoding="ascii") as history:
            try:
                for stage_index, stage in enumerate(config["stages"], start=1):
                    for within in range(stage["updates"]):
                        _guard(args)
                        repeats = stage["train_repeats"][within % len(stage["train_repeats"])]
                        x_np, y_np, _ = training_batch(train_rng, dimension, repeats)
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        x = torch.from_numpy(x_np); targets = torch.from_numpy(y_np)
                        logits = model(x)[:, :dimension]
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite training loss")
                        loss.backward()
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step(); update += 1
                        if update % 50 == 0:
                            record = {"update": update, "stage": stage_index, "R": repeats,
                                      "loss": float(loss.detach()),
                                      "gradient_norm": float(gradient_norm),
                                      "elapsed_seconds": time.monotonic() - started}
                            history.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                            history.flush(); os.fsync(history.fileno())
                            reporter.log({"training/loss": record["loss"],
                                          "training/gradient_norm": record["gradient_norm"],
                                          "training/update": update,
                                          "training/stage": stage_index}, step=report_step)
                            report_step += 1
                            atomic_json(out / "status.json", {
                                "status": "training", "dimension": dimension, "seed": seed,
                                "updates": update, "stage": stage_index,
                                "validation_passed": None,
                                "elapsed_seconds": time.monotonic() - started})
                    history.flush(); os.fsync(history.fileno())
                    checks = [_validate_model(model, torch, validation_bits, validation_designs,
                                              dimension, r, args, out)
                              for r in stage["validate_repeats"]]
                    stage_passed = all(competence_pass(item, 0.95) for item in checks)
                    stage_result = {"stage": stage_index, "updates": update,
                                    "required_repeats": stage["validate_repeats"],
                                    "passed": stage_passed, "checks": checks}
                    stage_results.append(stage_result)
                    atomic_json(out / f"stage-{stage_index}-validation.json", stage_result)
                    reporter.log({f"validation/R{item['repeats']}/min_bit_polarity_arm_accuracy":
                                  item["all_active_bits_polarities_arms_min_accuracy"] for item in checks},
                                 step=report_step)
                    report_step += 1
                    passed = stage_passed
                    checkpoint_meta.update(updates=update, stage=stage_index,
                                           competence_passed=passed)
                    checkpoint_sha = _save_checkpoint(out / "checkpoint.pt", torch, model,
                                                      optimizer, dict(checkpoint_meta))
                    atomic_json(out / "checkpoint.json", {
                        "sha256": checkpoint_sha, "updates": update,
                        "stage": stage_index, "competence_passed": passed})
                    atomic_json(out / "status.json", {
                        "status": "validation_passed" if passed else "competence_failed",
                        "dimension": dimension, "seed": seed, "updates": update,
                        "stage": stage_index, "validation_passed": passed,
                        "elapsed_seconds": time.monotonic() - started,
                        "checkpoint_sha256": checkpoint_sha})
                    if not passed:
                        break
            except StopRequested:
                stopped = True
                checkpoint_meta.update(updates=update, stage=len(stage_results) + 1,
                                       competence_passed=False, stopped=True)
                checkpoint_sha = _save_checkpoint(out / "checkpoint.pt", torch, model,
                                                  optimizer, dict(checkpoint_meta))
                atomic_json(out / "checkpoint.json", {"sha256": checkpoint_sha,
                                                       "updates": update, "stopped": True})
                atomic_json(out / "status.json", {
                    "status": "stopped_partial", "dimension": dimension, "seed": seed,
                    "updates": update, "stage": len(stage_results) + 1,
                    "validation_passed": False, "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_sha256": checkpoint_sha})

        if passed and not stopped:
            test_rng = np.random.default_rng(derived_seed(dimension, seed, "test"))
            test_bits = balanced_bits(test_rng, 256, dimension)
            condition_designs = {}
            for repeats, dwell in config["test_conditions"]:
                condition_designs[(repeats, dwell)] = body_pairs(
                    test_rng, config["test_forcing_units"], repeats, dwell)
            atomic_json(out / "evaluation-design.json", {
                "dimension": dimension, "seed": seed, "seed_stream": "test",
                "cue_bits": test_bits.tolist(),
                "forcing_unit_ids": list(range(config["test_forcing_units"])),
                "conditions": {f"R{r}-q{q}": [[p, c] for p, c in pairs]
                               for (r, q), pairs in condition_designs.items()},
            })
            test_design_sha = file_sha256(out / "evaluation-design.json")
            results = []
            fixed_model_sha = _model_digest(model)
            atomic_json(out / "status.json", {
                "status": "heldout_evaluation", "dimension": dimension, "seed": seed,
                "updates": update, "stage": len(stage_results),
                "validation_passed": True, "completed_conditions": 0,
                "active_condition": None, "elapsed_seconds": time.monotonic() - started,
                "checkpoint_sha256": file_sha256(out / "checkpoint.pt")})
            try:
                for repeats, dwell in config["test_conditions"]:
                    _guard(args)
                    label = f"R{repeats}-q{dwell}"
                    atomic_json(out / "status.json", {
                        "status": "heldout_evaluation", "dimension": dimension, "seed": seed,
                        "updates": update, "stage": len(stage_results),
                        "validation_passed": True, "completed_conditions": len(results),
                        "active_condition": label, "elapsed_seconds": time.monotonic() - started,
                        "checkpoint_sha256": file_sha256(out / "checkpoint.pt")})
                    result = _evaluate_paired_design(
                        model, torch, test_bits, condition_designs[(repeats, dwell)],
                        dimension, args, condition=label, chunk_rows=config["test_chunk"],
                        partial_path=out / f"{label}.partial.npz")
                    result["condition"] = label
                    result["repeats"] = repeats; result["dwell"] = dwell
                    results.append(result)
                    atomic_json(out / f"{label}-summary.json", result)
                    geometry_rows = result["conditional_complete_state_geometry"]["conditional_spectra"]
                    finite_pr = [row["participation_ratio"] for row in geometry_rows
                                 if row["participation_ratio"] is not None]
                    reporter.log({f"evaluation/{label}/paired_error_difference":
                                  result["paired_error_difference_periodic_minus_control"],
                                  f"evaluation/{label}/mean_bit_accuracy": float(np.mean([
                                      *result["per_bit_accuracy_by_arm"]["periodic"],
                                      *result["per_bit_accuracy_by_arm"]["control"]])),
                              f"evaluation/{label}/mean_bit_error_periodic":
                                  result["mean_bit_error_by_arm"]["periodic"],
                              f"evaluation/{label}/mean_bit_error_control":
                                  result["mean_bit_error_by_arm"]["control"],
                                  f"evaluation/{label}/exact_accuracy_periodic":
                                  result["whole_vector_exact_accuracy_by_arm"]["periodic"],
                                  f"evaluation/{label}/exact_accuracy_control":
                                  result["whole_vector_exact_accuracy_by_arm"]["control"],
                                  f"evaluation/{label}/zero_state_mean_bit_accuracy": float(np.mean(
                                      result["zero_state_readout_bit_accuracy"])),
                              f"evaluation/{label}/zero_state_whole_vector_accuracy":
                                  result["zero_state_readout_whole_vector_exact_accuracy"],
                                  f"evaluation/{label}/mean_complete_state_trace": float(np.mean([
                                      row["trace"] for row in geometry_rows])),
                              f"evaluation/{label}/mean_complete_state_pr": float(np.mean(finite_pr))
                                  if finite_pr else None,
                              f"evaluation/{label}/defined_complete_state_pr_spectra": len(finite_pr)},
                             step=report_step)
                    report_step += 1
                    if _model_digest(model) != fixed_model_sha:
                        raise RuntimeError("model weights changed during fixed-weight evaluation")
            except StopRequested:
                stopped = True
            by_condition = {item["condition"]: item for item in results}
            primary_delta = by_condition.get("R32-q1", {}).get(
                "paired_error_difference_periodic_minus_control")
            anchor_delta = by_condition.get("R2-q1", {}).get(
                "paired_error_difference_periodic_minus_control")
            did = (primary_delta - anchor_delta
                   if primary_delta is not None and anchor_delta is not None else None)
            result = {"schema": "multibit-memory-result-v1",
                      "status": "stopped_partial" if stopped else "competence_passed",
                      "dimension": dimension, "seed": seed, "updates": update,
                      "stage_results": stage_results, "evaluation": results,
                      "primary_R32q1_periodic_minus_control": primary_delta,
                      "anchor_R2q1_periodic_minus_control": anchor_delta,
                      "R32q1_minus_R2q1_difference_in_differences": did,
                      "checkpoint_sha256": file_sha256(out / "checkpoint.pt"),
                      "validation_design_sha256": design_sha,
                      "evaluation_design_sha256": test_design_sha,
                      "source_sha256": source_sha, "protocol_sha256": protocol_sha,
                      "config_sha256": config_sha, "heldout_opened": True,
                      "limitations": ["one recurrent architecture", "finite cue/body designs",
                                      "no causal representation intervention", "no Floquet/tangent analysis"]}
            atomic_json(out / "result.json", result)
            reporter.log({"result/primary_R32q1_delta": primary_delta,
                          "result/anchor_R2q1_delta": anchor_delta,
                          "result/R32q1_minus_R2q1_DID": did}, step=report_step)
            report_step += 1
            reporter.log_manifest("result", metadata={"dimension": dimension,
                                                       "seed": seed, "updates": update,
                                                       "competence_passed": True},
                                  identities={"result_sha256": file_sha256(out / "result.json"),
                                              "checkpoint_sha256": result["checkpoint_sha256"]})
        else:
            result = {"schema": "multibit-memory-result-v1",
                      "status": "stopped_partial" if stopped else "competence_failed",
                      "dimension": dimension, "seed": seed, "updates": update,
                      "stage_results": stage_results,
                      "evaluation": locals().get("results", []),
                      "primary_R32q1_periodic_minus_control": None,
                      "anchor_R2q1_periodic_minus_control": None,
                      "R32q1_minus_R2q1_difference_in_differences": None,
                      "checkpoint_sha256": file_sha256(out / "checkpoint.pt"),
                      "validation_design_sha256": design_sha,
                      "evaluation_design_sha256": locals().get("test_design_sha"),
                      "source_sha256": source_sha, "protocol_sha256": protocol_sha,
                      "config_sha256": config_sha,
                      "heldout_opened": bool(passed and (stopped or locals().get("results"))),
                      "limitations": (["held-out test remains sealed without all-stage competence"]
                                      if not passed else ["held-out evaluation stopped before all conditions completed"])}
            atomic_json(out / "result.json", result)
        final_status = {"status": result["status"], "dimension": dimension, "seed": seed,
                        "updates": update, "stage": len(stage_results),
                        "validation_passed": bool(passed and not stopped),
                        "elapsed_seconds": time.monotonic() - started,
                        "checkpoint_sha256": result["checkpoint_sha256"]}
        atomic_json(out / "status.json", final_status)
        inventory = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                     for p in sorted(out.iterdir()) if p.is_file()}
        atomic_json(out / "artifact-inventory.json", inventory)
        reporter.log({"result/competence_passed": int(passed and not stopped),
                      "result/updates": update}, step=report_step)
        reporter.log_manifest("artifact-inventory", metadata={"files": len(inventory)},
                              identities={"inventory_sha256": file_sha256(out / "artifact-inventory.json")})
    return result


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def run(*, config_path: Path, protocol_path: Path, output: Path, run_id: str,
        deadline_unix: float, stop_file: Path) -> dict[str, Any]:
    if (not _SAFE_RUN.fullmatch(run_id)
            or any(not _SAFE_RUN.fullmatch(f"{run_id}-D{d}-s{s}")
                   for d in (1, 4, 16, 64) for s in (0, 1, 2))):
        raise ValueError("run_id and derived fit IDs must use the frozen safe identifier syntax")
    if type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix):
        raise ValueError("deadline_unix must be finite")
    if deadline_unix <= time.time():
        raise ValueError("deadline_unix must be in the future")
    config_bytes = config_path.read_bytes()
    config = validate_config(json.loads(config_bytes.decode("ascii")))
    protocol_path = protocol_path.resolve(strict=True)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    protocol_sha = file_sha256(protocol_path)
    source_sha = file_sha256(Path(__file__).resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    args = _Args(stop_file.resolve(), float(deadline_unix), output)
    global _SIGNAL_STOP
    _SIGNAL_STOP = False
    old_term = signal.signal(signal.SIGTERM, _signal_stop)
    results = []
    try:
        import torch
        if torch.__version__ != "2.0.0+cpu" or np.__version__ != "1.24.4":
            raise RuntimeError("runtime differs from pinned CPU Torch/NumPy experiment environment")
        _configure_torch(torch)
        atomic_json(output / "run-identity.json", {
            "schema": "multibit-memory-run-v1", "run_id": run_id,
            "source_sha256": source_sha, "protocol_sha256": protocol_sha,
            "config_sha256": config_sha, "git_commit": _git_commit(),
            "deadline_unix": deadline_unix, "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version_expected": "2.0.0+cpu", "numpy_version": np.__version__,
        })
        atomic_json(output / "run-status.json", {"status": "running", "fits_completed": 0})
        for dimension in config["dimensions"]:
            for seed in config["seeds"]:
                try:
                    _guard(args)
                except StopRequested:
                    break
                results.append(_fit_one(config, dimension, seed, args, protocol_sha,
                                        config_sha, source_sha, run_id))
                atomic_json(output / "run-status.json", {
                    "status": "stopped_partial" if results[-1]["status"] == "stopped_partial" else "running",
                    "fits_completed": len(results), "last_dimension": dimension,
                    "last_seed": seed, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
                if results[-1]["status"] == "stopped_partial":
                    break
            if results and results[-1]["status"] == "stopped_partial":
                break
        final = {"schema": "multibit-memory-run-result-v1", "run_id": run_id,
                 "status": "stopped_partial" if _SIGNAL_STOP or stop_file.exists() or time.time() >= deadline_unix else "completed",
                 "fits": results, "source_sha256": source_sha,
                 "protocol_sha256": protocol_sha, "config_sha256": config_sha}
        atomic_json(output / "run-result.json", final)
        atomic_json(output / "run-status.json", {"status": final["status"],
                                                   "fits_completed": len(results),
                                                   "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        return final
    finally:
        signal.signal(signal.SIGTERM, old_term)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--stop-file", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(config_path=args.config, protocol_path=args.protocol, output=args.output,
                 run_id=args.run_id, deadline_unix=args.deadline_unix,
                 stop_file=args.stop_file)
    print(json.dumps({"status": result["status"], "fits": len(result["fits"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
