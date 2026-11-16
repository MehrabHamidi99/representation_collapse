"""Frozen higher-load extension of the multibit repetition experiment.

The runner fits independent fixed-architecture LSTMs at larger binary loads.
It reports task and complete-state covariance measurements; it does not infer
causal collapse, capacity, or Floquet stability from these outcomes.
"""
from __future__ import annotations

import argparse
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
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import multibit_memory as v1
from .reporting import ExperimentReporter, file_sha256


SCHEMA = "multibit-load-ladder-v2"
DIMENSIONS = (4, 8, 12, 16, 24, 32, 48, 64)
SEEDS = (0, 1, 2)
STAGES = (
    {"updates": 6000, "train_repeats": [0], "validate_repeats": [0]},
    {"updates": 12000, "train_repeats": [0, 1, 2], "validate_repeats": [0, 1, 2]},
    {"updates": 16000, "train_repeats": [0, 1, 2, 4, 8], "validate_repeats": [0, 1, 2, 4, 8]},
)
TEST_CONDITIONS = tuple((r, 1) for r in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) + (
    (128, 4), (256, 2))
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def default_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dimensions": list(DIMENSIONS),
        "seeds": list(SEEDS),
        "batch_cues": 32,
        "stages": [dict(stage) for stage in STAGES],
        "competence_threshold": 0.95,
        "validation_cue_vectors": 256,
        "test_cue_vectors": 256,
        "test_forcing_units": 4,
        "test_conditions": [list(c) for c in TEST_CONDITIONS],
        "test_chunk": 64,
        "learning_rate": 0.001,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii") + b"\n"


def _atomic_json(path: Path, value: Any) -> None:
    v1.atomic_json(path, value)


def validate_config(value: Any) -> dict[str, Any]:
    expected = default_config()
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("config must contain exactly the frozen load-ladder v2 keys")
    if value != expected:
        raise ValueError("config differs from the frozen higher-load v2 design")
    return json.loads(_canonical(value))


def derive_seed(dimension: int, seed: int, stream: str) -> int:
    raw = f"{SCHEMA}\0{dimension}\0{seed}\0{stream}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def _all_gates_passed(records: Sequence[Mapping[str, Any]], stage_count: int) -> bool:
    return len(records) == stage_count and all(bool(item.get("passed")) for item in records)


def _condition_horizon(repeats: int, dwell: int) -> int:
    if type(repeats) is not int or repeats < 0 or type(dwell) is not int or dwell < 1:
        raise ValueError("invalid repetition/dwell condition")
    return 12 * repeats * dwell


def _sha_np(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _fit_one(config: Mapping[str, Any], dimension: int, seed: int, args, *,
             run_id: str, source_sha: str, primitive_sha: str,
             protocol_sha: str, config_sha: str) -> dict[str, Any]:
    import torch

    fit_dir = args.output / f"D{dimension}"
    fit_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    fit_run_id = f"{run_id}-D{dimension}-s{seed}"
    if not _SAFE_ID.fullmatch(fit_run_id):
        raise ValueError("derived W&B fit ID exceeds safe identifier syntax")
    stage_digest = hashlib.sha256(_canonical(config["stages"])).hexdigest()
    report_config = {
        "schema": SCHEMA, "dimension": dimension, "seed": seed,
        "input_channels": 69, "hidden_size": 64, "complete_state_dimension": 128,
        "head_outputs": 64, "active_outputs": dimension, "batch_cues": 32,
        "stage_count": len(config["stages"]),
        "total_updates": sum(s["updates"] for s in config["stages"]),
        "competence_threshold": 0.95, "loss": "mean_active_bit_BCEWithLogits",
        "optimizer": "Adam(lr=0.001,weight_decay=0)", "gradient_clip_norm": 1.0,
    }
    identities = {"source_sha256": source_sha, "primitive_source_sha256": primitive_sha,
                  "protocol_sha256": protocol_sha, "config_sha256": config_sha,
                  "stage_schedule_sha256": stage_digest}
    reporter = ExperimentReporter(fit_dir / "reporting", project="representation-collapse",
                                  mode="offline", run_id=fit_run_id,
                                  config=report_config, identities=identities)
    start = time.monotonic()
    report_step = 0
    update = 0
    stage_records = []
    endpoint_passed = False
    interrupted = False
    checkpoint_records = []
    fit_identity = {
        "schema": "multibit-load-ladder-fit-v2", "seed": seed, "dimension": dimension,
        "model_seed": 7000 + seed, "train_rng_seed": derive_seed(dimension, seed, "train"),
        "validation_rng_seed": derive_seed(dimension, seed, "validation"),
        "test_rng_seed": derive_seed(dimension, seed, "test"),
        "architecture": "torch.nn.LSTM(69,64,batch_first=True)+Linear(64,64)",
        "state_layout": "hidden_then_cell", "state_dimension": 128,
        "source_sha256": source_sha, "primitive_source_sha256": primitive_sha,
        "protocol_sha256": protocol_sha, "config_sha256": config_sha,
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "python_version": platform.python_version(), "git_commit": _git_commit(),
    }
    validation_rng = np.random.default_rng(derive_seed(dimension, seed, "validation"))
    val_bits, val_designs = v1.validation_design(
        validation_rng, config["validation_cue_vectors"], dimension,
        sorted({r for stage in config["stages"] for r in stage["validate_repeats"]}))

    with reporter:
        torch.manual_seed(7000 + seed)  # common initialization across D at each seed
        random.seed(7000 + seed)
        model = v1._make_model(torch).cpu().float()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        train_rng = np.random.default_rng(derive_seed(dimension, seed, "train"))
        _atomic_json(fit_dir / "identity.json", {
            **fit_identity, "initial_model_sha256": v1._model_digest(model),
            "validation_design_sha256": None,
            "created_at_utc": datetime.now(timezone.utc).isoformat()})
        validation_design_value = {
            "dimension": dimension, "seed": seed, "rng_namespace": SCHEMA,
            "cue_bits": val_bits.tolist(),
            "body_pairs": {str(r): [[p, c] for p, c in pairs]
                           for r, pairs in val_designs.items()},
        }
        _atomic_json(fit_dir / "validation-design.json", validation_design_value)
        val_design_sha = file_sha256(fit_dir / "validation-design.json")
        fit_identity["validation_design_sha256"] = val_design_sha
        _atomic_json(fit_dir / "identity.json", {
            **fit_identity, "initial_model_sha256": v1._model_digest(model),
            "created_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(fit_dir / "status.json", {"status": "training", "dimension": dimension,
                                                "seed": seed, "update": 0, "stage": 0})

        with (fit_dir / "training.jsonl").open("x", encoding="ascii") as history:
            try:
                for stage_no, stage in enumerate(config["stages"], start=1):
                    stage_start = update
                    for within in range(stage["updates"]):
                        v1._guard(args)
                        repeat = stage["train_repeats"][within % len(stage["train_repeats"])]
                        x_np, y_np, _ = v1.training_batch(train_rng, dimension, repeat, 32)
                        model.train(); optimizer.zero_grad(set_to_none=True)
                        x = torch.from_numpy(x_np); target = torch.from_numpy(y_np)
                        logits = model(x)[:, :dimension]
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite training loss")
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                              error_if_nonfinite=True)
                        optimizer.step(); update += 1
                        if update % 50 == 0:
                            entry = {"update": update, "stage": stage_no, "R": repeat,
                                     "loss": float(loss.detach()), "gradient_norm": float(grad),
                                     "elapsed_seconds": time.monotonic() - start}
                            history.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                            history.flush(); os.fsync(history.fileno())
                            reporter.log({"training/loss": entry["loss"],
                                          "training/gradient_norm": entry["gradient_norm"],
                                          "training/update": update,
                                          "training/stage": stage_no}, step=report_step)
                            report_step += 1
                            _atomic_json(fit_dir / "status.json", {
                                "status": "training", "dimension": dimension, "seed": seed,
                                "update": update, "stage": stage_no,
                                "elapsed_seconds": time.monotonic() - start})
                    history.flush(); os.fsync(history.fileno())
                    ckpt_path = fit_dir / f"stage-{stage_no:02d}-checkpoint.pt"
                    ckpt_meta = {**fit_identity, "stage": stage_no, "update": update,
                                 "stage_updates": update - stage_start,
                                 "model_sha256": v1._model_digest(model)}
                    checkpoint_sha = v1._save_checkpoint(ckpt_path, torch, model, optimizer, ckpt_meta)
                    checkpoint_records.append({"stage": stage_no, "path": ckpt_path.name,
                                               "sha256": checkpoint_sha, "update": update})
                    _atomic_json(fit_dir / f"stage-{stage_no:02d}-checkpoint.json",
                                 checkpoint_records[-1])

                    validation = _validation_stage_save(
                        model, torch, val_bits, val_designs, dimension,
                        stage["validate_repeats"], args,
                        fit_dir / f"stage-{stage_no}-validation.npz",
                        fit_dir / f"stage-{stage_no}-validation.partial.npz")
                    validation.update({"stage": stage_no, "update": update,
                                       "passed": all(v1.competence_pass(item, 0.95)
                                                     for item in validation["checks"]),
                                       "checkpoint_sha256": checkpoint_sha,
                                       "raw_file": f"stage-{stage_no}-validation.npz"})
                    stage_records.append(validation)
                    _atomic_json(fit_dir / f"stage-{stage_no}-validation.json", validation)
                    reporter.log({f"validation/stage{stage_no}/min_bit_polarity_arm_accuracy":
                                  min(item["all_active_bits_polarities_arms_min_accuracy"]
                                      for item in validation["checks"]),
                                  f"validation/stage{stage_no}/competence_passed": int(validation["passed"])},
                                 step=report_step)
                    report_step += 1
                    _atomic_json(fit_dir / "status.json", {
                        "status": "validation_passed" if validation["passed"] else "competence_failed",
                        "dimension": dimension, "seed": seed, "update": update,
                        "stage": stage_no, "validation_passed": validation["passed"],
                        "checkpoint_sha256": checkpoint_sha})
                    if not validation["passed"]:
                        break
                endpoint_passed = _all_gates_passed(stage_records, len(config["stages"]))
            except v1.StopRequested:
                interrupted = True
                partial_path = fit_dir / f"partial-checkpoint-update-{update:08d}.pt"
                partial_sha = v1._save_checkpoint(partial_path, torch, model, optimizer, {
                    **fit_identity, "update": update, "partial": True})
                checkpoint_records.append({"stage": len(stage_records) + 1,
                                           "path": partial_path.name, "sha256": partial_sha,
                                           "update": update, "partial": True})
                _atomic_json(fit_dir / "status.json", {
                    "status": "stopped_partial", "dimension": dimension, "seed": seed,
                    "update": update, "stage": len(stage_records) + 1,
                    "checkpoint_sha256": partial_sha})

        replay_record = None
        test_results = []
        heldout_opened = False
        if endpoint_passed and not interrupted:
            final_ckpt = fit_dir / f"stage-{len(config['stages']):02d}-checkpoint.pt"
            before_digest = v1._model_digest(model)
            if file_sha256(final_ckpt) != checkpoint_records[-1]["sha256"]:
                raise RuntimeError("final checkpoint hash changed before replay")
            loaded = torch.load(final_ckpt, map_location="cpu", weights_only=True)
            if (loaded.get("metadata", {}).get("dimension") != dimension
                    or loaded.get("metadata", {}).get("seed") != seed
                    or loaded.get("metadata", {}).get("update") != update
                    or v1._model_digest(model) != before_digest):
                raise RuntimeError("final checkpoint metadata/model identity mismatch")
            replay_model = v1._make_model(torch).cpu().float()
            replay_model.load_state_dict(loaded["state_dict"], strict=True)
            replay_model.eval()
            if v1._model_digest(replay_model) != before_digest:
                raise RuntimeError("reloaded endpoint checkpoint differs from fitted model")
            args.validation_partial_path = fit_dir / "endpoint-replay.partial.npz"
            replay_validation, replay_arrays = _endpoint_validation_replay(
                replay_model, torch, val_bits, val_designs, dimension,
                config["stages"][-1]["validate_repeats"], args)
            if replay_validation is None:
                interrupted = True
                result = {
                    "schema": "multibit-load-ladder-result-v2", "status": "stopped_partial",
                    "dimension": dimension, "seed": seed, "updates": update,
                    "stage_results": stage_records, "stage_checkpoints": checkpoint_records,
                    "endpoint_replay": None, "evaluation": [],
                    "checkpoint_sha256": file_sha256(final_ckpt),
                    "validation_design_sha256": val_design_sha,
                    "source_sha256": source_sha, "primitive_source_sha256": primitive_sha,
                    "protocol_sha256": protocol_sha, "config_sha256": config_sha,
                    "heldout_opened": False,
                    "limitations": ["endpoint replay interrupted; held-out panel remains sealed"],
                }
                _atomic_json(fit_dir / "result.json", result)
            else:
                # The whole held-out branch is nested under a successful endpoint replay.
                endpoint_raw = np.load(fit_dir / f"stage-{len(config['stages'])}-validation.npz",
                                       allow_pickle=False)
                comparison = []
                for repeat in config["stages"][-1]["validate_repeats"]:
                    expected = np.asarray(endpoint_raw[f"logits_R{repeat}"])
                    actual = replay_arrays[f"logits_R{repeat}"]
                    if not np.allclose(expected, actual, rtol=0, atol=0):
                        raise RuntimeError(f"reloaded checkpoint validation logits differ at R={repeat}")
                    comparison.append({"repeats": repeat, "exact": True,
                                       "original_logits_sha256": _sha_np(expected),
                                       "reloaded_logits_sha256": _sha_np(actual)})
                endpoint_raw.close()
                replay_record = {"checkpoint_sha256": file_sha256(final_ckpt),
                                 "model_sha256": v1._model_digest(replay_model),
                                 "torch_version": torch.__version__, "rtol": 0.0, "atol": 0.0,
                                 "passed": True, "comparisons": comparison}
                _atomic_json(fit_dir / "endpoint-replay.json", replay_record)
                if not replay_validation["checks"] or not all(
                        v1.competence_pass(item, 0.95) for item in replay_validation["checks"]):
                    raise RuntimeError("reloaded endpoint validation lost competence")

                heldout_opened = True
                test_rng = np.random.default_rng(derive_seed(dimension, seed, "test"))
                test_bits = v1.balanced_bits(test_rng, config["test_cue_vectors"], dimension)
                designs = {tuple(condition): v1.body_pairs(
                    test_rng, config["test_forcing_units"], condition[0], condition[1])
                    for condition in config["test_conditions"]}
                test_design = {"dimension": dimension, "seed": seed, "rng_namespace": SCHEMA,
                               "cue_bits": test_bits.tolist(),
                               "forcing_unit_ids": list(range(config["test_forcing_units"])),
                               "conditions": {f"R{r}-q{q}": [[p, c] for p, c in pairs]
                                              for (r, q), pairs in designs.items()}}
                _atomic_json(fit_dir / "evaluation-design.json", test_design)
                test_design_sha = file_sha256(fit_dir / "evaluation-design.json")
                frozen_digest = v1._model_digest(replay_model)
                _atomic_json(fit_dir / "status.json", {"status": "heldout_evaluation",
                                                       "dimension": dimension, "seed": seed,
                                                       "update": update, "completed_conditions": 0,
                                                       "active_condition": None,
                                                       "checkpoint_sha256": replay_record["checkpoint_sha256"]})
                try:
                    for repeats, dwell in config["test_conditions"]:
                        v1._guard(args)
                        label = f"R{repeats}-q{dwell}"
                        result = v1._evaluate_paired_design(
                            replay_model, torch, test_bits, designs[(repeats, dwell)],
                            dimension, args, condition=label, chunk_rows=config["test_chunk"],
                            partial_path=fit_dir / f"{label}.partial.npz")
                        result.update({"condition": label, "repeats": repeats, "dwell": dwell,
                                       "horizon": _condition_horizon(repeats, dwell)})
                        test_results.append(result)
                        _atomic_json(fit_dir / f"{label}-summary.json", result)
                        reporter.log(_evaluation_metrics(label, result), step=report_step)
                        report_step += 1
                        if v1._model_digest(replay_model) != frozen_digest:
                            raise RuntimeError("weights changed during fixed-checkpoint test evaluation")
                except v1.StopRequested:
                    interrupted = True

                by_condition = {item["condition"]: item for item in test_results}
                primary = by_condition.get("R32-q1", {}).get(
                    "paired_error_difference_periodic_minus_control")
                anchor = by_condition.get("R2-q1", {}).get(
                    "paired_error_difference_periodic_minus_control")
                result = {
                    "schema": "multibit-load-ladder-result-v2",
                    "status": "stopped_partial" if interrupted else "competence_passed",
                    "dimension": dimension, "seed": seed, "updates": update,
                    "stage_results": stage_records, "stage_checkpoints": checkpoint_records,
                    "endpoint_replay": replay_record, "evaluation": test_results,
                    "primary_R32q1_periodic_minus_control": primary,
                    "anchor_R2q1_periodic_minus_control": anchor,
                    "R32q1_minus_R2q1_difference_in_differences": (
                        primary - anchor if primary is not None and anchor is not None else None),
                    "checkpoint_sha256": replay_record["checkpoint_sha256"],
                    "validation_design_sha256": val_design_sha,
                    "evaluation_design_sha256": test_design_sha,
                    "source_sha256": source_sha, "primitive_source_sha256": primitive_sha,
                    "protocol_sha256": protocol_sha, "config_sha256": config_sha,
                    "heldout_opened": True,
                    "interpretation_scope": "descriptive task retention and conditional covariance; not collapse or Floquet evidence",
                }
                _atomic_json(fit_dir / "result.json", result)
                reporter.log({"result/primary_R32q1_delta": primary,
                              "result/anchor_R2q1_delta": anchor,
                              "result/R32q1_minus_R2q1_DID": (
                                  primary - anchor if primary is not None and anchor is not None else None),
                              "result/competence_passed": 1,
                              "result/interrupted": int(interrupted)}, step=report_step)
                report_step += 1
                reporter.log_manifest("result", metadata={"dimension": dimension, "seed": seed,
                                                           "updates": update, "competence_passed": True,
                                                           "interrupted": interrupted},
                                      identities={"result_sha256": file_sha256(fit_dir / "result.json"),
                                                  "checkpoint_sha256": replay_record["checkpoint_sha256"]})
        else:
            result = {
                "schema": "multibit-load-ladder-result-v2",
                "status": "stopped_partial" if interrupted else "competence_failed",
                "dimension": dimension, "seed": seed, "updates": update,
                "stage_results": stage_records, "stage_checkpoints": checkpoint_records,
                "endpoint_replay": replay_record, "evaluation": [],
                "checkpoint_sha256": checkpoint_records[-1]["sha256"] if checkpoint_records else None,
                "validation_design_sha256": val_design_sha,
                "source_sha256": source_sha, "primitive_source_sha256": primitive_sha,
                "protocol_sha256": protocol_sha, "config_sha256": config_sha,
                "heldout_opened": False,
                "limitations": ["held-out panel remains sealed until every frozen competence gate and endpoint replay pass"],
            }
            _atomic_json(fit_dir / "result.json", result)

        all_gates = _all_gates_passed(stage_records, len(config["stages"]))
        _atomic_json(fit_dir / "status.json", {
            "status": result["status"], "dimension": dimension, "seed": seed,
            "update": update, "stage": len(stage_records),
            "all_competence_gates_passed": all_gates,
            "heldout_opened": heldout_opened,
            "elapsed_seconds": time.monotonic() - start,
            "checkpoint_sha256": result["checkpoint_sha256"]})
        inventory = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                     for p in sorted(fit_dir.iterdir()) if p.is_file()}
        _atomic_json(fit_dir / "artifact-inventory.json", inventory)
        reporter.log({"result/all_competence_gates_passed": int(all_gates),
                      "result/heldout_opened": int(heldout_opened),
                      "result/interrupted": int(interrupted), "result/updates": update},
                     step=report_step)
        reporter.log_manifest("artifact-inventory", metadata={"files": len(inventory)},
                              identities={"inventory_sha256": file_sha256(fit_dir / "artifact-inventory.json")})
    return result


def _validation_stage(model, torch, bits, designs, dimension, repeats, args):
    """Retained as a named boundary for testing endpoint and partial semantics."""
    all_arrays: dict[str, np.ndarray] = {}
    summaries = []
    for repeat in repeats:
        logits = np.zeros((len(bits), 2, dimension), dtype=np.float32)
        completed = 0
        try:
            model.eval()
            with torch.no_grad():
                for start in range(0, len(bits), 32):
                    v1._guard(args)
                    stop = min(len(bits), start + 32)
                    x_np = v1.render_pairs(bits[start:stop], designs[repeat][start:stop], dimension)
                    output = model(torch.from_numpy(x_np))[:, :dimension]
                    if not torch.isfinite(output).all():
                        raise RuntimeError("nonfinite endpoint validation output")
                    logits[start:stop] = output.cpu().numpy().reshape(stop - start, 2, dimension)
                    completed = stop
        except v1.StopRequested:
            target = np.broadcast_to((bits > 0)[:, None, :], logits.shape).astype(np.uint8)
            all_arrays.update({f"logits_R{repeat}": logits, f"labels_R{repeat}": target,
                               f"completed_cues_R{repeat}": np.asarray(completed, dtype=np.int32),
                               f"complete_R{repeat}": np.asarray(False)})
            _save_npz(args.validation_partial_path, all_arrays)
            raise
        target = np.broadcast_to((bits > 0)[:, None, :], logits.shape).astype(np.uint8)
        predictions = (logits >= 0).astype(np.uint8)
        correct = (predictions == target).astype(np.uint8)
        summary = v1._accuracy(logits[None], bits)
        summary.update({"repeats": repeat, "cue_vectors": len(bits), "active_bits": dimension,
                        "bit_correct_counts_by_arm": correct.sum(axis=0).tolist(),
                        "bit_denominators_by_arm": [len(bits)] * 2,
                        "polarity_order": ["positive", "negative"],
                        "polarity_correct_counts_by_arm": {
                            arm: [[int(correct[bits[:, bit] > 0, arm_idx, bit].sum()),
                                   int(correct[bits[:, bit] < 0, arm_idx, bit].sum())]
                                  for bit in range(dimension)]
                            for arm_idx, arm in enumerate(v1.ARM_NAMES)},
                        "polarity_denominators_per_bit": [
                            [int(np.sum(bits[:, bit] > 0)), int(np.sum(bits[:, bit] < 0))]
                            for bit in range(dimension)]})
        summaries.append(summary)
        all_arrays.update({f"logits_R{repeat}": logits, f"labels_R{repeat}": target,
                           f"predictions_R{repeat}": predictions, f"correct_R{repeat}": correct,
                           f"bit_correct_counts_R{repeat}": correct.sum(axis=0).astype(np.int32),
                           f"bit_denominators_R{repeat}": np.full((2, dimension), len(bits), dtype=np.int32),
                           f"polarity_correct_counts_R{repeat}": np.asarray(
                               [summary["polarity_correct_counts_by_arm"][arm]
                                for arm in v1.ARM_NAMES], dtype=np.int32),
                           f"polarity_denominators_R{repeat}": np.asarray(
                               summary["polarity_denominators_per_bit"], dtype=np.int32),
                           f"complete_R{repeat}": np.asarray(True)})
    return {"repeats": list(repeats), "checks": summaries}, all_arrays


def _endpoint_validation_replay(model, torch, bits, designs, dimension, repeats, args):
    """Return None on an orderly stop; its partial validation NPZ is already durable."""
    try:
        return _validation_stage(model, torch, bits, designs, dimension, repeats, args)
    except v1.StopRequested:
        return None, {}


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    v1._write_npz_atomic(path, **arrays)


def _validation_stage_save(model, torch, bits, designs, dimension, repeats, args,
                           raw_path: Path, partial_path: Path):
    args.validation_partial_path = partial_path
    value, arrays = _validation_stage(model, torch, bits, designs, dimension, repeats, args)
    arrays["complete"] = np.asarray(True)
    arrays["cue_bits"] = bits.astype(np.int8)
    arrays["arm_names"] = np.asarray(v1.ARM_NAMES)
    arrays["polarity_names"] = np.asarray(("positive", "negative"))
    _save_npz(raw_path, arrays)
    return value


def _evaluation_metrics(label: str, item: Mapping[str, Any]) -> dict[str, Any]:
    geometry = item["conditional_complete_state_geometry"]["conditional_spectra"]
    prs = [row["participation_ratio"] for row in geometry if row["participation_ratio"] is not None]
    return {
        f"evaluation/{label}/paired_error_difference": item["paired_error_difference_periodic_minus_control"],
        f"evaluation/{label}/mean_bit_accuracy": float(np.mean([
            *item["per_bit_accuracy_by_arm"]["periodic"],
            *item["per_bit_accuracy_by_arm"]["control"]])),
        f"evaluation/{label}/mean_bit_error_periodic": item["mean_bit_error_by_arm"]["periodic"],
        f"evaluation/{label}/mean_bit_error_control": item["mean_bit_error_by_arm"]["control"],
        f"evaluation/{label}/exact_accuracy_periodic": item["whole_vector_exact_accuracy_by_arm"]["periodic"],
        f"evaluation/{label}/exact_accuracy_control": item["whole_vector_exact_accuracy_by_arm"]["control"],
        f"evaluation/{label}/zero_state_mean_bit_accuracy": float(np.mean(item["zero_state_readout_bit_accuracy"])),
        f"evaluation/{label}/mean_complete_state_trace": float(np.mean([row["trace"] for row in geometry])),
        f"evaluation/{label}/mean_complete_state_pr": float(np.mean(prs)) if prs else None,
        f"evaluation/{label}/defined_complete_state_pr_spectra": len(prs),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _logits_sha(value: np.ndarray) -> str:
    return _sha_np(value.astype(np.float32, copy=False))


def _run_one_seed(config_path: Path, protocol_path: Path, output: Path, run_id: str,
                  seed: int, deadline_unix: float, stop_file: Path) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(run_id) or not _SAFE_ID.fullmatch(f"{run_id}-s{seed}"):
        raise ValueError("run ID must use frozen safe syntax")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed must be one of the configured independent seeds")
    if type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix) or deadline_unix <= time.time():
        raise ValueError("deadline must be finite and in the future")
    config_bytes = config_path.read_bytes()
    config = validate_config(json.loads(config_bytes.decode("ascii")))
    if seed not in config["seeds"]:
        raise ValueError("CLI seed is absent from the frozen config")
    protocol_path = protocol_path.resolve(strict=True)
    source_path = Path(__file__).resolve()
    primitive_path = Path(v1.__file__).resolve()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    protocol_sha = file_sha256(protocol_path)
    source_sha = file_sha256(source_path)
    primitive_sha = file_sha256(primitive_path)
    output = output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    args = v1._Args(stop_file.resolve(), float(deadline_unix), output)
    v1._SIGNAL_STOP = False
    prior_handler = signal.signal(signal.SIGTERM, v1._signal_stop)
    results = []
    try:
        import torch
        if torch.__version__ != "2.0.0+cpu" or np.__version__ != "1.24.4":
            raise RuntimeError("runtime differs from pinned CPU Torch/NumPy environment")
        v1._configure_torch(torch)
        run_identity = {"schema": "multibit-load-ladder-seed-run-v2", "run_id": run_id,
                        "seed": seed, "source_sha256": source_sha,
                        "primitive_source_sha256": primitive_sha,
                        "protocol_sha256": protocol_sha, "config_sha256": config_sha,
                        "git_commit": _git_commit(), "hostname": socket.gethostname(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "deadline_unix": deadline_unix, "torch_version": torch.__version__,
                        "numpy_version": np.__version__,
                        "python_version": platform.python_version(),
                        "created_at_utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(output / "run-identity.json", run_identity)
        _atomic_json(output / "campaign-status.json", {"status": "running", "seed": seed,
                                                        "completed_dimensions": []})
        for dimension in config["dimensions"]:
            try:
                v1._guard(args)
            except v1.StopRequested:
                break
            try:
                item = _fit_one(config, dimension, seed, args, run_id=run_id,
                                source_sha=source_sha, primitive_sha=primitive_sha,
                                protocol_sha=protocol_sha, config_sha=config_sha)
            except Exception as exc:
                _atomic_json(output / f"D{dimension}-engineering-failure.json", {
                    "status": "engineering_failed", "dimension": dimension, "seed": seed,
                    "error_type": type(exc).__name__, "message": str(exc)[:500]})
                raise
            results.append(item)
            _atomic_json(output / "campaign-status.json", {
                "status": "stopped_partial" if item["status"] == "stopped_partial" else "running",
                "seed": seed, "completed_dimensions": [x["dimension"] for x in results],
                "last_dimension": dimension, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            if item["status"] == "stopped_partial":
                break
        final_status = ("stopped_partial" if v1._SIGNAL_STOP or stop_file.exists()
                        or time.time() >= deadline_unix else "completed")
        final = {"schema": "multibit-load-ladder-campaign-v2", "status": final_status,
                 "seed": seed, "fits": results, "source_sha256": source_sha,
                 "primitive_source_sha256": primitive_sha, "protocol_sha256": protocol_sha,
                 "config_sha256": config_sha}
        _atomic_json(output / "campaign-result.json", final)
        _atomic_json(output / "campaign-status.json", {
            "status": final_status, "seed": seed,
            "completed_dimensions": [x["dimension"] for x in results],
            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        return final
    except Exception as exc:
        _atomic_json(output / "campaign-status.json", {
            "status": "engineering_failed", "seed": seed,
            "completed_dimensions": [x["dimension"] for x in results],
            "error_type": type(exc).__name__, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_handler)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--stop-file", required=True, type=Path)
    args = parser.parse_args(argv)
    result = _run_one_seed(args.config, args.protocol, args.output, args.run_id,
                           args.seed, args.deadline_unix, args.stop_file)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "seed": args.seed, "dimensions_completed": len(result["fits"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
