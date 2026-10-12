"""Bounded additive native-memory recovery trainer; no scientific acceptance.

The frozen v1 pilot is an architecture/atomic-writer dependency only. This engine
does not change its task, training, checkpoints or reported outcomes. No service,
scheduler, dataset download or automatic learning-rate sweep is invoked here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from .competence import IMMEDIATE, R0, R1, R2, NATIVE_PHASE, PROTOCOL, CompetenceBatch, _batch_hash, assess_stage_competence, evaluate_competence, make_competence_batch
from .diagnostics import SingleWriteBatch, paired_batch_with_zero, single_write_episode
from .pilot import VectorMemoryModel, atomic_checkpoint, atomic_json, atomic_npz

LR_GRID = [0.0005, 0.001, 0.003]
TRAIN_ASSAYS = [[IMMEDIATE], [R0], [R0, R1], [R0, R1, R2],
                [f"full_prefix_r{r}" for r in (1, 2, 4, 8, 16)]]
STAGE_UPDATES = [500, 500, 500, 500, 1500]
SHORT_ASSAYS = [R0, R1, R2]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config):
    """Validate the full frozen protocol before any engineering reductions."""
    if config.get("lr_grid") != LR_GRID or config.get("selection_horizons") != [1, 2, 4, 8, 16]:
        raise ValueError("frozen learning-rate grid and selection horizons required")
    dimensions = config.get("dimensions", {})
    if any(dimensions.get(k) != v for k, v in {"hidden_size": 128, "input_bottleneck": 64, "batch_size": 32}.items()):
        raise ValueError("frozen dimensions are hidden128/encoder64/batch32")
    optimizer = config.get("optimizer", {})
    if any(optimizer.get(k) != v for k, v in {"name": "AdamW", "weight_decay": 0.01, "gradient_clip": 1.0}.items()):
        raise ValueError("frozen AdamW/weight-decay/gradient-clip contract required")
    if config.get("validation_interval") != 100:
        raise ValueError("validation interval must be100")
    _integer(config.get("cpu_threads", 1), "cpu_threads")
    stages = config.get("stages", [])
    if len(stages) != 5:
        raise ValueError("five declared ladder stages required")
    for index, stage in enumerate(stages):
        if stage.get("stage") != index + 1 or stage.get("updates") != STAGE_UPDATES[index] or stage.get("train_assays") != TRAIN_ASSAYS[index]:
            raise ValueError("stage schedule differs from the frozen ladder")
    diagnostic = config.get("diagnostics", {})
    if diagnostic.get("fixed_batch_updates") != 100 or diagnostic.get("fixed_batch_examples_per_coordinate") != 4 or diagnostic.get("stop_on_fixed_batch_failure") is not True:
        raise ValueError("stage1 requires100 fixed-batch updates,4examples/coordinate and failure stop")
    validation = config.get("validation", {})
    expected_validation = {"episodes_per_coordinate_per_arm": 128, "required_coordinates": list(range(8)),
                           "categorical_min_accuracy": 0.95, "continuous_max_population_normalized_mse": 0.05,
                           "population_variance": 1 / 3, "query_phase": NATIVE_PHASE}
    if any(validation.get(k) != v for k, v in expected_validation.items()):
        raise ValueError("frozen native coordinate validation contract required")
    precision = config.get("precision", {})
    expected_precision = {"training_dtype": "float32", "cuda_matmul_allow_tf32": False,
                          "cudnn_allow_tf32": False, "float32_matmul_precision": "highest",
                          "deterministic_algorithms": True, "record_actual_flags": True}
    if any(precision.get(k) != v for k, v in expected_precision.items()):
        raise ValueError("purefloat32 deterministic precision contract required")


def _effective(config, overrides):
    result = deepcopy(config)
    if not overrides:
        return result
    allowed = {"stage_updates", "hidden_size", "input_bottleneck", "batch_size", "validation_interval", "fixed_batch_updates", "fixed_batch_examples_per_coordinate"}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise ValueError("unknown engineering override")
    for key in ("hidden_size", "input_bottleneck", "batch_size"):
        if key in overrides:
            value = _integer(overrides[key], key)
            if value > result["dimensions"][key] or (key == "batch_size" and value % 2):
                raise ValueError("engineering dimensions must reduce the budget; paired batch must be even")
            result["dimensions"][key] = value
    if "stage_updates" in overrides:
        values = overrides["stage_updates"]
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError("five engineering stage budgets required")
        for index, value in enumerate(values):
            if _integer(value, "stage_updates") > STAGE_UPDATES[index]:
                raise ValueError("engineering stage budget cannot expand")
            result["stages"][index]["updates"] = value
    for key in ("fixed_batch_updates", "fixed_batch_examples_per_coordinate"):
        if key in overrides:
            if _integer(overrides[key], key) > config["diagnostics"][key]:
                raise ValueError("engineering fixed-batch budget cannot expand")
            result["diagnostics"][key] = overrides[key]
    if result["diagnostics"]["fixed_batch_updates"] > result["stages"][0]["updates"]:
        raise ValueError("fixed-batch updates exceed the immediate-stage budget")
    if "validation_interval" in overrides:
        if _integer(overrides["validation_interval"], "validation_interval") > 100:
            raise ValueError("engineering validation interval cannot expand")
        result["validation_interval"] = overrides["validation_interval"]
    return result


def _loss(logits, targets, task):
    return F.cross_entropy(logits, targets) if task == "categorical" else F.mse_loss(logits[:, 0], targets)


def _targets(values, task, device):
    return torch.as_tensor(np.array(values, copy=True), dtype=torch.long if task == "categorical" else torch.float32, device=device)


def _finite(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise FloatingPointError(f"nonfinite {name}")
    return result


def _module_norm(module, gradients=False):
    values = [p.grad if gradients else p.detach() for p in module.parameters()]
    return _finite(torch.sqrt(sum(v.double().square().sum() for v in values if v is not None)), "module norm")


def _fixed_batch(rng, count_per_coordinate, task):
    batches = [single_write_episode(rng, count_per_coordinate, task, coordinate) for coordinate in range(8)]
    return SingleWriteBatch(*(np.concatenate([getattr(b, name) for b in batches]) for name in ("inputs", "targets", "queries")))


def _fixed_overfit(model, batch, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(batch.inputs, dtype=torch.float32, device=device))
        predictions = logits.argmax(1).cpu().numpy() if model.task == "categorical" else logits[:, 0].cpu().numpy()
    rows = []
    for coordinate in range(8):
        selected = batch.queries == coordinate
        if model.task == "categorical":
            metric = float((predictions[selected] == batch.targets[selected]).mean())
            passed = metric >= PROTOCOL.categorical_min_accuracy - PROTOCOL.comparison_ulps * np.spacing(PROTOCOL.categorical_min_accuracy)
        else:
            metric = _finite(np.square(predictions[selected].astype(np.float64) - batch.targets[selected]).mean() / (1 / 3), "fixed-batch normalizedMSE")
            passed = metric <= PROTOCOL.continuous_max_normalized_mse + PROTOCOL.comparison_ulps * np.spacing(PROTOCOL.continuous_max_normalized_mse)
        rows.append({"coordinate": coordinate, "episodes": int(selected.sum()), "metric": metric, "passed": bool(passed)})
    return {"passed": all(r["passed"] for r in rows), "coordinates": rows, "scope": "fixed training batch diagnostic; not128episode competence", "scientific_acceptance": False}


def _long_manifest(repeats, task, seed):
    rng = np.random.default_rng(seed)
    inputs, controls, targets, queries = [], [], [], []
    for coordinate in range(8):
        batch = paired_batch_with_zero(rng, 128, repeats, task)
        x = batch.inputs.copy()
        x[:, -1, 5:21] = 0
        x[:, -1, 5 + coordinate] = 1
        inputs.append(x[:128]); controls.append(x[128:])
        targets.append(batch.memory[:, coordinate]); queries.append(np.full(128, coordinate, dtype=np.int64))
    return {"inputs": np.concatenate(inputs + controls), "targets": np.tile(np.concatenate(targets), 2),
            "queries": np.tile(np.concatenate(queries), 2), "pair_ids": np.tile(np.arange(1024), 2), "sampling_seed": seed}


def _manifest_arrays(batch):
    return {key: np.asarray(batch[key] if isinstance(batch, dict) else getattr(batch, key)) for key in ("inputs", "targets", "queries", "pair_ids")}


def _validation_identity(assay, task, batch):
    """Same metadata/dtype/shape/byte hash as the coordinate-score contract."""
    if isinstance(batch, dict):
        canonical = CompetenceBatch(assay, task, **_manifest_arrays(batch), sampling_seed=batch["sampling_seed"])
    else:
        canonical = batch
    return {"batch_sha256": _batch_hash(canonical), "sampling_seed": int(canonical.sampling_seed)}


def _verify_validation_freeze(manifests, assays, task, validation_manifest, immediate_manifest):
    for assay in assays:
        expected = immediate_manifest if assay == IMMEDIATE else validation_manifest[f"R{int(assay.rsplit('r', 1)[1])}"]
        if _validation_identity(assay, task, manifests[assay]) != expected:
            raise ValueError("validation batch differs from the pre-optimization caller freeze")


def evaluate_assays(model, manifests, assays, device, seed, checkpoint_id, model_kind):
    """Chunked native chronology checks and native CE/rawMSE; no test split."""
    from .native_state import native_phase_capture
    records, losses, raw, chronology = [], {}, {}, {}
    for assay in assays:
        batch = manifests[assay]
        arrays = _manifest_arrays(batch)
        logits_parts, phase_parts, checks = [], {}, []
        for start in range(0, len(arrays["targets"]), 64):
            inputs = torch.as_tensor(np.array(arrays["inputs"][start:start + 64], copy=True), dtype=torch.float32, device=device)
            captured = native_phase_capture(model, inputs, write_count=1 if assay == IMMEDIATE else 16)
            if captured["chronology"]["passed"] is not True:
                raise FloatingPointError("forked/uninterrupted native query chronology failed")
            # Native behavior is the uninterrupted full forward; forked output
            # is a tolerance-qualified chronology check, never its replacement.
            logits_parts.append(captured["uninterrupted_logits"].detach().cpu().numpy())
            checks.append(captured["chronology"])
            for name, state in captured["phase_states"].items():
                phase_parts.setdefault(name, []).append(state.detach().cpu().numpy())
        logits = np.concatenate(logits_parts)
        if not np.isfinite(logits).all():
            raise FloatingPointError("nonfinite validation logits")
        prediction = logits.argmax(1) if model.task == "categorical" else logits[:, 0]
        arm_count = 1 if assay == IMMEDIATE else 2
        n = len(prediction) // arm_count
        arm_loss = {}
        for index, arm in enumerate(("immediate",) if arm_count == 1 else ("periodic", "control")):
            sl = slice(index * n, (index + 1) * n)
            arm_loss[arm] = _finite(_loss(torch.as_tensor(logits[sl]), _targets(arrays["targets"][sl], model.task, "cpu"), model.task), "validation task loss")
        losses[assay] = arm_loss
        if assay in (IMMEDIATE, R0, R1, R2):
            record = evaluate_competence(batch, prediction, trained_seed=seed, checkpoint_id=checkpoint_id, measurement_phase=NATIVE_PHASE)
            record["model"] = model_kind
            records.append(record)
        chronology[assay] = {"passed": True, "chunks": checks,
                            "validation_manifest": _validation_identity(assay, model.task, batch)}
        if assay not in (IMMEDIATE, R0, R1, R2):
            # Report longer-horizon competence without using it for eligibility.
            long_scores = {}
            for arm_index, arm in enumerate(("periodic", "control")):
                sl = slice(arm_index * n, (arm_index + 1) * n)
                rows = []
                for coordinate in range(8):
                    mask = arrays["queries"][sl] == coordinate
                    if model.task == "categorical":
                        metric = _finite((prediction[sl][mask] == arrays["targets"][sl][mask]).mean(), "long-coordinate accuracy")
                        passed = metric >= PROTOCOL.categorical_min_accuracy - PROTOCOL.comparison_ulps * np.spacing(0.95)
                    else:
                        metric = _finite(np.square(prediction[sl][mask].astype(np.float64) - arrays["targets"][sl][mask]).mean() / (1 / 3), "long-coordinate normalizedMSE")
                        passed = metric <= PROTOCOL.continuous_max_normalized_mse + PROTOCOL.comparison_ulps * np.spacing(0.05)
                    rows.append({"coordinate": coordinate, "episodes": int(mask.sum()), "metric": metric, "passed": bool(passed) and int(mask.sum()) == 128})
                long_scores[arm] = {"coordinates": rows, "passed": all(r["passed"] for r in rows)}
            chronology[assay]["long_horizon_coordinate_competence"] = {"arms": long_scores, "selection_eligibility": False, "scientific_acceptance": False}
        for name, value in {**arrays, "native_logits": logits, "predictions": prediction,
                            **{k: np.concatenate(v) for k, v in phase_parts.items()}}.items():
            raw[f"{assay}_{name}"] = value
    return records, losses, raw, chronology


def boundary_decision(stage, updates_completed, stage_budget, current_records):
    """A historical best checkpoint cannot release a current failing boundary."""
    if updates_completed != stage_budget:
        return {"at_boundary": False, "advance": False, "reason": "only declared boundary can advance"}
    report = assess_stage_competence(current_records, stage)
    return {"at_boundary": True, "advance": report["coordinate_requirement_met"], "coordinate_report": report}


def _hash_files(output):
    atomic_json(output / "manifest.json", {"schema_version": 1, "sha256": {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")}})


def run(config, output, model_kind, task, seed, learning_rate, *, device="cpu", engineering_overrides=None, config_path=None):
    """Run one explicitly chosen LR fit; all stops and failed gates are retained."""
    from .competence_selection import select_checkpoint
    validate_config(config)
    if model_kind not in ("gru", "lstm") or task not in ("categorical", "continuous"):
        raise ValueError("unsupported model/task")
    _integer(seed, "seed", 0)
    if seed not in config.get("discovery_seeds", [0]):
        raise ValueError("seed is outside the declared discovery registry")
    if learning_rate not in LR_GRID:
        raise ValueError("learning rate must be explicitly chosen from the frozen grid")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested device unavailable")
    effective = _effective(config, engineering_overrides)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"state": "running", "updated_at": _now(), "scientific_acceptance": False})
    effective.update(model=model_kind, task=task, seed=seed, learning_rate=learning_rate, device=device,
                     engineering_overrides=deepcopy(engineering_overrides), scientific_acceptance=False,
                     validation_sampling_seed_base=seed + 10000, fixed_batch_sampling_seed=seed + 5000)
    atomic_json(output / "config.json", effective)
    history, candidates, stage_best, stage_results, manifests, exposure = [], [], {}, [], {}, Counter()
    step, outcome, fixed_result = 0, "completed_schedule", None
    try:
        torch.set_num_threads(effective.get("cpu_threads", 1))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        dimensions = effective["dimensions"]
        model = VectorMemoryModel(model_kind, task, dimensions["hidden_size"], dimensions["input_bottleneck"]).to(device=device, dtype=torch.float32)
        # Freeze and persist every selection/diagnostic batch before the first
        # optimization update, including horizons not yet released by gates.
        manifest_directory = output / "validation_manifests"
        manifest_directory.mkdir()
        for assay_index, name in enumerate([IMMEDIATE, R0, R1, R2, "full_prefix_r4", "full_prefix_r8", "full_prefix_r16"]):
            sampling_seed = seed + 10000 + assay_index
            manifests[name] = make_competence_batch(name, task, sampling_seed) if name in (IMMEDIATE, R0, R1, R2) else _long_manifest(int(name.rsplit("r", 1)[1]), task, sampling_seed)
            atomic_npz(manifest_directory / f"{name}.npz", _manifest_arrays(manifests[name]))
        validation_manifest = {f"R{r}": _validation_identity(f"full_prefix_r{r}", task, manifests[f"full_prefix_r{r}"])
                               for r in (0, 1, 2, 4, 8, 16)}
        immediate_manifest = _validation_identity(IMMEDIATE, task, manifests[IMMEDIATE])
        effective.update(validation_manifest=deepcopy(validation_manifest),
                         immediate_validation_manifest=immediate_manifest, validation_manifest_frozen_at_step=0)
        atomic_json(output / "config.json", effective)
        atomic_json(output / "validation_manifest.json", {"selection_manifests": validation_manifest,
            "immediate_manifest": immediate_manifest, "frozen_at_step": 0, "policy": "caller freeze before optimization; no outcome-based replacement"})
        root = Path(__file__).resolve().parents[2]
        files = [Path(__file__), root / "uv.lock", root / "pyproject.toml", Path(__file__).with_name("pilot.py"), Path(__file__).with_name("competence.py"), Path(__file__).with_name("native_state.py"), Path(__file__).with_name("competence_selection.py")]
        if config_path is not None: files.append(Path(config_path))
        atomic_json(output / "provenance.json", {"schema_version": 1, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "dirty_checkout": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
            "source_sha256": {str(p.relative_to(root)) if p.is_relative_to(root) else p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": device,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "validation_manifest": deepcopy(validation_manifest), "immediate_validation_manifest": immediate_manifest,
            "validation_manifest_frozen_at_step": 0,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_version": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scientific_acceptance": False, "started_at": _now()})
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        fixed = _fixed_batch(np.random.default_rng(seed + 5000), effective["diagnostics"]["fixed_batch_examples_per_coordinate"], task)
        atomic_npz(output / "fixed_batch.npz", {name: getattr(fixed, name) for name in ("inputs", "targets", "queries")})
        initial_fixed_loss = _finite(_loss(model(torch.as_tensor(fixed.inputs, device=device)), _targets(fixed.targets, task, device), task).detach(), "initial fixed loss")
        fixed_module_changes = {name: 0.0 for name in ("encoder", "core", "head")}
        for stage_config in effective["stages"]:
            stage, budget = stage_config["stage"], stage_config["updates"]
            for local_step in range(1, budget + 1):
                assay = stage_config["train_assays"][(local_step - 1) % len(stage_config["train_assays"])]
                exposure[f"stage{stage}:{assay}"] += 1
                if stage == 1:
                    batch = fixed if local_step <= effective["diagnostics"]["fixed_batch_updates"] else single_write_episode(rng, dimensions["batch_size"], task)
                else:
                    batch = paired_batch_with_zero(rng, dimensions["batch_size"] // 2, int(assay.rsplit("r", 1)[1]), task)
                inputs = torch.as_tensor(np.array(batch.inputs, copy=True), device=device).requires_grad_(True)
                target = _targets(batch.targets, task, device)
                model.train(); optimizer.zero_grad(set_to_none=True)
                before = {name: [p.detach().clone() for p in getattr(model, name).parameters()] for name in fixed_module_changes}
                loss = _loss(model(inputs), target, task)
                _finite(loss.detach(), "training loss")
                loss.backward()
                gradients = {name: _module_norm(getattr(model, name), True) for name in before}
                query = np.asarray(batch.queries)
                if stage != 1: query = np.tile(query, 2)
                write_indices = np.zeros(len(query), dtype=np.int64) if stage == 1 else query
                write_gradient = _finite(inputs.grad[torch.arange(len(query), device=device), torch.as_tensor(write_indices, device=device)].double().norm(), "queried WRITE gradient norm")
                gradient_norm = _finite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True), "preclip gradient norm")
                optimizer.step()
                changes = {name: _finite(torch.sqrt(sum((p.detach().double() - old.double()).square().sum() for p, old in zip(getattr(model, name).parameters(), values))), "parameter update norm") for name, values in before.items()}
                if stage == 1 and local_step <= effective["diagnostics"]["fixed_batch_updates"]:
                    for name in changes: fixed_module_changes[name] += changes[name]
                for parameter in model.parameters():
                    if not torch.isfinite(parameter).all(): raise FloatingPointError("nonfinite updated parameter")
                step += 1
                row = {"step": step, "stage": stage, "stage_step": local_step, "assay": assay, "train_loss": float(loss.detach()),
                       "module_gradient_norms": gradients, "module_parameter_change_norms": changes,
                       "queried_write_input_gradient_norm": write_gradient, "gradient_norm_before_clip": gradient_norm, "clipped": gradient_norm > 1}
                history.append(row)
                fixed_boundary = stage == 1 and local_step == effective["diagnostics"]["fixed_batch_updates"]
                validation_due = local_step % effective["validation_interval"] == 0 or local_step == budget or fixed_boundary
                if not validation_due: continue
                model.eval()
                checkpoint_id = f"checkpoints/{model_kind}-{task}-seed{seed}-lr{learning_rate:g}-step-{step:06d}.pt"
                checkpoint_path = output / checkpoint_id
                checkpoint_path.parent.mkdir(exist_ok=True)
                atomic_checkpoint(checkpoint_path, {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": step, "stage": stage, "config": effective, "scientific_acceptance": False})
                required = [IMMEDIATE] if stage == 1 else SHORT_ASSAYS[:stage - 1] if stage < 4 else SHORT_ASSAYS.copy()
                if stage == 5: required += [f"full_prefix_r{r}" for r in (4, 8, 16)]
                _verify_validation_freeze(manifests, required, task, validation_manifest, immediate_manifest)
                records, losses, raw, chronology = evaluate_assays(model, manifests, required, device, seed, checkpoint_id, model_kind)
                raw_directory = output / "raw_validation"
                raw_directory.mkdir(exist_ok=True)
                atomic_npz(raw_directory / f"step-{step:06d}.npz", raw)
                row.update(validation_records=records, validation_loss_by_assay=losses, native_chronology=chronology)
                balanced_loss = _finite(np.mean([v for name in required for v in losses[name].values()]), "stage balanced validation loss")
                if stage not in stage_best or balanced_loss < stage_best[stage]["validation_loss"]:
                    stage_best[stage] = {"checkpoint_id": checkpoint_id, "step": step, "validation_loss": balanced_loss, "diagnostic_only": True}
                if fixed_boundary:
                    fixed_result = _fixed_overfit(model, fixed, device)
                    fixed_result.update(initial_loss=initial_fixed_loss, module_parameter_changes=fixed_module_changes.copy())
                    fixed_result["optimization_changes_observed"] = all(v > 0 for v in fixed_module_changes.values())
                    row["fixed_batch_diagnostic"] = fixed_result
                    if not fixed_result["passed"] or not fixed_result["optimization_changes_observed"]:
                        outcome = "stopped_fixed_batch_failure"
                if stage == 5:
                    candidates.append({"model": model_kind, "task": task, "trained_seed": seed, "learning_rate": learning_rate,
                        "step": step, "stage": 5, "checkpoint_id": checkpoint_id, "checkpoint_path": str(checkpoint_path),
                        "split": "validation", "native_validation": True, "short_horizon_records": records,
                        "validation_manifest": deepcopy(validation_manifest),
                        "validation_loss_by_horizon": {f"R{r}": losses[f"full_prefix_r{r}"] for r in (1, 2, 4, 8, 16)}})
                atomic_json(output / "history.json", history)
                atomic_json(output / "status.json", {"state": "running", "step": step, "stage": stage, "updated_at": _now(), "scientific_acceptance": False})
                if outcome != "completed_schedule": break
                if local_step == budget:
                    boundary = boundary_decision(stage, local_step, budget, records)
                    stage_results.append({"stage": stage, "checkpoint_id": checkpoint_id, **boundary})
                    if stage < 5 and not boundary["advance"]:
                        outcome = "stopped_coordinate_gate_failure"
                        break
            if outcome != "completed_schedule": break
        selection = select_checkpoint(candidates, LR_GRID, validation_manifest=validation_manifest) if candidates else {
            "selected_index": None, "selected_checkpoint_id": None, "selected": None,
            "candidate_reviews": [], "scientific_acceptance": False,
            "status": "no_stage5_candidate; acquisition_or_optimization_unresolved"}
        metrics = {"schema_version": 1, "kind": "bounded_competence_recovery_discovery", "scientific_acceptance": False,
            "model": model_kind, "task": task, "seed": seed, "learning_rate": learning_rate, "steps": step,
            "outcome": outcome, "stage_results": stage_results, "stage_best_diagnostics": stage_best,
            "horizon_exposure_updates": dict(exposure), "selection": selection, "fixed_batch_diagnostic": fixed_result,
            "validation_manifest": deepcopy(validation_manifest), "validation_manifest_frozen_at_step": 0,
            "engineering_override": bool(engineering_overrides), "statistical_unit": "one trained seed; discovery evidence only",
            "remaining_requirements": ["independent artifact acceptance", "repeat-run determinism", "activation/gate saturation assay", "untouched feasibility/confirmation seeds", "causal intervention/sham and precision-resolved measurements"]}
        atomic_json(output / "candidates.json", candidates)
        atomic_json(output / "metrics.json", metrics)
        atomic_json(output / "status.json", {"state": "completed", "outcome": outcome, "step": step, "updated_at": _now(), "scientific_acceptance": False})
        _hash_files(output)
        return metrics
    except Exception as error:
        atomic_json(output / "history.json", history)
        atomic_json(output / "status.json", {"state": "failed", "step": step, "error_type": type(error).__name__, "error": str(error), "updated_at": _now(), "scientific_acceptance": False})
        _hash_files(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("gru", "lstm"), required=True)
    parser.add_argument("--task", choices=("categorical", "continuous"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--engineering-overrides", type=Path)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.output, args.model, args.task, args.seed, args.learning_rate,
                 device=args.device, engineering_overrides=json.loads(args.engineering_overrides.read_text()) if args.engineering_overrides else None, config_path=args.config)
    print(json.dumps({"output": str(args.output), "outcome": result["outcome"], "steps": result["steps"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
