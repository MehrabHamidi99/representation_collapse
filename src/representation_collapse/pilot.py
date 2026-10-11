"""Bounded Torch vector-memory engineering pilot; not scientific acceptance."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import covariance_dimension
from .vector_task import INPUT_DIM, MOTIFS, body_from_ids, paired_batch, paired_body_ids, render, sample_memory


class VectorMemoryModel(nn.Module):
    """One-layer recurrent adapter; query updates state only after measurement."""

    def __init__(self, model: str, task: str, hidden_size: int = 128, bottleneck: int = 64):
        super().__init__()
        if model not in {"gru", "lstm"} or task not in {"categorical", "continuous"}:
            raise ValueError("unsupported model/task")
        self.model_kind, self.task, self.hidden_size = model, task, hidden_size
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, bottleneck), nn.Tanh())
        recurrent = nn.GRU if model == "gru" else nn.LSTM
        self.core = recurrent(bottleneck, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 16 if task == "categorical" else 1)

    def pack_state(self, state) -> torch.Tensor:
        if self.model_kind == "lstm":
            h, c = state
            return torch.cat([h[0], c[0]], dim=1)
        return state[0]

    def pre_query_state(self, inputs: torch.Tensor):
        _, state = self.core(self.encoder(inputs[:, :-1]))
        return state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.core(self.encoder(inputs))
        return self.head(output[:, -1])

    def read_query(self, query: torch.Tensor, state) -> torch.Tensor:
        output, _ = self.core(self.encoder(query[:, None]), state)
        return self.head(output[:, -1])


def atomic_json(path: Path, value) -> None:
    if path.name == "status.json":
        value = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "cluster": os.environ.get("CLUV_CLUSTER_NAME"), **value}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, values: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tensor(values, device, task: str | None = None):
    dtype = torch.long if task == "categorical" else torch.float32
    return torch.as_tensor(values, dtype=dtype, device=device)


def _loss(prediction, targets, task):
    if task == "categorical":
        return F.cross_entropy(prediction, targets)
    return F.mse_loss(prediction[:, 0], targets)


def _predictions(prediction, task):
    return prediction.argmax(dim=1) if task == "categorical" else prediction[:, 0]


@torch.no_grad()
def evaluate_batch(model, batch, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    logits = model(_tensor(batch.inputs, device))
    targets = _tensor(batch.targets, device, model.task)
    prediction = _predictions(logits, model.task)
    errors = (prediction != targets).float() if model.task == "categorical" else (prediction - targets).square()
    return float(_loss(logits, targets, model.task)), prediction.cpu().numpy(), errors.cpu().numpy()


@torch.no_grad()
def state_cloud(model, memory, ids, q, device) -> np.ndarray:
    inputs = render(memory, np.zeros(len(memory), dtype=np.int64), [body_from_ids(ids, q)] * len(memory), model.task)
    return model.pack_state(model.pre_query_state(_tensor(inputs, device))).cpu().numpy()


def ridge_probe(train_states, test_states, train_tags, test_tags, task, alpha, rng):
    """Fixed-alpha linear probes; labels and controls never fit on test cues."""
    x = np.asarray(train_states, dtype=np.float64)
    z = np.asarray(test_states, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    x, z = (x - mean) / scale, (z - mean) / scale
    y = np.eye(16)[train_tags.astype(np.int64)].reshape(len(x), -1) if task == "categorical" else train_tags.astype(np.float64)
    permutation = rng.permutation(len(y))
    results, arrays = {}, {"probe_feature_mean": mean, "probe_feature_scale": scale, "control_permutation": permutation}
    system = x.T @ x + float(alpha) * len(x) * np.eye(x.shape[1])
    for name, labels in (("real", y), ("label_control", y[permutation])):
        intercept = labels.mean(axis=0)
        coefficients = np.linalg.solve(system, x.T @ (labels - intercept))
        prediction = z @ coefficients + intercept
        if task == "categorical":
            prediction = prediction.reshape(len(z), 8, 16).argmax(axis=2)
            score = (prediction == test_tags).mean(axis=0)
            results[name] = {"mean_accuracy": float(score.mean()), "coordinate_accuracy": score.tolist(), "chance_accuracy": 1 / 16}
        else:
            score = np.square(prediction - test_tags).mean(axis=0)
            results[name] = {"mean_mse": float(score.mean()), "coordinate_mse": score.tolist(), "zero_predictor_mse": float(np.square(test_tags).mean())}
        arrays[name + "_predictions"] = prediction
        arrays[name + "_coefficients"] = coefficients
        arrays[name + "_intercept"] = intercept
    return results, arrays


def validate_config(config: dict) -> None:
    positive = ["hidden_size", "input_bottleneck", "training_steps", "batch_size", "validation_interval", "validation_pairs_per_horizon", "evaluation_pairs_per_horizon", "cloud_driver_pairs_per_horizon", "cloud_samples", "probe_test_samples", "cpu_threads"]
    for key in positive:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["batch_size"] % 2 or config["cloud_samples"] < 2 or config["probe_test_samples"] < 2:
        raise ValueError("paired training needs even batch size; clouds need at least two samples")
    if config["dwell"] != 1 or config["train_repeats"] != [1, 2, 4, 8, 16]:
        raise ValueError("v1 supports preregistered q=1 and train horizons1,2,4,8,16")
    if not {1, 2, 16}.issubset(config["evaluation_repeats"]) or any(r not in {1, 2, 4, 8, 16} for r in config["evaluation_repeats"]) or any(r not in {32, 64} for r in config["secondary_repeats"]):
        raise ValueError("evaluation requires1,2,16; optional secondary horizons32/64")
    if len(set(config["evaluation_repeats"])) != len(config["evaluation_repeats"]) or len(set(config["secondary_repeats"])) != len(config["secondary_repeats"]):
        raise ValueError("duplicate evaluation horizons are not allowed")
    for key in ["learning_rate", "gradient_clip", "probe_ridge"]:
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["optimizer"] != "AdamW" or config["write_coordinates"] != 16 or config["queried_coordinates"] != 8 or config["categorical_levels"] != 16 or config["continuous_range"] != [-1, 1]:
        raise ValueError("unsupported task or optimizer configuration")
    if not np.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not config["competence"]["horizons"] or not set(config["competence"]["horizons"]).issubset(config["evaluation_repeats"]):
        raise ValueError("competence horizons must be in primary evaluation")


def provenance(config_path: Path | None, device: str) -> dict:
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), Path(__file__).with_name("vector_task.py"), Path(__file__).with_name("metrics.py"), root / "uv.lock", root / "pyproject.toml"]
    if config_path is not None:
        files.append(config_path.resolve())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, True
    accelerator = None
    if device == "cuda":
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        accelerator = {"index": index, "name": properties.name,
                       "total_memory_bytes": properties.total_memory,
                       "compute_capability": [properties.major, properties.minor],
                       "cuda_runtime": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version()}
    return {
        "schema_version": 1, "git_revision": revision, "dirty_checkout": dirty,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()},
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "device": device, "platform": platform.platform(), "hostname": platform.node(),
        "accelerator": accelerator,
        "allocation": {name: os.environ.get(name) for name in (
            "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS",
            "CUDA_VISIBLE_DEVICES", "CLUV_CLUSTER_NAME")},
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": _now(),
        "deterministic_algorithms": True, "precision": "float32 model/state; float64 covariance/probes",
    }


def _manifest(output):
    atomic_json(output / "manifest.json", {"schema_version": 1, "sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")
    }})


def run(config: dict, output: Path, model_kind: str, task: str, seed: int,
        device: str = "cpu", max_steps: int | None = None, config_path: Path | None = None) -> dict:
    config = deepcopy(config)
    validate_config(config)
    if model_kind not in config["models"] or task not in config["tasks"] or seed not in config["seeds"]:
        raise ValueError("model/task/seed must be in configured pilot registry")
    if max_steps is not None and (max_steps < 1 or max_steps > config["training_steps"]):
        raise ValueError("max-steps must be positive and cannot expand the declared step budget")
    steps = max_steps or config["training_steps"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("requested device is unavailable")
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(seed + 10000)
    evaluation_rng = np.random.default_rng(seed + 20000)
    cloud_rng = np.random.default_rng(seed + 30000)
    effective = {**config, "model": model_kind, "task": task, "seed": seed, "effective_training_steps": steps,
                 "max_steps_override": max_steps is not None, "scientific_acceptance": False,
                 "sampling_stream_seeds": {"training": seed, "validation": seed + 10000, "evaluation": seed + 20000, "clouds": seed + 30000},
                 "motif_library": list(MOTIFS), "device": device}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "config.json", effective)
    atomic_json(output / "provenance.json", provenance(config_path, device))
    atomic_json(output / "status.json", {"state": "running", "updated_at": _now()})
    started = time.monotonic()
    try:
        model = VectorMemoryModel(model_kind, task, config["hidden_size"], config["input_bottleneck"]).to(device)
        core_parameters = sum(p.numel() for p in list(model.core.parameters()) + list(model.head.parameters()))
        persistent_bytes = config["hidden_size"] * (2 if model_kind == "lstm" else 1) * 4
        if core_parameters > 1000000 or persistent_bytes > 1048576:
            raise ValueError("controlled pilot exceeds core/head parameter or persistent-state budget")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        validation = [paired_batch(validation_rng, config["validation_pairs_per_horizon"], r, task) for r in config["train_repeats"]]
        history, best_loss, selected_step = [], float("inf"), 0
        for step in range(1, steps + 1):
            repeats = config["train_repeats"][(step - 1) % len(config["train_repeats"])]
            batch = paired_batch(rng, config["batch_size"] // 2, repeats, task)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(_tensor(batch.inputs, device)), _tensor(batch.targets, device, task), task)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step, "repeats": repeats, "train_loss": float(loss.detach()), "gradient_norm_before_clip": float(gradient_norm)}
            if step % config["validation_interval"] == 0 or step == steps:
                losses = [evaluate_batch(model, b, device)[0] for b in validation]
                validation_loss = float(np.mean(losses))
                row.update(validation_loss=validation_loss, validation_loss_by_horizon=dict(zip(map(str, config["train_repeats"]), losses)))
                if validation_loss < best_loss:
                    best_loss, selected_step = validation_loss, step
                    atomic_checkpoint(output / "checkpoint.pt", {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "selected_step": step, "validation_loss": best_loss, "config": effective})
            history.append(row)
            if "validation_loss" in row:
                atomic_json(output / "history.json", history)
                atomic_json(output / "status.json", {"state": "running", "step": step, "selected_step": selected_step, "updated_at": _now()})
                print(json.dumps({"step": step, "validation_loss": row["validation_loss"], "selected_step": selected_step}), flush=True)
        checkpoint = torch.load(output / "checkpoint.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        behavior, raw = {}, {}
        for repeats in config["evaluation_repeats"] + config["secondary_repeats"]:
            batch = paired_batch(evaluation_rng, config["evaluation_pairs_per_horizon"], repeats, task)
            _, predictions, errors = evaluate_batch(model, batch, device)
            n = config["evaluation_pairs_per_horizon"]
            periodic_error, control_error = float(errors[:n].mean()), float(errors[n:].mean())
            behavior[str(repeats)] = {"periodic_error": periodic_error, "control_error": control_error,
                "delta": periodic_error - control_error, "paired_error_difference_mean": float((errors[:n] - errors[n:]).mean()),
                "pairs": n, "body_updates": repeats * 12, "scope": "secondary_extrapolation" if repeats in config["secondary_repeats"] else "within_training_support"}
            for name, values in {"periodic_errors": errors[:n], "control_errors": errors[n:], "periodic_predictions": predictions[:n],
                "control_predictions": predictions[n:], "targets": batch.targets[:n], "memory": batch.memory, "queries": batch.queries,
                "periodic_motif_ids": batch.periodic_ids, "control_motif_ids": batch.control_ids}.items():
                raw[f"R{repeats}_{name}"] = values
        competence = {}
        for repeats in config["competence"]["horizons"]:
            row = behavior[str(repeats)]
            errors = [row["periodic_error"], row["control_error"]]
            passed = all(1 - e >= config["competence"]["categorical_min_accuracy"] for e in errors) if task == "categorical" else all(e <= config["competence"]["continuous_max_mse"] for e in errors)
            competence[str(repeats)] = {"passed": bool(passed), "periodic_error": errors[0], "control_error": errors[1]}
        cloud_results, clouds = {}, {}
        for repeats in [2, 16]:
            for panel in range(config["cloud_driver_pairs_per_horizon"]):
                periodic_ids, control_ids = paired_body_ids(cloud_rng, repeats)
                train_memory = sample_memory(cloud_rng, config["cloud_samples"], task)
                test_memory = sample_memory(cloud_rng, config["probe_test_samples"], task)
                nuisance = sample_memory(cloud_rng, 1, task)[0, 8:]
                train_memory[:, 8:], test_memory[:, 8:] = nuisance, nuisance
                # Pair the randomized-label control across arms as well as cues.
                probe_control_seed = seed + 40000 + 100 * repeats + panel
                for arm, ids in (("periodic", periodic_ids), ("control", control_ids)):
                    key = f"R{repeats}_panel{panel}_{arm}"
                    train_states = state_cloud(model, train_memory, ids, 1, device)
                    test_states = state_cloud(model, test_memory, ids, 1, device)
                    probe, probe_arrays = ridge_probe(train_states, test_states, train_memory[:, :8], test_memory[:, :8], task, config["probe_ridge"], np.random.default_rng(probe_control_seed))
                    cloud_results[key] = {"geometry": covariance_dimension(train_states), "probes": probe,
                        "body_motif_ids": ids.tolist(), "fixed_nuisance": nuisance.tolist(),
                        "sample_rank_ceiling": min(len(train_states) - 1, train_states.shape[1]), "probe_control_seed": probe_control_seed,
                        "measurement": "finite-time_complete_state_immediately_before_query", "body_updates": 12 * repeats}
                    arrays = {"train_states": train_states, "test_states": test_states, "train_memory": train_memory,
                        "test_memory": test_memory, "motif_ids": ids, **probe_arrays}
                    for name, values in arrays.items():
                        clouds[key + "_" + name] = values.copy()
        state_dim = config["hidden_size"] * (2 if model_kind == "lstm" else 1)
        delta2, delta16 = behavior["2"]["delta"], behavior["16"]["delta"]
        metrics = {"schema_version": 1, "scientific_acceptance": False, "kind": "engineering_vector_memory_pilot",
            "model": model_kind, "task": task, "seed": seed, "selected_step": selected_step, "selection_validation_loss": best_loss,
            "behavior": behavior, "delta2": delta2, "delta16": delta16, "difference_in_differences": delta16 - delta2,
            "contrast_interpretation": "positive DID means relative deterioration, not necessarily positive periodic harm atR16",
            "baseline_competence": competence, "baseline_competence_passed": all(x["passed"] for x in competence.values()),
            "chance_error": 15 / 16 if task == "categorical" else 1 / 3,
            "clouds": cloud_results, "state": {"dimension": state_dim, "persistent_fp32_bytes_per_sequence": 4 * state_dim,
                "components": ["hidden", "cell"] if model_kind == "lstm" else ["hidden"], "measurement_phase": "before terminal query"},
            "parameters": {"total": sum(p.numel() for p in model.parameters()), "core_plus_head": sum(p.numel() for p in list(model.core.parameters()) + list(model.head.parameters()))},
            "training_episodes": steps * config["batch_size"], "runtime_seconds": time.monotonic() - started,
            "limitations": config["scope"]["limitations"], "statistical_unit": "one trained seed; no across-seed inference in this artifact"}
        atomic_npz(output / "paired_errors.npz", raw)
        atomic_npz(output / "cue_clouds.npz", clouds)
        atomic_json(output / "metrics.json", metrics)
        atomic_json(output / "status.json", {"state": "completed", "scientific_acceptance": False, "selected_step": selected_step, "updated_at": _now()})
        _manifest(output)
        return metrics
    except Exception as error:
        atomic_json(output / "status.json", {"state": "failed", "error_type": type(error).__name__, "error": str(error), "updated_at": _now()})
        _manifest(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new run directory; refuses overwrite")
    parser.add_argument("--model", required=True, choices=["gru", "lstm"])
    parser.add_argument("--task", required=True, choices=["categorical", "continuous"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, help="reduce step budget; recorded as an engineering override")
    args = parser.parse_args()
    metrics = run(json.loads(args.config.read_text()), args.output, args.model, args.task, args.seed,
                  args.device, args.max_steps, args.config)
    print(json.dumps({"output": str(args.output), "selected_step": metrics["selected_step"],
        "baseline_competence": metrics["baseline_competence_passed"], "DID": metrics["difference_in_differences"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
