"""Prospective, CPU-only prefix acquisition; production requires a source freeze.

This module never submits jobs or invokes a model service. Its explicit fixture
API reduces execution budgets for infrastructure tests only. It cannot establish
periodicity-induced harm, select a checkpoint, or rescue a frozen v2 fit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .competence import R0, NATIVE_PHASE, _batch_hash, evaluate_competence, make_competence_batch
from .diagnostics import paired_batch_with_zero
from .native_state import native_phase_capture
from .pilot import VectorMemoryModel, atomic_checkpoint, atomic_json, atomic_npz

PROTOCOL_ID = "prefix-acquisition-discovery-v3"
CONDITIONS = ("fresh_seed0", "immediate_step500_transfer_reset_optimizer")
CELLS = tuple((model, task, condition) for model in ("gru", "lstm")
              for task in ("categorical", "continuous") for condition in CONDITIONS)
WARM_SOURCE = "1add2b6aba948714673f45f1b7854eff6212833d"


class DeadlineReached(RuntimeError):
    """A censored execution, never a fabricated coordinate-gate failure."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _relative(project_root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("a project-relative path without parent traversal is required")
    result = project_root / path
    cursor = result
    while cursor != project_root:
        if cursor.is_symlink():
            raise ValueError("symlinked evidence is rejected")
        cursor = cursor.parent
    return result


def validate_config(config):
    expected = {
        "protocol_id": PROTOCOL_ID, "models": ["gru", "lstm"],
        "tasks": ["categorical", "continuous"], "conditions": list(CONDITIONS),
        "seed": 0, "learning_rate": 0.001, "device": "cpu", "cpu_threads": 1,
        "dimensions": {"hidden_size": 128, "input_bottleneck": 64, "batch_size": 32},
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "gradient_clip": 1.0,
                      "new_optimizer_for_each_condition": True},
        "training_stream_seed": 2026101301, "validation_sampling_seed": 20001,
        "validation_episodes_per_coordinate_per_arm": 128, "required_coordinates": list(range(8)),
        "categorical_accuracy_gate": 0.95, "continuous_population_normalized_mse_gate": 0.05,
        "population_variance": 1 / 3, "evaluation_every_updates": 500,
        "updates_per_fit": 5000, "maximum_fits": 8,
        "maximum_total_optimizer_updates": 40000, "maximum_total_wall_seconds": 1200,
        "confirmation": False, "cluster_submissions": 0, "provider_calls": 0,
        "precision": "float32; deterministic_algorithms; CUDA/cuDNN TF32off; matmul_precision highest",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("configuration differs from the prospective v3 scientific/budget contract")
    for key in ("cpu_threads", "training_stream_seed", "validation_sampling_seed",
                "validation_episodes_per_coordinate_per_arm", "evaluation_every_updates", "updates_per_fit",
                "maximum_fits", "maximum_total_optimizer_updates", "maximum_total_wall_seconds"):
        _positive_int(config[key], key)
    if any(type(config[key]) is not int for key in ("seed", "cluster_submissions", "provider_calls")) or config["confirmation"] is not False:
        raise ValueError("integer zero counters and an explicit nonconfirmation flag are required")
    pins = config.get("warm_start_checkpoints", [])
    if len(pins) != 4 or {(p.get("model"), p.get("task")) for p in pins} != {
        (model, task) for model in ("gru", "lstm") for task in ("categorical", "continuous")}:
        raise ValueError("exactly four distinct warm-start checkpoint pins are required")
    for pin in pins:
        if pin.get("source_commit") != WARM_SOURCE or pin.get("step") != 500:
            raise ValueError("warm starts must be the recorded v2 step500 source")
        digest = pin.get("checkpoint_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("full lowercase checkpoint SHA256 required")
        _relative(Path("/validation-root"), pin["checkpoint_relative_to_project"])


def verify_freeze(config_path, receipt_path, project_root):
    """Verify a coordinator freeze and independent review before making outputs."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    if not config_path.is_relative_to(project_root):
        raise ValueError("configuration must be within the committed project")
    receipt = json.loads(Path(receipt_path).read_text())
    required = {"kind": "prefix_acquisition_v3_freeze", "protocol_id": PROTOCOL_ID,
                "local_execution_authorized": True, "scientific_acceptance": False}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("explicit local source/config freeze receipt required")
    commit = receipt.get("source_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("full frozen source commit required")
    relative = str(config_path.relative_to(project_root))
    if receipt.get("config_relative_to_project") != relative or receipt.get("config_sha256") != _sha(config_path):
        raise ValueError("configuration does not match the pre-execution freeze")
    review = receipt.get("independent_review", {})
    review_path = _relative(project_root, review.get("receipt_relative_to_project", ""))
    if review.get("accepted") is not True or not review_path.is_file() or _sha(review_path) != review.get("receipt_sha256"):
        raise ValueError("hash-bound independent source/chronology review required")
    if review.get("implementation_sha256") != _sha(__file__) or review.get("config_sha256") != _sha(config_path):
        raise ValueError("review must bind the executing implementation and exact configuration")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=project_root, text=True).strip()
    if actual != commit or dirty:
        raise ValueError("source must be clean and committed at the frozen revision")
    committed_config = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=project_root)
    if hashlib.sha256(committed_config).hexdigest() != receipt["config_sha256"]:
        raise ValueError("the exact configuration must belong to the frozen commit")
    module = Path(__file__).resolve()
    if not module.is_relative_to(project_root):
        raise ValueError("executing module is outside the frozen project source")
    module_relative = str(module.relative_to(project_root))
    if subprocess.check_output(["git", "show", f"{commit}:{module_relative}"], cwd=project_root) != module.read_bytes():
        raise ValueError("executing implementation differs from the frozen commit")
    return {"source_commit": commit, "config_sha256": receipt["config_sha256"],
            "freeze_receipt_sha256": _sha(receipt_path), "independent_review_sha256": review["receipt_sha256"]}


@dataclass(frozen=True)
class FixtureBudget:
    """Explicit infrastructure fixture; only update/fit/time budgets decrease."""
    updates_per_fit: int = 2
    evaluate_every: int = 1
    fit_indices: tuple[int, ...] = (0, 1)
    wall_seconds: float = 60.0

    def validate(self):
        if _positive_int(self.updates_per_fit, "fixture updates") > 5000:
            raise ValueError("fixture update budget cannot expand production")
        if _positive_int(self.evaluate_every, "fixture evaluation interval") > 500 or self.updates_per_fit % self.evaluate_every:
            raise ValueError("fixture evaluations must include the fixed endpoint")
        if not self.fit_indices or len(self.fit_indices) > 8 or len(set(self.fit_indices)) != len(self.fit_indices) or any(
            isinstance(index, bool) or not isinstance(index, int) or index not in range(8) for index in self.fit_indices):
            raise ValueError("fixture registry must be a distinct subset of declared fits")
        if isinstance(self.wall_seconds, bool) or not np.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 1200:
            raise ValueError("fixture wall budget must reduce the production ceiling")
        if self.updates_per_fit == 5000 and len(self.fit_indices) == 8 and self.wall_seconds == 1200:
            raise ValueError("fixture execution requires an actual update, fit or walltime budget reduction")


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineReached("overall wall deadline; incomplete/censored execution")


def batch_identity(batch):
    digest = hashlib.sha256()
    for name in ("inputs", "targets", "queries"):
        array = np.ascontiguousarray(getattr(batch, name))
        digest.update(json.dumps({"name": name, "dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def target_oracle(inputs, targets, queries, task):
    """Direct addressed-payload oracle, independent of renderer/evaluation."""
    x, y, q = np.asarray(inputs), np.asarray(targets), np.asarray(queries)
    if task not in ("categorical", "continuous") or x.ndim != 3 or x.shape[1:] != (17, 38) or y.ndim != 1 or q.ndim != 1 or len(x) != len(y) or len(q) != len(y):
        raise ValueError("full-prefix R0 requires [batch,17,38] and matching targets/selectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or q.dtype.kind not in "iu" or np.any((q < 0) | (q >= 8)):
        raise ValueError("finite R0 data and required-coordinate selectors are mandatory")
    types = x[:, :, :5]
    if not np.all((types == 0) | (types == 1)) or not np.all(types.sum(2) == 1) or not np.all(types[:, :16].argmax(2) == 0) or not np.all(types[:, -1].argmax(1) == 4):
        raise ValueError("R0 has exactly sixteen WRITE observations and one terminal QUERY")
    all_addresses = x[:, :, 5:21]
    if not np.all((all_addresses == 0) | (all_addresses == 1)) or not np.all(all_addresses.sum(2) == 1) or np.any(x[:, -1, 21:] != 0):
        raise ValueError("WRITE/QUERY addresses must be one-hot and QUERY cannot carry a payload")
    if not np.array_equal(x[:, -1, 5:21].argmax(1), q):
        raise ValueError("terminal QUERY does not match its selector")
    addresses = x[:, :16, 5:21]
    if not np.array_equal(addresses.argmax(2), np.broadcast_to(np.arange(16), (len(x), 16))):
        raise ValueError("R0 WRITE addresses must be the sixteen ordered coordinates")
    payload = x[np.arange(len(x)), q]
    categorical = x[:, :16, 21:37]
    continuous = x[:, :16, 37]
    if task == "categorical":
        if not np.all((categorical == 0) | (categorical == 1)) or not np.all(categorical.sum(2) == 1) or np.any(continuous != 0):
            raise ValueError("categorical WRITE payloads require sixteen-class one-hot values only")
    elif np.any(categorical != 0) or np.any(np.abs(continuous) > 1):
        raise ValueError("continuous WRITE payloads require real values in[-1,1] only")
    oracle = payload[:, 21:37].argmax(1) if task == "categorical" else payload[:, 37]
    if not np.array_equal(oracle, y):
        raise ValueError("target differs from the addressed required WRITE payload")
    return len(y)


def _rolling_start(task, updates):
    return hashlib.sha256(json.dumps({"protocol": PROTOCOL_ID, "task": task,
        "seed": 2026101301, "updates": updates, "batch_size": 32}, sort_keys=True).encode())


def prehash_stream(task, updates, deadline):
    """Stream every batch once before fitting; retain hashes, never all arrays."""
    rng = np.random.default_rng(2026101301)
    rolling = _rolling_start(task, updates)
    batch_hashes, prefix_hashes = [], []
    for step in range(1, updates + 1):
        _check_deadline(deadline)
        batch = paired_batch_with_zero(rng, 16, 0, task)
        target_oracle(batch.inputs, batch.targets, np.tile(batch.queries, 2), task)
        digest = batch_identity(batch)
        rolling.update(step.to_bytes(8, "big") + bytes.fromhex(digest))
        batch_hashes.append(digest); prefix_hashes.append(rolling.hexdigest())
    return {"schema_version": 1, "task": task, "sampling_seed": 2026101301,
            "updates": updates, "batch_size": 32, "batch_sha256": batch_hashes,
            "prefix_sha256": prefix_hashes, "stream_sha256": rolling.hexdigest(),
            "persisted_before_first_fit": True, "arrays_retained": False}


def verify_stream_batch(batch, step, expected, rolling):
    digest = batch_identity(batch)
    if digest != expected["batch_sha256"][step - 1]:
        raise ValueError("regenerated training batch differs from the prefit manifest")
    rolling.update(step.to_bytes(8, "big") + bytes.fromhex(digest))
    if rolling.hexdigest() != expected["prefix_sha256"][step - 1]:
        raise ValueError("regenerated rolling training identity differs from the prefit manifest")


def _warm_model(model, model_kind, task, config, project_root):
    pin = next(p for p in config["warm_start_checkpoints"] if p["model"] == model_kind and p["task"] == task)
    path = _relative(project_root, pin["checkpoint_relative_to_project"])
    if not path.is_file() or _sha(path) != pin["checkpoint_sha256"]:
        raise ValueError("warm step500 checkpoint hash mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    prior = checkpoint.get("config", {})
    if checkpoint.get("step") != 500 or checkpoint.get("stage") != 1 or any(
        prior.get(key) != value for key, value in {"model": model_kind, "task": task,
        "seed": 0, "learning_rate": 0.001, "dimensions": config["dimensions"]}.items()):
        raise ValueError("warm checkpoint metadata differs from the pinned immediate step500 condition")
    if not isinstance(checkpoint.get("model_state_dict"), dict) or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in checkpoint["model_state_dict"].values()):
        raise ValueError("serialized warm weights must already be finite FP32 tensors")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("warm weights must be finite native FP32")
    return {"checkpoint_sha256": pin["checkpoint_sha256"], "source_commit": pin["source_commit"],
            "inherited_updates": 500, "inherited_optimizer_state": False}


def _weight_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def _evaluate(model, batch, checkpoint_id, model_kind, task, deadline):
    model.eval()
    arrays = {name: np.asarray(getattr(batch, name)) for name in ("inputs", "targets", "queries", "pair_ids")}
    target_oracle(arrays["inputs"], arrays["targets"], arrays["queries"], task)
    pieces, states, chronology = [], {phase: [] for phase in ("post_write", "pre_query", "post_query")}, []
    for start in range(0, len(batch.targets), 64):
        _check_deadline(deadline)
        capture = native_phase_capture(model, torch.from_numpy(batch.inputs[start:start + 64].copy()), 16)
        pieces.append(capture["uninterrupted_logits"].numpy())
        chronology.append(capture["chronology"])
        for phase in states:
            states[phase].append(capture["phase_states"][phase].numpy())
    logits = np.concatenate(pieces)
    if not np.isfinite(logits).all():
        raise FloatingPointError("nonfinite validation logits")
    prediction = logits.argmax(1) if task == "categorical" else logits[:, 0]
    record = evaluate_competence(batch, prediction, trained_seed=0, checkpoint_id=checkpoint_id, measurement_phase=NATIVE_PHASE)
    record.update(model=model_kind, scope="v3 R0 acquisition discovery; no distractor body")
    raw = {**arrays, "native_logits": logits, "predictions": prediction,
           **{phase: np.concatenate(values) for phase, values in states.items()}}
    return record, raw, {"passed": all(chunk["passed"] for chunk in chronology), "chunks": chronology}


def _hash_outputs(output):
    atomic_json(output / "manifest.json", {"schema_version": 1, "sha256": {
        str(path.relative_to(output)): _sha(path) for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".tmp")}})


def _engine(config, output, project_root, identity, fixture):
    validate_config(config)
    if fixture is not None:
        fixture.validate()
    updates = 5000 if fixture is None else fixture.updates_per_fit
    interval = 500 if fixture is None else fixture.evaluate_every
    registry = CELLS if fixture is None else tuple(CELLS[i] for i in fixture.fit_indices)
    wall_limit = 1200 if fixture is None else fixture.wall_seconds
    started = time.monotonic(); deadline = started + wall_limit
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    context = {"protocol_id": PROTOCOL_ID, "scientific_acceptance": False,
               "fixture_only": fixture is not None, "infrastructure_only": fixture is not None,
               "source_identity": identity, "updates_per_fit": updates, "evaluate_every": interval,
               "planned_fits": len(registry), "maximum_total_updates": updates * len(registry),
               "maximum_wall_seconds": wall_limit, "device": "cpu", "cpu_threads": 1}
    atomic_json(output / "config.json", {**config, "execution": context})
    status = {**context, "state": "running", "started_at_utc": _now(), "fits": []}
    status["fit_registry"] = [{"model": model, "task": task, "condition": condition}
                              for model, task, condition in registry]
    atomic_json(output / "status.json", status)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    atomic_json(output / "precision.json", {"torch": str(torch.__version__), "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "threads": torch.get_num_threads(), "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "parameter_dtype": "torch.float32",
        "fixture_only": fixture is not None, "scientific_acceptance": False})
    source_root = Path(__file__).resolve().parents[2]
    source_files = ["uv.lock", "pyproject.toml", ".python-version"] + [
        f"src/representation_collapse/{name}.py" for name in
        ("prefix_acquisition", "pilot", "vector_task", "diagnostics", "competence", "native_state")]
    atomic_json(output / "source-provenance.json", {**context,
        "source_sha256": {name: _sha(source_root / name) for name in source_files},
        "canonical_configuration_sha256": _digest(config)})
    current_path, history, fit_status, current_model = None, [], None, None
    try:
        manifests, streams = {}, {}
        manifest_dir = output / "prefit_manifests"; manifest_dir.mkdir()
        for task in sorted({cell[1] for cell in registry}):
            _check_deadline(deadline)
            batch = make_competence_batch(R0, task, 20001)
            target_oracle(batch.inputs, batch.targets, batch.queries, task)
            manifests[task] = batch
            atomic_npz(manifest_dir / f"validation-{task}.npz", {name: getattr(batch, name) for name in ("inputs", "targets", "queries", "pair_ids")})
            streams[task] = prehash_stream(task, updates, deadline)
            atomic_json(manifest_dir / f"train-stream-{task}.json", streams[task])
        # Warm identities are checked before any fit starts, including fresh fits.
        warm_pins = []
        for model_kind, task in sorted({(cell[0], cell[1]) for cell in registry}):
            _check_deadline(deadline)
            with torch.device("cpu"):
                test_model = VectorMemoryModel(model_kind, task, 128, 64)
            warm_pins.append(_warm_model(test_model, model_kind, task, config, project_root))
        atomic_json(output / "prefit-freeze.json", {"validation_manifest_sha256": {task: _batch_hash(batch) for task, batch in manifests.items()},
            "validation_file_sha256": {task: _sha(manifest_dir / f"validation-{task}.npz") for task in manifests},
            "training_manifest_file_sha256": {task: _sha(manifest_dir / f"train-stream-{task}.json") for task in streams},
            "training_stream_sha256": {task: stream["stream_sha256"] for task, stream in streams.items()},
            "warm_start_identities": warm_pins, "frozen_before_first_optimization": True, **context})
        for model_kind, task, condition in registry:
            _check_deadline(deadline)
            prefit = json.loads((output / "prefit-freeze.json").read_text())
            if _sha(manifest_dir / f"train-stream-{task}.json") != prefit["training_manifest_file_sha256"][task]:
                raise ValueError("stored training manifest changed after the prefit freeze")
            name = f"{model_kind}-{task}-{condition}"; current_path = output / name
            current_path.mkdir()
            history = []
            current_model = None
            fit_status = {"model": model_kind, "task": task, "condition": condition,
                          "state": "running", "updates_completed": 0, "started_at_utc": _now(), **context}
            status["fits"].append(fit_status)
            atomic_json(output / "status.json", status)
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            with torch.device("cpu"):
                model = VectorMemoryModel(model_kind, task, 128, 64).to(device="cpu", dtype=torch.float32)
            current_model = model
            warm = _warm_model(model, model_kind, task, config, project_root) if condition == CONDITIONS[1] else None
            # Never load optimizer state. Both conditions create a new AdamW here.
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            if optimizer.state:
                raise ValueError("new optimizer unexpectedly contains inherited state")
            initial_hash = _weight_hash(model)
            atomic_checkpoint(current_path / "initial-checkpoint.pt", {"model_state_dict": model.state_dict(),
                "step": 0, "condition": condition, "warm_start": warm, "fixture_only": fixture is not None,
                "scientific_acceptance": False})
            atomic_json(current_path / "provenance.json", {**context, "model": model_kind, "task": task,
                "condition": condition, "seed": 0, "learning_rate": 0.001, "warm_start": warm,
                "initial_weight_sha256": initial_hash, "initial_optimizer_state_entries": len(optimizer.state),
                "validation_manifest_sha256": _batch_hash(manifests[task]), "validation_sampling_seed": 20001,
                "training_stream_sha256": streams[task]["stream_sha256"], "training_sampling_seed": 2026101301})
            rng = np.random.default_rng(2026101301); rolling = _rolling_start(task, updates)
            first_pass = None
            for step in range(1, updates + 1):
                _check_deadline(deadline)
                batch = paired_batch_with_zero(rng, 16, 0, task)
                verify_stream_batch(batch, step, streams[task], rolling)
                target_oracle(batch.inputs, batch.targets, np.tile(batch.queries, 2), task)
                inputs = torch.from_numpy(batch.inputs.copy())
                targets = torch.as_tensor(batch.targets, dtype=torch.long if task == "categorical" else torch.float32, device="cpu")
                model.train(); optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets) if task == "categorical" else F.mse_loss(logits[:, 0], targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                # Count a successfully returned optimizer step even if its
                # resulting weights fail the subsequent numerical check.
                row = {"step": step, "train_loss": float(loss.detach()), "gradient_norm_before_clip": float(gradient_norm),
                       "training_batch_sha256": streams[task]["batch_sha256"][step - 1], "training_prefix_sha256": rolling.hexdigest()}
                history.append(row); fit_status["updates_completed"] = step
                parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
                row["post_update_parameters_finite"] = parameters_finite
                if not parameters_finite:
                    fit_status["nonfinite_parameters_after_completed_update"] = True
                    raise FloatingPointError("nonfinite parameters after completed optimizer update")
                if step % interval == 0:
                    _check_deadline(deadline)
                    checkpoint_id = f"checkpoint-step-{step:06d}.pt"
                    atomic_checkpoint(current_path / checkpoint_id, {"model_state_dict": model.state_dict(), "step": step,
                        "condition": condition, "config": config, "fixture_only": fixture is not None, "scientific_acceptance": False})
                    if _batch_hash(manifests[task]) != read_manifest_hash(output, task):
                        raise ValueError("validation manifest changed after the prefit freeze")
                    if _sha(manifest_dir / f"validation-{task}.npz") != prefit["validation_file_sha256"][task]:
                        raise ValueError("stored validation arrays changed after the prefit freeze")
                    record, raw, chronology = _evaluate(model, manifests[task], checkpoint_id, model_kind, task, deadline)
                    atomic_npz(current_path / f"validation-step-{step:06d}.npz", raw)
                    row.update(validation=record, native_chronology=chronology)
                    if record["coordinate_gate_passed"] and first_pass is None:
                        first_pass = step
                    atomic_json(current_path / "history.json", history)
                    atomic_json(output / "status.json", status)
                    _check_deadline(deadline)
            if rolling.hexdigest() != streams[task]["stream_sha256"]:
                raise ValueError("final train stream differs from the prefit identity")
            endpoint = history[-1]["validation"]
            fit_status.update(state="completed", completed_at_utc=_now(), first_passing_validation_step=first_pass,
                endpoint_step=updates, endpoint_coordinate_gate_passed=endpoint["coordinate_gate_passed"],
                selected_checkpoint=None, final_train_stream_sha256=rolling.hexdigest())
            atomic_json(current_path / "metrics.json", {**fit_status, "endpoint": endpoint,
                "first_pass_is_descriptive_only": True, "initial_weight_sha256": initial_hash,
                "endpoint_weight_sha256": _weight_hash(model), "history_entries": len(history)})
            atomic_json(current_path / "status.json", fit_status)
            _hash_outputs(current_path)
            atomic_json(output / "status.json", status)
            current_path, fit_status, current_model = None, None, None
        status["state"] = "completed"
    except DeadlineReached:
        status["state"] = "incomplete_censored_overall_deadline"
        if fit_status is not None:
            fit_status.update(state="incomplete_censored_overall_deadline", endpoint_coordinate_gate_passed=None,
                              selected_checkpoint=None, completed_at_utc=_now())
    except Exception as error:
        status.update(state="failed_execution_identity_or_numerics", error_type=type(error).__name__, error=str(error))
        if fit_status is not None:
            fit_status.update(state=status["state"], error_type=type(error).__name__, endpoint_coordinate_gate_passed=None,
                              selected_checkpoint=None)
        raise
    finally:
        if current_path is not None:
            # A deadline between evaluations must retain the actual current
            # weights, rather than imply the last scheduled checkpoint is current.
            if current_model is not None:
                realized_step = fit_status["updates_completed"]
                censored_id = f"censored-current-step-{realized_step:06d}.pt"
                parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in current_model.parameters())
                atomic_checkpoint(current_path / censored_id, {
                    "model_state_dict": current_model.state_dict(), "step": realized_step,
                    "condition": fit_status["condition"], "config": config,
                    "execution_state": fit_status["state"], "censored": True,
                    "endpoint_competence_known": False, "selected_checkpoint": None,
                    "parameters_finite": parameters_finite, "usable_for_evaluation": False,
                    "diagnostic_preservation_only": True,
                    "optimizer_state_saved": False, "resume_authorized": False,
                    "fixture_only": fixture is not None, "scientific_acceptance": False})
                fit_status.update(censored_current_checkpoint_id=censored_id,
                    censored_current_checkpoint_sha256=_sha(current_path / censored_id),
                    checkpoint_realized_step=realized_step, preserved_checkpoint_parameters_finite=parameters_finite,
                    preserved_checkpoint_usable_for_evaluation=False, endpoint_coordinate_gate_passed=None)
            atomic_json(current_path / "history.json", history)
            atomic_json(current_path / "status.json", fit_status)
            _hash_outputs(current_path)
        status.update(finished_at_utc=_now(), elapsed_seconds=time.monotonic() - started,
                      total_optimizer_updates=sum(fit["updates_completed"] for fit in status["fits"]))
        status["completed_fits"] = sum(fit["state"] == "completed" for fit in status["fits"])
        status["unstarted_fits"] = status["fit_registry"][len(status["fits"]):]
        atomic_json(output / "status.json", status)
        _hash_outputs(output)
    return status


def read_manifest_hash(output, task):
    return json.loads((Path(output) / "prefit-freeze.json").read_text())["validation_manifest_sha256"][task]


def run(config_path, output, freeze_receipt):
    """Production entrypoint; no overrides and no fixture implicit fallback."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text()); validate_config(config)
    if torch.__version__.split("+")[0] != "2.6.0":
        raise ValueError("production requires the frozen Torch2.6.0 environment")
    project_root = Path(__file__).resolve().parents[2]
    identity = verify_freeze(config_path, freeze_receipt, project_root)
    return _engine(config, output, project_root, identity, None)


def run_fixture(config, output, project_root, budget: FixtureBudget):
    """Only infrastructure fixtures; does not grant production execution."""
    if not isinstance(budget, FixtureBudget):
        raise ValueError("an explicit FixtureBudget is required")
    return _engine(config, output, Path(project_root).resolve(), {"fixture_only": True,
        "source_module_sha256": _sha(__file__), "scientific_acceptance": False}, budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output, args.freeze_receipt)
    print(json.dumps({"state": result["state"], "fits": len(result["fits"]),
                      "total_optimizer_updates": result["total_optimizer_updates"], "scientific_acceptance": False}))


if __name__ == "__main__":
    main()
