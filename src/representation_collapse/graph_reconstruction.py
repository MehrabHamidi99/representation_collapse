"""Auditable contracts for the disabled T-GRAB reconstruction pilot.

Importing this module neither imports Torch/PyG nor constructs a model. The
native factory loads the actual pinned public wrappers and installed GC-LSTM
cell only after fail-closed preflight. NumPy/spies test the engineering
contracts; they provide no native graph parity or scientific evidence.
There is deliberately no CLI, dataset converter, checkpoint loader or launcher.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np


PUBLIC_COMMIT = "31293102b065feef7817ff86211427d7594edfdb"
TEMPORAL_COMMIT = "46abd39abec0cd8de23c0e489a617c320c3fc5e8"
CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
FROZEN_CANDIDATE_SHA256 = "dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092"
WRAPPER_SHA256 = {
    "model/gclstm.py": "86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609",
    "model/link_predictor.py": "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86",
    "model/node_emb.py": "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29",
}
SOURCE_REQUIREMENTS = {
    "torch": "2.0.0", "torch-geometric": "2.6.1",
    "torch-geometric-temporal": "0.54.0", "numpy": "1.24.4",
    "torch-cluster": "1.6.3", "torch-scatter": "2.1.2",
    "torch-sparse": "0.6.18", "torch-spline-conv": "1.2.2",
}


class ContractError(ValueError):
    """Invalid graph, state, chronology or frozen protocol."""


class PreflightError(RuntimeError):
    """Native integration is not admitted or lacks verified dependencies."""


class PilotStopped(RuntimeError):
    """A finite bound or owner stop was reached at a safe boundary."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"regular nonsymlink file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ContractError(f"nonfinite {name}")
    return value


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported conditions/settings instead of substituting defaults."""
    try:
        c = config["candidate"]
        a, d, o, e, s = (c[k] for k in ("architecture", "data", "optimization", "evaluation", "stopping"))
        expected = (
            config["source_pins"]["public_commit"] == PUBLIC_COMMIT,
            config["source_pins"]["temporal_commit"] == TEMPORAL_COMMIT,
            config["source_pins"]["cell_sha256"] == CELL_SHA256,
            c["condition"] == "primary_Pdet_2_32", c["num_models"] == 1,
            c["model_seed"] == 0, c["negative_sampling_seed"] == 1,
            c["seed_before_construction"] is True, c["fresh_model_and_optimizer"] is True,
            a["layers"] == 1, a["input_channels"] == 100,
            a["hidden_channels"] == 128, a["chebyshev_order_K"] == 2,
            a["joint_state_dimension"] == 25600, a["laplacian_normalization"] == "sym",
            d["num_nodes"] == 100, d["train_periods"] == 40,
            d["validation_periods"] == 4, d["test_periods"] == 4,
            d["split_timestamp_intervals_half_open"] == {"train": [0, 2560], "validation": [2560, 2816], "test": [2816, 3072]},
            d["shuffle"] is False, d["batch_snapshots"] == 1,
            o["learning_rate"] == 1e-4, o["betas"] == [0.9, 0.999],
            o["epsilon"] == 1e-8, o["weight_decay"] == 0, o["amsgrad"] is False,
            o["precision"] == "float32", o["amp"] is False,
            e["pairs_per_target"] == 9900, e["threshold_probability_ge"] == 0.5,
            s["hard_cap_complete_epochs"] == 200,
            s["hard_cap_optimizer_updates"] == 511800,
            s["hard_cap_entire_pilot_elapsed_seconds"] == 3600,
            s["loss_patience_nonimproving_complete_epochs"] == 100,
            s["test_metrics_affect_stopping_or_selection"] is False,
        )
    except (KeyError, TypeError) as exc:
        raise ContractError("incomplete reconstruction-v1 config") from exc
    try:
        candidate_sha = hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError) as exc:
        raise ContractError("invalid frozen candidate serialization") from exc
    if not all(expected) or candidate_sha != FROZEN_CANDIDATE_SHA256:
        raise ContractError("unsupported change to frozen reconstruction-v1 settings")


@dataclass(frozen=True)
class PinnedJSON:
    path: Path
    sha256: str

    def read(self) -> dict:
        if _sha256(self.path) != self.sha256:
            raise ContractError(f"receipt hash mismatch: {self.path}")
        value = json.loads(self.path.read_text())
        if not isinstance(value, dict):
            raise ContractError("receipt must be a JSON object")
        return value


@dataclass(frozen=True)
class Readiness:
    """Pinned independent receipts, not caller-provided readiness booleans."""
    environment: PinnedJSON | None = None
    conversion: PinnedJSON | None = None
    engineering_review: PinnedJSON | None = None
    scientific_review: PinnedJSON | None = None
    reporting: PinnedJSON | None = None
    admission: PinnedJSON | None = None


def dependency_probe() -> dict:
    """Metadata/source text only: no native package imports or model calls."""
    versions = {}
    for name in SOURCE_REQUIREMENTS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cell = None
    try:
        dist = importlib.metadata.distribution("torch-geometric-temporal")
        cell = Path(dist.locate_file("torch_geometric_temporal/nn/recurrent/gc_lstm.py"))
        cell_identity = _sha256(cell)
    except (importlib.metadata.PackageNotFoundError, OSError, ContractError):
        cell_identity = None
    return {"versions": versions, "cell_path": str(cell) if cell else None,
            "cell_sha256": cell_identity, "native_packages_imported": False}


def preflight(config_path: Path, source_root: Path, readiness: Readiness = Readiness(),
              *, probe: Callable[[], dict] = dependency_probe) -> dict:
    """Return explicit blockers without importing Torch or reading data payloads.

    A matching package version alone is insufficient: environment/conversion,
    independent reviews, reporting and fresh admission receipts are mandatory.
    Receipts must bind this exact config SHA. Launch remains disabled in v1.
    """
    config_sha = _sha256(config_path)
    config = json.loads(config_path.read_text())
    validate_frozen_config(config)
    blockers = []
    if config["launch"]["enabled"] is not True:
        blockers.append("launch_disabled_by_frozen_configuration")
    wrapper_identities = {}
    for relative, expected in WRAPPER_SHA256.items():
        try:
            wrapper_identities[relative] = _sha256(source_root / relative)
        except (OSError, ContractError):
            wrapper_identities[relative] = None
        if wrapper_identities[relative] != expected:
            blockers.append(f"pinned_wrapper_unverified:{relative}")
    dependencies = probe()
    for name, expected in SOURCE_REQUIREMENTS.items():
        if dependencies["versions"].get(name) != expected:
            blockers.append(f"dependency_mismatch:{name}:{dependencies['versions'].get(name)}:requires:{expected}")
    if dependencies["cell_sha256"] != CELL_SHA256:
        blockers.append("installed_GCLSTM_source_unverified")
    receipts = {}
    for name in ("environment", "conversion", "engineering_review", "scientific_review", "reporting", "admission"):
        pin = getattr(readiness, name)
        if pin is None:
            blockers.append(f"missing_pinned_{name}_receipt")
            continue
        try:
            receipt = pin.read()
            if receipt.get("kind") != f"graph_reconstruction_{name}" or receipt.get("config_sha256") != config_sha:
                raise ContractError("receipt kind/config binding mismatch")
            expected_status = "approved" if name == "admission" else "verified"
            if receipt.get("status") != expected_status:
                raise ContractError(f"receipt requires {expected_status} status")
            if name == "environment" and not all(receipt.get(k) for k in
                    ("lock_sha256", "python_identity", "compiled_extension_identities", "native_import_smoke_verified")):
                raise ContractError("incomplete graph environment receipt")
            if name == "conversion" and (not all(receipt.get(k) for k in
                    ("raw_sha256", "conversion_sha256", "snapshot_manifest_sha256")) or
                    receipt.get("node_ids") != list(range(100)) or receipt.get("period_steps") != 64):
                raise ContractError("incomplete primary data-conversion receipt")
            if name == "reporting" and receipt.get("mode") not in ("online_verified", "offline_verified"):
                raise ContractError("reporting must be explicitly verified online or offline")
            if name == "admission" and receipt.get("bounds") != {"epochs": 200, "updates": 511800, "seconds": 3600}:
                raise ContractError("admission bounds mismatch")
            receipts[name] = {"path": str(pin.path), "sha256": pin.sha256}
        except (OSError, ValueError, ContractError) as exc:
            blockers.append(f"invalid_{name}_receipt:{exc}")
    # The controller is implemented, but durable native checkpoint/reporting
    # integration is intentionally not disguised as a runnable experiment.
    blockers.append("native_checkpoint_and_reporting_runner_not_implemented")
    return {"ready": not blockers, "blockers": blockers, "config_sha256": config_sha,
            "public_commit": PUBLIC_COMMIT, "temporal_commit": TEMPORAL_COMMIT,
            "wrapper_sha256": wrapper_identities, "dependencies": dependencies,
            "readiness_receipts": receipts, "native_model_calls": 0, "data_payload_reads": 0}


@dataclass(frozen=True)
class StateLayout:
    layers: int
    nodes: int
    hidden: int
    node_ids: tuple[int, ...]

    def __post_init__(self):
        for key in ("layers", "nodes", "hidden"):
            _integer(getattr(self, key), key, 1)
        if not isinstance(self.node_ids, tuple) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in self.node_ids) or self.node_ids != tuple(range(self.nodes)):
            raise ContractError("canonical ascending integer node IDs required")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.layers, self.nodes, self.hidden

    @property
    def joint_dimension(self) -> int:
        return 2 * self.layers * self.nodes * self.hidden


@dataclass(frozen=True)
class CompleteState:
    hidden: Any
    cell: Any
    next_target_t: int
    node_ids: tuple[int, ...]

    def validate(self, layout: StateLayout) -> None:
        if tuple(self.hidden.shape) != layout.shape or tuple(self.cell.shape) != layout.shape:
            raise ContractError("complete hidden/cell shape mismatch")
        if self.node_ids != layout.node_ids:
            raise ContractError("state node ordering mismatch")
        _integer(self.next_target_t, "next_target_t")


def flatten_numpy_state(state: CompleteState, layout: StateLayout) -> np.ndarray:
    """Synthetic reference: component, layer, node, feature; no pooling."""
    state.validate(layout)
    if not isinstance(state.hidden, np.ndarray) or not isinstance(state.cell, np.ndarray):
        raise ContractError("NumPy fixture tensors required; no implicit native conversion")
    if not np.isfinite(state.hidden).all() or not np.isfinite(state.cell).all():
        raise ContractError("nonfinite complete state")
    return np.concatenate((state.hidden.ravel(order="C"), state.cell.ravel(order="C")))


def ordered_offdiagonal_pairs(nodes: int) -> tuple[tuple[int, int], ...]:
    nodes = _integer(nodes, "nodes", 2)
    return tuple((u, v) for u in range(nodes) for v in range(nodes) if u != v)


@dataclass(frozen=True)
class Snapshot:
    timestamp: int
    nodes: int
    positive_pairs: tuple[tuple[int, int], ...]
    node_ids: tuple[int, ...]

    def __post_init__(self):
        _integer(self.timestamp, "timestamp")
        _integer(self.nodes, "nodes", 2)
        if not isinstance(self.node_ids, tuple) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in self.node_ids) or self.node_ids != tuple(range(self.nodes)):
            raise ContractError("snapshot node ordering must be canonical")
        if not isinstance(self.positive_pairs, tuple) or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in self.positive_pairs):
            raise ContractError("immutable tuple of edge tuples required")
        for u, v in self.positive_pairs:
            _integer(u, "source"); _integer(v, "destination")
            if u >= self.nodes or v >= self.nodes or u == v:
                raise ContractError("positive pairs must be valid offdiagonal integer IDs")
        if len(set(self.positive_pairs)) != len(self.positive_pairs):
            raise ContractError("conversion must explicitly remove duplicate edges")

    def positive_mask(self) -> np.ndarray:
        mask = np.zeros((self.nodes, self.nodes), dtype=bool)
        for u, v in self.positive_pairs:
            mask[u, v] = True
        return mask


def validate_negatives(snapshot: Snapshot, negative_pairs: Sequence[tuple[int, int]]) -> None:
    if len(negative_pairs) != len(snapshot.positive_pairs):
        raise ContractError("one negative per positive required")
    full_positive = set(snapshot.positive_pairs)
    for (expected_u, _), (u, v) in zip(snapshot.positive_pairs, negative_pairs):
        _integer(u, "negative source"); _integer(v, "negative destination")
        if u != expected_u or v >= snapshot.nodes or u == v or (u, v) in full_positive:
            raise ContractError("negative source, diagonal or full-positive collision")


def sample_negatives(snapshot: Snapshot, rng: Any) -> tuple[tuple[int, int], ...]:
    """Finite same-source sampling with replacement against ALL positive edges."""
    if not snapshot.positive_pairs:
        raise ContractError("candidate training requires nonempty positive targets")
    full_positive = set(snapshot.positive_pairs)
    admissible = {}
    result = []
    for u, _ in snapshot.positive_pairs:
        if u not in admissible:
            admissible[u] = tuple(v for v in range(snapshot.nodes) if v != u and (u, v) not in full_positive)
        if not admissible[u]:
            raise ContractError(f"no admissible negative destination for source {u}")
        index = _integer(rng.integers(0, len(admissible[u])), "RNG index")
        if index >= len(admissible[u]):
            raise ContractError("RNG index outside admissible range")
        result.append((u, admissible[u][index]))
    result = tuple(result)
    validate_negatives(snapshot, result)
    return result


def phase_record(timestamp: int, *, period: int = 64, block: int = 32) -> dict:
    timestamp = _integer(timestamp, "target timestamp", 1)
    period = _integer(period, "period", 2); block = _integer(block, "block", 1)
    if period != 2 * block:
        raise ContractError("two equal pattern blocks required")
    target = timestamp % period
    return {"target_timestamp": timestamp, "source_timestamp": timestamp - 1,
            "target_phase": target, "source_phase": (timestamp - 1) % period,
            "phase_transition_expected": target in (0, block)}


def snapshot_metrics(snapshot: Snapshot, probabilities: Sequence[float],
                     previous: Snapshot, *, period: int = 64, block: int = 32) -> dict:
    if previous.timestamp + 1 != snapshot.timestamp or previous.node_ids != snapshot.node_ids:
        raise ContractError("metric predecessor/timestamp/node ordering mismatch")
    pairs = ordered_offdiagonal_pairs(snapshot.nodes)
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(pairs),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ContractError("finite probabilities for every ordered offdiagonal pair required")
    positive = set(snapshot.positive_pairs)
    truths = np.asarray([pair in positive for pair in pairs])
    predictions = values >= 0.5
    tp = int(np.count_nonzero(truths & predictions)); fp = int(np.count_nonzero(~truths & predictions))
    fn = int(np.count_nonzero(truths & ~predictions)); tn = len(pairs) - tp - fp - fn
    return {**phase_record(snapshot.timestamp, period=period, block=block),
            "change_point": set(snapshot.positive_pairs) != set(previous.positive_pairs),
            "pairs": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / len(pairs)}


def time_mean_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ContractError("cannot average an empty evaluation sequence")
    return math.fsum(_finite(row["f1"], "F1") for row in rows) / len(rows)


@dataclass
class LossPatience:
    patience: int = 100
    relative_tolerance: float = 1e-6
    best: float | None = None
    counter: int = 0

    def observe_complete_epoch(self, loss: float) -> bool:
        loss = _finite(loss, "epoch training loss")
        if loss < 0:
            raise ContractError("BCE loss cannot be negative")
        if self.best is None or (self.best - loss) / max(abs(self.best), 1e-12) > self.relative_tolerance:
            self.best = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


@dataclass(frozen=True)
class CheckpointReference:
    epoch: int
    checkpoint_sha256: str
    recurrent_sha256: str
    link_head_sha256: str

    def __post_init__(self):
        _integer(self.epoch, "checkpoint epoch", 1)
        for value in (self.checkpoint_sha256, self.recurrent_sha256, self.link_head_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("both model components require SHA-256 identities")


@dataclass
class ValidationOnlySelection:
    selected: CheckpointReference | None = None
    best_f1: float | None = None
    last_epoch: int = 0
    finalized: bool = False

    def consider(self, reference: CheckpointReference, f1: float, *, split: str,
                 epoch_complete: bool) -> bool:
        if self.finalized or split != "validation" or epoch_complete is not True:
            raise ContractError("only complete-epoch validation may select before finalization")
        if reference.epoch <= self.last_epoch:
            raise ContractError("validation epochs must increase strictly")
        f1 = _finite(f1, "validation F1")
        if not 0 <= f1 <= 1:
            raise ContractError("validation F1 outside [0,1]")
        self.last_epoch = reference.epoch
        if self.best_f1 is None or f1 > self.best_f1:
            self.selected, self.best_f1 = reference, f1
            return True
        return False

    def finalize(self) -> CheckpointReference:
        if self.selected is None or self.finalized:
            raise ContractError("a valid selection must be finalized exactly once")
        self.finalized = True
        return self.selected


@dataclass
class PilotBudget:
    max_epochs: int = 200
    max_updates: int = 511800
    max_seconds: float = 3600.0
    clock: Callable[[], float] = time.monotonic
    owner_stop: Callable[[], bool] = lambda: False
    epochs: int = 0
    updates: int = 0
    started: float = field(init=False)

    def __post_init__(self):
        _integer(self.max_epochs, "max epochs", 1); _integer(self.max_updates, "max updates", 1)
        if _finite(self.max_seconds, "walltime") <= 0:
            raise ContractError("positive finite walltime required")
        self.started = self.clock()

    def check_boundary(self, *, training: bool = False) -> None:
        elapsed = _finite(self.clock() - self.started, "elapsed pilot seconds")
        if elapsed < 0:
            raise ContractError("monotonic pilot clock went backwards")
        if self.owner_stop():
            raise PilotStopped("owner_stop")
        if elapsed >= self.max_seconds:
            raise PilotStopped("entire_pilot_walltime_cap")
        if training and self.epochs >= self.max_epochs:
            raise PilotStopped("complete_epoch_cap")
        if training and self.updates >= self.max_updates:
            raise PilotStopped("optimizer_update_cap")


class ContractBackend(Protocol):
    """Inject a reviewed native backend or an explicitly synthetic spy."""
    def training_mode(self): ...
    def evaluation_mode(self): ...
    def reset(self) -> CompleteState: ...
    def detach(self, state: CompleteState) -> CompleteState: ...
    def consume(self, state: CompleteState, snapshot: Snapshot) -> CompleteState: ...
    def score_pairs(self, state: CompleteState, pairs: Sequence[tuple[int, int]]) -> Sequence[float]: ...
    def loss_and_update(self, state: CompleteState, target: Snapshot, negatives: Sequence[tuple[int, int]]) -> float: ...
    def load_selected(self, reference: CheckpointReference) -> None: ...


class CausalController:
    """One-step causal flow. This is not a launch/admission/reporting runner."""

    def __init__(self, backend: ContractBackend, layout: StateLayout, budget: PilotBudget,
                 *, negative_seed: int = 1, period: int = 64, block: int = 32):
        self.backend, self.layout, self.budget = backend, layout, budget
        self.negative_rng = np.random.default_rng(negative_seed)
        self.period, self.block = period, block
        self.test_attempted = False
        self.test_opened = False

    def _consume(self, state: CompleteState, graph: Snapshot) -> CompleteState:
        self.budget.check_boundary()
        state.validate(self.layout)
        if state.next_target_t != graph.timestamp or graph.node_ids != self.layout.node_ids:
            raise ContractError("graph must be the next input in canonical node order")
        result = self.backend.consume(state, graph)
        result.validate(self.layout)
        if result.next_target_t != graph.timestamp + 1:
            raise ContractError("backend did not consume exactly one input")
        return result

    def training_epoch(self, graphs: Sequence[Snapshot], *, expected_updates: int = 2559) -> float:
        """Reset each epoch; t0 conditions; loss/update precedes next consume.

        Partial epochs raise and do not increment complete epochs. The caller
        must never pass their metrics/checkpoint to validation selection.
        """
        if self.test_attempted:
            raise ContractError("training cannot continue after a test attempt")
        if len(graphs) != expected_updates + 1 or not graphs or graphs[0].timestamp != 0:
            raise ContractError("one conditioning graph plus the complete training target count required")
        self.budget.check_boundary(training=True)
        losses = []
        with self.backend.training_mode():
            state = self.backend.reset()
            for predecessor, target in zip(graphs, graphs[1:]):
                if target.timestamp != predecessor.timestamp + 1:
                    raise ContractError("training timestamp gap")
                self.budget.check_boundary(training=True)
                state = self._consume(self.backend.detach(state), predecessor)
                negatives = sample_negatives(target, self.negative_rng)
                self.budget.check_boundary(training=True)
                loss = _finite(self.backend.loss_and_update(state, target, negatives), "update loss")
                self.budget.updates += 1
                if loss < 0:
                    raise ContractError("negative BCE loss")
                losses.append(loss)
            self.budget.check_boundary()
        self.budget.epochs += 1
        return math.fsum(losses) / len(losses)

    def _evaluate(self, conditioning: Sequence[Snapshot], targets: Iterable[Snapshot]) -> list[dict]:
        if not conditioning or conditioning[0].timestamp != 0:
            raise ContractError("evaluation requires a fresh training-prefix replay beginning at zero")
        with self.backend.evaluation_mode():
            state = self.backend.reset()
            for graph in conditioning:
                state = self._consume(state, graph)
            previous = conditioning[-1]
            rows = []
            for target in targets:
                self.budget.check_boundary()
                state.validate(self.layout)
                if state.next_target_t != target.timestamp:
                    raise ContractError("evaluation target/predecessor mismatch")
                # Score API receives complete state and fixed candidate IDs,
                # never the current target graph/labels/weights.
                probabilities = self.backend.score_pairs(state, ordered_offdiagonal_pairs(self.layout.nodes))
                rows.append(snapshot_metrics(target, probabilities, previous, period=self.period, block=self.block))
                state = self._consume(state, target)
                previous = target
            self.budget.check_boundary()
        if not rows:
            raise ContractError("empty held-out evaluation")
        return rows

    def validation(self, training_prefix: Sequence[Snapshot], targets: Iterable[Snapshot]) -> list[dict]:
        if self.test_attempted:
            raise ContractError("validation cannot continue after a test attempt")
        return self._evaluate(training_prefix, targets)

    def test_once(self, selection: ValidationOnlySelection,
                  training_and_validation: Sequence[Snapshot],
                  open_test: Callable[[], Iterable[Snapshot]]) -> list[dict]:
        """Charge one terminal attempt before loading the selected checkpoint.

        Eligibility and the initial budget check do not charge an attempt.
        After charging, load failures and all later failures forbid training,
        validation and another test attempt. ``test_opened`` records invocation
        of ``open_test``, including a failed invocation; a lazy iterator may
        still have read no test payload. These are cooperative controller flags,
        not protection against deliberate mutation or direct backend calls.
        """
        if self.test_attempted or not selection.finalized or selection.selected is None:
            raise ContractError("test requires finalized validation-only selection and opens once")
        self.budget.check_boundary()
        self.test_attempted = True
        self.backend.load_selected(selection.selected)
        self.budget.check_boundary()
        self.test_opened = True  # Callback attempted; lazy payload may remain unread.
        targets = open_test()
        self.budget.check_boundary()
        return self._evaluate(training_and_validation, targets)


def fresh_objects(seed: int, *, seed_all: Callable[[int], None],
                  model_factory: Callable[[], Any], optimizer_factory: Callable[[Any], Any]) -> tuple[Any, Any]:
    """Call order is seed -> fresh model construction -> fresh optimizer."""
    seed_all(_integer(seed, "model seed"))
    model = model_factory()
    optimizer = optimizer_factory(model)
    return model, optimizer


def _load_exact_wrappers(source_root: Path):
    for name, expected in WRAPPER_SHA256.items():
        if _sha256(source_root / name) != expected:
            raise PreflightError(f"source changed before native import: {name}")
    namespace = f"_tgrab_pinned_model_{PUBLIC_COMMIT}"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str((source_root / "model").resolve())]
        sys.modules[namespace] = package
    recurrent = importlib.import_module(f"{namespace}.gclstm")
    head = importlib.import_module(f"{namespace}.link_predictor")
    for module, relative in ((recurrent, "model/gclstm.py"), (head, "model/link_predictor.py"),
                             (sys.modules[f"{namespace}.node_emb"], "model/node_emb.py")):
        if Path(module.__file__).resolve() != (source_root / relative).resolve() or _sha256(Path(module.__file__)) != WRAPPER_SHA256[relative]:
            raise PreflightError("cached native wrapper path/hash mismatch")
    return recurrent, head


class NativeGCLSTMBackend:
    """Actual upstream recurrent/head modules; never an approximation.

    Only the guarded factory constructs this backend. Native numerical parity,
    gradients, extension compatibility and checkpoint I/O remain unverified.
    """

    def __init__(self, torch_module, models, optimizer, *, device: str):
        self.torch, self.models, self.optimizer = torch_module, models, optimizer
        self.device = device
        self.layout = StateLayout(1, 100, 128, tuple(range(100)))
        self.features = self.torch.eye(100, dtype=self.torch.float32, device=device)

    @contextmanager
    def _mode(self, training: bool):
        saved = [(module, module.training) for model in self.models.values() for module in model.modules()]
        for model in self.models.values():
            model.train(training)
        try:
            with self.torch.set_grad_enabled(training):
                yield
        finally:
            for module, flag in saved:
                module.training = flag

    def training_mode(self):
        return self._mode(True)

    def evaluation_mode(self):
        return self._mode(False)

    def reset(self) -> CompleteState:
        return CompleteState(self.torch.zeros(self.layout.shape, dtype=self.torch.float32, device=self.device),
                             self.torch.zeros(self.layout.shape, dtype=self.torch.float32, device=self.device),
                             0, self.layout.node_ids)

    def _validate_native(self, state: CompleteState):
        state.validate(self.layout)
        for value in (state.hidden, state.cell):
            if not isinstance(value, self.torch.Tensor) or value.dtype != self.torch.float32 or str(value.device) != self.device:
                raise ContractError("native state must have frozen float32 dtype and configured device")
            if not self.torch.isfinite(value).all():
                raise ContractError("nonfinite native hidden/cell state")

    def detach(self, state: CompleteState) -> CompleteState:
        self._validate_native(state)
        return CompleteState(state.hidden.detach(), state.cell.detach(), state.next_target_t, state.node_ids)

    def consume(self, state: CompleteState, snapshot: Snapshot) -> CompleteState:
        self._validate_native(state)
        if snapshot.node_ids != self.layout.node_ids or state.next_target_t != snapshot.timestamp:
            raise ContractError("native input/state timestamp or ordering mismatch")
        positives = set(snapshot.positive_pairs)
        if any((v, u) not in positives for u, v in positives):
            raise ContractError("native candidate requires bidirectional undirected input edges")
        edges = self.torch.tensor(snapshot.positive_pairs, dtype=self.torch.long, device=self.device).reshape(-1, 2).T.contiguous()
        weights = self.torch.ones(len(snapshot.positive_pairs), dtype=self.torch.float32, device=self.device)
        hidden, cell, z = self.models["node_emb"](self.features, edges, weights, state.hidden, state.cell)
        result = CompleteState(hidden, cell, snapshot.timestamp + 1, snapshot.node_ids)
        self._validate_native(result)
        if not self.torch.equal(z, hidden[-1]):
            raise ContractError("upstream output z must equal last-layer hidden state")
        return result

    def _probabilities(self, state: CompleteState, pairs: Sequence[tuple[int, int]]):
        self._validate_native(state)
        indices = self.torch.tensor(pairs, dtype=self.torch.long, device=self.device).reshape(-1, 2)
        return self.models["link_pred"](state.hidden[-1, indices[:, 0]], state.hidden[-1, indices[:, 1]]).reshape(-1)

    def score_pairs(self, state: CompleteState, pairs: Sequence[tuple[int, int]]) -> Sequence[float]:
        return self._probabilities(state, pairs).detach().cpu().numpy()

    def loss_and_update(self, state: CompleteState, target: Snapshot,
                        negatives: Sequence[tuple[int, int]]) -> float:
        if state.next_target_t != target.timestamp:
            raise ContractError("native loss must score current target before consumption")
        validate_negatives(target, negatives)
        if not target.positive_pairs:
            raise ContractError("empty positive training targets are outside this candidate")
        self.optimizer.zero_grad(set_to_none=True)
        positive = self._probabilities(state, target.positive_pairs)
        negative = self._probabilities(state, negatives)
        loss = self.torch.nn.functional.binary_cross_entropy(positive, self.torch.ones_like(positive))
        loss = loss + self.torch.nn.functional.binary_cross_entropy(negative, self.torch.zeros_like(negative))
        if not self.torch.isfinite(loss):
            raise ContractError("nonfinite native BCE loss")
        loss.backward()
        for model in self.models.values():
            for parameter in model.parameters():
                if parameter.grad is not None and not self.torch.isfinite(parameter.grad).all():
                    raise ContractError("nonfinite native gradient before optimizer step")
        self.optimizer.step()
        for model in self.models.values():
            if any(not self.torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ContractError("nonfinite native parameter after optimizer step")
        return float(loss.detach().cpu())

    def load_selected(self, reference: CheckpointReference) -> None:
        raise PreflightError("native checkpoint schema/hash reader is not implemented; no implicit torch.load")


def build_native_backend(config_path: Path, source_root: Path, readiness: Readiness = Readiness(),
                         *, device: str = "cpu") -> NativeGCLSTMBackend:
    """Guarded future construction; currently always blocked by runner gap.

    No caller can substitute a dependency probe or alternative recurrent cell.
    A future integration must remove the explicit runner-gap blocker only
    after implementing/reviewing durable checkpoints and reporting.
    """
    report = preflight(config_path, source_root, readiness)
    if not report["ready"]:
        raise PreflightError("native construction blocked: " + "; ".join(report["blockers"]))
    torch = importlib.import_module("torch")
    recurrent, head = _load_exact_wrappers(source_root)
    installed_cell = Path(importlib.import_module("inspect").getfile(recurrent.GCLSTM))
    if _sha256(installed_cell) != CELL_SHA256:
        raise PreflightError("imported GCLSTM differs from pinned actual cell")
    config = json.loads(config_path.read_text())
    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def model_factory():
        return {"node_emb": recurrent.MultiLayerGCLSTM(1, recurrent.GCLSTMParam(100, 128, 2)).to(device=device, dtype=torch.float32),
                "link_pred": head.LinkPredictor(128, num_layers=2, p_dropout=0.2).to(device=device, dtype=torch.float32)}
    def optimizer_factory(models):
        named = [(f"{component}.{name}", parameter) for component in ("node_emb", "link_pred")
                 for name, parameter in sorted(models[component].named_parameters())]
        if len({id(parameter) for _, parameter in named}) != len(named):
            raise ContractError("optimizer parameters unexpectedly aliased")
        return torch.optim.Adam([parameter for _, parameter in named], lr=1e-4,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
    models, optimizer = fresh_objects(config["candidate"]["model_seed"], seed_all=seed_all,
                                      model_factory=model_factory, optimizer_factory=optimizer_factory)
    return NativeGCLSTMBackend(torch, models, optimizer, device=str(torch.device(device)))
