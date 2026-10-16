"""Production adapter binding Figure 10 lifecycle to the pinned graph backend.

This module contains no native-model import or checkpoint/test-data interface.
The guarded factory in ``graph_reconstruction`` remains the only constructor
for the actual Torch/PyG backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import graph_reconstruction as gr
from .tgrab_figure10_fit import (
    FitPolicy,
    FitProtocolError,
    OptimizerAttemptError,
    run_figure10_fit,
)


@dataclass(frozen=True)
class SeededComponents:
    models: Mapping[str, Any]
    optimizer: Any
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class SeedApplicationReceipt:
    """Callback attestation of applied model RNG state; not an external receipt."""

    model_seed: int
    streams: tuple[str, ...]
    state_sha256: str


_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class FitConstructionReceipt:
    """In-process evidence that this adapter's objects came from its factory."""

    model_seed: int
    models: Mapping[str, Any]
    optimizer: Any
    parameter_names: tuple[str, ...]
    parameter_ids: tuple[int, ...]
    seed_state_sha256: str
    seed_streams: tuple[str, ...]
    _token: Any


def _ordered_named_parameters(models: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    if set(models) != {"node_emb", "link_pred"}:
        raise FitProtocolError("expected exactly node_emb and link_pred model components")
    ordered = []
    for component in ("node_emb", "link_pred"):
        model = models[component]
        if not callable(getattr(model, "named_parameters", None)):
            raise FitProtocolError(f"{component} must expose named_parameters()")
        local = list(model.named_parameters())
        if any(type(name) is not str or not name for name, _ in local):
            raise FitProtocolError(f"{component} has an invalid parameter name")
        if len({name for name, _ in local}) != len(local):
            raise FitProtocolError(f"{component} has duplicate parameter names")
        ordered.extend((f"{component}.{name}", parameter)
                       for name, parameter in sorted(local, key=lambda item: item[0]))
    identities = [id(parameter) for _, parameter in ordered]
    if len(set(identities)) != len(identities):
        raise FitProtocolError("parameters alias across names or model components")
    return tuple(ordered)


def _optimizer_parameter_ids(optimizer: Any) -> tuple[int, ...]:
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, (list, tuple)):
        raise FitProtocolError("optimizer must expose ordered param_groups")
    params = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), (list, tuple)):
            raise FitProtocolError("optimizer parameter group is malformed")
        params.extend(group["params"])
    identities = tuple(id(parameter) for parameter in params)
    if len(set(identities)) != len(identities):
        raise FitProtocolError("optimizer contains aliased or repeated parameters")
    return identities


def make_seeded_components(*, model_seed: int,
                           seed_all: Callable[[int], None],
                           model_factory: Callable[[], Mapping[str, Any]],
                           optimizer_factory: Callable[[Mapping[str, Any]], Any]) -> SeededComponents:
    """Seed first, create fresh components/optimizer, verify canonical ordering.

    ``graph_reconstruction.fresh_objects`` owns the seed -> model -> optimizer
    call order. The optimizer must carry parameters in node-embedding name order
    followed by link-head name order, lexical within each component.
    """
    if type(model_seed) is not int or model_seed != 0:
        raise FitProtocolError("the frozen Figure 10 model seed is exactly 0")
    seed_receipts: list[SeedApplicationReceipt] = []

    def apply_and_witness(seed: int) -> None:
        receipt = seed_all(seed)
        if (not isinstance(receipt, SeedApplicationReceipt)
                or receipt.model_seed != model_seed
                or set(receipt.streams) != {"python", "numpy", "torch_cpu"}
                or len(receipt.streams) != 3
                or len(receipt.state_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in receipt.state_sha256)):
            raise FitProtocolError("seed callback must attest Python/NumPy/Torch CPU RNG state")
        seed_receipts.append(receipt)
    models, optimizer = gr.fresh_objects(
        model_seed, seed_all=apply_and_witness, model_factory=model_factory,
        optimizer_factory=optimizer_factory)
    if len(seed_receipts) != 1:
        raise FitProtocolError("seed callback must run exactly once before model construction")
    ordered = _ordered_named_parameters(models)
    if not ordered:
        raise FitProtocolError("fresh model has no trainable parameters")
    expected = tuple(id(parameter) for _, parameter in ordered)
    if _optimizer_parameter_ids(optimizer) != expected:
        raise FitProtocolError("optimizer parameter order differs from canonical named order")
    return SeededComponents(models=models, optimizer=optimizer,
                            parameter_names=tuple(name for name, _ in ordered))


def build_figure10_fit_adapter(*, model_seed: int,
                               seed_all: Callable[[int], None],
                               model_factory: Callable[[], Mapping[str, Any]],
                               optimizer_factory: Callable[[Mapping[str, Any]], Any],
                               backend_factory: Callable[[Mapping[str, Any], Any], Any],
                               training_graphs: Sequence[gr.Snapshot],
                               validation_graphs: Sequence[gr.Snapshot],
                               negative_seed: int = 1,
                               period: int = 64, block: int = 32,
                               policy: FitPolicy = FitPolicy()) -> "Figure10NativeFitAdapter":
    """Construct a seed-0 model/optimizer pair and its identity-bound adapter."""
    if type(policy) is not FitPolicy or policy != FitPolicy():
        raise FitProtocolError("adapter construction requires the exact frozen FitPolicy")
    seed_receipts: list[SeedApplicationReceipt] = []

    def witnessed_seed(seed: int):
        receipt = seed_all(seed)
        seed_receipts.append(receipt)
        return receipt

    components = make_seeded_components(
        model_seed=model_seed, seed_all=witnessed_seed,
        model_factory=model_factory, optimizer_factory=optimizer_factory)
    backend = backend_factory(components.models, components.optimizer)
    receipt = FitConstructionReceipt(
        model_seed=model_seed, models=components.models,
        optimizer=components.optimizer,
        parameter_names=components.parameter_names,
        parameter_ids=tuple(id(parameter) for _, parameter in
                            _ordered_named_parameters(components.models)),
        seed_state_sha256=seed_receipts[0].state_sha256,
        seed_streams=seed_receipts[0].streams,
        _token=_FACTORY_TOKEN)
    return Figure10NativeFitAdapter(
        backend, training_graphs, validation_graphs,
        construction_receipt=receipt, negative_seed=negative_seed,
        period=period, block=block, policy=policy)


class Figure10NativeFitAdapter:
    """Implements the lifecycle backend API over NativeGCLSTMBackend.

    The caller supplies only the train and validation snapshots. No interface
    accepts or opens another split. One adapter instance is single-use so its
    independent negative RNG and row receipts cannot be silently reused.
    """

    def __init__(self, backend: gr.NativeGCLSTMBackend,
                 training_graphs: Sequence[gr.Snapshot],
                 validation_graphs: Sequence[gr.Snapshot], *,
                 construction_receipt: FitConstructionReceipt,
                 negative_seed: int = 1,
                 period: int = 64, block: int = 32,
                 policy: FitPolicy = FitPolicy()):
        if type(policy) is not FitPolicy or policy != FitPolicy():
            raise FitProtocolError("adapter construction requires the exact frozen FitPolicy")
        if not isinstance(backend, gr.NativeGCLSTMBackend):
            raise FitProtocolError("adapter requires the guarded NativeGCLSTMBackend")
        self._verify_construction_receipt(backend, construction_receipt)
        if type(negative_seed) is not int or negative_seed != 1:
            raise FitProtocolError("the frozen negative-sampling seed is 1")
        if type(period) is not int or period != 64 or type(block) is not int or block != 32:
            raise FitProtocolError("the frozen primary condition is period 64 with block length 32")
        gr_snapshot_type = gr.Snapshot
        if any(not isinstance(graph, gr_snapshot_type) for graph in (*training_graphs, *validation_graphs)):
            raise FitProtocolError("training and validation entries must be pinned Snapshot objects")
        if len(training_graphs) != 2560 or [g.timestamp for g in training_graphs] != list(range(2560)):
            raise FitProtocolError("training sequence must be timestamps 0..2559")
        if len(validation_graphs) != 256 or [g.timestamp for g in validation_graphs] != list(range(2560, 2816)):
            raise FitProtocolError("validation sequence must be timestamps 2560..2815")
        if any(graph.nodes != 100 or graph.node_ids != tuple(range(100))
               for graph in (*training_graphs, *validation_graphs)):
            raise FitProtocolError("canonical 100-node identity ordering required")
        self.backend = backend
        self.policy = policy
        self.training_graphs = tuple(training_graphs)
        self.validation_graphs = tuple(validation_graphs)
        self._graphs_by_timestamp = {g.timestamp: g for g in (*self.training_graphs,
                                                               *self.validation_graphs)}
        self.negative_rng = np.random.default_rng(negative_seed)
        self.period, self.block = period, block
        self.validation_rows: list[dict[str, Any]] = []
        self._used = False
        self._fit_active = False
        self._fit_thread_id: int | None = None
        self._phase: str | None = None
        self._evaluation_epoch = 0
        self._step_lock = threading.Lock()

    @staticmethod
    def _verify_construction_receipt(backend, receipt: FitConstructionReceipt) -> None:
        if not isinstance(receipt, FitConstructionReceipt) or receipt._token is not _FACTORY_TOKEN:
            raise FitProtocolError("adapter requires an authentic fit-factory receipt")
        if receipt.model_seed != 0:
            raise FitProtocolError("construction receipt must bind model seed 0")
        if set(receipt.seed_streams) != {"python", "numpy", "torch_cpu"}:
            raise FitProtocolError("construction receipt lacks required seeded RNG streams")
        if (len(receipt.seed_state_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in receipt.seed_state_sha256)):
            raise FitProtocolError("construction receipt has invalid RNG-state digest")
        if getattr(backend, "models", None) is not receipt.models:
            raise FitProtocolError("backend model components differ from factory receipt")
        if getattr(backend, "optimizer", None) is not receipt.optimizer:
            raise FitProtocolError("backend optimizer differs from factory receipt")
        ordered = _ordered_named_parameters(receipt.models)
        names = tuple(name for name, _ in ordered)
        ids = tuple(id(parameter) for _, parameter in ordered)
        if names != receipt.parameter_names or ids != receipt.parameter_ids:
            raise FitProtocolError("model parameter identity/order changed after construction")
        if _optimizer_parameter_ids(receipt.optimizer) != ids:
            raise FitProtocolError("backend optimizer parameter order differs from canonical model order")

    @contextmanager
    def training_mode(self):
        if not self._in_fit_thread() or self._phase is not None:
            raise FitProtocolError("training mode can only be entered by the active fit lifecycle")
        self._phase = "train"
        try:
            with self.backend.training_mode():
                yield
        finally:
            self._phase = None

    @contextmanager
    def evaluation_mode(self):
        if not self._in_fit_thread() or self._phase is not None:
            raise FitProtocolError("evaluation mode is controlled by the complete-epoch fit lifecycle")
        self._evaluation_epoch += 1
        self._phase = "eval"
        try:
            with self.backend.evaluation_mode():
                yield
        finally:
            self._phase = None

    def _in_fit_thread(self) -> bool:
        return (self._fit_active and self._fit_thread_id == threading.get_ident())

    def reset(self):
        return self.backend.reset()

    def detach(self, state):
        return self.backend.detach(state)

    def consume(self, state, graph):
        return self.backend.consume(state, graph)

    def _tracked_loss_update(self, state, target, negatives) -> float:
        optimizer = getattr(self.backend, "optimizer", None)
        if optimizer is None or not callable(getattr(optimizer, "step", None)):
            raise OptimizerAttemptError("native optimizer step is unavailable",
                                        step_status="not_started")
        if not self._step_lock.acquire(blocking=False):
            raise OptimizerAttemptError("concurrent optimizer update refused",
                                        step_status="unknown")
        sentinel = object()
        prior_instance_value = getattr(optimizer, "__dict__", {}).get("step", sentinel)
        original_step = optimizer.step
        witness = {"entered": 0, "returned": 0}
        installed = False

        def observed_status() -> str:
            if witness["entered"] == 0:
                return "not_started"
            if witness["entered"] == witness["returned"] == 1:
                return "completed"
            return "unknown"

        def tracked_step(*args, **kwargs):
            witness["entered"] += 1
            try:
                result = original_step(*args, **kwargs)
            except BaseException:
                raise
            witness["returned"] += 1
            return result

        try:
            optimizer.step = tracked_step
            installed = getattr(optimizer, "step") is tracked_step
            if not installed:
                raise RuntimeError("could not install optimizer step observer")
            loss = self.backend.loss_and_update(state, target, negatives)
            if witness["entered"] != 1 or witness["returned"] != 1:
                status = observed_status()
                raise OptimizerAttemptError("backend update did not perform exactly one returned optimizer.step",
                                            step_status=status)
            return float(loss)
        except BaseException as exc:
            status = observed_status()
            if isinstance(exc, OptimizerAttemptError) and exc.step_status == status:
                raise
            raise OptimizerAttemptError("native update failed; optimizer mutation status: " + status,
                                        step_status=status) from exc
        finally:
            try:
                if installed:
                    if prior_instance_value is sentinel:
                        del optimizer.step
                    else:
                        optimizer.step = prior_instance_value
            finally:
                self._step_lock.release()

    def train_update(self, state, target) -> float:
        if not self._in_fit_thread() or self._phase != "train":
            raise FitProtocolError("optimizer updates are private to the active training epoch")
        if not 1 <= target.timestamp < 2560 or state.next_target_t != target.timestamp:
            raise OptimizerAttemptError("training target/state chronology mismatch",
                                        step_status="not_started")
        try:
            negatives = gr.sample_negatives(target, self.negative_rng)
        except BaseException as exc:
            raise OptimizerAttemptError("negative sampling failed before optimizer step",
                                        step_status="not_started") from exc
        return self._tracked_loss_update(state, target, negatives)

    def score_f1(self, state, target) -> float:
        if not self._in_fit_thread() or self._phase != "eval" or self._evaluation_epoch < 1:
            raise FitProtocolError("validation scoring requires an active fit epoch")
        if target.timestamp not in range(2560, 2816):
            raise FitProtocolError("score_f1 accepts validation timestamps only")
        if state.next_target_t != target.timestamp:
            raise FitProtocolError("validation state must precede its target")
        if self._graphs_by_timestamp.get(target.timestamp) is not target:
            raise FitProtocolError("validation target is not the pinned adapter input")
        previous = self._graphs_by_timestamp.get(target.timestamp - 1)
        if previous is None:
            raise FitProtocolError("validation predecessor is unavailable")
        pairs = gr.ordered_offdiagonal_pairs(target.nodes)
        probabilities = self.backend.score_pairs(state, pairs)
        row = gr.snapshot_metrics(target, probabilities, previous,
                                  period=self.period, block=self.block)
        self.validation_rows.append({"epoch": self._evaluation_epoch, **row})
        return float(row["f1"])

    def fit(self, *, guard: Callable[[], None] = lambda: None,
            policy: FitPolicy | None = None,
            on_attempt: Callable[[dict[str, Any]], None] = lambda _: None,
            on_update: Callable[[dict], None] = lambda _: None,
            on_selection: Callable[[Any], None] = lambda _: None):
        """Run with the exact frozen production policy and adapter inputs."""
        policy = self.policy if policy is None else policy
        if type(policy) is not FitPolicy or policy != FitPolicy():
            raise FitProtocolError("production adapter fit requires the exact frozen FitPolicy")
        if self._used:
            raise FitProtocolError("fit adapter instances are single-use")
        self._used = True
        self._fit_active = True
        self._fit_thread_id = threading.get_ident()
        try:
            result = run_figure10_fit(backend=self,
                                      training_graphs=self.training_graphs,
                                      validation_graphs=self.validation_graphs,
                                      guard=guard, policy=policy,
                                      on_attempt=on_attempt,
                                      on_update=on_update,
                                      on_selection=on_selection)
        finally:
            self._fit_active = False
            self._fit_thread_id = None
            self._phase = None
        return result, tuple(self.validation_rows)
