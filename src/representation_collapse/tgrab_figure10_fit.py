"""Complete-epoch lifecycle for the corrected T-GRAB Figure 10 fit.

This module coordinates a supplied numerical backend. It does not load data,
construct models, serialize checkpoints, or expose a held-out-split interface.
The caller owns immutable inputs, backend parity, admission, and persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, ContextManager, Protocol, Sequence


TRAIN_START = 0
TRAIN_STOP = 2560
VALIDATION_START = 2560
VALIDATION_STOP = 2816
UPDATES_PER_EPOCH = TRAIN_STOP - TRAIN_START - 1


class FitProtocolError(ValueError):
    """Invalid lifecycle inputs or backend chronology."""


class FitInterrupted(RuntimeError):
    """A guard stopped the fit at a safe boundary."""

    def __init__(self, reason: str, *, epoch: int, attempted_updates: int,
                 completed_updates: int, optimizer_steps_confirmed: int,
                 epoch_complete: bool):
        super().__init__(reason)
        self.reason = reason
        self.epoch = epoch
        self.attempted_updates = attempted_updates
        self.completed_updates = completed_updates
        self.optimizer_steps_confirmed = optimizer_steps_confirmed
        self.epoch_complete = epoch_complete


class OptimizerAttemptError(RuntimeError):
    """Backend update exception with an explicit step-status witness.

    ``unknown`` is the safe default: an exception after entering optimizer
    code is not evidence that no parameter update occurred. The engine never
    retries an attempt.
    """

    STATUSES = {"not_started", "completed", "unknown"}

    def __init__(self, message: str, *, step_status: str = "unknown"):
        if step_status not in self.STATUSES:
            raise ValueError("invalid optimizer step status")
        super().__init__(message)
        self.step_status = step_status


class FitBackend(Protocol):
    """Minimal interface expected from a pinned, independently checked backend."""

    def training_mode(self) -> ContextManager[Any]: ...
    def evaluation_mode(self) -> ContextManager[Any]: ...
    def reset(self) -> Any: ...
    def detach(self, state: Any) -> Any: ...
    def consume(self, state: Any, graph: Any) -> Any: ...
    def train_update(self, state: Any, target: Any) -> float: ...
    def score_f1(self, state: Any, target: Any) -> float: ...


@dataclass(frozen=True)
class FitPolicy:
    """Frozen protocol defaults; smaller values are useful only for spy tests."""

    max_complete_epochs: int = 200
    loss_patience: int = 100
    relative_loss_tolerance: float = 1e-6
    expected_updates_per_epoch: int = UPDATES_PER_EPOCH

    def validate(self) -> None:
        if (type(self.max_complete_epochs) is not int
                or not 1 <= self.max_complete_epochs <= 200):
            raise FitProtocolError("max_complete_epochs must be an integer in [1, 200]")
        if type(self.loss_patience) is not int or not 1 <= self.loss_patience <= 100:
            raise FitProtocolError("loss_patience must be an integer in [1, 100]")
        if (type(self.expected_updates_per_epoch) is not int
                or self.expected_updates_per_epoch != UPDATES_PER_EPOCH):
            raise FitProtocolError("the Figure 10 protocol requires exactly 2559 updates per epoch")
        if (isinstance(self.relative_loss_tolerance, bool)
                or not isinstance(self.relative_loss_tolerance, (int, float))
                or self.relative_loss_tolerance != 1e-6):
            raise FitProtocolError("relative_loss_tolerance must equal the frozen value 1e-6")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    attempted_updates: int
    completed_updates: int
    optimizer_steps_confirmed: int
    mean_training_loss: float
    validation_mean_f1: float
    loss_improved: bool
    validation_selected: bool
    nonimproving_loss_epochs: int


@dataclass(frozen=True)
class FitResult:
    stop_reason: str
    epochs_completed: int
    attempted_updates: int
    completed_updates: int
    optimizer_steps_confirmed: int
    best_loss_epoch: int
    best_training_loss: float
    selected_validation_epoch: int
    selected_validation_mean_f1: float
    history: tuple[EpochRecord, ...] = field(default_factory=tuple)


def _timestamp(graph: Any) -> int:
    value = getattr(graph, "timestamp", None)
    if type(value) is not int:
        raise FitProtocolError("each graph must expose an integer timestamp")
    return value


def _check_sequence(graphs: Sequence[Any], start: int, stop: int, name: str) -> None:
    if len(graphs) != stop - start:
        raise FitProtocolError(f"{name} must contain exactly {stop - start} snapshots")
    for offset, graph in enumerate(graphs):
        expected = start + offset
        if _timestamp(graph) != expected:
            raise FitProtocolError(f"{name} timestamp mismatch at offset {offset}: expected {expected}")


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FitProtocolError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FitProtocolError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise FitProtocolError(f"{name} must be a finite nonnegative number")
    return result


def _guard(guard: Callable[[], None], *, epoch: int, attempted: int,
           completed: int, optimizer_steps_confirmed: int,
           epoch_complete: bool) -> None:
    try:
        guard()
    except BaseException as exc:
        raise FitInterrupted(str(exc) or type(exc).__name__, epoch=epoch,
                             attempted_updates=attempted,
                             completed_updates=completed,
                             optimizer_steps_confirmed=optimizer_steps_confirmed,
                             epoch_complete=epoch_complete) from exc


def _relative_loss_improved(current: float, best: float, tolerance: float) -> bool:
    return (best - current) / max(abs(best), 1e-12) > tolerance


def run_figure10_fit(*, backend: FitBackend, training_graphs: Sequence[Any],
                     validation_graphs: Sequence[Any],
                     guard: Callable[[], None] = lambda: None,
                     policy: FitPolicy = FitPolicy(),
                     on_attempt: Callable[[dict[str, Any]], None] = lambda _: None,
                     on_update: Callable[[dict[str, Any]], None] = lambda _: None,
                     on_selection: Callable[[EpochRecord], None] = lambda _: None,
                     on_epoch_complete: Callable[[EpochRecord, Any], None] = lambda *_: None) -> FitResult:
    """Production entry point; accepts only the exact frozen Figure 10 policy."""
    if type(policy) is not FitPolicy or policy != FitPolicy():
        raise FitProtocolError("production Figure 10 fit requires the exact frozen FitPolicy")
    return _run_figure10_fit_lifecycle(
        backend=backend, training_graphs=training_graphs,
        validation_graphs=validation_graphs, guard=guard, policy=policy,
        on_attempt=on_attempt, on_update=on_update, on_selection=on_selection,
        on_epoch_complete=on_epoch_complete)


def _run_figure10_fit_lifecycle(*, backend: FitBackend,
                     training_graphs: Sequence[Any], validation_graphs: Sequence[Any],
                     guard: Callable[[], None] = lambda: None,
                     policy: FitPolicy = FitPolicy(),
                     on_attempt: Callable[[dict[str, Any]], None] = lambda _: None,
                     on_update: Callable[[dict[str, Any]], None] = lambda _: None,
                     on_selection: Callable[[EpochRecord], None] = lambda _: None,
                     on_epoch_complete: Callable[[EpochRecord, Any], None] = lambda *_: None) -> FitResult:
    """Private lifecycle core; reduced policies are for synthetic tests only.

    The function intentionally accepts no held-out graph sequence and contains
    no loader/opener for one. At epoch boundaries it resets recurrent state,
    performs 2,559 predecessor-driven updates over targets 1..2559, then resets
    again and replays the 2,560 training snapshots in eval mode before scoring
    validation targets 2560..2815 score-before-consume. A partial epoch never
    updates selection or loss patience.

    `on_attempt` runs immediately before each `train_update` invocation with a
    unique sequence and chronology identity. Its successful return is only a
    barrier: durable charging is guaranteed only if the caller's callback
    commits it durably before returning. The default callback is a no-op and
    provides no durable accounting. If the callback raises, the backend update
    is not entered. `on_update` is a post-attempt status hook, not a retry
    mechanism. Attempt increments before `train_update`; if that call raises,
    its step status is `unknown` unless the exception explicitly supplies
    `step_status`. If the call succeeds, the returned update counts as
    completed before callbacks or subsequent stop/deadline checks.

    `on_epoch_complete` receives the immutable record and final state after
    the 256th validation target has been consumed, only after the complete
    epoch guard succeeds. It runs before the selection callback and does not
    imply durable persistence unless its caller implements that guarantee.
    """
    policy.validate()
    _check_sequence(training_graphs, TRAIN_START, TRAIN_STOP, "training_graphs")
    _check_sequence(validation_graphs, VALIDATION_START, VALIDATION_STOP,
                    "validation_graphs")

    total_attempted = 0
    total_completed = 0
    total_steps_confirmed = 0
    history: list[EpochRecord] = []
    best_loss = math.inf
    best_loss_epoch = 0
    stale_loss_epochs = 0
    best_validation = -math.inf
    selected_epoch = 0
    completed_epochs = 0

    for epoch in range(1, policy.max_complete_epochs + 1):
        epoch_attempted = 0
        epoch_completed = 0
        epoch_steps_confirmed = 0
        epoch_losses: list[float] = []
        state: Any = None
        _guard(guard, epoch=epoch, attempted=total_attempted,
               completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
               epoch_complete=False)
        with backend.training_mode():
            _guard(guard, epoch=epoch, attempted=total_attempted,
                   completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
                   epoch_complete=False)
            state = backend.reset()
            for target_t in range(1, TRAIN_STOP):
                _guard(guard, epoch=epoch, attempted=total_attempted,
                       completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
                       epoch_complete=False)
                predecessor = training_graphs[target_t - 1]
                target = training_graphs[target_t]
                state = backend.consume(backend.detach(state), predecessor)
                next_attempt = total_attempted + 1
                on_attempt({"kind": "tgrab-figure10-fit-attempt-v1",
                            "sequence": next_attempt, "epoch": epoch,
                            "training_order_index": target_t - TRAIN_START - 1,
                            "target_timestamp": target_t,
                            "consumed_timestamp": target_t - 1})
                total_attempted += 1
                epoch_attempted += 1
                try:
                    raw_loss = backend.train_update(state, target)
                except BaseException as exc:
                    # The attempt is recorded, but not silently classified as
                    # a no-op: optimizer code may have stepped before raising.
                    status = getattr(exc, "step_status", "unknown")
                    if status not in OptimizerAttemptError.STATUSES:
                        status = "unknown"
                    if status == "completed":
                        total_steps_confirmed += 1
                        epoch_steps_confirmed += 1
                    on_update({"epoch": epoch, "target_timestamp": target_t,
                               "consumed_timestamp": target_t - 1,
                               "attempt": total_attempted,
                               "attempted_updates": total_attempted,
                               "completed_updates": total_completed,
                               "epoch_attempted_updates": epoch_attempted,
                               "epoch_completed_updates": epoch_completed,
                               "optimizer_steps_confirmed": total_steps_confirmed,
                               "completed": False, "optimizer_step_status": status,
                               "error_type": type(exc).__name__})
                    raise
                total_completed += 1
                total_steps_confirmed += 1
                epoch_completed += 1
                epoch_steps_confirmed += 1
                try:
                    loss = _finite_nonnegative(raw_loss, "training loss")
                except BaseException as exc:
                    # The backend returned, so its update is completed even
                    # when the returned diagnostic loss is malformed.
                    on_update({"epoch": epoch, "target_timestamp": target_t,
                               "consumed_timestamp": target_t - 1,
                               "attempt": total_attempted,
                               "attempted_updates": total_attempted,
                               "completed_updates": total_completed,
                               "epoch_attempted_updates": epoch_attempted,
                               "epoch_completed_updates": epoch_completed,
                               "optimizer_steps_confirmed": total_steps_confirmed,
                               "completed": True, "optimizer_step_status": "completed",
                               "error_type": type(exc).__name__})
                    raise
                epoch_losses.append(loss)
                on_update({"epoch": epoch, "target_timestamp": target_t,
                           "consumed_timestamp": target_t - 1,
                           "attempt": total_attempted,
                           "attempted_updates": total_attempted,
                           "completed_updates": total_completed,
                           "epoch_attempted_updates": epoch_attempted,
                           "epoch_completed_updates": epoch_completed,
                           "optimizer_steps_confirmed": total_steps_confirmed,
                           "completed": True, "optimizer_step_status": "completed",
                           "loss": loss})

        if epoch_attempted != UPDATES_PER_EPOCH or epoch_completed != UPDATES_PER_EPOCH:
            raise FitProtocolError("an epoch may proceed to validation only after all 2559 updates complete")
        _guard(guard, epoch=epoch, attempted=total_attempted,
               completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
               epoch_complete=False)
        mean_loss = math.fsum(epoch_losses) / UPDATES_PER_EPOCH
        if best_loss == math.inf or _relative_loss_improved(
                mean_loss, best_loss, policy.relative_loss_tolerance):
            best_loss = mean_loss
            best_loss_epoch = epoch
            stale_loss_epochs = 0
            loss_improved = True
        else:
            stale_loss_epochs += 1
            loss_improved = False

        f1_values: list[float] = []
        with backend.evaluation_mode():
            _guard(guard, epoch=epoch, attempted=total_attempted,
                   completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
                   epoch_complete=False)
            # Recreate the complete training prefix under frozen epoch weights;
            # never continue from the training-mode state or cached state.
            state = backend.reset()
            for graph in training_graphs:
                _guard(guard, epoch=epoch, attempted=total_attempted,
                       completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
                       epoch_complete=False)
                state = backend.consume(state, graph)
            for target in validation_graphs:
                _guard(guard, epoch=epoch, attempted=total_attempted,
                       completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
                       epoch_complete=False)
                f1 = _finite_nonnegative(backend.score_f1(state, target), "validation F1")
                if f1 > 1:
                    raise FitProtocolError("validation F1 must lie in [0,1]")
                f1_values.append(f1)
                state = backend.consume(state, target)

        if len(f1_values) != VALIDATION_STOP - VALIDATION_START:
            raise FitProtocolError("validation must score all 256 snapshots")
        _guard(guard, epoch=epoch, attempted=total_attempted,
               completed=total_completed, optimizer_steps_confirmed=total_steps_confirmed,
               epoch_complete=True)
        mean_f1 = math.fsum(f1_values) / len(f1_values)
        selected = mean_f1 > best_validation  # strict: earliest epoch wins ties
        if selected:
            best_validation = mean_f1
            selected_epoch = epoch
        completed_epochs = epoch
        record = EpochRecord(epoch=epoch, attempted_updates=epoch_attempted,
                             completed_updates=epoch_completed,
                             optimizer_steps_confirmed=epoch_steps_confirmed,
                             mean_training_loss=mean_loss,
                             validation_mean_f1=mean_f1,
                             loss_improved=loss_improved,
                             validation_selected=selected,
                             nonimproving_loss_epochs=stale_loss_epochs)
        history.append(record)
        # `state` is s[2816]: validation target 2815 has been consumed under
        # the frozen epoch-final weights. This hook runs only after all 256
        # rows and the complete-epoch guard succeed, before selection callback.
        on_epoch_complete(record, state)
        if selected:
            on_selection(record)
        if stale_loss_epochs >= policy.loss_patience:
            stop_reason = "training_loss_patience"
            break
    else:
        stop_reason = "complete_epoch_cap"

    return FitResult(stop_reason=stop_reason,
                     epochs_completed=completed_epochs,
                     attempted_updates=total_attempted,
                     completed_updates=total_completed,
                     optimizer_steps_confirmed=total_steps_confirmed,
                     best_loss_epoch=best_loss_epoch,
                     best_training_loss=best_loss,
                     selected_validation_epoch=selected_epoch,
                     selected_validation_mean_f1=best_validation,
                     history=tuple(history))
