from __future__ import annotations

import contextlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse.tgrab_figure10_fit import (  # noqa: E402
    FitInterrupted,
    FitPolicy,
    OptimizerAttemptError,
    _run_figure10_fit_lifecycle,
    run_figure10_fit,
)


class Graph:
    def __init__(self, timestamp: int):
        self.timestamp = timestamp


class State:
    def __init__(self, next_timestamp: int):
        self.next_timestamp = next_timestamp


class SpyBackend:
    """Chronology witness only; does not execute a model or optimizer."""

    def __init__(self, losses=None, validation_scores=None, *, exception_at=None,
                 exception_step_status="unknown"):
        self.losses = list(losses or [1.0])
        self.validation_scores = list(validation_scores or [0.5])
        self.exception_at = exception_at
        self.exception_step_status = exception_step_status
        self.mode = None
        self.epoch = 0
        self.resets = []
        self.train_consumed = []
        self.eval_consumed = []
        self.updates = []
        self.scored = []
        self.optimizer_steps = 0

    @contextlib.contextmanager
    def training_mode(self):
        previous = self.mode
        self.mode = "train"
        self.epoch += 1
        try:
            yield
        finally:
            self.mode = previous

    @contextlib.contextmanager
    def evaluation_mode(self):
        previous = self.mode
        self.mode = "eval"
        try:
            yield
        finally:
            self.mode = previous

    def reset(self):
        self.resets.append(self.mode)
        return State(0)

    def detach(self, state):
        return state

    def consume(self, state, graph):
        if graph.timestamp != state.next_timestamp:
            raise AssertionError((self.mode, state.next_timestamp, graph.timestamp))
        if self.mode == "train":
            self.train_consumed.append(graph.timestamp)
        elif self.mode == "eval":
            self.eval_consumed.append(graph.timestamp)
        else:
            raise AssertionError("consume outside declared backend mode")
        return State(graph.timestamp + 1)

    def train_update(self, state, target):
        if self.mode != "train" or state.next_timestamp != target.timestamp:
            raise AssertionError("training must score the next target from its predecessor state")
        self.updates.append((self.epoch, target.timestamp, state.next_timestamp))
        if target.timestamp == self.exception_at:
            if self.exception_step_status == "completed":
                self.optimizer_steps += 1
            raise OptimizerAttemptError("injected update failure",
                                        step_status=self.exception_step_status)
        self.optimizer_steps += 1
        return self.losses[min(self.epoch - 1, len(self.losses) - 1)]

    def score_f1(self, state, target):
        if self.mode != "eval" or state.next_timestamp != target.timestamp:
            raise AssertionError("validation must score before consuming target")
        self.scored.append((self.epoch, target.timestamp, state.next_timestamp))
        return self.validation_scores[min(self.epoch - 1, len(self.validation_scores) - 1)]


def sequences():
    return ([Graph(t) for t in range(2560)],
            [Graph(t) for t in range(2560, 2816)])


class Figure10FitLifecycleTests(unittest.TestCase):
    def test_attempt_callback_precedes_each_update_with_sequence_identity(self):
        train, val = sequences()
        backend = SpyBackend()
        observations = []

        def before_update(event):
            observations.append((event.copy(), len(backend.updates)))

        _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
            validation_graphs=val, on_attempt=before_update,
            policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(len(observations), 2559)
        for index, (event, updates_seen) in enumerate(observations, start=1):
            self.assertEqual(updates_seen, index - 1)
            self.assertEqual(event, {
                "kind": "tgrab-figure10-fit-attempt-v1",
                "sequence": index,
                "epoch": 1,
                "training_order_index": index - 1,
                "target_timestamp": index,
                "consumed_timestamp": index - 1,
            })

    def test_attempt_callback_failure_prevents_corresponding_update(self):
        train, val = sequences()
        backend = SpyBackend()
        events = []

        def fail_before_second(event):
            events.append(event)
            if event["sequence"] == 2:
                raise RuntimeError("journal commit failed")

        with self.assertRaisesRegex(RuntimeError, "journal commit failed"):
            _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
                validation_graphs=val, on_attempt=fail_before_second,
                policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual([event["sequence"] for event in events], [1, 2])
        self.assertEqual([target for _, target, _ in backend.updates], [1])

    def test_frozen_policy_rejects_limit_or_tolerance_drift(self):
        invalid = (
            FitPolicy(max_complete_epochs=201),
            FitPolicy(loss_patience=101),
            FitPolicy(relative_loss_tolerance=1e-5),
            FitPolicy(relative_loss_tolerance=True),
        )
        for policy in invalid:
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                policy.validate()

    def test_public_entry_rejects_shortened_policy(self):
        with self.assertRaisesRegex(ValueError, "exact frozen FitPolicy"):
            run_figure10_fit(backend=SpyBackend(), training_graphs=(),
                validation_graphs=(), policy=FitPolicy(max_complete_epochs=1))

    def test_invalid_returned_loss_preserves_completed_optimizer_accounting(self):
        train, val = sequences()
        for invalid_loss in (float("nan"), -0.1, object()):
            with self.subTest(loss=type(invalid_loss).__name__):
                backend = SpyBackend(losses=[invalid_loss])
                receipts = []
                with self.assertRaises(ValueError):
                    _run_figure10_fit_lifecycle(backend=backend,
                        training_graphs=train, validation_graphs=val,
                        on_update=receipts.append,
                        policy=FitPolicy(max_complete_epochs=1))
                self.assertEqual(len(backend.updates), 1)
                self.assertEqual(backend.optimizer_steps, 1)
                self.assertEqual(len(receipts), 1)
                self.assertTrue(receipts[0]["completed"])
                self.assertEqual(receipts[0]["optimizer_step_status"], "completed")
                self.assertEqual(receipts[0]["attempted_updates"], 1)
                self.assertEqual(receipts[0]["completed_updates"], 1)
                self.assertEqual(receipts[0]["epoch_completed_updates"], 1)
                self.assertEqual(receipts[0]["optimizer_steps_confirmed"], 1)

    def test_one_epoch_has_exact_updates_reset_replay_and_score_before_consume(self):
        train, val = sequences()
        backend = SpyBackend()
        result = _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
                                  validation_graphs=val,
                                  policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(result.stop_reason, "complete_epoch_cap")
        self.assertEqual(result.epochs_completed, 1)
        self.assertEqual(result.attempted_updates, 2559)
        self.assertEqual(result.completed_updates, 2559)
        self.assertEqual(result.optimizer_steps_confirmed, 2559)
        self.assertEqual(backend.resets, ["train", "eval"])
        self.assertEqual(backend.train_consumed, list(range(2559)))
        self.assertEqual([t for _, t, _ in backend.updates], list(range(1, 2560)))
        self.assertTrue(all(state_t == target_t
                            for _, target_t, state_t in backend.updates))
        self.assertEqual(backend.eval_consumed, list(range(2816)))
        self.assertEqual([t for _, t, _ in backend.scored], list(range(2560, 2816)))
        self.assertTrue(all(state_t == target_t
                            for _, target_t, state_t in backend.scored))
        self.assertEqual(result.selected_validation_epoch, 1)
        self.assertEqual(result.selected_validation_mean_f1, 0.5)

    def test_validation_selection_is_strict_and_earliest_epoch_wins_ties(self):
        train, val = sequences()
        backend = SpyBackend(validation_scores=[0.7, 0.8, 0.8])
        selected = []
        result = _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
            validation_graphs=val, policy=FitPolicy(max_complete_epochs=3),
            on_selection=lambda record: selected.append(record.epoch))
        self.assertEqual(result.selected_validation_epoch, 2)
        self.assertAlmostEqual(result.selected_validation_mean_f1, 0.8)
        self.assertEqual(selected, [1, 2])
        self.assertFalse(result.history[2].validation_selected)

    def test_loss_patience_uses_complete_training_losses_only_and_resets_on_improvement(self):
        train, val = sequences()
        backend = SpyBackend(losses=[1.0, 0.9999995, 0.999998, 0.9999981, 0.9999982],
                             validation_scores=[0.2, 0.3, 0.4, 0.5, 0.6])
        result = _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
            validation_graphs=val,
            policy=FitPolicy(max_complete_epochs=10, loss_patience=2))
        self.assertEqual(result.stop_reason, "training_loss_patience")
        self.assertEqual(result.epochs_completed, 5)
        self.assertEqual([r.loss_improved for r in result.history],
                         [True, False, True, False, False])
        self.assertEqual([r.nonimproving_loss_epochs for r in result.history],
                         [0, 1, 0, 1, 2])
        self.assertEqual(result.selected_validation_epoch, 5)
        self.assertEqual(result.attempted_updates, 5 * 2559)

    def test_stop_guard_interrupts_without_completing_or_selecting_partial_epoch(self):
        train, val = sequences()
        backend = SpyBackend()
        records = []

        def guard():
            if len(records) >= 2:
                raise RuntimeError("STOP")

        with self.assertRaises(FitInterrupted) as caught:
            _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
                validation_graphs=val, guard=guard,
                on_update=records.append,
                policy=FitPolicy(max_complete_epochs=1))
        exc = caught.exception
        self.assertEqual(exc.reason, "STOP")
        self.assertFalse(exc.epoch_complete)
        self.assertEqual(exc.attempted_updates, 2)
        self.assertEqual(exc.completed_updates, 2)
        self.assertEqual(exc.optimizer_steps_confirmed, 2)
        self.assertEqual(len(backend.scored), 0)
        self.assertEqual(backend.resets, ["train"])

    def test_failed_attempt_is_not_retried_and_step_status_is_not_assumed(self):
        train, val = sequences()
        backend = SpyBackend(exception_at=3, exception_step_status="unknown")
        receipts = []
        with self.assertRaises(OptimizerAttemptError):
            _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
                validation_graphs=val, on_update=receipts.append,
                policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(len(backend.updates), 3)
        self.assertEqual([r["attempt"] for r in receipts], [1, 2, 3])
        self.assertEqual([r["completed"] for r in receipts], [True, True, False])
        self.assertEqual(receipts[-1]["optimizer_step_status"], "unknown")
        self.assertEqual(receipts[-1]["completed_updates"], 2)
        self.assertEqual(receipts[-1]["optimizer_steps_confirmed"], 2)

    def test_exception_with_step_witness_counts_confirmed_step_without_completed_update(self):
        train, val = sequences()
        backend = SpyBackend(exception_at=3, exception_step_status="completed")
        receipts = []
        with self.assertRaises(OptimizerAttemptError):
            _run_figure10_fit_lifecycle(backend=backend, training_graphs=train,
                validation_graphs=val, on_update=receipts.append,
                policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(receipts[-1]["completed_updates"], 2)
        self.assertEqual(receipts[-1]["optimizer_steps_confirmed"], 3)
        self.assertEqual(receipts[-1]["optimizer_step_status"], "completed")

    def test_bad_sequence_and_partial_update_count_fail_closed(self):
        train, val = sequences()
        bad_train = list(train)
        bad_train[100] = Graph(101)
        backend = SpyBackend()
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            _run_figure10_fit_lifecycle(backend=backend, training_graphs=bad_train,
                             validation_graphs=val,
                             policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(len(backend.updates), 0)
        with self.assertRaisesRegex(ValueError, "exactly 2559"):
            FitPolicy(expected_updates_per_epoch=2558).validate()


if __name__ == "__main__":
    unittest.main()
