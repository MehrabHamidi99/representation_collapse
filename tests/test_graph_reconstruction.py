"""Synthetic graph-contract fixtures/spies only; no Torch, PyG or optimizer.

These checks are not native GC-LSTM parity evidence or training experiments.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from representation_collapse.graph_reconstruction import (
    CELL_SHA256, SOURCE_REQUIREMENTS, CausalController, CheckpointReference,
    CompleteState, ContractError, LossPatience, PilotBudget, PilotStopped,
    PreflightError, Readiness, Snapshot, StateLayout, ValidationOnlySelection,
    build_native_backend, flatten_numpy_state, fresh_objects,
    ordered_offdiagonal_pairs, phase_record, preflight, sample_negatives,
    snapshot_metrics, time_mean_f1, validate_frozen_config, validate_negatives,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/graph/tgrab-reconstruction-v1.json"
SOURCE = ROOT / "runs/graph/upstream-pin-20261012/source"
A = ((0, 1), (1, 0))
B = ((0, 2), (2, 0))


def graph(t, pairs=A):
    return Snapshot(t, 3, pairs, (0, 1, 2))


def reference(epoch):
    return CheckpointReference(epoch, "a" * 64, "b" * 64, "c" * 64)


class ZeroIndexRNG:
    def integers(self, low, high):
        return 0


class SpyBackend:
    """Tagged arrays and event logs; loss/update is a spy, no optimization."""
    def __init__(self, layout):
        self.layout = layout
        self.events = []
        self.is_training = None

    @contextmanager
    def _mode(self, training):
        previous = self.is_training
        self.is_training = training
        self.events.append(("mode", training))
        try:
            yield
        finally:
            self.is_training = previous

    def training_mode(self):
        return self._mode(True)

    def evaluation_mode(self):
        return self._mode(False)

    def reset(self):
        self.events.append(("reset",))
        return CompleteState(np.zeros(self.layout.shape), np.zeros(self.layout.shape), 0, self.layout.node_ids)

    def detach(self, state):
        self.events.append(("detach", state.next_target_t))
        return CompleteState(state.hidden.copy(), state.cell.copy(), state.next_target_t, state.node_ids)

    def consume(self, state, snapshot):
        self.events.append(("consume", snapshot.timestamp, snapshot.positive_pairs))
        return CompleteState(state.hidden + snapshot.timestamp + 1, state.cell + 10 * (snapshot.timestamp + 1),
                             snapshot.timestamp + 1, state.node_ids)

    def score_pairs(self, state, pairs):
        self.events.append(("score", state.next_target_t, tuple(pairs), self.is_training))
        return np.zeros(len(pairs))

    def loss_and_update(self, state, target, negatives):
        validate_negatives(target, negatives)
        self.events.append(("loss_update_spy", target.timestamp, state.next_target_t, self.is_training))
        return 1 / (target.timestamp + 1)

    def load_selected(self, selected):
        self.events.append(("load_selected_spy", selected.epoch))


class GraphContractTests(unittest.TestCase):
    def controller(self, **budget_kwargs):
        layout = StateLayout(1, 3, 2, (0, 1, 2))
        backend = SpyBackend(layout)
        budget = PilotBudget(clock=lambda: 0, **budget_kwargs)
        return CausalController(backend, layout, budget, period=4, block=2), backend

    def test_import_has_no_native_dependency_side_effect(self):
        result = subprocess.run([sys.executable, "-c",
            "import sys; import representation_collapse.graph_reconstruction; "
            "assert not any(k == 'torch' or k.startswith('torch_geometric') for k in sys.modules)"],
            check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_config_rejects_changed_limits(self):
        config = json.loads(CONFIG.read_text())
        validate_frozen_config(config)
        for path in (("architecture", "hidden_channels"), ("stopping", "hard_cap_optimizer_updates"),
                     ("evaluation", "threshold_probability_ge")):
            candidate = deepcopy(config)
            candidate["candidate"][path[0]][path[1]] = 17
            with self.subTest(path=path), self.assertRaises(ContractError):
                validate_frozen_config(candidate)

    def test_preflight_missing_environment_is_explicit_without_native_import(self):
        def missing():
            return {"versions": {}, "cell_path": None, "cell_sha256": None}
        with patch("representation_collapse.graph_reconstruction.importlib.import_module", side_effect=AssertionError("native import forbidden")):
            report = preflight(CONFIG, SOURCE, probe=missing)
        self.assertFalse(report["ready"])
        self.assertIn("launch_disabled_by_frozen_configuration", report["blockers"])
        self.assertIn("missing_pinned_environment_receipt", report["blockers"])
        self.assertIn("missing_pinned_conversion_receipt", report["blockers"])
        self.assertIn("native_checkpoint_and_reporting_runner_not_implemented", report["blockers"])
        self.assertEqual(report["native_model_calls"], 0)

    def test_matching_dependency_metadata_alone_cannot_make_native_ready(self):
        report = preflight(CONFIG, SOURCE, Readiness(), probe=lambda: {
            "versions": dict(SOURCE_REQUIREMENTS), "cell_path": "unexecuted-fixture-path", "cell_sha256": CELL_SHA256})
        self.assertFalse(report["ready"])
        self.assertNotIn("installed_GCLSTM_source_unverified", report["blockers"])
        self.assertIn("missing_pinned_admission_receipt", report["blockers"])

    def test_native_builder_fails_before_import_or_construction(self):
        with patch("representation_collapse.graph_reconstruction.importlib.import_module", side_effect=AssertionError("native import forbidden")) as imported:
            with self.assertRaisesRegex(PreflightError, "native construction blocked"):
                build_native_backend(CONFIG, SOURCE)
            imported.assert_not_called()

    def test_preflight_missing_source_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            report = preflight(CONFIG, Path(directory), probe=lambda: {"versions": {}, "cell_sha256": None})
        self.assertIn("pinned_wrapper_unverified:model/gclstm.py", report["blockers"])

    def test_complete_state_flattening_includes_all_layers_nodes_and_cell(self):
        layout = StateLayout(2, 3, 2, (0, 1, 2))
        hidden = np.arange(12).reshape(layout.shape)
        cell = (100 + np.arange(12)).reshape(layout.shape)
        state = CompleteState(hidden, cell, 3, layout.node_ids)
        result = flatten_numpy_state(state, layout)
        np.testing.assert_array_equal(result, np.r_[np.arange(12), 100 + np.arange(12)])
        self.assertEqual(layout.joint_dimension, 24)
        self.assertEqual(StateLayout(1, 100, 128, tuple(range(100))).joint_dimension, 25600)

    def test_state_order_shape_and_nonfinite_rejected(self):
        layout = StateLayout(1, 3, 2, (0, 1, 2))
        with self.assertRaises(ContractError):
            StateLayout(1, 3, 2, (1, 0, 2))
        for hidden, cell, order in ((np.zeros((1, 2, 2)), np.zeros(layout.shape), layout.node_ids),
                                    (np.zeros(layout.shape), np.zeros(layout.shape), (2, 1, 0)),
                                    (np.full(layout.shape, np.nan), np.zeros(layout.shape), layout.node_ids)):
            with self.assertRaises(ContractError):
                flatten_numpy_state(CompleteState(hidden, cell, 0, order), layout)

    def test_positive_pairs_require_integer_offdiagonal_unique_ids(self):
        for pairs in (((0, 0),), ((0, 3),), ((True, 0),), ((0, 1), (0, 1))):
            with self.subTest(pairs=pairs), self.assertRaises(ContractError):
                graph(0, pairs)

    def test_negatives_check_full_positive_set_not_resampled_subset(self):
        snapshot = Snapshot(1, 4, ((0, 1), (0, 2), (1, 0)), (0, 1, 2, 3))
        result = sample_negatives(snapshot, ZeroIndexRNG())
        self.assertEqual(result[:2], ((0, 3), (0, 3)))  # Duplicates are permitted.
        validate_negatives(snapshot, result)
        for bad in (((0, 1), (0, 3), (1, 2)), ((0, 3), (0, 2), (1, 2)),
                    ((0, 0), (0, 3), (1, 2)), ((1, 3), (0, 3), (1, 2))):
            with self.assertRaises(ContractError):
                validate_negatives(snapshot, bad)

    def test_negative_sampling_saturated_row_and_empty_positive_stop(self):
        for snapshot in (graph(1, ((0, 1), (0, 2))), graph(1, ())):
            with self.assertRaises(ContractError):
                sample_negatives(snapshot, np.random.default_rng(1))

    def test_dedicated_negative_rng_reproducible(self):
        snapshot = Snapshot(1, 5, ((0, 1), (1, 0)), tuple(range(5)))
        left = np.random.default_rng(1); right = np.random.default_rng(1)
        values = [sample_negatives(snapshot, left) for _ in range(10)]
        np.random.seed(42); np.random.random(30)
        self.assertEqual(values, [sample_negatives(snapshot, right) for _ in range(10)])

    def test_metric_pair_count_threshold_equality_and_zero_positive(self):
        self.assertEqual(len(ordered_offdiagonal_pairs(100)), 9900)
        self.assertEqual(ordered_offdiagonal_pairs(3), ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)))
        row = snapshot_metrics(graph(1), [0.5, 0, 0.5, 0, 0, 0], graph(0), period=4, block=2)
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["f1"]), (2, 0, 0, 1))
        empty = snapshot_metrics(graph(1, ()), np.zeros(6), graph(0, ()), period=4, block=2)
        self.assertEqual((empty["precision"], empty["recall"], empty["f1"]), (0, 0, 0))

    def test_time_mean_is_not_pooled_confusion_f1(self):
        rows = [{"f1": 1, "tp": 1, "fp": 0, "fn": 0}, {"f1": 0, "tp": 0, "fp": 0, "fn": 9}]
        self.assertEqual(time_mean_f1(rows), 0.5)
        self.assertNotEqual(time_mean_f1(rows), 2 / 11)

    def test_phase_boundary_uses_target_not_source_phase(self):
        first = phase_record(2816)
        self.assertEqual((first["target_phase"], first["source_phase"]), (0, 63))
        self.assertEqual((phase_record(2848)["target_phase"], phase_record(2848)["source_phase"]), (32, 31))
        records = [phase_record(t) for t in range(2816, 3072)]
        self.assertEqual(sum(r["phase_transition_expected"] for r in records), 8)
        self.assertEqual(len(records), 256)

    def test_actual_transition_comes_from_graph_equality(self):
        row = snapshot_metrics(graph(2, B), np.zeros(6), graph(1, A), period=4, block=2)
        self.assertTrue(row["change_point"])
        same = snapshot_metrics(graph(2, A), np.zeros(6), graph(1, A), period=4, block=2)
        self.assertFalse(same["change_point"])
        self.assertTrue(same["phase_transition_expected"])

    def test_training_scores_before_target_consumption_and_resets_every_epoch(self):
        controller, backend = self.controller()
        graphs = [graph(0), graph(1), graph(2, B), graph(3, B)]
        mean = controller.training_epoch(graphs, expected_updates=3)
        self.assertAlmostEqual(mean, (1 / 2 + 1 / 3 + 1 / 4) / 3)
        events = [(e[0], e[1]) for e in backend.events if e[0] in ("consume", "loss_update_spy")]
        self.assertEqual(events, [("consume", 0), ("loss_update_spy", 1), ("consume", 1),
                                  ("loss_update_spy", 2), ("consume", 2), ("loss_update_spy", 3)])
        controller.training_epoch(graphs, expected_updates=3)
        self.assertEqual(sum(e[0] == "reset" for e in backend.events), 2)
        self.assertEqual((controller.budget.epochs, controller.budget.updates), (2, 6))
        self.assertIsNone(backend.is_training)

    def test_validation_replays_fresh_prefix_and_scores_before_each_consume(self):
        controller, backend = self.controller()
        controller.validation([graph(0), graph(1)], [graph(2, B), graph(3, B)])
        events = [(e[0], e[1]) for e in backend.events if e[0] in ("score", "consume")]
        self.assertEqual(events, [("consume", 0), ("consume", 1), ("score", 2), ("consume", 2), ("score", 3), ("consume", 3)])
        self.assertTrue(all(e[-1] is False for e in backend.events if e[0] == "score"))
        self.assertFalse(any(e[0] == "loss_update_spy" for e in backend.events))
        self.assertEqual(controller.budget.updates, 0)

    def test_split_predecessor_and_backend_double_consumption_rejected(self):
        controller, backend = self.controller()
        with self.assertRaises(ContractError):
            controller.validation([graph(0), graph(2)], [graph(3)])
        original = backend.consume
        def wrong(state, snapshot):
            result = original(state, snapshot)
            return CompleteState(result.hidden, result.cell, result.next_target_t + 1, result.node_ids)
        backend.consume = wrong
        with self.assertRaisesRegex(ContractError, "exactly one"):
            controller.validation([graph(0)], [graph(1)])

    def test_test_stays_closed_until_selection_and_opens_once(self):
        controller, backend = self.controller()
        selection = ValidationOnlySelection()
        opened = []
        def opener():
            opened.append(True)
            return [graph(2, B), graph(3, B)]
        with self.assertRaises(ContractError):
            controller.test_once(selection, [graph(0), graph(1)], opener)
        self.assertEqual(opened, [])
        selection.consider(reference(1), 0.4, split="validation", epoch_complete=True)
        selection.finalize()
        rows = controller.test_once(selection, [graph(0), graph(1)], opener)
        self.assertEqual(len(rows), 2)
        self.assertEqual(opened, [True])
        self.assertLess(backend.events.index(("load_selected_spy", 1)), backend.events.index(("reset",)))
        with self.assertRaises(ContractError):
            controller.test_once(selection, [graph(0), graph(1)], opener)
        with self.assertRaises(ContractError):
            controller.validation([graph(0)], [graph(1)])

    def test_validation_only_earliest_tie_and_complete_epoch_selection(self):
        selection = ValidationOnlySelection()
        for split, complete in (("test", True), ("validation", False)):
            with self.assertRaises(ContractError):
                selection.consider(reference(1), 1, split=split, epoch_complete=complete)
        self.assertTrue(selection.consider(reference(1), 0.8, split="validation", epoch_complete=True))
        self.assertFalse(selection.consider(reference(2), 0.8, split="validation", epoch_complete=True))
        self.assertEqual(selection.finalize().epoch, 1)
        with self.assertRaises(ContractError):
            selection.consider(reference(3), 1, split="validation", epoch_complete=True)

    def test_training_is_terminal_after_test_opens(self):
        controller, backend = self.controller()
        selection = ValidationOnlySelection()
        selection.consider(reference(1), 0.4, split="validation", epoch_complete=True)
        selection.finalize()
        controller.test_once(selection, [graph(0)], lambda: [graph(1)])
        prior_events = list(backend.events)
        with self.assertRaisesRegex(ContractError, "training cannot continue"):
            controller.training_epoch([graph(0), graph(1)], expected_updates=1)
        self.assertEqual(backend.events, prior_events)
        self.assertEqual(controller.budget.updates, 0)

    def test_test_load_expiry_or_stop_prevents_opener(self):
        for condition in ("walltime", "owner_stop"):
            with self.subTest(condition=condition):
                now, stopped, opened = [0.0], [False], []
                layout = StateLayout(1, 3, 2, (0, 1, 2))
                backend = SpyBackend(layout)
                budget = PilotBudget(clock=lambda: now[0], max_seconds=1, owner_stop=lambda: stopped[0])
                controller = CausalController(backend, layout, budget, period=4, block=2)
                selection = ValidationOnlySelection()
                selection.consider(reference(1), 0.4, split="validation", epoch_complete=True)
                selection.finalize()
                original_load = backend.load_selected
                def load(selected):
                    original_load(selected)
                    if condition == "walltime":
                        now[0] = 1
                    else:
                        stopped[0] = True
                backend.load_selected = load
                def opener():
                    opened.append(True)
                    return [graph(1)]
                with self.assertRaises(PilotStopped):
                    controller.test_once(selection, [graph(0)], opener)
                self.assertEqual(opened, [])
                self.assertFalse(controller.test_opened)
                self.assertFalse(any(event[0] == "reset" for event in backend.events))

    def test_test_opener_expiry_or_stop_prevents_reset_and_replay(self):
        for condition in ("walltime", "owner_stop"):
            with self.subTest(condition=condition):
                now, stopped, opened = [0.0], [False], []
                layout = StateLayout(1, 3, 2, (0, 1, 2))
                backend = SpyBackend(layout)
                budget = PilotBudget(clock=lambda: now[0], max_seconds=1, owner_stop=lambda: stopped[0])
                controller = CausalController(backend, layout, budget, period=4, block=2)
                selection = ValidationOnlySelection()
                selection.consider(reference(1), 0.4, split="validation", epoch_complete=True)
                selection.finalize()
                def opener():
                    opened.append(True)
                    if condition == "walltime":
                        now[0] = 1
                    else:
                        stopped[0] = True
                    return [graph(1)]
                with self.assertRaises(PilotStopped):
                    controller.test_once(selection, [graph(0)], opener)
                self.assertEqual(opened, [True])
                self.assertTrue(controller.test_opened)
                self.assertFalse(any(event[0] == "reset" for event in backend.events))

    def test_checkpoint_reference_requires_recurrent_and_head_hashes(self):
        with self.assertRaises(ContractError):
            CheckpointReference(1, "a" * 64, "b" * 64, "")

    def test_loss_plateau_strict_patience_and_zero_best_floor(self):
        patience = LossPatience(patience=2)
        self.assertFalse(patience.observe_complete_epoch(1))
        self.assertFalse(patience.observe_complete_epoch(1))
        self.assertTrue(patience.observe_complete_epoch(1))
        self.assertFalse(patience.observe_complete_epoch(0.5))
        zero = LossPatience(patience=1)
        self.assertFalse(zero.observe_complete_epoch(0))
        self.assertTrue(zero.observe_complete_epoch(0))
        with self.assertRaises(ContractError):
            zero.observe_complete_epoch(float("nan"))

    def test_finite_caps_and_partial_epoch_not_completed(self):
        controller, backend = self.controller(max_updates=1)
        with self.assertRaisesRegex(PilotStopped, "optimizer_update_cap"):
            controller.training_epoch([graph(0), graph(1), graph(2, B)], expected_updates=2)
        self.assertEqual((controller.budget.updates, controller.budget.epochs), (1, 0))
        self.assertEqual(sum(e[0] == "loss_update_spy" for e in backend.events), 1)
        now = [0.0]
        wall = PilotBudget(clock=lambda: now[0], max_seconds=1)
        now[0] = 1
        with self.assertRaisesRegex(PilotStopped, "walltime"):
            wall.check_boundary()
        epochs = PilotBudget(clock=lambda: 0, max_epochs=1)
        epochs.epochs = 1
        with self.assertRaisesRegex(PilotStopped, "epoch_cap"):
            epochs.check_boundary(training=True)
        owner = PilotBudget(clock=lambda: 0, owner_stop=lambda: True)
        with self.assertRaisesRegex(PilotStopped, "owner_stop"):
            owner.check_boundary()

    def test_seed_precedes_each_fresh_model_and_optimizer_construction(self):
        events = []
        def seed_all(seed): events.append(("seed", seed))
        def model_factory():
            model = object(); events.append(("model", model)); return model
        def optimizer_factory(model):
            optimizer = object(); events.append(("optimizer", model, optimizer)); return optimizer
        first = fresh_objects(0, seed_all=seed_all, model_factory=model_factory, optimizer_factory=optimizer_factory)
        second = fresh_objects(0, seed_all=seed_all, model_factory=model_factory, optimizer_factory=optimizer_factory)
        self.assertEqual([e[0] for e in events], ["seed", "model", "optimizer"] * 2)
        self.assertIsNot(first[0], second[0]); self.assertIsNot(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
