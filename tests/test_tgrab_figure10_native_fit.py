from __future__ import annotations

import contextlib
from pathlib import Path
import sys
import threading
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_collapse import graph_reconstruction as gr  # noqa: E402
from representation_collapse.tgrab_figure10_fit import (  # noqa: E402
    FitPolicy,
    OptimizerAttemptError,
    _run_figure10_fit_lifecycle,
)
from representation_collapse.tgrab_figure10_native_fit import (  # noqa: E402
    Figure10EpochEvidence,
    Figure10NativeFitAdapter,
    SeedApplicationReceipt,
    _expected_state_dict_specs,
    build_figure10_fit_adapter,
    make_seeded_components,
)


class Parameter:
    def __init__(self, name):
        self.name = name


class SyntheticTensor:
    """Metadata-only tensor shim; never used by model or optimizer math."""

    def __init__(self, shape, *, dtype="torch.float32", device="cpu", value=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return SyntheticTensor(self.shape, dtype=self.dtype, device="cpu", value=self.value)

    def clone(self):
        return SyntheticTensor(self.shape, dtype=self.dtype, device=self.device, value=self.value)


class Model:
    def __init__(self, names, state_tensors=None):
        self.parameters = {name: Parameter(name) for name in names}
        self._state_tensors = dict(state_tensors or {})

    def named_parameters(self):
        return iter(self.parameters.items())

    def state_dict(self):
        return dict(self._state_tensors)


class FakeOptimizer:
    def __init__(self, params, *, raises=False):
        self.param_groups = [{"params": list(params)}]
        self.raises = raises
        self.mutations = 0

    def step(self):
        self.mutations += 1
        if self.raises:
            raise RuntimeError("step raised after possible mutation")


class FakeNativeBackend(gr.NativeGCLSTMBackend):
    """Native-backend-shaped fixture with no Torch/model imports."""

    def __init__(self, *, update_failure=None, optimizer_failure=False, device="cpu"):
        self.optimizer = FakeOptimizer([], raises=optimizer_failure)
        self.models = None
        self.device = device
        self.optimizer_failure = optimizer_failure
        self.update_failure = update_failure
        self.mode = None
        self.epoch = 0
        self.update_count = 0
        self.consumed = []
        self.negatives = []
        self.last_pairs = None
        self.score_calls = 0
        self.terminal_validation_states = []

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
        return gr.CompleteState(SyntheticTensor((1, 100, 128), device=self.device),
                                SyntheticTensor((1, 100, 128), device=self.device),
                                0, tuple(range(100)))

    def detach(self, state):
        return state

    def consume(self, state, snapshot):
        if state.next_target_t != snapshot.timestamp:
            raise AssertionError("unexpected consume order")
        self.consumed.append((self.mode, snapshot.timestamp))
        state.hidden.value = snapshot.timestamp + 1
        state.cell.value = -(snapshot.timestamp + 1)
        result = gr.CompleteState(state.hidden, state.cell, snapshot.timestamp + 1,
                                  state.node_ids)
        if self.mode == "eval" and snapshot.timestamp == 2815:
            self.terminal_validation_states.append(result)
        return result

    def loss_and_update(self, state, target, negatives):
        if state.next_target_t != target.timestamp:
            raise AssertionError("target scored from wrong state")
        self.negatives.append(tuple(negatives))
        self.update_count += 1
        if self.update_failure == "before":
            raise RuntimeError("pre-step failure")
        self.optimizer.step()
        if self.models is not None:
            for model in self.models.values():
                for tensor in model.state_dict().values():
                    tensor.value += 1
        if self.update_failure == "after":
            raise RuntimeError("post-step failure")
        return 0.25

    def score_pairs(self, state, pairs):
        self.score_calls += 1
        self.last_pairs = tuple(pairs)
        return np.zeros(len(pairs), dtype=np.float64)


def graph(timestamp, pairs=((0, 1), (1, 0))):
    return gr.Snapshot(timestamp, 100, tuple(pairs), tuple(range(100)))


def seed_receipt(seed=0):
    return SeedApplicationReceipt(seed, ("python", "numpy", "torch_cpu"), "a" * 64)


def make_adapter(train, val, backend=None, *, model_seed=0, negative_seed=1,
                 corrupt_backend_binding=None, policy=FitPolicy()):
    backend = backend or FakeNativeBackend()

    def model_factory():
        states = {}
        for component in ("node_emb", "link_pred"):
            states[component] = {
                name: SyntheticTensor(shape, dtype=dtype, device=backend.device)
                for name, shape, dtype in _expected_state_dict_specs()[component]
            }
        return {"node_emb": Model(["state", "gate"], states["node_emb"]),
                "link_pred": Model(["readout", "bias"], states["link_pred"])}

    def optimizer_factory(models):
        params = [models[component].parameters[name]
                  for component in ("node_emb", "link_pred")
                  for name in sorted(models[component].parameters)]
        return FakeOptimizer(params, raises=backend.optimizer_failure)

    def backend_factory(models, optimizer):
        backend.models = models
        backend.optimizer = optimizer
        if corrupt_backend_binding == "models":
            backend.models = dict(models)
        elif corrupt_backend_binding == "optimizer":
            backend.optimizer = FakeOptimizer([])
        return backend

    return build_figure10_fit_adapter(
        model_seed=model_seed, seed_all=seed_receipt,
        model_factory=model_factory, optimizer_factory=optimizer_factory,
        backend_factory=backend_factory, training_graphs=train,
        validation_graphs=val, negative_seed=negative_seed, policy=policy)


def run_synthetic_adapter_fit(adapter, *, policy, on_attempt=lambda _: None,
                              on_update=lambda _: None, on_selection=lambda _: None,
                              on_epoch_complete=None):
    """Test-only short-run harness; production adapter has no such route."""
    policy.validate()
    if adapter._used:
        raise ValueError("fit adapter instances are single-use")
    adapter._used = True
    adapter._fit_active = True
    adapter._fit_thread_id = threading.get_ident()
    try:
        result = _run_figure10_fit_lifecycle(
            backend=adapter, training_graphs=adapter.training_graphs,
            validation_graphs=adapter.validation_graphs, policy=policy,
            on_attempt=on_attempt, on_update=on_update,
            on_selection=on_selection,
            on_epoch_complete=(
                (lambda record, state: on_epoch_complete(
                    adapter._snapshot_epoch_evidence(record, state)))
                if on_epoch_complete is not None else lambda *_: None))
    finally:
        adapter._fit_active = False
        adapter._fit_thread_id = None
        adapter._phase = None
    return result, tuple(adapter.validation_rows)


class NativeFitAdapterTests(unittest.TestCase):
    def test_seed_precedes_fresh_model_and_optimizer_and_names_are_canonical(self):
        events = []

        def seed(value):
            events.append(("seed", value))
            return seed_receipt(value)

        def build_models():
            events.append(("model", None))
            return {"link_pred": Model(["z", "a"]),
                    "node_emb": Model(["q", "b"])}

        def build_optimizer(models):
            events.append(("optimizer", None))
            ordered = [models[name].parameters[key]
                       for name in ("node_emb", "link_pred")
                       for key in sorted(models[name].parameters)]
            return FakeOptimizer(ordered)

        built = make_seeded_components(model_seed=0, seed_all=seed,
            model_factory=build_models, optimizer_factory=build_optimizer)
        self.assertEqual([event[0] for event in events], ["seed", "model", "optimizer"])
        self.assertEqual(events[0], ("seed", 0))
        self.assertEqual(built.parameter_names,
                         ("node_emb.b", "node_emb.q", "link_pred.a", "link_pred.z"))
        with self.assertRaisesRegex(ValueError, "seed is exactly 0"):
            make_seeded_components(model_seed=1, seed_all=seed_receipt,
                model_factory=build_models,
                optimizer_factory=lambda models: FakeOptimizer([
                    models["node_emb"].parameters[key]
                    for key in sorted(models["node_emb"].parameters)] +
                    [models["link_pred"].parameters[key]
                     for key in sorted(models["link_pred"].parameters)]))

    def test_factory_receipt_binds_backend_model_and_optimizer_identity(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        for binding in ("models", "optimizer"):
            with self.assertRaisesRegex(ValueError, "factory receipt"):
                make_adapter(train, val, corrupt_backend_binding=binding)
        with self.assertRaises(TypeError):
            Figure10NativeFitAdapter(FakeNativeBackend(), train, val)

    def test_builder_rejects_short_policy_before_any_factory_runs(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        calls = []

        def seed(seed):
            calls.append("seed")
            return seed_receipt(seed)

        def models():
            calls.append("model")
            return {"node_emb": Model(["a"]), "link_pred": Model(["b"])}

        def optimizer(model_map):
            calls.append("optimizer")
            return FakeOptimizer([model_map["node_emb"].parameters["a"],
                                  model_map["link_pred"].parameters["b"]])

        def backend(model_map, opt):
            calls.append("backend")
            result = FakeNativeBackend()
            result.models, result.optimizer = model_map, opt
            return result

        with self.assertRaisesRegex(ValueError, "exact frozen FitPolicy"):
            build_figure10_fit_adapter(
                model_seed=0, seed_all=seed, model_factory=models,
                optimizer_factory=optimizer, backend_factory=backend,
                training_graphs=train, validation_graphs=val,
                policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(calls, [])

    def test_parameter_order_mismatch_is_rejected(self):
        def build_models():
            return {"node_emb": Model(["a", "b"]), "link_pred": Model(["c"])}

        with self.assertRaisesRegex(ValueError, "order differs"):
            make_seeded_components(model_seed=0, seed_all=seed_receipt,
                model_factory=build_models,
                optimizer_factory=lambda models: FakeOptimizer([
                    models["node_emb"].parameters["b"],
                    models["node_emb"].parameters["a"],
                    models["link_pred"].parameters["c"]]))

    def test_seed_noop_callback_without_rng_attestation_fails_before_model_factory(self):
        model_calls = []
        with self.assertRaisesRegex(ValueError, "attest Python/NumPy/Torch CPU RNG"):
            make_seeded_components(model_seed=0, seed_all=lambda _: None,
                model_factory=lambda: model_calls.append("constructed") or {
                    "node_emb": Model(["a"]), "link_pred": Model(["b"])},
                optimizer_factory=lambda _: FakeOptimizer([]))
        self.assertEqual(model_calls, [])

    def test_negative_sampler_is_dedicated_reproducible_seed_one(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        first_backend, second_backend = FakeNativeBackend(), FakeNativeBackend()
        first = make_adapter(train, val, first_backend)
        second = make_adapter(train, val, second_backend)
        run_synthetic_adapter_fit(first, policy=FitPolicy(max_complete_epochs=1))
        run_synthetic_adapter_fit(second, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(first_backend.negatives, second_backend.negatives)
        gr.validate_negatives(train[1], first_backend.negatives[0])
        with self.assertRaisesRegex(ValueError, "seed is 1"):
            make_adapter(train, val, negative_seed=2)

    def test_frozen_fit_lifecycle_wires_adapter_metrics_and_exact_update_count(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)
        result, rows = run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(result.attempted_updates, 2559)
        self.assertEqual(result.completed_updates, 2559)
        self.assertEqual(backend.optimizer.mutations, 2559)
        self.assertEqual(len(backend.negatives), 2559)
        self.assertEqual(len(rows), 256)
        self.assertTrue(all(row["epoch"] == 1 for row in rows))
        self.assertEqual([row["target_timestamp"] for row in rows], list(range(2560, 2816)))
        self.assertEqual(rows[0]["pairs"], 9900)
        self.assertEqual(rows[0]["f1"], 0.0)
        self.assertEqual(backend.last_pairs, gr.ordered_offdiagonal_pairs(100))
        state = gr.CompleteState(np.zeros((1, 100, 1)), np.zeros((1, 100, 1)),
                                 1, tuple(range(100)))
        with self.assertRaisesRegex(ValueError, "private to the active training epoch"):
            adapter.train_update(state, train[1])
        self.assertEqual(backend.optimizer.mutations, 2559)
        with self.assertRaisesRegex(ValueError, "single-use"):
            run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))

    def test_public_fit_rejects_short_policy_before_backend_activity(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)
        with self.assertRaisesRegex(ValueError, "exact frozen FitPolicy"):
            adapter.fit(policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(backend.update_count, 0)
        self.assertEqual(backend.consumed, [])
        self.assertEqual(backend.score_calls, 0)
        self.assertFalse(adapter._used)

    def test_public_fit_forwards_attempt_callback_before_each_native_update(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)
        seen = []

        def before_update(event):
            seen.append((event.copy(), backend.update_count))

        # Stop on the first callback after confirming the public path forwards
        # the journal barrier without entering the backend update.
        def stop_after_first(event):
            before_update(event)
            raise RuntimeError("test stop")

        with self.assertRaisesRegex(RuntimeError, "test stop"):
            adapter.fit(on_attempt=stop_after_first)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][1], 0)
        self.assertEqual(seen[0][0]["sequence"], 1)
        self.assertEqual(seen[0][0]["epoch"], 1)
        self.assertEqual(seen[0][0]["training_order_index"], 0)
        self.assertEqual(seen[0][0]["target_timestamp"], 1)
        self.assertEqual(seen[0][0]["consumed_timestamp"], 0)
        self.assertEqual(backend.update_count, 0)

    def test_epoch_evidence_clones_catalog_state_and_keeps_each_epoch_rows(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)
        evidence = []
        selections = []
        run_synthetic_adapter_fit(
            adapter, policy=FitPolicy(max_complete_epochs=2),
            on_epoch_complete=evidence.append,
            on_selection=lambda record: selections.append(record.epoch))

        self.assertEqual([item.record.epoch for item in evidence], [1, 2])
        self.assertEqual([item.record.validation_selected for item in evidence], [True, False])
        self.assertEqual(selections, [1])
        for epoch, item in enumerate(evidence, start=1):
            self.assertIsInstance(item, Figure10EpochEvidence)
            self.assertEqual(item.terminal_state.next_target_t, 2816)
            self.assertEqual(item.terminal_state.node_ids, tuple(range(100)))
            self.assertEqual(item.terminal_state.hidden.value, 2816)
            self.assertEqual(item.terminal_state.cell.value, -2816)
            self.assertEqual(len(item.validation_rows), 256)
            self.assertEqual([row["epoch"] for row in item.validation_rows], [epoch] * 256)
            self.assertEqual([row["target_timestamp"] for row in item.validation_rows],
                             list(range(2560, 2816)))
            self.assertIsNot(item.terminal_state.hidden,
                             backend.terminal_validation_states[epoch - 1].hidden)
            self.assertEqual(set(item.model_state_dicts), {"node_emb", "link_pred"})
            for backend_key, tensors in item.model_state_dicts.items():
                self.assertEqual(tuple(tensors), tuple(
                    name for name, _, _ in _expected_state_dict_specs()[backend_key]))
                for name, shape, dtype in _expected_state_dict_specs()[backend_key]:
                    tensor = tensors[name]
                    self.assertEqual(tuple(tensor.shape), shape)
                    self.assertEqual(tensor.dtype, dtype)
                    self.assertEqual(tensor.device, "cpu")
                    self.assertEqual(tensor.value, epoch * 2559)
        # The first evidence owns independent values despite later optimizer
        # mutations; nested mappings are also read-only.
        first_name = next(iter(evidence[0].model_state_dicts["node_emb"]))
        self.assertEqual(evidence[0].model_state_dicts["node_emb"][first_name].value, 2559)
        self.assertEqual(adapter.backend.models["node_emb"].state_dict()[first_name].value, 5118)
        with self.assertRaises(TypeError):
            evidence[0].model_state_dicts["node_emb"]["injected"] = SyntheticTensor(())
        with self.assertRaises(TypeError):
            evidence[0].validation_rows[0]["epoch"] = 99

    def test_cuda_source_is_rejected_by_frozen_cpu_policy_and_mixed_device_fails_closed(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend(device="cuda:0")
        with self.assertRaisesRegex(ValueError, "requires a CPU backend"):
            make_adapter(train, val, backend)
        self.assertEqual(backend.update_count, 0)
        source_name = _expected_state_dict_specs()["link_pred"][0][0]
        backend = FakeNativeBackend(device="cpu")
        adapter = make_adapter(train, val, backend)
        backend.models["link_pred"].state_dict()[source_name].device = "cuda:1"
        with self.assertRaisesRegex(ValueError, "pinned CPU tensor catalog"):
            adapter.fit()
        self.assertEqual(backend.update_count, 0)

    def test_f1_adapter_scores_exact_ordered_pair_fixture(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)

        def exact_probabilities(state, pairs):
            positives = set(val[0].positive_pairs)
            return np.asarray([0.9 if pair in positives else 0.1 for pair in pairs])

        backend.score_pairs = exact_probabilities
        result, rows = run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(result.epochs_completed, 1)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[0]["tp"], 2)
        self.assertEqual(rows[0]["fp"], 0)
        self.assertEqual(rows[0]["fn"], 0)
        self.assertTrue(rows[0]["change_point"] is False)
        self.assertTrue(all(row["f1"] == 1.0 for row in rows))

    def test_step_status_distinguishes_before_after_and_inside_step_exceptions(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        state = gr.CompleteState(np.zeros((1, 100, 1)), np.zeros((1, 100, 1)),
                                 1, tuple(range(100)))

        backend = FakeNativeBackend(update_failure="before")
        adapter = make_adapter(train, val, backend)
        with self.assertRaises(OptimizerAttemptError) as before:
            run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(before.exception.step_status, "not_started")
        self.assertEqual(backend.optimizer.mutations, 0)

        backend = FakeNativeBackend(update_failure="after")
        adapter = make_adapter(train, val, backend)
        with self.assertRaises(OptimizerAttemptError) as after:
            run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(after.exception.step_status, "completed")
        self.assertEqual(backend.optimizer.mutations, 1)

        backend = FakeNativeBackend(optimizer_failure=True)
        adapter = make_adapter(train, val, backend)
        with self.assertRaises(OptimizerAttemptError) as inside:
            run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertEqual(inside.exception.step_status, "unknown")
        self.assertEqual(backend.optimizer.mutations, 1)

    def test_optimizer_step_wrapper_is_restored_after_success_or_error(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend(update_failure="after")
        adapter = make_adapter(train, val, backend)
        original = backend.optimizer.step
        with self.assertRaises(OptimizerAttemptError):
            run_synthetic_adapter_fit(adapter, policy=FitPolicy(max_complete_epochs=1))
        self.assertIs(backend.optimizer.step.__self__, original.__self__)
        self.assertEqual(backend.optimizer.step.__func__, original.__func__)

    def test_update_and_validation_score_calls_outside_fit_are_rejected_before_backend_work(self):
        train = [graph(t) for t in range(2560)]
        val = [graph(t) for t in range(2560, 2816)]
        backend = FakeNativeBackend()
        adapter = make_adapter(train, val, backend)
        train_state = gr.CompleteState(np.zeros((1, 100, 1)), np.zeros((1, 100, 1)),
                                       1, tuple(range(100)))
        val_state = gr.CompleteState(np.zeros((1, 100, 1)), np.zeros((1, 100, 1)),
                                     2560, tuple(range(100)))
        with self.assertRaisesRegex(ValueError, "private to the active training epoch"):
            adapter.train_update(train_state, train[1])
        with self.assertRaisesRegex(ValueError, "active fit epoch"):
            adapter.score_f1(val_state, val[0])
        self.assertEqual(backend.update_count, 0)
        self.assertEqual(backend.optimizer.mutations, 0)
        self.assertEqual(backend.score_calls, 0)
        self.assertEqual(backend.negatives, [])


if __name__ == "__main__":
    unittest.main()
