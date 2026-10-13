"""Tiny CPU calibrations of complete-state matrix-free derivatives."""

import json
import math
import unittest
from unittest.mock import patch

import torch

from representation_collapse.metrics import ordered_product
from representation_collapse.periodicity import recurrent_step, tangent_window
from representation_collapse.pilot import VectorMemoryModel
from representation_collapse.tangent_operator import (
    StateComponent,
    StateLayout,
    TrajectoryTangentOperator,
    estimate_leading_singular_gain,
)


def flat_layout(dimension):
    return StateLayout((StateComponent("complete_state", (dimension,)),))


def linear_operator(matrices, initial=None, dtype=torch.float64):
    dimension = matrices[0].shape[0] if matrices else initial.numel()
    if initial is None:
        initial = torch.linspace(-0.2, 0.3, dimension, dtype=dtype)
    states = [initial]
    for matrix in matrices:
        states.append(matrix @ states[-1])
    return TrajectoryTangentOperator(
        lambda state, matrix: matrix @ state, states, matrices,
        layout=flat_layout(dimension), trajectory_atol=0, trajectory_rtol=0,
    )


class TangentOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_json_diagnostics(self, result):
        json.dumps(result.diagnostics, allow_nan=False)
        self.assertFalse(result.diagnostics["floquet_certified"])
        self.assertFalse(result.diagnostics["leading_pair_certified"])
        self.assertFalse(result.diagnostics["all_directions_contraction_certified"])
        self.assertFalse(result.diagnostics["scientific_acceptance"])

    def test_layout_includes_layers_nodes_hidden_and_cell_in_explicit_order(self):
        layout = StateLayout((StateComponent("h", (2, 3, 4)), StateComponent("c", (2, 3, 4))))
        hidden = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        cell = hidden + 100
        flat = layout.pack({"c": cell, "h": hidden})
        self.assertEqual(layout.dimension, 48)
        torch.testing.assert_close(flat[:24], hidden.reshape(-1), atol=0, rtol=0)
        recovered = layout.unpack(flat)
        torch.testing.assert_close(recovered["h"], hidden, atol=0, rtol=0)
        torch.testing.assert_close(recovered["c"], cell, atol=0, rtol=0)
        self.assertEqual(layout.record()["component_order"], ["h", "c"])
        with self.assertRaises(ValueError):
            layout.pack({"h": hidden})
        with self.assertRaises(ValueError):
            layout.pack({"h": hidden, "c": cell.reshape(24)})
        with self.assertRaises(ValueError):
            StateLayout((StateComponent("h", (2,)), StateComponent("h", (2,))))

    def test_noncommuting_chronological_product_and_reverse_transpose(self):
        a = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64)
        b = torch.tensor([[0.5, 0.0], [1.0, 0.75]], dtype=torch.float64)
        operator = linear_operator([a, b])
        vector = torch.tensor([0.6, -0.8], dtype=torch.float64)
        product = b @ a
        torch.testing.assert_close(product, torch.from_numpy(ordered_product(torch.stack([a, b]).numpy())), atol=0, rtol=0)
        self.assertFalse(torch.allclose(a @ b, product))
        torch.testing.assert_close(operator.jvp(vector), product @ vector, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(operator.vjp(vector), product.T @ vector, atol=1e-14, rtol=1e-14)
        wrong_order = a @ b @ vector
        self.assertGreater(float(torch.linalg.vector_norm(operator.jvp(vector) - wrong_order)), 0.1)

    def graph_fixture(self, dtype=torch.float64):
        # A fully coupled tiny h/c recurrence; this is not upstream GC-LSTM.
        layout = StateLayout((StateComponent("h", (1, 2, 2)), StateComponent("c", (1, 2, 2))))
        def step(state, fixed):
            components = layout.unpack(state)
            h, c = components["h"], components["c"]
            neighbors = torch.einsum("ij,ljk->lik", fixed["adjacency"], h)
            x = fixed["features"]
            forget = torch.sigmoid(0.3 * h + 0.2 * neighbors + x)
            write = torch.sigmoid(-0.1 * h + 0.15 * c - 0.3 * x)
            candidate = torch.tanh(0.4 * neighbors + 0.2 * h + x)
            c_new = forget * c + write * candidate
            h_new = torch.sigmoid(0.25 * c_new - 0.15 * neighbors) * torch.tanh(c_new)
            return layout.pack({"h": h_new, "c": c_new})
        initial = torch.linspace(-0.3, 0.4, 8, dtype=dtype)
        inputs = [
            {"adjacency": torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=dtype),
             "features": torch.tensor([[[0.1, -0.2], [0.3, 0.15]]], dtype=dtype)},
            {"adjacency": torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=dtype),
             "features": torch.tensor([[[-0.15, 0.2], [-0.1, 0.05]]], dtype=dtype)},
            {"adjacency": torch.tensor([[0.1, 0.9], [0.6, 0.4]], dtype=dtype),
             "features": torch.tensor([[[0.2, 0.1], [0.1, -0.25]]], dtype=dtype)},
        ]
        states = [initial]
        for fixed in inputs:
            states.append(step(states[-1], fixed).detach())
        operator = TrajectoryTangentOperator(step, states, inputs, layout=layout,
                                             trajectory_atol=0, trajectory_rtol=0)
        return operator, step, states, inputs

    def test_coupled_complete_hc_jvp_vjp_dense_full_rollout_and_finite_difference(self):
        operator, step, states, inputs = self.graph_fixture()
        product = torch.eye(8, dtype=torch.float64)
        for state, fixed in zip(states, inputs):
            jacobian = torch.autograd.functional.jacobian(lambda value: step(value, fixed), state)
            # Cross-node derivative prevents a per-node diagonal interpretation.
            self.assertGreater(float(jacobian[0, 2].abs()), 1e-3)
            product = jacobian @ product
        def full(value):
            for fixed in inputs:
                value = step(value, fixed)
            return value
        full_jacobian = torch.autograd.functional.jacobian(full, states[0])
        torch.testing.assert_close(product, full_jacobian, atol=1e-13, rtol=1e-12)
        v = torch.linspace(0.2, -0.3, 8, dtype=torch.float64)
        w = torch.linspace(-0.5, 0.1, 8, dtype=torch.float64)
        torch.testing.assert_close(operator.jvp(v), product @ v, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(operator.vjp(w), product.T @ w, atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(torch.dot(w, operator.jvp(v)), torch.dot(operator.vjp(w), v), atol=1e-13, rtol=1e-12)
        epsilon = 1e-6
        difference = (full(states[0] + epsilon * v) - full(states[0] - epsilon * v)) / (2 * epsilon)
        torch.testing.assert_close(operator.jvp(v), difference, atol=3e-11, rtol=1e-8)
        result = estimate_leading_singular_gain(operator, seed=812, max_iterations=200, residual_tolerance=1e-10)
        self.assertTrue(result.diagnostics["converged"])
        self.assertAlmostEqual(result.diagnostics["gain"], float(torch.linalg.svdvals(product)[0]), places=11)
        self.assert_json_diagnostics(result)

    def test_float32_coupled_fixture_against_dense_reference(self):
        operator, step, states, inputs = self.graph_fixture(torch.float32)
        dense = torch.eye(8)
        for state, fixed in zip(states, inputs):
            dense = torch.autograd.functional.jacobian(lambda value: step(value, fixed), state) @ dense
        vector = torch.linspace(-0.4, 0.3, 8)
        torch.testing.assert_close(operator.jvp(vector), dense @ vector, atol=2e-8, rtol=2e-5)
        torch.testing.assert_close(operator.vjp(vector), dense.T @ vector, atol=2e-8, rtol=2e-5)
        result = estimate_leading_singular_gain(operator, seed=812, residual_tolerance=1e-5)
        self.assertTrue(result.diagnostics["converged"])
        self.assertLess(abs(result.diagnostics["gain"] - float(torch.linalg.svdvals(dense)[0])), 1e-7)
        self.assert_json_diagnostics(result)

    def test_existing_vector_gate_equations_match_current_dense_window_reference(self):
        # This checks the same derivative assay as periodicity.py. It does not
        # replace the separate native-state chronology/adapter-equivalence gate.
        for kind in ("gru", "lstm"):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(kind=kind, dtype=dtype), torch.random.fork_rng():
                    torch.manual_seed(913)
                    model = VectorMemoryModel(kind, "continuous", hidden_size=3, bottleneck=4).to(dtype=dtype).eval()
                    inputs = torch.randn(3, 38, dtype=dtype)
                    components = (StateComponent("h", (1, 1, 3)),)
                    if kind == "lstm":
                        components += (StateComponent("c", (1, 1, 3)),)
                    layout = StateLayout(components)
                    states = [torch.linspace(-0.3, 0.2, layout.dimension, dtype=dtype)]
                    for fixed in inputs:
                        states.append(recurrent_step(model, states[-1], fixed).detach())
                    operator = TrajectoryTangentOperator(lambda s, x: recurrent_step(model, s, x), states, list(inputs),
                                                         layout=layout, trajectory_atol=0, trajectory_rtol=0,
                                                         evaluation_modules=[model])
                    dense = torch.from_numpy(tangent_window(model, states[0], inputs)["product"])
                    vector = torch.linspace(0.1, 0.4, layout.dimension, dtype=dtype)
                    atol, rtol = (2e-8, 2e-5) if dtype == torch.float32 else (1e-13, 1e-12)
                    torch.testing.assert_close(operator.jvp(vector), dense @ vector, atol=atol, rtol=rtol)
                    torch.testing.assert_close(operator.vjp(vector), dense.T @ vector, atol=atol, rtol=rtol)

    def test_nonnormal_gain_above_one_despite_all_eigenvalues_below_one(self):
        matrix = torch.tensor([[0.5, 4.0], [0.0, 0.5]], dtype=torch.float64)
        operator = linear_operator([matrix])
        result = estimate_leading_singular_gain(operator, seed=32, residual_tolerance=1e-12)
        self.assertTrue(result.diagnostics["converged"])
        self.assertGreater(result.diagnostics["gain"], 4)
        self.assertLess(float(torch.linalg.eigvals(matrix).abs().max()), 1)
        self.assertAlmostEqual(result.diagnostics["gain"], float(torch.linalg.svdvals(matrix)[0]), places=12)
        self.assert_json_diagnostics(result)

    def test_rotation_and_uniform_contraction_singular_pair_residuals(self):
        for matrix in (
            torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64),
            torch.eye(2, dtype=torch.float64) * 0.2,
        ):
            operator = linear_operator([matrix, matrix])
            result = estimate_leading_singular_gain(operator, seed=192)
            self.assertTrue(result.diagnostics["converged"])
            self.assertEqual(result.diagnostics["iterations"], 1)
            self.assertAlmostEqual(result.diagnostics["gain"], float(torch.linalg.svdvals(matrix @ matrix)[0]), places=14)
            self.assert_json_diagnostics(result)

    def test_singular_map_keeps_initial_null_direction_failure_and_other_pair(self):
        matrix = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64))
        operator = linear_operator([matrix])
        null = estimate_leading_singular_gain(operator, initial_vector=torch.tensor([0.0, 1.0], dtype=torch.float64))
        self.assertFalse(null.diagnostics["converged"])
        self.assertEqual(null.diagnostics["status"], "direction_annihilated_at_working_precision")
        self.assertEqual(null.diagnostics["gain"], 0)
        self.assertIsNone(null.diagnostics["log_gain"])
        self.assert_json_diagnostics(null)
        result = estimate_leading_singular_gain(operator, seed=74)
        self.assertTrue(result.diagnostics["converged"])
        self.assertAlmostEqual(result.diagnostics["gain"], 2, places=14)
        self.assert_json_diagnostics(result)

    def test_zero_map_does_not_claim_complete_zero_spectrum(self):
        operator = linear_operator([torch.zeros((2, 2), dtype=torch.float64)])
        result = estimate_leading_singular_gain(operator, seed=62)
        self.assertFalse(result.diagnostics["spectrum_complete"])
        self.assertEqual(result.diagnostics["converged_pairs"], 0)
        self.assert_json_diagnostics(result)

    def test_orthogonal_initialization_can_converge_to_subleading_pair_without_certificate(self):
        operator = linear_operator([torch.diag(torch.tensor([3.0, 0.5], dtype=torch.float64))])
        result = estimate_leading_singular_gain(operator, initial_vector=torch.tensor([0.0, 1.0], dtype=torch.float64))
        self.assertTrue(result.diagnostics["converged"])
        self.assertAlmostEqual(result.diagnostics["gain"], 0.5)
        self.assertFalse(result.diagnostics["leading_pair_certified"])
        self.assert_json_diagnostics(result)

    def test_bounded_nonconvergence_preserves_residual_and_initial_vector(self):
        operator = linear_operator([torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))])
        initial = torch.tensor([1.0, 1.0], dtype=torch.float64)
        result = estimate_leading_singular_gain(operator, initial_vector=initial, max_iterations=1, residual_tolerance=1e-12)
        self.assertFalse(result.diagnostics["converged"])
        self.assertEqual(result.diagnostics["status"], "maximum_iterations_reached")
        self.assertGreater(result.diagnostics["relative_right_singular_residual"], 0.1)
        torch.testing.assert_close(result.initial_vector, initial, atol=0, rtol=0)
        torch.testing.assert_close(result.right_vector, initial / math.sqrt(2), atol=1e-15, rtol=1e-15)
        self.assert_json_diagnostics(result)

    def test_stepwise_scaling_preserves_logs_when_raw_long_product_underflows_or_overflows(self):
        for scale in (1e-200, 1e200):
            matrices = [torch.eye(2, dtype=torch.float64) * scale for _ in range(3)]
            # The zero reference trajectory is finite at every update.
            operator = linear_operator(matrices, initial=torch.zeros(2, dtype=torch.float64))
            initial = torch.tensor([1.0, 0.0], dtype=torch.float64)
            result = estimate_leading_singular_gain(operator, initial_vector=initial, residual_tolerance=1e-10)
            self.assertTrue(result.diagnostics["converged"])
            self.assertIsNone(result.diagnostics["gain"])
            self.assertAlmostEqual(result.diagnostics["log_gain"], 3 * math.log(scale), places=11)
            self.assertAlmostEqual(result.diagnostics["log_gain_per_update"], math.log(scale), places=11)
            self.assert_json_diagnostics(result)
            if scale < 1:
                self.assertTrue(torch.equal(operator.jvp(initial), torch.zeros_like(initial)))
            else:
                with self.assertRaises(FloatingPointError):
                    operator.jvp(initial)

    def test_scaled_transpose_reconstructs_dense_product_and_adjoint_inner_product(self):
        a = torch.tensor([[0.2, 0.3], [-0.1, 0.4]], dtype=torch.float64)
        b = torch.tensor([[1.0, 0.2], [0.7, -0.1]], dtype=torch.float64)
        operator = linear_operator([a, b])
        v = torch.tensor([3.0, -2.0], dtype=torch.float64)
        for action, dense in ((operator.scaled_jvp, b @ a), (operator.scaled_vjp, (b @ a).T)):
            unit, log_norm = action(v)
            torch.testing.assert_close(unit * math.exp(log_norm), dense @ v, atol=1e-14, rtol=1e-14)

    def test_empty_window_is_identity_with_undefined_per_update_rate(self):
        initial = torch.tensor([0.2, 0.1], dtype=torch.float64)
        operator = linear_operator([], initial=initial)
        torch.testing.assert_close(operator.jvp(initial), initial, atol=0, rtol=0)
        torch.testing.assert_close(operator.vjp(initial), initial, atol=0, rtol=0)
        result = estimate_leading_singular_gain(operator, seed=9)
        self.assertAlmostEqual(result.diagnostics["gain"], 1, places=14)
        self.assertIsNone(result.diagnostics["log_gain_per_update"])
        self.assert_json_diagnostics(result)

    def test_snapshots_are_independent_of_caller_mutation_and_rng_global_state(self):
        matrix = torch.tensor([[2.0, 0.1], [0.2, 0.5]], dtype=torch.float64)
        state = torch.tensor([0.2, -0.3], dtype=torch.float64)
        inputs, states = [matrix], [state, matrix @ state]
        operator = TrajectoryTangentOperator(lambda s, x: x @ s, states, inputs,
                                             layout=flat_layout(2), trajectory_atol=0, trajectory_rtol=0)
        expected = operator.jvp(state.clone())
        state.zero_(); matrix.zero_(); states[-1].zero_()
        torch.testing.assert_close(operator.jvp(torch.tensor([0.2, -0.3], dtype=torch.float64)), expected, atol=0, rtol=0)
        rng_before = torch.random.get_rng_state().clone()
        first = estimate_leading_singular_gain(operator, seed=137)
        second = estimate_leading_singular_gain(operator, seed=137)
        torch.testing.assert_close(first.initial_vector, second.initial_vector, atol=0, rtol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, atol=0, rtol=0)

    def test_weights_and_evaluation_mode_are_guarded_without_parameter_gradients(self):
        model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float64).eval()
        state = torch.tensor([0.2, -0.3], dtype=torch.float64)
        operator = TrajectoryTangentOperator(lambda s, x: model(s), [state, model(state).detach()], [None],
                                             layout=flat_layout(2), trajectory_atol=0, trajectory_rtol=0,
                                             evaluation_modules=[model])
        operator.jvp(state); operator.vjp(state)
        self.assertIsNone(model.weight.grad)
        model.train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            operator.jvp(state)
        model.eval()
        with torch.no_grad():
            model.weight.add_(0.01)
        with self.assertRaisesRegex(ValueError, "changed"):
            operator.jvp(state)

    def test_replaced_guarded_parameter_and_module_identity_are_rejected(self):
        state = torch.tensor([0.2, -0.3], dtype=torch.float64)
        for replacement in ("parameter", "module"):
            model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)).to(dtype=torch.float64).eval()
            operator = TrajectoryTangentOperator(lambda s, x: model(s), [state, model(state).detach()], [None],
                                                 layout=flat_layout(2), trajectory_atol=0, trajectory_rtol=0,
                                                 evaluation_modules=[model])
            if replacement == "parameter":
                model[0].weight = torch.nn.Parameter(model[0].weight.detach().clone())
            else:
                model[0] = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.float64).eval()
            with self.assertRaisesRegex(ValueError, "identity changed"):
                operator.jvp(state)

    def test_broken_trajectory_and_observable_stochastic_replay_rejected(self):
        state = torch.ones(2, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "trajectory replay"):
            TrajectoryTangentOperator(lambda s, x: s * 2, [state, state], [None],
                                      layout=flat_layout(2), trajectory_atol=0, trajectory_rtol=0)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            TrajectoryTangentOperator(lambda s, x: s + torch.rand_like(s), [state, state], [None],
                                      layout=flat_layout(2), trajectory_atol=2, trajectory_rtol=0)

    def test_no_dense_jacobian_calls_in_operator_or_iteration(self):
        operator = linear_operator([torch.diag(torch.tensor([1.2, 0.3], dtype=torch.float64))])
        with patch("torch.autograd.functional.jacobian", side_effect=AssertionError("dense forbidden")), \
             patch("torch.func.jacrev", side_effect=AssertionError("dense forbidden")), \
             patch("torch.func.jacfwd", side_effect=AssertionError("dense forbidden")):
            operator.jvp(torch.ones(2, dtype=torch.float64))
            operator.vjp(torch.ones(2, dtype=torch.float64))
            result = estimate_leading_singular_gain(operator, seed=51)
        self.assertTrue(result.diagnostics["converged"])

    def test_nonfinite_and_unsupported_derivative_actions_are_retained_as_failures(self):
        operator = linear_operator([torch.eye(2, dtype=torch.float64)])
        for error, status in ((FloatingPointError("fixture"), "nonfinite_tangent_or_residual"),
                              (NotImplementedError("fixture"), "autodiff_action_failed")):
            with patch.object(operator, "scaled_jvp", side_effect=error):
                result = estimate_leading_singular_gain(operator, seed=12)
            self.assertFalse(result.diagnostics["converged"])
            self.assertEqual(result.diagnostics["status"], status)
            self.assertIsNone(result.diagnostics["gain"])
            self.assert_json_diagnostics(result)

    def test_failure_after_one_iteration_does_not_return_stale_gain_for_new_vector(self):
        operator = linear_operator([torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))])
        original = operator.scaled_jvp
        calls = 0
        def fail_second(vector):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("fixture")
            return original(vector)
        with patch.object(operator, "scaled_jvp", side_effect=fail_second):
            result = estimate_leading_singular_gain(operator, initial_vector=torch.ones(2, dtype=torch.float64), residual_tolerance=1e-12)
        self.assertEqual(result.diagnostics["iterations"], 2)
        self.assertEqual(len(result.diagnostics["history"]), 1)
        self.assertIsNone(result.diagnostics["gain"])
        self.assertIsNone(result.diagnostics["log_gain"])
        self.assertIsNone(result.left_vector)
        self.assert_json_diagnostics(result)

    def test_actual_infinite_derivative_at_finite_primal_is_a_numerical_failure(self):
        zero = torch.zeros(2, dtype=torch.float64)
        operator = TrajectoryTangentOperator(lambda s, x: torch.sqrt(s), [zero, zero], [None],
                                             layout=flat_layout(2), trajectory_atol=0, trajectory_rtol=0)
        result = estimate_leading_singular_gain(operator, seed=48)
        self.assertEqual(result.diagnostics["status"], "nonfinite_tangent_or_residual")
        self.assertFalse(result.diagnostics["converged"])
        self.assertIsNone(result.diagnostics["gain"])
        self.assert_json_diagnostics(result)

    def test_adjoint_breakdown_is_distinct_from_observed_forward_zero(self):
        operator = linear_operator([torch.eye(2, dtype=torch.float64)])
        with patch.object(operator, "scaled_vjp", return_value=(None, None)):
            result = estimate_leading_singular_gain(operator, seed=48)
        self.assertEqual(result.diagnostics["status"], "adjoint_breakdown_at_working_precision")
        self.assertIsNone(result.diagnostics["gain"])
        self.assertIsNone(result.diagnostics["log_gain"])
        self.assertIsNone(result.left_vector)
        self.assert_json_diagnostics(result)

    def test_invalid_precision_vector_initialization_tolerances_and_inference_mode_rejected(self):
        operator = linear_operator([torch.eye(2, dtype=torch.float64)])
        for vector in (torch.ones(3, dtype=torch.float64), torch.ones(2, dtype=torch.float32),
                       torch.tensor([float("nan"), 0.0], dtype=torch.float64)):
            with self.assertRaises(ValueError):
                operator.jvp(vector)
        for options in ({}, {"seed": True}, {"seed": -1}, {"seed": 2**63},
                        {"seed": 1, "initial_vector": torch.ones(2, dtype=torch.float64)},
                        {"initial_vector": torch.zeros(2, dtype=torch.float64)},
                        {"seed": 1, "max_iterations": 0}, {"seed": 1, "residual_tolerance": 0}):
            with self.assertRaises(ValueError):
                estimate_leading_singular_gain(operator, **options)
        with torch.inference_mode(), self.assertRaises(ValueError):
            operator.jvp(torch.ones(2, dtype=torch.float64))
        with torch.no_grad():
            torch.testing.assert_close(operator.jvp(torch.ones(2, dtype=torch.float64)), torch.ones(2, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
