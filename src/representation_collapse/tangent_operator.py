"""Matrix-free finite-time derivatives of a caller-declared complete state.

Steps must be pure deterministic functions ``F(state_flat, fixed_input)``. The
layout declares every persistent component, including all nodes/layers and
LSTM h/c; the operator cannot discover omitted state. Weights and evaluation
mode can be guarded explicitly. No orbit closure or Floquet test is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StateComponent:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("state component names must be nonempty strings")
        if not isinstance(self.shape, tuple) or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.shape
        ):
            raise ValueError("state component shapes must contain positive integer dimensions")

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class StateLayout:
    """Components in declared order, each flattened in contiguous row-major order."""

    components: tuple[StateComponent, ...]

    def __post_init__(self):
        if not isinstance(self.components, tuple) or not self.components or any(
            not isinstance(component, StateComponent) for component in self.components
        ):
            raise ValueError("a nonempty tuple of StateComponent entries is required")
        names = [component.name for component in self.components]
        if len(set(names)) != len(names):
            raise ValueError("state component names must be unique")

    @property
    def dimension(self) -> int:
        return sum(component.size for component in self.components)

    def pack(self, values: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(values, Mapping) or set(values) != {c.name for c in self.components}:
            raise ValueError("state components must match the complete declared layout")
        parts = []
        for component in self.components:
            value = values[component.name]
            if not isinstance(value, Tensor) or tuple(value.shape) != component.shape:
                raise ValueError("state component shape mismatch: " + component.name)
            parts.append(value.reshape(-1))
        if any(part.dtype != parts[0].dtype or part.device != parts[0].device for part in parts):
            raise ValueError("all state components must share dtype and device")
        return torch.cat(parts)

    def unpack(self, flat: Tensor) -> dict[str, Tensor]:
        if not isinstance(flat, Tensor) or flat.ndim != 1 or flat.numel() != self.dimension:
            raise ValueError("a flat vector matching the complete declared layout is required")
        values, offset = {}, 0
        for component in self.components:
            values[component.name] = flat[offset : offset + component.size].reshape(component.shape)
            offset += component.size
        return values

    def record(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "component_order": [component.name for component in self.components],
            "component_shapes": {component.name: list(component.shape) for component in self.components},
            "within_component_order": "contiguous row-major",
            "completeness": "caller assertion; omitted persistent state cannot be detected",
        }


def _freeze_input(value: Any) -> Any:
    # Plain nested tensor containers cover features, edge indices and fixed
    # teacher-forced inputs without persisting an autograd graph.
    if isinstance(value, Tensor):
        if value.is_complex() or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
            raise ValueError("fixed inputs must be real and finite")
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_freeze_input(part) for part in value)
    if isinstance(value, list):
        return [_freeze_input(part) for part in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _freeze_input(part) for key, part in value.items()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("fixed inputs must use tensors and plain finite Python containers/scalars")


def _unit_and_log_norm(value: Tensor) -> tuple[Tensor | None, float | None]:
    """Normalize without squaring extreme values or reconstructing a huge norm."""
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("nonfinite tangent vector")
    maximum = float(value.abs().max())
    if maximum == 0:
        return None, None
    scaled = value / maximum
    scaled_norm = float(torch.linalg.vector_norm(scaled))
    return scaled / scaled_norm, math.log(maximum) + math.log(scaled_norm)


class TrajectoryTangentOperator:
    """Apply ``P = J[L-1] ... J[0]`` and ``P.T`` without dense Jacobians.

    ``states`` contains L+1 flat states, with ``states[t+1]=F_t(states[t],x[t])``
    at the declared tolerance. External inputs are fixed, so this is a
    conditional derivative. In a closed loop the caller must include generated
    inputs and their derivative inside the step and declare all persistent
    state. A readout belongs outside this state operator unless it is itself a
    declared persistent component.

    Functions that close over weights must supply ``frozen_tensors`` or their
    ``evaluation_modules`` to guard those values. Callable identity/purity and
    completeness remain caller responsibilities; repeated primal replay is a
    check, not a proof of either property.
    """

    def __init__(
        self,
        steps: Callable[[Tensor, Any], Tensor] | Sequence[Callable[[Tensor, Any], Tensor]],
        states: Sequence[Tensor],
        inputs: Sequence[Any],
        *,
        layout: StateLayout,
        trajectory_atol: float,
        trajectory_rtol: float,
        frozen_tensors: Sequence[Tensor] = (),
        evaluation_modules: Sequence[nn.Module] = (),
    ):
        if not isinstance(layout, StateLayout):
            raise ValueError("an explicit complete-state layout is required")
        for tolerance in (trajectory_atol, trajectory_rtol):
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("trajectory tolerances must be finite and nonnegative")
        self.layout = layout
        self.trajectory_atol, self.trajectory_rtol = float(trajectory_atol), float(trajectory_rtol)
        if len(states) != len(inputs) + 1:
            raise ValueError("a trajectory requires L+1 states and L fixed inputs")
        self._states = tuple(self._validate_state(state).detach().clone() for state in states)
        if not self._states:
            raise ValueError("at least one state is required")
        self.dtype, self.device = self._states[0].dtype, self._states[0].device
        if any(state.dtype != self.dtype or state.device != self.device for state in self._states):
            raise ValueError("all trajectory states must share dtype and device")
        self._inputs = tuple(_freeze_input(value) for value in inputs)
        self._steps = (steps,) * len(inputs) if callable(steps) else tuple(steps)
        if len(self._steps) != len(inputs) or any(not callable(step) for step in self._steps):
            raise ValueError("one callable step per update is required")
        self._modules = tuple(evaluation_modules)
        if any(not isinstance(module, nn.Module) for module in self._modules):
            raise ValueError("evaluation_modules must contain torch modules")
        self._module_structure = tuple(
            (tuple((name, id(child)) for name, child in module.named_modules()),
             tuple((name, id(value)) for name, value in module.named_parameters()),
             tuple((name, id(value)) for name, value in module.named_buffers()))
            for module in self._modules
        )
        tensors = list(frozen_tensors)
        for module in self._modules:
            tensors.extend(module.parameters())
            tensors.extend(module.buffers())
        self._frozen = []
        seen = set()
        for value in tensors:
            if not isinstance(value, Tensor) or value.is_complex() or (value.is_floating_point() and not bool(torch.isfinite(value).all())):
                raise ValueError("guarded weights/buffers must be real finite tensors")
            if id(value) not in seen:
                self._frozen.append((value, value.detach().clone()))
                seen.add(id(value))
        self._check_frozen()
        self._check_ad_mode()
        # Two replays reject observable nondeterminism and a broken trajectory.
        for index in range(self.length):
            with torch.no_grad():
                first = self._step(index, self._states[index].clone())
                second = self._step(index, self._states[index].clone())
            self._check_primal(index, first)
            self._check_primal(index, second)
            if not torch.equal(first, second):
                raise ValueError("step replay is not bitwise deterministic at update " + str(index))
        self._check_frozen()

    @property
    def length(self) -> int:
        return len(self._inputs)

    def _validate_state(self, state: Tensor) -> Tensor:
        if not isinstance(state, Tensor) or state.ndim != 1 or state.numel() != self.layout.dimension:
            raise ValueError("trajectory states must be flat complete-state vectors")
        if state.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(state).all()):
            raise ValueError("states must use finite float32 or float64 values")
        return state

    def _check_ad_mode(self):
        if torch.is_inference_mode_enabled():
            raise ValueError("torch inference_mode is incompatible with tangent autodiff")

    def _check_frozen(self):
        if any(child.training for module in self._modules for child in module.modules()):
            raise ValueError("guarded model and all child modules must remain in evaluation mode")
        for module, original in zip(self._modules, self._module_structure):
            current = (tuple((name, id(child)) for name, child in module.named_modules()),
                       tuple((name, id(value)) for name, value in module.named_parameters()),
                       tuple((name, id(value)) for name, value in module.named_buffers()))
            if current != original:
                raise ValueError("guarded module, parameter or buffer identity changed")
        for current, frozen in self._frozen:
            if current.shape != frozen.shape or current.dtype != frozen.dtype or current.device != frozen.device or not torch.equal(current.detach(), frozen):
                raise ValueError("guarded weights or persistent buffers changed")

    def _step(self, index: int, state: Tensor) -> Tensor:
        # A fresh copy prevents accidental mutation of the stored input snapshot.
        return self._steps[index](state, _freeze_input(self._inputs[index]))

    def _check_primal(self, index: int, primal: Tensor):
        self._validate_state(primal)
        if primal.dtype != self.dtype or primal.device != self.device:
            raise ValueError("step output changed state dtype or device")
        if not torch.allclose(primal, self._states[index + 1], atol=self.trajectory_atol, rtol=self.trajectory_rtol):
            raise ValueError("recorded trajectory replay failed at update " + str(index))

    def _vector(self, vector: Tensor) -> Tensor:
        self._validate_state(vector)
        if vector.dtype != self.dtype or vector.device != self.device:
            raise ValueError("tangent vector must match trajectory dtype and device")
        return vector.detach().clone()

    def _apply(self, vector: Tensor, *, transpose: bool, scaled: bool):
        self._check_ad_mode()
        self._check_frozen()
        result = self._vector(vector)
        log_scale = 0.0
        if scaled:
            result, initial_log = _unit_and_log_norm(result)
            if result is None:
                return None, None
            log_scale += initial_log
        indices = range(self.length - 1, -1, -1) if transpose else range(self.length)
        with torch.enable_grad():
            for index in indices:
                state = self._states[index].clone()
                function = lambda value, index=index: self._step(index, value)
                if transpose:
                    primal, pullback = torch.func.vjp(function, state)
                    result = pullback(result)[0]
                else:
                    primal, result = torch.func.jvp(function, (state,), (result,))
                self._check_primal(index, primal)
                result = result.detach()
                if not bool(torch.isfinite(result).all()):
                    raise FloatingPointError("nonfinite tangent update at index " + str(index))
                if scaled:
                    result, step_log = _unit_and_log_norm(result)
                    if result is None:
                        self._check_frozen()
                        return None, None
                    log_scale += step_log
        self._check_frozen()
        return (result, log_scale) if scaled else result

    def jvp(self, vector: Tensor) -> Tensor:
        """Unscaled chronological product action; long products may underflow."""
        return self._apply(vector, transpose=False, scaled=False)

    def vjp(self, vector: Tensor) -> Tensor:
        """Unscaled transpose action: steps are differentiated in reverse order."""
        return self._apply(vector, transpose=True, scaled=False)

    def scaled_jvp(self, vector: Tensor) -> tuple[Tensor | None, float | None]:
        """Unit output and log output norm, rescaled between every update.

        A ``(None, None)`` output denotes an observed zero at the working dtype,
        which may reflect a genuine null direction or numerical underflow.
        """
        return self._apply(vector, transpose=False, scaled=True)

    def scaled_vjp(self, vector: Tensor) -> tuple[Tensor | None, float | None]:
        return self._apply(vector, transpose=True, scaled=True)

    def record(self) -> dict[str, Any]:
        return {
            "operator": "finite-time conditional tangent propagator",
            "product_order": "J[L-1] ... J[0]",
            "transpose_order": "J[0].T ... J[L-1].T; applied from final update backward",
            "window_updates": self.length,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "layout": self.layout.record(),
            "trajectory_atol": self.trajectory_atol,
            "trajectory_rtol": self.trajectory_rtol,
            "guarded_tensors": len(self._frozen),
            "guarded_evaluation_modules": len(self._modules),
            "floquet_certified": False,
            "dense_jacobians_constructed": False,
        }


@dataclass(frozen=True)
class SingularGainEstimate:
    initial_vector: Tensor
    right_vector: Tensor
    left_vector: Tensor | None
    diagnostics: dict[str, Any]


def estimate_leading_singular_gain(
    operator: TrajectoryTangentOperator,
    *,
    initial_vector: Tensor | None = None,
    seed: int | None = None,
    max_iterations: int = 100,
    residual_tolerance: float = 1e-8,
) -> SingularGainEstimate:
    """Power iteration on P.T P, with one honest candidate singular pair.

    Exactly one explicit initial vector or local RNG seed is required. A small
    residual certifies stationarity of this singular pair, not that it is the
    leading pair: an initialization can be orthogonal to the leading subspace.
    No complete-spectrum, rank, eigenvalue-contraction or Floquet claim follows.
    Each product is rescaled between steps; zeros/unsupported autodiff/nonfinite
    actions are retained as failures rather than replaced by arbitrary floors.
    """
    if not isinstance(operator, TrajectoryTangentOperator):
        raise ValueError("a TrajectoryTangentOperator is required")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(residual_tolerance, bool) or not isinstance(residual_tolerance, (int, float)) or not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be positive and finite")
    if (initial_vector is None) == (seed is None):
        raise ValueError("provide exactly one initial_vector or seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial_vector = torch.randn(operator.layout.dimension, dtype=operator.dtype, generator=generator).to(operator.device)
    initial = operator._vector(initial_vector)
    vector, _ = _unit_and_log_norm(initial)
    if vector is None:
        raise ValueError("initial vector must be nonzero")
    diagnostics = {
        **operator.record(),
        "algorithm": "rescaled power iteration on P.T P",
        "initialization": "explicit vector" if seed is None else "local seeded CPU Gaussian vector",
        "seed": seed,
        "requested_pairs": 1,
        "converged_pairs": 0,
        "iterations": 0,
        "max_iterations": max_iterations,
        "residual_tolerance": float(residual_tolerance),
        "relative_right_singular_residual": None,
        "residual_definition": "||P.T u - sigma v||_2 / sigma; u=P v/||P v||_2",
        "gain": None,
        "log_gain": None,
        "log_gain_per_update": None,
        "gain_representation": "unavailable",
        "converged": False,
        "leading_pair_certified": False,
        "spectrum_complete": False,
        "all_directions_contraction_certified": False,
        "scientific_acceptance": False,
        "status": "maximum_iterations_reached",
        "history": [],
    }
    left = None
    for iteration in range(1, max_iterations + 1):
        diagnostics["iterations"] = iteration
        try:
            left, forward_log = operator.scaled_jvp(vector)
            if left is None:
                diagnostics.update(status="direction_annihilated_at_working_precision", gain=0.0,
                                   log_gain=None, log_gain_per_update=None, relative_right_singular_residual=None,
                                   gain_representation="observed directional zero; nullspace or numerical underflow")
                break
            next_vector, backward_log = operator.scaled_vjp(left)
            if next_vector is None:
                diagnostics["status"] = "adjoint_breakdown_at_working_precision"
                left = None
                diagnostics.update(gain=None, log_gain=None, log_gain_per_update=None,
                                   relative_right_singular_residual=None, gain_representation="unavailable after adjoint breakdown")
                break
            ratio = math.exp(backward_log - forward_log)
            if not math.isfinite(ratio):
                raise FloatingPointError("nonfinite singular residual scale")
            residual_vector = next_vector * ratio - vector
            _, residual_log = _unit_and_log_norm(residual_vector)
            residual = 0.0 if residual_log is None else math.exp(residual_log)
            if not math.isfinite(residual):
                raise FloatingPointError("nonfinite singular residual")
        except (FloatingPointError, OverflowError):
            diagnostics["status"] = "nonfinite_tangent_or_residual"
            left = None
            diagnostics.update(gain=None, log_gain=None, log_gain_per_update=None,
                               relative_right_singular_residual=None, gain_representation="unavailable after action failure")
            break
        except (NotImplementedError, RuntimeError):
            # Torch transforms can lack a derivative rule. Preserve failure,
            # without inferring that the numerical spectrum is empty or zero.
            diagnostics["status"] = "autodiff_action_failed"
            left = None
            diagnostics.update(gain=None, log_gain=None, log_gain_per_update=None,
                               relative_right_singular_residual=None, gain_representation="unavailable after action failure")
            break
        diagnostics["log_gain"] = forward_log
        diagnostics["log_gain_per_update"] = forward_log / operator.length if operator.length else None
        try:
            gain = math.exp(forward_log)
        except OverflowError:
            gain = math.inf
        if math.isfinite(gain) and gain > 0:
            diagnostics["gain"], diagnostics["gain_representation"] = gain, "finite Python float; log_gain retained"
        else:
            diagnostics["gain"], diagnostics["gain_representation"] = None, "gain outside Python float range; log_gain retained"
        diagnostics["relative_right_singular_residual"] = residual
        diagnostics["history"].append({"iteration": iteration, "log_gain": forward_log, "relative_right_singular_residual": residual})
        if residual <= residual_tolerance:
            diagnostics.update(status="converged_singular_pair", converged=True, converged_pairs=1)
            break
        if iteration < max_iterations:
            vector = next_vector
    return SingularGainEstimate(initial, vector.detach().clone(), None if left is None else left.detach().clone(), diagnostics)
