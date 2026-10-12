"""Complete native recurrent state at WRITE/body/QUERY boundaries.

This is a chronology and adapter-equivalence assay for the frozen one-layer
vector model. Native PyTorch GRU/LSTM kernels perform every update. Numerical
agreement is not a cue-relative equivalence test and certifies neither a
periodic orbit, Floquet analysis, useful dimension nor harmful collapse.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch
from torch import nn

from .vector_task import A, B, X, INPUT_DIM, QUERY, WRITE

TOLERANCES = {
    torch.float32: {"absolute": 1e-6, "relative": 1e-5},
    torch.float64: {"absolute": 1e-12, "relative": 1e-10},
}


def _tensor_hash(value):
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _weights_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_hash(value).encode())
    return digest.hexdigest()


def _packed(state, kind):
    if kind == "lstm":
        hidden, cell = state
        return torch.cat((hidden[0], cell[0]), dim=1)
    return state[0]


def _copy_state(state):
    return tuple(value.detach().clone() for value in state) if isinstance(state, tuple) else state.detach().clone()


def _agreement(left, right, tolerance):
    if left.shape != right.shape:
        raise ValueError("native adapter comparison has mismatched shapes")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("native state/logit capture is nonfinite")
    delta = torch.abs(left - right)
    coordinate_tolerance = tolerance["absolute"] + tolerance["relative"] * torch.abs(right)
    passed = bool(torch.all(delta <= coordinate_tolerance))
    record = {"passed": passed, "max_absolute_difference": float(delta.max()),
              "changed_coordinates": int(torch.count_nonzero(left != right)),
              "coordinates": left.numel(), "bitwise_equal": bool(torch.equal(left, right)),
              "absolute_tolerance": tolerance["absolute"], "relative_tolerance": tolerance["relative"],
              "comparison": "elementwise |candidate-reference| <= atol + rtol*|reference|"}
    if not passed:
        raise FloatingPointError("native chronology/continuation agreement failed: " + json.dumps(record))
    return record


def _validate(model, inputs, write_count):
    if isinstance(write_count, bool) or not isinstance(write_count, int) or write_count not in (1, 16):
        raise ValueError("write_count must be 1 (immediate) or 16 (complete prefix)")
    if not isinstance(inputs, torch.Tensor) or inputs.dtype not in TOLERANCES:
        raise ValueError("native inputs must be a float32/float64 Torch tensor")
    if inputs.ndim != 3 or inputs.shape[0] < 1 or inputs.shape[2] != INPUT_DIM or inputs.shape[1] <= write_count:
        raise ValueError("nonempty [batch, WRITE+body+QUERY,38] inputs required")
    if not torch.isfinite(inputs).all():
        raise ValueError("native inputs must be finite")
    kind = getattr(model, "model_kind", None)
    task = getattr(model, "task", None)
    if kind not in ("gru", "lstm") or task not in ("categorical", "continuous"):
        raise ValueError("the one-layer vector GRU/LSTM model and task identity are required")
    core = getattr(model, "core", None)
    expected = nn.LSTM if kind == "lstm" else nn.GRU
    if not isinstance(core, expected) or core.num_layers != 1 or not core.batch_first or core.bidirectional or core.dropout != 0:
        raise ValueError("capture supports one-layer, batch-first, unidirectional native GRU/LSTM without dropout")
    if getattr(core, "proj_size", 0) != 0:
        raise ValueError("projected LSTM state is outside this complete-state contract")
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Sequential) or len(encoder) != 2 or not isinstance(encoder[0], nn.Linear) or not isinstance(encoder[1], nn.Tanh):
        raise ValueError("the frozen pointwise Linear/Tanh encoder is required")
    if not isinstance(getattr(model, "head", None), nn.Linear):
        raise ValueError("the native vector task head is required")
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != inputs.dtype or p.device != inputs.device for p in parameters):
        raise ValueError("all native model parameters must share the inputs' dtype and device")
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("native model parameters must be finite")
    types = inputs[..., :5]
    if not torch.all((types == 0) | (types == 1)) or not torch.all(types.sum(dim=-1) == 1):
        raise ValueError("every observation requires one-hot WRITE/A/B/X/QUERY type")
    symbols = types.argmax(dim=-1)
    if not torch.all(symbols[:, :write_count] == WRITE) or not torch.all(symbols[:, -1] == QUERY):
        raise ValueError("capture requires initial WRITEs and one terminal QUERY")
    body = symbols[:, write_count:-1]
    if not torch.all((body == A) | (body == B) | (body == X)):
        raise ValueError("only A/B/X updates may separate WRITE prefix and QUERY")
    if write_count == 1 and inputs.shape[1] != 2:
        raise ValueError("the immediate diagnostic has exactly one WRITE and one QUERY")
    addresses = inputs[..., 5:21]
    used = torch.cat((addresses[:, :write_count], addresses[:, -1:]), dim=1)
    if not torch.all((used == 0) | (used == 1)) or not torch.all(used.sum(dim=-1) == 1):
        raise ValueError("WRITE/QUERY addresses must be one-hot")
    if torch.any(addresses[:, -1].argmax(dim=-1) >= 8):
        raise ValueError("QUERY must select a required coordinate 0..7")
    if write_count == 16 and not torch.all(addresses[:, :16].argmax(dim=-1) == torch.arange(16, device=inputs.device)):
        raise ValueError("the complete prefix requires the sixteen ordered WRITEs")
    if write_count == 1 and torch.any(addresses[:, 0].argmax(dim=-1) >= 8):
        raise ValueError("immediate WRITE must use a required coordinate")
    if write_count == 1 and not torch.equal(addresses[:, 0], addresses[:, -1]):
        raise ValueError("immediate QUERY must address the single written coordinate")
    if torch.any(addresses[:, write_count:-1] != 0) or torch.any(inputs[:, write_count:, 21:] != 0):
        raise ValueError("body/QUERY may not contain memory values or body addresses")
    values = inputs[:, :write_count, 21:37]
    real = inputs[:, :write_count, 37]
    if task == "categorical":
        if not torch.all((values == 0) | (values == 1)) or not torch.all(values.sum(dim=-1) == 1) or torch.any(real != 0):
            raise ValueError("categorical WRITE requires one-hot class payload")
    elif torch.any(values != 0) or torch.any(torch.abs(real) > 1):
        raise ValueError("continuous WRITE requires only a real payload in [-1,1]")
    return kind


@torch.no_grad()
def native_phase_capture(model, inputs: torch.Tensor, write_count: int) -> dict:
    """Capture detached complete states and verify native query continuation.

    Returns packed ``post_write/pre_query/post_query`` tensors of shape [B,H]
    for GRU and [B,2H] for LSTM (hidden then cell). ``phase_states`` references
    those same output tensors; ``component_states`` contains independent copies
    of each hidden/cell component. ``native_logits`` equals uninterrupted
    ``model(inputs)``; ``forked_logits`` applies the original head only after
    the terminal native query update, forked from the copied pre-query state.
    No targets, original cues or decoder are accepted by this interface.

    Module train/eval flags are restored even if a check fails. Dtype, measured
    kernel flags, input/weight/source hashes, chronology and numerical residuals
    are in JSON-compatible ``chronology``. Tolerances are declared above and
    are adapter tolerances, not a noise floor or evidence of state periodicity.
    """
    kind = _validate(model, inputs, write_count)
    tolerance = TOLERANCES[inputs.dtype]
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        encoded = model.encoder(inputs)
        _, write_state = model.core(encoded[:, :write_count])
        if inputs.shape[1] > write_count + 1:
            _, pre_state = model.core(encoded[:, write_count:-1], _copy_state(write_state))
        else:
            pre_state = _copy_state(write_state)
        fork_output, fork_state = model.core(encoded[:, -1:], _copy_state(pre_state))
        forked_logits = model.head(fork_output[:, -1])
        uninterrupted_output, uninterrupted_state = model.core(encoded)
        uninterrupted_logits = model(inputs)
        full_core_logits = model.head(uninterrupted_output[:, -1])
        _, full_pre_state = model.core(encoded[:, :-1])
        checks = {
            "segmented_vs_uninterrupted_pre_query_state": _agreement(_packed(pre_state, kind), _packed(full_pre_state, kind), tolerance),
            "forked_vs_uninterrupted_post_query_state": _agreement(_packed(fork_state, kind), _packed(uninterrupted_state, kind), tolerance),
            "forked_vs_uninterrupted_logits": _agreement(forked_logits, uninterrupted_logits, tolerance),
            "native_core_head_vs_model_forward_logits": _agreement(full_core_logits, uninterrupted_logits, tolerance),
        }
        if callable(getattr(model, "read_query", None)):
            read_logits = model.read_query(inputs[:, -1], _copy_state(pre_state))
            checks["native_read_query_vs_forked_logits"] = _agreement(read_logits, forked_logits, tolerance)
        states = {"post_write": write_state, "pre_query": pre_state, "post_query": fork_state}
        phase_states, component_states = {}, {}
        for phase, state in states.items():
            packed = _packed(state, kind)
            if not torch.isfinite(packed).all():
                raise FloatingPointError("native complete state is nonfinite")
            if callable(getattr(model, "pack_state", None)):
                checks[phase + "_model_pack_preserves_complete_state"] = _agreement(model.pack_state(state), packed, tolerance)
            phase_states[phase] = packed.detach().clone()
            hidden = state[0] if kind == "lstm" else state
            component_states[phase] = {"hidden": hidden[0].detach().clone()}
            if kind == "lstm":
                component_states[phase]["cell"] = state[1][0].detach().clone()
        sources = {"capture_module": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        try:
            model_file = Path(inspect.getfile(type(model)))
            sources["native_model_module"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
        except (OSError, TypeError):
            sources["native_model_module"] = None
        chronology = {
            "schema_version": 1, "kind": "native_three_phase_chronology", "passed": True,
            "scientific_acceptance": False, "floquet_certified": False,
            "model": kind, "task": model.task, "dtype": str(inputs.dtype), "device": str(inputs.device),
            "batch_size": inputs.shape[0], "sequence_updates": inputs.shape[1],
            "write_updates": write_count, "body_updates": inputs.shape[1] - write_count - 1,
            "phase_after_update_indices_zero_based": {"post_write": write_count - 1, "pre_query": inputs.shape[1] - 2, "post_query": inputs.shape[1] - 1},
            "native_query_update_included": True, "fork_origin": "copied_complete_pre_query_state",
            "state_dimension": phase_states["pre_query"].shape[1],
            "packing_order": "hidden_then_cell" if kind == "lstm" else "hidden",
            "absolute_tolerance": tolerance["absolute"], "relative_tolerance": tolerance["relative"],
            "tolerance_scope": "native execution adapter agreement; not bitwise or cue-relative equivalence",
            "checks": checks, "inputs_sha256": _tensor_hash(inputs), "weights_sha256": _weights_hash(model),
            "source_sha256": sources, "torch_version": str(torch.__version__),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        return {**phase_states, "phase_states": phase_states, "component_states": component_states,
                "components": ["hidden", "cell"] if kind == "lstm" else ["hidden"],
                "forked_logits": forked_logits.detach().clone(),
                "uninterrupted_logits": uninterrupted_logits.detach().clone(),
                "native_logits": uninterrupted_logits.detach().clone(), "chronology": chronology}
    finally:
        for module, training in previous_modes:
            module.training = training
