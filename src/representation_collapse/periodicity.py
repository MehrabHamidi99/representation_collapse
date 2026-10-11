"""Checkpoint-only complete-state periodicity and finite-time tangent diagnostics.

Numerical closure is a diagnostic, never proof of an exact periodic orbit.
FP64 replay casts a copy of fixed weights before recurrence, not saved states.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from .pilot import VectorMemoryModel, atomic_json, atomic_npz, provenance, _manifest
from .vector_task import MOTIFS, body_from_ids, proper_period, render, sample_memory


def unpack_state(model: VectorMemoryModel, state: torch.Tensor):
    """Packed order is h[0:H], then c[H:2H], with one layer and one sample."""
    if state.ndim != 1 or state.numel() != model.hidden_size * (2 if model.model_kind == "lstm" else 1):
        raise ValueError("invalid complete-state shape")
    h = state[:model.hidden_size].reshape(1, 1, -1)
    return (h, state[model.hidden_size:].reshape(1, 1, -1)) if model.model_kind == "lstm" else h


def recurrent_step(model: VectorMemoryModel, state: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
    """Exact documented PyTorch gate conventions; frozen input, full-state AD."""
    encoded = model.encoder(input_)
    h = state[:model.hidden_size]
    gi = F.linear(encoded, model.core.weight_ih_l0, model.core.bias_ih_l0)
    gh = F.linear(h, model.core.weight_hh_l0, model.core.bias_hh_l0)
    if model.model_kind == "gru":
        ir, iz, inn = gi.chunk(3)
        hr, hz, hn = gh.chunk(3)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        return (1 - z) * n + z * h
    i, f, g, o = (gi + gh).chunk(4)
    c = torch.sigmoid(f) * state[model.hidden_size:] + torch.sigmoid(i) * torch.tanh(g)
    return torch.cat([torch.sigmoid(o) * torch.tanh(c), c])


def window_map(model, state, inputs):
    for input_ in inputs:
        state = recurrent_step(model, state, input_)
    return state


def dense_jacobian(model, state, input_):
    return torch.autograd.functional.jacobian(
        lambda z: recurrent_step(model, z, input_), state, vectorize=True)


def step_jacobians(model, start, inputs):
    state = start.detach()
    jacobians = []
    for input_ in inputs:
        jacobians.append(dense_jacobian(model, state, input_).detach())
        state = recurrent_step(model, state, input_).detach()
    return torch.stack(jacobians)


def tangent_window(model, start, inputs) -> dict[str, np.ndarray]:
    """J[t] acts first; returned product is J[T-1] ... J[0]."""
    steps = step_jacobians(model, start, inputs)
    # Multiply in replay dtype; separately retain float64 accumulation of saved J.
    product = torch.eye(start.numel(), dtype=start.dtype, device=start.device)
    for step in steps:
        product = step @ product
    direct = torch.autograd.functional.jacobian(
        lambda z: window_map(model, z, inputs), start.detach(), vectorize=True).detach()
    arrays = {"step_jacobians": steps.cpu().numpy(), "product": product.cpu().numpy(),
              "direct_jacobian": direct.cpu().numpy()}
    p64 = np.eye(start.numel())
    for step in arrays["step_jacobians"]:
        p64 = step.astype(np.float64) @ p64
    eigenvalues = np.linalg.eigvals(p64)
    arrays.update(product_float64_accumulation=p64,
                  singular_values=np.linalg.svd(p64, compute_uv=False),
                  eigenvalues_real=eigenvalues.real, eigenvalues_imag=eigenvalues.imag)
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError("nonfinite tangent result")
    return arrays


def distance(a, b, absolute, relative, floor=1e-12) -> dict:
    """Global Frobenius/L2 closure; require both absolute and relative bounds."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("closure comparison requires matching finite arrays")
    delta = float(np.linalg.norm(a - b))
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), floor)
    coordinate_delta = np.abs(a-b)
    coordinate_gate = absolute + relative*np.maximum(np.abs(a), np.abs(b))
    return {"absolute": delta, "relative": delta / scale, "reference_scale": scale,
            "maximum_coordinate_absolute": float(coordinate_delta.max()),
            "coordinatewise_passed": bool(np.all(coordinate_delta <= coordinate_gate)),
            "absolute_tolerance": absolute, "relative_tolerance": relative,
            "scale_floor": floor, "passed": delta <= absolute and delta / scale <= relative and bool(np.all(coordinate_delta <= coordinate_gate))}


def replay(model, inputs, *, atol=1e-6, rtol=1e-5, return_native=False):
    """Retain initial+every pre-query state; validate native step/full replay."""
    size = model.hidden_size * (2 if model.model_kind == "lstm" else 1)
    state = inputs.new_zeros(size)
    native_state = unpack_state(model, state)
    states = [state.detach().cpu().numpy()]
    native_steps = [states[0]]
    passed = True
    with torch.no_grad():
        for input_ in inputs[:-1]:
            state = recurrent_step(model, state, input_)
            _, native_state = model.core(model.encoder(input_)[None, None], native_state)
            native = model.pack_state(native_state)[0]
            passed = passed and bool(torch.allclose(state, native, atol=atol, rtol=rtol))
            states.append(state.cpu().numpy())
            native_steps.append(native.cpu().numpy())
        encoded = model.encoder(inputs[None, :-1])
        full_output, full_state = model.core(encoded)
        passed = passed and bool(torch.allclose(state, model.pack_state(full_state)[0], atol=atol, rtol=rtol))
        trajectory = torch.as_tensor(np.stack(states[1:]), device=inputs.device)[:, :model.hidden_size]
        passed = passed and bool(torch.allclose(trajectory, full_output[0], atol=atol, rtol=rtol))
        query_state = recurrent_step(model, state, inputs[-1])
        manual_prediction = model.head(query_state[:model.hidden_size])
        native_prediction = model(inputs[None])[0]
        passed = passed and bool(torch.allclose(manual_prediction, native_prediction, atol=atol, rtol=rtol))
    raw = np.stack(states)
    native_raw = np.stack(native_steps)
    if not np.isfinite(raw).all() or not np.isfinite(native_raw).all() or not np.isfinite(native_prediction.cpu().numpy()).all():
        raise FloatingPointError("nonfinite replay")
    checks = {"passed": passed, "manual_vs_native_step_max_absolute": float(np.max(np.abs(raw-native_raw))),
              "manual_vs_full_native_state_max_absolute": float(torch.max(torch.abs(state-model.pack_state(full_state)[0]))),
              "manual_vs_full_native_hidden_trajectory_max_absolute": float(torch.max(torch.abs(trajectory-full_output[0]))),
              "manual_vs_native_query_output_max_absolute": float(torch.max(torch.abs(manual_prediction-native_prediction))),
              "absolute_tolerance": atol, "relative_tolerance": rtol}
    result=raw,checks,native_prediction.cpu().numpy()
    native_record={"step_states":native_raw,
                   "full_pre_query_state":model.pack_state(full_state)[0].cpu().numpy(),
                   "full_hidden_trajectory":full_output[0].cpu().numpy()}
    return (*result,native_record) if return_native else result


def validate_config(config):
    if config["horizons"] != [1, 2, 16, 32] or config["derivative_horizons"] != [2, 16] or config.get("dwell", 1) != 1:
        raise ValueError("bounded diagnostic requires q1, horizons1/2/16/32, derivatives2/16")
    for key in ["cue_histories", "driver_panels", "derivative_cues", "cpu_threads"]:
        if not isinstance(config[key], int) or not 1 <= config[key] <= (4 if key == "cpu_threads" else 2):
            raise ValueError("invalid bounded count: " + key)
    if config["derivative_cues"] > config["cue_histories"] or config["precisions"] != ["float32", "float64"]:
        raise ValueError("both separate precision assays and available derivative cues required")
    for kind in ["float32", "float64"]:
        tolerances = config["tolerances"][kind]
        for name in ["state_absolute", "state_relative", "jacobian_absolute", "jacobian_relative", "adapter_absolute", "adapter_relative", "product_absolute", "product_relative", "scale_floor"]:
            if not np.isfinite(tolerances[name]) or tolerances[name] <= 0:
                raise ValueError("finite positive tolerance required: " + name)


def fixed_panels(seed, count, maximum=32):
    """Same final motif across horizons, no accidental tiling in control bodies."""
    rng = np.random.default_rng(seed)
    result = []
    for final in rng.choice(len(MOTIFS), size=count, replace=False):
        final = int(final)
        for _attempt in range(10000):
            control = rng.integers(len(MOTIFS), size=maximum)
            control[-1] = final
            if all(proper_period(body_from_ids(np.r_[control[:r-1], final])) is None for r in [2, 16, 32]):
                break
        else:
            raise RuntimeError("cannot construct matched control")
        result.append((final, control))
    return result


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config, checkpoint_path, output, model_kind, task, seed, source_commit,
        cluster="local", device="cpu", config_path=None, *, verify_source=True, checkpoint_sha256=None):
    """Read-only weights; exclusive output, fail-closed status and hashed artifacts."""
    config = deepcopy(config)
    validate_config(config)
    output, checkpoint_path = Path(output), Path(checkpoint_path)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw,trajectories,tangents,results={},{},{},{}
    atomic_json(output/"status.json", {"state":"running", "cluster":cluster, "scientific_acceptance":False})
    try:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("unavailable device")
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        if verify_source and (revision != source_commit or dirty):
            raise ValueError("diagnostic source must match clean immutable --source-commit")
        checkpoint_hash = _hash(checkpoint_path)
        if checkpoint_sha256 is None or checkpoint_hash != checkpoint_sha256:
            raise ValueError("checkpoint must match trusted --checkpoint-sha256 before loading")
        registry = config.get("cells", config.get("checkpoints"))
        if registry is not None:
            cell = next((c for c in registry if c["model"] == model_kind and c["task"] == task and c["seed"] == seed), None)
            if cell is None or checkpoint_hash != cell.get("checkpoint_sha256",cell.get("sha256")):
                raise ValueError("checkpoint is outside frozen registry")
            if cell.get("cluster",cluster)!=cluster or ("checkpoint" in cell and Path(cell["checkpoint"]).resolve()!=checkpoint_path.resolve()):
                raise ValueError("checkpoint path/cluster mismatch with frozen registry")
        parent = checkpoint_path.parent
        parent_manifest=json.loads((parent/"manifest.json").read_text())["sha256"]
        for name in ["checkpoint.pt","provenance.json","config.json","status.json"]:
            if parent_manifest.get(name)!=_hash(parent/name):raise ValueError("training manifest mismatch: "+name)
        training_origin=json.loads((parent/"provenance.json").read_text())
        expected_training=config.get("training_source_commit","9004a0c114a8b7c5335dca3d5247449fd10b5ffa")
        if training_origin["git_revision"]!=expected_training or training_origin["dirty_checkout"]:
            raise ValueError("training checkpoint source mismatch/dirty checkout")
        if json.loads((parent/"status.json").read_text())["state"]!="completed":raise ValueError("training run incomplete")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved = checkpoint["config"]
        if saved!=json.loads((parent/"config.json").read_text()):raise ValueError("checkpoint/config mismatch")
        if any(saved[k] != v for k,v in [("model",model_kind),("task",task),("seed",seed)]):
            raise ValueError("checkpoint model/task/seed mismatch")
        if saved["hidden_size"] > 128 or saved["input_bottleneck"] > 64:
            raise ValueError("checkpoint exceeds bounded dense-state budget")
        for key in ["hidden_size","input_bottleneck"]:
            if key in config and saved[key]!=config[key]:raise ValueError("checkpoint/config dimension mismatch: "+key)
        if saved["effective_training_steps"]!=1500 or saved["max_steps_override"] or not 1<=checkpoint["selected_step"]<=1500:
            raise ValueError("initial checkpoint is not the full-budget selected snapshot")
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        base = VectorMemoryModel(model_kind, task, saved["hidden_size"], saved["input_bottleneck"])
        base.load_state_dict(checkpoint["model_state_dict"], strict=True)
        base.eval()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        effective = {**config, "model":model_kind,"task":task,"seed":seed,"source_commit":source_commit,
                     "checkpoint_sha256":checkpoint_hash,"selected_step":checkpoint["selected_step"],
                     "cluster":cluster,"device":device,"scientific_acceptance":False,
                     "state_order":["h","c"] if model_kind=="lstm" else ["h"],
                     "state_dimension":saved["hidden_size"]*(2 if model_kind=="lstm" else 1),
                     "checkpoint_config":saved, "no_retraining":True,
                     "finite_difference":{"precision":"float64","epsilon":[1e-4,1e-5,1e-6],"absolute_tolerance":1e-8,"relative_tolerance":1e-4,"directions":1},
                     "trajectory_assays":"manual equation adapter and nativecore sequential states stored separately; tolerance agreement does not imply bitwise or cue-relative equivalence",
                     "allow_tf32":False,"float32_matmul_precision":"highest",
                     "tangent_assay":"autograd of manual PyTorch gate equations along manual-equation replay"}
        atomic_json(output/"config.json",effective)
        origin=provenance(config_path, device)
        origin.update(checkpoint_path=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,
                      diagnostic_source_commit=revision, source_verified=verify_source,
                      training_origin=training_origin,diagnostic_module_sha256=_hash(__file__),
                      cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                      cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                      float32_matmul_precision=torch.get_float32_matmul_precision(),
                      precision="FP32 native replay and FP64 cast-copy fixed-weight replay; derivatives in assay dtype")
        atomic_json(output/"provenance.json",origin)
        memory_rng=np.random.default_rng(seed+config.get("sampling_seed_offset",60000))
        panel_memories=[]
        for panel in range(config["driver_panels"]):
            memory=sample_memory(memory_rng,config["cue_histories"],task)
            memory[:,8:]=memory[0,8:]
            panel_memories.append(memory)
            raw[f"panel{panel}_memory"]=memory
            raw[f"panel{panel}_query_targets"]=memory[:,0].copy()
        raw["query_coordinates"]=np.zeros(config["cue_histories"],dtype=np.int64)
        panels = fixed_panels(seed+config.get("driver_seed_offset",70000),config["driver_panels"])
        numerical_failures=[]
        runtime_limit=min(840,config.get("cluster_resources",{}).get("walltime_minutes",15)*60-60)
        if runtime_limit<1:raise ValueError("invalid runtime cap")
        for precision,dtype in [("float32",torch.float32),("float64",torch.float64)]:
            model=deepcopy(base).to(device=device,dtype=dtype)
            tol=config["tolerances"][precision]
            for panel,(final,control) in enumerate(panels):
                memory=panel_memories[panel]
                for repeats in config["horizons"]:
                    periodic_ids=np.full(repeats,final,dtype=np.int64)
                    control_ids=np.r_[control[:repeats-1],final].astype(np.int64)
                    for arm,ids in [("periodic",periodic_ids),("control",control_ids)]:
                        key=f"{precision}_panel{panel}_R{repeats}_{arm}"
                        rendered=render(memory,np.zeros(len(memory),dtype=np.int64),[body_from_ids(ids)]*len(memory),task)
                        raw[key+"_inputs"]=rendered
                        raw[key+"_motif_ids"]=ids
                        driver=rendered[0,16:-1]
                        driver_closed=bool(repeats>=2 and np.array_equal(driver[:-12],driver[12:]))
                        packet={"driver_periodic_over_observed_body":driver_closed,"body_steps":len(driver),
                                "primitive_motif_steps":12,"query_included_in_dynamics":False,"cues":[]}
                        for cue in range(len(memory)):
                            if time.monotonic()-started>runtime_limit:raise TimeoutError("bounded runtime cap reached; partial diagnostic retained")
                            inputs=torch.as_tensor(rendered[cue],dtype=dtype,device=device)
                            states,adapter,prediction,native_record=replay(model,inputs,atol=tol["adapter_absolute"],rtol=tol["adapter_relative"],return_native=True)
                            ckey=key+f"_cue{cue}"
                            trajectories[ckey+"_states"]=states
                            trajectories[ckey+"_native_core_states"]=native_record["step_states"]
                            trajectories[ckey+"_native_full_pre_query_state"]=native_record["full_pre_query_state"]
                            trajectories[ckey+"_native_full_hidden_trajectory"]=native_record["full_hidden_trajectory"]
                            trajectories[ckey+"_native_query_prediction"]=prediction
                            body_states=states[16:]
                            cycle_abs=[];cycle_rel=[]
                            for cycle in range(repeats-1):
                                pair=distance(body_states[cycle*12:(cycle+1)*12+1],body_states[cycle*12+12:(cycle+2)*12+1],tol["state_absolute"],tol["state_relative"],tol["scale_floor"])
                                cycle_abs.append(pair["absolute"]);cycle_rel.append(pair["relative"])
                            trajectories[ckey+"_cycle_state_closure_absolute"]=np.asarray(cycle_abs)
                            trajectories[ckey+"_cycle_state_closure_relative"]=np.asarray(cycle_rel)
                            cue_packet={"adapter":adapter,"measurement":"manual equation complete-state replay; nativecore trajectory separately retained","tangent":None}
                            if not adapter["passed"]:numerical_failures.append(ckey+":native_adapter")
                            if repeats in config["derivative_horizons"] and cue<config["derivative_cues"]:
                                phase=(repeats-1)*12
                                current_start=torch.as_tensor(body_states[phase],dtype=dtype,device=device)
                                previous_start=torch.as_tensor(body_states[phase-12],dtype=dtype,device=device)
                                current_inputs=inputs[16+phase:16+phase+12]
                                previous_inputs=inputs[16+phase-12:16+phase]
                                current=tangent_window(model,current_start,current_inputs)
                                previous_j=step_jacobians(model,previous_start,previous_inputs).detach().cpu().numpy()
                                for name,value in current.items():tangents[ckey+"_"+name]=value
                                tangents[ckey+"_previous_step_jacobians"]=previous_j
                                closure=distance(body_states[phase-12:phase+1],body_states[phase:phase+13],tol["state_absolute"],tol["state_relative"],tol["scale_floor"])
                                jclosure=distance(previous_j,current["step_jacobians"],tol["jacobian_absolute"],tol["jacobian_relative"],tol["scale_floor"])
                                agreement=distance(current["product"],current["direct_jacobian"],tol["product_absolute"],tol["product_relative"],tol["scale_floor"])
                                if not agreement["passed"]:numerical_failures.append(ckey+":product_direct")
                                compatible=arm=="periodic" and driver_closed and closure["passed"] and jclosure["passed"]
                                cue_packet["tangent"]={"interpretation":"finite_time_one_motif_propagator",
                                    "numerical_closure_compatible":compatible,"exact_orbit_verified":False,"floquet_acceptance":False,
                                    "state_closure":closure,"jacobian_closure":jclosure,"product_direct_agreement":agreement,
                                    "window_body_start":phase,"window_steps":12,"step_order":"J[11] @ ... @ J[0]",
                                    "spectra_from":"float64 ordered accumulation of saved assay-dtype step Jacobians",
                                    "spectral_radius":float(np.hypot(current["eigenvalues_real"],current["eigenvalues_imag"]).max()),
                                    "largest_singular_value":float(current["singular_values"][0])}
                                if precision=="float64":
                                    direction=torch.linspace(-1,1,current_start.numel(),device=device,dtype=dtype)
                                    if current_start.numel()==1:direction=torch.ones_like(direction)
                                    direction=direction/torch.linalg.vector_norm(direction)
                                    _,jvp=torch.autograd.functional.jvp(lambda z:window_map(model,z,current_inputs),current_start,direction)
                                    epsilon=np.asarray([1e-4,1e-5,1e-6])
                                    with torch.no_grad():
                                        fd=torch.stack([(window_map(model,current_start+float(e)*direction,current_inputs)-window_map(model,current_start-float(e)*direction,current_inputs))/(2*float(e)) for e in epsilon])
                                    expected=current["direct_jacobian"]@direction.cpu().numpy()
                                    fd_array=fd.cpu().numpy()
                                    tangents[ckey+"_fd_epsilon"]=epsilon
                                    tangents[ckey+"_fd_direction"]=direction.cpu().numpy()
                                    tangents[ckey+"_fd_derivatives"]=fd_array
                                    tangents[ckey+"_autograd_jvp"]=jvp.detach().cpu().numpy()
                                    fd_checks=[distance(v,expected,1e-8,1e-4,tol["scale_floor"]) for v in fd_array]
                                    jvp_check=distance(jvp.detach().cpu().numpy(),expected,tol["product_absolute"],tol["product_relative"],tol["scale_floor"])
                                    cue_packet["tangent"]["directional_finite_difference"]={"epsilon":epsilon.tolist(),"checks":fd_checks,"jvp_agreement":jvp_check,
                                        "passed":all(c["passed"] for c in fd_checks) and jvp_check["passed"],"scope":"one deterministic normalized direction; local numerical check"}
                                    if not all(c["passed"] for c in fd_checks) or not jvp_check["passed"]:numerical_failures.append(ckey+":finite_difference_jvp")
                            packet["cues"].append(cue_packet)
                        if len(memory)==2:
                            first=trajectories[key+"_cue0_states"].astype(np.float64)
                            second=trajectories[key+"_cue1_states"].astype(np.float64)
                            separation=np.linalg.norm(first-second,axis=1)
                            trajectories[key+"_paired_cue_separation_l2"]=separation
                            native_first=trajectories[key+"_cue0_native_core_states"].astype(np.float64)
                            native_second=trajectories[key+"_cue1_native_core_states"].astype(np.float64)
                            native_separation=np.linalg.norm(native_first-native_second,axis=1)
                            trajectories[key+"_native_paired_cue_separation_l2"]=native_separation
                            packet["paired_cue_separation"]={"after_write":float(separation[16]),"pre_query":float(separation[-1]),
                                "native_after_write":float(native_separation[16]),"native_pre_query":float(native_separation[-1]),
                                "scope":"two fixed required-history samples; shared nuisance; no covariance/population dimension claim"}
                            if model_kind=="lstm":
                                for component,sl in [("h",slice(0,model.hidden_size)),("c",slice(model.hidden_size,None))]:
                                    trajectories[key+"_paired_cue_separation_"+component]=np.linalg.norm((first-second)[:,sl],axis=1)
                                    trajectories[key+"_native_paired_cue_separation_"+component]=np.linalg.norm((native_first-native_second)[:,sl],axis=1)
                        packet["observed_adjacent_cycle_comparisons"]=repeats-1
                        results[key]=packet
        metrics={"schema_version":1,"kind":"checkpoint_periodicity_diagnostic","scientific_acceptance":False,
                 "native_competence_established":False,"no_retraining":True,"assays":results,
                 "numerical_checks_passed":not numerical_failures,"numerical_failures":numerical_failures,
                 "dense_windows":sum(k.endswith("_direct_jacobian") for k in tangents),"tangent_derivative_cues":config["derivative_cues"],
                 "step_jacobian_evaluations":sum(v.shape[0] for k,v in tangents.items() if k.endswith("step_jacobians")),
                 "query_readout_scope":"coordinate0 only; original head unchanged; no memory competence assay",
                 "runtime_seconds":time.monotonic()-started,
                 "limitations":"Selected failed-competence checkpoints; no exact orbit, causal collapse, training replication or precision equivalence claim."}
        atomic_npz(output/"inputs.npz",raw);atomic_npz(output/"trajectories.npz",trajectories);atomic_npz(output/"tangents.npz",tangents)
        atomic_json(output/"metrics.json",metrics)
        if _hash(checkpoint_path)!=checkpoint_hash:raise ValueError("checkpoint changed during diagnostic")
        if numerical_failures:raise ValueError("numerical checks failed; raw diagnostic quarantined")
        atomic_json(output/"status.json",{"state":"completed","cluster":cluster,"scientific_acceptance":False,"finished_at":datetime.now(timezone.utc).isoformat()})
        _manifest(output)
        return metrics
    except Exception as error:
        for name,values in [("inputs",raw),("trajectories",trajectories),("tangents",tangents)]:
            if values:atomic_npz(output/(name+".npz"),values)
        atomic_json(output/"status.json",{"state":"failed","cluster":cluster,"scientific_acceptance":False,"error_type":type(error).__name__,"error":str(error)})
        _manifest(output)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["config","checkpoint","output"]:parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--model",choices=["gru","lstm"],required=True)
    parser.add_argument("--task",choices=["categorical","continuous"],required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--checkpoint-sha256",required=True)
    parser.add_argument("--cluster",required=True)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    metrics=run(json.loads(args.config.read_text()),args.checkpoint,args.output,args.model,args.task,args.seed,args.source_commit,args.cluster,args.device,args.config,checkpoint_sha256=args.checkpoint_sha256)
    print(json.dumps({"output":str(args.output),"assays":len(metrics["assays"]),"scientific_acceptance":False}))


if __name__=="__main__":main()
