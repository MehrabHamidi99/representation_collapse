"""Query-averaged original-task acquisition, with committed source admission.

Training forks retain the entire autograd graph and LSTM h+c. Diagnostic native
capture is used only for evaluation. No task, head or old protocol is changed.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from .address_acquisition import reference_preflight
from .competence import R0, NATIVE_PHASE, _batch_hash, make_competence_batch, evaluate_competence
from .diagnostics import paired_batch_with_zero
from .native_state import native_phase_capture
from .pilot import VectorMemoryModel, atomic_checkpoint, atomic_json, atomic_npz
from .vector_task import render
from .prefix_acquisition import batch_identity, _relative, _sha, _weight_hash, _hash_outputs

PROTOCOL="query-averaged-acquisition-v5"
CELLS=(("gru","categorical"),("gru","continuous"),("lstm","categorical"),("lstm","continuous"))
TOL={torch.float32:(1e-6,1e-5),torch.float64:(1e-12,1e-10)}
class DeadlineReached(RuntimeError):pass
def _check(deadline):
    if time.monotonic()>=deadline:raise DeadlineReached("overall deadline; endpoint censored")
def _now():return datetime.now(timezone.utc).isoformat()
def _pack(state,kind):return torch.cat((state[0][0],state[1][0]),1) if kind=="lstm" else state[0]
def _copy_graph(state):return tuple(v.clone() for v in state) if isinstance(state,tuple) else state.clone()
def _loss(logits,targets,task):return F.cross_entropy(logits,targets) if task=="categorical" else F.mse_loss(logits[:,0],targets)
def _gradvector(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(grads,params)])

def validate_config(c):
    expected={"protocol_id":PROTOCOL,"models":["gru","lstm"],"tasks":["categorical","continuous"],"seed":0,"learning_rate":.001,
        "model_source_commit":"999d88d8679c0f353c6bb7fb4f5ab8433a7e730c","dimensions":{"hidden_size":128,"input_bottleneck":64,"batch_size":32},
        "optimizer":{"name":"AdamW","weight_decay":.01,"gradient_clip":1.,"new_optimizer_per_fit":True,"clip_once_after_mean_loss_backward":True,"updates_per_batch":1},
        "device":"cpu","cpu_threads":1,"precision":"float32; deterministic_algorithms; CUDA/cuDNN TF32off; matmul_precision highest",
        "training_stream_seed":2026101301,"required_queries":list(range(8)),"validation_sampling_seed":20005,
        "validation_episodes_per_coordinate_per_arm":128,"categorical_accuracy_gate":.95,"continuous_population_normalized_mse_gate":.05,
        "population_variance":1/3,"comparison_ulps_float64":4,"evaluation_every_updates":500,"updates_per_fit":5000,
        "maximum_fits":4,"maximum_total_optimizer_updates":20000,"maximum_total_wall_seconds":1200,
        "scientific_acceptance":False,"confirmation":False,"cluster_submissions":0,"provider_calls":0}
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError("v5 scientific/execution contract changed")
    for k in ("cpu_threads","training_stream_seed","validation_sampling_seed","updates_per_fit","evaluation_every_updates","maximum_fits","maximum_total_optimizer_updates","maximum_total_wall_seconds"):
        if type(c[k]) is not int:raise ValueError("integer execution bounds required")
    if type(c["seed"]) is not int:raise ValueError("integer seed required")
    refs=c.get("v3_fresh_references",[]);streams=c.get("v3_base_training_streams",[])
    if len(refs)!=4 or {(r.get("model"),r.get("task")) for r in refs}!=set(CELLS) or any(r.get("condition")!="fresh_seed0" or r.get("endpoint_updates")!=5000 or r.get("source_commit")!=c["model_source_commit"] for r in refs):raise ValueError("all four pinned original v3 fresh references required")
    if len(streams)!=2 or {p.get("task") for p in streams}!={"categorical","continuous"} or any(p.get("updates")!=5000 or p.get("sampling_seed")!=2026101301 for p in streams):raise ValueError("both exact original v3 training stream pins required")
    if c.get("exposure")!={"arm_instances_per_update":32,"branch_loss_evaluations_per_update":256,"equal_optimizer_update_and_prefix_exposure":True,"equal_query_supervision_or_compute":False,"independent_prefix_histories_per_update":16,"native_row_updates_per_update":768,"nominal_recurrence_ratio":"24/17","queries_per_prefix":8,"v3_row_updates_per_update":544}:raise ValueError("declared unequal QUERY/compute exposure required")
    if c.get("pre_fit_calibration")!={"model_dimensions":"actual128/64/batch32","dtypes":["float32","float64"],"reference":"mean8 independent fullnative17update trajectories; compareeverytheta gradient andcompletequery state","fp32_atol":1e-6,"fp32_rtol":1e-5,"fp64_atol":1e-12,"fp64_rtol":1e-10,"negative_controls":["detach_prefix","sequential_query_chaining","omit_lstm_cell","missing_mean_normalization"],"optimizer_updates":0}:raise ValueError("mandatory actual-helper calibration changed")

def derive_all_queries(base,task):
    """Consume no randomness; old sampled QUERY remains provenance only."""
    x=base.inputs
    if task not in ("categorical","continuous") or x.shape!=(32,17,38) or x.dtype!=np.float32 or base.memory.shape!=(16,16) or base.queries.shape!=(16,):raise ValueError("frozen paired batch32/16WRITE chronology required")
    # Full original rendering is an independent oracle for every input feature,
    # including event/address one-hotness and the absence of QUERY value leakage.
    expected=render(np.concatenate((base.memory,base.memory)),np.concatenate((base.queries,base.queries)),[""]*32,task)
    values=base.memory[np.arange(16),base.queries]
    if base.targets.shape!=(32,) or not np.array_equal(x,expected) or not np.array_equal(base.targets,np.concatenate((values,values))):raise ValueError("original R0 rendering/target oracle mismatch or QUERY payload leakage")
    queries=np.zeros((8,32,38),dtype=np.float32)
    for q in range(8):queries[q,:,4]=1;queries[q,:,5+q]=1
    targets=x[:,:8,21:37].argmax(2).T.astype(np.int64) if task=="categorical" else x[:,:8,37].T.copy()
    if not np.array_equal(x[:16,:16],x[16:,:16]):raise ValueError("paired R0 histories must agree")
    return {"prefix":x[:,:16].copy(),"queries":queries,"targets":targets,
        "original_sampled_queries":base.queries.copy(),"original_sampled_targets":base.targets.copy()}

def query_identity(arrays):
    h=hashlib.sha256()
    for name in ("prefix","queries","targets","original_sampled_queries","original_sampled_targets"):
        a=np.ascontiguousarray(arrays[name]);h.update(json.dumps({"name":name,"dtype":a.dtype.str,"shape":list(a.shape)},sort_keys=True).encode());h.update(a.tobytes())
    return h.hexdigest()
def _tensors(arrays,task,dtype):
    return (torch.as_tensor(arrays["prefix"],dtype=dtype,device="cpu"),torch.as_tensor(arrays["queries"],dtype=dtype,device="cpu"),
        torch.as_tensor(arrays["targets"],dtype=torch.long if task=="categorical" else dtype,device="cpu"))

def average_query_loss(model,prefix,queries,targets):
    """Correct differentiable fork; no detached training state or branch chaining."""
    if prefix.shape!=(32,16,38) or queries.shape!=(8,32,38) or targets.shape!=(8,32):raise ValueError("frozen8x32 query graph geometry required")
    dtype=next(model.parameters()).dtype
    if prefix.device.type!="cpu" or queries.device.type!="cpu" or targets.device.type!="cpu" or prefix.dtype!=dtype or queries.dtype!=dtype or targets.dtype!=(torch.long if model.task=="categorical" else dtype):raise ValueError("CPU inputs and native target units/dtypes required")
    _,origin=model.core(model.encoder(prefix))
    if any(not torch.isfinite(v).all() for v in (origin if isinstance(origin,tuple) else (origin,))):raise FloatingPointError("nonfinite complete prefix state")
    logits,states,losses=[],[],[]
    for q in range(8):
        output,state=model.core(model.encoder(queries[q][:,None]),_copy_graph(origin))
        prediction=model.head(output[:,-1]);logits.append(prediction);states.append(_pack(state,model.model_kind));losses.append(_loss(prediction,targets[q],model.task))
    losses=torch.stack(losses);mean=losses.mean()
    if not torch.isfinite(mean):raise FloatingPointError("nonfinite averaged query loss")
    return mean,{"logits":torch.stack(logits),"states":torch.stack(states),"losses":losses,"prefix_state":_pack(origin,model.model_kind)}

def _wrong_loss(model,prefix,queries,targets,kind):
    """Intentionally invalid zero-update calibration controls; never train."""
    _,origin=model.core(model.encoder(prefix));state=origin
    if kind=="detach_prefix":state=tuple(v.detach() for v in state) if isinstance(state,tuple) else state.detach()
    if kind=="omit_lstm_cell":state=(state[0],torch.zeros_like(state[1]))
    losses=[]
    for q in range(8):
        out,post=model.core(model.encoder(queries[q][:,None]),_copy_graph(state));losses.append(_loss(model.head(out[:,-1]),targets[q],model.task))
        if kind=="sequential_query_chaining":state=post
    total=torch.stack(losses)
    return total.sum() if kind=="missing_mean_normalization" else total.mean()

def _agree(a,b,dtype):
    atol,rtol=TOL[dtype];gap=torch.abs(a-b)
    passed=bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.all(gap<=atol+rtol*torch.abs(b)))
    if not passed:raise FloatingPointError("actual-helper gradient/native calibration mismatch")
    return {"passed":passed,"max_absolute_difference":float(gap.max()),"atol":atol,"rtol":rtol,"entries":a.numel()}

def calibrate_helper(initial,base,output,deadline):
    """Actual dimensions, same FP32 theta/input represented in FP64; zero steps."""
    if initial.hidden_size!=128 or initial.core.input_size!=64 or next(initial.parameters()).dtype!=torch.float32:raise ValueError("actual128/64 model initialized in FP32 required")
    output=Path(output);output.mkdir(exist_ok=False);arrays=derive_all_queries(base,initial.task);records=[];mean_grads={}
    for dtype in (torch.float32,torch.float64):
        _check(deadline);model=deepcopy(initial).to(device="cpu",dtype=dtype);reference=deepcopy(initial).to(device="cpu",dtype=dtype)
        prefix,queries,targets=_tensors(arrays,model.task,dtype);params=list(model.parameters());reference_params=list(reference.parameters())
        qlogits,qstates,qloss,qgrad=[],[],[],[]
        for q in range(8):
            _check(deadline);captured=[]
            hook=reference.core.register_forward_hook(lambda _m,_i,o:captured.append(_pack(o[1],reference.model_kind)))
            try:pred=reference(torch.cat((prefix,queries[q][:,None]),1))
            finally:hook.remove()
            loss=_loss(pred,targets[q],model.task);g=_gradvector(torch.autograd.grad(loss,reference_params),reference_params)
            qlogits.append(pred.detach());qstates.append(captured[0].detach());qloss.append(loss.detach());qgrad.append(g.detach())
        expected=torch.stack(qgrad).mean(0);mean,branch=average_query_loss(model,prefix,queries,targets)
        actual=_gradvector(torch.autograd.grad(mean,params),params);mean_grads[dtype]=actual.detach()
        checks={"mean_loss":_agree(mean.detach().reshape(1),torch.stack(qloss).mean().reshape(1),dtype),
            "query_logits":_agree(branch["logits"].detach(),torch.stack(qlogits),dtype),
            "complete_query_states":_agree(branch["states"].detach(),torch.stack(qstates),dtype),"all_theta_gradient":_agree(actual,expected,dtype)}
        parameter_records=[];offset=0
        for name,param in model.named_parameters():
            end=offset+param.numel();checks["gradient:"+name]=_agree(actual[offset:end],expected[offset:end],dtype)
            if float(expected[offset:end].norm())==0:raise ValueError("zero reference parameter gradient cannot certify coverage")
            parameter_records.append({"name":name,"shape":list(param.shape),"start":offset,"end":end,"reference_l2_norm":float(expected[offset:end].norm())});offset=end
        negative={};negative_arrays={}
        for kind in ("detach_prefix","sequential_query_chaining","missing_mean_normalization")+(("omit_lstm_cell",) if model.model_kind=="lstm" else ()):
            _check(deadline);wrong=deepcopy(initial).to(device="cpu",dtype=dtype);wp=list(wrong.parameters())
            g=_gradvector(torch.autograd.grad(_wrong_loss(wrong,prefix,queries,targets,kind),wp,allow_unused=True),wp)
            atol,rtol=TOL[dtype];detected=not bool(torch.all(torch.abs(g-expected)<=atol+rtol*torch.abs(expected)))
            if not detected:raise FloatingPointError("invalid fork control was not detected")
            negative[kind]={"detected":True,"gradient_max_absolute_difference":float(torch.abs(g-expected).max())};negative_arrays[kind+"_gradient"]=g.detach().numpy()
        name=str(dtype).split(".")[-1];atomic_npz(output/(name+".npz"),{**arrays,"fork_prefix_state":branch["prefix_state"].detach().numpy(),"fork_logits":branch["logits"].detach().numpy(),"native_logits":torch.stack(qlogits).numpy(),
            "fork_states":branch["states"].detach().numpy(),"native_states":torch.stack(qstates).numpy(),"fork_gradient":actual.detach().numpy(),
            "native_query_gradients":torch.stack(qgrad).numpy(),"native_mean_gradient":expected.numpy(),"query_losses":torch.stack(qloss).numpy(),**negative_arrays})
        records.append({"dtype":str(dtype),"model":model.model_kind,"task":model.task,"hidden":128,"encoder":64,"batch":32,"optimizer_updates":0,
            "parameters":parameter_records,"checks":checks,"negative_controls":negative,"raw_sha256":_sha(output/(name+".npz"))})
    cross_precision=_agree(mean_grads[torch.float32].double(),mean_grads[torch.float64],torch.float32)
    result={"kind":"v5_actual_helper_zero_update_calibration","optimizer_updates":0,"scientific_acceptance":False,"cases":records,"fp32_fp64_gradient":cross_precision,"fp32_fp64_gradient_max_difference":cross_precision["max_absolute_difference"]}
    atomic_json(output/"results.json",result);_hash_outputs(output);return result

def prehash(task,updates,base_manifest,deadline):
    rng=np.random.default_rng(2026101301);base_roll=hashlib.sha256(json.dumps({"protocol":"prefix-acquisition-discovery-v3","task":task,"seed":2026101301,"updates":5000,"batch_size":32},sort_keys=True).encode())
    new_roll=hashlib.sha256(json.dumps({"protocol":PROTOCOL,"task":task,"seed":2026101301,"updates":updates,"queries":list(range(8))},sort_keys=True).encode())
    record={"base_batch_sha256":[],"base_prefix_sha256":[],"all_query_batch_sha256":[],"all_query_prefix_sha256":[]}
    for step in range(1,updates+1):
        _check(deadline);base=paired_batch_with_zero(rng,16,0,task);d=batch_identity(base);base_roll.update(step.to_bytes(8,"big")+bytes.fromhex(d))
        if d!=base_manifest["batch_sha256"][step-1] or base_roll.hexdigest()!=base_manifest["prefix_sha256"][step-1]:raise ValueError("v3 original RNG/base stream mismatch")
        allq=query_identity(derive_all_queries(base,task));new_roll.update(step.to_bytes(8,"big")+bytes.fromhex(allq))
        for k,value in (("base_batch_sha256",d),("base_prefix_sha256",base_roll.hexdigest()),("all_query_batch_sha256",allq),("all_query_prefix_sha256",new_roll.hexdigest())):record[k].append(value)
    if updates==5000 and base_roll.hexdigest()!=base_manifest["stream_sha256"]:raise ValueError("full v3 base stream mismatch")
    return {**record,"task":task,"updates":updates,"base_v3_stream_sha256":base_manifest["stream_sha256"],"all_query_stream_sha256":new_roll.hexdigest(),"arrays_retained":False,"frozen_prefit":True}

def evaluate(model,batch,checkpoint,deadline,*,fixture_only=False):
    logits=[];states={k:[] for k in ("post_write","pre_query","post_query")};reports=[]
    for start in range(0,len(batch.targets),64):
        _check(deadline);cap=native_phase_capture(model,torch.from_numpy(batch.inputs[start:start+64].copy()),16)
        logits.append(cap["uninterrupted_logits"].numpy());reports.append(cap["chronology"])
        for k in states:states[k].append(cap["phase_states"][k].numpy())
    native=np.concatenate(logits);prediction=native.argmax(1) if model.task=="categorical" else native[:,0]
    record=evaluate_competence(batch,prediction,trained_seed=0,checkpoint_id=checkpoint,measurement_phase=NATIVE_PHASE,split="fixture" if fixture_only else "validation")
    return record,{**{k:getattr(batch,k) for k in ("inputs","targets","queries","pair_ids")},"native_logits":native,"predictions":prediction,**{k:np.concatenate(v) for k,v in states.items()}},reports

def verify_freeze(config_path,receipt_path,root):
    path=Path(config_path).resolve();root=Path(root).resolve();r=json.loads(Path(receipt_path).read_text())
    if not path.is_relative_to(root) or any(r.get(k)!=v for k,v in {"kind":"query_acquisition_v5_freeze","protocol_id":PROTOCOL,"local_execution_authorized":True,"scientific_acceptance":False}.items()):raise ValueError("coordinator source/config freeze required")
    commit=r.get("source_commit","");review=r.get("independent_review",{});rp=_relative(root,review.get("receipt_relative_to_project",""))
    if len(commit)!=40 or any(c not in "0123456789abcdef" for c in commit) or r.get("config_relative_to_project")!=str(path.relative_to(root)) or r.get("config_sha256")!=_sha(path):raise ValueError("exact source/config binding required")
    if review.get("accepted") is not True or not rp.is_file() or _sha(rp)!=review.get("receipt_sha256") or review.get("implementation_sha256")!=_sha(__file__) or review.get("config_sha256")!=_sha(path):raise ValueError("current independently reviewed hashes required")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()!=commit or subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=root,text=True).strip():raise ValueError("clean committed source required")
    for p in (path,Path(__file__).resolve()):
        if not p.is_relative_to(root) or subprocess.check_output(["git","show",f"{commit}:{p.relative_to(root)}"],cwd=root)!=p.read_bytes():raise ValueError("executed bytes differ from committed source")
    return {"source_commit":commit,"config_sha256":r["config_sha256"],"freeze_receipt_sha256":_sha(receipt_path),"review_sha256":review["receipt_sha256"]}

@dataclass(frozen=True)
class FixtureBudget:
    updates_per_fit:int=2
    evaluate_every:int=1
    fit_indices:tuple[int,...]=(0,)
    wall_seconds:float=120
    def validate(self):
        if type(self.updates_per_fit) is not int or not 1<=self.updates_per_fit<=5000 or type(self.evaluate_every) is not int or not 1<=self.evaluate_every<=500 or self.updates_per_fit%self.evaluate_every:raise ValueError("valid reduced fixture schedule required")
        if not self.fit_indices or len(set(self.fit_indices))!=len(self.fit_indices) or any(type(i) is not int or i not in range(4) for i in self.fit_indices):raise ValueError("declared distinct fixture subset required")
        if isinstance(self.wall_seconds,bool) or not np.isfinite(self.wall_seconds) or not 0<self.wall_seconds<=1200 or (self.updates_per_fit==5000 and len(self.fit_indices)==4 and self.wall_seconds==1200):raise ValueError("strict budget reduction required")

def _engine(config,output,root,identity,budget):
    validate_config(config)
    if budget is not None:budget.validate()
    updates=5000 if budget is None else budget.updates_per_fit;interval=500 if budget is None else budget.evaluate_every;registry=CELLS if budget is None else tuple(CELLS[i] for i in budget.fit_indices)
    wall=1200 if budget is None else budget.wall_seconds;start=time.monotonic();deadline=start+wall;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    context={"protocol_id":PROTOCOL,"fixture_only":budget is not None,"scientific_acceptance":False,"source_identity":identity,"planned_fits":len(registry),"updates_per_fit":updates,"maximum_total_updates":updates*len(registry),"maximum_wall_seconds":wall,"device":"cpu","exposure":config["exposure"]}
    status={**context,"state":"running","started_at_utc":_now(),"fits":[],"fit_registry":[{"model":m,"task":t} for m,t in registry]}
    atomic_json(output/"config.json",{**config,"execution":context});atomic_json(output/"status.json",status)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    atomic_json(output/"precision.json",{"torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),"threads":torch.get_num_threads(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),"cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,"float32_matmul_precision":torch.get_float32_matmul_precision()})
    source_root=Path(__file__).resolve().parents[2];files=["uv.lock","pyproject.toml",".python-version"]+[f"src/representation_collapse/{n}.py" for n in ("query_acquisition","address_acquisition","prefix_acquisition","pilot","diagnostics","vector_task","competence","native_state")]
    atomic_json(output/"source-provenance.json",{"source_sha256":{n:_sha(source_root/n) for n in files},"model_source_commit":config["model_source_commit"],**context})
    current,model,fit,history=None,None,None,[]
    try:
        refs,base_streams=reference_preflight(config,root);prefit=output/"prefit";prefit.mkdir();manifests={};streams={}
        for task in sorted({t for m,t in registry}):
            _check(deadline);batch=make_competence_batch(R0,task,20005);manifests[task]=batch;atomic_npz(prefit/f"validation-{task}.npz",{k:getattr(batch,k) for k in ("inputs","targets","queries","pair_ids")})
            streams[task]=prehash(task,updates,base_streams[task],deadline);atomic_json(prefit/f"all-query-stream-{task}.json",streams[task])
        frozen={p.name:_sha(p) for p in prefit.iterdir()};atomic_json(output/"prefit-freeze.json",{"file_sha256":frozen,"validation_identity":{t:_batch_hash(b) for t,b in manifests.items()},"before_first_optimizer_step":True,**context})
        calibration=output/"calibration";calibration.mkdir()
        for m,t in registry:
            _check(deadline);torch.manual_seed(0)
            with torch.device("cpu"):initial=VectorMemoryModel(m,t,128,64)
            if _weight_hash(initial)!=refs[(m,t)][0]["initial_weight_sha256"]:raise ValueError("calibration initial theta mismatch")
            calibrate_helper(initial,paired_batch_with_zero(np.random.default_rng(2026101301),16,0,t),calibration/f"{m}-{t}",deadline)
        atomic_json(output/"calibration-admission.json",{"all_declared_helpers_calibrated":True,"optimizer_updates":0,"calibration_results_sha256":{f"{m}-{t}":_sha(calibration/f"{m}-{t}/results.json") for m,t in registry},**context})
        baselines={};baseline_dir=output/"baseline";baseline_dir.mkdir()
        for m,t in registry:
            _check(deadline)
            with torch.device("cpu"):baseline=VectorMemoryModel(m,t,128,64)
            baseline.load_state_dict(refs[(m,t)][2]["model_state_dict"]);record,raw,chronology=evaluate(baseline,manifests[t],"pinned_v3_fresh_step5000",deadline,fixture_only=budget is not None)
            baselines[(m,t)]=raw;atomic_npz(baseline_dir/f"{m}-{t}.npz",raw);atomic_json(baseline_dir/f"{m}-{t}.json",{"record":record,"chronology":chronology,"optimizer_updates":0,"checkpoint_sha256":refs[(m,t)][0]["endpoint_checkpoint_sha256"],**context})
        for m,t in registry:
            _check(deadline);fit_start=time.monotonic();current=output/f"{m}-{t}";current.mkdir();history=[];model=None
            fit={"model":m,"task":t,"state":"running","updates_completed":0,"started_at_utc":_now(),**context};status["fits"].append(fit)
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            with torch.device("cpu"):model=VectorMemoryModel(m,t,128,64)
            ref,expected,_=refs[(m,t)]
            if _weight_hash(model)!=ref["initial_weight_sha256"] or any(not torch.equal(model.state_dict()[k],v) for k,v in expected["model_state_dict"].items()):raise ValueError("v3 initial tensor identity mismatch")
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            if optimizer.state:raise ValueError("empty optimizer state required")
            atomic_checkpoint(current/"initial-checkpoint.pt",{"model_state_dict":model.state_dict(),"step":0,"fixture_only":budget is not None,"scientific_acceptance":False})
            atomic_json(current/"provenance.json",{"initial_weight_sha256":_weight_hash(model),"initial_optimizer_state_entries":0,"validation_seed":20005,"training_seed":2026101301,"stream_sha256":streams[t]["all_query_stream_sha256"],"query_branches":8,"normalization":1/(32*8),**context})
            atomic_json(output/"status.json",status)
            rng=np.random.default_rng(2026101301);rolling=hashlib.sha256(json.dumps({"protocol":PROTOCOL,"task":t,"seed":2026101301,"updates":updates,"queries":list(range(8))},sort_keys=True).encode());first_pass=None
            for step in range(1,updates+1):
                _check(deadline);base=paired_batch_with_zero(rng,16,0,t);arrays=derive_all_queries(base,t);d=query_identity(arrays);rolling.update(step.to_bytes(8,"big")+bytes.fromhex(d));stream=streams[t]
                if batch_identity(base)!=stream["base_batch_sha256"][step-1] or d!=stream["all_query_batch_sha256"][step-1] or rolling.hexdigest()!=stream["all_query_prefix_sha256"][step-1]:raise ValueError("prefit query-view/base stream changed")
                prefix,queries,targets=_tensors(arrays,t,torch.float32);model.train();optimizer.zero_grad(set_to_none=True);loss,branch=average_query_loss(model,prefix,queries,targets)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                row={"step":step,"train_mean8_raw_loss":float(loss.detach()),"query_raw_losses":branch["losses"].detach().tolist(),"gradient_norm_before_single_clip":float(norm),"all_query_batch_sha256":d,"all_query_prefix_sha256":rolling.hexdigest()};history.append(row);fit["updates_completed"]=step
                finite=all(bool(torch.isfinite(p).all()) for p in model.parameters());row["post_update_parameters_finite"]=finite
                if not finite:fit["nonfinite_after_completed_update"]=True;raise FloatingPointError("nonfinite weights after completed update")
                if step%interval==0:
                    _check(deadline)
                    for filename in (f"validation-{t}.npz",f"all-query-stream-{t}.json"):
                        if _sha(prefit/filename)!=frozen[filename]:raise ValueError("stored prefit manifest changed")
                    if _batch_hash(manifests[t])!=json.loads((output/"prefit-freeze.json").read_text())["validation_identity"][t]:raise ValueError("validation identity changed")
                    cp=f"checkpoint-step-{step:06d}.pt";atomic_checkpoint(current/cp,{"model_state_dict":model.state_dict(),"step":step,"config":config,"fixture_only":budget is not None,"scientific_acceptance":False})
                    record,raw,chronology=evaluate(model,manifests[t],cp,deadline,fixture_only=budget is not None);atomic_npz(current/f"validation-step-{step:06d}.npz",raw);row.update(validation=record,native_chronology=chronology)
                    if record["coordinate_gate_passed"] and first_pass is None:first_pass=step
                    atomic_json(current/"history.json",history);atomic_json(output/"status.json",status);_check(deadline)
            if rolling.hexdigest()!=streams[t]["all_query_stream_sha256"]:raise ValueError("final query-view stream mismatch")
            base_raw=baselines[(m,t)]
            if any(not np.array_equal(raw[k],base_raw[k]) for k in ("inputs","targets","queries","pair_ids")):raise ValueError("baseline/endpoint histories are not identical")
            ledger=[]
            for q in range(8):
                mask=raw["queries"][:1024]==q;y=raw["targets"][:1024][mask];a=raw["predictions"][:1024][mask];b=base_raw["predictions"][:1024][mask]
                ea=(a!=y).astype(np.float64) if t=="categorical" else (a.astype(np.float64)-y.astype(np.float64))**2/(1/3);eb=(b!=y).astype(np.float64) if t=="categorical" else (b.astype(np.float64)-y.astype(np.float64))**2/(1/3)
                if int(mask.sum())!=128:raise ValueError("coordinate pairing count changed")
                ledger.append({"coordinate":q,"paired_episodes":128,"error_difference_averaged_minus_pinned_sampled":(ea-eb).tolist(),"mean_error_difference":float((ea-eb).mean()),"units":"error probability" if t=="categorical" else "MSE/(1/3)","p_value":None,"confidence_interval":None})
            duration=time.monotonic()-fit_start;fit.update(state="completed",endpoint_step=updates,endpoint_coordinate_gate_passed=record["coordinate_gate_passed"],first_passing_validation_step=first_pass,selected_checkpoint=None,wall_seconds=duration,useful_updates_per_second=updates/duration,
                branch_loss_evaluations=updates*32*8,native_recurrent_row_updates=updates*32*24,independent_prefix_history_instances=updates*16)
            atomic_json(current/"metrics.json",{**fit,"endpoint":record,"paired_baseline_coordinate_ledger":ledger,"endpoint_weight_sha256":_weight_hash(model)});atomic_json(current/"status.json",fit);_hash_outputs(current);atomic_json(output/"status.json",status);current,model,fit=None,None,None
        status["state"]="completed"
    except DeadlineReached:
        status["state"]="incomplete_censored_overall_deadline"
        if fit is not None:fit.update(state=status["state"],endpoint_coordinate_gate_passed=None,selected_checkpoint=None)
    except Exception as error:
        status.update(state="failed_execution_identity_or_numerics",error_type=type(error).__name__,error=str(error))
        if fit is not None:fit.update(state=status["state"],endpoint_coordinate_gate_passed=None,selected_checkpoint=None)
        raise
    finally:
        if current is not None:
            if model is not None:
                step=fit["updates_completed"];cp=f"censored-current-step-{step:06d}.pt";finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
                atomic_checkpoint(current/cp,{"model_state_dict":model.state_dict(),"step":step,"censored":True,"parameters_finite":finite,"usable_for_evaluation":False,"endpoint_competence_known":False,"resume_authorized":False,"selected_checkpoint":None,"fixture_only":budget is not None,"scientific_acceptance":False})
                fit.update(censored_current_checkpoint_id=cp,censored_current_checkpoint_sha256=_sha(current/cp),checkpoint_realized_step=step,preserved_parameters_finite=finite)
            atomic_json(current/"history.json",history);atomic_json(current/"status.json",fit);_hash_outputs(current)
        realized=sum(f["updates_completed"] for f in status["fits"]);status.update(finished_at_utc=_now(),elapsed_seconds=time.monotonic()-start,total_optimizer_updates=realized,total_branch_loss_evaluations=realized*256,total_native_recurrent_row_updates=realized*768,completed_fits=sum(f["state"]=="completed" for f in status["fits"]),unstarted_fits=status["fit_registry"][len(status["fits"]):])
        atomic_json(output/"status.json",status);_hash_outputs(output)
    return status

def run(config_path,output,freeze_receipt):
    c=json.loads(Path(config_path).read_text());validate_config(c)
    if torch.__version__.split("+")[0]!="2.6.0":raise ValueError("frozen Torch2.6.0 required")
    root=Path(__file__).resolve().parents[2];identity=verify_freeze(config_path,freeze_receipt,root);return _engine(c,output,root,identity,None)
def run_fixture(config,output,project_root,budget):
    if not isinstance(budget,FixtureBudget):raise ValueError("explicit fixture budget required")
    return _engine(config,output,Path(project_root).resolve(),{"fixture_only":True,"implementation_sha256":_sha(__file__)},budget)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--freeze-receipt",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);a=parser.parse_args();s=run(a.config,a.output,a.freeze_receipt);print(json.dumps({"state":s["state"],"optimizer_updates":s["total_optimizer_updates"],"scientific_acceptance":False}))
if __name__=="__main__":main()
