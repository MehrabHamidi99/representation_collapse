"""Separate singleton-address R0 discovery; never relaxes v2/v3 guards.

Production requires committed-clean source/config/review freeze. The explicit
fixture API is infrastructure only, with strictly reduced execution bounds.
"""
from __future__ import annotations
import argparse
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
from .diagnostics import paired_batch_with_zero
from .pilot import VectorMemoryModel, atomic_checkpoint, atomic_json, atomic_npz
from .prefix_acquisition import (_positive_int, _relative, _sha, _weight_hash,
    _hash_outputs, batch_identity, DeadlineReached, _check_deadline)

PROTOCOL = "address-acquisition-discovery-v4"
MODEL_SOURCE = "999d88d8679c0f353c6bb7fb4f5ab8433a7e730c"
ADDRESSES = (0, 7, 15)
CELLS = tuple((m,t,q) for m in ("gru","lstm") for t in ("categorical","continuous") for q in ADDRESSES)

def _now(): return datetime.now(timezone.utc).isoformat()

def validate_config(config):
    expected = {"protocol_id":PROTOCOL,"models":["gru","lstm"],"tasks":["categorical","continuous"],
        "conditions":list(ADDRESSES),"model_source_commit":MODEL_SOURCE,"seed":0,"learning_rate":.001,
        "dimensions":{"hidden_size":128,"input_bottleneck":64,"batch_size":32},
        "optimizer":{"name":"AdamW","weight_decay":.01,"gradient_clip":1.0,"new_optimizer_per_fit":True},
        "device":"cpu","cpu_threads":1,"training_stream_seed":2026101301,"validation_sampling_seed":20003,
        "validation_episodes_per_address_per_arm":128,"categorical_accuracy_gate":.95,
        "continuous_population_normalized_mse_gate":.05,"population_variance":1/3,"comparison_ulps_float64":4,
        "evaluation_every_updates":500,"updates_per_fit":5000,"maximum_fits":12,
        "maximum_total_optimizer_updates":60000,"maximum_total_wall_seconds":1200,
        "confirmation":False,"scientific_acceptance":False,"cluster_submissions":0,"provider_calls":0,
        "baseline_comparison_addresses":[0,7],"baseline_excluded_address":15,"baseline_new_optimizer_updates":0,
        "precision":"float32; deterministic_algorithms; CUDA/cuDNN TF32off; matmul_precision highest"}
    if any(config.get(k)!=v for k,v in expected.items()): raise ValueError("v4 scientific or execution contract changed")
    for k in ("cpu_threads","training_stream_seed","validation_sampling_seed","updates_per_fit","evaluation_every_updates",
              "maximum_fits","maximum_total_optimizer_updates","maximum_total_wall_seconds"):_positive_int(config[k],k)
    if any(type(config[k]) is not int for k in ("seed","cluster_submissions","provider_calls","baseline_new_optimizer_updates")):
        raise ValueError("integer seed/budget counters required")
    refs=config.get("v3_fresh_references",[])
    if len(refs)!=4 or {(r.get("model"),r.get("task")) for r in refs}!={(m,t) for m in ("gru","lstm") for t in ("categorical","continuous")}:
        raise ValueError("four distinct pinned v3 fresh references required")
    for ref in refs:
        if ref.get("source_commit")!=MODEL_SOURCE or ref.get("condition")!="fresh_seed0" or ref.get("endpoint_updates")!=5000:
            raise ValueError("references must be fixed v3 fresh endpoints")
    if len(config.get("v3_base_training_streams",[]))!=2: raise ValueError("both v3 base stream identities required")
    for packet in refs+config["v3_base_training_streams"]:
        for k,v in packet.items():
            if k.endswith("_relative_to_project"):_relative(Path("/validation-root"),v)
            if k.endswith("sha256") and (not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v)):
                raise ValueError("full lowercase evidence hash required")

@dataclass(frozen=True)
class AddressBatch:
    inputs: np.ndarray
    targets: np.ndarray
    queries: np.ndarray
    pair_ids: np.ndarray

def target_oracle(inputs, targets, queries, task):
    """Validate all sixteen address positions; targets never enter QUERY."""
    x,y,q=np.asarray(inputs),np.asarray(targets),np.asarray(queries)
    if task not in ("categorical","continuous") or x.ndim!=3 or x.shape[1:]!=(17,38) or y.ndim!=1 or q.ndim!=1 or len(x)!=len(y) or len(q)!=len(y):
        raise ValueError("[batch,17,38] and scalar labels/selectors required")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or q.dtype.kind not in "iu" or np.any((q<0)|(q>15)):
        raise ValueError("finite values and QUERY domain0..15 required")
    types=x[:,:,:5];address=x[:,:,5:21]
    if not np.all((types==0)|(types==1)) or not np.all(types.sum(2)==1) or not np.all(types[:,:16].argmax(2)==0) or not np.all(types[:,-1].argmax(1)==4):
        raise ValueError("sixteen WRITEs followed by QUERY required")
    if not np.all((address==0)|(address==1)) or not np.all(address.sum(2)==1) or not np.array_equal(address[:,:16].argmax(2),np.broadcast_to(np.arange(16),(len(x),16))) or not np.array_equal(address[:,-1].argmax(1),q):
        raise ValueError("ordered WRITE and matching one-hot QUERY addresses required")
    if np.any(x[:,-1,21:]!=0):raise ValueError("QUERY may not leak a target payload")
    cat=x[:,:16,21:37];real=x[:,:16,37]
    if task=="categorical":
        if not np.all((cat==0)|(cat==1)) or not np.all(cat.sum(2)==1) or np.any(real!=0):raise ValueError("one-hot class payloads required")
    elif np.any(cat!=0) or np.any(np.abs(real)>1):raise ValueError("bounded real-only payloads required")
    selected=x[np.arange(len(x)),q]
    expected=selected[:,21:37].argmax(1) if task=="categorical" else selected[:,37]
    if not np.array_equal(expected,y):raise ValueError("label differs from addressed WRITE")
    return len(y)

def derive_batch(base, address, task):
    if type(address) is not int or address not in ADDRESSES:raise ValueError("declared singleton address required")
    x=base.inputs.copy();n=len(x);x[:,-1,5:21]=0;x[:,-1,5+address]=1
    q=np.full(n,address,dtype=np.int64)
    y=x[:,address,21:37].argmax(1).astype(np.int64) if task=="categorical" else x[:,address,37].copy()
    batch=AddressBatch(x,y,q,np.tile(np.arange(n//2,dtype=np.int64),2))
    target_oracle(x,y,q,task)
    return batch

def address_identity(batch):
    h=hashlib.sha256()
    for name in ("inputs","targets","queries","pair_ids"):
        a=np.ascontiguousarray(getattr(batch,name));h.update(json.dumps({"name":name,"dtype":a.dtype.str,"shape":list(a.shape)},sort_keys=True).encode());h.update(a.tobytes())
    return h.hexdigest()

def _base_rolling(task):
    return hashlib.sha256(json.dumps({"protocol":"prefix-acquisition-discovery-v3","task":task,"seed":2026101301,"updates":5000,"batch_size":32},sort_keys=True).encode())

def _address_rolling(task,address,updates):
    return hashlib.sha256(json.dumps({"protocol":PROTOCOL,"task":task,"address":address,"seed":2026101301,"updates":updates,"batch_size":32},sort_keys=True).encode())

def prehash_streams(task,updates,addresses,base_manifest,deadline):
    rng=np.random.default_rng(2026101301);base_rolling=_base_rolling(task)
    rolling={q:_address_rolling(task,q,updates) for q in addresses}
    records={q:{"batch_sha256":[],"prefix_sha256":[],"base_batch_sha256":[],"base_prefix_sha256":[]} for q in addresses}
    for step in range(1,updates+1):
        _check_deadline(deadline);base=paired_batch_with_zero(rng,16,0,task);digest=batch_identity(base)
        base_rolling.update(step.to_bytes(8,"big")+bytes.fromhex(digest))
        if digest!=base_manifest["batch_sha256"][step-1] or base_rolling.hexdigest()!=base_manifest["prefix_sha256"][step-1]:
            raise ValueError("base draw/RNG stream differs from pinned v3")
        for q in addresses:
            new=derive_batch(base,q,task);d=address_identity(new);rolling[q].update(step.to_bytes(8,"big")+bytes.fromhex(d))
            r=records[q];r["batch_sha256"].append(d);r["prefix_sha256"].append(rolling[q].hexdigest());r["base_batch_sha256"].append(digest);r["base_prefix_sha256"].append(base_rolling.hexdigest())
    if updates==5000 and base_rolling.hexdigest()!=base_manifest["stream_sha256"]:raise ValueError("complete base stream mismatch")
    return {q:{**records[q],"task":task,"address":q,"updates":updates,"stream_sha256":rolling[q].hexdigest(),
        "base_prefix_sha256_final":base_rolling.hexdigest(),"base_v3_stream_sha256":base_manifest["stream_sha256"],"arrays_retained":False,"prefit":True} for q in addresses}

def _pack(state,kind):return torch.cat((state[0][0],state[1][0]),1) if kind=="lstm" else state[0]

def _agreement(a,b):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError("nonfinite/malformed native state")
    delta=torch.abs(a-b);passed=bool(torch.all(delta<=1e-6+1e-5*torch.abs(b)))
    if not passed:raise FloatingPointError("native adapter mismatch")
    return {"passed":passed,"max_absolute_difference":float(delta.max()),"bitwise_equal":bool(torch.equal(a,b)),"absolute_tolerance":1e-6,"relative_tolerance":1e-5}

@torch.no_grad()
def native_capture(model,inputs):
    """New sixteen-address adapter; no call to restricted v2/v3 capture."""
    x=inputs.detach().cpu().numpy();q=x[:,-1,5:21].argmax(1).astype(np.int64)
    y=x[np.arange(len(x)),q,21:37].argmax(1) if model.task=="categorical" else x[np.arange(len(x)),q,37]
    target_oracle(x,y,q,model.task)
    expected=torch.nn.GRU if model.model_kind=="gru" else torch.nn.LSTM
    if not isinstance(model.core,expected) or model.core.num_layers!=1 or model.core.bidirectional or not model.core.batch_first or model.core.dropout!=0 or getattr(model.core,"proj_size",0)!=0:
        raise ValueError("complete one-layer native state contract required")
    if inputs.dtype!=torch.float32 or inputs.device.type!="cpu" or any(p.dtype!=torch.float32 or p.device.type!="cpu" for p in model.parameters()):
        raise ValueError("native CPU FP32 required")
    model.eval();encoded=model.encoder(inputs);_,write=model.core(encoded[:,:16]);out,post=model.core(encoded)
    copied=tuple(v.clone() for v in write) if isinstance(write,tuple) else write.clone()
    fork_out,fork_post=model.core(encoded[:,-1:],copied)
    native_logits=model(inputs);core_logits=model.head(out[:,-1]);fork_logits=model.head(fork_out[:,-1])
    checks={"core_vs_uninterrupted_logits":_agreement(core_logits,native_logits),"fork_vs_uninterrupted_logits":_agreement(fork_logits,native_logits),
        "fork_vs_uninterrupted_complete_state":_agreement(_pack(fork_post,model.model_kind),_pack(post,model.model_kind))}
    states={"post_write":_pack(write,model.model_kind),"pre_query":_pack(write,model.model_kind),"post_query":_pack(post,model.model_kind)}
    return {"native_logits":native_logits.detach(),"states":{k:v.detach().clone() for k,v in states.items()},
        "chronology":{"passed":True,"adapter":"address-domain0..15-v4","state_dimension":states["post_query"].shape[1],
            "packing_order":"hidden_then_cell" if model.model_kind=="lstm" else "hidden","write_updates":16,"body_updates":0,"query_updates":1,
            "query_addresses":np.unique(q).tolist(),"checks":checks,"weights_sha256":_weight_hash(model),
            "inputs_sha256":hashlib.sha256(x.tobytes()).hexdigest(),
            "source_sha256":{"adapter":_sha(__file__),"model":_sha(Path(__file__).with_name("pilot.py"))},
            "phase_after_update_indices_zero_based":{"post_write":15,"pre_query":15,"post_query":16},
            "floquet_certified":False,"scientific_acceptance":False}}

def score_batch(batch,prediction,task,address):
    target_oracle(batch.inputs,batch.targets,batch.queries,task)
    prediction=np.asarray(prediction)
    if type(address) is not int or address not in ADDRESSES or len(batch.targets)!=256 or prediction.shape!=(256,) or not np.all(batch.queries==address) or not np.isfinite(prediction).all():raise ValueError("128 paired singleton episodes per arm required")
    arms={}
    for arm,start,end in (("periodic",0,128),("control",128,256)):
        a,b=prediction[start:end],batch.targets[start:end]
        metric=float(np.mean(a==b)) if task=="categorical" else float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2)/(1/3))
        if not np.isfinite(metric):raise FloatingPointError("nonfinite singleton evaluation metric")
        passed=metric>=.95-4*np.spacing(.95) if task=="categorical" else metric<=.05+4*np.spacing(.05)
        arms[arm]={"metric":metric,"passed":bool(passed),"episodes":128}
    return {"kind":"v4_singleton_coordinate_competence","address":address,"arms":arms,
        "coordinate_gate_passed":all(a["passed"] for a in arms.values()),"metric_units":"accuracy" if task=="categorical" else "MSE/(1/3)",
        "trained_seed_units":1,"paired_episode_units":128,"scientific_acceptance":False,"not_main_v2_v3_eligibility":True}

def evaluate(model,batch,address,deadline):
    pieces=[];states={k:[] for k in ("post_write","pre_query","post_query")};reports=[]
    for start in range(0,len(batch.targets),64):
        _check_deadline(deadline);cap=native_capture(model,torch.from_numpy(batch.inputs[start:start+64].copy()))
        pieces.append(cap["native_logits"].numpy());reports.append(cap["chronology"])
        for k in states:states[k].append(cap["states"][k].numpy())
    logits=np.concatenate(pieces);pred=logits.argmax(1) if model.task=="categorical" else logits[:,0]
    return score_batch(batch,pred,model.task,address),{**{k:getattr(batch,k) for k in ("inputs","targets","queries","pair_ids")},
        "native_logits":logits,"predictions":pred,**{k:np.concatenate(v) for k,v in states.items()}},reports

def verify_freeze(config_path,receipt_path,root):
    config_path=Path(config_path).resolve();root=Path(root).resolve();receipt=json.loads(Path(receipt_path).read_text())
    if not config_path.is_relative_to(root):raise ValueError("configuration outside committed project")
    if any(receipt.get(k)!=v for k,v in {"kind":"address_acquisition_v4_freeze","protocol_id":PROTOCOL,"local_execution_authorized":True,"scientific_acceptance":False}.items()):raise ValueError("coordinator freeze required")
    commit=receipt.get("source_commit","");relative=str(config_path.relative_to(root));review=receipt.get("independent_review",{})
    if len(commit)!=40 or any(c not in "0123456789abcdef" for c in commit):raise ValueError("full source commit required")
    if receipt.get("config_relative_to_project")!=relative or receipt.get("config_sha256")!=_sha(config_path):raise ValueError("configuration freeze mismatch")
    rp=_relative(root,review.get("receipt_relative_to_project",""))
    if review.get("accepted") is not True or not rp.is_file() or _sha(rp)!=review.get("receipt_sha256") or review.get("implementation_sha256")!=_sha(__file__) or review.get("config_sha256")!=_sha(config_path):raise ValueError("current hash-bound independent review required")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()!=commit or subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=root,text=True).strip():raise ValueError("clean committed source required")
    for path in (config_path,Path(__file__).resolve()):
        if not path.is_relative_to(root) or subprocess.check_output(["git","show",f"{commit}:{path.relative_to(root)}"],cwd=root)!=path.read_bytes():raise ValueError("source/config not at committed bytes")
    return {"source_commit":commit,"config_sha256":receipt["config_sha256"],"freeze_receipt_sha256":_sha(receipt_path),"review_sha256":review["receipt_sha256"]}

@dataclass(frozen=True)
class FixtureBudget:
    updates_per_fit:int=2
    evaluate_every:int=1
    fit_indices:tuple[int,...]=(0,2)
    wall_seconds:float=60
    def validate(self):
        if _positive_int(self.updates_per_fit,"updates")>5000 or _positive_int(self.evaluate_every,"evaluation")>500 or self.updates_per_fit%self.evaluate_every:raise ValueError("reduced valid fixture schedule required")
        if not self.fit_indices or len(set(self.fit_indices))!=len(self.fit_indices) or any(type(i) is not int or i not in range(12) for i in self.fit_indices):raise ValueError("distinct declared fixture subset required")
        if isinstance(self.wall_seconds,bool) or not np.isfinite(self.wall_seconds) or not 0<self.wall_seconds<=1200 or (self.updates_per_fit==5000 and len(self.fit_indices)==12 and self.wall_seconds==1200):raise ValueError("strictly reduced fixture envelope required")

def reference_preflight(config,root):
    for name,digest in config["unchanged_source_sha256"].items():
        if _sha(Path(__file__).resolve().parents[2]/name)!=digest:raise ValueError("model/data source differs from pinned v3")
    result_path=_relative(root,config["v3_result_receipt_relative_to_project"])
    if _sha(result_path)!=config["v3_result_receipt_sha256"]:raise ValueError("v3 result reference mismatch")
    result=json.loads(result_path.read_text())
    if result.get("source_commit")!=MODEL_SOURCE:raise ValueError("v3 reference source mismatch")
    refs={}
    for ref in config["v3_fresh_references"]:
        for k in ("initial_checkpoint","endpoint_checkpoint","metrics","provenance"):
            path=_relative(root,ref[k+"_relative_to_project"])
            if not path.is_file() or _sha(path)!=ref[k+"_sha256"]:raise ValueError("pinned v3 fresh artifact mismatch")
        initial=torch.load(_relative(root,ref["initial_checkpoint_relative_to_project"]),map_location="cpu",weights_only=True)
        endpoint=torch.load(_relative(root,ref["endpoint_checkpoint_relative_to_project"]),map_location="cpu",weights_only=True)
        if initial.get("step")!=0 or initial.get("condition")!="fresh_seed0" or endpoint.get("step")!=5000 or endpoint.get("condition")!="fresh_seed0" or _weight_hash_from_state(initial["model_state_dict"])!=ref["initial_weight_sha256"]:raise ValueError("v3 initialization/endpoint semantics mismatch")
        accepted=next(f for f in result["fits"] if f["model"]==ref["model"] and f["task"]==ref["task"] and f["condition"]=="fresh_seed0")
        if accepted["endpoint_checkpoint_sha256"]!=ref["endpoint_checkpoint_sha256"] or accepted["initial_weight_sha256"]!=ref["initial_weight_sha256"]:raise ValueError("reference not bound to accepted v3 reduction")
        refs[(ref["model"],ref["task"])]=(ref,initial,endpoint)
    streams={}
    for pin in config["v3_base_training_streams"]:
        path=_relative(root,pin["manifest_relative_to_project"])
        if _sha(path)!=pin["manifest_sha256"]:raise ValueError("v3 stream manifest mismatch")
        stream=json.loads(path.read_text())
        if stream["stream_sha256"]!=pin["stream_sha256"] or stream["updates"]!=5000 or stream["sampling_seed"]!=2026101301:raise ValueError("v3 stream identity mismatch")
        streams[pin["task"]]=stream
    return refs,streams

def _weight_hash_from_state(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def _engine(config,output,root,identity,budget):
    validate_config(config)
    if budget is not None:budget.validate()
    updates=5000 if budget is None else budget.updates_per_fit;interval=500 if budget is None else budget.evaluate_every
    registry=CELLS if budget is None else tuple(CELLS[i] for i in budget.fit_indices);wall=1200 if budget is None else budget.wall_seconds
    started=time.monotonic();deadline=started+wall;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    context={"protocol_id":PROTOCOL,"fixture_only":budget is not None,"scientific_acceptance":False,"source_identity":identity,
        "updates_per_fit":updates,"planned_fits":len(registry),"maximum_total_updates":updates*len(registry),"maximum_wall_seconds":wall,
        "device":"cpu","endpoint_selection":False,"baseline_new_optimizer_updates":0}
    status={**context,"state":"running","started_at_utc":_now(),"fits":[],"fit_registry":[{"model":m,"task":t,"address":q} for m,t,q in registry]}
    atomic_json(output/"config.json",{**config,"execution":context});atomic_json(output/"status.json",status)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision("highest")
    atomic_json(output/"precision.json",{"torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),"threads":torch.get_num_threads(),
        "cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,"float32_matmul_precision":torch.get_float32_matmul_precision(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled()})
    source_root=Path(__file__).resolve().parents[2]
    source_files=["uv.lock","pyproject.toml",".python-version"]+[f"src/representation_collapse/{name}.py" for name in ("address_acquisition","prefix_acquisition","pilot","diagnostics","vector_task","metrics")]
    atomic_json(output/"source-provenance.json",{"source_sha256":{name:_sha(source_root/name) for name in source_files},"model_source_commit":MODEL_SOURCE,**context})
    current_path,current_model,fit,history=None,None,None,[]
    try:
        refs,base_streams=reference_preflight(config,root);manifests={};streams={};prefit=output/"prefit";prefit.mkdir()
        for task in sorted({t for m,t,q in registry}):
            addresses=sorted({q for m,t,q in registry if t==task});base=paired_batch_with_zero(np.random.default_rng(20003),128,0,task)
            for q in addresses:
                batch=derive_batch(base,q,task);manifests[(task,q)]=batch
                atomic_npz(prefit/f"validation-{task}-q{q}.npz",{k:getattr(batch,k) for k in ("inputs","targets","queries","pair_ids")})
            frozen=prehash_streams(task,updates,addresses,base_streams[task],deadline)
            for q,stream in frozen.items():streams[(task,q)]=stream;atomic_json(prefit/f"train-{task}-q{q}.json",stream)
        frozen_files={p.name:_sha(p) for p in prefit.iterdir() if p.is_file()}
        atomic_json(output/"prefit-freeze.json",{"file_sha256":frozen_files,"validation_array_sha256":{f"{t}-q{q}":address_identity(b) for (t,q),b in manifests.items()},"before_first_optimizer_step":True,**context})
        baselines={};baseline_dir=output/"baseline";baseline_dir.mkdir()
        for m,t,q in registry:
            if q not in (0,7):continue
            _check_deadline(deadline)
            with torch.device("cpu"):model=VectorMemoryModel(m,t,128,64).to(device="cpu",dtype=torch.float32)
            model.load_state_dict(refs[(m,t)][2]["model_state_dict"])
            score,raw,chronology=evaluate(model,manifests[(t,q)],q,deadline);baselines[(m,t,q)]=raw
            atomic_npz(baseline_dir/f"{m}-{t}-q{q}.npz",raw);atomic_json(baseline_dir/f"{m}-{t}-q{q}.json",{"score":score,"chronology":chronology,"pinned_checkpoint_sha256":refs[(m,t)][0]["endpoint_checkpoint_sha256"],"optimizer_updates":0,"reevaluation_only":True,"query_in_original_support":True,**context})
        for m,t,q in registry:
            _check_deadline(deadline);current_path=output/f"{m}-{t}-q{q}";current_path.mkdir();history=[];current_model=None
            fit={"model":m,"task":t,"address":q,"state":"running","updates_completed":0,"started_at_utc":_now(),**context};status["fits"].append(fit)
            random.seed(0);np.random.seed(0);torch.manual_seed(0)
            with torch.device("cpu"):current_model=VectorMemoryModel(m,t,128,64).to(device="cpu",dtype=torch.float32)
            initial=current_model.state_dict();ref,expected,_=refs[(m,t)]
            if _weight_hash(current_model)!=ref["initial_weight_sha256"] or any(not torch.equal(initial[k],v) for k,v in expected["model_state_dict"].items()):raise ValueError("initial theta differs from pinned v3 fresh initialization")
            optimizer=torch.optim.AdamW(current_model.parameters(),lr=.001,weight_decay=.01)
            if optimizer.state:raise ValueError("optimizer state must start empty")
            atomic_checkpoint(current_path/"initial-checkpoint.pt",{"model_state_dict":initial,"step":0,"address":q,"fixture_only":budget is not None,"scientific_acceptance":False})
            atomic_json(current_path/"provenance.json",{"initial_weight_sha256":_weight_hash(current_model),"reference_initial_checkpoint_sha256":ref["initial_checkpoint_sha256"],"initial_optimizer_state_entries":0,"intervening_WRITE_updates":15-q,"terminal_QUERY_updates":1,
                "training_stream_sha256":streams[(t,q)]["stream_sha256"],"training_seed":2026101301,"validation_seed":20003,**context})
            rng=np.random.default_rng(2026101301);rolling=_address_rolling(t,q,updates);first_pass=None
            for step in range(1,updates+1):
                _check_deadline(deadline);base=paired_batch_with_zero(rng,16,0,t);batch=derive_batch(base,q,t);digest=address_identity(batch)
                rolling.update(step.to_bytes(8,"big")+bytes.fromhex(digest));stream=streams[(t,q)]
                if batch_identity(base)!=stream["base_batch_sha256"][step-1] or digest!=stream["batch_sha256"][step-1] or rolling.hexdigest()!=stream["prefix_sha256"][step-1]:raise ValueError("training draw differs from prefit paired stream")
                current_model.train();optimizer.zero_grad(set_to_none=True);logits=current_model(torch.from_numpy(batch.inputs.copy()));target=torch.as_tensor(batch.targets,dtype=torch.long if t=="categorical" else torch.float32,device="cpu")
                loss=F.cross_entropy(logits,target) if t=="categorical" else F.mse_loss(logits[:,0],target)
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(current_model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                row={"step":step,"train_loss":float(loss.detach()),"gradient_norm_before_clip":float(norm),"training_batch_sha256":digest,"training_prefix_sha256":rolling.hexdigest()};history.append(row);fit["updates_completed"]=step
                finite=all(bool(torch.isfinite(p).all()) for p in current_model.parameters());row["post_update_parameters_finite"]=finite
                if not finite:fit["nonfinite_after_completed_update"]=True;raise FloatingPointError("nonfinite parameters after completed update")
                if step%interval==0:
                    _check_deadline(deadline)
                    for filename in (f"validation-{t}-q{q}.npz",f"train-{t}-q{q}.json"):
                        if _sha(prefit/filename)!=frozen_files[filename]:raise ValueError("stored prefit identity changed")
                    if address_identity(manifests[(t,q)])!=json.loads((output/"prefit-freeze.json").read_text())["validation_array_sha256"][f"{t}-q{q}"]:raise ValueError("inmemory validation identity changed")
                    cp=f"checkpoint-step-{step:06d}.pt";atomic_checkpoint(current_path/cp,{"model_state_dict":current_model.state_dict(),"step":step,"address":q,"config":config,"fixture_only":budget is not None,"scientific_acceptance":False})
                    score,raw,chronology=evaluate(current_model,manifests[(t,q)],q,deadline);atomic_npz(current_path/f"validation-step-{step:06d}.npz",raw);row.update(validation=score,native_chronology=chronology)
                    if score["coordinate_gate_passed"] and first_pass is None:first_pass=step
                    atomic_json(current_path/"history.json",history);atomic_json(output/"status.json",status);_check_deadline(deadline)
            if rolling.hexdigest()!=streams[(t,q)]["stream_sha256"]:raise ValueError("final paired training stream changed")
            score=history[-1]["validation"];contrast=None
            if q in (0,7):
                base_raw=baselines[(m,t,q)];assert np.array_equal(raw["inputs"],base_raw["inputs"]) and np.array_equal(raw["targets"],base_raw["targets"])
                y=raw["targets"][:128];a=raw["predictions"][:128];b=base_raw["predictions"][:128]
                ea=(a!=y).astype(np.float64) if t=="categorical" else (a.astype(np.float64)-y.astype(np.float64))**2/(1/3)
                eb=(b!=y).astype(np.float64) if t=="categorical" else (b.astype(np.float64)-y.astype(np.float64))**2/(1/3)
                contrast={"kind":"singleton_minus_pinned_random_QUERY_baseline_native_error","paired_episodes":128,"paired_error_difference":(ea-eb).tolist(),"mean_paired_error_difference":float((ea-eb).mean()),"units":"error probability" if t=="categorical" else "MSE/(1/3)","p_value":None,"confidence_interval":None,"scientific_acceptance":False}
            fit.update(state="completed",endpoint_step=updates,endpoint_coordinate_gate_passed=score["coordinate_gate_passed"],first_passing_validation_step=first_pass,selected_checkpoint=None,final_training_stream_sha256=rolling.hexdigest())
            atomic_json(current_path/"metrics.json",{**fit,"endpoint":score,"paired_baseline_contrast":contrast,"baseline_exclusion_reason":"QUERY15 outsideoriginaltrainedQUERYsupport" if q==15 else None,"endpoint_weight_sha256":_weight_hash(current_model)})
            atomic_json(current_path/"status.json",fit);_hash_outputs(current_path);atomic_json(output/"status.json",status);current_path,current_model,fit=None,None,None
        status["state"]="completed"
    except DeadlineReached:
        status["state"]="incomplete_censored_overall_deadline"
        if fit is not None:fit.update(state=status["state"],endpoint_coordinate_gate_passed=None,selected_checkpoint=None)
    except Exception as error:
        status.update(state="failed_execution_identity_or_numerics",error_type=type(error).__name__,error=str(error))
        if fit is not None:fit.update(state=status["state"],endpoint_coordinate_gate_passed=None,selected_checkpoint=None)
        raise
    finally:
        if current_path is not None:
            if current_model is not None:
                step=fit["updates_completed"];cp=f"censored-current-step-{step:06d}.pt";finite=all(bool(torch.isfinite(p).all()) for p in current_model.parameters())
                atomic_checkpoint(current_path/cp,{"model_state_dict":current_model.state_dict(),"step":step,"address":fit["address"],"censored":True,"parameters_finite":finite,"usable_for_evaluation":False,"endpoint_competence_known":False,"resume_authorized":False,"selected_checkpoint":None,"fixture_only":budget is not None,"scientific_acceptance":False})
                fit.update(censored_current_checkpoint_id=cp,censored_current_checkpoint_sha256=_sha(current_path/cp),checkpoint_realized_step=step,preserved_parameters_finite=finite)
            atomic_json(current_path/"history.json",history);atomic_json(current_path/"status.json",fit);_hash_outputs(current_path)
        status.update(finished_at_utc=_now(),elapsed_seconds=time.monotonic()-started,total_optimizer_updates=sum(f["updates_completed"] for f in status["fits"]),completed_fits=sum(f["state"]=="completed" for f in status["fits"]),unstarted_fits=status["fit_registry"][len(status["fits"]):])
        atomic_json(output/"status.json",status);_hash_outputs(output)
    return status

def run(config_path,output,freeze_receipt):
    config_path=Path(config_path);config=json.loads(config_path.read_text());validate_config(config)
    if torch.__version__.split("+")[0]!="2.6.0":raise ValueError("frozen Torch2.6.0 required")
    root=Path(__file__).resolve().parents[2];identity=verify_freeze(config_path,freeze_receipt,root)
    return _engine(config,output,root,identity,None)

def run_fixture(config,output,project_root,budget):
    if not isinstance(budget,FixtureBudget):raise ValueError("explicit fixture budget required")
    return _engine(config,output,Path(project_root).resolve(),{"fixture_only":True,"implementation_sha256":_sha(__file__)},budget)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--freeze-receipt",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();s=run(args.config,args.output,args.freeze_receipt);print(json.dumps({"state":s["state"],"total_optimizer_updates":s["total_optimizer_updates"],"scientific_acceptance":False}))

if __name__=="__main__":main()
