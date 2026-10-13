"""Prospective label-free native chronology diagnostic; no fitting or scoring.

The full registry needs an external clean-source/review freeze. Only the
explicit tiny synthetic fixture API bypasses that admission; it never reads
the known trained checkpoint or its validation inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from .pilot import VectorMemoryModel
from .vector_task import render, sample_memory

PROTOCOL = "native-chronology-calibration-v1"
CELLS = (("gru", "categorical"), ("gru", "continuous"),
         ("lstm", "categorical"), ("lstm", "continuous"))
DTYPES = {"float32": torch.float32, "float64": torch.float64}
TOLERANCES = {"float32": (1e-6, 1e-5), "float64": (1e-12, 1e-10)}
PINNED_SOURCES = {
    "src/representation_collapse/pilot.py": "0aa1c337ffd0ab576056e70b4ecb0dfeee073c1d3453e87389447f77f1dfbbd3",
    "src/representation_collapse/vector_task.py": "df3843330ac9c59205823c8d55868e7878e903c0065c5af8bc89d1308243a297",
    "src/representation_collapse/native_state.py": "eeadc7a7332045c6209c6b823af2ec35740dbe3eb662dc0de25755e82041fdeb",
}
KNOWN_CHECKPOINT = "runs/query-acquisition-v5-20261013/experiment/gru-categorical/censored-current-step-004000.pt"
KNOWN_INPUTS = "runs/query-acquisition-v5-20261013/experiment/prefit/validation-categorical.npz"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_contract():
    return {"protocol_id": PROTOCOL, "schema_version": 1,
        "proposal_sha256": "9438ec2626b30b3f85aca2f784ed5d9d16bc7af18ccc6ad6216181d36cd83f32",
        "proposal_review_sha256": "ec445dea80061e72446881ac838d30f3621aed441952f76a68cf15a321575c3f",
        "models_tasks": [list(c) for c in CELLS], "initialization_seeds": [0, 17],
        "input_seeds": [30013, 30017], "histories": 64, "chunk_sizes": [64, 32, 16, 8, 1],
        "dtype_order": ["float64", "float32"], "repeats": 2,
        "dimensions": {"hidden": 128, "encoder": 64}, "device": "cpu", "cpu_threads": 1,
        "torch_version": "2.6.0", "maximum_core_calls": 120000, "maximum_wall_seconds": 600,
        "core_call_registry": {"primary": 116288, "known_regression": 2944,
                               "N1": 368, "N2": 320, "N3": 80},
        "tolerances": {d: {"atol": a, "rtol": r} for d, (a, r) in TOLERANCES.items()},
        "canonical_only_admitted": False, "scientific_acceptance": False,
        "training_authorized": False, "targets_read": False,
        "source_pins": PINNED_SOURCES,
        "known_failed_source": "f766df5e792c8a6dedaa3ea17f2c92fd630f067d",
        "known_checkpoint": {"path": KNOWN_CHECKPOINT,
            "sha256": "b57b645f80a8e85a1743385e01df8075706904e122a57923b6216310365f3788"},
        "known_inputs": {"path": KNOWN_INPUTS,
            "sha256": "5f86500b5378bde6a179074a2c1ee4d682ea1989b6e185020843ac452ddee877"}}


def validate_config(config):
    if json.dumps(config, sort_keys=True) != json.dumps(configuration_contract(), sort_keys=True):
        raise ValueError("configuration differs from the exact reviewed numerical contract")


def _relative(root, value):
    p = Path(value)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError("project-relative path without traversal required")
    result = root / p
    cursor = result
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError("symlinked references are rejected")
        cursor = cursor.parent
    return result


def verify_freeze(config_path, receipt_path, root):
    """No trained/input loading or output creation occurs before this returns."""
    root = Path(root).resolve(); config_path = Path(config_path).resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("configuration outside project")
    receipt = json.loads(Path(receipt_path).read_text())
    required = {"kind": "native_chronology_calibration_v1_freeze", "protocol_id": PROTOCOL,
                "calibration_execution_authorized": True, "scientific_acceptance": False,
                "training_authorized": False, "canonical_only_admitted": False}
    if any(receipt.get(k) is not v if isinstance(v, bool) else receipt.get(k) != v for k, v in required.items()):
        raise ValueError("external numerical-only execution freeze required")
    commit = receipt.get("source_commit", "")
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("full committed source identity required")
    relative = str(config_path.relative_to(root))
    if receipt.get("config_relative_to_project") != relative or receipt.get("config_sha256") != sha(config_path):
        raise ValueError("configuration freeze mismatch")
    review = receipt.get("independent_review", {})
    path = _relative(root, review.get("receipt_relative_to_project", ""))
    if review.get("accepted") is not True or not path.is_file() or sha(path) != review.get("receipt_sha256"):
        raise ValueError("independent code/config review receipt required")
    evidence = json.loads(path.read_text())
    if evidence.get("implementation_preparation_accepted") is not True or evidence.get("scientific_acceptance") is not False:
        raise ValueError("independent implementation verdict missing")
    for obj in (review, evidence):
        if obj.get("implementation_sha256") != sha(__file__) or obj.get("config_sha256") != sha(config_path):
            raise ValueError("review must bind final executing code and configuration")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True).strip()
    if current != commit or dirty:
        raise ValueError("clean immutable committed source required")
    for file in (relative, str(Path(__file__).resolve().relative_to(root)), *PINNED_SOURCES):
        if subprocess.check_output(["git", "show", f"{commit}:{file}"], cwd=root) != (root / file).read_bytes():
            raise ValueError("executing source/config differs from commit")
    for file, digest in PINNED_SOURCES.items():
        if sha(root / file) != digest:
            raise ValueError("unchanged native source pin differs")
    return {"source_commit": commit, "config_sha256": sha(config_path),
            "freeze_receipt_sha256": sha(receipt_path), "review_receipt_sha256": sha(path)}


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class CallBudget:
    maximum: int
    wall_seconds: float
    started: float
    attempted: int = 0
    completed: int = 0
    stage: str = "unscoped"
    by_stage: dict = field(default_factory=dict)

    def check(self):
        if time.monotonic() - self.started >= self.wall_seconds:
            raise BudgetExhausted("shared wall deadline")

    def before(self):
        self.check()
        if self.attempted >= self.maximum:
            raise BudgetExhausted("native-core call ceiling")
        self.attempted += 1
        self.by_stage[self.stage]=self.by_stage.get(self.stage,0)+1

    def after(self):
        self.completed += 1
        self.check()


@contextmanager
def counted_core(model, budget):
    before = model.core.register_forward_pre_hook(lambda *_: budget.before())
    after = model.core.register_forward_hook(lambda *_: budget.after())
    try:
        yield
    finally:
        before.remove(); after.remove()


def _copy(state):
    return tuple(s.detach().clone() for s in state) if isinstance(state, tuple) else state.detach().clone()


def _components(state, kind):
    return {"hidden": state[0][0], "cell": state[1][0]} if kind == "lstm" else {"hidden": state[0]}


def _fields(encoded, prefix, state, logits, kind):
    fields = {"prefix_encoding": encoded[:, :16], "query_encoding": encoded[:, -1], "logits": logits}
    fields.update({"terminal_" + k: v for k, v in _components(state, kind).items()})
    fields.update({"prefix_" + k: v for k, v in _components(prefix, kind).items()})
    return fields


@torch.no_grad()
def native_paths(model, inputs, budget, paths=("P0", "P1", "P2", "P3", "P4")):
    """Single P0 hook observation; P3 reuses P1, no hidden forwards."""
    if tuple(paths) not in (("P0", "P1", "P2", "P3", "P4"), ("P0", "P1", "P4")):
        raise ValueError("fixed path schedule required")
    kind = model.model_kind; capture = {}
    with counted_core(model, budget):
        eh = model.encoder.register_forward_hook(lambda _m, _i, o: capture.update(encoded=o))
        ch = model.core.register_forward_hook(lambda _m, _i, o: capture.update(output=o[0], state=o[1]))
        try:
            logits = model(inputs)
        finally:
            eh.remove(); ch.remove()
        encoded = capture["encoded"]; full = capture["output"]; state = capture["state"]
        p0 = {"prefix_encoding": encoded[:, :16], "query_encoding": encoded[:, -1],
              "logits": logits, "prefix_hidden": full[:, 15], "sequence_hidden": full}
        p0.update({"terminal_" + k: v for k, v in _components(state, kind).items()})
        output, origin = model.core(encoded[:, :16])
        output, terminal = model.core(encoded[:, -1:], _copy(origin))
        result = {"P0": p0, "P1": _fields(encoded, origin, terminal, model.head(output[:, -1]), kind)}
        if "P2" in paths:
            prefix_encoded = model.encoder(inputs[:, :16])
            query_encoded = model.encoder(inputs[:, -1][:, None])
            _, separate_origin = model.core(prefix_encoded)
            qoutput, terminal = model.core(query_encoded, _copy(separate_origin))
            result["P2"] = _fields(encoded, separate_origin, terminal, model.head(qoutput[:, -1]), kind)
            result["P2"]["prefix_encoding"] = prefix_encoded
            result["P2"]["query_encoding"] = query_encoded[:, 0]
            query_capture = {}
            eh = model.encoder.register_forward_hook(lambda _m, _i, o: query_capture.update(encoded=o))
            ch = model.core.register_forward_hook(lambda _m, _i, o: query_capture.update(state=o[1]))
            try:
                read_logits = model.read_query(inputs[:, -1], _copy(origin))
            finally:
                eh.remove(); ch.remove()
            p3 = _fields(encoded, origin, query_capture["state"], read_logits, kind)
            p3["query_encoding"] = query_capture["encoded"][:, 0]
            result["P3"] = p3
        state = None; sequence = []; prefix = None
        for step in range(17):
            qoutput, state = model.core(encoded[:, step:step+1], state)
            parts = _components(state, kind)
            sequence.append(torch.cat(list(parts.values()), 1))
            if step == 15:
                prefix = _copy(state)
        p4 = _fields(encoded, prefix, state, model.head(qoutput[:, -1]), kind)
        p4["sequence_complete"] = torch.stack(sequence, 1)
        result["P4"] = p4
    return result


def _array(value):
    return value.detach().cpu().contiguous().numpy()


def ordered_ulp_distances(candidate, reference):
    """Python integer subtraction avoids unsigned overflow; ±0 distance zero."""
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.dtype != b.dtype or a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("ULP operands require the same IEEE float dtype")
    if a.size==0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("ULP requires matched finite arrays")
    bits = 32 if a.dtype == np.float32 else 64
    uint = np.uint32 if bits == 32 else np.uint64
    sign, mask = 1 << (bits - 1), (1 << bits)-1
    def key(value):
        # +1 collapses -0 and +0 globally, also for intervals crossing zero.
        return ((~value & mask)+1) if value & sign else value | sign
    values = [0 if x == 0 and y == 0 else abs(key(int(u))-key(int(v)))
              for x, y, u, v in zip(a.ravel(), b.ravel(), a.view(uint).ravel(), b.view(uint).ravel())]
    return np.asarray(values, dtype=np.uint64).reshape(a.shape)


def compare(candidate, reference, tolerance_dtype, *, bitwise=False):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.size==0 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError("nonfinite or mismatched required observable")
    left, right = a.astype(np.float64), b.astype(np.float64)
    with np.errstate(over="raise",invalid="raise"):
        gap = np.abs(left-right)
    atol, rtol = TOLERANCES[tolerance_dtype]
    bound = atol+rtol*np.abs(right); failures = gap > bound
    with np.errstate(over="ignore",divide="ignore",invalid="ignore"):
        threshold_ratio=gap/bound
        relative=gap[right!=0]/np.abs(right[right!=0])
    threshold_overflow=int(np.count_nonzero(~np.isfinite(threshold_ratio)))
    relative_overflow=int(np.count_nonzero(~np.isfinite(relative)))
    # np.array_equal treats signed zero as equal; bitwise identity uses raw bytes.
    exact = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    record = {"passed": exact if bitwise else not bool(failures.any()), "entries": a.size,
              "bitwise_equal": exact, "violation_count": int(failures.sum()),
              "violation_indices": np.argwhere(failures).tolist(),
              "max_absolute_difference": float(gap.max(initial=0)),
              "max_threshold_ratio": None if threshold_overflow else float(threshold_ratio.max()),
              "threshold_ratio_overflow_count": threshold_overflow,
              "absolute_quantiles": np.quantile(gap, [0, .5, .95, 1]).tolist(),
              "relative_undefined_at_zero_count": int(np.count_nonzero(right == 0)),
              "relative_max_on_nonzero": None if relative_overflow or not relative.size else float(relative.max()),
              "relative_ratio_overflow_count": relative_overflow,
              "unrepresentable_ratio_policy": "null summary plus count; finite absolute residuals and true violations retained",
              "signed_difference_min": float((left-right).min()),
              "signed_difference_max": float((left-right).max()),
              "atol": atol, "rtol": rtol, "rule": "bitwise" if bitwise else "abs(candidate-reference)<=atol+rtol*abs(reference)"}
    if a.dtype == b.dtype:
        ulp = ordered_ulp_distances(a, b)
        record["max_ordered_ulp_distance"] = int(ulp.max(initial=0))
        record["signed_zero_policy"] = "ULP distance zero; bitwise identity remains separate"
    else:
        record["max_ordered_ulp_distance"] = None
        record["ulp_unavailable_reason"] = "cross-dtype; no artificial quantization"
    return record


def validate_observables(arrays, kind, task):
    if kind not in ("gru","lstm") or task not in ("categorical","continuous"):
        raise ValueError("explicit native model/task observability required")
    if set(arrays) not in ({"P0","P1","P2","P3","P4"},{"P0","P1","P4"}):
        raise ValueError("required native path missing")
    for path,fields in arrays.items():
        required={"prefix_encoding","query_encoding","logits","prefix_hidden","terminal_hidden"}
        if path=="P0":required.add("sequence_hidden")
        if path=="P4":required.add("sequence_complete")
        if kind=="lstm":
            required.add("terminal_cell")
            if path!="P0":required.add("prefix_cell")
        if set(fields)!=required:
            raise ValueError("missing/invented observable for "+kind+"/"+path)
        rows=fields["logits"].shape[0];hidden=fields["terminal_hidden"].shape[1]
        if fields["logits"].shape!=(rows,16 if task=="categorical" else 1):
            raise ValueError("native head shape changed")
        for field,value in fields.items():
            if value.shape[0]!=rows or not np.isfinite(value).all():
                raise FloatingPointError("invalid required observable")
            if field in ("prefix_hidden","terminal_hidden","prefix_cell","terminal_cell") and value.shape!=(rows,hidden):
                raise ValueError("native h/c component shape changed")
        if path=="P4" and fields["sequence_complete"].shape!=(rows,17,hidden*(2 if kind=="lstm" else 1)):
            raise ValueError("complete native one-step state shape changed")


def directed_checks(arrays, dtype, kind, task):
    validate_observables(arrays,kind,task)
    checks = []
    def edge(cp, cf, rp, rf, exact=False):
        checks.append({"candidate": cp+"/"+cf, "reference": rp+"/"+rf,
                       **compare(arrays[cp][cf], arrays[rp][rf], dtype, bitwise=exact)})
    for path in arrays:
        if path != "P0":
            for field in ("logits", "terminal_hidden", "terminal_cell"):
                if field in arrays["P0"]:
                    edge(path, field, "P0", field)
    for path in ("P2", "P4"):
        if path in arrays:
            for field in ("prefix_hidden", "prefix_cell"):
                if field in arrays["P1"]:
                    edge(path, field, "P1", field)
    edge("P1", "prefix_hidden", "P0", "prefix_hidden")
    if "P2" in arrays:
        edge("P2", "prefix_encoding", "P0", "prefix_encoding")
        for path in ("P2", "P3"):
            edge(path, "query_encoding", "P0", "query_encoding")
        for field in ("prefix_hidden", "prefix_cell"):
            if field in arrays["P1"]:
                edge("P3", field, "P1", field, True)
    for path in ("P1", "P4"):
        edge(path, "prefix_encoding", "P0", "prefix_encoding", True)
        edge(path, "query_encoding", "P0", "query_encoding", True)
    return checks


def validate_copy_observables(arrays, kind, task):
    required={"logits","prefix_hidden","terminal_hidden"}
    if kind=="lstm":required.update(("prefix_cell","terminal_cell"))
    if set(arrays)!={"original","clone","twice_clone","contiguous_clone"}:
        raise ValueError("complete copy-null schedule missing")
    for fields in arrays.values():
        if set(fields)!=required:raise ValueError("missing/invented complete copy-null component")
        for value in fields.values():
            if not value.size or not np.isfinite(value).all():raise FloatingPointError("invalid complete copy-null observable")
        rows,hidden=fields["terminal_hidden"].shape
        if fields["logits"].shape!=(rows,16 if task=="categorical" else 1):raise ValueError("copy-null head changed")
        if any(v.shape!=(rows,hidden) for k,v in fields.items() if k!="logits"):
            raise ValueError("copy-null h/c component shape changed")


def registry():
    """Pure call schedule; no native calls or reference loads."""
    calls = {"primary": len(CELLS)*2*2*2*2*sum(64//b for b in (64,32,16,8,1))*23,
             "known_regression": (2048//64)*2*2*23,
             "N1": len(CELLS)*2*2*23, "N2": len(CELLS)*2*2*20,
             "N3": len(CELLS)*2*2*5}
    assert calls == configuration_contract()["core_call_registry"] and sum(calls.values()) == 120000
    return calls


def case_registry(fixture=None):
    """Fixed complete condition ledger, without loading inputs or models."""
    cells=CELLS if fixture is None else (fixture.cell,)
    count=64 if fixture is None else fixture.histories
    weights=(0,17) if fixture is None else (0,)
    seeds=(30013,30017) if fixture is None else (30013,)
    chunks=(64,32,16,8,1) if fixture is None else (count,)
    stages=["primary"]+(["known_regression","N1","N2","N3"] if fixture is None else (["N1","N2","N3"] if fixture.include_nulls else []))
    result=[]
    for stage in stages:
        if stage=="primary":contexts=[(m,t,w,s,count) for m,t in cells for w in weights for s in seeds];stage_chunks=chunks
        elif stage=="known_regression":contexts=[("gru","categorical",0,0,2048)];stage_chunks=(64,)
        else:contexts=[(m,t,0,30013,count) for m,t in cells];stage_chunks=(count,)
        for m,t,w,s,rows in contexts:
            for dtype in ("float64","float32"):
                for repeat in (1,2):
                    for chunk in stage_chunks:
                        calls={"N1":23,"N2":20,"N3":5}.get(stage,23)*(rows//chunk)
                        result.append({"id":f"{stage}-{m}-{t}-w{w}-s{s}-{dtype}-r{repeat}-b{chunk}",
                            "stage":stage,"rows":rows,"chunk":chunk,"dtype":dtype,"repeat":repeat,"core_calls":calls})
    return result


@dataclass(frozen=True)
class FixtureSpec:
    cell: tuple[str, str] = ("gru", "categorical")
    histories: int = 2
    hidden: int = 5
    encoder: int = 4
    include_nulls: bool = False
    maximum_calls: int = 500
    wall_seconds: float = 120.0

    def validate(self):
        if self.cell not in CELLS or any(type(v) is not int or v < 1 for v in (self.histories,self.hidden,self.encoder,self.maximum_calls)):
            raise ValueError("explicit positive tiny synthetic fixture dimensions required")
        if self.histories > 2 or self.hidden > 5 or self.encoder > 4 or self.maximum_calls > 500:
            raise ValueError("fixture must be strictly tiny and at most500 core calls")
        if type(self.include_nulls) is not bool or not 0 < self.wall_seconds <= 120:
            raise ValueError("fixture runtime/scope expansion rejected")
        if self.include_nulls and self.histories!=2:
            raise ValueError("paired-copy tiny nulls require exactly2 rows")
        if isinstance(self.wall_seconds,bool) or not np.isfinite(self.wall_seconds):
            raise ValueError("finite real fixture deadline required")


def synthetic_inputs(task, seed, count):
    # Payloads are solely WRITE inputs. No addressed target is extracted.
    memory = sample_memory(np.random.default_rng(seed), count, task)
    return render(memory, np.arange(count, dtype=np.int64)%8, [""]*count, task)


def _tensor_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(json.dumps({"dtype":str(a.dtype),"shape":list(a.shape)},sort_keys=True).encode()+a.tobytes()).hexdigest()


def _weights_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(_tensor_hash(_array(value)).encode())
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path, arrays):
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def _flat(arrays):
    return {path+"__"+field: value for path, fields in arrays.items() for field,value in fields.items()}


def _read_arrays(path):
    record=json.loads(path.with_suffix(".json").read_text())
    if sha(path)!=record["arrays_sha256"]:
        raise ValueError("stored directed reference changed")
    result={}
    with np.load(path,allow_pickle=False) as saved:
        for key in saved.files:
            p,field=key.split("__",1);result.setdefault(p,{})[field]=saved[key]
    return result


def _model(kind, task, seed, hidden, encoder, dtype, null=None, known=None):
    torch.manual_seed(seed)
    original_default=torch.get_default_dtype();torch.set_default_dtype(torch.float32)
    try:
        with torch.device("cpu"):model=VectorMemoryModel(kind,task,hidden,encoder)
    finally:torch.set_default_dtype(original_default)
    if known is not None:
        model.load_state_dict(known)
    if null=="N1":
        with torch.no_grad():model.head.weight.zero_();model.head.bias.zero_()
    elif null=="N2":
        with torch.no_grad():
            for p in model.parameters():p.zero_()
    originals={k:v.detach().clone() for k,v in model.state_dict().items()}
    fp32_hash=_weights_hash(model)
    model=model.to(dtype=DTYPES[dtype],device="cpu").eval()
    promotion=all(torch.equal(model.state_dict()[k],v.to(dtype=DTYPES[dtype])) for k,v in originals.items())
    if not promotion:raise ValueError("exact FP32 parameter promotion failed")
    model._chronology_fp32_weight_sha256=fp32_hash
    model._chronology_parameter_promotion_verified=promotion
    return model


def _copy_null(model, inputs, budget):
    with torch.no_grad(),counted_core(model,budget):
        encoded=model.encoder(inputs);_,origin=model.core(encoded[:,:16])
        before={k:_array(v).copy() for k,v in _components(origin,model.model_kind).items()}
        copies=[origin,_copy(origin),_copy(_copy(origin)),tuple(s.detach().clone().contiguous() for s in origin) if isinstance(origin,tuple) else origin.detach().clone().contiguous()]
        arrays={}
        for name,state in zip(("original","clone","twice_clone","contiguous_clone"),copies):
            out,terminal=model.core(encoded[:,-1:],state)
            fields={"logits":_array(model.head(out[:,-1]))}
            fields.update({"terminal_"+k:_array(v) for k,v in _components(terminal,model.model_kind).items()})
            fields.update({"prefix_"+k:_array(v) for k,v in _components(state,model.model_kind).items()})
            arrays[name]=fields
        after={k:_array(v) for k,v in _components(origin,model.model_kind).items()}
    validate_copy_observables(arrays,model.model_kind,model.task)
    checks=[];dtype=str(inputs.dtype).split(".")[-1]
    for k in before:
        checks.append({"candidate":"origin_after/"+k,"reference":"origin_before/"+k,**compare(after[k],before[k],dtype,bitwise=True)})
    for name in arrays:
        if name!="original":
            for field in arrays[name]:
                checks.append({"candidate":name+"/"+field,"reference":"original/"+field,
                    **compare(arrays[name][field],arrays["original"][field],dtype,bitwise=field.startswith("prefix_"))})
    return arrays,checks


def _case(output, identifier, model, inputs, chunk, repeat, dtype, budget, refs, null=None):
    budget.check();start=budget.attempted;parts={};layouts=[]
    weights_before=_weights_hash(model);inputs_before=_tensor_hash(inputs)
    if inputs.dtype!=np.float32:raise ValueError("inputs must originate as exact FP32 values")
    for begin in range(0,len(inputs),chunk):
        tensor=torch.tensor(inputs[begin:begin+chunk],dtype=DTYPES[dtype])
        tensor_before=_tensor_hash(_array(tensor))
        if not np.array_equal(_array(tensor),inputs[begin:begin+chunk].astype(_array(tensor).dtype)):
            raise ValueError("exact FP32 input promotion failed")
        layouts.append({"row_start":begin,"shape":list(tensor.shape),"stride":list(tensor.stride()),"contiguous":tensor.is_contiguous()})
        if null=="N3":
            arrays,local_checks=_copy_null(model,tensor,budget)
        else:
            native=native_paths(model,tensor,budget,paths=("P0","P1","P4") if null=="N2" else ("P0","P1","P2","P3","P4"))
            layouts[-1]["path_fields"]={p:{f:{"shape":list(v.shape),"stride":list(v.stride()),"contiguous":v.is_contiguous(),"dtype":str(v.dtype)} for f,v in fields.items()} for p,fields in native.items()}
            arrays={p:{f:_array(v) for f,v in fields.items()} for p,fields in native.items()}
            local_checks=[]
        for p,fields in arrays.items():
            for field,value in fields.items():
                parts.setdefault(p,{}).setdefault(field,[]).append(value)
        if _tensor_hash(_array(tensor))!=tensor_before:raise ValueError("native input mutated")
    arrays={p:{f:np.concatenate(v,axis=0) for f,v in fields.items()} for p,fields in parts.items()}
    checks=local_checks if null=="N3" else directed_checks(arrays,dtype,model.model_kind,model.task)
    for relation,reference in refs:
        previous=_read_arrays(output/reference)
        if null!="N3":validate_observables(previous,model.model_kind,model.task)
        else:validate_copy_observables(previous,model.model_kind,model.task)
        for p,fields in arrays.items():
            for field,value in fields.items():
                if p not in previous or field not in previous[p]:
                    raise ValueError("missing required directed reference observable")
                checks.append({"candidate":p+"/"+field,"reference":reference+":"+p+"/"+field,"relation":relation,
                    **compare(value,previous[p][field],dtype,bitwise=relation=="repeat")})
    if null in ("N1","N2"):
        for p,fields in arrays.items():
            for field,value in fields.items():
                if field=="logits" or (null=="N2" and ("hidden" in field or "cell" in field or field=="sequence_complete")):
                    checks.append({"candidate":p+"/"+field,"reference":"exact_zero_null",**compare(value,np.zeros_like(value),dtype,bitwise=True)})
                if null=="N1":
                    half=len(value)//2
                    checks.append({"candidate":p+"/"+field+"/duplicate_half","reference":p+"/"+field+"/first_half",**compare(value[half:],value[:half],dtype)})
    flat=_flat(arrays)
    if _weights_hash(model)!=weights_before or _tensor_hash(inputs)!=inputs_before:
        raise ValueError("weights/inputs changed during fixed-path probe")
    file=identifier+".npz";_atomic_npz(output/file,flat)
    unavailable={"P0/prefix_cell":{"observed":False,"reason":"native fused full17 LSTM exposes terminal c only"}} if model.model_kind=="lstm" and null!="N3" else {}
    record={"id":identifier,"dtype":dtype,"chunk":chunk,"repeat":repeat,"model":model.model_kind,"task":model.task,
        "attempted_core_calls":budget.attempted-start,"inputs_sha256":inputs_before,"weights_sha256":weights_before,
        "fp32_origin_weight_sha256":model._chronology_fp32_weight_sha256,
        "parameter_promotion_verified":model._chronology_parameter_promotion_verified,
        "input_promotion_verified":True,"before_after_weight_input_hashes_equal":True,
        "input_layouts":layouts,"arrays":file,"arrays_sha256":sha(output/file),"unavailable":unavailable,
        "checks":checks,"required_comparisons_passed":all(c["passed"] for c in checks),"targets_read":False}
    _atomic_json(output/(identifier+".json"),record);budget.check()
    return record,file


def _known(config,root,budget):
    budget.check()
    paths={k:_relative(root,config[k]["path"]) for k in ("known_checkpoint","known_inputs")}
    for k,p in paths.items():
        if sha(p)!=config[k]["sha256"]:raise ValueError("known numerical regression pin mismatch")
        budget.check()
    checkpoint=torch.load(paths["known_checkpoint"],map_location="cpu",weights_only=True)
    if checkpoint.get("step")!=4000:raise ValueError("known checkpoint must be exact current4000")
    budget.check()
    with np.load(paths["known_inputs"],allow_pickle=False) as saved:
        inputs=saved["inputs"].copy()  # No target/label/score array is accessed.
        pair_ids=saved["pair_ids"].copy()
    if inputs.dtype!=np.float32 or inputs.shape!=(2048,17,38) or pair_ids.shape!=(2048,) or not np.isfinite(inputs).all():
        raise ValueError("known input geometry/precision changed")
    budget.check();return checkpoint["model_state_dict"],inputs,pair_ids


def _environment():
    return {"torch":str(torch.__version__),"numpy":np.__version__,"python":platform.python_version(),
        "platform":platform.platform(),"machine":platform.machine(),"cpu_threads":torch.get_num_threads(),
        "interop_threads":torch.get_num_interop_threads(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "mkldnn_available":torch.backends.mkldnn.is_available(),"mkldnn_enabled":torch.backends.mkldnn.enabled,
        "cudnn_enabled":torch.backends.cudnn.enabled,"cudnn_deterministic":torch.backends.cudnn.deterministic,
        "cudnn_benchmark":torch.backends.cudnn.benchmark,"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn":torch.backends.cudnn.allow_tf32,"matmul_precision":torch.get_float32_matmul_precision(),
        "torch_build_configuration":torch.__config__.show(),
        "native_python_rnn_source_sha256":sha(torch.nn.modules.rnn.__file__),
        "subnormal_flush_mode":"unknown; not inferred from integer ULP arithmetic"}


def _source_provenance(root):
    actual={p:sha(root/p) for p in PINNED_SOURCES}
    if actual!=PINNED_SOURCES:raise ValueError("unchanged native source identity mismatch")
    return {**actual,"src/representation_collapse/chronology_calibration.py":sha(__file__),
            "uv.lock":sha(root/"uv.lock")}


def _engine(config, output, root, identity, fixture, _began=None):
    began=time.monotonic() if _began is None else _began
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    maximum=120000 if fixture is None else fixture.maximum_calls
    wall=600 if fixture is None else fixture.wall_seconds
    budget=CallBudget(maximum,wall,began);records=[];status={"protocol_id":PROTOCOL,"fixture_only":fixture is not None,
        "state":"running","scientific_acceptance":False,"training_authorized":False,"canonical_only_admitted":False,
        "targets_read":False,"optimizer_updates":0,"native_core_calls_attempted":0,"matrix_completed":False,
        "strict_interoperability_passed":False,"source_identity":identity}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    _atomic_json(output/"environment.json",_environment());_atomic_json(output/"config.json",config)
    planned=case_registry(fixture);expected=sum(c["core_calls"] for c in planned)
    _atomic_json(output/"condition-registry.json",{"conditions":planned,"planned_core_calls":expected})
    stage_actual={};current=None
    try:
        budget.check();sources_before=_source_provenance(root)
        _atomic_json(output/"source-provenance.json",sources_before)
        cells=CELLS if fixture is None else (fixture.cell,)
        count=64 if fixture is None else fixture.histories;hidden=128 if fixture is None else fixture.hidden;encoder=64 if fixture is None else fixture.encoder
        weight_seeds=(0,17) if fixture is None else (0,);input_seeds=(30013,30017) if fixture is None else (30013,)
        chunks=(64,32,16,8,1) if fixture is None else (count,)
        stages=["primary"]+(["known_regression","N1","N2","N3"] if fixture is None else (["N1","N2","N3"] if fixture.include_nulls else []))
        known_state=known_inputs=None
        for stage in stages:
            budget.stage=stage
            stage_start=budget.attempted
            if stage=="known_regression":
                known_state,known_inputs,pair_ids=_known(config,root,budget)
                _atomic_npz(output/"known-input-identities.npz",{"row_ids":np.arange(2048),"pair_ids":pair_ids})
                contexts=[("gru","categorical",0,0,known_inputs)];stage_chunks=(64,)
            elif stage=="primary":
                contexts=[(m,t,w,s,synthetic_inputs(t,s,count)) for m,t in cells for w in weight_seeds for s in input_seeds]
                stage_chunks=chunks
            else:
                contexts=[];stage_chunks=(count,)
                for m,t in cells:
                    inputs=synthetic_inputs(t,30013,count)
                    if stage=="N2":inputs=np.zeros_like(inputs)
                    else:
                        # Full null has32prefixes duplicated64; tiny fixture1prefix duplicated2.
                        inputs=np.concatenate((inputs[:count//2],inputs[:count//2]))
                    contexts.append((m,t,0,30013,inputs))
            for m,t,w,s,inputs in contexts:
                base=f"{stage}-{m}-{t}-w{w}-s{s}"
                _atomic_npz(output/(base+"-inputs.npz"),{"inputs":inputs,"row_ids":np.arange(len(inputs),dtype=np.int64)})
                files={}
                for dtype in ("float64","float32"):
                    for repeat in (1,2):
                        for chunk in stage_chunks:
                            budget.check();current=f"{base}-{dtype}-r{repeat}-b{chunk}"
                            model=_model(m,t,w,hidden,encoder,dtype,null=stage if stage in ("N1","N2") else None,
                                         known=known_state if stage=="known_regression" else None)
                            _atomic_json(output/(current+"-intent.json"),{"weights_sha256":_weights_hash(model),
                                "fp32_origin_weight_sha256":model._chronology_fp32_weight_sha256,"inputs_sha256":_tensor_hash(inputs),
                                "dtype":dtype,"stage":stage,"targets_read":False,"state":"prepared_before_native_calls"})
                            refs=[]
                            if repeat==2:refs.append(("repeat",files[(dtype,1,chunk)]))
                            if chunk!=stage_chunks[0]:refs.append(("cross_chunk",files[(dtype,repeat,stage_chunks[0])]))
                            if dtype=="float32":refs.append(("cross_dtype",files[("float64",repeat,chunk)]))
                            record,file=_case(output,current,model,inputs,chunk,repeat,dtype,budget,refs,null=stage if stage.startswith("N") else None)
                            files[(dtype,repeat,chunk)]=file;record["stage"]=stage;records.append(record)
                            status.update(native_core_calls_attempted=budget.attempted,cases_completed=len(records))
                            _atomic_json(output/"status.json",status)
            stage_actual[stage]=budget.attempted-stage_start
        if budget.attempted!=expected or budget.completed!=expected:raise ValueError("actual core-call registry mismatch")
        if fixture is None and stage_actual!=registry():raise ValueError("production stage counts differ from fixed registry")
        if _source_provenance(root)!=sources_before:raise ValueError("executing/native source changed during matrix")
        status.update(state="diagnostic_matrix_completed",matrix_completed=True,
                      strict_interoperability_passed=all(r["required_comparisons_passed"] for r in records))
    except BudgetExhausted as error:
        status.update(state="incomplete_censored_budget",error=str(error),current_case=current)
    except Exception as error:
        status.update(state="failed_execution_identity_or_numerics",error_type=type(error).__name__,error=str(error),current_case=current)
    finally:
        complete_ids={r["id"] for r in records}
        interrupted=current if current not in complete_ids else None
        stage_actual={s:budget.by_stage.get(s,0) for s in dict.fromkeys(c["stage"] for c in planned)}
        status.update(native_core_calls_attempted=budget.attempted,native_core_calls_completed=budget.completed,
            expected_core_calls=expected,stage_actual_core_calls=stage_actual,cases_completed=len(records),elapsed_seconds=time.monotonic()-began,
            planned_condition_count=len(planned),interrupted_condition_id=interrupted,
            unstarted_conditions=[c for c in planned if c["id"] not in complete_ids and c["id"]!=interrupted])
        _atomic_json(output/"results.json",{"fixture_only":fixture is not None,"scientific_acceptance":False,"targets_read":False,
            "records":records,"known_source_failure_not_waived":True,
            "known_regression_completed":stage_actual.get("known_regression")==2944})
        _atomic_json(output/"status.json",status)
        _atomic_json(output/"manifest.json",{"files":{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="manifest.json"}})
        if time.monotonic()-began>=wall:
            status.update(state="incomplete_censored_budget",matrix_completed=False,strict_interoperability_passed=False,
                          finalization_exceeded_deadline=True,elapsed_seconds=time.monotonic()-began)
            _atomic_json(output/"status.json",status)
            _atomic_json(output/"manifest.json",{"files":{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!="manifest.json"}})
    return status


def run(config_path, output, freeze_receipt):
    began=time.monotonic()
    config=json.loads(Path(config_path).read_text());validate_config(config)
    root=Path(__file__).resolve().parents[2]
    identity=verify_freeze(config_path,freeze_receipt,root)
    if torch.__version__.split("+")[0]!="2.6.0":raise ValueError("frozen Torch2.6.0 required")
    return _engine(config,output,root,identity,None,_began=began)


def run_fixture(output, spec=FixtureSpec()):
    spec.validate();root=Path(__file__).resolve().parents[2]
    # This branch never invokes _known/torch.load/validationnp.load.
    return _engine(configuration_contract(),output,root,{"kind":"tiny_synthetic_fixture","implementation_sha256":sha(__file__)},spec)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--freeze-receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    status=run(args.config,args.output,args.freeze_receipt)
    print(json.dumps({k:status[k] for k in ("state","matrix_completed","strict_interoperability_passed","native_core_calls_attempted")}))
    if not status["matrix_completed"]:raise SystemExit(1)


if __name__=="__main__":main()
