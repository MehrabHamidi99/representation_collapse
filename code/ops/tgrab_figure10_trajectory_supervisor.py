#!/usr/bin/env python3
"""Admitted descriptive trajectory analysis, sharing reviewed CPU ownership helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import types

PROTOCOL_ID = "figure10_descriptive_state_trajectory_v1"
HELPER_SHA256 = "3023286daa4d9142fe43c6c1c8898fc7d7cc1060ddd8e191a440f8f6c0638e1d"
WORKER = "src/representation_collapse/tgrab_figure10_trajectory_analysis.py"
LONG_CONTRACT_SHA256 = "aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381"
LEGACY_ACCEPTANCE = "figure10_selected_evaluation_execution_acceptance_v2"
PRIMITIVE_V2_ACCEPTANCE = "figure10_primitive_v2_selected_evaluation_execution_acceptance_v3"


def validate_evaluation_acceptance(acceptance, h):
    """Dispatch actual receipt versions without inventing legacy identities."""
    if (type(acceptance) is not dict
            or acceptance.get("kind") not in (LEGACY_ACCEPTANCE, PRIMITIVE_V2_ACCEPTANCE)
            or acceptance.get("status") != "accepted"
            or acceptance.get("terminal_replay_verified") is not True
            or acceptance.get("test_rows_verified") is not True):
        h.fail("independent selected evaluation acceptance required")
    for key in ("evaluation_result_sha256", "trajectory_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(acceptance.get(key, ""))):
            h.fail("invalid accepted evaluation hash: " + key)
    for key in ("evaluation_result_path", "trajectory_path"):
        if type(acceptance.get(key)) is not str or not Path(acceptance[key]).is_absolute():
            h.fail("absolute accepted evaluation path required: " + key)
    if acceptance["kind"] == PRIMITIVE_V2_ACCEPTANCE:
        if (acceptance.get("checkpoint_format") != "primitive_v2"
                or acceptance.get("executed_long_contract_sha256") != LONG_CONTRACT_SHA256
                or "final_head_sha256" in acceptance):
            h.fail("primitive-v2 acceptance identity mismatch")
        keys = ("candidate_manifest_sha256", "selected_pointer_sha256")
    else:
        if any(key in acceptance for key in ("checkpoint_format", "selected_pointer_sha256", "executed_long_contract_sha256")):
            h.fail("mixed legacy and primitive-v2 acceptance identities")
        keys = ("checkpoint_sha256", "final_head_sha256")
    for key in keys:
        if not re.fullmatch(r"[0-9a-f]{64}", str(acceptance.get(key, ""))):
            h.fail("invalid accepted checkpoint hash: " + key)


def validate_result_binding(result, acceptance, h):
    descriptor = result.get("state_trajectory", {})
    if (type(descriptor) is not dict or descriptor.get("sha256") != acceptance["trajectory_sha256"]
            or descriptor.get("filename") != Path(acceptance["trajectory_path"]).name
            or Path(acceptance["trajectory_path"]).parent != Path(acceptance["evaluation_result_path"]).parent):
        h.fail("accepted result trajectory binding differs")
    keys = (("checkpoint_format", "executed_long_contract_sha256", "selected_pointer_sha256", "candidate_manifest_sha256")
            if acceptance["kind"] == PRIMITIVE_V2_ACCEPTANCE else ("checkpoint_sha256", "final_head_sha256"))
    # The legacy result has no head field; that identity remains in its pinned receipt.
    for key in keys:
        if key == "final_head_sha256":
            continue
        if result.get(key) != acceptance[key]:
            h.fail("accepted evaluation result identity differs: " + key)
    if acceptance["kind"] == PRIMITIVE_V2_ACCEPTANCE and "final_head_sha256" in result:
        h.fail("primitive-v2 result cannot carry a legacy final head")


def helper():
    path = Path(__file__).with_name("tgrab_figure10_canary_supervisor.py")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA256:
        raise RuntimeError("reviewed CPU supervisor helpers changed")
    module = types.ModuleType("_reviewed_prefix_supervisor")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def source_closure(repo: Path) -> dict[str, str]:
    paths = ["code/ops/tgrab_figure10_trajectory_job.sh",
             "code/ops/tgrab_figure10_trajectory_supervisor.py",
             "code/ops/tgrab_figure10_canary_supervisor.py",
             "configs/graph/tgrab-reconstruction-v1.json",
             "configs/graph/tgrab-selected-checkpoint-schema-v1.json",
             "configs/graph/tgrab-selected-tensor-catalog-v1.json"]
    paths.append("doc/auto/figure10-trajectory-diagnostics-2026-10-19.md")
    paths += [str(p.relative_to(repo)) for p in
              sorted((repo / "src/representation_collapse").glob("*.py"))]
    h = helper()
    return {name: h.sha256_path(repo / name) for name in paths}


def validate_admission(d, h, *, actual=None, memory=None):
    if (type(d) is not dict or d.get("schema") != "tgrab-figure10-trajectory-admission-v1"
            or d.get("protocol_id") != PROTOCOL_ID
            or not re.fullmatch(r"[0-9a-f]{40}", str(d.get("source_commit", "")))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", str(d.get("run_id", "")))):
        h.fail("invalid trajectory analysis admission identity")
    expected = {"account": "mila", "partition": "main-cpu", "qos": "normal",
                "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
                "time_limit": "00:10:00", "gpu_allocation": "0",
                "memory_max_bytes": 2 * 1024**3}
    actual = h.scheduler_identity() if actual is None else actual
    memory = h.cgroup_memory() if memory is None else memory
    job = d.get("job")
    if type(job) is not dict or any(str(job.get(k)) != str(v) for k, v in expected.items()):
        h.fail("trajectory analysis job differs from admitted CPU envelope")
    for key in expected:
        if key != "memory_max_bytes" and str(actual.get(key)) != str(job[key]):
            h.fail("actual trajectory analysis allocation differs: " + key)
    if memory["effective_limit_bytes"] != expected["memory_max_bytes"]:
        h.fail("actual trajectory analysis memory ceiling differs")
    if job.get("job_id") not in (None, actual["job_id"]) or job.get("node_list") not in (None, actual["node_list"]):
        h.fail("actual job/node differs from admission")
    if (d.get("work_seconds") != 540 or type(d.get("work_seconds")) is not int
            or d.get("outer_wall_seconds") != 570
            or os.environ.get("RC_JOB_WALL_SECONDS") != "570"
            or "deadline_unix" in d):
        h.fail("invalid trajectory analysis relative timing")
    argv = d.get("canary_args")
    if type(argv) is not list or not all(type(v) is str for v in argv) or len(argv) % 2:
        h.fail("invalid trajectory analysis argument pairs")
    flags = argv[::2]
    if len(set(flags)) != len(flags) or "--deadline-unix" in flags:
        h.fail("duplicate/prequeue deadline flag")
    for flag, value in (("--run-id", d["run_id"]),):
        h._require_arg_pair(argv, flag, value)
    h.regular_bytes(d["environment_receipt"]["path"], d["environment_receipt"]["sha256"])
    acceptance = json.loads(h.regular_bytes(d["evaluation_acceptance"]["path"], d["evaluation_acceptance"]["sha256"]))
    validate_evaluation_acceptance(acceptance, h)
    for key, flag in (("evaluation_result_path", "--evaluation-result"),
                      ("evaluation_result_sha256", "--evaluation-result-sha256"),
                      ("trajectory_path", "--trajectory"), ("trajectory_sha256", "--trajectory-sha256")):
        h._require_arg_pair(argv, flag, acceptance[key])
    result = json.loads(h.regular_bytes(acceptance["evaluation_result_path"], acceptance["evaluation_result_sha256"]))
    validate_result_binding(result, acceptance, h)
    # Numerical trajectory bytes are first opened/hashed inside the entered
    # mandatory reporter. This supervisor authenticates compact metadata only.
    return actual, memory


BOOTSTRAP = r'''import hashlib,importlib.abc,importlib.util,json,os,sys
sys.dont_write_bytecode=True
root=os.environ["RC_PROJECT_ROOT"]; expected=json.loads(os.environ["RC_MODULE_HASHES"])
modules={}
for name in expected:
 if name.startswith("src/representation_collapse/") and name.endswith(".py"):
  module=name[4:-3].replace("/", ".")
  if module.endswith(".__init__"): module=module[:-9]
  modules[module]=name
class Loader(importlib.abc.Loader):
 def __init__(self,name): self.name=name
 def create_module(self,spec): return None
 def exec_module(self,module):
  name=modules[self.name]; filename=os.path.join(root,name); raw=open(filename,"rb").read()
  if hashlib.sha256(raw).hexdigest()!=expected[name]: raise ImportError("admitted source changed: "+name)
  module.__file__=filename; exec(compile(raw,filename,"exec"),module.__dict__)
class Finder(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in modules: return importlib.util.spec_from_loader(fullname,Loader(fullname),is_package=(fullname=="representation_collapse"))
sys.meta_path.insert(0,Finder());sys.path.insert(0,os.path.join(root,"src"))
path=os.environ["RC_WORKER_SOURCE"];raw=open(path,"rb").read()
if hashlib.sha256(raw).hexdigest()!=os.environ["RC_EXPECTED_WORKER_SHA256"]: raise SystemExit("worker source changed")
ns={"__name__":"__main__","__file__":path,"__package__":"representation_collapse"}
exec(compile(raw,path,"exec"),ns,ns)
'''


def run(path, digest):
    h = helper()
    d = json.loads(h.regular_bytes(path, digest))
    repo = Path(__file__).resolve().parents[2]
    if h._git_identity(repo) != d["source_commit"] or source_closure(repo) != d["closure_sha256"]:
        h.fail("trajectory analysis immutable source identity differs")
    h._verify_interpreter(d["interpreter"])
    h._verify_environment_receipt(d["environment_receipt"])
    actual, memory = validate_admission(d, h)
    started_ns = int(os.environ.get("RC_WALL_START_EPOCH_NS", "0"))
    deadline = started_ns / 1e9 + d["work_seconds"]
    if started_ns <= 0 or not math.isfinite(deadline) or not 0 <= time.time() - started_ns / 1e9 < d["work_seconds"]:
        h.fail("invalid trajectory analysis job-start clock")
    args = list(d["canary_args"])
    out = Path(args[args.index("--output") + 1])
    stop = Path(d["stop_file"])
    scratch = Path("/network/scratch/m/mehrab.hamidi/representation-collapse/runs/figure10-trajectory")
    if out.parent != scratch or out.name != d["run_id"] or os.path.lexists(out) or os.path.lexists(stop):
        h.fail("trajectory analysis output must be fresh and project scoped")
    scratch.mkdir(mode=0o700, exist_ok=True)
    control = out.with_name("." + out.name)
    child_argv = [sys.executable, "-B", "-c", BOOTSTRAP, *args]
    argv_sha = h.sha256_bytes(json.dumps(child_argv, separators=(",", ":")).encode())
    receipt = {"schema": "tgrab-figure10-trajectory-supervisor-receipt-v1", "protocol_id": PROTOCOL_ID,
               "run_id": d["run_id"], "admission_sha256": digest, "source_commit": d["source_commit"],
               "closure_sha256": d["closure_sha256"], "source_sha256": d["closure_sha256"][WORKER],
               "job_id": actual["job_id"], "job": {**d["job"], "job_id": actual["job_id"], "node_list": actual["node_list"]},
               "scheduler_record": actual["scheduler_record"], "slurm_environment": actual["slurm_environment"],
               "canary_args": args, "admitted_canary_args": d["canary_args"],
               "child_argv": child_argv, "argv_sha256": argv_sha, "sys_executable": sys.executable,
               "interpreter": d["interpreter"], "environment_receipt": d["environment_receipt"],
               "evaluation_acceptance": d["evaluation_acceptance"],
               "work_seconds": d["work_seconds"], "outer_wall_seconds": d["outer_wall_seconds"],
               "wall_started_epoch_ns": started_ns, "deadline_unix": deadline, "cgroup_memory": memory}
    receipt_path = control.with_name(control.name + ".supervisor-receipt.json")
    h.atomic_json(receipt_path, receipt)
    journal_path = control.with_name(control.name + ".supervisor-journal.jsonl")
    fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    identity = os.fstat(fd); os.close(fd)

    def journal(event, **fields):
        fd = os.open(journal_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino, st.st_nlink) != (identity.st_dev, identity.st_ino, 1):
                h.fail("trajectory analysis journal changed")
            raw = (json.dumps({"time_epoch_ns": time.time_ns(), "event": event, **fields}, sort_keys=True) + "\n").encode()
            while raw:
                n = os.write(fd, raw)
                if n <= 0: raise OSError("short journal write")
                raw = raw[n:]
            os.fsync(fd)
        finally:
            os.close(fd)

    env = {**os.environ, "RC_SUPERVISOR_RECEIPT": str(receipt_path),
           "RC_SUPERVISOR_RECEIPT_SHA256": h.sha256_path(receipt_path),
           "RC_PROJECT_ROOT": str(repo), "RC_WORKER_SOURCE": str(repo / WORKER),
           "RC_EXPECTED_WORKER_SHA256": d["closure_sha256"][WORKER],
           "RC_MODULE_HASHES": json.dumps(d["closure_sha256"]), "PYTHONPATH": str(repo / "src")}
    h.enable_child_subreaper()
    signal_state = {"number": None}
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    def interrupted(number, _frame): signal_state["number"] = number
    for s in old: signal.signal(s, interrupted)
    child = None
    try:
        child = subprocess.Popen(child_argv, cwd=repo, env=env, start_new_session=True)
        journal("child_started", pid=child.pid, argv_sha256=argv_sha)
        # The independent supervisor bounds this compact derived-state analysis.
        status, reason = h.supervise_child(child, stop_path=stop,
            deadline_unix=started_ns / 1e9 + 540, memory=memory,
            memory_limit=memory["effective_limit_bytes"], signal_state=signal_state,
            check_interval=1, journal=journal)
        journal("child_finished", pid=child.pid, returncode=status, stop_reason=reason)
        return status if reason is None else 137
    except BaseException:
        if child is not None: h.terminate_and_reap_group(child, journal=lambda *a, **kw: None)
        raise
    finally:
        for s, handler in old.items(): signal.signal(s, handler)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--admission" or sys.argv[3] != "--sha256":
        raise SystemExit("usage: trajectory_supervisor --admission PATH --sha256 SHA")
    raise SystemExit(run(sys.argv[2], sys.argv[4]))
