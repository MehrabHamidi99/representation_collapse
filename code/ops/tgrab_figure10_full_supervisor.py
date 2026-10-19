#!/usr/bin/env python3
"""Admitted one-hour fit-only execution, sharing reviewed CPU ownership helpers."""
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

PROTOCOL_ID = "tgrab_figure10_full_fit_v1"
HELPER_SHA256 = "3023286daa4d9142fe43c6c1c8898fc7d7cc1060ddd8e191a440f8f6c0638e1d"
WORKER = "src/representation_collapse/tgrab_figure10_full_fit.py"


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
    paths = ["code/ops/tgrab_figure10_full_job.sh",
             "code/ops/tgrab_figure10_full_supervisor.py",
             "code/ops/tgrab_figure10_canary_supervisor.py",
             "configs/graph/tgrab-reconstruction-v1.json",
             "configs/graph/tgrab-selected-checkpoint-schema-v1.json",
             "configs/graph/tgrab-selected-tensor-catalog-v1.json"]
    paths.append("doc/auto/tgrab-primary-data-conversion-v1.md")
    paths += [str(p.relative_to(repo)) for p in
              sorted((repo / "src/representation_collapse").glob("*.py"))]
    h = helper()
    return {name: h.sha256_path(repo / name) for name in paths}


def validate_admission(d, h, *, actual=None, memory=None):
    if (type(d) is not dict or d.get("schema") != "tgrab-figure10-full-admission-v1"
            or d.get("protocol_id") != PROTOCOL_ID
            or not re.fullmatch(r"[0-9a-f]{40}", str(d.get("source_commit", "")))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", str(d.get("run_id", "")))):
        h.fail("invalid full-fit admission identity")
    expected = {"account": "mila", "partition": "main-cpu", "qos": "normal",
                "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
                "time_limit": "01:00:00", "gpu_allocation": "0",
                "memory_max_bytes": 2 * 1024**3}
    actual = h.scheduler_identity() if actual is None else actual
    memory = h.cgroup_memory() if memory is None else memory
    job = d.get("job")
    if type(job) is not dict or any(str(job.get(k)) != str(v) for k, v in expected.items()):
        h.fail("full-fit job differs from admitted CPU envelope")
    for key in expected:
        if key != "memory_max_bytes" and str(actual.get(key)) != str(job[key]):
            h.fail("actual full-fit allocation differs: " + key)
    if memory["effective_limit_bytes"] != expected["memory_max_bytes"]:
        h.fail("actual full-fit memory ceiling differs")
    if job.get("job_id") not in (None, actual["job_id"]) or job.get("node_list") not in (None, actual["node_list"]):
        h.fail("actual job/node differs from admission")
    if (d.get("work_seconds") != 3480 or type(d.get("work_seconds")) is not int
            or d.get("outer_wall_seconds") != 3570
            or os.environ.get("RC_JOB_WALL_SECONDS") != "3570"
            or "deadline_unix" in d):
        h.fail("invalid full-fit relative timing")
    argv = d.get("canary_args")
    if type(argv) is not list or not all(type(v) is str for v in argv) or len(argv) % 2:
        h.fail("invalid full-fit argument pairs")
    flags = argv[::2]
    if len(set(flags)) != len(flags) or "--deadline-unix" in flags:
        h.fail("duplicate/prequeue deadline flag")
    for flag, value in (("--run-id", d["run_id"]), ("--source-commit", d["source_commit"]),
                        ("--runtime-seconds", "3480")):
        h._require_arg_pair(argv, flag, value)
    for key, flag in (("environment_receipt", "--environment-receipt"),
                      ("node2a_receipt", "--node2a-receipt"),
                      ("conversion_receipt", "--conversion-receipt")):
        pin = d.get(key, {})
        h.regular_bytes(pin.get("path", ""), pin.get("sha256", ""))
        h._require_arg_pair(argv, flag, pin["path"])
        h._require_arg_pair(argv, flag + "-sha256", pin["sha256"])
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
        h.fail("full-fit immutable source identity differs")
    h._verify_interpreter(d["interpreter"])
    h._verify_environment_receipt(d["environment_receipt"])
    h._verify_node2a(d["node2a_receipt"])
    actual, memory = validate_admission(d, h)
    started_ns = int(os.environ.get("RC_WALL_START_EPOCH_NS", "0"))
    deadline = started_ns / 1e9 + d["work_seconds"]
    if started_ns <= 0 or not math.isfinite(deadline) or not 0 <= time.time() - started_ns / 1e9 < d["work_seconds"]:
        h.fail("invalid full-fit job-start clock")
    args = [*d["canary_args"], "--deadline-unix", str(deadline)]
    out = Path(args[args.index("--run-dir") + 1])
    stop = Path(args[args.index("--stop-file") + 1])
    scratch = Path("/network/scratch/m/mehrab.hamidi/representation-collapse/runs/figure10-full")
    if out.parent != scratch or out.name != d["run_id"] or os.path.lexists(out) or os.path.lexists(stop):
        h.fail("full-fit output must be fresh and project scoped")
    scratch.mkdir(mode=0o700, exist_ok=True)
    control = out.with_name("." + out.name)
    child_argv = [sys.executable, "-B", "-c", BOOTSTRAP, *args]
    argv_sha = h.sha256_bytes(json.dumps(child_argv, separators=(",", ":")).encode())
    receipt = {"schema": "tgrab-figure10-supervisor-receipt-v2", "protocol_id": PROTOCOL_ID,
               "run_id": d["run_id"], "admission_sha256": digest, "source_commit": d["source_commit"],
               "closure_sha256": d["closure_sha256"], "source_sha256": d["closure_sha256"][WORKER],
               "job_id": actual["job_id"], "job": {**d["job"], "job_id": actual["job_id"], "node_list": actual["node_list"]},
               "scheduler_record": actual["scheduler_record"], "slurm_environment": actual["slurm_environment"],
               "canary_args": args, "admitted_canary_args": d["canary_args"],
               "child_argv": child_argv, "argv_sha256": argv_sha, "sys_executable": sys.executable,
               "interpreter": d["interpreter"], "environment_receipt": d["environment_receipt"],
               "node2a_receipt": d["node2a_receipt"], "conversion_receipt": d["conversion_receipt"],
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
                h.fail("full-fit journal changed")
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
        # The worker owns the 3480-second graceful fit stop; the supervisor
        # reserves 60 more seconds for selection/reporting before forced cleanup.
        status, reason = h.supervise_child(child, stop_path=stop,
            deadline_unix=started_ns / 1e9 + 3540, memory=memory,
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
        raise SystemExit("usage: full_supervisor --admission PATH --sha256 SHA")
    raise SystemExit(run(sys.argv[2], sys.argv[4]))
