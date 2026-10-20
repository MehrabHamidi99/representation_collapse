#!/usr/bin/env python3
"""Admitted descriptive T-GCN training prefix, sharing reviewed CPU ownership helpers."""
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

PROTOCOL_ID = "tgrab_tgcn_training_prefix_v1"
HELPER_SHA256 = "3023286daa4d9142fe43c6c1c8898fc7d7cc1060ddd8e191a440f8f6c0638e1d"
WORKER = "src/representation_collapse/tgrab_tgcn_prefix.py"
SOURCE_INPUTS = {
 "--tgcn-wrapper": ("tgcn.py", "028a85f0527833a7e62099e8bb826ac144df29738ee120795d205d96a8c6f78c"),
 "--link-head": ("link_predictor.py", "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86"),
}

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
    paths = ["code/ops/tgrab_tgcn_prefix_job.sh",
             "code/ops/tgrab_tgcn_prefix_supervisor.py",
             "code/ops/tgrab_figure10_canary_supervisor.py", WORKER,
             "doc/auto/tgcn-production-prefix-protocol-2026-10-20.md",
             "code/analysis/tgrab_tgcn_native_calibration.py"]
    paths += [str(p.relative_to(repo)) for p in
              sorted((repo / "src/representation_collapse").glob("*.py"))]
    h = helper()
    return {name: h.sha256_path(repo / name) for name in paths}


def validate_admission(d, h, *, actual=None, memory=None):
    if (type(d) is not dict or d.get("schema") != "tgrab-tgcn-prefix-admission-v1"
            or d.get("protocol_id") != PROTOCOL_ID
            or not re.fullmatch(r"[0-9a-f]{40}", str(d.get("source_commit", "")))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", str(d.get("run_id", "")))):
        h.fail("invalid T-GCN training prefix admission identity")
    expected = {"account": "mila", "partition": "main-cpu", "qos": "normal",
                "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
                "time_limit": "00:30:00", "gpu_allocation": "0",
                "memory_max_bytes": 2 * 1024**3}
    actual = h.scheduler_identity() if actual is None else actual
    memory = h.cgroup_memory() if memory is None else memory
    job = d.get("job")
    if type(job) is not dict or any(str(job.get(k)) != str(v) for k, v in expected.items()):
        h.fail("T-GCN training prefix job differs from admitted CPU envelope")
    for key in expected:
        if key != "memory_max_bytes" and str(actual.get(key)) != str(job[key]):
            h.fail("actual T-GCN training prefix allocation differs: " + key)
    if memory["effective_limit_bytes"] != expected["memory_max_bytes"]:
        h.fail("actual T-GCN training prefix memory ceiling differs")
    if job.get("job_id") not in (None, actual["job_id"]) or job.get("node_list") not in (None, actual["node_list"]):
        h.fail("actual job/node differs from admission")
    if (d.get("work_seconds") != 1740 or type(d.get("work_seconds")) is not int
            or d.get("outer_wall_seconds") != 1770
            or os.environ.get("RC_JOB_WALL_SECONDS") != "1770"
            or "deadline_unix" in d):
        h.fail("invalid T-GCN training prefix relative timing")
    argv = d.get("canary_args")
    if type(argv) is not list or not all(type(v) is str for v in argv) or len(argv) % 2:
        h.fail("invalid T-GCN training prefix argument pairs")
    flags = argv[::2]
    if len(set(flags)) != len(flags) or "--deadline-unix" in flags:
        h.fail("duplicate/prequeue deadline flag")
    for flag, value in (("--run-id", d["run_id"]),):
        h._require_arg_pair(argv, flag, value)
    h.regular_bytes(d["environment_receipt"]["path"], d["environment_receipt"]["sha256"])
    if set(flags) != {"--output", "--run-id", *SOURCE_INPUTS, "--data-root", "--data-manifest-sha256", "--calibration-acceptance", "--calibration-acceptance-sha256"}:
        h.fail("prefix accepts only frozen source/data/calibration arguments")
    if d.get("training_prefix_only") is not True or d.get("new_test_opener_calls") != 0:
        h.fail("training prefix excludes validation/test and saved-weight reads")
    source_root = Path("/network/scratch/m/mehrab.hamidi/representation-collapse/sources/tgcn-native-20261020")
    for flag, (filename, digest) in SOURCE_INPUTS.items():
        path = source_root / filename
        h._require_arg_pair(argv, flag, str(path))
        h.regular_bytes(path, digest)
    h.regular_bytes(source_root / "node_emb.py", "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29")
    data_root = "/network/scratch/m/mehrab.hamidi/representation-collapse/runs/primary-conversion-verified-20261016/attempt001/worker/converted"
    data_sha = "a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce"
    h._require_arg_pair(argv, "--data-root", data_root)
    h._require_arg_pair(argv, "--data-manifest-sha256", data_sha)
    h.regular_bytes(Path(data_root) / "manifest.json", data_sha)
    accepted = d["calibration_acceptance"]
    h._require_arg_pair(argv, "--calibration-acceptance", accepted["path"])
    h._require_arg_pair(argv, "--calibration-acceptance-sha256", accepted["sha256"])
    calibration = json.loads(h.regular_bytes(accepted["path"], accepted["sha256"]))
    if (calibration.get("kind") != "tgcn_native_calibration_compact_acceptance_v1"
            or calibration.get("status") != "accepted_synthetic_execution"
            or calibration.get("checks_verified") != 248
            or calibration.get("source_commit") != "e0e6c20a72a98134aa1f2a2450f9eeb011718ab2"
            or calibration.get("result_sha256") != "23b68f8c78e0f4b1fd00959c812463318cb687d8b9b95404c709c6ade5001619"
            or calibration.get("actual_offline_history_rows") != 1):
        h.fail("accepted native T-GCN calibration required before prefix")
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
        h.fail("T-GCN training prefix immutable source identity differs")
    h._verify_interpreter(d["interpreter"])
    h._verify_environment_receipt(d["environment_receipt"])
    actual, memory = validate_admission(d, h)
    started_ns = int(os.environ.get("RC_WALL_START_EPOCH_NS", "0"))
    deadline = started_ns / 1e9 + d["work_seconds"]
    if started_ns <= 0 or not math.isfinite(deadline) or not 0 <= time.time() - started_ns / 1e9 < d["work_seconds"]:
        h.fail("invalid T-GCN training prefix job-start clock")
    args = list(d["canary_args"])
    out = Path(args[args.index("--output") + 1])
    stop = Path(d["stop_file"])
    scratch = Path("/network/scratch/m/mehrab.hamidi/representation-collapse/runs/tgcn-training-prefix")
    if out.parent != scratch or out.name != d["run_id"] or stop != Path(str(out) + ".STOP") or os.path.lexists(out) or os.path.lexists(stop):
        h.fail("T-GCN training prefix output must be fresh and project scoped")
    scratch.mkdir(mode=0o700, exist_ok=True)
    control = out.with_name("." + out.name)
    worker_args = [*args, "--deadline-unix", str(deadline), "--stop-file", str(stop)]
    child_argv = [sys.executable, "-B", "-c", BOOTSTRAP, *worker_args]
    argv_sha = h.sha256_bytes(json.dumps(child_argv, separators=(",", ":")).encode())
    receipt = {"schema": "tgrab-tgcn-prefix-supervisor-receipt-v1", "protocol_id": PROTOCOL_ID,
               "run_id": d["run_id"], "admission_sha256": digest, "source_commit": d["source_commit"],
               "closure_sha256": d["closure_sha256"], "source_sha256": d["closure_sha256"][WORKER],
               "job_id": actual["job_id"], "job": {**d["job"], "job_id": actual["job_id"], "node_list": actual["node_list"]},
               "scheduler_record": actual["scheduler_record"], "slurm_environment": actual["slurm_environment"],
               "canary_args": worker_args, "admitted_canary_args": d["canary_args"],
               "child_argv": child_argv, "argv_sha256": argv_sha, "sys_executable": sys.executable,
               "interpreter": d["interpreter"], "environment_receipt": d["environment_receipt"],
               "calibration_acceptance": d["calibration_acceptance"],
               "data_root": "/network/scratch/m/mehrab.hamidi/representation-collapse/runs/primary-conversion-verified-20261016/attempt001/worker/converted",
               "data_manifest_sha256": "a6ff50183de570ff17cfbe92b3f1976bac9d2d85557ecc11379e24b2c737d1ce",
               "source_inputs": {k: {"filename": v[0], "sha256": v[1]} for k, v in SOURCE_INPUTS.items()},
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
                h.fail("T-GCN training prefix journal changed")
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
            deadline_unix=started_ns / 1e9 + 1760, memory=memory,
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
        raise SystemExit("usage: tgcn_calibration_supervisor --admission PATH --sha256 SHA")
    raise SystemExit(run(sys.argv[2], sys.argv[4]))
