#!/usr/bin/env python3
"""Stdlib-only allocation/STOP/deadline supervisor for synthetic node2a."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import signal
import subprocess
import sys
import time

SCRATCH = Path("/network/scratch/m/mehrab.hamidi/representation-collapse")
MODEL_SOURCE = SCRATCH / "runs/graph/upstream-pin-20261012/source"
MODEL_SOURCE_REVISION = "31293102b065feef7817ff86211427d7594edfdb"
MODEL_WRAPPERS = {
    "model/gclstm.py": "86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609",
    "model/link_predictor.py": "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86",
    "model/node_emb.py": "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29",
}
INTERPRETER = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/venv/bin/python"
RECEIPT = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/preparation-receipt.json"
ENV_INVENTORY = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/node2a-environment-identity.json"
EXPECTED = {"SLURM_JOB_ACCOUNT": "mila", "SLURM_JOB_PARTITION": "main-cpu",
            "SLURM_JOB_QOS": "normal", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "1",
            "SLURM_MEM_PER_NODE": "2048"}
EXPECTED_PLAN_ALLOCATION = {"cluster": "mila", "account": "mila", "qos": "normal",
    "partition": "main-cpu", "nodes": 1, "ntasks": 1, "cpus_per_task": 1,
    "memory": "2G", "walltime": "00:10:00", "gpus": 0, "submissions": 1,
    "retry": False, "requeue": False, "scientific_acceptance": False}
PINNED_INTERPRETER = SCRATCH / "toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10"
PINNED_INTERPRETER_SHA256 = "b081289003128d701e48880da11800d7e4410a5777fec6afaa9e31d07419c52f"
PINNED_CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
PINNED_IMPORT_SHA256 = "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344"
PINNED_VERSIONS = {"torch": "2.0.0+cpu", "torch-geometric": "2.6.1",
    "torch-geometric-temporal": "0.54.0", "numpy": "1.24.4",
    "torch-cluster": "1.6.3+pt20cpu", "torch-scatter": "2.1.2+pt20cpu",
    "torch-sparse": "0.6.18+pt20cpu", "torch-spline-conv": "1.2.2+pt20cpu",
    "wandb": "0.30.0"}
STOP = False


def fixture_output_path(run_output: Path) -> Path:
    """Keep atomic fixture publication separate from the supervisor journal."""
    return Path(run_output) / "fixture"


def verify_model_source(plan: dict) -> dict:
    """Require the separate pinned upstream model wrappers, not project source."""
    if (plan.get("model_source_root") != str(MODEL_SOURCE)
            or plan.get("model_source_revision") != MODEL_SOURCE_REVISION
            or plan.get("model_wrapper_sha256") != MODEL_WRAPPERS):
        raise SystemExit("upstream model source plan differs from fixed pins")
    if not MODEL_SOURCE.is_dir() or MODEL_SOURCE.is_symlink() or MODEL_SOURCE.resolve(strict=True) != MODEL_SOURCE:
        raise SystemExit("pinned upstream model source root is unavailable or noncanonical")
    observed = {}
    for relative, expected in MODEL_WRAPPERS.items():
        path = MODEL_SOURCE / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64_000:
            raise SystemExit("pinned upstream wrapper is not a bounded regular file: " + relative)
        digest = sha(path)
        if digest != expected:
            raise SystemExit("pinned upstream wrapper digest mismatch: " + relative)
        observed[relative] = digest
    return observed


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def stop_signal(_signum, _frame):
    global STOP
    STOP = True


def validate_plan_contract(plan: dict) -> None:
    """Reject even correctly rehashed plans that alter fixed stop/resource bounds."""
    run_id = plan.get("run_id")
    if type(run_id) is not str or not re.fullmatch(
            r"node2a-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", run_id):
        raise SystemExit("invalid plan run identity")
    expected_output = str(SCRATCH / "runs/figure10-node2a" / run_id)
    if plan.get("output") != expected_output:
        raise SystemExit("plan output must be the unique run directory under the frozen scratch root")
    if plan.get("stop_file") != expected_output + "/STOP":
        raise SystemExit("plan STOP path differs from its run directory")
    if type(plan.get("deadline_seconds")) is not int or plan["deadline_seconds"] != 570:
        raise SystemExit("plan deadline differs from the reviewed 570-second bound")
    if plan.get("allocation") != EXPECTED_PLAN_ALLOCATION:
        raise SystemExit("plan allocation differs from the reviewed one-shot CPU envelope")
    if (plan.get("model_source_root") != str(MODEL_SOURCE)
            or plan.get("model_source_revision") != MODEL_SOURCE_REVISION
            or plan.get("model_wrapper_sha256") != MODEL_WRAPPERS):
        raise SystemExit("plan upstream model source differs from fixed pins")
    if plan.get("fixture") != "synthetic-node2a-zero-update-only" or plan.get("scientific_acceptance") is not False:
        raise SystemExit("plan is not the synthetic-only fixture")


def verify_scheduler_record(job_id: str) -> None:
    result = subprocess.run(["scontrol", "show", "job", "--oneliner", job_id],
                            check=True, capture_output=True, text=True, timeout=5)
    fields = dict(item.split("=", 1) for item in result.stdout.split() if "=" in item)
    expected = {"JobState": "RUNNING", "Account": "mila", "QOS": "normal",
                "Partition": "main-cpu", "NumNodes": "1", "NumCPUs": "1",
                "CPUs/Task": "1", "TimeLimit": "00:10:00", "Requeue": "0",
                "Restarts": "0"}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise SystemExit("authoritative Slurm job record differs from the one-shot envelope")
    for key in ("ReqTRES", "AllocTRES"):
        tres = fields.get(key, "").lower()
        if not tres or "gpu" in tres:
            raise SystemExit(f"Slurm {key} is missing or includes a GPU allocation")


def verify_environment_restart_count(value: str | None) -> None:
    """Mila may omit this variable; authoritative scheduler fields are mandatory."""
    if value not in (None, "0"):
        raise SystemExit("restarted/requeued job is not accepted")


def cgroup_memory() -> dict:
    membership = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            membership = PurePosixPath(fields[2]); break
    if membership is None:
        raise SystemExit("cgroup v2 membership unavailable")
    candidates = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        sides = line.split(" - ", 1)
        if len(sides) != 2: continue
        left, right = sides; lf, rf = left.split(), right.split()
        if not rf or rf[0] != "cgroup2" or len(lf) < 5: continue
        root = PurePosixPath(lf[3].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\"))
        mount = Path(lf[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\"))
        try: relative = membership.relative_to(root)
        except ValueError: continue
        candidates.append((len(root.parts), root, mount, relative))
    if not candidates:
        raise SystemExit("cannot map process cgroup to a cgroup2 mount")
    _, root, mount, relative = max(candidates, key=lambda x: x[0])
    mount = mount.resolve(strict=True); leaf = mount.joinpath(*relative.parts).resolve(strict=True)
    if leaf != mount and mount not in leaf.parents:
        raise SystemExit("cgroup path escaped its mounted root")
    ancestors = []; current = leaf
    while True:
        ancestors.append(current)
        if current == mount: break
        current = current.parent
    return read_cgroup_memory_files(ancestors, leaf)


def read_cgroup_memory_files(ancestors: list[Path], leaf: Path) -> dict:
    """Use exposed memory limits while allowing controller-less cgroup ancestors."""
    limits = []
    for ancestor in ancestors:
        path = ancestor / "memory.max"
        if not path.is_file():
            continue
        value = path.read_text().strip()
        if value != "max":
            try:
                limits.append((int(value), str(path)))
            except ValueError as exc:
                raise SystemExit("invalid cgroup memory.max") from exc
    leaf_limit = leaf / "memory.max"
    if not leaf_limit.is_file():
        raise SystemExit("job cgroup memory.max unavailable")
    if not limits:
        raise SystemExit("no finite cgroup memory limit")
    current_path, peak_path = leaf / "memory.current", leaf / "memory.peak"
    if not current_path.is_file():
        raise SystemExit("job cgroup memory.current unavailable")
    minimum = min(value for value, _ in limits)
    return {"effective_limit_bytes": minimum,
            "limiting_ancestors": [path for value, path in limits if value == minimum],
            "memory_current_bytes": int(current_path.read_text().strip()),
            "memory_peak_bytes": int(peak_path.read_text().strip()) if peak_path.is_file() else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != args.plan_sha256:
        raise SystemExit("plan digest mismatch")
    validate_plan_contract(plan)
    if (plan.get("interpreter") != str(INTERPRETER)
            or Path(sys.executable).resolve() != PINNED_INTERPRETER.resolve()
            or sha(PINNED_INTERPRETER) != PINNED_INTERPRETER_SHA256):
        raise SystemExit("runtime interpreter path mismatch")
    if Path(sys.argv[0]).resolve() != (SCRATCH / "source/code/ops/tgrab_node2a_supervisor.py").resolve():
        raise SystemExit("supervisor source path mismatch")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ID") in ("0", "undefined"):
        raise SystemExit("live Slurm job ID required")
    for key, value in EXPECTED.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"Slurm allocation mismatch: {key}")
    verify_environment_restart_count(os.environ.get("SLURM_RESTART_COUNT"))
    if any(os.environ.get(key, "") not in ("", "0", "NoDevFiles")
           for key in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")):
        raise SystemExit("GPU allocation/environment detected")
    verify_scheduler_record(os.environ["SLURM_JOB_ID"])
    memory_before = cgroup_memory()
    if memory_before["effective_limit_bytes"] != 2 * 1024**3:
        raise SystemExit("effective physical cgroup memory limit differs from 2 GiB")
    model_source_before = verify_model_source(plan)
    root = Path(plan["source_root"]).resolve(strict=True)
    if root != (SCRATCH / "source").resolve():
        raise SystemExit("unexpected source root")
    if not (root / ".git").exists():
        raise SystemExit("source root is not a Git checkout")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if status or head != plan["source_commit"] or tree != plan["source_tree"]:
        raise SystemExit("source checkout is dirty or not pinned")
    if sha(root / "src/representation_collapse/tgrab_figure10_bootstrap.py") != plan["bootstrap_sha256"]:
        raise SystemExit("bootstrap module digest mismatch")
    if (plan.get("pinned_graph_import_sha256") != PINNED_IMPORT_SHA256
            or sha(root / "src/representation_collapse/pinned_graph_import.py") != PINNED_IMPORT_SHA256):
        raise SystemExit("pinned graph importer digest mismatch")
    if sha(root / "code/ops/tgrab_node2a_job.sh") != plan["job_sha256"]:
        raise SystemExit("job wrapper digest mismatch")
    try:
        receipt_stat = RECEIPT.lstat()
        if not stat.S_ISREG(receipt_stat.st_mode) or receipt_stat.st_size > 1_000_000:
            raise SystemExit("environment receipt is not a bounded regular file")
        receipt_raw = RECEIPT.read_bytes()
        receipt = json.loads(receipt_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("verified remote environment receipt unavailable or invalid") from exc
    if sha(RECEIPT) != plan["environment_receipt_sha256"]:
        raise SystemExit("verified remote environment receipt mismatch")
    if (receipt.get("state") != "installed_metadata_verified_native_smoke_pending"
            or receipt.get("installed_metadata_verified") is not True
            or receipt.get("requirements_sha256") != "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
            or receipt.get("artifact_manifest_sha256") != "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"
            or receipt.get("installed_cell_sha256") != PINNED_CELL_SHA256
            or type(receipt.get("installed_versions")) is not dict
            or any(receipt["installed_versions"].get(name) != version for name, version in PINNED_VERSIONS.items())
            or receipt.get("venv") != str(INTERPRETER.parent.parent)):
        raise SystemExit("environment receipt does not certify the pinned CPython 3.10 environment")
    inventory_info = ENV_INVENTORY.lstat()
    if not stat.S_ISREG(inventory_info.st_mode) or inventory_info.st_size > 262_144:
        raise SystemExit("environment identity inventory is not a bounded regular file")
    inventory_raw = ENV_INVENTORY.read_bytes()
    if hashlib.sha256(inventory_raw).hexdigest() != plan["environment_inventory_sha256"]:
        raise SystemExit("environment identity inventory digest differs from reviewed pin")
    try:
        identity_doc = json.loads(inventory_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit("invalid environment identity inventory JSON") from exc
    if (type(identity_doc) is not dict or set(identity_doc) != {"schema", "identity"}
            or identity_doc.get("schema") != "tgrab-node2a-environment-identity-v1"
            or type(identity_doc.get("identity")) is not dict):
        raise SystemExit("environment identity inventory schema mismatch")
    output = Path(plan["output"])
    if output.parent != SCRATCH / "runs/figure10-node2a" or output.exists() or output.is_symlink():
        raise SystemExit("output must be a new unique run under Mila scratch")
    stop_file = Path(plan["stop_file"])
    if stop_file.exists() or stop_file.is_symlink():
        raise SystemExit("STOP already requested")
    if sha(Path(__file__)) != plan["supervisor_sha256"]:
        raise SystemExit("supervisor digest mismatch")
    argv = plan["argv"]
    expected_argv = [str(INTERPRETER), "-B", "-m", "representation_collapse.tgrab_figure10_bootstrap",
                     "--source-root", str(MODEL_SOURCE), "--output", str(fixture_output_path(output)),
                     "--environment-identity", str(ENV_INVENTORY),
                     "--environment-identity-sha256", plan["environment_inventory_sha256"]]
    if argv != expected_argv:
        raise SystemExit("unexpected fixture argv")
    output.mkdir(parents=True, mode=0o700)
    atomic_json(output / "supervisor-start.json", {"schema": "tgrab-node2a-job-v1",
        "job_id": os.environ["SLURM_JOB_ID"], "plan_sha256": args.plan_sha256,
        "source_commit": plan["source_commit"], "environment_receipt_sha256": plan["environment_receipt_sha256"],
        "pinned_graph_import_sha256": PINNED_IMPORT_SHA256,
        "environment_inventory_sha256": plan["environment_inventory_sha256"],
        "model_source_root": str(MODEL_SOURCE), "model_source_revision": MODEL_SOURCE_REVISION,
        "model_wrapper_sha256": model_source_before,
        "interpreter": sys.executable, "allocation": {k: os.environ[k] for k in EXPECTED},
        "memory_before": memory_before,
        "scientific_acceptance": False})
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop_signal)
    pycache = output / "isolated-pycache"
    pycache.mkdir(mode=0o700)
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX=str(pycache))
    child = subprocess.Popen(expected_argv, cwd=root, env=env, start_new_session=True,
                             stdin=subprocess.DEVNULL)
    reason = "exit"
    deadline = started + int(plan["deadline_seconds"])
    while child.poll() is None:
        if STOP or stop_file.exists() or stop_file.is_symlink():
            reason = "stop"
            break
        if time.monotonic() >= deadline:
            reason = "deadline"
            break
        time.sleep(0.5)
    if reason != "exit":
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
    code = child.wait()
    model_source_error = None
    try:
        model_source_after = verify_model_source(plan)
        if model_source_after != model_source_before:
            raise SystemExit("pinned upstream model source changed during fixture")
    except SystemExit as exc:
        model_source_after = None
        model_source_error = str(exc)
    completed = code == 0 and reason == "exit" and model_source_error is None
    receipt = {"schema": "tgrab-node2a-job-terminal-v1", "state": "completed" if completed else "failed",
               "reason": reason, "exit_code": code, "job_id": os.environ["SLURM_JOB_ID"],
               "elapsed_seconds": time.monotonic() - started, "plan_sha256": args.plan_sha256,
               "model_source_before": model_source_before, "model_source_after": model_source_after,
               "model_source_error": model_source_error,
               "memory_after": cgroup_memory(),
               "scientific_acceptance": False}
    atomic_json(output / "supervisor-terminal.json", receipt)
    return 0 if receipt["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
