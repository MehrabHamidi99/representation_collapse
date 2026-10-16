#!/usr/bin/env python3
"""Fail-closed stdlib supervisor for the one-shot node2a identity capture."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import PurePosixPath

SCRATCH = Path("/network/scratch/m/mehrab.hamidi/representation-collapse")
SOURCE = SCRATCH / "source"
ENV_DIR = SCRATCH / "environments/graph-native-cpu-20261015-calibration01"
INTERPRETER = ENV_DIR / "venv/bin/python"
TOOLCHAIN_PYTHON = SCRATCH / "toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10"
INTERPRETER_SHA256 = "b081289003128d701e48880da11800d7e4410a5777fec6afaa9e31d07419c52f"
BOOTSTRAP_SHA256 = "37a0c0839e167d3a71cc4e1186f97a9d91c211cf006ad2df00dcc64652e62253"
IMPORTER_SHA256 = "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344"
ENV_RECEIPT = ENV_DIR / "preparation-receipt.json"
ENV_RECEIPT_SHA256 = "4e753d13680738a7b48374950ae41e33f0b7f4935c889f882b149678301eb6c2"
INSTALLED_CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
INVENTORY = ENV_DIR / "node2a-environment-identity.json"
RUNS = SCRATCH / "runs/figure10-node2a-capture"
REVIEW_RECEIPT = SCRATCH / "runs/review/figure10-node2a-capture-review.json"
JOB_REL = "code/ops/tgrab_node2a_capture_job.sh"
SUPERVISOR_REL = "code/ops/tgrab_node2a_capture_supervisor.py"
ALLOCATION = {"cluster": "mila", "account": "mila", "qos": "normal", "partition": "main-cpu",
    "nodes": 1, "ntasks": 1, "cpus_per_task": 1, "memory": "2G", "walltime": "00:10:00",
    "gpus": 0, "submissions": 1, "retry": False, "requeue": False, "scientific_acceptance": False}
SLURM_FIELDS = {"SLURM_JOB_ACCOUNT": "mila", "SLURM_JOB_PARTITION": "main-cpu",
    "SLURM_JOB_QOS": "normal", "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
    "SLURM_CPUS_PER_TASK": "1", "SLURM_MEM_PER_NODE": "2048"}
STOP_REQUESTED = False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pinned_bootstrap(path: Path) -> bytes:
    """Read once, reject links/non-files, and return only the reviewed bytes."""
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise SystemExit("bootstrap source must be a bounded regular file")
        content = path.read_bytes()
    except OSError as exc:
        raise SystemExit("pinned bootstrap source is unavailable") from exc
    if hashlib.sha256(content).hexdigest() != BOOTSTRAP_SHA256:
        raise SystemExit("bootstrap bytes differ from pinned source")
    return content


def verify_source_closure(root: Path, plan: dict, bootstrap_bytes: bytes | None = None) -> dict:
    """Check committed tree and every executable source pin at an admission boundary."""
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True, capture_output=True, text=True).stdout.strip()
    if status or commit != plan["source_commit"] or tree != plan["source_tree"]:
        raise SystemExit("remote source worktree/commit/tree differs from reviewed pin")
    files = {
        "bootstrap_sha256": (root / "src/representation_collapse/tgrab_figure10_bootstrap.py", BOOTSTRAP_SHA256),
        "pinned_graph_import_sha256": (root / "src/representation_collapse/pinned_graph_import.py", IMPORTER_SHA256),
        "job_sha256": (root / JOB_REL, plan["job_sha256"]),
        "supervisor_sha256": (root / SUPERVISOR_REL, plan["supervisor_sha256"]),
    }
    for name, (path, expected) in files.items():
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"capture source closure mismatch: {name}")
        actual = (hashlib.sha256(bootstrap_bytes).hexdigest()
                  if name == "bootstrap_sha256" and bootstrap_bytes is not None else sha(path))
        if actual != expected:
            raise SystemExit(f"capture source closure mismatch: {name}")
    own_path = Path(__file__)
    if own_path.is_symlink() or sha(own_path) != plan["supervisor_sha256"]:
        raise SystemExit("running supervisor bytes differ from reviewed pin")
    return {"commit": commit, "tree": tree,
        **{name: digest for name, (_, digest) in files.items()}}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def validate_plan_contract(plan: dict) -> None:
    required = {"schema", "run_id", "source_commit", "source_tree", "bootstrap_sha256",
        "pinned_graph_import_sha256", "job_sha256", "supervisor_sha256", "environment_receipt_sha256",
        "review_receipt_path", "review_receipt_sha256", "inventory_path", "run_dir", "stop_file",
        "deadline_seconds", "allocation"}
    if type(plan) is not dict or set(plan) != required or plan.get("schema") != "tgrab-node2a-capture-plan-v1":
        raise SystemExit("capture plan schema mismatch")
    if not re.fullmatch(r"node2a-capture-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", str(plan["run_id"])):
        raise SystemExit("invalid capture run identity")
    for field in ("source_commit", "source_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(plan[field])):
            raise SystemExit(f"invalid immutable source {field}")
    fixed_hashes = {"bootstrap_sha256": BOOTSTRAP_SHA256,
                    "pinned_graph_import_sha256": IMPORTER_SHA256,
                    "environment_receipt_sha256": ENV_RECEIPT_SHA256}
    for key, expected in fixed_hashes.items():
        if plan[key] != expected:
            raise SystemExit(f"plan {key} differs from reviewed pin")
    for field in ("job_sha256", "supervisor_sha256", "review_receipt_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(plan[field])):
            raise SystemExit(f"invalid plan {field}")
    run_dir = str(RUNS / plan["run_id"])
    if plan["run_dir"] != run_dir:
        raise SystemExit("capture journal path must be unique under the fixed scratch root")
    if plan["stop_file"] != run_dir + "/STOP":
        raise SystemExit("capture STOP path must be exactly run_dir/STOP")
    if plan["inventory_path"] != str(INVENTORY):
        raise SystemExit("capture inventory path differs from the unique fixed output")
    if type(plan["deadline_seconds"]) is not int or plan["deadline_seconds"] != 570:
        raise SystemExit("capture deadline differs from reviewed bound")
    if plan["allocation"] != ALLOCATION:
        raise SystemExit("capture allocation differs from fixed CPU-only envelope")


def _signal(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def stop_requested(stop_file: Path) -> bool:
    return STOP_REQUESTED or stop_file.exists() or stop_file.is_symlink()


def terminate_owned_group(child: subprocess.Popen, graceful_seconds: float = 5.0) -> None:
    """Reap the direct child and ensure no process in its private session survives."""
    try: os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    until = time.monotonic() + graceful_seconds
    while time.monotonic() < until:
        child.poll()
        try: os.killpg(child.pid, 0)
        except ProcessLookupError: break
        time.sleep(0.05)
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    child.wait()
    try: os.killpg(child.pid, 0)
    except ProcessLookupError: return
    raise SystemExit("owned capture process group remains after TERM/KILL cleanup")


def verify_slurm(job_id: str, startup_timeout: float = 5.0,
                 poll_interval: float = 0.25) -> dict:
    expected = {"Account": "mila", "QOS": "normal", "Partition": "main-cpu",
        "NumNodes": "1", "NumCPUs": "1", "CPUs/Task": "1", "TimeLimit": "00:10:00",
        "Requeue": "0", "Restarts": "0"}
    deadline = time.monotonic() + startup_timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit("Slurm job did not reach RUNNING within the 5-second startup bound")
        try:
            result = subprocess.run(["scontrol", "show", "job", "--oneliner", job_id],
                check=True, capture_output=True, text=True, timeout=min(1.0, remaining))
        except subprocess.TimeoutExpired as exc:
            raise SystemExit("scontrol timed out while verifying Slurm allocation") from exc
        fields = dict(item.split("=", 1) for item in result.stdout.split() if "=" in item)
        mismatches = [f"{key} expected={value!r} actual={fields.get(key)!r}"
                      for key, value in expected.items() if fields.get(key) != value]
        expected_tres = {"cpu": "1", "mem": "2G", "node": "1"}
        def parse_tres(name: str, required: bool) -> None:
            raw = fields.get(name)
            if raw is None and not required:
                return
            values = {}
            for entry in (raw or "").split(","):
                if "=" not in entry:
                    values = {}; break
                key, value = entry.split("=", 1)
                if not key or not value or key in values:
                    values = {}; break
                values[key.lower()] = value
            valid = (all(values.get(key) == value for key, value in expected_tres.items())
                     and set(values).issubset(set(expected_tres) | {"billing"})
                     and values.get("billing", "1") == "1")
            if not valid:
                mismatches.append(f"{name} expected={expected_tres!r} with optional billing=1 actual={raw!r}")
        parse_tres("ReqTRES", required=True)
        pending = fields.get("JobState") == "PENDING"
        parse_tres("AllocTRES", required=not pending)
        if mismatches:
            raise SystemExit("authoritative Slurm allocation mismatch: " + "; ".join(mismatches))
        state = fields.get("JobState")
        if state == "RUNNING":
            return {"job_id": job_id, "fields": {**{key: fields[key] for key in expected},
                    "JobState": state}, "requested_tres": fields["ReqTRES"],
                    "allocated_tres": fields.get("AllocTRES")}
        if state not in {"CONFIGURING", "PENDING"}:
            raise SystemExit(f"authoritative Slurm job state is {state!r}; expected CONFIGURING/PENDING or RUNNING")
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def require_zero_restart_count(value: str | None) -> None:
    if value is not None and value != "0":
        raise SystemExit(f"SLURM_RESTART_COUNT, when present, must be exactly '0'; actual={value!r}")


def verify_environment_receipt(path: Path, expected_sha: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000 or sha(path) != expected_sha:
        raise SystemExit("environment receipt type/size/hash mismatch")
    receipt = json.loads(path.read_text())
    if (receipt.get("state") != "installed_metadata_verified_native_smoke_pending"
            or receipt.get("installed_metadata_verified") is not True
            or receipt.get("requirements_sha256") != "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
            or receipt.get("artifact_manifest_sha256") != "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"
            or receipt.get("installed_cell_sha256") != INSTALLED_CELL_SHA256
            or receipt.get("venv") != str(INTERPRETER.parent.parent)):
        raise SystemExit("environment receipt does not match isolated CPython 3.10 install")


def verify_child_interpreter(interpreter: Path = INTERPRETER,
                             toolchain: Path = TOOLCHAIN_PYTHON) -> dict:
    """Require venv launcher symlink to resolve to the pinned standalone binary."""
    try:
        mode = interpreter.lstat().st_mode
        if not stat.S_ISLNK(mode):
            raise SystemExit("native venv Python must be a symlink to the pinned toolchain")
        resolved = interpreter.resolve(strict=True)
        pinned = toolchain.resolve(strict=True)
        if resolved != pinned or not stat.S_ISREG(pinned.stat().st_mode):
            raise SystemExit("native venv Python does not resolve to the pinned toolchain")
        digest = sha(pinned)
    except OSError as exc:
        raise SystemExit("native venv Python/toolchain is unavailable") from exc
    if digest != INTERPRETER_SHA256:
        raise SystemExit("native venv Python target hash differs from pinned toolchain")
    return {"path": str(interpreter), "lstat_type": "symlink",
            "resolved_path": str(resolved), "sha256": digest}


def verify_review(plan: dict) -> None:
    path = Path(plan["review_receipt_path"])
    if path != REVIEW_RECEIPT:
        raise SystemExit("review receipt must be staged at the fixed approved path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000 or sha(path) != plan["review_receipt_sha256"]:
        raise SystemExit("independent capture-job review receipt missing or hash mismatch")
    review = json.loads(path.read_text())
    expected = {"schema": "tgrab-node2a-capture-job-review-v1", "status": "accepted",
        "source_commit": plan["source_commit"], "source_tree": plan["source_tree"],
        "bootstrap_sha256": BOOTSTRAP_SHA256, "pinned_graph_import_sha256": IMPORTER_SHA256,
        "job_sha256": plan["job_sha256"], "supervisor_sha256": plan["supervisor_sha256"],
        "environment_receipt_sha256": ENV_RECEIPT_SHA256, "allocation": ALLOCATION}
    if review != expected:
        raise SystemExit("independent capture review does not bind this exact source/job/environment envelope")


def verify_inventory(path: Path) -> dict:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 262_144:
        raise SystemExit("captured identity is not a bounded regular file")
    document = json.loads(path.read_text())
    if (type(document) is not dict or set(document) != {"schema", "identity"}
            or document.get("schema") != "tgrab-node2a-environment-identity-v1"
            or type(document.get("identity")) is not dict
            or document["identity"].get("python_version") != [3, 10, 19]
            or document["identity"].get("prefix") != str(INTERPRETER.parent.parent)
            or document["identity"].get("source_identity", {}).get("pinned_graph_import_sha256") != IMPORTER_SHA256):
        raise SystemExit("captured inventory schema/runtime/source identity mismatch")
    return {"sha256": sha(path), "bytes": info.st_size,
            "source_identity": document["identity"]["source_identity"]}


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
        unescape = lambda s: s.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
        root = PurePosixPath(unescape(lf[3])); mount = Path(unescape(lf[4]))
        try: relative = membership.relative_to(root)
        except ValueError: continue
        candidates.append((len(root.parts), mount, relative))
    if not candidates: raise SystemExit("cannot map process cgroup to cgroup2 mount")
    _, mount, relative = max(candidates, key=lambda item: item[0])
    mount = mount.resolve(strict=True); leaf = mount.joinpath(*relative.parts).resolve(strict=True)
    if leaf != mount and mount not in leaf.parents: raise SystemExit("cgroup path escaped mount")
    ancestors = []; current = leaf
    while True:
        ancestors.append(current)
        if current == mount: break
        current = current.parent
    return read_cgroup_memory_files(ancestors, leaf)


def read_cgroup_memory_files(ancestors: list[Path], leaf: Path) -> dict:
    """Use every exposed ancestor limit; some Mila cgroup roots omit memory.max."""
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
        raise SystemExit("no finite effective cgroup memory limit")
    current_file, peak_file = leaf / "memory.current", leaf / "memory.peak"
    if not current_file.is_file():
        raise SystemExit("job cgroup memory.current unavailable")
    minimum = min(value for value, _ in limits)
    return {"effective_limit_bytes": minimum,
        "limiting_ancestors": [path for value, path in limits if value == minimum],
        "memory_current_bytes": int(current_file.read_text().strip()),
        "memory_peak_bytes": int(peak_file.read_text().strip()) if peak_file.is_file() else None}


def ensure_private_run_dir(path: Path) -> None:
    if path != RUNS / path.name or not re.fullmatch(r"node2a-capture-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", path.name):
        raise SystemExit("invalid capture run directory")
    if RUNS.is_symlink(): raise SystemExit("capture run root cannot be a symlink")
    RUNS.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = RUNS.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit("capture run root must be an owner-only directory")
    if path.exists() or path.is_symlink(): raise SystemExit("capture journal already exists; no retry")
    try: path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc: raise SystemExit("capture journal already exists; no retry") from exc


def require_absent_inventory(path: Path) -> None:
    if path != INVENTORY or path.exists() or path.is_symlink():
        raise SystemExit("identity output must be the unique absent fixed inventory path")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    plan_path = args.plan.resolve(strict=True)
    if sha(plan_path) != args.plan_sha256:
        raise SystemExit("capture plan digest mismatch")
    plan = json.loads(plan_path.read_text())
    validate_plan_contract(plan)
    if Path(sys.executable).resolve() != TOOLCHAIN_PYTHON.resolve() or sha(TOOLCHAIN_PYTHON) != INTERPRETER_SHA256:
        raise SystemExit("capture did not start with the pinned CPython 3.10.19 binary")
    if Path(sys.argv[0]).resolve() != (SOURCE / SUPERVISOR_REL).resolve():
        raise SystemExit("unexpected capture supervisor source path")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ID") in ("0", "undefined"):
        raise SystemExit("live Slurm job ID required")
    for key, value in SLURM_FIELDS.items():
        if os.environ.get(key) != value:
            raise SystemExit(f"Slurm allocation mismatch: {key}")
    require_zero_restart_count(os.environ.get("SLURM_RESTART_COUNT"))
    if any(
            os.environ.get(key, "") not in ("", "0", "NoDevFiles")
            for key in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")):
        raise SystemExit("GPU-allocated capture job refused")
    slurm_record = verify_slurm(os.environ["SLURM_JOB_ID"])
    memory_before = cgroup_memory()
    if memory_before["effective_limit_bytes"] != 2 * 1024**3:
        raise SystemExit("effective cgroup memory limit differs from 2 GiB envelope")
    root = SOURCE.resolve(strict=True)
    bootstrap = root / "src/representation_collapse/tgrab_figure10_bootstrap.py"
    bootstrap_bytes = read_pinned_bootstrap(bootstrap)
    source_identity_before = verify_source_closure(root, plan, bootstrap_bytes=bootstrap_bytes)
    verify_environment_receipt(ENV_RECEIPT, plan["environment_receipt_sha256"])
    verify_review(plan)
    out = Path(plan["inventory_path"])
    require_absent_inventory(out)
    run_dir = Path(plan["run_dir"])
    ensure_private_run_dir(run_dir)
    stop_file = Path(plan["stop_file"])
    if stop_file.exists() or stop_file.is_symlink():
        raise SystemExit("STOP requested before capture")
    child_interpreter_before = verify_child_interpreter()
    atomic_json(run_dir / "capture-start.json", {"schema": "tgrab-node2a-capture-start-v1",
        "job_id": os.environ["SLURM_JOB_ID"], "plan_sha256": args.plan_sha256,
        "source_commit": source_identity_before["commit"], "source_tree": source_identity_before["tree"],
        "bootstrap_sha256": BOOTSTRAP_SHA256,
        "pinned_graph_import_sha256": IMPORTER_SHA256,
        "environment_receipt_sha256": plan["environment_receipt_sha256"],
        "child_interpreter": child_interpreter_before,
        "allocation": slurm_record, "inventory_path": str(out), "scientific_acceptance": False})
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _signal)
    pycache = run_dir / "isolated-pycache"; pycache.mkdir(mode=0o700)
    bootstrap_b64 = base64.b64encode(bootstrap_bytes).decode("ascii")
    if len(bootstrap_b64) > 100_000:
        raise SystemExit("pinned bootstrap exceeds safe direct-exec argument bound")
    bootstrap_code = ("import base64,sys; p=sys.argv[1]; sys.path.insert(0, sys.argv[2]); "
        "raw=base64.b64decode(sys.argv[3], validate=True); sys.argv=[p] + sys.argv[4:]; "
        "exec(compile(raw, p, 'exec'), "
        "{'__name__':'__main__','__package__':'representation_collapse','__file__':p})")
    child_argv = [str(INTERPRETER), "-B", "-c", bootstrap_code, str(bootstrap), str(root / "src"),
        bootstrap_b64,
        "--source-root", str(root), "--capture-environment-identity", str(out)]
    child = subprocess.Popen(child_argv, cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
                 PYTHONPYCACHEPREFIX=str(pycache)), start_new_session=True, stdin=subprocess.DEVNULL)
    deadline = started + plan["deadline_seconds"]
    reason = "exit"
    while child.poll() is None:
        if stop_requested(stop_file):
            reason = "stop"; break
        if time.monotonic() >= deadline:
            reason = "deadline"; break
        time.sleep(0.5)
    terminate_owned_group(child)
    exit_code = child.wait()
    interpreter_error = None
    try:
        child_interpreter_after = verify_child_interpreter()
        if child_interpreter_after != child_interpreter_before:
            raise SystemExit("native venv Python identity changed during capture")
    except SystemExit as exc:
        child_interpreter_after = None
        interpreter_error = str(exc)
    source_error = None
    try:
        source_identity_after = verify_source_closure(root, plan)
        if source_identity_after != source_identity_before:
            raise SystemExit("source identity changed during capture")
    except SystemExit as exc:
        source_identity_after = None
        source_error = str(exc)
    artifact = None
    if reason == "exit" and exit_code == 0 and interpreter_error is None and source_error is None:
        artifact = verify_inventory(out)
    terminal = {"schema": "tgrab-node2a-capture-terminal-v1",
        "state": "completed" if reason == "exit" and exit_code == 0 and artifact else "failed",
        "reason": reason, "exit_code": exit_code, "job_id": os.environ["SLURM_JOB_ID"],
        "elapsed_seconds": time.monotonic() - started, "plan_sha256": args.plan_sha256,
        "inventory": artifact, "memory_before": memory_before, "memory_after": cgroup_memory(),
        "child_interpreter_before": child_interpreter_before,
        "child_interpreter_after": child_interpreter_after,
        "child_interpreter_error": interpreter_error,
        "source_identity_before": source_identity_before,
        "source_identity_after": source_identity_after,
        "source_identity_error": source_error,
        "scientific_acceptance": False}
    atomic_json(run_dir / "capture-terminal.json", terminal)
    return 0 if terminal["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
