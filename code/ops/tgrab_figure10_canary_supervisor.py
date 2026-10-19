#!/usr/bin/env python3
"""stdlib-only admission and process supervisor for the prefix runtime diagnostic."""
import hashlib
import math
import re
import ctypes
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import subprocess
import sys
import time


class AdmissionError(SystemExit):
    pass


def fail(message):
    raise AdmissionError("supervisor: " + message)


PINNED_INTERPRETER = "/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/bin/python"
PINNED_ENVIRONMENT_RECEIPT = "/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/preparation-receipt.json"
PROTOCOL_ID = "tgrab_figure10_256_prefix_runtime_diagnostic_v1"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path):
    p = Path(path)
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        fail("expected regular nonsymlink file: " + str(p))
    h = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            fail("expected single-link regular file: " + str(p))
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            h.update(chunk)
        after, path_stat = os.fstat(fd), p.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(path_stat):
            fail("source changed while hashing: " + str(p))
        return h.hexdigest()
    finally:
        os.close(fd)


def regular_bytes(path, expected, limit=1_000_000):
    p = Path(path)
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        fail("admission must be a bounded, single-link regular file")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            fail("admission must be a bounded, single-link regular file")
        chunks, size = [], 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - size))
            if not chunk: break
            chunks.append(chunk); size += len(chunk)
            if size > limit: fail("admission file exceeds size bound")
        after, path_stat = os.fstat(fd), p.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        raw = b"".join(chunks)
        if identity(before) != identity(after) or identity(after) != identity(path_stat) or size != before.st_size:
            fail("admission changed while reading")
        if sha256_bytes(raw) != expected: fail("admission hash mismatch")
        return raw
    finally:
        os.close(fd)


def _unescape_mountinfo(value):
    return value.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def resolve_cgroup2(proc_cgroup_text, mountinfo_text):
    membership = None
    for line in proc_cgroup_text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            membership = PurePosixPath(fields[2])
            break
    if membership is None:
        fail("cgroup v2 membership unavailable")
    candidates = []
    for line in mountinfo_text.splitlines():
        left_right = line.split(" - ", 1)
        if len(left_right) != 2:
            continue
        left, right = left_right
        rfields, lfields = right.split(), left.split()
        if not rfields or rfields[0] != "cgroup2" or len(lfields) < 5:
            continue
        root = PurePosixPath(_unescape_mountinfo(lfields[3]))
        mountpoint = Path(_unescape_mountinfo(lfields[4]))
        try:
            relative = membership.relative_to(root)
        except ValueError:
            continue
        candidates.append((len(root.parts), root, mountpoint, relative))
    if not candidates:
        fail("cannot map cgroup membership to a cgroup2 mount")
    _, root, mountpoint, relative = max(candidates, key=lambda item: item[0])
    mount_real = mountpoint.resolve(strict=True)
    leaf = mount_real.joinpath(*relative.parts)
    leaf_real = leaf.resolve(strict=True)
    if leaf_real != mount_real and mount_real not in leaf_real.parents:
        fail("resolved cgroup escapes its cgroup2 mount")
    ancestors = []
    current = leaf_real
    while True:
        ancestors.append(current)
        if current == mount_real:
            break
        if mount_real not in current.parents:
            fail("cgroup ancestry escaped mount")
        current = current.parent
    return {"mountpoint": str(mount_real), "mount_root": str(root), "membership": str(membership),
            "leaf": str(leaf_real), "ancestors": [str(x) for x in ancestors]}


def cgroup_memory(proc_cgroup_path="/proc/self/cgroup", mountinfo_path="/proc/self/mountinfo"):
    mapping = resolve_cgroup2(Path(proc_cgroup_path).read_text(), Path(mountinfo_path).read_text())
    leaf = Path(mapping["leaf"])
    finite_limits = []
    for ancestor in mapping["ancestors"]:
        limit_path = Path(ancestor) / "memory.max"
        if not limit_path.is_file():
            # cgroup2 mount roots can lack the enabled memory controller.
            continue
        value = limit_path.read_text().strip()
        if value != "max":
            try:
                limit = int(value)
                if limit <= 0:
                    fail("nonpositive cgroup memory.max")
                finite_limits.append((limit, str(limit_path)))
            except ValueError:
                fail("invalid cgroup memory.max")
    if not (leaf / "memory.max").is_file():
        fail("job cgroup memory.max unavailable")
    if not finite_limits:
        fail("no finite cgroup-v2 ancestor memory limit")
    current_path = leaf / "memory.current"
    peak_path = leaf / "memory.peak"
    if not current_path.is_file():
        fail("cgroup memory.current unavailable")
    return {"effective_limit_bytes": min(x[0] for x in finite_limits),
            "limiting_ancestors": [path for value, path in finite_limits if value == min(x[0] for x in finite_limits)],
            "current_path": str(current_path),
            "peak_path": str(peak_path) if peak_path.is_file() else None,
            "identity": mapping}


def _required_slurm(name):
    value = os.environ.get(name)
    if value is None or value == "":
        fail("authoritative Slurm variable missing: " + name)
    return value


def scheduler_identity():
    """Use the live scheduler record; optional environment fields cannot replace it."""
    job_id = _required_slurm("SLURM_JOB_ID")
    if not re.fullmatch(r"[1-9][0-9]*", job_id):
        fail("invalid live Slurm job ID")
    if os.environ.get("SLURM_RESTART_COUNT") not in (None, "0"):
        fail("restarted/requeued job is not accepted")
    result = subprocess.run(["scontrol", "show", "job", "--oneliner", job_id],
                            check=True, capture_output=True, text=True, timeout=5)
    fields = dict(item.split("=", 1) for item in result.stdout.split() if "=" in item)
    required = ("JobId", "JobState", "Account", "Partition", "QOS", "NumNodes",
                "NumTasks", "NumCPUs", "CPUs/Task", "NodeList", "TimeLimit",
                "ReqTRES", "AllocTRES", "Requeue", "Restarts")
    if any(not fields.get(name) for name in required):
        fail("authoritative Slurm job record is incomplete")
    if (fields["JobId"] != job_id or fields["JobState"] != "RUNNING"
            or fields["Requeue"] != "0" or fields["Restarts"] != "0"):
        fail("live one-shot RUNNING Slurm job required")
    def tres(name):
        values = {}
        for entry in fields[name].split(","):
            pair = entry.split("=", 1)
            if len(pair) != 2 or not all(pair) or pair[0] in values:
                fail("malformed authoritative Slurm " + name)
            values[pair[0]] = pair[1]
        if not all(key in values for key in ("cpu", "mem", "node")):
            fail("incomplete authoritative Slurm " + name)
        if any("gpu" in key.lower() for key in values):
            fail("GPU allocation/request not admitted")
        if values["cpu"] != fields["NumCPUs"] or values["node"] != fields["NumNodes"]:
            fail("authoritative Slurm resource counts disagree")
        return values
    tres("ReqTRES"); tres("AllocTRES")
    env_map = {"SLURM_JOB_ACCOUNT": "Account", "SLURM_JOB_PARTITION": "Partition",
               "SLURM_JOB_QOS": "QOS", "SLURM_JOB_NUM_NODES": "NumNodes",
               "SLURM_NTASKS": "NumTasks", "SLURM_CPUS_ON_NODE": "NumCPUs",
               "SLURM_CPUS_PER_TASK": "CPUs/Task", "SLURM_JOB_NODELIST": "NodeList"}
    for env_name, field in env_map.items():
        if env_name in os.environ and os.environ[env_name] != fields[field]:
            fail("Slurm environment disagrees with scheduler: " + env_name)
    if "SLURM_TIMELIMIT" in os.environ and parse_time_limit(os.environ["SLURM_TIMELIMIT"]) != parse_time_limit(fields["TimeLimit"]):
        fail("Slurm time-limit environment disagrees with scheduler")
    for name in ("SLURM_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name, "") not in ("", "0", "NoDevFiles"):
            fail("GPU allocation/environment detected")
    return {"job_id": job_id, "account": fields["Account"], "partition": fields["Partition"],
            "qos": fields["QOS"], "nodes": int(fields["NumNodes"]), "tasks": int(fields["NumTasks"]),
            "cpus_on_node": fields["NumCPUs"], "cpus_per_task": fields["CPUs/Task"],
            "node_list": fields["NodeList"], "time_limit": fields["TimeLimit"],
            "gpu_allocation": "0", "gpu_allocation_source": "scontrol.AllocTRES",
            "slurm_environment": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
            "scheduler_record": fields}


def parse_time_limit(value):
    if value.upper() in ("UNLIMITED", "INFINITE"):
        fail("finite Slurm wall limit required")
    if value.isdigit():
        return int(value) * 60
    day = 0
    if "-" in value:
        d, value = value.split("-", 1)
        day = int(d)
    fields = [int(part) for part in value.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, (minutes, seconds) = 0, fields
    else:
        fail("unrecognized Slurm time-limit format")
    return day * 86400 + hours * 3600 + minutes * 60 + seconds


def validate_admission(admission, *, actual=None, memory=None):
    if type(admission) is not dict or admission.get("schema") != "tgrab-figure10-prefix-admission-v2":
        fail("unsupported admission schema")
    required = ("protocol_id", "source_commit", "closure_sha256", "run_id", "canary_args", "job", "work_seconds",
                "outer_wall_seconds", "interpreter", "environment_receipt", "node2a_receipt")
    if any(key not in admission for key in required):
        fail("admission missing required field")
    if not isinstance(admission["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", admission["source_commit"]):
        fail("full immutable Git commit required")
    if admission["protocol_id"] != PROTOCOL_ID:
        fail("only the separately frozen 256-prefix runtime diagnostic is admitted")
    if type(admission["closure_sha256"]) is not dict:
        fail("per-file source closure hashes required")
    actual = scheduler_identity() if actual is None else actual
    expected_job = admission["job"]
    fixed = {"account": "mila", "partition": "main-cpu", "qos": "normal",
             "nodes": 1, "tasks": 1, "cpus_on_node": "1", "cpus_per_task": "1",
             "time_limit": "00:10:00", "gpu_allocation": "0", "memory_max_bytes": 2 * 1024**3}
    if type(expected_job) is not dict or any(str(expected_job.get(key)) != str(value) for key, value in fixed.items()):
        fail("allocation differs from the reviewed one-CPU/2-GiB/600-second prefix envelope")
    if expected_job.get("job_id") not in (None, actual["job_id"]):
        fail("Slurm job ID mismatch")
    for key in ("account", "partition", "qos", "nodes", "tasks", "cpus_on_node", "cpus_per_task", "time_limit", "gpu_allocation"):
        if key not in expected_job or str(expected_job[key]) != str(actual[key]):
            fail("Slurm allocation mismatch for " + key)
    if expected_job.get("node_list") not in (None, actual["node_list"]):
        fail("Slurm allocation mismatch for node_list")
    if expected_job.get("nodes") != 1 or expected_job.get("tasks") != 1:
        fail("one-node/one-task admission required")
    if str(expected_job.get("gpu_allocation")) not in ("", "0"):
        fail("CPU-only diagnostic requires zero allocated GPUs")
    memory = cgroup_memory() if memory is None else memory
    if type(expected_job.get("memory_max_bytes")) is not int or expected_job["memory_max_bytes"] != memory["effective_limit_bytes"]:
        fail("effective physical cgroup memory differs from admission")
    if type(admission["canary_args"]) is not list or not all(type(x) is str for x in admission["canary_args"]):
        fail("malformed canary argv")
    if "--run-id" not in admission["canary_args"]:
        fail("admitted argv has no run ID")
    idx = admission["canary_args"].index("--run-id")
    if idx + 1 >= len(admission["canary_args"]) or admission["canary_args"][idx + 1] != admission["run_id"]:
        fail("run ID does not match admitted argv")
    if admission["canary_args"].count("--run-id") != 1:
        fail("run ID flag must be unique")
    if type(admission["work_seconds"]) is not int or admission["work_seconds"] != 540:
        fail("prefix diagnostic requires the reviewed 540-second work bound")
    if type(admission["outer_wall_seconds"]) is not int or admission["outer_wall_seconds"] != 570 or str(admission["outer_wall_seconds"]) != os.environ.get("RC_JOB_WALL_SECONDS"):
        fail("outer shell timeout differs from reviewed 570-second admission")
    if admission["outer_wall_seconds"] > parse_time_limit(actual["time_limit"]):
        fail("outer wall duration exceeds actual Slurm allocation time limit")
    if "--deadline-unix" in admission["canary_args"] or "deadline_unix" in admission:
        fail("prequeue absolute deadlines are not admitted")
    argv = admission["canary_args"]
    if argv.count("--runtime-seconds") != 1:
        fail("admitted argv must contain exactly one runtime bound")
    i = argv.index("--runtime-seconds")
    if i + 1 >= len(argv) or argv[i + 1] != str(admission["work_seconds"]):
        fail("argv and admission runtime bound differ")
    return actual, memory


def group_members(pgid):
    members = []
    proc = Path("/proc")
    if proc.is_dir():
        for item in proc.iterdir():
            if not item.name.isdigit():
                continue
            try:
                stat_text = (item / "stat").read_text()
                tail = stat_text[stat_text.rfind(")") + 2:].split()
                if len(tail) > 2 and int(tail[2]) == pgid:
                    members.append(int(item.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
                continue
        return members
    if sys.platform == "darwin":
        try:
            os.killpg(pgid, 0)
            return [pgid]
        except ProcessLookupError:
            return []
        except PermissionError:
            return [pgid]
    try:
        os.killpg(pgid, 0)
        return [pgid]
    except ProcessLookupError:
        return []


def enable_child_subreaper():
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        fail("cannot enable child-subreeper ownership")
    return True


def reap_adopted_group_children(pgid):
    while True:
        try:
            pid, _status = os.waitpid(-pgid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def terminate_and_reap_group(child, *, grace_seconds=3.0, verify_seconds=3.0, journal=lambda *a, **k: None):
    pgid = child.pid
    journal_errors = []
    def note(event, **fields):
        try:
            journal(event, **fields)
        except BaseException as exc:
            journal_errors.append(type(exc).__name__)
    if group_members(pgid):
        note("process_group_term", pgid=pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and group_members(pgid):
        time.sleep(0.05)
    remaining = group_members(pgid)
    if remaining:
        note("process_group_kill", pgid=pgid, remaining=remaining)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=verify_seconds)
    except subprocess.TimeoutExpired:
        try: os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError: pass
        child.wait(timeout=verify_seconds)
    deadline = time.monotonic() + verify_seconds
    while time.monotonic() < deadline:
        reap_adopted_group_children(pgid)
        if not group_members(pgid):
            break
        time.sleep(0.05)
    reap_adopted_group_children(pgid)
    remaining = group_members(pgid)
    if remaining:
        raise RuntimeError("owned process group remains after TERM/KILL: " + repr(remaining))
    note("process_group_empty", pgid=pgid)
    if journal_errors:
        raise RuntimeError("journal write failed during child cleanup: " + ",".join(journal_errors))
    return child.returncode


def supervise_child(child, *, stop_path, deadline_unix, memory, memory_limit,
                    signal_state=None, check_interval=0.05, journal=lambda *a, **k: None,
                    cleanup_grace=0.5):
    reason = None
    try:
        while child.poll() is None:
            if signal_state and signal_state.get("number") is not None:
                reason = "outer_signal_" + str(signal_state["number"])
            if time.time() >= deadline_unix:
                reason = "deadline"
            current = int(Path(memory["current_path"]).read_text().strip()) if memory and memory.get("current_path") else 0
            peak_path = memory.get("peak_path") if memory else None
            peak = int(Path(peak_path).read_text().strip()) if peak_path else None
            journal("resource_sample", cgroup_memory_current_bytes=current,
                    cgroup_memory_peak_bytes=peak)
            if memory_limit is not None and current >= memory_limit:
                reason = "cgroup_memory_limit"
            if os.path.lexists(stop_path):
                reason = "STOP"
            if reason:
                break
            time.sleep(check_interval)
    except BaseException:
        terminate_and_reap_group(child, journal=lambda *a, **k: None)
        raise
    status = terminate_and_reap_group(child, grace_seconds=cleanup_grace, journal=journal)
    return status, reason


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".pending")
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path, follow_symlinks=False)
        os.unlink(tmp)
    except BaseException:
        try: tmp.unlink()
        except FileNotFoundError: pass
        raise


BOOTSTRAP = r'''import hashlib, os, sys
import importlib.abc, importlib.util, json
path = os.environ["RC_CANARY_SOURCE"]
raw = open(path, "rb").read()
if hashlib.sha256(raw).hexdigest() != os.environ["RC_EXPECTED_CANARY_SHA256"]:
    raise SystemExit("canary bytes changed after admission")
sys.dont_write_bytecode = True
src = os.environ["RC_PROJECT_SRC"]
expected = json.loads(os.environ["RC_MODULE_HASHES"])
modules = {"representation_collapse": "src/representation_collapse/__init__.py",
           "representation_collapse.graph_data": "src/representation_collapse/graph_data.py",
           "representation_collapse.graph_reconstruction": "src/representation_collapse/graph_reconstruction.py",
           "representation_collapse.pinned_graph_import": "src/representation_collapse/pinned_graph_import.py",
           "representation_collapse.reporting": "src/representation_collapse/reporting.py",
           "representation_collapse.tgrab_figure10_attempt_journal": "src/representation_collapse/tgrab_figure10_attempt_journal.py",
           "representation_collapse.tgrab_selected_checkpoint": "src/representation_collapse/tgrab_selected_checkpoint.py",
           "representation_collapse.tgrab_selection_transaction": "src/representation_collapse/tgrab_selection_transaction.py"}
class SourceOnlyLoader(importlib.abc.Loader):
    def __init__(self, name): self.name = name
    def create_module(self, spec): return None
    def exec_module(self, module):
        rel = modules[self.name]
        filename = os.path.join(os.path.dirname(src), rel)
        data = open(filename, "rb").read()
        key = rel.removeprefix("src/")
        if hashlib.sha256(data).hexdigest() != expected[key]:
            raise ImportError("source closure changed: " + key)
        module.__file__ = filename
        exec(compile(data, filename, "exec"), module.__dict__)
class SourceOnlyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in modules:
            return importlib.util.spec_from_loader(fullname, SourceOnlyLoader(fullname),
                                                   is_package=(fullname == "representation_collapse"))
        return None
sys.meta_path.insert(0, SourceOnlyFinder())
sys.path.insert(0, os.environ["RC_PROJECT_SRC"])
ns = {"__name__": "__main__", "__file__": path, "__package__": "representation_collapse"}
exec(compile(raw, path, "exec"), ns, ns)
'''


def _git_identity(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True)
    if status:
        fail("source checkout must be clean at admitted commit")
    return commit


def source_closure(repo):
    paths = ("code/ops/tgrab_figure10_canary_job.sh", "code/ops/tgrab_figure10_canary_supervisor.py",
             "src/representation_collapse/tgrab_figure10_canary.py", "src/representation_collapse/graph_data.py",
             "src/representation_collapse/graph_reconstruction.py", "src/representation_collapse/pinned_graph_import.py",
             "src/representation_collapse/reporting.py", "src/representation_collapse/__init__.py",
             "src/representation_collapse/tgrab_figure10_attempt_journal.py",
             "src/representation_collapse/tgrab_selected_checkpoint.py",
             "src/representation_collapse/tgrab_selection_transaction.py")
    return {name: sha256_path(repo / name) for name in paths}


def _verify_interpreter(spec):
    if type(spec) is not dict or not isinstance(spec.get("path"), str) or not isinstance(spec.get("sha256"), str):
        fail("pinned environment interpreter identity required")
    if spec["path"] != PINNED_INTERPRETER:
        fail("only the reviewed native CPU environment interpreter path is allowed")
    observed_path = str(Path(sys.executable).resolve(strict=True))
    expected_path = str(Path(spec["path"]).resolve(strict=True))
    if observed_path != expected_path or sha256_path(observed_path) != spec["sha256"]:
        fail("supervisor is not running from the admitted pinned interpreter")


def _verify_environment_receipt(spec):
    if type(spec) is not dict or not isinstance(spec.get("path"), str) or not isinstance(spec.get("sha256"), str):
        fail("pinned remote environment receipt path and hash required")
    if spec["path"] != PINNED_ENVIRONMENT_RECEIPT:
        fail("environment receipt path differs from the verified preparation result")
    regular_bytes(spec["path"], spec["sha256"], 1_000_000)
    return spec


def _verify_node2a(spec):
    if type(spec) is not dict or not isinstance(spec.get("path"), str) or not isinstance(spec.get("sha256"), str):
        fail("node2a pass artifact path and hash required")
    raw = regular_bytes(spec["path"], spec["sha256"], 1_000_000)
    doc = json.loads(raw)
    keys = ("both_model_components_roundtripped", "complete_hc_state_roundtripped",
            "forward_loss_gradient_parity", "mode_dropout_parity", "transaction_crash_reopen")
    if type(doc) is not dict or doc.get("schema") != "node2a-synthetic-bootstrap-pass-v1" or doc.get("status") != "pass":
        fail("node2a synthetic pre-fit pass artifact missing")
    if any(doc.get(key) is not True for key in keys) or type(doc.get("optimizer_updates")) is not int or doc.get("optimizer_updates") != 0:
        fail("node2a artifact lacks required no-training parity evidence")
    if any(doc.get(key) is not False for key in ("primary_payload_opened", "validation_payload_opened", "test_payload_opened")):
        fail("node2a artifact violates synthetic no-data boundary")
    return spec


def _require_arg_pair(argv, flag, value):
    if argv.count(flag) != 1:
        fail("admitted argv must contain exactly one " + flag)
    index = argv.index(flag)
    if index + 1 >= len(argv) or str(Path(argv[index + 1])) != str(Path(value)):
        fail("admission path differs from argv for " + flag)


def run(admission_path, admission_sha):
    wall_started_ns = int(os.environ.get("RC_WALL_START_EPOCH_NS", "0"))
    if wall_started_ns <= 0:
        fail("outer shell wall-start timestamp missing")
    raw = regular_bytes(admission_path, admission_sha)
    admission = json.loads(raw)
    repo = Path(__file__).resolve().parents[2]
    if _git_identity(repo) != admission["source_commit"]:
        fail("Git commit differs from admission")
    closure = source_closure(repo)
    if closure != admission["closure_sha256"]:
        fail("source closure differs from admission")
    _verify_interpreter(admission["interpreter"])
    env_receipt = _verify_environment_receipt(admission["environment_receipt"])
    node2a_receipt = _verify_node2a(admission["node2a_receipt"])
    _require_arg_pair(admission["canary_args"], "--environment-receipt", env_receipt["path"])
    _require_arg_pair(admission["canary_args"], "--environment-receipt-sha256", env_receipt["sha256"])
    _require_arg_pair(admission["canary_args"], "--node2a-receipt", node2a_receipt["path"])
    _require_arg_pair(admission["canary_args"], "--node2a-receipt-sha256", node2a_receipt["sha256"])
    actual, memory = validate_admission(admission)
    if memory["effective_limit_bytes"] != admission["job"]["memory_max_bytes"]:
        fail("physical memory admission mismatch")
    canary = repo / "src/representation_collapse/tgrab_figure10_canary.py"
    run_id = admission["run_id"]
    deadline_unix = wall_started_ns / 1_000_000_000 + admission["work_seconds"]
    if not math.isfinite(deadline_unix) or not 0 <= time.time() - wall_started_ns / 1_000_000_000 < admission["work_seconds"]:
        fail("invalid or exhausted job-start-relative wall clock")
    arglist = [*admission["canary_args"], "--deadline-unix", str(deadline_unix)]
    for flag in ("--run-dir", "--stop-file"):
        if arglist.count(flag) != 1 or arglist.index(flag) + 1 >= len(arglist):
            fail("missing/duplicate path flag " + flag)
    run_dir = Path(arglist[arglist.index("--run-dir") + 1])
    receipt_path = run_dir.with_name("." + run_dir.name + ".supervisor-receipt.json")
    stop_path = Path(arglist[arglist.index("--stop-file") + 1])
    child_argv = [sys.executable, "-B", "-c", BOOTSTRAP, *arglist]
    argv_hash = sha256_bytes(json.dumps(child_argv, separators=(",", ":")).encode())
    receipt = {"schema": "tgrab-figure10-supervisor-receipt-v2", "protocol_id": PROTOCOL_ID,
               "admission_sha256": admission_sha,
               "source_commit": admission["source_commit"], "closure_sha256": closure,
               "source_sha256": closure["src/representation_collapse/tgrab_figure10_canary.py"],
               "job_id": actual["job_id"],
               "job": {**admission["job"], "job_id": actual["job_id"], "node_list": actual["node_list"]},
               "slurm_environment": actual["slurm_environment"], "scheduler_record": actual["scheduler_record"],
               "argv_sha256": argv_hash,
               "child_argv": child_argv, "canary_args": arglist,
               "admitted_canary_args": admission["canary_args"], "work_seconds": admission["work_seconds"],
               "outer_wall_seconds": admission["outer_wall_seconds"],
               "environment_receipt": env_receipt, "interpreter": admission["interpreter"],
               "sys_executable": sys.executable,
               "node2a_receipt": node2a_receipt,
               "cgroup_memory": {**memory, "initial_bytes": int(Path(memory["current_path"]).read_text().strip()),
                                 "initial_peak_bytes": int(Path(memory["peak_path"]).read_text().strip()) if memory["peak_path"] else None},
               "wall_started_epoch_ns": wall_started_ns, "deadline_unix": deadline_unix}
    atomic_json(receipt_path, receipt)
    receipt_hash = sha256_path(receipt_path)
    journal_path = receipt_path.with_name("." + run_dir.name + ".supervisor-journal.jsonl")
    journal_fd = os.open(journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    journal_identity = os.fstat(journal_fd)
    os.close(journal_fd)
    def journal(event, **fields):
        fd = os.open(journal_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
                    or (st.st_dev, st.st_ino) != (journal_identity.st_dev, journal_identity.st_ino)):
                fail("supervisor journal identity changed")
            payload = (json.dumps({"time_epoch_ns": time.time_ns(), "event": event, **fields}, sort_keys=True) + "\n").encode()
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    fail("short supervisor journal write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "RC_SUPERVISOR_RECEIPT": str(receipt_path), "RC_SUPERVISOR_RECEIPT_SHA256": receipt_hash,
                "RC_CANARY_SOURCE": str(canary), "RC_EXPECTED_CANARY_SHA256": receipt["source_sha256"],
                "RC_PROJECT_SRC": str(repo / "src"), "RC_MODULE_HASHES": json.dumps(
                    {name.removeprefix("src/"): digest for name, digest in closure.items()
                     if name.startswith("src/representation_collapse/")}),
                "PYTHONPATH": str(repo / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    timeout = deadline_unix - time.time()
    if timeout <= 0:
        fail("deadline expired before child launch")
    if timeout > admission["outer_wall_seconds"]:
        fail("absolute work deadline exceeds the import-inclusive outer wall cap")
    enable_child_subreaper()
    stop_signal = {"number": None}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    def request_stop(number, _frame):
        stop_signal["number"] = number
    for sig in previous:
        signal.signal(sig, request_stop)
    child = None
    reason = None
    started = time.monotonic()
    try:
        child = subprocess.Popen(child_argv, cwd=repo, env=env, start_new_session=True)
        journal("child_started", pid=child.pid, pgid=child.pid, argv_sha256=argv_hash)
        status, reason = supervise_child(child, stop_path=stop_path,
            deadline_unix=deadline_unix, memory=memory,
            memory_limit=memory["effective_limit_bytes"], signal_state=stop_signal,
            check_interval=min(1.0, max(0.05, timeout)), journal=journal)
        journal("child_finished", pid=child.pid, returncode=status, stop_reason=reason,
                elapsed_seconds=time.monotonic() - started,
                final_cgroup_memory_current_bytes=int(Path(memory["current_path"]).read_text().strip()),
                final_cgroup_memory_peak_bytes=int(Path(memory["peak_path"]).read_text().strip()) if memory["peak_path"] else None)
    except BaseException:
        if child is not None:
            try:
                terminate_and_reap_group(child, journal=lambda *a, **k: None)
            except BaseException as cleanup_error:
                try: journal("cleanup_failed", error_type=type(cleanup_error).__name__)
                except BaseException: pass
        raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
    if reason == "deadline": return 124
    if reason == "STOP" or reason == "cgroup_memory_limit" or reason and reason.startswith("outer_signal_"): return 137
    return status


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--admission" or sys.argv[3] != "--sha256":
        fail("usage: supervisor --admission PATH --sha256 HASH")
    raise SystemExit(run(sys.argv[2], sys.argv[4]))
