#!/usr/bin/env python3
"""Submit one owner-admitted primary conversion through pinned native cluv.

Import is stdlib only. Preparation/tests confer no allocation authority.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import socket
import stat
import subprocess
import sys

SCRATCH = Path("/network/scratch/m/mehrab.hamidi/representation-collapse")
CONTROLLER = "code/ops/submit_primary_conversion.py"
PROTOCOL = "doc/auto/primary-conversion-cluv-controller-2026-10-16.md"
WORKER = "code/ops/graph_data_conversion_job.py"
SHELL = "code/ops/graph_data_conversion_job.sh"
CONFIG = "configs/graph/primary-data-conversion-v1.json"
REQUEST = "doc/auto/graph-data-conversion-launch-request-2026-10-16.md"
LIBRARY_PINS = {
    "cluv/cache.py": "54fb84245bc4a9804b01bb450048a25ae7d4ffe2d96801ae30c466efd2beb742",
    "cluv/cli/submit.py": "b6e6f2668b04b956174c98ea3f3d2266a3d3c394765a4fc151ec8be62ac49f04",
    "cluv/cli/sync.py": "336d49af8c9effc9fe5ab52914f8069d34ca7e655dfc932e6d07fc5e971c3d59",
    "cluv/config.py": "1648a71704059d56a2e6654e370ac3987c6b7ec7c78c259e08ef4e485135040d",
    "cluv/remote.py": "62db4b34a721ab6c2bcfc09d908c6dfe0f35a5828f1dac620ce828f98aa6f883",
    "cluv/sbatch_args.py": "7df36316a700f35e47a050085c91f4b89b1213c8c91c7adad30814401f0ac9d5",
}
ALLOCATION = {"cluster": "mila", "account": "mila", "qos": "normal", "partition": "main-cpu",
              "max_submissions": 1, "retries": 0, "requeue": False, "race": False}
RESOURCES = {"cpus": 1, "memory_bytes": 2147483648, "gpus": 0, "wall_seconds": 600, "work_seconds": 590}
SBATCH = {"account": "mila", "qos": "normal", "partition": "main-cpu", "nodes": 1, "ntasks": 1,
          "cpus-per-task": 1, "mem": "2G", "time": "00:10:00", "no-requeue": True,
          "job-name": "rc-primary-conversion-v1"}
BINDINGS = ("source_commit", "source_files", "output", "stop_path", "limits", "max_submissions", "retries")
LOGIN_GUARD = """for rc_primary_sbatch_option in ${!SBATCH_@}; do
  printf '%s\\n' 'primary conversion: inherited scheduler input rejected' >&2
  exit 2
done
if [[ ${SLURM_CLUSTERS+x} || -n ${PYTHONOPTIMIZE:-} ]]; then
  printf '%s\\n' 'primary conversion: inherited cluster or optimization input rejected' >&2
  exit 2
fi
"""


class SubmissionError(RuntimeError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")


def exact(left, right):
    return encoded(left) == encoded(right)


def regular(path, limit=2 * 1024 * 1024):
    path = Path(path)
    for item in (path, *path.parents):
        if item.is_symlink():
            raise SubmissionError("symlink component refused")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size > limit:
        raise SubmissionError("bounded single-link regular file required")
    return path


def sha256(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def read_bytes(path, limit=2 * 1024 * 1024):
    path = regular(path, limit)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise SubmissionError("stable bounded metadata required")
        chunks, count = [], 0
        while True:
            block = os.read(fd, min(65536, limit + 1 - count))
            if not block:
                break
            chunks.append(block); count += len(block)
            if count > limit:
                raise SubmissionError("metadata grew beyond bound")
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(os.fstat(fd)) or identity(before) != identity(path.lstat()) or count != before.st_size:
            raise SubmissionError("metadata changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def decode_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise SubmissionError("duplicate metadata key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(SubmissionError("nonfinite metadata")))


def read_json(path):
    return decode_json(read_bytes(path))


def write_new(path, data):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def scratch_path(value, *, absent=False):
    path = Path(value)
    if (not path.is_absolute() or str(path) != os.path.abspath(path) or SCRATCH not in path.parents
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)) or ".." in path.parts):
        raise SubmissionError("canonical shell-safe project scratch path required")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise SubmissionError("scratch symlink refused")
        if item.exists() and (item == SCRATCH or SCRATCH in item.parents) and item.stat().st_uid != os.getuid():
            raise SubmissionError("owned project scratch paths required")
    if absent and path.exists():
        raise SubmissionError("preexisting output/STOP/claim/controller evidence refused")
    return path


def reject_environment():
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise SubmissionError("normal nonoptimized Python required")
    if any(key.startswith("SBATCH_") or key == "SLURM_CLUSTERS" for key in os.environ):
        raise SubmissionError("inherited scheduler input rejected")


def claim_payload(plan):
    return {"kind": "root_exclusive_submission_claim", "status": "verified", "exclusive": True,
            "claim_id": plan["claim_id"], "claimed_submissions": 1,
            **{key: plan[key] for key in BINDINGS}}


def pinned_record(record):
    data = read_bytes(record["path"])
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise SubmissionError("external receipt hash mismatch")
    return decode_json(data)


def preflight(plan_path, plan_sha):
    reject_environment()
    if not socket.gethostname().endswith(".server.mila.quebec"):
        raise SubmissionError("actual Mila login host required")
    if any(os.environ.get(key) not in (None, "mila") for key in ("CLUV_CLUSTER", "CC_CLUSTER")):
        raise SubmissionError("ambient cluster identity mismatch")
    plan_path = scratch_path(plan_path)
    plan_bytes = read_bytes(plan_path)
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha) or hashlib.sha256(plan_bytes).hexdigest() != plan_sha:
        raise SubmissionError("external plan hash mismatch")
    plan = decode_json(plan_bytes)
    expected = {"kind": "primary_data_conversion_single_submission_plan", "owner": "root coordinator",
                "protocol_id": "primary_data_conversion_v1", "execution_authorized": True,
                "training_authorized": False, "scientific_acceptance": False, "max_submissions": 1, "retries": 0}
    if any(not exact(plan.get(key), value) for key, value in expected.items()):
        raise SubmissionError("exact root one-submission conversion plan required")
    root = scratch_path(plan["source_root"])
    if Path.cwd() != root or Path(__file__).absolute() != root / CONTROLLER:
        raise SubmissionError("execute exact controller from isolated published source cwd")
    config_bytes = read_bytes(root / CONFIG); config = decode_json(config_bytes)
    if hashlib.sha256(config_bytes).hexdigest() != plan["config_sha256"] or config.get("execution_authorized") is not True:
        raise SubmissionError("exact technically enabled config required")
    if config.get("training_authorized") is not False or config.get("scientific_acceptance") is not False or config.get("protocol_id") != expected["protocol_id"]:
        raise SubmissionError("conversion-only config required")
    for key, value in (("resources", RESOURCES), ("allocation", ALLOCATION)):
        if not exact(config[key], value) or not exact(plan[key], value):
            raise SubmissionError("allocation/resource expansion changed")
    if not exact(plan["limits"], config["limits"]):
        raise SubmissionError("finite count limits mismatch")
    required = {WORKER, SHELL, CONFIG, REQUEST, "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]}
    required.update(record["path"] for record in config["sources"].values())
    if CONTROLLER not in required or PROTOCOL not in required or set(plan["source_files"]) != required:
        raise SubmissionError("complete controller/worker/config/protocol source closure required")
    for relative, pin in plan["source_files"].items():
        if relative.startswith("/") or ".." in Path(relative).parts or not re.fullmatch(r"[0-9a-f]{64}", str(pin)) or sha256(root / relative) != pin:
            raise SubmissionError("source file identity mismatch")
    for record in (*config["sources"].values(), config["reconstruction_config"]):
        if record["sha256"] != plan["source_files"][record["path"]]:
            raise SubmissionError("configured helper source pin mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(plan["source_commit"])):
        raise SubmissionError("published commit identity required")
    for revision in ("HEAD", "refs/remotes/origin/main"):
        actual = subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True, timeout=2).strip()
        if actual != plan["source_commit"]:
            raise SubmissionError("HEAD/published origin main must equal approved commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, timeout=2).strip():
        raise SubmissionError("clean published source required")
    run_root = plan_path.parent.parent
    if plan_path != run_root / "inputs/plan.json":
        raise SubmissionError("fixed inputs plan path required")
    for key, expected_path in {
        "output": run_root / "attempt001", "stop_path": run_root / "STOP",
        "controller_cache": plan_path.parent / "controller-cache", "preview_receipt": plan_path.parent / "native-submission-preview.json",
        "submission_receipt": plan_path.parent / "native-submission-receipt.json"}.items():
        if scratch_path(plan[key], absent=True) != expected_path:
            raise SubmissionError("exact new run/control path required")
    if not run_root.is_dir() or not plan_path.parent.is_dir():
        raise SubmissionError("prepared owned input/run parents required")
    scratch_path(plan_path.parent / "native-submission-intent.json", absent=True)
    logs = scratch_path(plan["logs"])
    if logs != SCRATCH / "logs" / run_root.name or (logs.exists() and (not logs.is_dir() or next(logs.iterdir(), None) is not None)):
        raise SubmissionError("isolated fresh or empty exact run log root required")
    if root == run_root or root in run_root.parents or run_root in root.parents:
        raise SubmissionError("isolated source and run trees required")
    if plan["owner_approval"]["path"] != str(plan_path.parent / "owner-allocation-approval.json"):
        raise SubmissionError("unique run owner approval path required")
    approval = pinned_record(plan["owner_approval"])
    if (approval.get("kind") != "primary_data_conversion_owner_allocation_approval" or approval.get("status") != "verified"
            or approval.get("approved_by") != "owner" or approval.get("approved") is not True):
        raise SubmissionError("fresh explicit owner-approved allocation receipt required")
    for key in (*BINDINGS, "config_sha256", "allocation", "resources"):
        if not exact(approval.get(key), plan.get(key)):
            raise SubmissionError("owner receipt must bind exact source/config/output/STOP/caps/resources")
    for key in ("implementation_review", "reporting_readiness", "environment", "storage_gate"):
        record = pinned_record(plan[key])
        if record.get("status") != "verified":
            raise SubmissionError("verified worker readiness receipts required")
        if key in ("implementation_review", "reporting_readiness") and record.get("config_sha256") != plan["config_sha256"]:
            raise SubmissionError("config-bound review/reporting readiness required")
        if key == "environment" and (record.get("python") != "3.10.19" or record.get("wandb") != "0.30.0" or record.get("python_executable") != config["python"]):
            raise SubmissionError("exact existing worker environment required")
        if key == "storage_gate":
            used, reserved = record.get("used_bytes"), record.get("reserved_bytes")
            if (type(used) is not int or type(reserved) is not int or used < 0 or reserved < 1073741824
                    or used + reserved > config["limits"]["project_scratch_bytes"] or record.get("source_root") != str(root) or record.get("output") != plan["output"]):
                raise SubmissionError("whole-project source/output/cache/log reservation required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", str(plan["claim_id"])):
        raise SubmissionError("explicit unique claim ID required")
    claim = scratch_path(plan["submission_claim"]["path"], absent=True)
    if claim != plan_path.parent / "native-submission-claim.json" or hashlib.sha256(encoded(claim_payload(plan))).hexdigest() != plan["submission_claim"]["sha256"]:
        raise SubmissionError("exclusive claim path/precomputed byte hash mismatch")
    supervisor = [config["python"], "-B", str(root / WORKER), "--plan", str(plan_path), "--plan-sha256", "{plan_sha256}"]
    if plan["supervisor_argv"] != supervisor or not exact(plan["native_cluv_source_sha256"], LIBRARY_PINS):
        raise SubmissionError("exact supervisor argv and audited native library pins required")
    return {"plan": plan, "config": config, "root": root, "plan_path": plan_path, "plan_sha256": plan_sha}


def verify_native_library():
    distribution = importlib.metadata.distribution("cluster-uv")
    if distribution.version != "0.1.5":
        raise SubmissionError("exact native cluster-uv0.1.5 required")
    paths = {member: Path(distribution.locate_file(member)).absolute() for member in LIBRARY_PINS}
    for member, pin in LIBRARY_PINS.items():
        if sha256(paths[member]) != pin:
            raise SubmissionError("actual native cluv library source identity mismatch")
    return paths


def prepare_native(context):
    reject_environment(); paths = verify_native_library()
    if any(name == "cluv" or name.startswith("cluv.") for name in sys.modules):
        raise SubmissionError("fresh process before native cluv import required")
    plan = context["plan"]
    Path(plan["controller_cache"]).mkdir(mode=0o700, exist_ok=False)
    os.environ["XDG_CACHE_HOME"] = plan["controller_cache"]
    config_module = importlib.import_module("cluv.config")
    module = importlib.import_module("cluv.cli.submit")
    for member, expected_path in paths.items():
        loaded = sys.modules.get(member[:-3].replace("/", "."))
        if loaded is None or Path(loaded.__file__).absolute() != expected_path or sha256(loaded.__file__) != LIBRARY_PINS[member]:
            raise SubmissionError("actual imported native module must match pinned installed source")
    config = config_module.get_cluv_config()
    config.env = {}; config.sbatch_args = {}; config.project_dir = str(context["root"])
    config.results_path = plan["logs"]; config.job_script_path = SHELL
    config.clusters = {"mila": dataclasses.replace(config.clusters["mila"], env={}, project_dir=config.project_dir,
        results_path=config.results_path, job_script_path=SHELL, sbatch_args=[dict(SBATCH)])}
    verify_native_library()
    return module


def guard_row(rows, context):
    plan, root = context["plan"], context["root"]
    if len(rows) != 1:
        raise SubmissionError("exactly one native submission row required")
    row = rows[0]
    arguments = ["--plan", str(context["plan_path"]), "--plan-sha256", context["plan_sha256"]]
    resources = {key: str(value) if type(value) is int else value for key, value in SBATCH.items()}
    resources.update({"job-name": "cluv-rc-primary-conversion-v1", "output": plan["logs"] + "/mila_%j/slurm-%j.out", "chdir": str(root), "export": "ALL"})
    if (row.cluster != "mila" or row.remote is not None or row.n_chunks is not None or not exact(row.sbatch_args, resources)
            or row.git_commit != plan["source_commit"] or row.job_script != Path(SHELL) or row.program_args != arguments):
        raise SubmissionError("native allocation/source/script/argv expansion differs from approved single job")
    words = shlex.split(row.sbatch_command)
    if len(words) != 4 or words[:3] != ["bash", "--login", "-c"]:
        raise SubmissionError("unexpected native login-shell command shape")
    prefix = f"GIT_COMMIT={plan['source_commit']} CLUV_CLUSTER=mila; cd {root} && sbatch --parsable "
    if not words[3].startswith(prefix):
        raise SubmissionError("native environment/source shell prefix mismatch")
    tokens = shlex.split(words[3][len(prefix):])
    position = next((index for index, token in enumerate(tokens) if token == SHELL), -1)
    if position < 0 or tokens[position + 1:] != arguments:
        raise SubmissionError("actual shell script/argv mismatch")
    actual = {}
    for token in tokens[:position]:
        if not token.startswith("--"):
            raise SubmissionError("unexpected actual sbatch token")
        key, separator, value = token[2:].partition("=")
        if key in actual:
            raise SubmissionError("duplicate actual sbatch flag")
        actual[key] = value if separator else True
    if not exact(actual, resources):
        raise SubmissionError("actual shell sbatch resources differ from approved row")
    guarded = dataclasses.replace(row, sbatch_command=shlex.join(words[:3] + [LOGIN_GUARD + words[3]]))
    return guarded


async def submit_once(context, native):
    plan = context["plan"]; original = native.get_submissions; calls = 0
    async def guarded_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise SubmissionError("native expansion attempted more than once")
        row = guard_row(await original(*args, **kwargs), context)
        write_new(plan["preview_receipt"], encoded({"kind": "primary_conversion_native_preview", "plan_sha256": context["plan_sha256"],
            "resources": row.sbatch_args, "program_args": row.program_args, "source_commit": row.git_commit,
            "guarded_native_sbatch_command": row.sbatch_command, "native_cluv": "0.1.5"}))
        return [row]
    native.get_submissions = guarded_get
    outcome = {"kind": "primary_conversion_submission_outcome", "plan_sha256": context["plan_sha256"],
               "claim_id": plan["claim_id"], "cluster": "mila", "retry_authorized": False, "job_id": None}
    intent_path = context["plan_path"].parent / "native-submission-intent.json"
    try:
        reject_environment()
        plan_bytes = read_bytes(context["plan_path"])
        if hashlib.sha256(plan_bytes).hexdigest() != context["plan_sha256"] or not exact(decode_json(plan_bytes), plan):
            raise SubmissionError("plan changed before exclusive claim")
        approval = pinned_record(plan["owner_approval"])
        if approval.get("approved") is not True or approval.get("approved_by") != "owner" or approval.get("status") != "verified":
            raise SubmissionError("owner approval changed before claim")
        if any(not exact(approval.get(key), plan.get(key)) for key in (*BINDINGS, "config_sha256", "allocation", "resources")):
            raise SubmissionError("owner binding changed before claim")
        for key in ("stop_path", "output"):
            if Path(plan[key]).exists() or Path(plan[key]).is_symlink():
                raise SubmissionError("owner STOP or existing output before claim")
        if hashlib.sha256(encoded(claim_payload(plan))).hexdigest() != plan["submission_claim"]["sha256"]:
            raise SubmissionError("claim bytes changed before exclusive publication")
        write_new(intent_path, encoded({**outcome, "state": "intent_recorded", "resources": SBATCH}))
        # Claim contains no plan hash, avoiding a circular plan->claim->plan hash.
        write_new(plan["submission_claim"]["path"], encoded(claim_payload(plan)))
        job = await native.submit("mila", Path(SHELL), [],
            ["--plan", str(context["plan_path"]), "--plan-sha256", context["plan_sha256"]], autocommit=False,
            chunking=None, vram=None, _skip_sync=True, sync_datasets=False, parsable=True)
        if calls != 1:
            raise SubmissionError("native submit did not use exactly one guarded expansion")
        if job is not None and (getattr(job, "cluster", None) != "mila" or not re.fullmatch(r"[1-9][0-9]*", str(job.job_id))):
            raise SubmissionError("native returned invalid job identity")
        outcome.update(state="native_returned_job_requires_scheduler_reconciliation" if job else "ambiguous_requires_reconciliation", job_id=str(job.job_id) if job else None)
    except BaseException as error:
        outcome.update(state="ambiguous_requires_reconciliation", error_type=type(error).__name__)
    finally:
        native.get_submissions = original
        outcome["guarded_expansion_attempts"] = calls
        write_new(plan["submission_receipt"], encoded(outcome))
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    context = preflight(args.plan, args.plan_sha256)
    native = prepare_native(context)
    result = asyncio.run(submit_once(context, native))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["job_id"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
