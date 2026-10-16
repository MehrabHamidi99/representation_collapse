#!/usr/bin/env python3
"""Prepare/supervise one separately admitted primary CSV conversion, never submit.

Stdlib-only import. The disabled checked-in configuration is not admission.
"""
from __future__ import annotations

import time
PROCESS_STARTED = time.monotonic()

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import runpy
import signal
import stat
import subprocess
import sys

PROTOCOL_ID = "primary_data_conversion_v1"
MODULE_PATH = "code/ops/graph_data_conversion_job.py"
SHELL_PATH = "code/ops/graph_data_conversion_job.sh"
CONFIG_PATH = "configs/graph/primary-data-conversion-v1.json"
REQUEST_PATH = "doc/auto/graph-data-conversion-launch-request-2026-10-16.md"
SCRATCH = Path("/network/scratch/m/mehrab.hamidi/representation-collapse")
POLICY_SHA256 = "b2b426987bbfc8a6766d239c50af94ec4658d9a1630631eed17bc9cc0d1d6534"


class ConversionJobError(RuntimeError):
    pass


def exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def regular(path):
    path = Path(os.path.abspath(os.fspath(path)))
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ConversionJobError("symlink path component refused")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ConversionJobError("single-link regular file required")
    return details


def sha256(path, limit=2 * 1024 * 1024, guard=lambda: None):
    """Only bounded source/config/receipt files before reporter entry."""
    guard()
    if regular(path).st_size > limit:
        raise ConversionJobError("file exceeds declared bound")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        count = 0
        for chunk in iter(lambda: stream.read(65536), b""):
            guard()
            count += len(chunk)
            if count > limit:
                raise ConversionJobError("file grew beyond bound")
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if regular(path).st_size > 2 * 1024 * 1024:
        raise ConversionJobError("oversized metadata")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ConversionJobError("duplicate metadata key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ConversionJobError("nonfinite metadata")))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def load_config(path):
    config = read_json(path)
    policy = json.loads(json.dumps(config))
    policy["execution_authorized"] = False
    for key in ("converter", "conversion_protocol"):
        policy["sources"][key]["sha256"] = None
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != POLICY_SHA256:
        raise ConversionJobError("frozen policy changed")
    return config


def pinned_record(record):
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))):
        raise ConversionJobError("concrete SHA256 pin required")
    if sha256(record["path"]) != record["sha256"]:
        raise ConversionJobError("receipt pin mismatch")
    return read_json(record["path"])


def scratch_path(value, *, absent=False):
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)) or SCRATCH not in path.parents:
        raise ConversionJobError("canonical project scratch path required")
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ConversionJobError("scratch symlink refused")
    if absent and path.exists():
        raise ConversionJobError("fresh output and absent STOP required")
    return path


def verify_plan(args):
    if sha256(args.plan) != args.plan_sha256:
        raise ConversionJobError("external coordinator plan pin mismatch")
    plan = read_json(args.plan)
    expected = {"kind": "primary_data_conversion_single_submission_plan", "owner": "root coordinator",
                "protocol_id": PROTOCOL_ID, "execution_authorized": True, "training_authorized": False,
                "scientific_acceptance": False, "max_submissions": 1, "retries": 0}
    if any(type(plan.get(k)) is not type(v) or plan.get(k) != v for k, v in expected.items()):
        raise ConversionJobError("explicit externally approved root plan required")
    root = scratch_path(plan["source_root"])
    if Path(__file__).absolute() != root / MODULE_PATH or Path.cwd() != root:
        raise ConversionJobError("isolated published source root required")
    config_path = root / CONFIG_PATH
    if sha256(config_path) != plan["config_sha256"]:
        raise ConversionJobError("plan config pin mismatch")
    config = load_config(config_path)
    if config["execution_authorized"] is not True:
        raise ConversionJobError("prepared configuration is disabled")
    required = {MODULE_PATH, SHELL_PATH, CONFIG_PATH, REQUEST_PATH, "src/representation_collapse/__init__.py", config["reconstruction_config"]["path"]}
    required.update(record["path"] for record in config["sources"].values())
    if set(plan["source_files"]) != required:
        raise ConversionJobError("exact source closure required")
    for relative in required:
        if relative.startswith("/") or ".." in Path(relative).parts or sha256(root / relative) != plan["source_files"][relative]:
            raise ConversionJobError("published source pin mismatch")
    for record in (*config["sources"].values(), config["reconstruction_config"]):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])) or sha256(root / record["path"]) != record["sha256"]:
            raise ConversionJobError("missing or nonexact helper/converter/protocol pin")
    if not all(exact(plan[key], config[key]) for key in ("allocation", "resources", "limits")):
        raise ConversionJobError("admission must bind allocation, resources and finite limits")
    if not re.fullmatch(r"[0-9a-f]{40}", str(plan["source_commit"])):
        raise ConversionJobError("published source commit required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=2).strip()
    if head != plan["source_commit"]:
        raise ConversionJobError("source commit mismatch")
    if str(Path(sys.executable).absolute()) != config["python"]:
        raise ConversionJobError("exact existing isolated Python executable required")
    # A literal placeholder avoids putting a plan's own hash inside its bytes.
    expected_argv = [config["python"], "-B", str(root / MODULE_PATH), "--plan", str(args.plan.absolute()), "--plan-sha256", "{plan_sha256}"]
    if plan["supervisor_argv"] != expected_argv:
        raise ConversionJobError("exact supervisor argv required")
    if not args.worker and [str(Path(sys.executable).absolute()), "-B", str(Path(__file__).absolute()), *sys.argv[1:]] != expected_argv[:-1] + [args.plan_sha256]:
        raise ConversionJobError("actual supervisor argv differs from frozen plan")
    output = scratch_path(plan["output"], absent=not args.worker)
    stop = scratch_path(plan["stop_path"], absent=True)
    if output == stop or output in stop.parents or stop in output.parents or not output.parent.is_dir() or not stop.parent.is_dir():
        raise ConversionJobError("distinct STOP outside fresh output with existing parent required")
    for name in ("implementation_review", "reporting_readiness", "environment", "storage_gate", "owner_approval", "submission_claim"):
        record = pinned_record(plan[name])
        if record.get("status") != "verified":
            raise ConversionJobError("verified admission receipts required")
        if name in ("implementation_review", "reporting_readiness", "owner_approval") and record.get("config_sha256") != plan["config_sha256"]:
            raise ConversionJobError("config-bound review/reporting/approval receipt required")
        if name == "owner_approval" and (not exact(record.get("allocation"), config["allocation"]) or not exact(record.get("resources"), config["resources"])):
            raise ConversionJobError("owner must approve exact new allocation")
        if name in ("owner_approval", "submission_claim") and any(not exact(record.get(key), plan.get(key)) for key in (
                "source_commit", "source_files", "output", "stop_path", "limits", "max_submissions", "retries")):
            raise ConversionJobError("root owner admission must bind exact source/output/STOP/count caps")
        if name == "submission_claim" and (record.get("kind") != "root_exclusive_submission_claim" or record.get("exclusive") is not True
                or type(record.get("claimed_submissions")) is not int or record["claimed_submissions"] != 1
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", str(record.get("claim_id", "")))):
            raise ConversionJobError("root exclusive single-submission claim required")
        if name == "environment" and (record.get("python") != "3.10.19" or record.get("wandb") != "0.30.0" or record.get("python_executable") != config["python"]):
            raise ConversionJobError("verified existing exact Python/W&B environment required")
        if name == "storage_gate":
            used, reserved = record.get("used_bytes"), record.get("reserved_bytes")
            if (type(used) is not int or type(reserved) is not int or used < 0 or reserved < config["limits"]["conversion_bytes"] + config["limits"]["evidence_bytes"]
                    or used + reserved > config["limits"]["project_scratch_bytes"] or record.get("source_root") != str(root) or record.get("output") != str(output)):
                raise ConversionJobError("root source/output scratch reservation gate required")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise ConversionJobError("actual one-CPU Slurm job required")
    for key, value in (("SLURM_JOB_ACCOUNT", "mila"), ("SLURM_JOB_PARTITION", "main-cpu"), ("SLURM_JOB_QOS", "normal")):
        if os.environ.get(key) is not None and os.environ.get(key) != value:
            raise ConversionJobError("actual Slurm account/partition/QOS mismatch")
    return plan, config, root, output


def scheduler_receipt(job_id):
    """One bounded, read-only query of this attempt's actual allocation only."""
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ConversionJobError("numeric actual job identity required")
    result = subprocess.run(["scontrol", "show", "job", "-o", job_id], capture_output=True, text=True, timeout=5, check=True)
    if len(result.stdout.encode()) > 65536:
        raise ConversionJobError("oversized scheduler response")
    expected = {"JobId": job_id, "Account": "mila", "QOS": "normal", "Partition": "main-cpu",
                "NumCPUs": "1", "NumNodes": "1", "NumTasks": "1", "TimeLimit": "00:10:00", "Requeue": "0"}
    wanted = set(expected) | {"ReqTRES", "AllocTRES"}
    selected = {}
    for token in result.stdout.split():
        key, separator, value = token.partition("=")
        if separator and key in wanted:
            if key in selected:
                raise ConversionJobError("duplicate scheduler field")
            selected[key] = value
    for key in ("ReqTRES", "AllocTRES"):
        tres = {}
        for part in selected.get(key, "").split(","):
            name, separator, value = part.partition("=")
            if not separator or name in tres:
                raise ConversionJobError("missing or duplicate scheduler TRES")
            tres[name] = value
        memory = tres.get("mem")
        if memory not in ("2G", "2048M", "2097152K", "2147483648"):
            raise ConversionJobError("scheduler TRES memory differs from2GiB")
        tres["mem"] = "2147483648"
        if tres != {"cpu": "1", "mem": "2147483648", "node": "1"}:
            raise ConversionJobError("scheduler TRES requires one CPU/node and zero GPUs")
        selected[key] = tres
    expected.update(ReqTRES={"cpu": "1", "mem": "2147483648", "node": "1"}, AllocTRES={"cpu": "1", "mem": "2147483648", "node": "1"})
    if selected != expected:
        raise ConversionJobError("actual scheduler allocation differs from approved envelope")
    return {"kind": "primary_conversion_scheduler_v1", "status": "verified", "fields": selected}


def kernel_memory_limit(proc_cgroup=Path("/proc/self/cgroup"), cgroup_root=Path("/sys/fs/cgroup")):
    candidates = []
    for hierarchy, controllers, relative in (line.split(":", 2) for line in proc_cgroup.read_text().splitlines()):
        if ".." in Path(relative).parts:
            raise ConversionJobError("unsafe cgroup path")
        if hierarchy == "0" and controllers == "":
            base, name = cgroup_root, "memory.max"
        elif "memory" in controllers.split(","):
            base, name = cgroup_root / "memory", "memory.limit_in_bytes"
        else:
            continue
        current = base / relative.lstrip("/")
        while True:
            path = current / name
            if path.is_file() and path.read_text().strip() != "max":
                value = int(path.read_text().strip())
                if value > 0:
                    candidates.append({"path": str(path), "physical_memory_limit_bytes": value})
            if current == base:
                break
            current = current.parent
    if not candidates:
        raise ConversionJobError("mandatory physical cgroup cap unavailable")
    return min(candidates, key=lambda item: item["physical_memory_limit_bytes"])


def runtime(config):
    if sys.platform != "linux" or platform.machine() != "x86_64" or platform.python_version() != "3.10.19" or sys.implementation.name != "cpython":
        raise ConversionJobError("exact existing Linux/x86_64/Python3.10.19 required")
    if importlib.metadata.version("wandb") != "0.30.0":
        raise ConversionJobError("exact W&B0.30.0 required")
    memory = kernel_memory_limit()
    if memory["physical_memory_limit_bytes"] > config["resources"]["memory_bytes"]:
        raise ConversionJobError("physical memory cgroup exceeds2GiB")
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) != 1:
        raise ConversionJobError("actual one-CPU affinity required")
    return {"memory": memory, "cpu_affinity": sorted(os.sched_getaffinity(0)), "python": platform.python_version(), "sys_version": sys.version,
            "implementation": sys.implementation.name, "wandb": "0.30.0"}


def inventory(path, *, max_files, max_bytes, guard=lambda: None, hashes=False, sdk_metadata=False, live=False):
    records, size, directory_count, vanished = [], 0, 0, []
    for parent, directories, files in os.walk(path, followlinks=False):
        guard()
        directory_count += len(directories)
        if directory_count > 2 * max_files:
            raise ConversionJobError("finite output directory bound exceeded")
        for name in directories:
            if (Path(parent) / name).is_symlink():
                raise ConversionJobError("output symlink refused")
        for name in files:
            guard(); file = Path(parent) / name
            relative = str(file.relative_to(path))
            sdk_scope = sdk_metadata and (relative.startswith("worker/reporting/") or relative.startswith("reporting-support/"))
            try:
                details = file.lstat()
                match = re.fullmatch(r"worker/reporting/wandb/offline-run-[0-9]{8}_[0-9]{6}-([A-Za-z0-9_.-]{1,96})/logs/debug-core\.log", relative)
                skipped = stat.S_ISLNK(details.st_mode) and sdk_metadata and match is not None
                if skipped:
                    reporting = read_json(Path(path) / "worker/reporting/reporting-receipt.json")
                    if reporting.get("mode") != "offline" or reporting.get("sdk_version_required") != "0.30.0" or reporting.get("run_id") != match.group(1):
                        raise ConversionJobError("SDK debug symlink must belong to this exact offline run")
                count = details.st_size if skipped else regular(file).st_size
            except FileNotFoundError:
                if not live or not sdk_scope or len(vanished) >= 16:
                    raise
                vanished.append(relative); continue
            size += count
            if len(records) >= max_files or size > max_bytes:
                raise ConversionJobError("finite output bound exceeded")
            record = {"path": relative, "bytes": count}
            if skipped:
                record["skipped_sdk_symlink"] = True
            elif hashes:
                record["sha256"] = sha256(file, max_bytes, guard)
            records.append(record)
    return {"files": sorted(records, key=lambda item: item["path"]), "bytes": size, "vanished_sdk_files": vanished}


def reporter_metadata(plan, config):
    return ({"protocol_id": PROTOCOL_ID, "nodes": 100, "timestamps": 3072, "cpu_threads": 1,
             "training": False, "generation": False, "module_path": MODULE_PATH, "config_path": CONFIG_PATH},
            {"source_commit": plan["source_commit"], "config_sha256": plan["config_sha256"],
             "converter_sha256": config["sources"]["converter"]["sha256"],
             "protocol_sha256": config["sources"]["conversion_protocol"]["sha256"],
             "reporter_sha256": config["sources"]["reporter"]["sha256"],
             "input_sha256": config["raw"]["sha256"], "environment_sha256": plan["environment"]["sha256"]})


def worker(args, plan, config, root, output):
    admission = pinned_record({"path": str(args.worker_admission), "sha256": args.worker_admission_sha256})
    started = admission["started_monotonic"]
    if type(started) not in (int, float) or not math.isfinite(started):
        raise ConversionJobError("finite original monotonic clock required")
    deadline = started + 590
    if (admission.get("plan_sha256") != args.plan_sha256 or admission.get("slurm_job_id") != os.environ["SLURM_JOB_ID"]
            or not 0 < deadline - time.monotonic() <= 590 or admission.get("worker_argv") != [str(Path(sys.executable).absolute()), "-B", str(root / MODULE_PATH), *sys.argv[1:-2]]
            or args.worker_admission != output / "worker-admission.json" or sys.argv[-2:] != ["--worker-admission-sha256", args.worker_admission_sha256]):
        raise ConversionJobError("shared clock/job/argv worker binding required")
    worker_output = output / "worker"; worker_output.mkdir(exist_ok=False)
    status = {"state": "initializing", "protocol_id": PROTOCOL_ID, "conversion_attempts": 0,
              "scientific_acceptance": False, "remote_delivery_verified": False}
    def persist():
        status["elapsed_seconds"] = time.monotonic() - started
        if sys.platform == "linux":
            import resource
            status["worker_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(worker_output / "status.json", status)
    def guard():
        if time.monotonic() >= deadline or Path(plan["stop_path"]).exists():
            raise ConversionJobError("work deadline or owner STOP")
    def alarm(_signum, _frame):
        raise ConversionJobError("physical worker deadline")
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
    persist()
    try:
        guard(); status["runtime"] = runtime(config); persist()
        import resource
        # A file-size cap complements the outer aggregate monitor. No virtual
        # address-space limit is used as a substitute for the memory cgroup.
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        scheduler = pinned_record(admission["scheduler"])
        if scheduler != {"kind": "primary_conversion_scheduler_v1", "status": "verified", "fields": {
                "JobId": os.environ["SLURM_JOB_ID"], "Account": "mila", "QOS": "normal", "Partition": "main-cpu",
                "NumCPUs": "1", "NumNodes": "1", "NumTasks": "1", "TimeLimit": "00:10:00", "Requeue": "0",
                "ReqTRES": {"cpu": "1", "mem": "2147483648", "node": "1"}, "AllocTRES": {"cpu": "1", "mem": "2147483648", "node": "1"}}}:
            raise ConversionJobError("supervisor-bound actual scheduler receipt required")
        helpers = runpy.run_path(str(root / config["sources"]["ownership_helper"]["path"]), run_name="_conversion_ownership")
        if not helpers["same_process"](admission["supervisor_identity"], helpers["process_identity"](os.getppid())):
            raise ConversionJobError("live owned supervisor parent/start required")
        helpers["install_spawn_journal"](output / "owned-processes.json")
        from representation_collapse.reporting import ExperimentReporter
        compact_config, identities = reporter_metadata(plan, config)
        with ExperimentReporter(worker_output / "reporting", project="representation-collapse", mode="offline",
                                config=compact_config, identities=identities) as report:
            guard()
            # The first primary-file stat/read/hash/parse occurs after SDK admission.
            if regular(config["raw"]["path"]).st_size != config["raw"]["bytes"]:
                raise ConversionJobError("primary size mismatch")
            converter = importlib.import_module("representation_collapse.graph_data")
            if Path(converter.__file__).absolute() != root / config["sources"]["converter"]["path"] or sha256(converter.__file__) != config["sources"]["converter"]["sha256"]:
                raise ConversionJobError("actual converter source mismatch")
            status.update(state="converting", conversion_attempts=1); persist()
            report.log({"conversion/attempts": 1, "conversion/started": True}, step=0)
            manifest_path = converter.convert_primary_csv(config["raw"]["path"], worker_output / "converted",
                expected_raw_sha256=config["raw"]["sha256"], expected_raw_bytes=config["raw"]["bytes"],
                config_path=root / config["reconstruction_config"]["path"], converter_revision=plan["source_commit"], guard=guard)
            guard()
            files = inventory(worker_output / "converted", max_files=4, max_bytes=config["limits"]["conversion_bytes"], guard=guard, hashes=True)
            if {item["path"] for item in files["files"]} != {"manifest.json", "train.json", "validation.json", "test.json"}:
                raise ConversionJobError("exact completed output closure required")
            manifest_sha = sha256(manifest_path)
            verified = converter.read_manifest(manifest_path, manifest_sha, guard)
            metadata = verified.metadata
            if (metadata["kind"] != "tgrab_primary_converted_v1" or metadata["converter"] != {"revision": plan["source_commit"], "sha256": config["sources"]["converter"]["sha256"]}
                    or metadata["counts"]["raw_rows"] > config["limits"]["raw_rows"] or len(metadata["snapshot_semantic_sha256"]) != 3072):
                raise ConversionJobError("completed primary manifest identity/count mismatch")
            atomic_json(worker_output / "conversion-receipt.json", {"kind": "primary_conversion_receipt_v1", "manifest_sha256": manifest_sha,
                "files": files, "counts": metadata["counts"], "identities": identities, "scientific_acceptance": False})
            report.log({"conversion/completed": True, "conversion/rows": metadata["counts"]["raw_rows"],
                        "conversion/output_bytes": files["bytes"], "conversion/timestamps": 3072}, step=1)
            guard()
        guard(); status.update(state="completed", manifest_sha256=manifest_sha); persist(); return 0
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__); persist(); return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def supervise(args, plan, config, root, output):
    started = PROCESS_STARTED; deadline = started + 600
    if sys.platform != "linux" or not hasattr(os, "sched_setaffinity"):
        raise ConversionJobError("Linux physical CPU supervision required")
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise ConversionJobError("CPU affinity unavailable")
    os.sched_setaffinity(0, {min(allowed)})
    environment = runtime(config)
    scheduler = scheduler_receipt(os.environ["SLURM_JOB_ID"])
    output.mkdir(exist_ok=False)
    helpers = runpy.run_path(str(root / config["sources"]["ownership_helper"]["path"]), run_name="_conversion_ownership")
    receipt = {"kind": "primary_conversion_supervisor_v1", "state": "initializing", "started_monotonic": started,
               "plan_sha256": args.plan_sha256, "source_commit": plan["source_commit"], "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "runtime": environment, "scientific_acceptance": False,
               "actual_supervisor_argv": [config["python"], "-B", str(root / MODULE_PATH), *sys.argv[1:]]}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "supervisor-receipt.json", receipt)
    persist()
    atomic_json(output / "scheduler-receipt.json", scheduler)
    for source, name in ((args.plan, "plan.json"), (root / CONFIG_PATH, "config.json")):
        with (output / name).open("xb") as stream:
            stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    command = [config["python"], "-B", str(root / MODULE_PATH), "--plan", str(args.plan.absolute()), "--plan-sha256", args.plan_sha256,
               "--worker", "--worker-admission", str(output / "worker-admission.json")]
    admission = {"plan_sha256": args.plan_sha256, "started_monotonic": started,
                 "slurm_job_id": os.environ["SLURM_JOB_ID"], "supervisor_identity": helpers["process_identity"](os.getpid()),
                 "scheduler": {"path": str(output / "scheduler-receipt.json"), "sha256": sha256(output / "scheduler-receipt.json")}}
    # Admission hash is an argv pin, excluded from the argv value it authenticates.
    admission["worker_argv"] = command
    atomic_json(output / "worker-admission.json", admission)
    command += ["--worker-admission-sha256", sha256(output / "worker-admission.json")]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "GOMAXPROCS", "OMP_THREAD_LIMIT"):
        env[key] = "1"
    env.update(PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="offline")
    for key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
        env[key] = str(output / "reporting-support" / key.lower())
    child, owned = None, None
    try:
        if time.monotonic() >= started + 590 or Path(plan["stop_path"]).exists():
            raise ConversionJobError("deadline or STOP before worker spawn")
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
            owned = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
            receipt.update(state="running", worker_pid=child.pid); persist()
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 1 or Path(plan["stop_path"]).exists():
                    raise ConversionJobError("outer deadline or owner STOP")
                def outer_guard():
                    if time.monotonic() >= deadline - 1 or Path(plan["stop_path"]).exists():
                        raise ConversionJobError("outer inventory deadline or owner STOP")
                inventory(output, max_files=config["limits"]["evidence_files"] + 4,
                          max_bytes=config["limits"]["evidence_bytes"] + config["limits"]["conversion_bytes"], guard=outer_guard, sdk_metadata=True, live=True)
                owned.refresh(timeout=min(0.5, remaining - 0.5))
                try:
                    child.wait(timeout=min(0.1, remaining - 1))
                except subprocess.TimeoutExpired:
                    pass
        receipt.update(state="completed" if child.returncode == 0 else "failed", worker_exit_code=child.returncode)
        if child.returncode == 0:
            status = read_json(output / "worker/status.json")
            reporting = read_json(output / "worker/reporting/reporting-receipt.json")
            regular(output / "worker/conversion-receipt.json")
            if (status.get("state") != "completed" or status.get("conversion_attempts") != 1
                    or reporting.get("state") != "completed" or reporting.get("finish_returned") is not True
                    or reporting.get("mode") != "offline" or reporting.get("last_completed_step") != 1):
                raise ConversionJobError("completed conversion and durable offline reporting required")
    except BaseException as error:
        receipt.update(state="incomplete", error_type=type(error).__name__)
    finally:
        if child is not None:
            if owned is None:
                owned = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
            errors = owned.cleanup(deadline=deadline)
            receipt.update(cleanup_errors=errors, owned_processes=list(owned.known.values()),
                           cleanup_signals=owned.signals, cleanup_reconciliation=owned.cleanup_reconciliation)
            if errors:
                receipt["state"] = "incomplete"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    receipt["state"] = "incomplete"
        if time.monotonic() >= deadline:
            receipt["state"] = "incomplete"
        persist()
        try:
            def guard():
                if time.monotonic() >= deadline:
                    raise ConversionJobError("outer inventory deadline")
            records = inventory(output, max_files=config["limits"]["evidence_files"] + 4,
                                max_bytes=config["limits"]["evidence_bytes"] + config["limits"]["conversion_bytes"], guard=guard, hashes=True, sdk_metadata=True)
            atomic_json(output / "evidence-inventory.json", records)
        except BaseException as error:
            receipt.update(state="incomplete", inventory_error_type=type(error).__name__); persist()
    return 0 if receipt["state"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-admission", type=Path)
    parser.add_argument("--worker-admission-sha256")
    args = parser.parse_args(argv)
    plan, config, root, output = verify_plan(args)
    if args.worker:
        if args.worker_admission is None or args.worker_admission_sha256 is None:
            raise ConversionJobError("supervised worker admission required")
        return worker(args, plan, config, root, output)
    return supervise(args, plan, config, root, output)


if __name__ == "__main__":
    raise SystemExit(main())
