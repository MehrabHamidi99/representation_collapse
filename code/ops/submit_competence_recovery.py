#!/usr/bin/env python3
"""Preview one competence canary cell; submit once only with exact owner approval.

Preview performs no SSH, model calls, cluv cache writes or filesystem mutations.
Execution uses native cluster-uv0.1.5 on the existing controller connections;
source and locked offline environments must have been staged separately.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CANARY_PATH = "configs/experiments/competence-canary-v2.json"
PROTOCOL_PATH = "configs/experiments/competence-recovery-v2.json"
WRAPPER_PATH = "code/ops/competence_job.sh"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
RESOURCES = {"max_submissions": 4, "max_concurrent_gpu_jobs": 2,
             "max_requested_gpu_hours": 2, "gpus_per_job": 1,
             "cpus_per_job": 4, "memory_gib": 16, "walltime": "00:30:00",
             "nodes": 1, "tasks": 1, "race": False,
             "automatic_retries": False, "requeue": False}
REQUIRED_SOURCE = (CANARY_PATH, PROTOCOL_PATH, WRAPPER_PATH, "uv.lock",
                   "code/ops/submit_competence_recovery.py",
                   "pyproject.toml", ".python-version",
                   "src/representation_collapse/competence_training.py",
                   "src/representation_collapse/competence.py",
                   "src/representation_collapse/competence_selection.py",
                   "src/representation_collapse/native_state.py",
                   "src/representation_collapse/diagnostics.py",
                   "src/representation_collapse/vector_task.py",
                   "src/representation_collapse/pilot.py")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(data: str | bytes) -> dict:
    def reject(value):
        raise ValueError("nonfinite JSON value: " + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    result = json.loads(data, parse_constant=reject, object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def plain_path(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute path without traversal required")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlinked path component rejected")


def atomic_json(path: Path, packet: dict) -> None:
    plain_path(path)
    temporary = path.with_name(path.name + ".tmp")
    plain_path(temporary)
    with temporary.open("x") as stream:
        json.dump(packet, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def ignored_path(path: Path) -> None:
    plain_path(path)
    if not path.is_relative_to(ROOT):
        raise ValueError("approval and journal must be inside this project")
    result = subprocess.run(["git", "check-ignore", "--quiet", "--", str(path)], cwd=ROOT)
    if result.returncode:
        raise ValueError("approval and runtime journal must be git-ignored")


def validate_plan(plan: dict, protocol: dict) -> None:
    if (plan.get("phase_id") != "competence-canary-v2" or plan.get("scientific_acceptance") is not False
            or plan.get("confirmation") is not False or plan.get("grid_selection") is not False
            or plan.get("protocol_path") != PROTOCOL_PATH or plan.get("wrapper_path") != WRAPPER_PATH
            or plan.get("trainer_module") != "representation_collapse.competence_training"
            or plan.get("resource_envelope") != RESOURCES or plan.get("learning_rate") != 0.001
            or plan.get("maximum_updates_per_fit") != 3500):
        raise ValueError("unexpected canary scope or resource envelope")
    if (protocol.get("protocol_id") != "competence-recovery-v2"
            or protocol.get("scientific_acceptance") is not False
            or protocol.get("maximum_fits") != 12 or protocol.get("maximum_updates_per_fit") != 3500
            or protocol.get("lr_grid") != [0.0005, 0.001, 0.003]
            or protocol.get("diagnostics", {}).get("fixed_batch_updates") != 100
            or protocol.get("diagnostics", {}).get("iid_immediate_updates_after_fixed_batch") != 400):
        raise ValueError("require the frozen 12-fit discovery protocol and100/400diagnostic split")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != 4:
        raise ValueError("exactly four canary cells required")
    expected = {(m, t, 0) for m in ("gru", "lstm") for t in ("categorical", "continuous")}
    if {(c.get("model"), c.get("task"), c.get("seed")) for c in cells} != expected:
        raise ValueError("canary must cover the four frozen seed-0 model/task cells")
    slots = {}
    for index, cell in enumerate(cells):
        if (type(cell.get("cell")) is not int or cell.get("cell") != index
                or type(cell.get("seed")) is not int or cell["seed"] != 0
                or type(cell.get("concurrency_slot")) is not int or cell.get("concurrency_slot") not in (0, 1)):
            raise ValueError("stable ordered cell IDs and two concurrency slots required")
        host = cell.get("cluster")
        if host not in plan.get("targets", {}) or host not in {"fir", "nibi", "narval", "rorqual"}:
            raise ValueError("require an explicitly selected, configured single cluster per cell")
        slot = cell["concurrency_slot"]
        predecessor = slots.get(slot)
        if cell.get("predecessor_cell") != predecessor:
            raise ValueError("each concurrency slot requires its registered immediate predecessor")
        if predecessor is not None and cells[predecessor]["cluster"] != host:
            raise ValueError("a dependency slot cannot change clusters")
        slots[slot] = index
    if len(slots) != 2:
        raise ValueError("exactly two dependency chains required")


def target_verified(target: dict) -> bool:
    try:
        stamp = datetime.fromisoformat(target.get("verified_at_utc") or "")
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
    except (ValueError, TypeError):
        return False
    return (0 <= age <= 86400 and target.get("scheduler_verified") is True
            and target.get("offline_environment_verified") is True
            and all(isinstance(target.get(k), str) and IDENTIFIER.fullmatch(target[k])
                    for k in ("account", "partition", "qos"))
            and isinstance(target.get("gres"), str)
            and re.fullmatch(r"gpu:[A-Za-z0-9_.-]+:1", target["gres"]) is not None
            and all(isinstance(target.get(k), str)
                    and re.fullmatch(r"/scratch/[A-Za-z0-9_./-]+", target[k]) is not None
                    and ".." not in Path(target[k]).parts and "//" not in target[k]
                    for k in ("source_directory", "scratch_root"))
            and target["source_directory"].startswith(target["scratch_root"] + "/representation-collapse/")
            and SHA64.fullmatch(target.get("verification_receipt_sha256") or "") is not None
            and SHA64.fullmatch(target.get("offline_preflight_receipt_sha256") or "") is not None)


def binding(plan: dict, source_commit: str, source_hashes: dict,
            run_prefix: str, journal: Path) -> dict:
    return {"phase_id": plan["phase_id"], "source_commit": source_commit,
            "canary_sha256": source_hashes[CANARY_PATH],
            "protocol_sha256": source_hashes[PROTOCOL_PATH],
            "resource_envelope": plan["resource_envelope"],
            "cells": plan["cells"], "targets": plan["targets"],
            "run_prefix": run_prefix, "journal_relative_path": str(journal.relative_to(ROOT))}


def validate_approval(approval: dict, expected: dict) -> dict:
    record = approval.get("competence_canary_v2")
    if not isinstance(record, dict) or record.get("approved") is not True or record.get("approved_by") != "owner":
        raise ValueError("explicit owner approval for this canary is absent")
    if record.get("approval_binding") != expected:
        raise ValueError("owner approval does not match exact source/config/targets/cells/resources")
    stamp = datetime.fromisoformat(record.get("approved_at_utc", ""))
    if stamp.utcoffset() is None or stamp > datetime.now(timezone.utc):
        raise ValueError("valid nonfuture owner-approval timestamp required")
    return {"schema_version": 1, "approved": True, "approved_by": "owner",
            "approved_at_utc": record["approved_at_utc"], "approval_binding": expected,
            "scientific_acceptance": False}


def read_source(source_commit: str) -> tuple[dict, dict, dict, list[str]]:
    hashes, missing = {}, []
    for name in REQUIRED_SOURCE:
        result = subprocess.run(["git", "show", f"{source_commit}:{name}"], cwd=ROOT, capture_output=True)
        if result.returncode:
            missing.append(name)
        else:
            hashes[name] = digest(result.stdout)
    plan_bytes = subprocess.run(["git", "show", f"{source_commit}:{CANARY_PATH}"], cwd=ROOT, capture_output=True)
    protocol_bytes = subprocess.run(["git", "show", f"{source_commit}:{PROTOCOL_PATH}"], cwd=ROOT, capture_output=True)
    # A precommit preparation preview is explicitly non-executable.
    plan = strict_json(plan_bytes.stdout if not plan_bytes.returncode else (ROOT / CANARY_PATH).read_bytes())
    protocol = strict_json(protocol_bytes.stdout if not protocol_bytes.returncode else (ROOT / PROTOCOL_PATH).read_bytes())
    validate_plan(plan, protocol)
    return plan, protocol, hashes, missing


def program_args(cell: dict, source_commit: str, run_id: str, plan: dict,
                 hashes: dict, approval_path: str, approval_sha256: str) -> list[str]:
    target = plan["targets"][cell["cluster"]]
    return ["--source-commit", source_commit, "--run-id", run_id,
            "--cluster", cell["cluster"], "--model", cell["model"], "--task", cell["task"],
            "--seed", str(cell["seed"]), "--learning-rate", "0.001", "--cell", str(cell["cell"]),
            "--config", PROTOCOL_PATH, "--config-sha256", hashes[PROTOCOL_PATH],
            "--canary-config", CANARY_PATH, "--canary-sha256", hashes[CANARY_PATH],
            "--approval-record", approval_path, "--approval-sha256", approval_sha256,
            "--account", target["account"], "--partition", target["partition"],
            "--qos", target["qos"], "--gres", target["gres"]]


# Executed by the cluster's system Python through native cluv Remote.run.
# The payload contains source/resource bindings only, never provider settings.
PREFLIGHT = r'''
import hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
p=json.loads(sys.argv[1]); root=Path(p['target']['source_directory'])
def run(args): return subprocess.check_output(args,text=True).strip()
def safe(path):
 if any(q.is_symlink() for q in (path,*path.parents)): raise RuntimeError('symlinked source/cache path')
safe(root)
if not root.is_dir() or run(['git','-C',str(root),'rev-parse','HEAD'])!=p['source_commit'] or run(['git','-C',str(root),'status','--porcelain','--untracked-files=all']): raise RuntimeError('clean pinned target source required')
if any(line.startswith('120000 ') for line in run(['git','-C',str(root),'ls-files','-s']).splitlines()): raise RuntimeError('tracked source symlinks rejected')
for name,expected in p['source_hashes'].items():
 path=root/name; safe(path)
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise RuntimeError('pinned source hash mismatch:'+name)
target=p['target']; scratch=Path(target['scratch_root']); safe(scratch)
if os.environ.get('SCRATCH')!=str(scratch): raise RuntimeError('SCRATCH differs from exact approved path')
part=run(['sinfo','--noheader','--partition='+target['partition'],'--format=%P|%a|%G'])
gputype=target['gres'].split(':')[1]
if not any('|up|' in line.lower() and ('gpu:'+gputype+':') in line for line in part.splitlines()): raise RuntimeError('approved UP partition/GPU type unavailable')
assoc=run(['sacctmgr','--noheader','--parsable2','show','assoc','where','user='+os.environ['USER'],'account='+target['account'],'format=Account,QOS'])
if not any(line.split('|')[0]==target['account'] and target['qos'] in line.split('|')[1].split(',') for line in assoc.splitlines() if '|' in line): raise RuntimeError('approved account/QOS association not verified')
for name in ('cache/uv','cache/python'):
 path=scratch/'representation-collapse'/name; safe(path)
 if not path.is_dir(): raise RuntimeError('staged scratch cache absent')
receipt=scratch/'representation-collapse'/'runs'/'ops'/('competence-canary-v2-offline-'+p['cluster']+'.json'); safe(receipt)
if not receipt.is_file() or hashlib.sha256(receipt.read_bytes()).hexdigest()!=target['offline_preflight_receipt_sha256']: raise RuntimeError('approved offline receipt missing or changed')
r=json.loads(receipt.read_text())
if r.get('offline_preflight_passed') is not True or r.get('uv_lock_sha256')!=p['source_hashes']['uv.lock'] or r.get('dependency_group')!='experiments' or r.get('gpu_package')!='torch==2.6.0': raise RuntimeError('locked experiments cache was not verified offline')
stamp=datetime.fromisoformat(r.get('checked_at_utc',''))
if stamp.utcoffset() is None or not 0<=(datetime.now(timezone.utc)-stamp).total_seconds()<=86400: raise RuntimeError('offline cache preflight receipt must be from the last24hours')
queue=run(['squeue','--noheader','--user='+os.environ['USER'],'--format=%i|%j|%T'])
matching=[]
for line in queue.splitlines():
 bits=line.split('|')
 if len(bits)==3 and bits[1].startswith(p['job_name_prefix']): matching.append({'job_id':bits[0],'name':bits[1],'state':bits[2]})
registered={str(row['job_id']) for row in p['known_jobs']}
if any(row['job_id'] not in registered for row in matching): raise RuntimeError('unregistered matching canary job; reconcile before retry')
sacct=[]
if registered:
 text=run(['sacct','--noheader','--parsable2','--jobs='+','.join(sorted(registered)),'--format=JobIDRaw,State,ExitCode,JobName%128'])
 for line in text.splitlines():
  bits=line.split('|')
  if bits[0] in registered:
   if not bits[3].startswith(p['job_name_prefix']): raise RuntimeError('known ID is not this canary job')
   sacct.append({'job_id':bits[0],'state':bits[1],'exit_code':bits[2],'name':bits[3]})
 if {r['job_id'] for r in sacct}!=registered: raise RuntimeError('known IDs not reconciled in accounting')
print(json.dumps({'scheduler_verified':True,'source_verified':True,'offline_receipt_verified':True,'queue':matching,'sacct':sacct}))
'''


STAGE_RECEIPT = """import os,sys
from pathlib import Path
p=Path(sys.argv[1]); b=sys.argv[2].encode(); tmp=p.with_name(p.name+'.tmp')
if any(q.is_symlink() for q in (p,tmp,*p.parents)) or not p.parent.is_dir(): raise RuntimeError('unsafe approval receipt path')
if p.exists():
 if p.read_bytes()!=b: raise RuntimeError('approval receipt already exists with different content')
else:
 with tmp.open('xb') as stream:
  stream.write(b); stream.flush(); os.fsync(stream.fileno())
 tmp.rename(p)
"""


def validate_embedded_python() -> list[str]:
    """Compile all generated programs locally, without executing their bodies."""
    programs = {"PREFLIGHT": PREFLIGHT, "STAGE_RECEIPT": STAGE_RECEIPT}
    wrapper = (ROOT / WRAPPER_PATH).read_text()
    bodies = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", wrapper, flags=re.DOTALL)
    if len(bodies) != 3:
        raise ValueError("wrapper must expose its three reviewed Python heredoc bodies")
    programs.update({"competence_job.sh:python_body_" + str(index + 1): body
                     for index, body in enumerate(bodies)})
    for name, body in programs.items():
        try:
            compile(body, name, "exec")
        except SyntaxError as error:
            raise ValueError("embedded Python program failed local compilation: " + name + ": " + str(error)) from error
    return list(programs)


async def execute(args, plan: dict, hashes: dict, approval: dict, preview: dict) -> dict:
    # All imports below are behind the explicit --submit and approval gate.
    from cluv.cache import save_job
    from cluv.cli.login import get_remote_without_2fa_prompt
    from cluv.cli.submit import display_commands, get_submissions, submit_job
    from cluv.config import get_cluv_config
    ignored_path(args.journal)
    if args.journal.exists():
        journal = strict_json(args.journal.read_bytes())
        if journal.get("approval_binding") != approval["approval_binding"]:
            raise ValueError("existing journal has a different immutable approval binding")
    else:
        journal = {"schema_version": 1, "approval_binding": approval["approval_binding"], "cells": {}}
    key = str(args.cell)
    if key in journal["cells"]:
        return {"existing_cell": journal["cells"][key], "resubmitted": False}
    if len(journal["cells"]) >= 4 or any(row.get("state") != "submitted" for row in journal["cells"].values()):
        raise ValueError("submission cap or ambiguous journal entry blocks all further submissions")
    cell = plan["cells"][args.cell]
    predecessor = cell["predecessor_cell"]
    dependency = None
    if predecessor is not None:
        parent = journal["cells"].get(str(predecessor), {})
        if parent.get("state") != "submitted" or not parent.get("job_id") or parent.get("cluster") != cell["cluster"]:
            raise ValueError("registered same-cluster predecessor required for concurrency bound")
        dependency = int(parent["job_id"])
    remotes, reconciliation = {}, {}
    for cluster, target in plan["targets"].items():
        if not target_verified(target):
            raise ValueError("target scheduler/source/offline-cache evidence remains unverified: " + cluster)
        remote = await get_remote_without_2fa_prompt(cluster)
        if remote is None:
            raise ValueError("no reusable native cluv master; refusing Duo/login/fallback: " + cluster)
        remotes[cluster] = remote
        payload = {"cluster": cluster, "target": target, "source_commit": args.source_commit,
                   "source_hashes": hashes, "job_name_prefix": preview["job_name_prefix"],
                   "known_jobs": [row for row in journal["cells"].values() if row["cluster"] == cluster]}
        result = await asyncio.wait_for(remote.run(shlex.join(["python3", "-c", PREFLIGHT, json.dumps(payload)]),
                                                   display=False, hide=True, warn=True), timeout=90)
        if result.returncode:
            raise ValueError("passive source/scheduler/cache/job reconciliation failed: " + cluster + ": " + result.stderr[-2000:])
        reconciliation[cluster] = strict_json(result.stdout)
    holding_states = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
    active = sum(sum(job["state"] in holding_states for job in row["queue"]) for row in reconciliation.values())
    if active > 2:
        raise ValueError("existing matching queue exceeds the two-slot bound")
    # Only the lock owner may stage the receipt, journal or actual submission.
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.journal.with_name(args.journal.name + ".lock")
    plain_path(lock_path)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another owner holds this serial submission journal") from error
        if args.journal.exists() and strict_json(args.journal.read_bytes()) != journal:
            raise ValueError("journal changed during passive checks; recompute preview, never duplicate")
        target = plan["targets"][cell["cluster"]]
        receipt_path = target["scratch_root"] + "/representation-collapse/runs/ops/" + preview["run_prefix"] + "-approval.json"
        approval_bytes = (json.dumps(approval, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        receipt_sha = digest(approval_bytes)
        # Whitelist-only receipt; no credentials or provider settings are uploaded.
        receipt_result = await asyncio.wait_for(remotes[cell["cluster"]].run(
            shlex.join(["python3", "-c", STAGE_RECEIPT, receipt_path, approval_bytes.decode()]),
            display=False, hide=True, warn=True), timeout=30)
        if receipt_result.returncode:
            raise ValueError("cannot stage exact sanitized approval receipt")
        native = get_cluv_config()
        native.clusters[cell["cluster"]] = replace(native.clusters[cell["cluster"]],
                                                   project_dir=target["source_directory"],
                                                   job_script_path=WRAPPER_PATH,
                                                   sbatch_args={},
                                                   results_path=target["scratch_root"] + "/representation-collapse/logs/cluv")
        # Remove inherited extra allocations/flags; this process owns one exact set.
        native.sbatch_args = {}
        resources = ["--account=" + target["account"], "--partition=" + target["partition"],
                     "--qos=" + target["qos"], "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
                     "--mem=16G", "--time=00:30:00", "--gres=" + target["gres"], "--no-requeue",
                     "--job-name=" + preview["job_name_prefix"].removeprefix("cluv-") + str(args.cell)]
        if dependency is not None:
            resources.append("--dependency=afterany:" + str(dependency))
        program = program_args(cell, args.source_commit, preview["run_id"], plan, hashes, receipt_path, receipt_sha)
        submissions = await get_submissions(cell["cluster"], remotes[cell["cluster"]],
            job_script=Path(WRAPPER_PATH), sbatch_args=resources, program_args=program,
            chunking=None, git_commit=args.source_commit, vram=None)
        if len(submissions) != 1 or submissions[0].n_chunks is not None:
            raise ValueError("require exactly one native submission; arrays/races excluded")
        submission = submissions[0]
        row = {"cell": args.cell, "cluster": cell["cluster"], "run_id": preview["run_id"],
               "model": cell["model"], "task": cell["task"], "seed": 0, "learning_rate": 0.001,
               "state": "submitting", "resources": submission.sbatch_args,
               "command": submission.sbatch_command, "approval_receipt_sha256": receipt_sha,
               "prepared_at_utc": now(), "reconciliation": reconciliation,
               "requested_gpu_hours": 0.5, "scientific_acceptance": False}
        journal["cells"][key] = row
        atomic_json(args.journal, journal)
        log = args.journal.parent / (preview["run_id"] + "-native-cluv.log")
        if log.exists():
            raise ValueError("submission log already exists; reconcile possible prior attempt")
        display_commands.set(False)
        try:
            job = await asyncio.wait_for(submit_job(submission, log), timeout=60)
        except BaseException:
            row.update(state="ambiguous", observed_at_utc=now(), retry_allowed=False)
            atomic_json(args.journal, journal)
            raise
        row.update(state="submitted", job_id=job.job_id, observed_submitted_at_utc=now())
        atomic_json(args.journal, journal)
        save_job(job)
        return row


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit")
    parser.add_argument("--cell", type=int, choices=range(4), required=True)
    parser.add_argument("--run-prefix", default="competence-canary-v2-20261012")
    parser.add_argument("--approval-record", type=Path, default=ROOT / "configs/approval.local.json")
    parser.add_argument("--journal", type=Path, default=ROOT / "runs/ops/competence-canary-v2-20261012.json")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    try:
        compiled_programs = validate_embedded_python()
        if version("cluster-uv") != "0.1.5":
            raise ValueError("native cluster-uv0.1.5 required")
        if not IDENTIFIER.fullmatch(args.run_prefix) or len(args.run_prefix) > 60:
            raise ValueError("run prefix requires at most60safe identifier characters")
        args.source_commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if not SHA40.fullmatch(args.source_commit):
            raise ValueError("full lowercase source SHA required")
        plan, protocol, hashes, missing = read_source(args.source_commit)
        controller_clean = not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
        cell = plan["cells"][args.cell]
        target = plan["targets"][cell["cluster"]]
        run_id = args.run_prefix + "-" + str(args.cell) + "-" + cell["model"] + "-" + cell["task"] + "-s0-lr001"
        if not IDENTIFIER.fullmatch(run_id):
            raise ValueError("derived unique run ID exceeds safe length")
        preview = {"kind": "competence_canary_preview", "remote_actions": 0, "mutations": 0,
                   "embedded_python_compilation_passed": compiled_programs,
                   "phase_id": plan["phase_id"], "source_commit": args.source_commit,
                   "source_sha256": hashes, "missing_committed_source": missing,
                   "source_ready": not missing, "target_verified": target_verified(target),
                   "controller_clean": controller_clean,
                   "approval_required": True, "approval_present": False, "scientific_acceptance": False,
                   "cell": cell, "run_prefix": args.run_prefix, "run_id": run_id,
                   "job_name_prefix": "cluv-rc-comp-v2-" + digest(args.run_prefix.encode())[:10] + "-",
                   "resources": RESOURCES, "target": target, "learning_rate": 0.001,
                   "maximum_updates": 3500, "requested_gpu_hours_per_job": 0.5,
                   "native_api": "cluv.cli.submit.get_submissions/submit_job; no sync, no race",
                   "outputs": (target.get("scratch_root") or "<verified SCRATCH>") + "/representation-collapse/logs/cluv/" + cell["cluster"] + "_<JOB_ID>/competence"}
        preview["command_after_approval"] = shlex.join(["uv", "run", "--locked", "--group", "ops", "python",
            "code/ops/submit_competence_recovery.py", "--source-commit", args.source_commit,
            "--cell", str(args.cell), "--run-prefix", args.run_prefix, "--submit"])
        if not missing:
            plain_path(args.journal)
            if not args.journal.is_relative_to(ROOT):
                raise ValueError("project-relative canonical journal required")
            preview["approval_binding"] = binding(plan, args.source_commit, hashes, args.run_prefix, args.journal)
        if not args.submit:
            print(json.dumps(preview, indent=2, sort_keys=True, allow_nan=False)); return
        if missing:
            raise ValueError("new trainer/wrapper/config must be committed before execution")
        if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != args.source_commit:
            raise ValueError("controller HEAD must equal exact approved source")
        if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip():
            raise ValueError("controller source must be clean")
        ignored_path(args.approval_record)
        approval = validate_approval(strict_json(args.approval_record.read_bytes()), preview["approval_binding"])
        result = await execute(args, plan, hashes, approval, preview)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError, asyncio.TimeoutError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    asyncio.run(main())
