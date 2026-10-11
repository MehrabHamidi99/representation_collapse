#!/usr/bin/env python3
"""Submit one authorized initial-phase cell through native cluv, once.

The default only previews. --execute requires the owner's existing envelope;
this program grants no compute authorization and never retries a submission.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib.metadata import version
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from cluv.cache import save_job
from cluv.cli.login import get_remote_without_2fa_prompt
from cluv.cli.submit import get_submissions, submit_job, display_commands
from cluv.config import find_pyproject


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cell", type=int, choices=range(5), required=True,
                        help="0 calibration;1 GRU cat;2 LSTM cat;3 GRU continuous;4 LSTM continuous")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if version("cluster-uv") != "0.1.5" or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("require pinned cluv 0.1.5 and a full source SHA")
    lock_file = None
    if args.execute:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        lock_path = args.journal.with_suffix(args.journal.suffix + ".lock")
        lock_file = os.fdopen(os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600), "a")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another process owns this phase journal; refusing concurrent submission")
        # Keep the descriptor alive for the complete async command. Process exit
        # releases it even if submission or transport fails.
    project = find_pyproject().parent.resolve()
    plan = json.loads(subprocess.check_output([
        "git", "show", f"{args.source_commit}:configs/experiments/initial-phase-v1.json"],
        cwd=project, text=True))
    if args.journal.exists():
        journal = json.loads(args.journal.read_text())
        if journal["source_commit"] != args.source_commit:
            parser.error("existing journal has a different immutable source")
    else:
        journal = {"schema_version": 1, "phase_id": plan["phase_id"],
                   "source_commit": args.source_commit, "cells": {}}
    key = str(args.cell)
    if key in journal["cells"]:
        # Even a prepared record without a job ID can mean an accepted sbatch.
        # Reconcile its native log and scheduler; never resubmit automatically.
        print(json.dumps({"existing_cell": journal["cells"][key], "resubmitted": False}))
        return
    if args.cell == 0:
        cell = {"kind": "calibration", "cluster": plan["calibration"]["preferred_cluster"]}
        resources = ["--account=def-ioannism_cpu", "--partition=cpubase_bycore_b1",
                     "--qos=normal", "--cpus-per-task=1", "--mem=2G", "--time=00:10:00"]
        suffix = "calibration"
    else:
        choice = plan["pilots"][args.cell - 1]
        cell = {"kind": "pilot", "cluster": choice["preferred_cluster"],
                "model": choice["model"], "task": choice["task"], "seed": choice["seed"]}
        gpu = plan["gpu_resources"]
        resources = ["--account=" + gpu["account"], "--partition=" + gpu["partition"],
                     "--qos=" + gpu["qos"], "--cpus-per-task=4", "--mem=16G",
                     "--time=01:00:00", "--gres=" + gpu["typed_gres"][cell["cluster"]]]
        suffix = f"{cell['model']}-{cell['task']}-s0"
        if args.cell in (3, 4):
            predecessor = journal["cells"].get(str(args.cell - 2), {})
            if not predecessor.get("job_id") or predecessor.get("cluster") != cell["cluster"]:
                parser.error("continuous cell requires its own cluster's submitted categorical predecessor")
            resources.append(f"--dependency=afterany:{int(predecessor['job_id'])}")
    resources += ["--nodes=1", "--ntasks=1", "--no-requeue", "--job-name=rc-p1-" + suffix]
    run_id = "p1-20261011-" + suffix
    program = ["--source-commit", args.source_commit, "--run-id", run_id,
               "--cluster", cell["cluster"], "--kind", cell["kind"]]
    if args.cell:
        program += ["--model", cell["model"], "--task", cell["task"], "--seed", "0"]
    remote = await get_remote_without_2fa_prompt(cell["cluster"]) if args.execute else None
    if args.execute and remote is None:
        parser.error("no reusable master; refusing interactive authentication")
    submissions = await get_submissions(cell["cluster"], remote,
        job_script=Path("code/ops/phase1_job.sh"), sbatch_args=resources,
        program_args=program, chunking=None, git_commit=args.source_commit, vram=None)
    if len(submissions) != 1:
        parser.error("initial phase requires exactly one native cluv submission")
    submission = submissions[0]
    cell.update(run_id=run_id, resources=submission.sbatch_args,
                command=submission.sbatch_command, state="prepared",
                prepared_at_utc=datetime.now(timezone.utc).isoformat())
    if not args.execute:
        print(json.dumps(cell, indent=2))
        return
    # Validate staged code before recording a submission attempt.
    check = await remote.run(shlex.join(("bash", "-lc",
        "cd /scratch/mehrab/representation-collapse/source && "
        "test \"$(git rev-parse HEAD)\" = " + args.source_commit +
        " && test -z \"$(git status --porcelain)\"")),
        display=False, warn=True, hide=True)
    if check.returncode:
        parser.error("target must be clean at the exact immutable source")
    journal["cells"][key] = cell
    log_path = args.journal.parent / f"phase1-native-cluv-cell-{args.cell}.log"
    cell["native_submission_log"] = str(log_path)
    atomic_json(args.journal, journal)
    display_commands.set(False)
    # Actual native cluv scheduling, with no sync, racing, duplicate configs,
    # automatic retries or waiting for the scheduler to start the job.
    job = await submit_job(submission, log_path)
    cell.update(job_id=job.job_id, state="submitted",
                submitted_at=job.submitted_at.isoformat(),
                observed_submitted_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(args.journal, journal)
    save_job(job)
    print(json.dumps(cell, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
