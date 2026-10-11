#!/usr/bin/env python3
"""Submit one owner-authorized checkpoint analysis through native cluv, once."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from cluv.cache import save_job
from cluv.cli.login import get_remote_without_2fa_prompt
from cluv.cli.submit import display_commands, get_submissions, submit_job
from cluv.config import find_pyproject, get_cluv_config
from cluv_controller import configure_ssh


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--cell', type=int, choices=range(4), required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if version('cluster-uv') != '0.1.5' or not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        parser.error('require cluv0.1.5 and a full source SHA')
    lock_file = None
    if args.execute:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        lock_file = os.fdopen(os.open(args.journal.with_suffix('.json.lock'), os.O_WRONLY | os.O_CREAT, 0o600), 'a')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another process owns this journal; refusing concurrent submission')
    project = find_pyproject().parent.resolve()
    plan = json.loads(subprocess.check_output(['git', 'show', args.source_commit + ':configs/experiments/periodicity-analysis-v1.json'], cwd=project, text=True))
    if plan['phase_id'] != 'periodicity-analysis-v1' or plan['retraining'] or plan['scientific_acceptance']:
        parser.error('require the frozen checkpoint-only diagnostic')
    if plan['resource_envelope'] != {'max_submissions': 4, 'max_requested_gpu_hours': 1, 'max_concurrent_gpu_jobs': 2, 'race': False, 'automatic_retries': False}:
        parser.error('unexpected analysis resource envelope')
    if args.journal.exists():
        journal = json.loads(args.journal.read_text())
        if journal['source_commit'] != args.source_commit or journal['phase_id'] != plan['phase_id']:
            parser.error('existing journal has a different phase or immutable source')
    else:
        journal = {'schema_version': 1, 'phase_id': plan['phase_id'], 'source_commit': args.source_commit, 'cells': {}}
    key = str(args.cell)
    if key in journal['cells']:
        print(json.dumps({'existing_cell': journal['cells'][key], 'resubmitted': False}))
        return
    if len(journal['cells']) >= 4:
        parser.error('four-cell submission cap exhausted')
    cell = dict(plan['cells'][args.cell])
    if cell['cell'] != args.cell or cell['seed'] != 0 or cell['cluster'] not in {'narval', 'rorqual'}:
        parser.error('unexpected frozen cell')
    gpu = plan['cluster_resources']
    if (gpu['gpus_per_job'], gpu['cpus_per_job'], gpu['memory_gib'], gpu['walltime_minutes']) != (1, 4, 16, 15):
        parser.error('unexpected resource counts')
    resources = ['--account=' + gpu['account'], '--partition=' + gpu['partition'], '--qos=' + gpu['qos'],
                 '--nodes=1', '--ntasks=1', '--cpus-per-task=4', '--mem=16G', '--time=00:15:00',
                 '--gres=' + gpu['typed_gres'][cell['cluster']], '--no-requeue',
                 '--job-name=rc-per-v1-' + cell['model'] + '-' + cell['task']]
    if args.cell in (2, 3):
        predecessor = journal['cells'].get(str(args.cell - 2), {})
        if not predecessor.get('job_id') or predecessor.get('cluster') != cell['cluster']:
            parser.error('successor requires its registered predecessor on the same cluster')
        resources.append('--dependency=afterany:' + str(int(predecessor['job_id'])))
    run_id = 'per-v1-20261011-' + cell['model'] + '-' + cell['task'] + '-s0'
    program = ['--source-commit', args.source_commit, '--run-id', run_id, '--cluster', cell['cluster'],
               '--model', cell['model'], '--task', cell['task'], '--seed', '0',
               '--checkpoint', cell['checkpoint'], '--checkpoint-sha256', cell['checkpoint_sha256']]
    if args.execute:
        configure_ssh(batch=True)
    remote = await get_remote_without_2fa_prompt(cell['cluster']) if args.execute else None
    if args.execute and remote is None:
        parser.error('no reusable master; refusing interactive authentication')
    # cluv0.1.5 prefers the configured script over an explicit CLI script.
    # Override only this process's cached config; preserve the disk default.
    config = get_cluv_config()
    config.job_script_path = 'code/ops/periodicity_job.sh'
    if config.get_cluster_config(cell['cluster']).job_script_path.as_posix() != config.job_script_path:
        parser.error('unexpected cluster-specific wrapper override')
    submissions = await get_submissions(cell['cluster'], remote, job_script=Path(config.job_script_path),
        sbatch_args=resources, program_args=program, chunking=None, git_commit=args.source_commit, vram=None)
    if len(submissions) != 1:
        parser.error('require exactly one native submission; racing is excluded')
    submission = submissions[0]
    cell.update(run_id=run_id, resources=submission.sbatch_args, command=submission.sbatch_command,
                state='prepared', prepared_at_utc=datetime.now(timezone.utc).isoformat())
    if not args.execute:
        print(json.dumps(cell, indent=2))
        return
    check_command = 'cd /scratch/mehrab/representation-collapse/source && test "$(git rev-parse HEAD)" = ' + args.source_commit + ' && test -z "$(git status --porcelain)" && test "$(sha256sum ' + shlex.quote(cell['checkpoint']) + ' | cut -d\' \'' + ' -f1)" = ' + cell['checkpoint_sha256']
    check = await remote.run(shlex.join(['bash', '-lc', check_command]), display=False, warn=True, hide=True)
    if check.returncode:
        parser.error('target source must be clean and checkpoint hash must match')
    journal['cells'][key] = cell
    log_path = args.journal.parent / ('periodicity-native-cluv-cell-' + key + '.log')
    cell['native_submission_log'] = str(log_path)
    atomic_json(args.journal, journal)
    display_commands.set(False)
    job = await submit_job(submission, log_path)
    cell.update(job_id=job.job_id, state='submitted', submitted_at=job.submitted_at.isoformat(),
                observed_submitted_at_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(args.journal, journal)
    save_job(job)
    print(json.dumps(cell, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
