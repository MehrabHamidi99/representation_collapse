#!/usr/bin/env python3
"""Instantiate the job-ID field in one coordinator-frozen calibration plan.

Stdlib-only; does not initialize SDKs, import numerical packages, install
dependencies, submit jobs or change the calibration configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('coordinator plan identity mismatch')
    plan = json.loads(args.plan.read_text())
    if (plan['kind'] != 'native_graph_calibration_single_submission_plan'
            or plan['max_submissions'] != 1 or plan['cluster'] != 'mila'
            or plan['retries'] != 0 or plan['gpus'] != 0):
        raise ValueError('single approved CPU calibration required')
    root = Path(plan['source_root']).resolve()
    if Path.cwd().resolve() != root or Path(__file__).resolve() != root / 'code/ops/native_graph_calibration_job.py':
        raise ValueError('isolated published source root required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=2).strip()
    admission = dict(plan['admission_template'])
    if admission['source_commit'] != head:
        raise ValueError('published source commit mismatch')
    for relative, expected in plan['job_source_files'].items():
        if digest(root / relative) != expected:
            raise ValueError('job wrapper source identity mismatch')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '1' or not os.environ.get('SLURM_JOB_ID', '').isdigit():
        raise ValueError('actual one-CPU Slurm allocation required')
    admission['slurm_job_id'] = os.environ['SLURM_JOB_ID']
    admission['submission_plan_sha256'] = args.plan_sha256
    admission_path = Path(plan['admission_path'])
    if not admission_path.is_absolute() or admission_path.exists() or not admission_path.parent.is_dir():
        raise ValueError('fresh absolute admission path required')
    with admission_path.open('x') as stream:
        json.dump(admission, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    command = admission['supervisor_argv']
    expected = [plan['python'], '-m', 'representation_collapse.native_graph_calibration',
                '--root', str(root), '--config', str(root / 'configs/graph/native-graph-calibration-v1.json'),
                '--source-root', admission['source_root'], '--admission', str(admission_path),
                '--output', admission['output']]
    if command != expected or not Path(command[0]).is_file():
        raise ValueError('exact isolated environment supervisor argv required')
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='offline')
    os.execve(command[0], command, environment)


if __name__ == '__main__':
    main()
