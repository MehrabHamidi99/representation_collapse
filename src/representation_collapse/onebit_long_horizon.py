"""Frozen-checkpoint long-horizon extension of the accepted one-bit discovery."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

import numpy as np

from .onebit_memory import atomic, evaluate, guard, make_model, model_digest, schedules
from .reporting import ExperimentReporter, file_sha256

PRIOR_RESULT_SHA256 = 'ab36f689f337d57ebfe165a523c772c4a6dd7404155781dadf3b5e92270886ba'
CHECKPOINT_SHA256 = '1af080a2cef09b68d627ef4389f2b2a1d03983098b10f8e08d38063d2e281c9e'
PRIOR_WORKER_SHA256 = '92d855488b675f82d318e646f9811a0d65d36aa0786f555c5f8cc0c592d50535'
GRID = ((64, 1), (128, 1), (256, 1), (512, 1), (256, 2), (128, 4))
FORCING_DRAWS = 512


def run(args):
    import json
    import torch

    prior = Path(args.prior)
    checkpoint = prior / 'checkpoint.pt'
    prior_result = prior / 'result.json'
    if file_sha256(checkpoint) != CHECKPOINT_SHA256 or file_sha256(prior_result) != PRIOR_RESULT_SHA256:
        raise RuntimeError('accepted checkpoint/result hash mismatch')
    if file_sha256(Path(__file__).with_name('onebit_memory.py')) != PRIOR_WORKER_SHA256:
        raise RuntimeError('native model/evaluator source changed')
    accepted = json.loads(prior_result.read_text())
    if accepted['schema'] != 'onebit-memory-discovery-v1' or accepted['status'] != 'competence_passed' or accepted['updates'] != 5000:
        raise RuntimeError('prior competence or update identity mismatch')
    if accepted['checkpoint_sha256'] != CHECKPOINT_SHA256 or accepted['source_sha256'] != PRIOR_WORKER_SHA256:
        raise RuntimeError('prior result does not bind model/source')
    pre = json.loads((prior / 'pre-evaluation-checkpoint.json').read_text())
    if not pre['eligible'] or pre['checkpoint_sha256'] != CHECKPOINT_SHA256 or pre['model_sha256'] != accepted['final_model_sha256']:
        raise RuntimeError('prior frozen checkpoint identity mismatch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = make_model(torch)
    payload = torch.load(checkpoint, map_location='cpu')
    if payload['update'] != 5000:
        raise RuntimeError('checkpoint update mismatch')
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    frozen_model_sha256 = model_digest(model)
    if frozen_model_sha256 != accepted['final_model_sha256']:
        raise RuntimeError('loaded parameter digest mismatch')
    guard(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    worker_sha256 = file_sha256(Path(__file__))
    protocol_sha256 = file_sha256(args.protocol)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    atomic(out / 'identity.json', {
        'source_commit': source_commit, 'worker_sha256': worker_sha256,
        'protocol_sha256': protocol_sha256, 'prior_result_sha256': PRIOR_RESULT_SHA256,
        'checkpoint_sha256': CHECKPOINT_SHA256, 'model_sha256': frozen_model_sha256,
        'test_rng_seed': 4000, 'torch': torch.__version__, 'numpy': np.__version__,
        'python': platform.python_version(), 'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    })
    with ExperimentReporter(out / 'reporting', project='representation-collapse', run_id=args.run_id, mode='offline',
                            config={'task': 'onebit-long-horizon', 'trained_seed': 0, 'test_rng_seed': 4000,
                                    'forcing_draws_per_condition': FORCING_DRAWS, 'condition_count': len(GRID)},
                            identities={'checkpoint_sha256': CHECKPOINT_SHA256, 'worker_sha256': worker_sha256,
                                        'protocol_sha256': protocol_sha256}) as reporter:
        rng = np.random.default_rng(4000)
        summaries = []
        with (out / 'evaluation.jsonl').open('x') as rows:
            for step, (r, q) in enumerate(GRID, 1):
                guard(args)
                condition = f'R{r}-q{q}'
                pairs = schedules(rng, FORCING_DRAWS, r, q)
                atomic(out / f'{condition}-design.json', pairs)
                summary = evaluate(model, pairs, torch, args, rows_file=rows, condition=condition, interventions=True)
                rows.flush()
                os.fsync(rows.fileno())
                atomic(out / f'{condition}-summary.json', summary)
                summaries.append(summary)
                reporter.log({'evaluation/body_steps': 12 * r * q,
                              'evaluation/periodic_accuracy': (summary['accuracy']['bit0_arm0'] + summary['accuracy']['bit1_arm0']) / 2,
                              'evaluation/control_accuracy': (summary['accuracy']['bit0_arm1'] + summary['accuracy']['bit1_arm1']) / 2,
                              'evaluation/paired_error_difference': summary['paired_error_difference']}, step=step)
        if file_sha256(checkpoint) != CHECKPOINT_SHA256 or model_digest(model) != frozen_model_sha256:
            raise RuntimeError('frozen checkpoint/model changed during extension')
        result = {'schema': 'onebit-long-horizon-v1', 'status': 'complete',
                  'source_commit': source_commit, 'worker_sha256': worker_sha256,
                  'protocol_sha256': protocol_sha256, 'prior_result_sha256': PRIOR_RESULT_SHA256,
                  'checkpoint_sha256': CHECKPOINT_SHA256, 'model_sha256': frozen_model_sha256,
                  'test_rng_seed': 4000, 'forcing_draws_per_condition': FORCING_DRAWS,
                  'evaluation': summaries}
        atomic(out / 'result.json', result)
        atomic(out / 'artifact-inventory.json', {p.name: {'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                                                  for p in sorted(out.iterdir()) if p.is_file()})
        reporter.log({'result/conditions': len(GRID), 'result/episode_rows': len(GRID) * FORCING_DRAWS * 4}, step=len(GRID) + 1)
        reporter.log_manifest('result', metadata={'conditions': len(GRID), 'payload_upload': False},
                              identities={'result_sha256': file_sha256(out / 'result.json')})
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('output', 'prior', 'protocol', 'run-id', 'stop-file'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
