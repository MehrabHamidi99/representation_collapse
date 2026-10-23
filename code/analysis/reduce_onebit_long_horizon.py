"""Independent compact reduction of one-bit long-horizon evaluation rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reduce_onebit_memory import compare_summary, designs, read, reduce_rows, require, rows, sha

GRID = ((64, 1), (128, 1), (256, 1), (512, 1), (256, 2), (128, 4))
N = 512


def audit(root: Path, result_sha256: str) -> dict:
    require(sha(root / 'result.json') == result_sha256, 'result hash')
    result = read(root / 'result.json')
    identity = read(root / 'identity.json')
    inventory = read(root / 'artifact-inventory.json')
    require(result['schema'] == 'onebit-long-horizon-v1' and result['status'] == 'complete', 'result schema/status')
    for key in ('source_commit', 'worker_sha256', 'protocol_sha256', 'prior_result_sha256',
                'checkpoint_sha256', 'model_sha256', 'test_rng_seed'):
        require(result[key] == identity[key], 'result/identity ' + key)
    require(result['test_rng_seed'] == 4000 and result['forcing_draws_per_condition'] == N,
            'test design identity')
    require(result['prior_result_sha256'] == 'ab36f689f337d57ebfe165a523c772c4a6dd7404155781dadf3b5e92270886ba'
            and result['checkpoint_sha256'] == '1af080a2cef09b68d627ef4389f2b2a1d03983098b10f8e08d38063d2e281c9e',
            'accepted parent mismatch')
    require({'result.json', 'identity.json', 'evaluation.jsonl'} <= set(inventory), 'required files')
    checked = 0
    uncollected = []
    for name, entry in inventory.items():
        require(Path(name).name == name, 'inventory path')
        file = root / name
        if not file.exists():
            require(name.endswith('-states.npz'), 'missing compact file ' + name)
            uncollected.append(name)
            continue
        require(file.stat().st_size == entry['bytes'] and sha(file) == entry['sha256'],
                'inventory mismatch ' + name)
        checked += 1
    require(inventory['result.json']['sha256'] == result_sha256, 'inventory result hash')
    data = rows(root / 'evaluation.jsonl')
    require(len(data) == len(GRID) * N * 4 and len(result['evaluation']) == len(GRID),
            'incomplete held-out rows/conditions')
    conditions = []
    for j, (r, q) in enumerate(GRID):
        name = f'R{r}-q{q}'
        require({f'{name}-design.json', f'{name}-summary.json', f'{name}-states.npz'} <= set(inventory),
                'condition inventory')
        designs(read(root / f'{name}-design.json'), N, r, q)
        item = reduce_rows(data[j*N*4:(j+1)*N*4], N, name, True)
        require(result['evaluation'][j] == read(root / f'{name}-summary.json'),
                'summary/result mismatch')
        compare_summary(result['evaluation'][j], item, True)
        item.update(R=r, q=q, body_steps=12*r*q)
        conditions.append(item)
    contrast = {x['condition']: x['paired_error_difference'] for x in conditions}
    return {'status': 'verified_compact', 'conditions': conditions,
            'contrasts': {'delta512': contrast['R512-q1'], 'delta64': contrast['R64-q1'],
                          'delta512_minus_delta64': contrast['R512-q1'] - contrast['R64-q1']},
            'episode_rows_verified': len(data), 'forcing_draws_per_condition': N,
            'compact_files_verified': checked, 'uncollected_state_payloads': uncollected,
            'input_result_sha256': result_sha256,
            'limitations': ['one trained seed and one stored bit',
                            'state arrays/model not independently reexecuted',
                            'scheduler and actual W&B audit separate',
                            'no capacity, erasure or Floquet conclusion']}


def plot(data: dict, output: Path) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    q1 = [x for x in data['conditions'] if x['q'] == 1]
    same_length = sorted([x for x in data['conditions'] if x['body_steps'] == 6144], key=lambda x: x['q'])
    for arm, label in [('0', 'Periodic'), ('1', 'Matched order control')]:
        axes[0].plot([x['R'] for x in q1], [x['arm_accuracy'][arm] for x in q1], 'o-', label=label)
        axes[1].plot([x['q'] for x in same_length], [x['arm_accuracy'][arm] for x in same_length], 'o-', label=label)
    axes[0].set(xlabel='Motif repetitions R (q=1)', ylabel='Native bit accuracy',
                title='Longer retention horizon', xticks=[64,128,256,512], xlim=(45,535))
    axes[1].set(xlabel='Symbol dwell q', ylabel='Native bit accuracy',
                title='Equal body length: 6144 updates', xticks=[1,2,4])
    for ax in axes:
        ax.set_ylim(-.02, 1.02)
        ax.axhline(.5, color='gray', ls=':')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Frozen one-bit checkpoint · held-out long horizons · discovery seed 0', fontsize=10)
    path = output / 'onebit-long-horizon.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return sha(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--result-sha256', required=True)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from representation_collapse.reporting import ExperimentReporter
    with ExperimentReporter(args.output / 'reporting', project='representation-collapse', mode='offline',
                            run_id=args.run_id, config={'task': 'onebit-long-horizon-reduction'},
                            identities={'result_sha256': args.result_sha256}) as reporter:
        reduced = audit(args.input, args.result_sha256)
        (args.output / 'results.json').write_text(json.dumps(reduced, indent=2, allow_nan=False) + '\n')
        plot_sha256 = plot(reduced, args.output)
        (args.output / 'plot-manifest.json').write_text(json.dumps({'onebit-long-horizon.png': plot_sha256}, indent=2) + '\n')
        reporter.log({'verification/episode_rows': reduced['episode_rows_verified'],
                      'verification/conditions': len(reduced['conditions'])}, step=0)
        reporter.log_manifest('onebit-long-horizon-reduction', metadata={'payload_upload': False},
                              identities={'results_sha256': sha(args.output / 'results.json')})


if __name__ == '__main__':
    main()
