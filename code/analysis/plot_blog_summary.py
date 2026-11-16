"""Replot verified compact results for the public research note; no new experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph_path = ROOT / 'doc/auto/figures/repetition-dwell-2026-10-20/results.json'
    memory_path = ROOT / 'doc/auto/figures/onebit-long-horizon-2026-10-23/reduction-results.json'
    graph = json.loads(graph_path.read_text())
    memory = json.loads(memory_path.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#263c34', 'text.color': '#263c34',
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    lengths = [8,16,32,64,128,256]
    means = []; controls = []
    for length in lengths:
        rows = [r for r in graph['conditions'] if r['dwell'] == length]
        assert len(rows) == 2 and {r['start'] for r in rows} == {'A','B'}
        means.append(sum(r['mean_f1'] for r in rows)/2)
        controls.append(sum(r['persistence_mean_f1_excluding_startup'] for r in rows)/2)
    fig, ax = plt.subplots(figsize=(7.4,4.5), layout='constrained')
    ax.axvline(32, color='#cbd6ce', linewidth=1.5)
    ax.plot(lengths, controls, 's--', color='#89958b', label='Repeat the last graph (persistence)', markersize=6)
    ax.plot(lengths, means, 'o-', color='#b85d3a', label='Trained GC-LSTM', linewidth=2.5, markersize=7)
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(lengths)); ax.xaxis.set_major_formatter(FixedFormatter(list(map(str,lengths))))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlabel='Consecutive repetitions of each graph', ylabel='Mean prediction F1',
           title='Longer graph blocks hurt this checkpoint', ylim=(0,1.07))
    ax.text(32,.36,'Trained at 32',ha='center',color='#53685d',fontsize=11)
    ax.legend(loc='lower left',frameon=False,fontsize=10)
    ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output/'repetition-graph.png',dpi=200); plt.close(fig)

    rows = [r for r in memory['conditions'] if r['q']==1]
    assert [r['body_steps'] for r in rows] == [768,1536,3072,6144]
    assert all(r['arm_accuracy']['0']==r['arm_accuracy']['1']==1 for r in memory['conditions'])
    fig, ax = plt.subplots(figsize=(7.4,3.8), layout='constrained')
    x=[r['body_steps'] for r in rows]
    ax.plot(x,[r['arm_accuracy']['0'] for r in rows], 'o-',color='#315d49',linewidth=2.5,markersize=7,
            label='Periodic and matched control (overlap)')
    ax.axhline(.5,color='#89958b',ls=':',label='Balanced constant guess')
    ax.set_xscale('log',base=2)
    ax.xaxis.set_major_locator(FixedLocator(x)); ax.xaxis.set_major_formatter(FixedFormatter([f'{v:,}' for v in x]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set(xlabel='Body steps between WRITE and QUERY',ylabel='Bit recall accuracy',
           title='A separate LSTM keeps its bit',ylim=(0,1.09))
    ax.legend(loc='lower left',frameon=False,fontsize=10); ax.grid(axis='y',alpha=.15)
    fig.savefig(args.output/'repetition-memory.png',dpi=200); plt.close(fig)
    provenance={'purpose':'presentation of previously verified results; no new scientific measurements',
                'wandb_required':False,'wandb_reason':'replotting existing accepted results',
                'inputs':{str(v.relative_to(ROOT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in [graph_path,memory_path]},
                'outputs':{v.name:hashlib.sha256(v.read_bytes()).hexdigest() for v in sorted(args.output.glob('repetition-*.png'))}}
    (args.output/'plot-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__ == '__main__':
    main()
