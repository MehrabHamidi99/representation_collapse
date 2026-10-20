"""Seed-0 discovery of native one-bit retention; never claims capacity collapse."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import platform
import socket
import subprocess
import time
import numpy as np
from .vector_task import paired_body_ids, body_from_ids
from .reporting import ExperimentReporter, file_sha256


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x') as f:
        f.write(canonical(value) + '\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def schedules(rng, count, repeats, q=1):
    if count < 1 or repeats < 0 or q < 1:
        raise ValueError('invalid schedule dimensions')
    return [('', '') if repeats == 0 else tuple(body_from_ids(ids, q) for ids in paired_body_ids(rng, repeats)) for _ in range(count)]


def render(bodies, bits):
    if len(bodies) != len(bits) or not bodies or len(set(map(len, bodies))) != 1:
        raise ValueError('equal nonempty batch and body horizons required')
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError('binary cue required')
    x = np.zeros((len(bodies), len(bodies[0]) + 2, 6), dtype=np.float32)
    x[:, 0, 0] = 1; x[:, 0, 5] = np.asarray(bits) * 2 - 1
    for i, body in enumerate(bodies):
        for t, s in enumerate(body, 1):
            x[i, t, {'A': 1, 'B': 2, 'X': 3}[s]] = 1
    x[:, -1, 4] = 1
    return x


def training_batch(rng, repeats):
    bits = np.array([0] * 16 + [1] * 16); rng.shuffle(bits)
    pairs = schedules(rng, 32, repeats)
    return render([s for pair in pairs for s in pair], np.repeat(bits, 2)), np.repeat(bits, 2)


def evaluation_batch(pairs):
    # schedule-major, cue-major, arm-minor: four dependent rows per forcing.
    bodies = [pair[arm] for pair in pairs for bit in (0, 1) for arm in (0, 1)]
    bits = np.tile([0, 0, 1, 1], len(pairs))
    return render(bodies, bits), bits


def model_digest(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_model(torch):
    class Memory(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.cell = torch.nn.LSTMCell(6, 32); self.head = torch.nn.Linear(32, 2)
        def before_query(self, x):
            h = x.new_zeros((len(x), 32)); c = h.clone(); write = None
            for t in range(x.shape[1] - 1):
                h, c = self.cell(x[:, t], (h, c))
                if t == 0:
                    write = torch.cat((h, c), 1)
            return h, c, write
        def query(self, x, h, c):
            h, c = self.cell(x[:, -1], (h, c)); return self.head(h)
        def forward(self, x):
            h, c, _ = self.before_query(x); return self.query(x, h, c)
    return Memory()


def guard(args):
    if Path(args.stop_file).exists() or time.time() >= args.deadline_unix:
        raise RuntimeError('STOP or deadline reached')


def evaluate(model, pairs, torch, args, *, rows_file=None, condition='', interventions=False):
    model.eval(); records = []; states = []; write_states = []
    with torch.no_grad():
        for start in range(0, len(pairs), 32):
            guard(args)
            x_np, labels = evaluation_batch(pairs[start:start + 32]); x = torch.from_numpy(x_np)
            h, c, write = model.before_query(x)
            logits = model.query(x, h, c)
            if not all(torch.isfinite(v).all() for v in (h, c, write, logits)):
                raise RuntimeError('nonfinite evaluation state/logits')
            variants = {'native': logits}
            if interventions:
                indices = torch.arange(len(x)).reshape(-1, 4)[:, [2, 3, 0, 1]].reshape(-1)
                same_cue = torch.arange(len(x)).reshape(-1, 4).roll(1, 0).reshape(-1)
                variants.update(swap=model.query(x, h[indices], c[indices]),
                                self_copy=model.query(x, h.clone(), c.clone()),
                                same_cue_copy=model.query(x, h[same_cue], c[same_cue]),
                                zero=model.query(x, torch.zeros_like(h), torch.zeros_like(c)))
                if not all(torch.isfinite(v).all() for v in variants.values()):
                    raise RuntimeError('nonfinite intervention logits')
                if not torch.equal(variants['same_cue_copy'], logits[same_cue]):
                    raise RuntimeError('same-cue donor native replay mismatch')
                if not torch.equal(variants['swap'], logits[indices]):
                    raise RuntimeError('swap donor native replay mismatch')
                if not torch.equal(variants['self_copy'], logits):
                    raise RuntimeError('self-copy replay mismatch')
                states.append(torch.cat((h, c), 1).numpy()); write_states.append(write.numpy())
            arrays = {key: value.numpy() for key, value in variants.items()}
            for row, label in enumerate(labels):
                record = {'condition': condition, 'forcing_unit': start + row // 4, 'bit': int(label), 'arm': row % 2,
                          'logits': {key: value[row].tolist() for key, value in arrays.items()},
                          'prediction': {key: int(value[row].argmax()) for key, value in arrays.items()}}
                record['oracle_prediction'] = int(x_np[row, 0, 5] > 0)
                record['error'] = {key: int(pred != label) for key, pred in record['prediction'].items()}
                record['swap_donor_correct'] = int(record['prediction'].get('swap', -1) == 1 - label)
                records.append(record)
                if rows_file:
                    rows_file.write(canonical(record) + '\n')
    accuracy = {f'bit{bit}_arm{arm}': float(np.mean([r['prediction']['native'] == bit for r in records if r['bit'] == bit and r['arm'] == arm])) for bit in (0, 1) for arm in (0, 1)}
    result = {'condition': condition, 'forcing_units': len(pairs), 'cue_body_units': 2 * len(pairs), 'accuracy': accuracy,
              'mean_accuracy': float(np.mean([r['error']['native'] == 0 for r in records])),
              'cross_entropy': {str(arm): float(np.mean([float(np.logaddexp(*r['logits']['native'])) - r['logits']['native'][r['bit']] for r in records if r['arm'] == arm])) for arm in (0, 1)},
              'constant_zero_readout_accuracy': float(np.mean([r['bit'] == 0 for r in records])),
              'register_oracle_accuracy': float(np.mean([r['oracle_prediction'] == r['bit'] for r in records]))}
    if interventions:
        result['intervention_accuracy_against_original'] = {key: float(np.mean([r['error'][key] == 0 for r in records])) for key in variants}
        result['swap_donor_accuracy'] = float(np.mean([r['swap_donor_correct'] for r in records]))
        result['paired_error_difference'] = float(np.mean([r['error']['native'] for r in records if r['arm'] == 0]) - np.mean([r['error']['native'] for r in records if r['arm'] == 1]))
        z = np.concatenate(states).reshape(-1, 2, 2, 64)
        w = np.concatenate(write_states).reshape(-1, 2, 2, 64)
        result['cue_separation'] = {name: {'mean_l2': float(np.linalg.norm(a[:, 1] - a[:, 0], axis=-1).mean()), 'maximum_absolute_difference': float(np.max(np.abs(a[:, 1] - a[:, 0]))), 'exact_equal_fraction': float(np.all(a[:, 1] == a[:, 0], axis=-1).mean())} for name, a in [('after_write', w), ('before_query', z)]}
        # Complete-state raw units, separate per arm and within body cue comparison.
        result['geometry'] = {}
        for arm in (0, 1):
            a = z[:, :, arm].reshape(-1, 64).astype(np.float64); a -= a.mean(0)
            eig = np.linalg.eigvalsh(a.T @ a / (len(a) - 1))
            tolerance = 64 * np.finfo(np.float64).eps * max(a.shape) * max(float(np.max(np.abs(eig))), np.finfo(np.float64).tiny)
            if not np.isfinite(eig).all() or float(eig.min()) < -tolerance:
                raise RuntimeError('materially negative or nonfinite covariance spectrum')
            negative_minimum = float(eig.min()); eig = np.maximum(eig, 0)
            trace = float(eig.sum())
            result['geometry'][str(arm)] = {'samples': len(a), 'features': 64, 'centering': 'pooled cues within arm', 'units': 'raw unrescaled H then C', 'rank_ceiling': min(len(a)-1, 64), 'negative_eigenvalue_tolerance': tolerance, 'minimum_unclamped_eigenvalue': negative_minimum, 'trace': trace, 'pr': float(trace * trace / (eig @ eig)) if trace else None, 'eigenvalues': eig.tolist()}
        np.savez(Path(args.output) / f'{condition}-states.npz', before_query=z, after_write=w)
    return result


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    source_hash = file_sha256(Path(__file__)); protocol_hash = file_sha256(args.protocol)
    with ExperimentReporter(out / 'reporting', project='representation-collapse', run_id=args.run_id, mode='offline',
                            config={'seed': 0, 'updates': 5000, 'hidden_size': 32, 'dtype': 'float32', 'task': 'onebit-memory'},
                            identities={'source_sha256': source_hash, 'protocol_sha256': protocol_hash}) as reporter:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        model = make_model(torch); initial_model_sha256 = model_digest(model); optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
        train_rng = np.random.default_rng(1000); val_rng = np.random.default_rng(2000)
        validation = {str(r): schedules(val_rng, 512, r) for r in (0, 1, 2)}
        atomic(out / 'validation-design.json', validation)
        design_hash = file_sha256(out / 'validation-design.json')
        atomic(out / 'identity.json', {'source_sha256': source_hash, 'protocol_sha256': protocol_hash, 'validation_sha256': design_hash,
                                     'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version(), 'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'initial_model_sha256': initial_model_sha256, 'seed': 0, 'training_rng_seed': 1000, 'validation_rng_seed': 2000, 'test_rng_seed': 3000})
        stage_results = []; update = 0; passed = True
        with (out / 'training.jsonl').open('x') as history:
            for stage, (count, horizons) in enumerate([(1000, [0]), (2000, [0, 1, 2]), (2000, [0, 1, 2, 4, 8])], 1):
                for within in range(count):
                    guard(args); r = horizons[within % len(horizons)]
                    x, labels = training_batch(train_rng, r); model.train(); optimizer.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(x)); loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(labels))
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); optimizer.step(); update += 1
                    history.write(canonical({'update': update, 'stage': stage, 'R': r, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}) + '\n')
                    if update % 25 == 0:
                        history.flush(); reporter.log({'training/loss': float(loss.detach()), 'training/update': update}, step=update)
                history.flush(); os.fsync(history.fileno())
                with (out / f'stage-{stage}-validation.jsonl').open('x') as validation_rows:
                    checks = [evaluate(model, validation[str(r)], torch, args, rows_file=validation_rows, condition=f'validation-R{r}') for r in ([0] if stage == 1 else [0, 1, 2])]
                    validation_rows.flush(); os.fsync(validation_rows.fileno())
                passed = all(a >= .95 for check in checks for a in check['accuracy'].values())
                stage_results.append({'stage': stage, 'update': update, 'passed': passed, 'checks': checks})
                atomic(out / f'stage-{stage}.json', stage_results[-1])
                if not passed: break
        checkpoint = out / 'checkpoint.pt'; torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'update': update}, checkpoint)
        checkpoint_sha256 = file_sha256(checkpoint); frozen_model_sha256 = model_digest(model)
        atomic(out / 'pre-evaluation-checkpoint.json', {'checkpoint_sha256': checkpoint_sha256, 'model_sha256': frozen_model_sha256, 'update': update, 'eligible': passed})
        results = []
        if passed:
            rng = np.random.default_rng(3000)
            with (out / 'evaluation.jsonl').open('x') as rows:
                for r, q in [(0, 1), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (32, 1), (16, 2), (8, 4)]:
                    pairs = schedules(rng, 1024, r, q); condition = f'R{r}-q{q}'
                    atomic(out / f'{condition}-design.json', pairs)
                    result = evaluate(model, pairs, torch, args, rows_file=rows, condition=condition, interventions=True)
                    results.append(result); atomic(out / f'{condition}-summary.json', result); rows.flush(); os.fsync(rows.fileno())
        if file_sha256(checkpoint) != checkpoint_sha256 or model_digest(model) != frozen_model_sha256:
            raise RuntimeError('frozen checkpoint/model changed during evaluation')
        result = {'schema': 'onebit-memory-discovery-v1', 'status': 'competence_passed' if passed else 'competence_failed', 'updates': update,
                  'stage_results': stage_results, 'evaluation': results, 'checkpoint_sha256': file_sha256(checkpoint), 'validation_sha256': design_hash,
                  'source_sha256': source_hash, 'protocol_sha256': protocol_hash, 'heldout_opened': passed, 'initial_model_sha256': initial_model_sha256, 'final_model_sha256': model_digest(model),
                  'limitations': ['one discovery seed', 'forcing units pair opposite cues and both arms', 'not a memory capacity test', 'no Floquet measurement']}
        atomic(out / 'result.json', result)
        atomic(out / 'artifact-inventory.json', {p.name: {'sha256': file_sha256(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()})
        reporter.log({'result/competence_passed': int(passed), 'result/updates': update}, step=update + 1)
        reporter.log_manifest('result', metadata={'competence_passed': passed, 'updates': update}, identities={'result_sha256': file_sha256(out / 'result.json')})
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--protocol', required=True)
    p.add_argument('--run-id', required=True); p.add_argument('--deadline-unix', required=True, type=float); p.add_argument('--stop-file', required=True)
    args = p.parse_args(); run(args)

if __name__ == '__main__': main()
