"""Independent saved-array audit of the frozen cached-forward v2 contract.

NumPy only: no model calls, checkpoints, labels, autodiff, or producer imports.
Author fixtures are engineering checks. Production requires an independently
reviewed admission and an offline reporting lifecycle before any NPZ is read.
The reducer cannot certify memory, competence, spectra, or harmful collapse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

CONFIG_SHA256 = '0170170b7724311a3903e5f1cde1917e91263b57fe45ebfc89f6d90b121a1740'
CONTRACT_SHA256 = '83c0b8c61fdefa9eb73e8386deab6c073d5ac04a82b13d251e1be699b25124d7'
TOL = {'float32': (1e-6, 1e-5), 'float64': (1e-12, 1e-10)}
ENTITY = 'mehrab-hamidi-milaquebec5431'
PROJECT = 'representation-collapse'


class AuditError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise AuditError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    a = np.asarray(a)
    header = json.dumps({'dtype': str(a.dtype), 'shape': list(a.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + a.tobytes(order='C')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def safe_file(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe artifact reference')
    result = Path(root) / p
    require(result.is_file() and not result.is_symlink(), 'missing/nonregular artifact: ' + str(name))
    require(result.resolve().is_relative_to(Path(root).resolve()), 'artifact escaped root')
    return result


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        require(len(z.files) == len(set(z.files)), 'duplicate NPZ member names')
        result = {k: z[k].copy() for k in z.files}
    require(all(not a.dtype.hasobject for a in result.values()), 'object arrays forbidden')
    require(not any('target' in k.lower() or 'label' in k.lower() for k in result), 'label arrays forbidden')
    return result


def compare(a, b, dtype, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.size > 0, 'mismatched/empty comparison')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite comparison operand')
    x, y = a.astype(np.float64), b.astype(np.float64)
    atol, rtol = TOL[dtype]
    with np.errstate(over='raise', invalid='raise'):
        gaps = np.abs(x-y); bounds = atol + rtol*np.abs(y)
    identity = a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')
    if exact:
        require(a.dtype == b.dtype, 'byte comparison dtype mismatch')
        left = np.ascontiguousarray(a).view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
        right = np.ascontiguousarray(b).view(np.uint8).reshape(b.shape + (b.dtype.itemsize,))
        bad = np.any(left != right, axis=-1)
    else:
        bad = gaps > bounds
    indices = np.argwhere(bad)
    return {'passed': not bool(np.any(bad)), 'bitwise_equal': identity,
            'rule': 'raw_bytes' if exact else 'abs(candidate-reference)<=atol+rtol*abs(reference)',
            'atol': atol, 'rtol': rtol, 'entries': int(a.size),
            'violation_count': len(indices), 'max_absolute_difference': float(gaps.max()),
            'violations': [{'index': i.tolist(), 'candidate': float(x[tuple(i)]),
                'reference': float(y[tuple(i)]), 'absolute_difference': float(gaps[tuple(i)]),
                'bound': float(bounds[tuple(i)])} for i in indices]}


def metadata_agrees(declared, actual, path='metadata'):
    """Derived JSON floats may differ by final FP64 rounding, never verdicts."""
    if isinstance(actual, dict):
        require(isinstance(declared, dict) and set(actual) <= set(declared), path + ' missing fields')
        for key, value in actual.items():
            metadata_agrees(declared[key], value, path + '/' + key)
    elif isinstance(actual, list):
        require(isinstance(declared, list) and len(actual) == len(declared), path + ' list mismatch')
        for i, (a, d) in enumerate(zip(actual, declared)):
            metadata_agrees(d, a, path + '/' + str(i))
    elif isinstance(actual, float):
        require(isinstance(declared, (int, float)) and not isinstance(declared, bool) and
                np.isfinite(declared) and np.isclose(actual, declared, atol=1e-14, rtol=1e-12), path + ' float mismatch')
    else:
        require(type(declared) is type(actual) and declared == actual, path + ' mismatch')


def behavior(candidate, reference, task, dtype):
    u, v = candidate.astype(np.float64), reference.astype(np.float64)
    compare(u, v, dtype)
    atol, rtol = TOL[dtype]; b = atol + rtol*np.abs(v); delta = u-v
    if task == 'categorical':
        a = np.argmax(v, axis=1); c = np.argmax(u, axis=1); rows = np.arange(len(v))
        margins = v[rows, a, None]-v; required = b[rows, a, None]+b
        strict = margins > required; strict[rows, a] = True
        unique = np.sum(v == v.max(axis=1, keepdims=True), axis=1) == 1
        return {'B0_argmax': a.tolist(), 'C_argmax': c.tolist(),
                'exact_prediction_agreement': (a == c).tolist(),
                'B0_ties': [np.flatnonzero(row == row.max()).tolist() for row in v],
                'C_ties': [np.flatnonzero(row == row.max()).tolist() for row in u],
                'margin_certified': (unique & np.all(strict, axis=1) & np.all(np.abs(delta) <= b, axis=1)).tolist(),
                'signed_differences': delta.tolist(), 'margins': margins.tolist(), 'required_margins': required.tolist(),
                'accuracy': None, 'reason': 'no_targets', 'B0_causal_bridge': False}
    require(task == 'continuous' and u.shape[1] == 1, 'unknown behavior task')
    e = np.abs(delta)
    with np.errstate(over='ignore', invalid='ignore'):
        actual = 3*e*(2*(np.abs(v)+1)+e); conditional = 3*b*(2*(np.abs(v)+1)+b)
    if not np.isfinite(actual).all() or not np.isfinite(conditional).all():
        return {'actual_MSE': None, 'reason': 'bound_overflow', 'bound': None}
    valid = e <= b
    return {'actual_MSE': None, 'reason': 'no_targets',
            'normalized_error_change_bound_per_row': actual[:, 0].tolist(),
            'mean_normalized_error_change_bound': float(np.mean(actual)),
            'tolerance_bound_per_row': [float(x) if ok else None for x, ok in zip(conditional[:, 0], valid[:, 0])],
            'tolerance_bound_defined': valid[:, 0].tolist(), 'gate_certified': False}


def effect(candidate, baseline, displacement):
    delta = candidate.astype(np.float64)-baseline.astype(np.float64)
    d = displacement.astype(np.float64)
    require(np.isfinite(delta).all() and np.isfinite(d).all(), 'nonfinite effect')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        norms = np.hypot.reduce(np.abs(d), axis=1)
        outputs = np.hypot.reduce(np.abs(delta), axis=1)
    zd, ze = np.all(d == 0, axis=1), np.all(delta == 0, axis=1)
    states, gains = [], []
    for n, o, z, e in zip(norms, outputs, zd, ze):
        state = 'represented_zero' if z else ('unresolved_at_this_perturbation' if e else 'resolved_C_effect')
        if not np.isfinite(n) or not np.isfinite(o): state = 'norm_overflow'
        if (n == 0 and not z) or (o == 0 and not e): state = 'norm_underflow'
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            gain = float(o/n) if n > 0 and np.isfinite(n) and np.isfinite(o) else None
        if gain is not None and not np.isfinite(gain): state, gain = 'gain_overflow', None
        if gain == 0 and not e and not z: state, gain = 'gain_underflow', None
        if state in ('norm_overflow', 'norm_underflow'): gain = None
        states.append(state); gains.append(gain)
    return {'signed_effect': delta.tolist(), 'status_per_row': states, 'gain_per_row': gains,
            'status_counts': {k: states.count(k) for k in sorted(set(states))},
            'undefined_gain_count': sum(x is None for x in gains),
            'represented_norm': [float(x) if np.isfinite(x) else None for x in norms],
            'effect_norm': [float(x) if np.isfinite(x) else None for x in outputs], 'scope': 'C_only'}


def operations(kind):
    require(kind in ('gru', 'lstm'), 'invalid model kind')
    return ['B0'] + ['C_step_' + str(t) for t in range(17)] + [
        'copy_query', 'zero_displacement_query', 'hidden_plus', 'hidden_minus'] + (
        ['cell_plus', 'cell_minus'] if kind == 'lstm' else [])


def registry(config):
    return [{'id': f"{c['id']}-{dt}-r{r}", 'context': c['id'], 'model': c['model'],
             'task': c['task'], 'dtype': dt, 'repeat': r, 'operations': operations(c['model'])}
            for c in config['contexts'] for dt in ('float64', 'float32') for r in (1, 2)]


def input_stream(task, seed=30019, histories=32):
    """Independent empty-body input rendering; never construct/read targets.

    Frozen production seeds are replayed only after root admission and offline
    reporting. Preparation tests use unrelated seeds and smaller geometry.
    """
    require(task in ('categorical', 'continuous') and histories > 0, 'input stream parameters')
    rng = np.random.Generator(np.random.PCG64(seed))
    values = (rng.integers(0, 16, size=(histories, 16), dtype=np.int64) if task == 'categorical'
              else rng.uniform(-1, 1, size=(histories, 16)).astype(np.float32))
    payload = np.concatenate((values, values), axis=0); count = 2*histories
    result = np.zeros((count, 17, 38), dtype=np.float32); rows = np.arange(count)
    result[:, :16, 0] = 1
    for position in range(16):
        result[:, position, 5+position] = 1
        if task == 'categorical': result[rows, position, 21+payload[:, position]] = 1
        else: result[:, position, 37] = payload[:, position]
    result[:, 16, 4] = 1; result[rows, 16, 5+rows%8] = 1
    return result


def direction_stream(contexts, seed=30029, histories=32, hidden=128):
    rng = np.random.Generator(np.random.PCG64(seed)); result = {}
    for context in contexts:
        for component in context['directions']:
            raw = rng.standard_normal((histories, hidden), dtype=np.float64)
            norms = np.sqrt(np.sum(raw*raw, axis=1, keepdims=True))
            require(np.isfinite(norms).all() and np.all(norms > 0), 'invalid frozen direction draw')
            normalized = raw/norms
            result[context['id']+'__'+component] = np.concatenate((normalized, normalized), axis=0)
    return result


def schema(kind, batch, hidden, encoder, outputs):
    components = ['hidden', 'cell'] if kind == 'lstm' else ['hidden']
    d = hidden*len(components); shapes = {}
    for base in ('B0', 'C'):
        shapes[base+'/cache'] = (batch, 17, encoder)
        shapes[base+'/hidden_sequence'] = (batch, 17, hidden)
        shapes[base+'/logits'] = (batch, outputs)
        for c in components: shapes[base+'/terminal/'+c] = (batch, hidden)
    shapes['C/complete_sequence'] = (batch, 18, d)
    for prefix in ('C/S0', 'C/origin_before', 'C/origin_after'):
        for c in components: shapes[prefix+'/'+c] = (batch, hidden)
    for op in operations(kind)[1:]:
        shapes[op+'/hidden_output'] = (batch, 1, hidden)
        for c in components:
            for phase in ('before', 'after', 'terminal'): shapes[op+'/'+phase+'/'+c] = (batch, hidden)
        for phase in ('before', 'after'): shapes[op+'/cache_'+phase] = (batch, 17, encoder)
        if not op.startswith('C_step_'):
            shapes[op+'/logits'] = (batch, outputs)
            shapes[op+'/requested_displacement'] = (batch, d)
            shapes[op+'/displacement'] = (batch, d)
            if op not in ('copy_query', 'zero_displacement_query'):
                shapes[op+'/direction'] = (batch, hidden); shapes[op+'/direction_norm'] = (batch,)
    return shapes, components


def local_edges(kind, keys):
    cs = ['hidden', 'cell'] if kind == 'lstm' else ['hidden']; edges = []
    for op in operations(kind)[1:]:
        edges += [(op+'/after/'+c, op+'/before/'+c, True) for c in cs]
        edges.append((op+'/cache_after', op+'/cache_before', True))
    edges += [('C/origin_after/'+c, 'C/origin_before/'+c, True) for c in cs]
    edges += [('C/'+field, 'B0/'+field, False) for field in ['logits', 'hidden_sequence'] + ['terminal/'+c for c in cs]]
    for op in ('copy_query', 'zero_displacement_query'):
        edges += [(op+'/'+field, 'C/'+field, False) for field in ['logits'] + ['terminal/'+c for c in cs]]
    edges += [(k+'/duplicate_half', k+'/first_half', False) for k in keys]
    return edges


def condition_audit(record, arrays, expected, geometry, directions):
    b, h, e = geometry; kind, dt = expected['model'], expected['dtype']
    for field in ('id', 'context', 'model', 'task', 'dtype', 'repeat'):
        require(record['condition'].get(field) == expected[field], 'condition mismatch: ' + field)
    if 'operations' in expected:
        require(record['condition'].get('operations') == expected['operations'], 'declared operation order')
    shapes, cs = schema(kind, b, h, e, 16 if expected['task'] == 'categorical' else 1)
    require(set(arrays) == set(shapes), 'missing/extra raw snapshots: ' + expected['id'])
    require(set(record['layouts']) == set(shapes), 'missing/extra layouts')
    require(record.get('scientific_acceptance') is False and record.get('core_calls') == len(operations(kind)), 'condition scope/call count')
    require(record.get('selected_rows') == list(range(b)), 'row ordering')
    require(record.get('state_packing') == ('hidden_then_cell' if kind == 'lstm' else 'hidden'), 'state packing')
    require(record.get('B0_intermediate_cell') is None and record.get('B0_intermediate_cell_reason') ==
            ('unobserved_fused_LSTM' if kind == 'lstm' else 'not_applicable'), 'unobserved B0 cell scope')
    for k, a in arrays.items():
        require(a.shape == shapes[k] and a.dtype == np.dtype('float64' if k.endswith('/direction_norm') else dt)
                and np.isfinite(a).all(), 'raw shape/dtype/finite: '+k)
        layout = record['layouts'][k]
        require(layout.get('source_shape') == list(a.shape) and layout.get('array_sha256') == array_sha(a), 'layout/hash: '+k)
        require(layout.get('dtype') == 'torch.'+str(a.dtype) and layout.get('row_ids') == list(range(b))
                and layout.get('feature_ids') == (list(range(a.shape[-1])) if a.ndim > 1 else None), 'layout identity: '+k)
        require(len(layout.get('source_stride', [])) == a.ndim and all(type(x) is int and x >= 0 for x in layout['source_stride']), 'invalid source strides')
    def packed(prefix): return np.concatenate([arrays[prefix+'/'+c] for c in cs], axis=1)
    def identical(a, ref, reason): require(compare(a, ref, dt, True)['passed'], reason)
    identical(packed('C/S0'), np.zeros((b, h*len(cs)), dtype=dt), 'noncanonical complete zero S0')
    identical(arrays['C/complete_sequence'][:, 0], packed('C/S0'), 'S0 export inconsistent')
    identical(arrays['B0/hidden_sequence'][:, -1], arrays['B0/terminal/hidden'], 'B0 terminal hidden inconsistent')
    for t in range(17):
        op = 'C_step_'+str(t)
        identical(packed(op+'/before'), arrays['C/complete_sequence'][:, t], 'trajectory chronology before '+op)
        identical(packed(op+'/terminal'), arrays['C/complete_sequence'][:, t+1], 'trajectory chronology terminal '+op)
        identical(arrays[op+'/hidden_output'][:, 0], arrays[op+'/terminal/hidden'], 'step output state inconsistent')
        identical(arrays['C/hidden_sequence'][:, t], arrays[op+'/hidden_output'][:, 0], 'hidden sequence inconsistent')
    identical(packed('C/terminal'), arrays['C/complete_sequence'][:, 17], 'C terminal inconsistent')
    identical(packed('C/origin_before'), arrays['C/complete_sequence'][:, 16], 'wrong query origin')
    construction = {}; recomputed_effects = {}
    for op in operations(kind)[18:]:
        requested = np.zeros((b, h*len(cs)), dtype=dt)
        supplied = {c: arrays['C/origin_before/'+c].copy() for c in cs}
        target = None
        if op == 'zero_displacement_query':
            supplied = {c: x + np.zeros_like(x) for c, x in supplied.items()}
        elif op != 'copy_query':
            target, sign = op.rsplit('_', 1); direction = directions[target].astype(dt)
            identical(arrays[op+'/direction'], direction, 'noncanonical direction')
            norm = np.sqrt(np.sum(direction.astype(np.float64)**2, axis=1))
            require(np.all(np.abs(norm-1) <= 1e-6) and np.all(norm > 0), 'direction norm')
            require(compare(arrays[op+'/direction_norm'], norm, 'float64')['passed'], 'saved norm wrong')
            step = direction*np.array(1e-3, dtype=dt)*(1 if sign == 'plus' else -1)
            supplied[target] += step; offset = cs.index(target)*h
            requested[:, offset:offset+h] = step
            actual_effect = effect(arrays[op+'/logits'], arrays['C/logits'], arrays[op+'/displacement'])
            metadata_agrees(record['effects'].get(op), actual_effect, 'effects/'+op)
            recomputed_effects[op] = actual_effect
        construction[op] = {}
        for c in cs:
            identical(arrays[op+'/before/'+c], supplied[c], 'arm construction wrong '+op+'/'+c)
            observed = {'targeted': c == target, **compare(arrays[op+'/before/'+c], arrays['C/origin_before/'+c], dt, True)}
            metadata_agrees(record['constructed_state_vs_origin'][op][c], observed, 'construction/'+op+'/'+c)
            construction[op][c] = observed
        identical(arrays[op+'/requested_displacement'], requested, 'requested displacement wrong')
        identical(arrays[op+'/displacement'], packed(op+'/before')-packed('C/origin_before'), 'represented displacement wrong')
    require(set(record['effects']) == set(recomputed_effects), 'effect arm inventory')
    for op in operations(kind)[1:]:
        identical(arrays[op+'/cache_before'], arrays['C/cache'], 'cache operand wrong')
    declared = {}
    for item in record['checks']:
        if 'relation' in item: continue
        edge = (item.get('candidate'), item.get('reference'))
        require(edge not in declared, 'duplicate declared comparison')
        declared[edge] = item
    expected_edges = local_edges(kind, arrays)
    require(set(declared) == {(a, r) for a, r, _ in expected_edges}, 'missing/unknown required comparison')
    checks = []
    for a, r, exact in expected_edges:
        if a.endswith('/duplicate_half'):
            key = a.removesuffix('/duplicate_half'); av, rv = arrays[key][b//2:], arrays[key][:b//2]
        else: av, rv = arrays[a], arrays[r]
        value = compare(av, rv, dt, exact)
        metadata_agrees(declared[(a, r)], value, 'comparison/'+a)
        checks.append({'candidate': a, 'reference': r, **value})
    result_behavior = behavior(arrays['C/logits'], arrays['B0/logits'], expected['task'], dt)
    metadata_agrees(record['behavior'], result_behavior, 'behavior')
    return {'id': expected['id'], 'checks': checks, 'behavior': result_behavior,
            'effects': recomputed_effects, 'construction_identity_separate_from_acceptance': construction,
            'numeric_comparisons_passed': all(c['passed'] for c in checks), 'scientific_acceptance': False}


def native_audit(entries, conditions, arrays_by_id, geometry, wall, reserve, offset=0, previous_end=-1.0):
    expected_calls = [(c, op) for c in conditions for op in operations(c['model'])]
    require(len(entries) == len(expected_calls), 'native registry incomplete')
    b, _, e = geometry
    for index, (entry, (c, op)) in enumerate(zip(entries, expected_calls), offset+1):
        require(entry.get('global_index') == index and all(entry.get(k) == c[k] for k in ('context', 'dtype', 'repeat'))
                and entry.get('operation') == op and entry.get('map') == ('B0' if op == 'B0' else 'C'), 'native order/identity')
        a = arrays_by_id[c['id']]
        t = int(op.removeprefix('C_step_')) if op.startswith('C_step_') else 16
        x = a['B0/cache'] if op == 'B0' else a['C/cache'][:, t:t+1]
        require(entry.get('input_hash') == array_sha(x) and entry.get('input_shape') == list(x.shape)
                and entry.get('input_dtype') == 'torch.'+c['dtype'] and entry.get('input_stride') == [17*e, e, 1], 'native input operand')
        parts = ['hidden', 'cell'] if c['model'] == 'lstm' else ['hidden']
        state = 'native_implicit_complete_zero' if op == 'B0' else {p: array_sha(a[op+'/before/'+p][None]) for p in parts}
        require(entry.get('complete_state_hash') == state, 'native complete-state operand')
        start, middle, end = (entry.get(k) for k in ('start_elapsed', 'native_start_elapsed', 'end_elapsed'))
        require(all(isinstance(x, (float, int)) and np.isfinite(x) for x in (start, middle, end))
                and previous_end <= start <= middle <= end < wall and middle < wall-reserve, 'native chronology/deadline')
        require(entry.get('completed') is True and entry.get('exception') is None, 'incomplete native invocation')
        previous_end = end
    return {'native_calls': len(entries), 'encoder_calls_expected': 2*len(conditions),
            'head_calls_expected': sum(8 if c['model'] == 'lstm' else 6 for c in conditions),
            'all_saved_complete_state_and_input_operands_verified': True, 'last_end_elapsed': previous_end}


def reduce(root, config, author_unit=False, reporter=None, admission=None):
    root = Path(root); geometry = (4, 3, 4) if author_unit else (64, 128, 64)
    if not author_unit:
        require(reporter is not None and getattr(reporter, '_state', None) == 'running'
                and reporter.__class__.__module__ == 'representation_collapse.reporting'
                and getattr(reporter, 'mode', None) == 'offline' and getattr(reporter, 'project', None) == PROJECT
                and getattr(reporter, 'entity', None) == ENTITY, 'production requires active reviewed offline reporter')
        require(isinstance(admission, dict) and admission.get('review_accepted') is True
                and admission.get('execution_authorized') is True and admission.get('reducer_sha256') == sha(__file__)
                and admission.get('input_root') == str(root.resolve()), 'production requires exact reviewed admission')
        lifecycle = read_json(reporter.receipt_path)
        require(lifecycle.get('work_may_start') is True and lifecycle.get('sdk_version') == '0.30.0'
                and lifecycle.get('state') == 'running', 'offline reporter not operational')
    if author_unit:
        receipt = read_json(safe_file(root, 'fixture-receipt.json'))
        require(receipt.get('kind') == 'cached_forward_v2_author_tiny_fixtures' and receipt.get('production_calls') == 0
                and receipt.get('checkpoint_deserializations') == 0 and receipt.get('scientific_acceptance') is False, 'not author fixtures')
        file_manifest = receipt['files']
        conditions = [{'id': f'{k}-{dt}-r{r}', 'context': k+'-tiny', 'model': k,
            'task': 'categorical' if k == 'gru' else 'continuous', 'dtype': dt, 'repeat': r}
            for k in ('gru', 'lstm') for dt in ('float64', 'float32') for r in (1, 2)]
        require(receipt['conditions'] == [c['id'] for c in conditions], 'fixture condition order')
        registry_name = 'engine-native-registry.json'; wall, reserve = 120, 5
    else:
        require(sha(safe_file(root, 'config.json')) == CONFIG_SHA256, 'raw config changed')
        conditions = registry(config)
        require(read_json(safe_file(root, 'condition-registry.json')) == conditions, '32-condition registry mismatch')
        file_manifest = read_json(safe_file(root, 'raw-manifest.json'))
        status = read_json(safe_file(root, 'status.json'))
        require(status.get('state') == 'completed' and status.get('native_attempted') == 736 and status.get('native_completed') == 736
                and status.get('encoder_calls') == 64 and status.get('head_calls') == 224
                and status.get('fits') == 0 and status.get('targets_read') is False and status.get('gradients') == 0
                and status.get('scientific_acceptance') is False, 'incomplete/invalid production status')
        registry_name = 'native-call-registry.json'; wall, reserve = 600, 10
    require(isinstance(file_manifest, dict) and file_manifest, 'missing artifact hashes')
    required_files = {registry_name} | {c['id']+ext for c in conditions for ext in ('.json', '.npz')}
    required_files |= {'gru-fixture-inputs.npz', 'lstm-fixture-inputs.npz'} if author_unit else {
        'input-categorical.npz', 'input-continuous.npz', 'canonical-directions.npz', 'config.json', 'condition-registry.json', 'environment.json', 'source-freeze.json'}
    if author_unit: required_files.add('independent-native-registry.json')
    require(required_files == set(file_manifest), 'missing/unknown declared artifact hash')
    for name, digest in file_manifest.items(): require(sha(safe_file(root, name)) == digest, 'artifact SHA mismatch: '+name)
    native = read_json(safe_file(root, registry_name))
    expected_count = sum(len(operations(c['model'])) for c in conditions)
    expected_heads = sum(8 if c['model'] == 'lstm' else 6 for c in conditions)
    require(native.get('attempted') == expected_count == native.get('completed') and native.get('encoder_calls') == 2*len(conditions)
            and native.get('head_calls') == expected_heads and len(native.get('entries', [])) == expected_count, 'terminal native/encoder/head counters')
    arrays_by_id = {}; reports = []; refs = {}; relation_checks = []; offset = 0; last_end = -1.0; prior_context = None; weight_hashes = {}
    directions_all = {} if author_unit else load_arrays(safe_file(root, 'canonical-directions.npz'))
    if not author_unit:
        require(set(directions_all) == {c['context']+'__'+p for c in conditions for p in
                (['hidden', 'cell'] if c['model'] == 'lstm' else ['hidden'])}, 'canonical direction inventory')
        replayed_directions = direction_stream(config['contexts'])
        for key in directions_all:
            require(compare(directions_all[key], replayed_directions[key], 'float64', True)['passed'], 'canonical PCG64 direction replay mismatch')
    inputs = {}
    for context in ({c['context']: c for c in conditions}).values():
        key = context['model'] if author_unit else context['task']
        if key not in inputs:
            inputs[key] = load_arrays(safe_file(root, key+'-fixture-inputs.npz' if author_unit else 'input-'+key+'.npz'))
            x = inputs[key]['inputs']
            require(x.shape == (geometry[0], 17, 38) and x.dtype == np.float32 and np.isfinite(x).all(), 'input snapshot geometry')
            require(compare(x[geometry[0]//2:], x[:geometry[0]//2], 'float32', True)['passed'], 'input histories not duplicated')
            if not author_unit:
                require(compare(x, input_stream(key), 'float32', True)['passed'], 'frozen PCG64/render input byte mismatch')
                require(set(inputs[key]) == {'inputs', 'row_ids', 'history_ids', 'query_ids'}, 'input registry fields')
                for field, divisor in [('row_ids', None), ('history_ids', 32), ('query_ids', 8)]:
                    v = np.arange(64) if divisor is None else np.arange(64)%divisor
                    require(np.array_equal(inputs[key][field], v), 'input row/history/query IDs')
    for c in conditions:
        if c['context'] != prior_context:
            arrays_by_id.clear(); refs.clear(); prior_context = c['context']
        cid = c['id']; record = read_json(safe_file(root, cid+'.json'))
        require(record.get('arrays_sha256') == sha(safe_file(root, cid+'.npz')), 'condition NPZ hash')
        arrays = load_arrays(safe_file(root, cid+'.npz')); arrays_by_id[cid] = arrays
        parts = ['hidden', 'cell'] if c['model'] == 'lstm' else ['hidden']
        canonical = {p: inputs[c['model']]['direction_'+p] if author_unit else directions_all[c['context']+'__'+p] for p in parts}
        for p, raw in canonical.items():
            require(raw.shape == (geometry[0], geometry[1]) and raw.dtype == np.float64 and np.isfinite(raw).all(), 'canonical direction shape')
            require(compare(raw[geometry[0]//2:], raw[:geometry[0]//2], 'float64', True)['passed'], 'canonical direction duplicates')
        reports.append(condition_audit(record, arrays, c, geometry, canonical))
        count = len(operations(c['model']))
        call_proof = native_audit(native['entries'][offset:offset+count], [c], {cid: arrays}, geometry, wall, reserve, offset, last_end)
        offset += count; last_end = call_proof['last_end_elapsed']
        references = []
        if c['repeat'] == 2: references.append(('repeat', refs[(c['context'], c['dtype'], 1)]))
        if c['dtype'] == 'float32': references.append(('cross_dtype', refs[(c['context'], 'float64', c['repeat'])]))
        expected_declared = {}
        for relation, refid in references:
            for field in arrays:
                value = compare(arrays[field], arrays_by_id[refid][field], 'float32' if relation == 'cross_dtype' else c['dtype'], relation == 'repeat')
                edge = {'relation': relation, 'candidate': cid+'/'+field, 'reference': refid+'/'+field, **value}
                relation_checks.append(edge); expected_declared[(relation, edge['candidate'], edge['reference'])] = edge
        if author_unit:
            require(not any('relation' in x for x in record['checks']), 'unexpected author-unit relation schema')
            declared = record.get('cross_dtype_checks', [])
            require(len(declared) == (len(arrays) if c['dtype'] == 'float32' else 0), 'fixture cross-dtype inventory')
            require(len({x['field'] for x in declared}) == len(declared), 'duplicate fixture cross-dtype field')
            for item in declared:
                require(item['field'] in arrays, 'unknown fixture cross-dtype field')
                refid = refs[(c['context'], 'float64', c['repeat'])]
                metadata_agrees(item, compare(arrays[item['field']], arrays_by_id[refid][item['field']], 'float32'), 'fixture cross-dtype')
        else:
            declared = {(x.get('relation'), x.get('candidate'), x.get('reference')): x for x in record['checks'] if 'relation' in x}
            require(len(declared) == sum('relation' in x for x in record['checks']) and set(declared) == set(expected_declared), 'missing/duplicate cross-condition operand references')
            for key, expected_edge in expected_declared.items(): metadata_agrees(declared[key], expected_edge, 'cross-condition comparison')
            context = next(x for x in config['contexts'] if x['id'] == c['context'])
            require(record.get('checkpoint_sha256') == context['checkpoint']['sha256'] and record.get('parameter_promotion_exact') is True
                    and record.get('input_promotion_exact') is True, 'checkpoint/promotion provenance declaration')
            digest = record.get('base_fp32_weight_hash')
            require(isinstance(digest, str) and len(digest) == 64 and all(x in '0123456789abcdef' for x in digest), 'base parameter hash declaration')
            require(weight_hashes.get(c['context'], digest) == digest, 'base parameter hash differs across dtype/repeats')
            weight_hashes[c['context']] = digest
        refs[(c['context'], c['dtype'], c['repeat'])] = cid
    proof = {'native_calls': expected_count, 'encoder_calls_expected': 2*len(conditions), 'head_calls_expected': expected_heads,
             'all_saved_complete_state_and_input_operands_verified': True, 'last_end_elapsed': last_end}
    if author_unit:
        independent = read_json(safe_file(root, 'independent-native-registry.json'))
        require(len(independent) == expected_count, 'independent fixture hook registry incomplete')
        expected_calls = [(c, op) for c in conditions for op in operations(c['model'])]
        for index, (entry, (condition, op)) in enumerate(zip(independent, expected_calls), 1):
            shape = [geometry[0], 17 if op == 'B0' else 1, geometry[2]]
            require(entry.get('index') == index and entry.get('condition') == condition['id']
                    and entry.get('input_shape') == shape and isinstance(entry.get('elapsed_seconds'), (float, int))
                    and 0 <= entry['elapsed_seconds'] < 120, 'independent fixture hook order/geometry')
        proof['independent_author_hook_calls_verified'] = expected_count
    else:
        expected_status = [{'id': report['id'], 'completed': True, 'failed_comparisons':
            sum(not c['passed'] for c in report['checks']) + sum(not c['passed'] for c in relation_checks if c['candidate'].startswith(report['id']+'/'))}
            for report in reports]
        require(status.get('conditions') == expected_status and status.get('error_type') is None,
                'terminal condition outcomes disagree with independent comparisons')
    return {'kind': 'cached_forward_v2_independent_numpy_audit', 'scope': 'author_unit' if author_unit else 'production_numerical_calibration',
            'condition_count': len(conditions), 'native_registry': proof, 'conditions': reports, 'cross_condition_checks': relation_checks,
            'artifact_hashes': file_manifest, 'calibrated_numeric_acceptance': all(r['numeric_comparisons_passed'] for r in reports) and
                all(x['passed'] for x in relation_checks), 'scientific_acceptance': False,
            'checkpoint_deserializations': 0, 'model_calls': 0, 'targets_read': False,
            'provenance_limit': 'Parameter promotion is frozen producer-source provenance, not re-executed; ' +
                ('author-unit streams are fixture artifacts.' if author_unit else 'production input/direction PCG64 streams independently replayed label-free.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/experiments/cached-state-forward-calibration-v2.json')
    parser.add_argument('--author-unit', action='store_true'); parser.add_argument('--admission')
    args = parser.parse_args(); output = Path(args.output).resolve(); input_root = Path(args.input).resolve()
    require(not output.exists() and not output.is_relative_to(input_root), 'unique output outside input required')
    require(sha(args.config) == CONFIG_SHA256, 'frozen configuration hash mismatch')
    config = read_json(args.config)
    if not args.author_unit:
        require(args.admission is not None, 'production needs reviewed independent-reduction admission')
        admission = read_json(args.admission)
        require(admission.get('kind') == 'cached_forward_v2_independent_reduction_admission' and admission.get('execution_authorized') is True
                and admission.get('scientific_acceptance') is False and admission.get('review_accepted') is True
                and admission.get('max_wall_seconds') == 600 and admission.get('cpu_threads') == 1
                and type(admission.get('external_guard_pid')) is int and admission['external_guard_pid'] > 0
                and admission.get('external_guard_pid') == int(os.environ.get('RC_REDUCER_EXTERNAL_GUARD_PID', '-1'))
                and admission.get('input_root') == str(input_root) and admission.get('output_root') == str(output)
                and admission.get('reducer_sha256') == sha(__file__) and admission.get('config_sha256') == CONFIG_SHA256
                and admission.get('raw_manifest_sha256') == sha(safe_file(input_root, 'raw-manifest.json')), 'invalid production admission')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')), 'one-CPU environment required')
        source_root = Path(__file__).resolve().parents[2]
        require(sha(source_root/'doc/auto/post-calibration-measurement-contract-v2-2026-10-14.md') == CONTRACT_SHA256
                and admission.get('contract_sha256') == CONTRACT_SHA256
                and admission.get('source_freeze_sha256') == sha(safe_file(input_root, 'source-freeze.json')), 'contract/input source-freeze identity')
        require(admission.get('reporter_sha256') == sha(source_root/'src/representation_collapse/reporting.py') and
                admission.get('lock_sha256') == sha(source_root/'uv.lock') and
                admission.get('producer_source_commit') == read_json(safe_file(input_root, 'source-freeze.json')).get('source_commit'), 'reviewed producer/reporting/lock identity')
        revision = admission.get('source_commit')
        require(isinstance(revision, str) and len(revision) == 40 and all(x in '0123456789abcdef' for x in revision), 'immutable reducer source revision')
        for relative in ('code/analysis/reduce_cached_forward_v2.py', 'src/representation_collapse/reporting.py',
                         'configs/experiments/cached-state-forward-calibration-v2.json', 'uv.lock'):
            committed = subprocess.check_output(['git', 'show', revision+':'+relative], cwd=source_root, stderr=subprocess.DEVNULL)
            require(committed == (source_root/relative).read_bytes(), 'committed reviewed source differs: '+relative)
        metadata_receipt = Path(admission.get('producer_metadata_review_path', ''))
        require(metadata_receipt.is_file() and not metadata_receipt.is_symlink()
                and admission.get('producer_metadata_review_sha256') == sha(metadata_receipt), 'reviewed producer terminal/deadline/offline metadata receipt required')
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        if args.author_unit:
            audit = reduce(input_root, config, True)
        else:
            from representation_collapse import reporting
            require(Path(reporting.__file__).resolve() == source_root/'src/representation_collapse/reporting.py', 'wrong reporter module')
            ExperimentReporter = reporting.ExperimentReporter
            with ExperimentReporter(output/'reporting', project=PROJECT, entity=ENTITY, mode='offline',
                    config={'kind': 'independent_cached_forward_reduction', 'scientific_acceptance': False, 'cpu_threads': 1,
                        'source_commit': admission['source_commit'], 'producer_source_commit': admission['producer_source_commit'],
                        'input_seed': 30019, 'direction_seed': 30029, 'python_version': platform.python_version(),
                        'numpy_version': np.__version__},
                    identities={'reducer_sha256': sha(__file__), 'config_sha256': CONFIG_SHA256,
                        'admission_sha256': sha(args.admission), 'reporter_sha256': admission['reporter_sha256'],
                        'lock_sha256': admission['lock_sha256'], 'raw_manifest_sha256': admission['raw_manifest_sha256'],
                        'producer_metadata_review_sha256': admission['producer_metadata_review_sha256'],
                        'source_freeze_sha256': admission['source_freeze_sha256'],
                        'producer_environment_sha256': sha(safe_file(input_root, 'environment.json'))}, run_id=admission['run_id']) as reporter:
                audit = reduce(input_root, config, False, reporter=reporter, admission=admission)
                reporter.log({'audit/condition_count': audit['condition_count'], 'audit/numerical_acceptance':
                    int(audit['calibrated_numeric_acceptance']), 'audit/scientific_acceptance': 0}, step=0)
                reporter.log_manifest('independent-reduction-identities', metadata={'condition_count': audit['condition_count']},
                    identities={'reducer_sha256': sha(__file__), 'raw_manifest_sha256': admission['raw_manifest_sha256']})
        audit.update(reducer_sha256=sha(__file__), config_sha256=CONFIG_SHA256, elapsed_seconds=time.monotonic()-started,
                     python_version=platform.python_version(), numpy_version=np.__version__)
        require(audit['elapsed_seconds'] < 600, 'reduction deadline exceeded')
        atomic_json(output/'audit.json', audit)
        atomic_json(output/'hash-manifest.json', {'audit.json': sha(output/'audit.json'), 'reducer_sha256': sha(__file__),
            'config_sha256': CONFIG_SHA256, 'scientific_acceptance': False})
        print(json.dumps({'state': 'completed', 'output': str(output), 'audit_sha256': sha(output/'audit.json'),
            'calibrated_numeric_acceptance': audit['calibrated_numeric_acceptance'], 'scientific_acceptance': False}))
    except BaseException as error:
        atomic_json(output/'failure.json', {'state': 'rejected', 'error_type': type(error).__name__, 'reason': str(error),
            'scientific_acceptance': False, 'calibrated_numeric_acceptance': False, 'elapsed_seconds': time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
