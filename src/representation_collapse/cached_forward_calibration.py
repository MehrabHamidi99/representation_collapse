"""Frozen 736-call B0/C forward calibration; production requires root admission.

Tiny unit fixtures call capture_condition directly under their own bounded
native-hook budget. This module has no training, target or derivative path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

PROTOCOL = 'cached-state-forward-calibration-v2'
CONFIG_SHA256 = '0170170b7724311a3903e5f1cde1917e91263b57fe45ebfc89f6d90b121a1740'
CONTRACT_SHA256 = '83c0b8c61fdefa9eb73e8386deab6c073d5ac04a82b13d251e1be699b25124d7'
TOLERANCES = {'float32': (1e-6, 1e-5), 'float64': (1e-12, 1e-10)}
ENGINE_PATH = 'src/representation_collapse/cached_forward_calibration.py'
LAUNCHER_PATH = 'code/ops/cached_forward_calibration_launcher.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def atomic_npz(path, arrays):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(json.dumps({'dtype': str(a.dtype), 'shape': list(a.shape)}, sort_keys=True).encode())
    h.update(a.tobytes()); return h.hexdigest()


def snapshot(value):
    return value.detach().cpu().numpy().copy()


def state_components(state, kind):
    import torch
    if kind == 'lstm':
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError('complete LSTM hidden and cell required')
        parts = dict(zip(('hidden', 'cell'), state))
    elif kind == 'gru' and isinstance(state, torch.Tensor):
        parts = {'hidden': state}
    else:
        raise ValueError('invalid native state kind/structure')
    shapes = {tuple(v.shape) for v in parts.values()}
    if len(shapes) != 1 or any(v.ndim != 3 or v.shape[0] != 1 for v in parts.values()):
        raise ValueError('complete one-layer native state shape required')
    return parts


def clone_state(state, kind):
    parts = [v.detach().clone().contiguous() for v in state_components(state, kind).values()]
    return tuple(parts) if kind == 'lstm' else parts[0]


def operations(kind):
    return ['B0'] + [f'C_step_{t}' for t in range(17)] + ['copy_query', 'zero_displacement_query',
        'hidden_plus', 'hidden_minus'] + (['cell_plus', 'cell_minus'] if kind == 'lstm' else [])


def condition_registry(config):
    conditions = []
    for context in config['contexts']:
        for dtype in config['dtype_order']:
            for repeat in config['repeat_order']:
                conditions.append({'id': f"{context['id']}-{dtype}-r{repeat}", 'context': context['id'],
                                   'model': context['model'], 'task': context['task'], 'dtype': dtype,
                                   'repeat': repeat, 'operations': operations(context['model'])})
    return conditions


class StopCalibration(RuntimeError):
    pass


@dataclass
class CallBudget:
    maximum: int
    started: float
    wall_seconds: float
    reserve: float
    stop_path: Path | None = None
    registry_path: Path | None = None
    attempted: int = 0
    completed: int = 0
    entries: list = field(default_factory=list)
    encoder_calls: int = 0
    head_calls: int = 0

    def check(self, *, native=False):
        elapsed = time.monotonic() - self.started
        if elapsed < 0 or elapsed >= self.wall_seconds - (self.reserve if native else 0):
            raise StopCalibration('shared deadline reached')
        if self.stop_path is not None and self.stop_path.exists():
            raise StopCalibration('owner STOP requested')
        if native and self.attempted >= self.maximum:
            raise StopCalibration('native invocation cap reached')

    def persist(self):
        if self.registry_path is not None:
            atomic_json(self.registry_path, {'attempted': self.attempted, 'completed': self.completed,
                'encoder_calls': self.encoder_calls, 'head_calls': self.head_calls, 'entries': self.entries})

    def begin(self, condition, operation, encoded, state, kind):
        self.check(native=True)
        state_hash = 'native_implicit_complete_zero' if state is None else {
            k: array_hash(snapshot(v)) for k, v in state_components(state, kind).items()}
        self.attempted += 1
        self.entries.append({'global_index': self.attempted, 'context': condition['context'],
            'dtype': condition['dtype'], 'repeat': condition['repeat'], 'operation': operation,
            'map': 'B0' if operation == 'B0' else 'C', 'input_hash': array_hash(snapshot(encoded)),
            'input_shape':list(encoded.shape),'input_stride':list(encoded.stride()),'input_dtype':str(encoded.dtype),
            'complete_state_hash': state_hash, 'start_elapsed': time.monotonic()-self.started,
            'native_start_elapsed': None, 'end_elapsed': None, 'completed': False, 'exception': None})
        self.persist()
        # Hashing and fsync consume the same clock. Do not cross the cutoff here.
        self.check(native=False)
        native_start = time.monotonic()-self.started
        if native_start >= self.wall_seconds-self.reserve:
            raise StopCalibration('native start cutoff reached during bookkeeping')
        self.entries[-1]['native_start_elapsed'] = native_start

    def end(self):
        self.completed += 1
        self.entries[-1].update(completed=True, end_elapsed=time.monotonic()-self.started)
        self.persist()


def compare(candidate, reference, dtype, *, bitwise=False):
    a, b = np.asarray(candidate), np.asarray(reference)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite matched nonempty comparison operands required')
    left, right = a.astype(np.float64), b.astype(np.float64)
    atol, rtol = TOLERANCES[dtype]
    with np.errstate(over='raise', invalid='raise'):
        differences = left-right; gaps = np.abs(differences); bounds = atol+rtol*np.abs(right)
    exact = a.dtype == b.dtype and a.tobytes() == b.tobytes()
    failures = gaps > bounds
    if bitwise:
        if a.dtype != b.dtype:
            raise ValueError('bitwise comparison requires same dtype')
        failures = (a.copy().view(np.uint8).reshape(a.shape + (a.dtype.itemsize,)) !=
                    b.copy().view(np.uint8).reshape(b.shape + (b.dtype.itemsize,))).any(axis=-1)
    idx = np.argwhere(failures)
    details = [{'index': i.tolist(), 'candidate': float(left[tuple(i)]), 'reference': float(right[tuple(i)]),
                'absolute_difference': float(gaps[tuple(i)]), 'bound': float(bounds[tuple(i)])} for i in idx]
    return {'passed': exact if bitwise else not bool(failures.any()), 'bitwise_equal': exact,
            'rule': 'raw_bytes' if bitwise else 'abs(candidate-reference)<=atol+rtol*abs(reference)',
            'atol': atol, 'rtol': rtol, 'entries': a.size, 'violations': details,
            'violation_count': len(details), 'max_absolute_difference': float(gaps.max())}


def behavior_bounds(candidate, reference, task, dtype):
    u, v = np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    compare(u, v, dtype)
    atol, rtol = TOLERANCES[dtype]; bound = atol+rtol*np.abs(v); delta = u-v
    if task == 'categorical':
        winner = v.argmax(axis=1); other = u.argmax(axis=1)
        rows = np.arange(len(v)); margins = v[rows,winner,None]-v
        needed = bound[rows,winner,None]+bound
        unique = (v == v.max(axis=1, keepdims=True)).sum(axis=1) == 1
        checks = margins > needed; checks[rows,winner] = True
        return {'B0_argmax': winner.tolist(), 'C_argmax': other.tolist(),
            'exact_prediction_agreement': (winner == other).tolist(),
            'B0_ties': [np.flatnonzero(row == row.max()).tolist() for row in v],
            'C_ties': [np.flatnonzero(row == row.max()).tolist() for row in u],
            'margin_certified': (unique & checks.all(axis=1) & (np.abs(delta)<=bound).all(axis=1)).tolist(),
            'signed_differences': delta.tolist(), 'margins': margins.tolist(), 'required_margins': needed.tolist(),
            'accuracy': None, 'reason': 'no_targets', 'B0_causal_bridge': False}
    if task != 'continuous' or u.shape[1] != 1:
        raise ValueError('declared native task/output required')
    e = np.abs(delta); actual = 3*e*(2*(np.abs(v)+1)+e)
    conditional = 3*bound*(2*(np.abs(v)+1)+bound)
    valid = e<=bound
    if not np.isfinite(actual).all() or not np.isfinite(conditional).all():
        return {'actual_MSE': None, 'reason': 'bound_overflow', 'bound': None}
    return {'actual_MSE': None, 'reason': 'no_targets', 'normalized_error_change_bound_per_row': actual[:,0].tolist(),
        'mean_normalized_error_change_bound': float(actual.mean()),
        'tolerance_bound_per_row': [float(x) if ok else None for x, ok in zip(conditional[:,0],valid[:,0])],
        'tolerance_bound_defined': valid[:,0].tolist(), 'gate_certified': False}


def perturbation_effect(candidate, baseline, displacement):
    delta = np.asarray(candidate, dtype=np.float64)-np.asarray(baseline, dtype=np.float64)
    d = np.asarray(displacement, dtype=np.float64)
    if not np.isfinite(delta).all() or not np.isfinite(d).all():
        raise ValueError('finite perturbation operands required')
    # hypot rescales intermediate squares, preserving subnormal nonzero rows.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        norm = np.hypot.reduce(np.abs(d), axis=1)
        effect_norm = np.hypot.reduce(np.abs(delta), axis=1)
    zero_d = (d == 0).all(axis=1); zero_effect = (delta == 0).all(axis=1)
    statuses, gains = [], []
    for n, e, zd, ze in zip(norm, effect_norm, zero_d, zero_effect):
        status = 'represented_zero' if zd else ('unresolved_at_this_perturbation' if ze else 'resolved_C_effect')
        if not np.isfinite(n) or not np.isfinite(e): status = 'norm_overflow'
        if (n == 0 and not zd) or (e == 0 and not ze): status = 'norm_underflow'
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            gain = float(e/n) if n > 0 and np.isfinite(n) and np.isfinite(e) else None
        if gain is not None and not np.isfinite(gain): gain = None; status = 'gain_overflow'
        if gain == 0 and not ze and not zd: gain = None; status = 'gain_underflow'
        if status in ('norm_overflow','norm_underflow'): gain = None
        gains.append(gain); statuses.append(status)
    return {'signed_effect': delta.tolist(), 'status_per_row': statuses, 'gain_per_row': gains,
        'status_counts': {status:statuses.count(status) for status in sorted(set(statuses))},
        'undefined_gain_count':sum(gain is None for gain in gains),
        'represented_norm': [float(x) if np.isfinite(x) else None for x in norm],
        'effect_norm': [float(x) if np.isfinite(x) else None for x in effect_norm], 'scope': 'C_only'}


def capture_condition(model, inputs, directions, condition, budget, arrays=None):
    """Execute precisely 22 GRU/24 LSTM native calls, preserving partial arrays."""
    import torch
    kind = model.model_kind; arrays = {} if arrays is None else arrays
    layouts = {}; checks = []; active = None; core_count = 0
    planned = operations(kind); batch, length, _ = inputs.shape
    if length != 17 or batch < 2 or batch % 2 or model.core.num_layers != 1:
        raise ValueError('paired complete 17-update geometry required')
    if model.training or torch.is_grad_enabled():
        raise ValueError('explicit eval/no_grad execution required')
    if set(directions) != ({'hidden', 'cell'} if kind == 'lstm' else {'hidden'}):
        raise ValueError('complete predetermined direction set required')
    dtype = str(inputs.dtype).split('.')[-1]
    def save(key, value):
        arrays[key] = snapshot(value)
        layouts[key] = {'source_shape': list(value.shape), 'source_stride': list(value.stride()),
            'dtype': str(value.dtype), 'array_sha256': array_hash(arrays[key]),
            'row_ids':list(range(batch)),
            'feature_ids':list(range(value.shape[-1])) if value.ndim>1 else None}
    def save_state(prefix, state):
        for component, value in state_components(state, kind).items(): save(prefix+'/'+component, value[0])
    def finite(*keys):
        if any(not np.isfinite(arrays[k]).all() for k in keys):
            raise FloatingPointError('nonfinite required native observable')
    def edge(candidate, reference, exact=False):
        checks.append({'candidate': candidate, 'reference': reference,
                       **compare(arrays[candidate], arrays[reference], dtype, bitwise=exact)})
    def pre(module, args):
        nonlocal core_count
        if core_count >= len(planned) or active != planned[core_count]:
            raise StopCalibration('unregistered native invocation')
        budget.begin(condition, active, args[0], args[1] if len(args)>1 else None, kind)
        core_count += 1
    def post(module, args, result):
        budget.end()
        if active == 'B0':
            save('B0/hidden_sequence', result[0]); save_state('B0/terminal', result[1])
    def encoded(module, args, result):
        budget.encoder_calls += 1
        if active == 'B0': save('B0/cache', result)
    def head_hook(module, args, result): budget.head_calls += 1
    handles = [model.core.register_forward_pre_hook(pre), model.core.register_forward_hook(post),
               model.encoder.register_forward_hook(encoded), model.head.register_forward_hook(head_hook)]
    weight_before = {k: snapshot(v) for k,v in model.state_dict().items()}
    try:
        active = 'B0'; save('B0/logits', model(inputs)); finite(*arrays)
        active = 'C_encoder'; cache = model.encoder(inputs); save('C/cache', cache)
        h = torch.zeros((1,batch,model.hidden_size), dtype=inputs.dtype, device='cpu')
        state = (h, h.clone()) if kind == 'lstm' else h
        sequence = []; packed = []; save_state('C/S0', state)
        def pack(s): return torch.cat([v[0] for v in state_components(s,kind).values()],dim=1)
        packed.append(pack(state).clone())
        def call(name, supplied, row):
            nonlocal active
            active = name
            save_state(name+'/before', supplied); save(name+'/cache_before', cache)
            try:
                output, returned = model.core(cache[:,row:row+1,:], supplied)
            finally:
                save_state(name+'/after', supplied); save(name+'/cache_after', cache)
            for component in state_components(supplied,kind): edge(name+'/after/'+component,name+'/before/'+component,True)
            edge(name+'/cache_after',name+'/cache_before',True)
            if any(not c['passed'] for c in checks[-(len(state_components(supplied,kind))+1):]):
                raise StopCalibration('state/cache mutation detected')
            save(name+'/hidden_output', output); save_state(name+'/terminal',returned)
            finite(*[k for k in arrays if k.startswith(name+'/')]); return output,returned
        origin = None
        for t in range(17):
            if t == 16: origin = clone_state(state,kind); save_state('C/origin_before',origin)
            output,state = call(f'C_step_{t}',clone_state(state,kind),t)
            sequence.append(output[:,0].clone()); packed.append(pack(state).clone())
        save('C/hidden_sequence',torch.stack(sequence,dim=1)); save('C/complete_sequence',torch.stack(packed,dim=1))
        save_state('C/terminal',state); save('C/logits',model.head(output[:,0])); finite('C/logits')
        origin_packed = pack(origin).clone()
        effects = {}; construction = {}
        for name in planned[18:]:
            supplied = clone_state(origin,kind)
            requested = torch.zeros_like(origin_packed)
            target_component = None
            if name == 'zero_displacement_query':
                parts = [v + torch.zeros_like(v) for v in state_components(supplied,kind).values()]
                supplied = tuple(parts) if kind == 'lstm' else parts[0]
            elif name not in ('copy_query',):
                component,sign = name.rsplit('_',1)
                target_component = component
                raw = directions[component]
                if raw.shape != (batch,model.hidden_size) or not np.isfinite(raw).all():
                    raise ValueError('invalid predetermined direction')
                direction = torch.as_tensor(raw,dtype=inputs.dtype)
                norms = np.linalg.norm(snapshot(direction).astype(np.float64),axis=1)
                if np.any(np.abs(norms-1)>1e-6) or np.any(norms<=0): raise ValueError('invalid raw unit direction')
                save(name+'/direction',direction); save(name+'/direction_norm',torch.from_numpy(norms.copy()))
                step = direction*torch.tensor(1e-3,dtype=inputs.dtype)*(1 if sign=='plus' else -1)
                offset = model.hidden_size if component=='cell' else 0
                requested[:,offset:offset+model.hidden_size] = step
                state_components(supplied,kind)[component].add_(step[None])
            displacement = pack(supplied)-origin_packed
            save(name+'/requested_displacement',requested)
            save(name+'/displacement',displacement)
            construction[name] = {component:{'targeted':component==target_component,
                **compare(snapshot(value[0]), arrays['C/origin_before/'+component],dtype,bitwise=True)}
                for component,value in state_components(supplied,kind).items()}
            armout, armstate = call(name,supplied,16)
            save(name+'/logits',model.head(armout[:,0])); finite(name+'/logits')
            if name in ('copy_query','zero_displacement_query'):
                edge(name+'/logits','C/logits')
                for component in state_components(armstate,kind): edge(name+'/terminal/'+component,'C/terminal/'+component)
            else: effects[name] = perturbation_effect(arrays[name+'/logits'],arrays['C/logits'],arrays[name+'/displacement'])
        save_state('C/origin_after',origin)
        for component in state_components(origin,kind): edge('C/origin_after/'+component,'C/origin_before/'+component,True)
        for field in ['logits','hidden_sequence'] + ['terminal/'+c for c in state_components(state,kind)]: edge('C/'+field,'B0/'+field)
        for key,value in arrays.items():
            if value.shape[0] != batch: raise ValueError('required row-first export changed')
            checks.append({'candidate':key+'/duplicate_half','reference':key+'/first_half',
                **compare(value[batch//2:],value[:batch//2],dtype)})
        if core_count != len(planned): raise StopCalibration('incomplete condition native registry')
        for k,v in model.state_dict().items():
            if not compare(snapshot(v),weight_before[k],dtype,bitwise=True)['passed']:
                raise StopCalibration('parameter identity mutated')
        return {'condition':condition,'layouts':layouts,'checks':checks,'effects':effects,
            'constructed_state_vs_origin':construction,
            'behavior':behavior_bounds(arrays['C/logits'],arrays['B0/logits'],model.task,dtype),
            'B0_intermediate_cell':None,'B0_intermediate_cell_reason':'unobserved_fused_LSTM' if kind=='lstm' else 'not_applicable',
            'state_packing':'hidden_then_cell' if kind=='lstm' else 'hidden',
            'selected_rows':list(range(batch)),'scientific_acceptance':False,'core_calls':core_count}
    except BaseException as error:
        if budget.entries and not budget.entries[-1]['completed']:
            budget.entries[-1].update(exception=type(error).__name__,end_elapsed=time.monotonic()-budget.started)
            budget.persist()
        raise
    finally:
        for handle in handles: handle.remove()


def verify_pin(root, pin):
    relative = Path(pin['path'])
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('project-relative pin required')
    path = root/relative
    if not path.is_file() or sha(path) != pin['sha256'] or path.stat().st_size != pin['bytes']:
        raise ValueError('pinned file mismatch: '+str(relative))
    return path


def validate_admission_fields(freeze, config_path, output):
    required = {'kind':'cached_forward_v2_root_admission', 'owner':'root coordinator',
                'execution_authorized':True, 'scientific_acceptance':False, 'training_authorized':False,
                'native_core_calls':736, 'wall_seconds':600, 'retries':0}
    for key,value in required.items():
        if type(freeze.get(key)) is not type(value) or freeze[key] != value:
            raise ValueError('external root-only numerical admission required: '+key)
    if freeze.get('config_sha256') != CONFIG_SHA256 or sha(config_path) != CONFIG_SHA256:
        raise ValueError('final config hash mismatch')
    if freeze.get('output') != str(Path(output).resolve()): raise ValueError('unique output freeze mismatch')
    commit = freeze.get('source_commit','')
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('full source commit required')
    stop = freeze.get('stop_path')
    if not isinstance(stop,str) or not Path(stop).is_absolute(): raise ValueError('absolute owner STOP path required')


def verify_admission(root, config_path, freeze_path, output):
    """Verify every gate before checkpoint deserialization or input generation.

    Required external receipt schema is explicit here. The root prepares it
    only after source publication and independent review; this module cannot.
    """
    root = Path(root).resolve(); config_path = Path(config_path).resolve(); freeze_path = Path(freeze_path).resolve()
    if config_path != root/'configs/experiments/cached-state-forward-calibration-v2.json':
        raise ValueError('canonical reviewed config path required')
    if freeze_path == config_path or freeze_path.is_relative_to(root/'src') or freeze_path.is_relative_to(root/'configs/experiments'):
        raise ValueError('external admission receipt required')
    freeze = json.loads(freeze_path.read_text()); validate_admission_fields(freeze,config_path,output)
    config = json.loads(config_path.read_text())
    if config['execution_authorized'] is not False or config['scientific_acceptance'] is not False:
        raise ValueError('proposal config must remain non-authorizing')
    if config['protocol_document']['sha256'] != CONTRACT_SHA256: raise ValueError('contract changed')
    verify_pin(root,config['protocol_document'])
    sources = freeze.get('source_files',{})
    required = {ENGINE_PATH, LAUNCHER_PATH, 'src/representation_collapse/reporting.py', 'uv.lock',
                'pyproject.toml','tests/test_cached_forward_calibration.py','code/ops/wandb_smoke.py'}
    if not required.issubset(sources): raise ValueError('complete implementation/reporting/source freeze required')
    for file,digest in sources.items():
        if Path(file).is_absolute() or '..' in Path(file).parts or sha(root/file) != digest:
            raise ValueError('implementation source pin mismatch')
    if Path(__file__).resolve() != root/ENGINE_PATH: raise ValueError('wrong imported driver source')
    evidence = {}; evidence_paths = {}
    for role,field in [('implementation_review','implementation_preparation_accepted'),
                       ('fixture_receipt','fixtures_passed'),('offline_smoke','completed')]:
        pin = freeze.get(role)
        if not isinstance(pin,dict): raise ValueError('missing '+role+' receipt')
        p = verify_pin(root,pin); record = json.loads(p.read_text())
        if record.get(field) is not True: raise ValueError(role+' is not accepted/completed')
        evidence[role] = record
        evidence_paths[role] = p
    review = evidence['implementation_review']
    if review.get('config_sha256') != CONFIG_SHA256 or review.get('source_files') != sources or review.get('scientific_acceptance') is not False:
        raise ValueError('independent review must bind exact source/config with no science acceptance')
    fixture = evidence['fixture_receipt']
    if fixture.get('native_core_attempts',201)>200 or fixture.get('wall_seconds',121)>120 or fixture.get('production_calls',1)!=0:
        raise ValueError('fixture scope proof missing/exceeded')
    if any(fixture.get('source_files',{}).get(file) != sources[file] for file in
           (ENGINE_PATH,LAUNCHER_PATH,'tests/test_cached_forward_calibration.py')):
        raise ValueError('fixture source mismatch')
    for file,digest in fixture.get('files',{}).items():
        if Path(file).name != file or sha(evidence_paths['fixture_receipt'].parent/file) != digest:
            raise ValueError('fixture evidence file mismatch')
    smoke = evidence['offline_smoke']
    if smoke.get('mode') != 'offline' or smoke.get('remote_history_verified') is not False or not smoke.get('stream_verification'):
        raise ValueError('durable verified offline lifecycle proof required')
    receipt_path = evidence_paths['offline_smoke'].parent/'reporting-receipt.json'
    if sha(receipt_path) != smoke.get('receipt_sha256'): raise ValueError('offline lifecycle receipt changed')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('sdk_version') != '0.30.0' or receipt.get('state') != 'completed' or
        receipt.get('identities',{}).get('reporter_source_sha256') != sources['src/representation_collapse/reporting.py'] or
        receipt.get('identities',{}).get('smoke_source_sha256') != sources['code/ops/wandb_smoke.py']):
        raise ValueError('offline lifecycle SDK/source mismatch')
    for stream in smoke.get('transaction_streams',[]):
        if sha(stream['path']) != stream['sha256'] or Path(stream['path']).stat().st_size != stream['bytes']:
            raise ValueError('offline transaction stream changed')
    if not smoke.get('transaction_streams') or any(v.get('history_records')!=2 or not all(v.get(k) is True for k in
        ('config_verified','completed_summary_verified','automatic_host_environment_code_files_absent',
         'metadata_artifact_record_present','metadata_artifact_bytes_in_durable_run_tree_verified')) for v in smoke['stream_verification']):
        raise ValueError('offline lifecycle content verification incomplete')
    # Scientific/native source, dependency lock, binaries and checkpoint bytes.
    for pin in config['source']['native_files'] + [config['source']['uv_lock']] + config['runtime']['native_file_pins']:
        verify_pin(root,pin)
    for context in config['contexts']: verify_pin(root,context['checkpoint'])
    current = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=root,text=True).strip()
    if current != freeze['source_commit'] or dirty: raise ValueError('clean committed source required')
    for file in required | {'configs/experiments/cached-state-forward-calibration-v2.json'}:
        if subprocess.check_output(['git','show',current+':'+file],cwd=root) != (root/file).read_bytes():
            raise ValueError('committed source/config mismatch')
    return config,freeze


def initialize_runtime(config):
    import torch
    from .pilot import VectorMemoryModel
    from . import pilot, vector_task
    root = Path(__file__).resolve().parents[2]
    if Path(pilot.__file__).resolve() != root/'src/representation_collapse/pilot.py' or Path(vector_task.__file__).resolve() != root/'src/representation_collapse/vector_task.py':
        raise ValueError('wrong imported scientific source')
    for k,v in config['runtime']['environment_threads'].items():
        if os.environ.get(k) != v: raise ValueError('one CPU environment not set before imports')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    actual = {'torch_version':str(torch.__version__),'numpy_version':np.__version__,
        'python_version':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
        'cpu_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'mkldnn_available':torch.backends.mkldnn.is_available(),'mkldnn_enabled':torch.backends.mkldnn.enabled,
        'cudnn_enabled':torch.backends.cudnn.enabled,'torch_build_configuration':torch.__config__.show()}
    for key,value in actual.items():
        if config['runtime'][key] != value: raise ValueError('native runtime identity mismatch: '+key)
    if sha(torch.nn.modules.rnn.__file__) != config['runtime']['native_python_rnn_source_sha256']:
        raise ValueError('native RNN implementation mismatch')
    return torch,VectorMemoryModel,actual


def production_inputs(task):
    from .vector_task import render
    rng = np.random.Generator(np.random.PCG64(30019))
    memory = (rng.integers(0,16,size=(32,16),dtype=np.int64) if task=='categorical' else
              rng.uniform(-1,1,size=(32,16)).astype(np.float32))
    memory = np.concatenate([memory,memory],axis=0)
    return render(memory,np.arange(64,dtype=np.int64)%8,['']*64,task)


def production_directions(contexts):
    rng = np.random.Generator(np.random.PCG64(30029)); result = {}
    for c in contexts:
        result[c['id']] = {}
        for component in c['directions']:
            raw = rng.standard_normal(size=(32,128),dtype=np.float64)
            norms = np.linalg.norm(raw,axis=1,keepdims=True)
            if np.any(norms==0) or not np.isfinite(norms).all(): raise StopCalibration('invalid direction draw; no redraw')
            unit = raw/norms; result[c['id']][component] = np.concatenate([unit,unit],axis=0)
    return result


def _weights_hash(values):
    digest = hashlib.sha256()
    for name,value in sorted(values.items()):
        digest.update(name.encode()); digest.update(array_hash(snapshot(value)).encode())
    return digest.hexdigest()


def production(config_path, freeze_path, root, output, started):
    """Only the deadline launcher may invoke this; every gate fails closed."""
    root = Path(root).resolve(); output = Path(output).resolve()
    if os.environ.get('RC_CACHED_FORWARD_LAUNCH') != '1': raise ValueError('external deadline launcher required')
    if not np.isfinite(started) or not 0 <= time.monotonic()-started < 600: raise ValueError('invalid shared clock')
    config,freeze = verify_admission(root,config_path,freeze_path,output)
    if freeze.get('started_monotonic') != started: raise ValueError('root admission shared clock mismatch')
    if output.exists(): raise ValueError('output must be unique and absent')
    output.mkdir(parents=True)
    budget = CallBudget(736,started,600,10,Path(freeze['stop_path']),output/'native-call-registry.json')
    records = []; error = None; reporting_dir = output/'reporting'; current = None
    try:
        budget.check(native=True)
        torch,Model,environment = initialize_runtime(config)
        atomic_json(output/'environment.json',environment)
        atomic_json(output/'config.json',config)
        atomic_json(output/'source-freeze.json',freeze)
        from .reporting import ExperimentReporter
        with ExperimentReporter(reporting_dir,project='representation-collapse',entity=None,mode='offline',
                config={'protocol_id':PROTOCOL,'cpu_threads':1,'native_calls_planned':736,'scientific_acceptance':False,
                        'source_commit':freeze['source_commit'],'input_seed':30019,'direction_seed':30029},
                identities={'config_sha256':CONFIG_SHA256,'engine_sha256':sha(__file__),
                            'lock_sha256':sha(root/'uv.lock'),'freeze_sha256':sha(freeze_path),
                            'environment_sha256':sha(output/'environment.json')}) as reporter:
            budget.check(native=True)
            input_by_task = {task:production_inputs(task) for task in ('categorical','continuous')}
            directions = production_directions(config['contexts'])
            input_paths = {}
            for task,inputs in input_by_task.items():
                p = output/f'input-{task}.npz'; atomic_npz(p,{'inputs':inputs,'row_ids':np.arange(64),
                    'history_ids':np.arange(64)%32,'query_ids':np.arange(64)%8}); input_paths[task]=sha(p)
            dp = output/'canonical-directions.npz'; atomic_npz(dp,{context+'__'+component:value for context,parts in directions.items() for component,value in parts.items()})
            reporter.log_manifest('calibration-input-identities',metadata={'rows':64,'independent_histories':32,'input_seed':30019,'direction_seed':30029},
                                  identities={**{'inputs_'+k:v for k,v in input_paths.items()},'directions':sha(dp),
                                      **{'checkpoint_'+str(i):c['checkpoint']['sha256'] for i,c in enumerate(config['contexts'])}})
            plan = condition_registry(config); atomic_json(output/'condition-registry.json',plan)
            contexts = {c['id']:c for c in config['contexts']}; saved = {}
            for index,condition in enumerate(plan):
                budget.check(native=True); current = condition['id']; context = contexts[condition['context']]
                # Fresh load for each repeat/dtype; only the declared state_dict is consumed.
                checkpoint_path = verify_pin(root,context['checkpoint'])
                loaded = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
                weights = loaded['model_state_dict']; del loaded
                if not isinstance(weights,dict) or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in weights.values()):
                    raise ValueError('finite stored FP32 model state required')
                base_hash = _weights_hash(weights)
                original_default=torch.get_default_dtype(); torch.set_default_dtype(torch.float32)
                try:model = Model(context['model'],context['task'],128,64)
                finally:torch.set_default_dtype(original_default)
                model.load_state_dict(weights,strict=True)
                dtype=getattr(torch,condition['dtype']); model=model.to(dtype=dtype,device='cpu').eval()
                if not all(torch.equal(model.state_dict()[k],v.to(dtype=dtype)) for k,v in weights.items()):
                    raise ValueError('exact parameter promotion failed')
                tensor=torch.tensor(input_by_task[context['task']],dtype=dtype,device='cpu')
                if not np.array_equal(snapshot(tensor),input_by_task[context['task']].astype(str(dtype).split('.')[-1])):
                    raise ValueError('exact input promotion failed')
                arrays={}; record=None
                try:
                    with torch.no_grad(): record=capture_condition(model,tensor,directions[context['id']],condition,budget,arrays)
                finally:
                    atomic_npz(output/(current+'.npz'),arrays)
                record.update(arrays_sha256=sha(output/(current+'.npz')),base_fp32_weight_hash=base_hash,
                    parameter_promotion_exact=True,input_promotion_exact=True,checkpoint_sha256=context['checkpoint']['sha256'])
                # Full saved operand comparisons; no native work for repeat/cross-dtype edges.
                references=[]
                if condition['repeat']==2: references.append(('repeat',saved[(condition['context'],condition['dtype'],1)]))
                if condition['dtype']=='float32': references.append(('cross_dtype',saved[(condition['context'],'float64',condition['repeat'])]))
                for relation,reference_id in references:
                    with np.load(output/(reference_id+'.npz'),allow_pickle=False) as reference:
                        if set(reference.files)!=set(arrays): raise ValueError('saved reference observable set differs')
                        for field,value in arrays.items():
                            record['checks'].append({'relation':relation,'candidate':current+'/'+field,'reference':reference_id+'/'+field,
                                **compare(value,reference[field],condition['dtype'],bitwise=relation=='repeat')})
                saved[(condition['context'],condition['dtype'],condition['repeat'])]=current
                atomic_json(output/(current+'.json'),record); records.append({'id':current,'completed':True,
                    'failed_comparisons':sum(not v['passed'] for v in record['checks'])})
                reporter.log({'condition_id':current,'context':condition['context'],'model':condition['model'],
                    'task':condition['task'],'dtype':condition['dtype'],'repeat':condition['repeat'],
                    'calibration/conditions_completed':len(records),'calibration/core_attempted':budget.attempted,
                    'calibration/core_completed':budget.completed,'calibration/failed_comparisons':records[-1]['failed_comparisons'],
                    'resources/elapsed_seconds':time.monotonic()-started,'resources/cpu_seconds':time.process_time(),
                    'resources/max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},step=index)
                del weights,model,tensor,arrays
            if (budget.attempted,budget.completed,budget.encoder_calls,budget.head_calls)!=(736,736,64,224):
                raise StopCalibration('terminal physical call registry mismatch')
            budget.check()
            budget.persist()  # include the final head count before hashing the registry
            manifest={str(p.relative_to(output)):sha(p) for p in sorted(output.glob('*')) if p.is_file()}
            atomic_json(output/'raw-manifest.json',manifest)
            reporter.log_manifest('calibration-completion',metadata={'conditions_completed':32,'core_calls':736,'scientific_acceptance':False},
                identities={'raw_manifest_sha256':sha(output/'raw-manifest.json'),'config_sha256':CONFIG_SHA256})
        budget.check()  # tracking finalization consumes the same envelope
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        budget.persist()
        state='completed' if error is None and len(records)==32 and time.monotonic()-started<600 else 'incomplete'
        atomic_json(output/'status.json',{'protocol_id':PROTOCOL,'state':state,'error_type':error,'current_condition':current,
            'conditions':records,'native_attempted':budget.attempted,'native_completed':budget.completed,
            'encoder_calls':budget.encoder_calls,'head_calls':budget.head_calls,'elapsed_seconds':time.monotonic()-started,
            'fits':0,'targets_read':False,'gradients':0,'scientific_acceptance':False,'sync_pending':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','freeze','output'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--started-monotonic',required=True,type=float)
    args=parser.parse_args()
    production(args.config,args.freeze,args.root,args.output,args.started_monotonic)


if __name__=='__main__': main()
