"""Hidden-only native T-GCN backend and source-derived parameter expectations.

No Torch/PyG import, model construction or checkpoint decoding occurs on import.
Callers must authenticate the supplied upstream wrapper/cell/head and initialize
reporting before construction. Native parity is a separate admission gate. Tiny
explicit layouts support synthetic calibration; production is one layer, N=100,
H=128, CPU FP32. This is not the GC-LSTM H+C state or checkpoint format.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

PUBLIC_COMMIT = "31293102b065feef7817ff86211427d7594edfdb"
WRAPPER_SHA256 = "028a85f0527833a7e62099e8bb826ac144df29738ee120795d205d96a8c6f78c"
CELL_SHA256 = "1f417ad10b1fb3ed6af24731efb0e80b614495a3e4df3fc16254e631c3c8819e"
HEAD_SHA256 = "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86"


class TGCNError(ValueError):
    pass


def require(value, message):
    if not value: raise TGCNError(message)


@dataclass(frozen=True)
class TGCNLayout:
    nodes: int = 100
    hidden: int = 128

    def __post_init__(self):
        require(type(self.nodes) is int and 2 <= self.nodes <= 100 and type(self.hidden) is int
                and 1 <= self.hidden <= 128, "bounded positive native calibration/production layout required")

    @property
    def shape(self): return (1, self.nodes, self.hidden)
    @property
    def node_ids(self): return tuple(range(self.nodes))
    @property
    def layers(self): return 1
    @property
    def joint_dimension(self): return self.nodes * self.hidden


@dataclass(frozen=True)
class HiddenOnlyState:
    hidden: Any
    next_target_t: int
    node_ids: tuple[int, ...]

    def validate(self, layout):
        require(tuple(self.hidden.shape) == layout.shape, "hidden-only state shape mismatch")
        require(type(self.node_ids) is tuple and all(type(n) is int for n in self.node_ids)
                and self.node_ids == layout.node_ids, "canonical state node ordering required")
        require(type(self.next_target_t) is int and self.next_target_t >= 0, "integer state timestamp required")


def source_derived_catalog(*, nodes=100, hidden=128):
    """Expected names/shapes from pinned source; not a native inventory receipt."""
    layout = TGCNLayout(nodes, hidden)
    recurrent = []
    for gate in ('z', 'r', 'h'):
        recurrent += [(f'units.0.conv_{gate}.bias', [hidden]),
            (f'units.0.conv_{gate}.lin.weight', [hidden, nodes]),
            (f'units.0.linear_{gate}.weight', [hidden, 2*hidden]),
            (f'units.0.linear_{gate}.bias', [hidden])]
    head = [('seq.1.weight', [hidden, 2*hidden]), ('seq.1.bias', [hidden]),
        ('seq.3.weight', [hidden, hidden]), ('seq.3.bias', [hidden]),
        ('seq.5.weight', [1, hidden]), ('seq.5.bias', [1])]
    components = {}
    for key, backend_key, entries in (('recurrent', 'node_emb', recurrent), ('link_head', 'link_pred', head)):
        offset = 0; rows = []
        for name, shape in sorted(entries):
            count = math.prod(shape)
            rows.append({'name': name, 'shape': shape, 'dtype': 'float32_le', 'order': 'C',
                'offset': offset, 'nbytes': count*4, 'numel': count})
            offset += count*4
        components[key] = {'backend_key': backend_key, 'native_state_dict_order': [name for name, _ in entries],
            'serialization_order': [r['name'] for r in rows], 'tensors': rows, 'parameter_count': len(rows),
            'numel': offset//4, 'nbytes': offset}
    return {'kind': 'tgcn_source_derived_parameter_catalog_v1', 'production_layout': layout.shape == (1,100,128),
        'layout': list(layout.shape), 'state_components': ['hidden'], 'components': components,
        'total_parameters': sum(c['numel'] for c in components.values()),
        'source_sha256': {'wrapper': WRAPPER_SHA256, 'cell': CELL_SHA256, 'head': HEAD_SHA256},
        'native_inventory_verified': False}


def construct_models(wrapper_type, parameter_type, head_type, *, layout=TGCNLayout()):
    """Construct supplied, externally authenticated actual upstream types.

    Fresh seeding, source imports, optimizer construction and reporting belong to
    the caller; this helper makes no RNG or source-authentication claim.
    """
    params = parameter_type(layout.nodes, layout.hidden, improved=False, cached=False)
    return {'node_emb': wrapper_type(1, params), 'link_pred': head_type(layout.hidden, num_layers=2, p_dropout=0.2)}


class NativeTGCNBackend:
    def __init__(self, torch_module, models, optimizer=None, *, device='cpu', layout=TGCNLayout()):
        require(device == 'cpu' and isinstance(layout, TGCNLayout), "CPU hidden-only layout required")
        require(set(models) == {'node_emb', 'link_pred'}, "both recurrent and link-head models required")
        self.torch, self.models, self.optimizer, self.device, self.layout = torch_module, models, optimizer, device, layout
        self.catalog = source_derived_catalog(nodes=layout.nodes, hidden=layout.hidden)
        recurrent = models['node_emb']
        require(recurrent.num_units == 1 and len(recurrent.units) == 1, "exactly one TGCN unit required")
        cell = recurrent.units[0]
        require(cell.__class__.__name__ == 'TGCN', "actual TGCN cell required, not TGCN2")
        for key, expected in {'in_channels': layout.nodes, 'out_channels': layout.hidden,
                              'improved': False, 'cached': False, 'add_self_loops': False}.items():
            value = getattr(cell, key, None)
            require(type(value) is type(expected) and value == expected, "cell architecture mismatch: " + key)
        for gate in ('z', 'r', 'h'):
            conv = getattr(cell, 'conv_'+gate)
            require(conv.cached is False and conv.add_self_loops is False and conv.improved is False
                    and conv.normalize is True and conv.flow == 'source_to_target',
                    "dynamic graph convolution flags differ")
            require(getattr(conv, '_cached_edge_index', None) is None and getattr(conv, '_cached_adj_t', None) is None,
                    "graph cache must be empty")
        self._validate_components({key: models[spec['backend_key']].state_dict() for key,spec in self.catalog['components'].items()})
        self._validate_model_storage()
        self.features = self.torch.eye(layout.nodes, dtype=self.torch.float32, device='cpu')

    def _tensor(self, value, shape):
        require(isinstance(value, self.torch.Tensor) and tuple(value.shape) == tuple(shape)
                and value.dtype == self.torch.float32 and str(value.device) == 'cpu', "finite CPU FP32 tensor with exact shape required")
        require(bool(self.torch.isfinite(value).all()), "nonfinite native tensor")

    def _validate_components(self, components):
        require(set(components) == {'recurrent', 'link_head'}, "both native component state dicts required")
        for key,spec in self.catalog['components'].items():
            values = components[key]
            require(list(values) == spec['native_state_dict_order'], "native component names/order differ from source catalog")
            for item in spec['tensors']: self._tensor(values[item['name']], item['shape'])

    def _validate_storage(self, components):
        pointers=set(); objects=set()
        for values in components.values():
            for value in values.values():
                storage=value.untyped_storage();pointer=storage.data_ptr()
                require(id(value) not in objects and pointer not in pointers and pointer != 0
                        and value.storage_offset()==0 and value.is_contiguous()
                        and storage.nbytes()==value.numel()*4,
                        'aliased, overlapping or noncompact tensor storage forbidden')
                objects.add(id(value));pointers.add(pointer)

    def _validate_model_storage(self):
        registered={}
        for key,spec in self.catalog['components'].items():
            entries=list(self.models[spec['backend_key']].named_parameters(remove_duplicate=False))
            require([name for name,_ in entries]==spec['native_state_dict_order'],
                    'native registered parameter inventory/order differs')
            registered[key]=dict(entries)
        self._validate_storage(registered)
        self._validate_storage({key:self.models[spec['backend_key']].state_dict() for key,spec in self.catalog['components'].items()})

    def _f32le_bytes(self,value):
        self._tensor(value,tuple(value.shape))
        return value.detach().cpu().contiguous().numpy().astype('<f4',copy=False).tobytes(order='C')

    def _state(self, state):
        require(isinstance(state, HiddenOnlyState), "dedicated hidden-only state required; no cell substitution")
        state.validate(self.layout); self._tensor(state.hidden, self.layout.shape)

    @contextmanager
    def _mode(self, training):
        saved = [(module, module.training) for model in self.models.values() for module in model.modules()]
        try:
            for model in self.models.values(): model.train(training)
            with self.torch.set_grad_enabled(training): yield
        finally:
            for module, flag in saved: module.training = flag

    def training_mode(self): return self._mode(True)
    def evaluation_mode(self): return self._mode(False)

    def reset(self):
        return HiddenOnlyState(self.torch.zeros(self.layout.shape,dtype=self.torch.float32,device='cpu'),0,self.layout.node_ids)

    def detach(self, state):
        self._state(state)
        return HiddenOnlyState(state.hidden.detach().clone(),state.next_target_t,state.node_ids)

    def consume(self, state, snapshot, *, edge_weight=None):
        self._state(state)
        require(snapshot.nodes == self.layout.nodes and snapshot.node_ids == self.layout.node_ids
                and snapshot.timestamp == state.next_target_t, "input/state timestamp or node ordering mismatch")
        pairs = snapshot.positive_pairs
        require(tuple(sorted(set(pairs))) == pairs and all(type(u) is int and type(v) is int and 0 <= u < self.layout.nodes
                and 0 <= v < self.layout.nodes and u != v for u,v in pairs), "sorted unique offdiagonal edges required")
        edgeset = set(pairs)
        require(all((v,u) in edgeset for u,v in pairs), "bidirectional undirected input required")
        edges = self.torch.tensor(pairs,dtype=self.torch.long,device='cpu').reshape(-1,2).T.contiguous()
        if edge_weight is None:
            weights = self.torch.ones(len(pairs),dtype=self.torch.float32,device='cpu')
        else:
            require(isinstance(edge_weight,(tuple,list)) and len(edge_weight)==len(pairs)
                    and all(type(w) in (float,int) and math.isfinite(w) and w>0 for w in edge_weight),
                    'finite positive edge weights in canonical pair order required')
            by_pair=dict(zip(pairs,edge_weight))
            require(all(by_pair[u,v]==by_pair[v,u] for u,v in pairs), 'undirected edge weights must match reverse pairs')
            weights=self.torch.tensor(edge_weight,dtype=self.torch.float32,device='cpu')
            self._tensor(weights,(len(pairs),))
        hidden,z = self.models['node_emb'](self.features,edges,state.hidden,edge_weight=weights)
        result = HiddenOnlyState(hidden,snapshot.timestamp+1,snapshot.node_ids); self._state(result)
        self._tensor(z,(self.layout.nodes,self.layout.hidden))
        require(bool(self.torch.equal(z,hidden[-1])), "upstream output must equal last-layer hidden state")
        return result

    def capture_state(self, state):
        """Independent detached H-only diagnostic capture; never an RNG resume."""
        return self.detach(state)

    def _probabilities(self, state, pairs):
        self._state(state)
        require(bool(pairs) and all(type(u) is int and type(v) is int and 0 <= u < self.layout.nodes
                and 0 <= v < self.layout.nodes and u != v for u,v in pairs), "valid nonempty offdiagonal node pairs required")
        indices = self.torch.tensor(pairs,dtype=self.torch.long,device='cpu').reshape(-1,2)
        p = self.models['link_pred'](state.hidden[-1,indices[:,0]],state.hidden[-1,indices[:,1]]).reshape(-1)
        self._tensor(p,(len(pairs),))
        require(bool(((p>=0)&(p<=1)).all()), "native link probabilities outside [0,1]")
        return p

    def score_pairs(self,state,pairs): return self._probabilities(state,pairs).detach().cpu().numpy()

    def loss_and_update(self,state,target,negatives):
        from .graph_reconstruction import validate_negatives
        self._state(state)
        require(self.optimizer is not None and state.next_target_t == target.timestamp
                and target.node_ids == self.layout.node_ids and target.nodes == self.layout.nodes, "optimizer and causal target required")
        validate_negatives(target,negatives)
        require(bool(target.positive_pairs), "nonempty positive training target required")
        self.optimizer.zero_grad(set_to_none=True)
        positive=self._probabilities(state,target.positive_pairs);negative=self._probabilities(state,negatives)
        loss=self.torch.nn.functional.binary_cross_entropy(positive,self.torch.ones_like(positive))
        loss=loss+self.torch.nn.functional.binary_cross_entropy(negative,self.torch.zeros_like(negative))
        require(bool(self.torch.isfinite(loss)), "nonfinite BCE loss")
        loss.backward()
        for model in self.models.values():
            for p in model.parameters():
                require(p.grad is None or bool(self.torch.isfinite(p.grad).all()), "nonfinite gradient before optimizer step")
        self.optimizer.step()
        self._validate_components({key:self.models[spec['backend_key']].state_dict() for key,spec in self.catalog['components'].items()})
        return float(loss.detach().cpu())

    def capture_components(self):
        self._validate_model_storage()
        self._validate_components({key:self.models[spec['backend_key']].state_dict() for key,spec in self.catalog['components'].items()})
        return {key:{name:value.detach().clone() for name,value in self.models[spec['backend_key']].state_dict().items()}
                for key,spec in self.catalog['components'].items()}

    def load_components(self, components):
        """Strict inference-only load of both validated dictionaries; no codec claim.

        All inputs are validated before either model changes. A later native
        mutation/readback failure is terminal to the caller; no rollback/resume.
        """
        self._validate_components(components)
        self._validate_storage(components)
        self._validate_components({key:self.models[spec['backend_key']].state_dict() for key,spec in self.catalog['components'].items()})
        self._validate_model_storage()
        copies={key:{name:value.detach().clone() for name,value in values.items()} for key,values in components.items()}
        for key,spec in self.catalog['components'].items():
            model=self.models[spec['backend_key']]
            report=model.load_state_dict(copies[key],strict=True)
            require(not report.missing_keys and not report.unexpected_keys, "strict dual-component load mismatch")
            require(all(self._f32le_bytes(value)==self._f32le_bytes(copies[key][name]) for name,value in model.state_dict().items()), "native FP32 byte readback differs")
            model.eval()
        self._validate_model_storage()

    def load_selected(self, reference):
        raise TGCNError("T-GCN checkpoint admission/codec not implemented; no implicit load or GC-LSTM format coercion")
