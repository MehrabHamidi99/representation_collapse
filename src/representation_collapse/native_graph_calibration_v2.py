"""One supervised, admitted native GC-LSTM engineering calibration.

Stdlib-only import. No native work is authorized by importing this module or
by its disabled config. Supervisor/worker share a pre-import physical deadline.
"""
from __future__ import annotations

import time
PROCESS_STARTED = time.monotonic()

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import runpy
import signal
import subprocess
import sys
from types import ModuleType
from typing import Callable

PROTOCOL_ID = "native_graph_calibration_v2"
FROZEN_CONFIG_CANONICAL_SHA256 = "c2e8490b954a5228f2f58f13c9b7e87526143fde80d4ab1774997485e9bf15b6"
MODULE_PATH = "src/representation_collapse/native_graph_calibration_v2.py"
CONFIG_PATH = "configs/graph/native-graph-calibration-v2.json"
PROTOCOL_PATH = "doc/auto/native-graph-calibration-v2.md"


class CalibrationError(RuntimeError):
    pass


def sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CalibrationError(f"regular source/receipt file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def frozen_config(path):
    config = json.loads(Path(path).read_text())
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if canonical != FROZEN_CONFIG_CANONICAL_SHA256:
        raise CalibrationError("frozen calibration config changed")
    return config


class Ledger:
    """Attempted entries counted before every declared API/native call."""
    def __init__(self, limits, guard: Callable[[], None]):
        self.limits = dict(limits)
        self.counts = {name: 0 for name in limits}
        self.guard = guard

    def charge(self, name):
        self.guard()
        if name not in self.limits or self.counts[name] >= self.limits[name]:
            raise CalibrationError(f"finite call cap: {name}")
        self.counts[name] += 1

    def complete(self):
        if self.counts != self.limits:
            raise CalibrationError("successful native/AD counts differ from frozen schedule")


def ordered_pairs(nodes=3):
    return tuple((u, v) for u in range(nodes) for v in range(nodes) if u != v)


def coordinate_permutation(new_to_old, hidden=2):
    if sorted(new_to_old) != list(range(len(new_to_old))):
        raise CalibrationError("node permutation is not bijective")
    nodes = len(new_to_old)
    inverse = [new_to_old.index(old) for old in range(nodes)]
    coordinates = [component * nodes * hidden + old * hidden + feature
                   for component in range(2) for old in new_to_old for feature in range(hidden)]
    return inverse, coordinates


def chronological_trace(inputs, state, *, consume, score, guard):
    """Native/spied wiring: t0 conditions; every score precedes its target."""
    events, predictions = [], []
    for t, graph in enumerate(inputs):
        guard()
        if t:
            predictions.append(score(state, ordered_pairs()))
            events.append(f"score_{t}")
        state = consume(state, graph)
        events.append(f"consume_{t}")
    return state, predictions, events


def _matmul(left, right):
    return [[sum(a * b for a, b in zip(row, column)) for column in zip(*right)] for row in left]


def analytic_order_reference():
    """Exact integer infrastructure reference; no numerical library/model."""
    a, b = [[1, 1], [0, 1]], [[1, 0], [1, 1]]
    return {"chronological": _matmul(b, a), "reversed": _matmul(a, b),
            "chronological_direction": [3, 5], "reversed_direction": [5, 3]}


def extend_real_operator(operator, value, *, combine):
    """Complex tangent extension only; the native state never becomes complex."""
    return combine(operator(value.real), operator(value.imag))


def kernel_memory_limit(proc_cgroup=Path("/proc/self/cgroup"), cgroup_root=Path("/sys/fs/cgroup")):
    """Read inherited kernel physical cap; virtual address space is unrelated."""
    rows = [line.split(":", 2) for line in proc_cgroup.read_text().splitlines()]
    candidates = []
    for hierarchy, controllers, relative in rows:
        if ".." in Path(relative).parts:
            raise CalibrationError("unsafe cgroup path")
        if hierarchy == "0" and controllers == "":
            base, filename = cgroup_root, "memory.max"
        elif "memory" in controllers.split(","):
            base, filename = cgroup_root / "memory", "memory.limit_in_bytes"
        else:
            continue
        current = base / relative.lstrip("/")
        while True:
            file = current / filename
            if file.is_file():
                value = file.read_text().strip()
                if value != "max":
                    limit = int(value)
                    if limit > 0:
                        candidates.append({"path": str(file), "physical_memory_limit_bytes": limit})
            if current == base:
                break
            current = current.parent
    if not candidates:
        raise CalibrationError("kernel physical memory cgroup cap unavailable")
    return min(candidates, key=lambda item: item["physical_memory_limit_bytes"])


def _pinned_json(record):
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise CalibrationError("receipt pin mismatch")
    return json.loads(path.read_text())


def verify_admission(args, config, *, worker=False):
    """No source execution, arrays, credentials or native imports here."""
    root = args.root.resolve(); admission = json.loads(args.admission.read_text())
    if Path(__file__).resolve() != (root / MODULE_PATH).resolve():
        raise CalibrationError("imported engine must come from the admitted published source root")
    expected = {"kind": "native_graph_calibration_root_admission", "owner": "root coordinator",
                "protocol_id": PROTOCOL_ID, "execution_authorized": True, "training_authorized": False,
                "scientific_acceptance": False, "retries": 0}
    if any(admission.get(key) != value for key, value in expected.items()):
        raise CalibrationError("explicit root calibration admission required")
    if admission.get("config_sha256") != sha256(args.config) or admission.get("finite_counts") != config["finite_counts"] or admission.get("resources") != config["resources"]:
        raise CalibrationError("admission config/count/resource mismatch")
    if admission.get("output") != str(args.output.resolve()) or admission.get("source_root") != str(args.source_root.resolve()):
        raise CalibrationError("admission output/source fixture path mismatch")
    stop = Path(admission["stop_path"])
    if not stop.is_absolute() or stop.exists():
        raise CalibrationError("absolute absent owner STOP required")
    if args.config.resolve() != (root / CONFIG_PATH).resolve():
        raise CalibrationError("v2 config must come from the admitted published source root")
    required = [MODULE_PATH, PROTOCOL_PATH, CONFIG_PATH,
                config["sources"]["ownership_helper"]["path"], config["sources"]["reporter"]["path"],
                config["sources"]["graph_import_helper"]["path"]]
    for relative in required:
        if admission.get("source_files", {}).get(relative) != sha256(root / relative):
            raise CalibrationError(f"published source identity mismatch: {relative}")
    for relative, expected_hash in config["sources"]["wrapper_files"].items():
        if sha256(args.source_root / relative) != expected_hash:
            raise CalibrationError(f"upstream wrapper bytes mismatch: {relative}")
    for record in (config["sources"]["ownership_helper"], config["sources"]["reporter"],
                   config["sources"]["graph_import_helper"]):
        if sha256(root / record["path"]) != record["sha256"]:
            raise CalibrationError("helper/reporter/import-helper source pin mismatch")
    for key in ("implementation_review", "offline_reporting_readiness"):
        value = _pinned_json(admission[key])
        if value.get("status") != "verified" or value.get("config_sha256") != sha256(args.config):
            raise CalibrationError(f"verified config-bound {key} receipt required")
    environment = _pinned_json(admission["environment"])
    if environment.get("status") != "verified" or environment.get("python_major_minor") != [3, 10] or environment.get("platform") != "linux-x86_64":
        raise CalibrationError("isolated verified legacy CPython3.10/LinuxCPU environment required")
    if any(environment.get("versions", {}).get(name) != version for name, version in config["required_distributions"].items()):
        raise CalibrationError("exact CPU-build environment distribution versions required")
    if not environment.get("lock_sha256") or not environment.get("artifact_manifest_sha256"):
        raise CalibrationError("environment lock/artifact identities required")
    if not worker:
        # Preserve the venv executable path: resolving its symlink can bypass
        # the isolated site-packages when a coordinator reuses this argv.
        actual = [str(Path(sys.executable).absolute()), "-m", "representation_collapse.native_graph_calibration_v2", *sys.argv[1:]]
        if admission.get("supervisor_argv") != actual:
            raise CalibrationError("root admission must bind exact supervisor argv")
    else:
        if admission.get("started_monotonic") != args.started_monotonic:
            raise CalibrationError("worker and supervisor clocks must share root binding")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=2).strip()
    if admission.get("source_commit") != head:
        raise CalibrationError("published source commit mismatch")
    return admission


def _runtime_preflight(config):
    if sys.version_info[:2] != (3, 10) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise CalibrationError("native environment must be isolated CPython3.10/Linux-x86_64")
    versions = {name: importlib.metadata.version(name) for name in config["required_distributions"]}
    if versions != config["required_distributions"]:
        raise CalibrationError("actual installed distribution/build tags differ from frozen CPU environment")
    dist = importlib.metadata.distribution("torch-geometric-temporal")
    cell_path = Path(dist.locate_file("torch_geometric_temporal/nn/recurrent/gc_lstm.py"))
    if sha256(cell_path) != config["sources"]["cell_sha256"]:
        raise CalibrationError("installed GC-LSTM source bytes differ from exact pin")
    memory = kernel_memory_limit()
    if memory["physical_memory_limit_bytes"] > config["resources"]["memory_bytes"]:
        raise CalibrationError("actual inherited physical cgroup exceeds approved2GiB cap")
    return {"versions": versions, "installed_cell_path": str(cell_path), "memory": memory,
            "python": platform.python_version(), "native_packages_imported": False}


def _execute_pinned_helper(path, data):
    """Execute verified helper source once, without reading cached bytecode."""
    namespace = "_native_graph_calibration_v2_import_helper"
    if namespace in sys.modules:
        raise CalibrationError("import-helper namespace already loaded; fresh worker required")
    module = ModuleType(namespace)
    module.__file__ = str(path)
    module.__package__ = "representation_collapse"
    sys.modules[namespace] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        if sys.modules.get(namespace) is module:
            del sys.modules[namespace]
        raise
    return module


def _load_wrappers(source_root, config, *, root, installed_cell_path, guard, provenance):
    """Load exact cell/wrappers inside reporting; full Temporal API stays unverified."""
    guard()
    if not isinstance(sys.modules.get("torch"), ModuleType):
        raise CalibrationError("Torch must already be loaded after mandatory reporting")
    record = config["sources"]["graph_import_helper"]
    helper_path = Path(root) / record["path"]
    if sha256(helper_path) != record["sha256"]:
        raise CalibrationError("import-helper source pin mismatch before execution")
    helper_bytes = helper_path.read_bytes()
    if hashlib.sha256(helper_bytes).hexdigest() != record["sha256"]:
        raise CalibrationError("import-helper bytes changed before execution")
    guard()
    helper = _execute_pinned_helper(helper_path.resolve(), helper_bytes)
    guard()
    if Path(helper.__file__).resolve() != helper_path.resolve() or sha256(helper_path) != record["sha256"]:
        raise CalibrationError("executed import-helper path/source mismatch")
    bundle = helper.load_pinned_graph_modules(source_root=Path(source_root),
        installed_cell_path=Path(installed_cell_path), cell_sha256=config["sources"]["cell_sha256"],
        wrapper_sha256=config["sources"]["wrapper_files"], guard=guard)
    guard()
    if bundle.provenance.get("source_release_hashes_match") is not True:
        raise CalibrationError("production isolated import must match actual released cell/wrapper hashes")
    if (bundle.recurrent.GCLSTM is not bundle.cell_type or bundle.cell.GCLSTM is not bundle.cell_type
            or bundle.recurrent.MultiLayerGCLSTM is not bundle.wrapper_type
            or bundle.recurrent.GCLSTMParam is not bundle.parameter_type
            or bundle.head.LinkPredictor is not bundle.head_type):
        raise CalibrationError("isolated imported class identity mismatch")
    inspect = importlib.import_module("inspect")
    checks = ((bundle.cell_type, Path(installed_cell_path), config["sources"]["cell_sha256"]),
              (bundle.wrapper_type, Path(source_root) / "model/gclstm.py", config["sources"]["wrapper_files"]["model/gclstm.py"]),
              (bundle.parameter_type, Path(source_root) / "model/gclstm.py", config["sources"]["wrapper_files"]["model/gclstm.py"]),
              (bundle.head_type, Path(source_root) / "model/link_predictor.py", config["sources"]["wrapper_files"]["model/link_predictor.py"]))
    for actual_class, expected_path, expected_hash in checks:
        guard()
        actual_path = Path(inspect.getfile(actual_class))
        if actual_path.resolve() != expected_path.resolve() or sha256(actual_path) != expected_hash:
            raise CalibrationError("isolated actual class path/source mismatch")
    details = bundle.provenance_dict()
    if (details.get("Temporal_package_initializers_executed") is not False
            or details.get("full_Temporal_package_API_verified") is not False
            or details.get("native_math_or_source_modified") is not False):
        raise CalibrationError("isolated import scope/provenance mismatch")
    provenance.update(details)
    provenance["helper"] = dict(record)
    guard()
    return bundle.recurrent, bundle.head


def _numerics(torch, recurrent, head_module, config, ledger, payload, progress):
    """Fixed finite native schedule; only reached inside initialized reporting."""
    raw, checks = payload["raw"], payload["comparisons"]
    f, tol = config["fixtures"], config["tolerances"]
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    old_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float32)
    try:
        ledger.charge("wrapper_constructors"); ledger.charge("cell_constructors")
        wrapper = recurrent.MultiLayerGCLSTM(1, recurrent.GCLSTMParam(3, 2, 2)).cpu()
        ledger.charge("head_constructors"); head = head_module.LinkPredictor(2, 2, p_dropout=0.2).cpu()
    finally:
        torch.set_default_dtype(old_dtype)
    wrapper.eval(); head.eval()
    if len(wrapper.units) != 1 or type(wrapper.units[0]) is not recurrent.GCLSTM:
        raise CalibrationError("wrapper must contain exactly the pinned actual cell")
    for module in (wrapper, head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            if parameter.dtype != torch.float32 or parameter.device.type != "cpu":
                raise CalibrationError("source default CPU FP32 initialization required")
    parameter_digest = hashlib.sha256()
    for component, module in (("node_emb", wrapper), ("link_pred", head)):
        for name, value in sorted(module.state_dict().items()):
            parameter_digest.update((component + "." + name).encode())
            parameter_digest.update(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode())
            parameter_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    payload["initialized_FP32_parameter_sha256"] = parameter_digest.hexdigest()
    cell = wrapper.units[0]
    cell.register_forward_pre_hook(lambda *_: ledger.charge("cell_forward_calls"))
    head.register_forward_pre_hook(lambda *_: ledger.charge("head_forward_calls"))
    head.seq.register_forward_pre_hook(lambda *_: ledger.charge("head_sequence_forward_calls"))
    x = torch.eye(3, dtype=torch.float64)
    edges = {key: torch.tensor(f[key + "_edges"], dtype=torch.long).T.contiguous() for key in ("A", "B")}
    s0 = torch.tensor(f["initial_state"], dtype=torch.float64)
    v = torch.tensor(f["direction_v"], dtype=torch.float64); v = v / torch.linalg.vector_norm(v)
    w = torch.tensor(f["cotangent_w"], dtype=torch.float64); w = w / torch.linalg.vector_norm(w)
    def pack(hidden, cell_state):
        return torch.cat((hidden.reshape(-1), cell_state.reshape(-1)))
    def step(state, graph="A", features=None, edge_index=None):
        ledger.guard()
        if state.shape != (12,) or not torch.isfinite(state).all():
            raise CalibrationError("finite full D12 state required")
        edge = edges[graph] if edge_index is None else edge_index
        hidden, cell_state, z = wrapper(x if features is None else features, edge,
            torch.ones(edge.shape[1], dtype=state.dtype), state[:6].reshape(1, 3, 2), state[6:].reshape(1, 3, 2))
        if not torch.equal(z, hidden[-1]):
            raise CalibrationError("z differs from complete last-layer hidden output")
        return pack(hidden, cell_state)
    def compare(name, candidate, reference, kind="derivative"):
        ledger.guard()
        raw[name + "_candidate"] = candidate.detach().clone(); raw[name + "_reference"] = reference.detach().clone()
        finite = bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
        if candidate.shape != reference.shape or not finite:
            raise CalibrationError("mismatched/nonfinite comparison: " + name)
        delta = torch.abs(candidate - reference)
        threshold = tol[kind + "_atol"] + tol[kind + "_rtol"] * torch.abs(reference)
        failures = int(torch.count_nonzero(delta > threshold))
        checks[name] = {"passed": failures == 0, "shape": list(candidate.shape), "finite": finite,
            "atol": tol[kind + "_atol"], "rtol": tol[kind + "_rtol"], "violations": failures,
            "max_absolute_error": float(delta.max()),
            "relative_norm_error": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-300))}
    def jac(fn, state):
        ledger.charge("dense_jacobian_api_calls")
        return torch.autograd.functional.jacobian(fn, state, create_graph=False, strict=True, vectorize=False)
    def jvp(fn, state, direction):
        ledger.charge("jvp_api_calls")
        return torch.autograd.functional.jvp(fn, state, direction, create_graph=False, strict=True)
    def vjp(fn, state, cotangent):
        ledger.charge("vjp_api_calls")
        return torch.autograd.functional.vjp(fn, state, cotangent, create_graph=False, strict=True)
    h, c = cell(x.float(), edges["A"], edge_weight=torch.ones(4, dtype=torch.float32), H=s0[:6].float().reshape(3, 2), C=s0[6:].float().reshape(3, 2))
    wrapped = step(s0.float(), features=x.float())
    compare("cell_wrapper_default_FP32", wrapped, pack(h, c), "forward_fp32")
    pairs = torch.tensor(ordered_pairs(), dtype=torch.long)
    hp = wrapped[:6].reshape(3, 2)
    predicted = head(hp[pairs[:, 0]], hp[pairs[:, 1]]).reshape(-1)
    structural = torch.sigmoid(head.seq(torch.cat((hp[pairs[:, 0]], hp[pairs[:, 1]]), dim=-1))).reshape(-1)
    compare("head_source_structural_eval_default_FP32", predicted, structural, "forward_fp32")
    progress("source_forward_wiring")
    wrapper.double(); head.double()
    for module in (wrapper, head):
        if any(parameter.dtype != torch.float64 for parameter in module.parameters()):
            raise CalibrationError("FP64 derivative fixture conversion failed")
    fa = lambda state: step(state, "A")
    fb = lambda state: step(state, "B")
    fab = lambda state: fb(fa(state))
    j0 = jac(fa, s0); s1 = fa(s0); j1 = jac(fb, s1)
    raw.update(s0=s0, s1=s1, direction_v=v, cotangent_w=w, J0=j0, J1=j1)
    _, av = jvp(fa, s0, v); _, bv = jvp(fb, s1, v)
    compare("step_A_jvp", av, j0 @ v); compare("step_B_jvp", bv, j1 @ v)
    _, aw = vjp(fa, s0, w); _, bw = vjp(fb, s1, w)
    compare("step_A_vjp", aw, j0.T @ w); compare("step_B_vjp", bw, j1.T @ w)
    product = j1 @ j0; reverse = j0 @ j1
    direct_j = jac(fab, s0)
    compare("composed_dense_order", direct_j, product)
    _, direct_v = jvp(fab, s0, v); _, direct_w = vjp(fab, s0, w)
    compare("composed_direct_jvp", direct_v, product @ v); compare("composed_direct_vjp", direct_w, product.T @ w)
    intermediate, av = jvp(fa, s0, v); _, composed_v = jvp(fb, intermediate, av)
    _, back_w = vjp(fb, s1, w); _, composed_w = vjp(fa, s0, back_w)
    compare("stepwise_composed_jvp", composed_v, product @ v); compare("stepwise_composed_vjp", composed_w, product.T @ w)
    commutator = float(torch.linalg.matrix_norm(product - reverse))
    wrong_order = float(torch.linalg.vector_norm((product - reverse) @ v))
    checks["native_noncommuting_order_sensitivity"] = {
        "passed": commutator > tol["native_commutator_min_frobenius"] and wrong_order > tol["native_wrong_order_min_direction_error"],
        "commutator_frobenius": commutator, "wrong_order_direction_error": wrong_order,
        "no_fixture_replacement_on_failure": True}
    raw.update(chronological_product=product, reversed_same_jacobians=reverse)
    progress("native_AD_order")
    complex_direction = torch.complex(v, w)
    real_operator = lambda direction: jvp(fab, s0, direction)[1]
    complex_result = extend_real_operator(real_operator, complex_direction, combine=torch.complex)
    imaginary_result = extend_real_operator(real_operator, 1j * complex_direction, combine=torch.complex)
    compare("complex_tangent_dense_extension", complex_result, product.to(torch.complex128) @ complex_direction)
    compare("complex_tangent_i_homogeneity", imaginary_result, 1j * complex_result)
    progress("real_imaginary_tangent_extension")
    for label, fn, derivative in (("one_step", fa, j0 @ v), ("two_step", fab, product @ v)):
        for index, epsilon in enumerate(tol["finite_difference_epsilons"]):
            ledger.charge("finite_difference_function_evaluations"); positive = fn(s0 + epsilon * v)
            ledger.charge("finite_difference_function_evaluations"); negative = fn(s0 - epsilon * v)
            compare(f"finite_difference_{label}_epsilon_{index}", (positive - negative) / (2 * epsilon), derivative, "finite_difference")
    progress("fixed_finite_differences")
    inverse, coordinates = coordinate_permutation(f["new_node_to_old_node_permutation"])
    p = torch.tensor(f["new_node_to_old_node_permutation"], dtype=torch.long)
    q = torch.tensor(coordinates, dtype=torch.long); inv = torch.tensor(inverse, dtype=torch.long)
    perm_edges = inv[edges["A"]]
    perm_fn = lambda state: step(state, "A", features=x[p], edge_index=perm_edges)
    perm_result = perm_fn(s0[q]); compare("node_forward_permutation_FP64", perm_result, s1[q], "forward")
    perm_j = jac(perm_fn, s0[q]); compare("node_Jacobian_permutation", perm_j, j0[q][:, q])
    hp_perm = perm_result[:6].reshape(3, 2); remapped_pairs = inv[pairs]
    hp_base = s1[:6].reshape(3, 2)
    base_head_FP64 = head(hp_base[pairs[:, 0]], hp_base[pairs[:, 1]]).reshape(-1)
    perm_head = head(hp_perm[remapped_pairs[:, 0]], hp_perm[remapped_pairs[:, 1]]).reshape(-1)
    compare("ordered_pair_head_permutation_FP64", perm_head, base_head_FP64, "forward")
    raw.update(permutation_coordinates=q, permuted_Jacobian=perm_j)
    ledger.charge("analytic_control_cases")
    aa = torch.tensor(f["analytic_A"], dtype=torch.float64); bb = torch.tensor(f["analytic_B"], dtype=torch.float64)
    compare("analytic_noncommuting_order", bb @ aa, torch.tensor(f["analytic_chronological_product"], dtype=torch.float64), "forward")
    analytic_reverse = aa @ bb
    compare("analytic_reversed_reference", analytic_reverse, torch.tensor(f["analytic_reversed_product"], dtype=torch.float64), "forward")
    spectral_norm = float(torch.linalg.svdvals(aa).max())
    checks["analytic_nonnormal_sensitivity"] = {"passed": spectral_norm > tol["analytic_nonnormal_min_spectral_norm"] and not torch.equal(bb @ aa, analytic_reverse), "spectral_norm": spectral_norm, "analytic_eigenvalue_magnitudes": [1, 1]}
    progress("permutation_and_analytic_control")
    captured = []
    def consume(state, graph):
        result = step(state, graph); captured.append(result.detach().clone()); return result
    def score(state, pair_ids):
        indices = torch.tensor(pair_ids, dtype=torch.long); hidden = state[:6].reshape(3, 2)
        return head(hidden[indices[:, 0]], hidden[indices[:, 1]]).reshape(-1)
    _, predictions, events = chronological_trace(f["chronology_inputs"], torch.zeros(12, dtype=torch.float64), consume=consume, score=score, guard=ledger.guard)
    payload["chronology_events"] = events
    checks["complete_state_native_chronology"] = {"passed": events == f["chronology_events"], "state_shape": [1, 3, 2], "joint_dimension": 12, "scored_targets": [1, 2], "conditioning_timestamp": 0}
    raw["chronology_states"] = torch.stack(captured); raw["chronology_predictions"] = torch.stack(predictions)
    ledger.complete()
    payload["engineering_passed"] = all(value["passed"] for value in checks.values())
    payload["native_threads"] = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}
    progress("complete")


def _worker(args):
    output = args.output / "worker"; output.mkdir(exist_ok=False)
    config = frozen_config(args.config)
    status = {"kind": "native_graph_calibration_worker", "protocol_id": PROTOCOL_ID, "state": "initializing", "scientific_acceptance": False, "native_work_started": False}
    payload = {"raw": {}, "comparisons": {}, "scientific_acceptance": False}
    admission = None
    def guard():
        if time.monotonic() >= args.work_deadline_monotonic:
            raise CalibrationError("import-inclusive worker deadline expired")
        if admission is not None and Path(admission["stop_path"]).exists():
            raise CalibrationError("owner STOP")
    def alarm(_signum, _frame):
        raise CalibrationError("physical worker work deadline")
    remaining = args.work_deadline_monotonic - time.monotonic()
    if not 0 < remaining <= 590 or args.work_deadline_monotonic - args.started_monotonic != 590:
        raise CalibrationError("shared pre-import deadline missing or invalid")
    signal.signal(signal.SIGALRM, alarm); signal.setitimer(signal.ITIMER_REAL, remaining)
    ledger = Ledger(config["finite_counts"], guard)
    def persist():
        status.update(elapsed_seconds=time.monotonic() - args.started_monotonic, attempted_counts=dict(ledger.counts))
        if sys.platform == "linux":
            resource = importlib.import_module("resource")
            status["worker_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            status["rss_source"] = "Linux getrusage ru_maxrss KiB converted to bytes; not virtual address space"
            if "runtime_preflight" in status:
                limit_path = Path(status["runtime_preflight"]["memory"]["path"])
                usage_path = limit_path.with_name("memory.current" if limit_path.name == "memory.max" else "memory.usage_in_bytes")
                if usage_path.is_file():
                    status["inherited_cgroup_current_physical_bytes"] = int(usage_path.read_text())
        atomic_json(output / "status.json", status)
    persist()
    try:
        guard(); admission = verify_admission(args, config, worker=True)
        runtime = _runtime_preflight(config); status["runtime_preflight"] = runtime; persist()
        helpers = runpy.run_path(str(args.root / config["sources"]["ownership_helper"]["path"]), run_name="_calibration_ownership_helpers")
        actual_parent = helpers["process_identity"](os.getppid())
        if not helpers["same_process"](admission.get("supervisor_identity"), actual_parent):
            raise CalibrationError("live physical supervisor parent/start identity required")
        helpers["install_spawn_journal"](args.output / "owned-processes.json")
        guard()
        from .reporting import ExperimentReporter
        identities = {"config_sha256": sha256(args.config), "source_sha256": sha256(Path(__file__)),
            "source_commit": admission["source_commit"], "protocol_sha256": sha256(args.root / PROTOCOL_PATH),
            "cell_sha256": config["sources"]["cell_sha256"], "reporter_sha256": config["sources"]["reporter"]["sha256"],
            "environment_manifest_sha256": admission["environment"]["sha256"],
            "graph_import_helper_sha256": config["sources"]["graph_import_helper"]["sha256"],
            "module_path": MODULE_PATH, "config_path": CONFIG_PATH, "protocol_path": PROTOCOL_PATH}
        with ExperimentReporter(output / "reporting", project=config["reporting"]["project"], entity=None, mode="offline",
                config={"protocol_id": PROTOCOL_ID, "nodes": 3, "hidden": 2, "joint_dimension": 12, "cpu_threads": 1, "training": False}, identities=identities) as report:
            guard(); torch = importlib.import_module("torch")
            import_provenance = {}
            recurrent, head = _load_wrappers(args.source_root, config, root=args.root,
                installed_cell_path=runtime["installed_cell_path"], guard=guard, provenance=import_provenance)
            status["isolated_import_provenance"] = import_provenance
            payload["isolated_import_provenance"] = import_provenance
            status.update(state="running", native_work_started=True); persist()
            progress_index = 0
            def progress(stage):
                nonlocal progress_index
                guard(); persist()
                report.log({"calibration/stage": stage, "calibration/cell_calls": ledger.counts["cell_forward_calls"],
                    "calibration/comparisons": len(payload["comparisons"]), "calibration/failures": sum(not v["passed"] for v in payload["comparisons"].values())}, step=progress_index)
                progress_index += 1
            _numerics(torch, recurrent, head, config, ledger, payload, progress)
            guard()
            numpy = importlib.import_module("numpy")
            arrays = {name: value.detach().cpu().numpy() for name, value in payload["raw"].items()}
            temp = output / "raw.tmp.npz"; numpy.savez(temp, **arrays); temp.replace(output / "raw.npz")
            summary = {key: value for key, value in payload.items() if key != "raw"}
            summary.update(attempted_counts=dict(ledger.counts), raw_npz_sha256=sha256(output / "raw.npz"), identities=identities)
            atomic_json(output / "calibration.json", summary)
            report.log({"calibration/engineering_passed": payload["engineering_passed"], "calibration/scientific_acceptance": False}, step=progress_index)
            guard()
        guard(); status.update(state="completed" if payload["engineering_passed"] else "engineering_failed", engineering_passed=payload["engineering_passed"]); persist()
        return 0 if payload["engineering_passed"] else 2
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, failure=str(error)[:512], partial_comparisons=payload["comparisons"])
        if payload["raw"] and time.monotonic() < args.work_deadline_monotonic:
            try:
                guard()
                numpy = importlib.import_module("numpy")
                temp = output / "partial-raw.tmp.npz"
                numpy.savez(temp, **{name: value.detach().cpu().numpy() for name, value in payload["raw"].items()})
                guard(); temp.replace(output / "partial-raw.npz")
                status["partial_raw_npz_sha256"] = sha256(output / "partial-raw.npz")
            except BaseException as save_error:
                status["partial_raw_save_error_type"] = type(save_error).__name__
        elif payload["raw"]:
            status["partial_raw_censored_by_work_deadline"] = True
        persist(); return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _supervise(args):
    started = PROCESS_STARTED
    config = frozen_config(args.config); admission = verify_admission(args, config)
    if sys.platform != "linux" or not hasattr(os, "sched_setaffinity"):
        raise CalibrationError("Slurm Linux physical resource supervision required")
    memory = kernel_memory_limit()
    if memory["physical_memory_limit_bytes"] > config["resources"]["memory_bytes"]:
        raise CalibrationError("actual Slurm/cgroup physical memory cap must be <=2GiB")
    allowed = os.sched_getaffinity(0)
    if not allowed:
        raise CalibrationError("CPU affinity unavailable")
    os.sched_setaffinity(0, {min(allowed)})
    if "SLURM_JOB_ID" not in os.environ or str(admission.get("slurm_job_id")) != os.environ["SLURM_JOB_ID"]:
        raise CalibrationError("admission must bind actual Slurm job identity")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise CalibrationError("actual Slurm cpus-per-task must match the one-CPU envelope")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    helpers = runpy.run_path(str(args.root / config["sources"]["ownership_helper"]["path"]), run_name="_calibration_ownership_helpers")
    deadline = started + 600
    child, owned = None, None
    receipt = {"kind": "native_graph_calibration_supervisor", "protocol_id": PROTOCOL_ID, "state": "initializing", "started_monotonic": started,
               "memory": memory, "cpu_affinity": sorted(os.sched_getaffinity(0)), "slurm_job_id": os.environ["SLURM_JOB_ID"], "scientific_acceptance": False}
    def persist():
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "supervisor-receipt.json", receipt)
    persist()
    # Root binding is immutable; shared start is carried in a unique derivative
    # admission whose parent hash and exact supervisor argv remain intact.
    child_admission = dict(admission); child_admission["started_monotonic"] = started
    child_admission["supervisor_identity"] = helpers["process_identity"](os.getpid())
    child_admission["parent_admission_sha256"] = sha256(args.admission)
    atomic_json(output / "worker-admission.json", child_admission)
    command = [sys.executable, "-m", "representation_collapse.native_graph_calibration_v2", "--worker",
        "--root", str(args.root.resolve()), "--config", str(args.config.resolve()), "--source-root", str(args.source_root.resolve()),
        "--admission", str(output / "worker-admission.json"), "--output", str(output),
        "--started-monotonic", str(started), "--work-deadline-monotonic", str(started + 590)]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "GOMAXPROCS"):
        env[key] = "1"
    env.update(PYTHONPATH=str(args.root.resolve() / "src"), PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="offline")
    for key in ("WANDB_CACHE_DIR", "WANDB_CONFIG_DIR", "WANDB_DATA_DIR"):
        env[key] = str(output / "reporting-support" / key.lower())
    try:
        if time.monotonic() >= started + 590:
            raise CalibrationError("deadline exhausted before worker spawn")
        with (output / "worker.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=log, cwd=args.root, env=env, start_new_session=True)
            owned = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
            receipt.update(state="running", worker_pid=child.pid); persist()
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 1 or Path(admission["stop_path"]).exists():
                    raise CalibrationError("outer physical deadline or owner STOP")
                owned.refresh(timeout=min(0.5, remaining - 0.5))
                try:
                    child.wait(timeout=min(0.1, remaining - 1))
                except subprocess.TimeoutExpired:
                    pass
        receipt.update(state="completed" if child.returncode == 0 else "failed", worker_exit_code=child.returncode)
    except BaseException as error:
        receipt.update(state="incomplete", error_type=type(error).__name__)
    finally:
        if child is not None:
            if owned is None:
                owned = helpers["OwnedProcesses"](child.pid, output / "owned-processes.json")
            errors = owned.cleanup(deadline=deadline)
            receipt.update(cleanup_errors=errors, owned_processes=list(owned.known.values()), cleanup_reconciliation=owned.cleanup_reconciliation)
            if errors:
                receipt["state"] = "incomplete"
            remaining = deadline - time.monotonic()
            if remaining > 0:
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    receipt["state"] = "incomplete"
        if time.monotonic() >= deadline:
            receipt["state"] = "incomplete"
        persist()
    return 0 if receipt["state"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "config", "source-root", "admission", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--started-monotonic", type=float)
    parser.add_argument("--work-deadline-monotonic", type=float)
    args = parser.parse_args(argv)
    if args.worker:
        if args.started_monotonic is None or args.work_deadline_monotonic is None:
            parser.error("worker requires shared import-inclusive deadlines")
        return _worker(args)
    return _supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
