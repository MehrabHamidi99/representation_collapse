"""Data-only semantic decoder for complete primitive-v2 inference candidates.

External fit acceptance supplies the expected execution identities. This reader
checks their consistency, not selection authority, and never imports Torch or
restores RNG/optimizer state. The v1 module supplies unchanged tensor encoding
primitives only; no v1 candidate identity or schema is asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Any

from . import tgrab_selected_checkpoint as enc

CONTRACT_SHA256 = "aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381"
TEMPLATE_SHA256 = "22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f"
IDENTITY_KEYS = frozenset(("executed_long_contract_sha256", "numerical_template_sha256", "conversion_config_sha256",
    "converted_manifest_sha256", "raw_sha256", "conversion_receipt_sha256", "environment_receipt_sha256",
    "node2a_receipt_sha256", "prefix_reference_sha256", "tensor_catalog_sha256", "actual_supervisor_receipt_sha256",
    "source_closure_sha256", "reporting_identity_sha256"))


class V2CodecError(enc.CheckpointError):
    pass


@dataclass(frozen=True)
class V2ExpectedIdentity:
    """Values authenticated by independent fit acceptance, never the bundle.

    Mapping fields are compared exactly. Callers must derive them from accepted
    supervisor/run/initialization records; copying self-asserted candidate fields
    would provide integrity only, not an authenticated admission.
    """
    run_id: str
    source_commit: str
    slurm_job_id: str
    identities: Mapping[str, str]
    source_closure: Mapping[str, str]
    runtime_environment: Mapping[str, Any]
    initialization: Mapping[str, Any]
    module_modes: Mapping[str, Any]
    module_state_dict_versions: Mapping[str, Any]
    optimizer_parameter_order: list[str]
    actual_job: Mapping[str, Any]


def _require(ok, message):
    if not ok:
        raise V2CodecError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise V2CodecError("nonfinite JSON: " + value)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise V2CodecError("invalid JSON metadata") from exc


def _sha(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256")
    return value


def _frozen():
    root = enc._repo_root()
    paths = {"contract": (root / "configs/graph/tgrab-long-fit-v2.json", CONTRACT_SHA256),
             "template": (root / "configs/graph/tgrab-reconstruction-v1.json", TEMPLATE_SHA256),
             "catalog": (root / enc.CATALOG_PATH, enc.CATALOG_SHA256)}
    values = {}
    for name, (path, sha) in paths.items():
        raw = path.read_bytes()
        _require(enc._sha(raw) == sha, "frozen " + name + " identity mismatch")
        values[name] = _json(raw)
    return values["contract"], values["catalog"]


def _validate_provenance(p, m, expected, contract, catalog):
    epoch = m["epoch"]
    expected_keys = {"kind", "schema_version", "run_id", "attempt_id", "source_commit", "slurm_job_id", "identities",
        "executed_long_contract_sha256", "numerical_template_sha256", "conversion_config_sha256", "initialization",
        "model_seed", "negative_seed", "seed_before_construction", "construction_order", "module_modes",
        "module_state_dict_versions", "optimizer_parameter_order", "runtime_environment", "actual_job", "runtime_policy",
        "optimizer_state_included", "rng_restore_allowed", "test_payload_opened", "complete_epoch", "complete_epoch_counters",
        "journal_epoch_seal", "created_at_utc"}
    _require(set(p) == expected_keys, "v2 provenance field set mismatch")
    fixed = {"kind": "tgrab_long_fit_candidate_provenance_v2", "schema_version": 2, "attempt_id": "longfit01",
        "executed_long_contract_sha256": CONTRACT_SHA256, "numerical_template_sha256": TEMPLATE_SHA256,
        "conversion_config_sha256": TEMPLATE_SHA256, "model_seed": 0, "negative_seed": 1,
        "seed_before_construction": True, "construction_order": ["seed_all", "recurrent", "link_head", "optimizer", "negative_generator"],
        "optimizer_state_included": False, "rng_restore_allowed": False, "test_payload_opened": False,
        "complete_epoch": epoch}
    for key, value in fixed.items():
        _require(type(p[key]) is type(value) and p[key] == value, "provenance policy mismatch: " + key)
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", expected.run_id) is not None, "invalid run id")
    _require(re.fullmatch(r"[0-9a-f]{40}", expected.source_commit) is not None, "invalid source commit")
    _require(str(expected.slurm_job_id).isdigit(), "invalid actual job id")
    for key in ("run_id", "source_commit", "slurm_job_id", "identities", "runtime_environment", "initialization", "module_modes",
                "module_state_dict_versions", "optimizer_parameter_order", "actual_job"):
        _require(enc._canonical(p[key]) == enc._canonical(getattr(expected, key)), "external execution identity mismatch: " + key)
    ids = p["identities"]
    _require(set(ids) == IDENTITY_KEYS, "exact v2 identity keys required")
    for value in ids.values():
        _sha(value)
    for key, value in {"executed_long_contract_sha256": CONTRACT_SHA256, "numerical_template_sha256": TEMPLATE_SHA256,
        "conversion_config_sha256": TEMPLATE_SHA256, "tensor_catalog_sha256": enc.CATALOG_SHA256,
        "raw_sha256": contract["data"]["raw_sha256"],
        "prefix_reference_sha256": contract["prefix_parity"]["accepted_reference_sha256"]}.items():
        _require(ids[key] == value, "frozen identity mismatch: " + key)
    _require(ids["source_closure_sha256"] == enc._sha(json.dumps(dict(expected.source_closure), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")), "source closure identity mismatch")
    for value in expected.source_closure.values():
        _sha(value)
    for name, sha in contract["input_pins"]["source_files_sha256"].items():
        _require(expected.source_closure.get("src/representation_collapse/" + name) == sha, "native source closure pin mismatch: " + name)
    init = p["initialization"]
    _require(set(init) == {"seeding_record", "post_model_rng_record", "negative_generator_record", "initial_component_semantic_sha256"}, "initialization observations incomplete")
    for key, filename, role in (("seeding_record", "seeding.json", "seeding_before_models"),
            ("post_model_rng_record", "post-model-rng.json", "post_model_construction_rng"),
            ("negative_generator_record", "negative-generator.json", "negative_generator_initialization")):
        ref = init[key]
        _require(set(ref) == {"relative_path", "bytes", "sha256", "role", "encoding"}
                 and ref["relative_path"] == filename and type(ref["bytes"]) is int and ref["bytes"] > 0
                 and ref["role"] == role and ref["encoding"] == "primitive_json", "initial RNG observation reference malformed")
        _sha(ref["sha256"])
    _require(set(init["initial_component_semantic_sha256"]) == {"recurrent", "link_head"}, "initial component identities incomplete")
    for value in init["initial_component_semantic_sha256"].values():
        _sha(value)
    modes = p["module_modes"]
    _require(set(modes) == {"relative_path", "bytes", "sha256"} and modes["relative_path"] == "module-modes.json"
             and type(modes["bytes"]) is int and modes["bytes"] > 0, "mode observation reference malformed")
    _sha(modes["sha256"])
    job = p["actual_job"]
    for key, value in {"job_id": expected.slurm_job_id, "account": "mila", "partition": "main-cpu", "qos": "normal",
            "nodes": "1", "tasks": "1", "cpus_on_node": "1", "cpus_per_task": "1", "gpu_allocation": "0",
            "memory_max_bytes": "2147483648"}.items():
        _require(str(job.get(key)) == str(value), "actual fit allocation mismatch: " + key)
    _require(job.get("time_limit") in {"2-00:00:00", "48:00:00"}, "actual fit walltime mismatch")
    env = p["runtime_environment"]
    _require(env.get("requirements_lock_sha256") == contract["input_pins"]["environment_lock_sha256"]
             and env.get("wheel_manifest_sha256") == contract["input_pins"]["wheel_manifest_sha256"], "environment lock/wheel mismatch")
    policy = {"device": "cpu", "precision": "float32", "cpu_threads": 1, "amp": False,
        "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "work_seconds": 172680, "slurm_seconds": 172800}
    _require(p["runtime_policy"] == policy, "runtime policy mismatch")
    counters = {"complete_epochs": epoch, "validation_epochs": epoch, "attempted_updates": epoch * 2559,
        "completed_updates": epoch * 2559, "optimizer_steps_confirmed": epoch * 2559}
    _require(p["complete_epoch_counters"] == counters and all(type(v) is int for v in p["complete_epoch_counters"].values()), "complete-epoch counters mismatch")
    seal = p["journal_epoch_seal"]
    _require(seal.get("epoch") == epoch and seal.get("records") == 5118, "journal seal epoch/record mismatch")
    for key in ("attempted_updates", "completed_updates", "optimizer_steps_confirmed"):
        _require(type(seal.get(key)) is int and seal[key] == epoch * 2559, "journal cumulative count mismatch")
    for key, value in {"kind": "figure10_longfit_epoch_chunk_seal_v2", "run_id": expected.run_id,
            "attempt_id": "longfit01", "contract_sha256": CONTRACT_SHA256,
            "admission_sha256": ids["actual_supervisor_receipt_sha256"],
            "filename": f"epoch-{epoch:03d}.jsonl", "receipt_filename": f"epoch-{epoch:03d}.seal.json"}.items():
        _require(seal.get(key) == value, "journal seal execution context mismatch: " + key)
    for key in ("chunk_sha256", "last_record_sha256", "receipt_sha256"):
        _sha(seal.get(key))
    _require(set(p["module_state_dict_versions"]) == {"recurrent", "link_head"}, "state dict versions incomplete")
    _require(len(p["optimizer_parameter_order"]) == sum(c["parameter_count"] for c in catalog["components"].values())
             and len(set(p["optimizer_parameter_order"])) == len(p["optimizer_parameter_order"]), "optimizer parameter order malformed")


def _validate_payloads(payloads, expected):
    contract, catalog = _frozen()
    m, p, v = (_json(payloads[name]) for name in ("manifest.json", "provenance.json", "validation.json"))
    _require(set(m) == {"kind", "schema_version", "epoch", "executed_long_contract_sha256", "numerical_template_sha256",
        "tensor_catalog_sha256", "components", "terminal_state", "validation_time_mean_f1", "files",
        "optimizer_state_included", "rng_restore_allowed", "test_payload_opened"}, "v2 manifest field set mismatch")
    for key, value in {"kind": "tgrab_complete_epoch_candidate_v2", "schema_version": 2,
        "executed_long_contract_sha256": CONTRACT_SHA256, "numerical_template_sha256": TEMPLATE_SHA256,
        "tensor_catalog_sha256": enc.CATALOG_SHA256, "optimizer_state_included": False,
        "rng_restore_allowed": False, "test_payload_opened": False}.items():
        _require(type(m[key]) is type(value) and m[key] == value, "manifest v2 policy mismatch: " + key)
    _require(type(m["epoch"]) is int and 1 <= m["epoch"] <= 200, "epoch outside v2 cap")
    _require(set(m["files"]) == set(enc.FILES) - {"manifest.json"}, "file descriptor inventory mismatch")
    for name, d in m["files"].items():
        _require(set(d) == {"bytes", "sha256"} and type(d["bytes"]) is int and d["bytes"] == len(payloads[name])
                 and _sha(d["sha256"]) == enc._sha(payloads[name]), "file digest/size mismatch: " + name)
    _require(set(m["components"]) == {"recurrent", "link_head"}, "both components required")
    for component, name in (("recurrent", "recurrent.f32le"), ("link_head", "link-head.f32le")):
        _require(set(m["components"][component]) == {"filename", "bytes", "sha256", "tensors"}, "component descriptor fields mismatch")
        d = m["components"][component]
        _require(type(d["bytes"]) is int, "component byte count must be integer")
        for tensor in d["tensors"]:
            _require(type(tensor.get("offset")) is int and type(tensor.get("nbytes")) is int
                     and isinstance(tensor.get("shape"), list) and all(type(x) is int for x in tensor["shape"]), "tensor shape/offset types mismatch")
        enc._verify_component(component, d, payloads[name], catalog)
    t, raw = m["terminal_state"], payloads["validation-terminal.f32le"]
    fixed = {"filename": "validation-terminal.f32le", "bytes": 102400, "sha256": enc._sha(raw),
        "next_target_t": 2816, "last_consumed_t": 2815, "construction": "fresh_zero_eval_replay_0_through_2815", "epoch": m["epoch"]}
    _require(set(t) == set(fixed) | {"tensors"} and len(raw) == 102400, "terminal descriptor/size mismatch")
    for key, value in fixed.items():
        _require(type(t[key]) is type(value) and t[key] == value, "terminal chronology mismatch: " + key)
    _require(len(t["tensors"]) == 2, "full H/C terminal required")
    for d, name, offset in zip(t["tensors"], ("hidden", "cell"), (0, 51200), strict=True):
        chunk = raw[offset:offset + 51200]
        row = {"name": name, "dtype": "float32_le", "shape": [1, 100, 128], "order": "C", "offset": offset,
            "nbytes": 51200, "data_sha256": enc._sha(chunk), "semantic_sha256": enc.tensor_semantic_sha256("terminal_state", name, [1,100,128], chunk)}
        _require(enc._canonical(d) == enc._canonical(row), "terminal H/C tensor descriptor mismatch")
        enc._finite_f32(chunk)
    _require(set(v) == {"kind", "schema_version", "split", "epoch", "rows", "time_mean_f1"}
             and v["kind"] == "tgrab_complete_validation_rows_v2" and type(v["schema_version"]) is int
             and v["schema_version"] == 2 and v["split"] == "validation", "v2 validation metadata mismatch")
    _require(isinstance(v["rows"], list) and len(v["rows"]) == 256, "complete validation rows required")
    keys = {"target_timestamp", "source_timestamp", "target_phase", "source_phase", "phase_transition_expected", "change_point",
            "pairs", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"}
    for timestamp, row in enumerate(v["rows"], 2560):
        _require(set(row) == keys, "validation row keys mismatch")
        for key, value in {"target_timestamp": timestamp, "source_timestamp": timestamp-1,
                "target_phase": timestamp % 64, "source_phase": (timestamp-1) % 64}.items():
            _require(type(row[key]) is int and row[key] == value, "validation chronology mismatch")
        for key in ("phase_transition_expected", "change_point"):
            _require(type(row[key]) is bool and row[key] == (timestamp % 32 == 0), "validation transition mismatch")
        _require(all(type(row[key]) is int and row[key] >= 0 for key in ("pairs", "tp", "fp", "fn", "tn")), "invalid confusion count")
        _require(all(type(row[key]) in (float, int) and math.isfinite(row[key]) for key in ("precision", "recall", "f1", "accuracy")), "invalid validation metric")
    enc._validate_validation_score({"epoch": m["epoch"], "validation": {"time_mean_f1": m["validation_time_mean_f1"]}}, v)
    _validate_provenance(p, m, expected, contract, catalog)
    return {"manifest": m, "provenance": p, "validation": v, "recurrent": payloads["recurrent.f32le"],
            "link_head": payloads["link-head.f32le"], "terminal_state": raw,
            "files_sha256": {name: enc._sha(data) for name, data in payloads.items()}}


def read_candidate_v2(path, expected_manifest_sha256, *, expected: V2ExpectedIdentity):
    """Verify the entire original v2 bundle before returning native-load bytes.

    No model mutation occurs here. An independently accepted finalized selection
    pointer must supply ``expected_manifest_sha256``; this is not its verifier.
    """
    _sha(expected_manifest_sha256)
    root = Path(path).absolute()
    pfd = enc._open_bundle_directory(root.parent)
    fd = None
    try:
        before = os.stat(root.name, dir_fd=pfd, follow_symlinks=False)
        _require(stat.S_ISDIR(before.st_mode), "candidate is not a real directory")
        fd = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=pfd)
        opened = os.fstat(fd)
        _require((before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino), "candidate replaced during open")
        _require(set(os.listdir(fd)) == set(enc.FILES), "exact six-file inventory required")
        payloads = {name: enc._read_stable_at(fd, name, 12_582_912 if name.endswith("f32le") else 2_000_000) for name in enc.FILES}
        _require(sum(map(len, payloads.values())) <= 12_582_912, "aggregate candidate bound exceeded")
        _require(enc._sha(payloads["manifest.json"]) == expected_manifest_sha256, "selected manifest SHA mismatch")
        result = _validate_payloads(payloads, expected)
        after = os.stat(root.name, dir_fd=pfd, follow_symlinks=False)
        _require((after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino)
                 and set(os.listdir(fd)) == set(enc.FILES), "candidate namespace changed during read")
        return result
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise V2CodecError("malformed or unsafe primitive-v2 candidate") from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(pfd)
