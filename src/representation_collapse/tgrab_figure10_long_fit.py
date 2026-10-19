"""Fresh seed-0 200-epoch-or-patience corrected fit under a separate v2 contract.

The unchanged v1 config is only a numerical/conversion template. This module
publishes primitive v2 inference bundles, never v1 admission claims or resumes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import tempfile
import time

from . import tgrab_figure10_canary as common
from .tgrab_figure10_full_fit import canonical, digest, utcnow, write_record, DurableEvents, FitStop, check_conversion

PROTOCOL_ID = "tgrab_figure10_long_fit_v2"
CONTRACT_SHA256 = "aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381"
PREFIX_SHA256 = "825f11e7c7d7e1c2e246666859133c822137da3d3551cf344a837776e9aecc9f"
WORK_SECONDS = 172680
OUTER_SECONDS = 172770
ATTEMPT_ID = "longfit01"
FILES = {"manifest.json", "recurrent.f32le", "link-head.f32le", "validation-terminal.f32le", "validation.json", "provenance.json"}


class LongFitError(RuntimeError):
    pass


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise LongFitError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise LongFitError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(fd)
    finally:
        os.close(fd)
    _sync(Path(path).parent)


def load_contract(path, expected):
    if expected != CONTRACT_SHA256:
        raise LongFitError("only the frozen executed long-fit v2 contract is eligible")
    value = _json(common.read_pinned_bytes(Path(path), expected, max_bytes=2_000_000))
    if value.get("protocol_id") != PROTOCOL_ID or value.get("schema_version") != 2:
        raise LongFitError("wrong long-fit contract version")
    return value


def validate_supervisor(receipt, args, argv):
    if (receipt.get("schema") != "tgrab-figure10-supervisor-receipt-v2"
            or receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("run_id") != args.run_id
            or receipt.get("source_commit") != args.source_commit
            or str(receipt.get("job_id")) != os.environ.get("SLURM_JOB_ID")
            or receipt.get("source_sha256") != common.sha256_file(Path(__file__))
            or receipt.get("canary_args") != list(argv)):
        raise LongFitError("long-fit supervisor source/run/job/arguments mismatch")
    started = receipt.get("wall_started_epoch_ns")
    if (type(started) is not int or started <= 0 or args.runtime_seconds != WORK_SECONDS
            or receipt.get("work_seconds") != WORK_SECONDS or receipt.get("outer_wall_seconds") != OUTER_SECONDS
            or args.deadline_unix != receipt.get("deadline_unix")
            or args.deadline_unix != started / 1e9 + WORK_SECONDS):
        raise LongFitError("long-fit admitted runtime mismatch")
    job = receipt.get("job", {})
    for key, expected in {"account": "mila", "partition": "main-cpu", "qos": "normal", "nodes": "1", "tasks": "1",
                          "cpus_on_node": "1", "cpus_per_task": "1", "gpu_allocation": "0",
                          "memory_max_bytes": str(2 * 1024**3)}.items():
        if str(job.get(key)) != expected:
            raise LongFitError("long-fit resource mismatch: " + key)
    for key, variable in {"job_id": "SLURM_JOB_ID", "account": "SLURM_JOB_ACCOUNT", "partition": "SLURM_JOB_PARTITION",
                          "cpus_on_node": "SLURM_CPUS_ON_NODE"}.items():
        if variable in os.environ and str(job.get(key)) != os.environ[variable]:
            raise LongFitError("actual scheduler identity mismatch")
    if job.get("time_limit") not in {"2-00:00:00", "48:00:00"}:
        raise LongFitError("registered Slurm ceiling is exactly 48 hours")
    for key in ("long_contract", "prefix_updates", "environment_receipt", "node2a_receipt", "conversion_receipt"):
        if receipt.get(key) != {"path": str(getattr(args, key)), "sha256": getattr(args, key + "_sha256")}:
            raise LongFitError("supervisor input identity mismatch: " + key)
    child = receipt.get("child_argv")
    if (receipt.get("sys_executable") != sys.executable or type(child) is not list
            or child[:3] != [sys.executable, "-B", "-c"] or child[-len(argv):] != list(argv)
            or receipt.get("argv_sha256") != digest(json.dumps(child, separators=(",", ":")).encode())):
        raise LongFitError("admitted bootstrap command differs")


def load_prefix_reference(path, expected):
    if expected != PREFIX_SHA256:
        raise LongFitError("independently accepted exact v1 prefix reference required")
    raw = common.read_pinned_bytes(Path(path), expected, max_bytes=2_000_000)
    if not raw.endswith(b"\n"):
        raise LongFitError("truncated prefix reference")
    rows = [_json(line) for line in raw.splitlines()]
    if len(rows) != 256:
        raise LongFitError("accepted prefix must contain 256 rows")
    for index, row in enumerate(rows, 1):
        if (row.get("target_timestamp") != index or row.get("completed") is not True
                or row.get("optimizer_step_status") != "completed"
                or type(row.get("loss")) not in (int, float) or not math.isfinite(row["loss"]) or row["loss"] < 0):
            raise LongFitError("invalid accepted prefix chronology/loss")
    return rows


class PrefixParity:
    def __init__(self, rows, run_dir, reference_hash):
        self.rows, self.root, self.hash = rows, Path(run_dir), reference_hash
        self.compared = 0
        self.poisoned = False

    def on_update(self, record):
        if self.poisoned:
            raise LongFitError("prefix parity gate is terminal after disagreement")
        if self.compared == 256:
            return
        index = self.compared
        expected = self.rows[index]
        if (record.get("attempt") != index + 1 or record.get("target_timestamp") != index + 1
                or record.get("completed") is not True or record.get("loss") != expected["loss"]):
            self.poisoned = True
            write_record(self.root, "prefix-parity-failure.json", {"kind": "long_fit_prefix_parity_v2", "status": "failed",
                "reference_sha256": self.hash, "target_timestamp": index + 1, "prior_exact_matches": index,
                "expected_loss": expected["loss"], "observed_loss": record.get("loss")})
            raise LongFitError("first-256 native loss differs from accepted reference; stop without retry")
        self.compared += 1
        if self.compared == 256:
            write_record(self.root, "prefix-parity.json", {"kind": "long_fit_prefix_parity_v2", "status": "pass",
                "reference_sha256": self.hash, "exact_loss_matches": 256, "comparison": "exact_python_float_equality",
                "test_payload_opened": False})


def read_candidate_v2(path, manifest_hash):
    """Verify file inventory/bytes/hashes; not a selected-inference admission."""
    path = Path(path)
    if set(entry.name for entry in path.iterdir()) != FILES:
        raise LongFitError("complete v2 candidate must have exactly six files")
    manifest_raw = common.read_pinned_bytes(path / "manifest.json", manifest_hash, max_bytes=2_000_000)
    manifest = _json(manifest_raw)
    if manifest.get("kind") != "tgrab_complete_epoch_candidate_v2" or manifest.get("schema_version") != 2:
        raise LongFitError("candidate is not primitive v2")
    if set(manifest.get("files", {})) != FILES - {"manifest.json"}:
        raise LongFitError("v2 candidate file descriptors incomplete")
    for name, descriptor in manifest["files"].items():
        raw = common.read_pinned_bytes(path / name, descriptor["sha256"], max_bytes=12_582_912)
        if type(descriptor["bytes"]) is not int or len(raw) != descriptor["bytes"]:
            raise LongFitError("v2 candidate file length mismatch")
    return manifest


def write_candidate_v2(root, evidence, provenance):
    from . import tgrab_selected_checkpoint as codec
    from .tgrab_figure10_candidate_encoder import _component_bytes, _terminal_bytes, _validation_rows
    root = Path(root)
    epoch = evidence.record.epoch
    if type(epoch) is not int or not 1 <= epoch <= 200:
        raise LongFitError("complete epoch outside v2 cap")
    for key in ("attempted_updates", "completed_updates", "optimizer_steps_confirmed"):
        if getattr(evidence.record, key) != 2559:
            raise LongFitError("only complete 2559-update epochs can publish a candidate")
    if (provenance.get("kind") != "tgrab_long_fit_candidate_provenance_v2"
            or provenance.get("executed_long_contract_sha256") != CONTRACT_SHA256
            or provenance.get("numerical_template_sha256") != common.CONFIG_SHA256
            or provenance.get("complete_epoch") != epoch):
        raise LongFitError("candidate provenance lacks actual v2/template/epoch identities")
    _, catalog = codec._schema_and_catalog()
    components, files = {}, {}
    for component, key in (("recurrent", "node_emb"), ("link_head", "link_pred")):
        raw, descriptor = _component_bytes(component, evidence.model_state_dicts[key], catalog)
        components[component] = descriptor
        files[descriptor["filename"]] = raw
    terminal_raw, terminal = _terminal_bytes(evidence, epoch)
    files[terminal["filename"]] = terminal_raw
    validation, mean = _validation_rows(evidence, epoch)
    validation.update(kind="tgrab_complete_validation_rows_v2", schema_version=2)
    files["validation.json"] = canonical(validation)
    files["provenance.json"] = canonical(provenance)
    descriptors = {name: {"bytes": len(raw), "sha256": digest(raw)} for name, raw in files.items()}
    manifest = {"kind": "tgrab_complete_epoch_candidate_v2", "schema_version": 2, "epoch": epoch,
        "executed_long_contract_sha256": CONTRACT_SHA256, "numerical_template_sha256": common.CONFIG_SHA256,
        "tensor_catalog_sha256": codec.CATALOG_SHA256, "components": components, "terminal_state": terminal,
        "validation_time_mean_f1": mean, "files": descriptors,
        "optimizer_state_included": False, "rng_restore_allowed": False, "test_payload_opened": False}
    files["manifest.json"] = canonical(manifest)
    manifest_hash = digest(files["manifest.json"])
    final = root / f"epoch-{epoch:03d}"
    if final.exists() or final.is_symlink():
        raise LongFitError("candidate publication is create-only")
    stage = Path(tempfile.mkdtemp(prefix=f".epoch-{epoch:03d}-", dir=root))
    try:
        for name, raw in files.items():
            _new(stage / name, raw)
        read_candidate_v2(stage, manifest_hash)
        _sync(stage)
        if final.exists() or final.is_symlink():
            raise LongFitError("candidate destination appeared before publication")
        os.rename(stage, final); _sync(root)
        read_candidate_v2(final, manifest_hash)
    except BaseException:
        # Keep a failed staging namespace for forensic inspection; never retry.
        raise
    return {"epoch": epoch, "manifest_relative_path": f"candidates/epoch-{epoch:03d}/manifest.json",
            "manifest_sha256": manifest_hash, "recurrent_sha256": descriptors["recurrent.f32le"]["sha256"],
            "link_head_sha256": descriptors["link-head.f32le"]["sha256"],
            "terminal_state_sha256": descriptors["validation-terminal.f32le"]["sha256"]}


class PrimitiveSelectionV2:
    """Fresh-only immutable decisions plus one atomic authoritative pointer."""
    def __init__(self, root, *, run_id, admission_hash, contract_hash=CONTRACT_SHA256):
        if (type(run_id) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id) is None
                or type(admission_hash) is not str or re.fullmatch(r"[0-9a-f]{64}", admission_hash) is None
                or contract_hash != CONTRACT_SHA256):
            raise LongFitError("valid run, actual supervisor receipt and frozen v2 contract required")
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        (self.root / "intents").mkdir(mode=0o700)
        (self.root / "receipts").mkdir(mode=0o700)
        _sync(self.root)
        self.context = {"run_id": run_id, "attempt_id": ATTEMPT_ID,
                        "admission_sha256": admission_hash, "executed_long_contract_sha256": contract_hash}
        self.selected = None; self.best_f1 = None; self.last_epoch = 0
        self.pointer_hash = None; self.receipt_reference = None
        self.finalized = False; self.poisoned = False

    def _check(self):
        if self.poisoned:
            raise LongFitError("selection is terminal after failure")
        path = self.root / "selected.json"
        if self.pointer_hash is None:
            if path.exists() or path.is_symlink():
                raise LongFitError("unexpected initial selection pointer")
            return None
        pointer = _json(common.read_pinned_bytes(path, self.pointer_hash, max_bytes=65536))
        if (pointer["selected"] != self.selected or pointer["best_f1"] != self.best_f1
                or pointer["last_epoch"] != self.last_epoch or pointer["finalized"] != self.finalized
                or pointer["receipt"] != self.receipt_reference):
            raise LongFitError("durable pointer and scalar selection disagree")
        return pointer

    def _publish(self, receipt, operation):
        epoch = receipt["last_epoch"]
        name = f"{operation}-{epoch:03d}.json"
        intent = {"kind": "tgrab_long_selection_intent_v2", **self.context, "operation": operation,
                  "previous_pointer_sha256": self.pointer_hash, "receipt": receipt}
        _new(self.root / "intents" / name, canonical(intent))
        receipt_raw = canonical(receipt)
        _new(self.root / "receipts" / name, receipt_raw)
        reference = {"relative_path": "receipts/" + name, "sha256": digest(receipt_raw), "bytes": len(receipt_raw)}
        pointer = {"kind": "tgrab_long_selected_pointer_v2", **self.context, "selected": receipt["selected"],
                   "best_f1": receipt["best_f1"], "last_epoch": epoch, "receipt": reference,
                   "finalized": receipt["finalized"], "previous_pointer_sha256": self.pointer_hash}
        self._check()
        raw = canonical(pointer)
        temporary = self.root / (".selected-" + name)
        _new(temporary, raw)
        os.replace(temporary, self.root / "selected.json"); _sync(self.root)
        self.selected, self.best_f1, self.last_epoch = receipt["selected"], receipt["best_f1"], epoch
        self.finalized = receipt["finalized"]
        self.pointer_hash, self.receipt_reference = digest(raw), reference
        self._check()
        return pointer

    def consider(self, candidate, score, *, selected_flag):
        try:
            self._check()
            epoch = candidate["epoch"]
            keys = {"epoch", "manifest_relative_path", "manifest_sha256", "recurrent_sha256", "link_head_sha256", "terminal_state_sha256"}
            if (set(candidate) != keys or type(epoch) is not int or not 1 <= epoch <= 200
                    or candidate["manifest_relative_path"] != f"candidates/epoch-{epoch:03d}/manifest.json"
                    or any(type(candidate[k]) is not str or re.fullmatch(r"[0-9a-f]{64}", candidate[k]) is None
                           for k in keys - {"epoch", "manifest_relative_path"})):
                raise LongFitError("complete v2 candidate identity required")
            candidate = _json(canonical(candidate))
            if self.finalized or epoch != self.last_epoch + 1 or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise LongFitError("invalid complete-epoch selection sequence/score")
            selected_now = self.best_f1 is None or score > self.best_f1
            if type(selected_flag) is not bool or selected_flag is not selected_now:
                raise LongFitError("native fit and strict earliest-tie selection disagree")
            receipt = {"kind": "tgrab_long_selection_decision_v2", **self.context,
                "epoch": epoch, "last_epoch": epoch, "candidate": candidate, "candidate_f1": score,
                "previous_selected": self.selected, "previous_best_f1": self.best_f1,
                "previous_receipt": self.receipt_reference,
                "selected": candidate if selected_now else self.selected,
                "best_f1": score if selected_now else self.best_f1, "selected_now": selected_now,
                "finalized": False, "test_authorized": False, "created_at_utc": utcnow()}
            return self._publish(receipt, "epoch")
        except BaseException:
            self.poisoned = True
            raise

    def audit(self):
        pointer = self._check()
        if pointer is None:
            return None
        reference = self.receipt_reference
        decisions = []
        while reference is not None:
            relative = Path(reference["relative_path"])
            if relative.is_absolute() or relative.parts[0] != "receipts" or len(relative.parts) != 2 or ".." in relative.parts:
                raise LongFitError("unsafe immutable receipt reference")
            raw = common.read_pinned_bytes(self.root / relative, reference["sha256"], max_bytes=65536)
            if len(raw) != reference["bytes"]:
                raise LongFitError("immutable receipt byte count differs")
            record = _json(raw)
            if any(record.get(key) != value for key, value in self.context.items()):
                raise LongFitError("immutable receipt context changed")
            if record["kind"] == "tgrab_long_selection_decision_v2":
                decisions.append(record)
            elif record["kind"] != "tgrab_long_selection_finalized_v2":
                raise LongFitError("unknown immutable selection record")
            reference = record["previous_receipt"]
        selected, best, epoch = None, None, 0
        for decision in reversed(decisions):
            epoch += 1
            score = decision["candidate_f1"]
            choose = best is None or score > best
            if (decision["epoch"] != epoch or decision["previous_selected"] != selected
                    or decision["previous_best_f1"] != best or decision["selected_now"] is not choose):
                raise LongFitError("immutable selection predecessor or tie rule changed")
            if choose:
                selected, best = decision["candidate"], score
            if decision["selected"] != selected or decision["best_f1"] != best:
                raise LongFitError("immutable selected tuple or score differs")
        if epoch != self.last_epoch or selected != self.selected or best != self.best_f1:
            raise LongFitError("selection history differs from authoritative pointer")
        return {"selected_pointer_sha256": self.pointer_hash, "pointer": pointer, "complete_epochs": epoch}

    def finalize(self, reason, accounting):
        try:
            self.audit()
            if self.finalized or self.selected is None or reason not in {"loss_patience", "complete_epoch_cap", "deadline", "owner_stop"}:
                raise LongFitError("only a valid complete selected candidate can finalize once")
            receipt = {"kind": "tgrab_long_selection_finalized_v2", **self.context,
                "last_epoch": self.last_epoch, "selected": self.selected, "best_f1": self.best_f1,
                "previous_receipt": self.receipt_reference, "finalized": True,
                "stop_reason": reason, "accounting": accounting, "test_authorized": False, "created_at_utc": utcnow()}
            self._publish(receipt, "final")
            return self.audit()
        except BaseException:
            self.poisoned = True
            raise



def summarize_native_result(adapter_return, completed_epochs):
    """Consume the actual native (FitResult, validation_rows) return contract."""
    from .tgrab_figure10_fit import FitResult
    result, rows = adapter_return
    if (type(result) is not FitResult or result.epochs_completed != completed_epochs
            or len(rows) != completed_epochs * 256):
        raise LongFitError("native return differs from complete published epochs")
    reason = {"training_loss_patience": "loss_patience", "complete_epoch_cap": "complete_epoch_cap"}[result.stop_reason]
    return reason, asdict(result)


def run_native(args, owner, guard, reporter, template_raw, contract, admission, admission_hash, runtime):
    import numpy as np
    import torch
    from . import graph_data as gd, graph_reconstruction as gr
    from . import tgrab_selected_checkpoint as codec
    from . import tgrab_figure10_native_fit as native
    from .tgrab_figure10_fit import FitInterrupted
    from .tgrab_figure10_bootstrap import actual_rng_snapshot
    from .tgrab_figure10_candidate_encoder import _component_bytes
    from .tgrab_figure10_chunk_journal import ChunkJournalContext, Figure10ChunkJournal
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256

    guard()
    # V1 validation is used solely to bind unchanged numerical settings and
    # conversion bytes. Actual executed runtime/admission is v2 throughout.
    template = _json(template_raw)
    gr.validate_frozen_config(template)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    modules = load_pinned_graph_modules(source_root=args.source_root, installed_cell_path=args.cell_path,
        cell_sha256=PINNED_CELL_SHA256, wrapper_sha256=PINNED_WRAPPER_SHA256, guard=guard)
    conversion = _json(common.read_pinned_bytes(args.conversion_receipt, args.conversion_receipt_sha256, max_bytes=1_000_000))
    manifest = gd.read_manifest(args.data_manifest, args.data_manifest_sha256, guard)
    check_conversion(conversion, manifest.metadata, manifest.sha256)
    if manifest.metadata["raw"]["sha256"] != contract["data"]["raw_sha256"]:
        raise LongFitError("v2 input differs from frozen accepted graph bytes")
    store = gd.open_train_validation(manifest, guard, snapshot_factory=gr.Snapshot)
    reference_rows = load_prefix_reference(args.prefix_updates, args.prefix_updates_sha256)
    parity = PrefixParity(reference_rows, owner.run_dir, args.prefix_updates_sha256)
    _, catalog = codec._schema_and_catalog()  # unchanged encoding catalog only
    observations = {}

    def seed_all(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        snapshot = actual_rng_snapshot(torch)
        observations["seeding_record"] = write_record(owner.run_dir, "seeding.json", snapshot, "seeding_before_models")
        return native.SeedApplicationReceipt(seed, ("python", "numpy", "torch_cpu"), snapshot["sha256"])

    def model_factory():
        guard()
        models = {"node_emb": modules.wrapper_type(1, modules.parameter_type(100, 128, 2)).to(device="cpu", dtype=torch.float32),
                  "link_pred": modules.head_type(128, num_layers=2, p_dropout=0.2).to(device="cpu", dtype=torch.float32)}
        observations["post_model_rng_record"] = write_record(owner.run_dir, "post-model-rng.json", actual_rng_snapshot(torch), "post_model_construction_rng")
        observations["initial_component_semantic_sha256"] = {}
        for component, key in (("recurrent", "node_emb"), ("link_head", "link_pred")):
            _, descriptor = _component_bytes(component, models[key].state_dict(), catalog)
            observations["initial_component_semantic_sha256"][component] = codec.component_semantic_sha256(descriptor["tensors"])
        return models

    def optimizer_factory(models):
        parameters = [p for key in ("node_emb", "link_pred") for _, p in sorted(models[key].named_parameters())]
        return torch.optim.Adam(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)

    adapter = native.build_figure10_fit_adapter(model_seed=0, seed_all=seed_all, model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        backend_factory=lambda models, optimizer: gr.NativeGCLSTMBackend(torch, models, optimizer, device="cpu"),
        training_graphs=store.train(), validation_graphs=store.validation())
    runtime["model_constructed"] = True
    observations["negative_generator_record"] = write_record(owner.run_dir, "negative-generator.json",
        {"seed": 1, "bit_generator": adapter.negative_rng.bit_generator.state}, "negative_generator_initialization")
    modes = {}
    with adapter.backend.training_mode():
        modes["train_flags"] = {key: {name: bool(module.training) for name, module in model.named_modules()}
                                for key, model in adapter.backend.models.items()}
    with adapter.backend.evaluation_mode():
        modes["eval_flags"] = {key: {name: bool(module.training) for name, module in model.named_modules()}
                               for key, model in adapter.backend.models.items()}
        modes["validation_grad_enabled"] = torch.is_grad_enabled()
    if (not all(v for flags in modes["train_flags"].values() for v in flags.values())
            or any(v for flags in modes["eval_flags"].values() for v in flags.values())
            or modes["validation_grad_enabled"] is not False):
        raise LongFitError("actual train/eval/dropout modes differ from frozen numerical policy")
    mode_ref = write_record(owner.run_dir, "module-modes.json", modes)
    reporting_ref = write_record(owner.run_dir, "reporting-identity.json", {
        "kind": "tgrab_long_reporting_identity_v2", "run_id": args.run_id, "mode": "offline",
        "project": "representation-collapse", "initial_receipt_sha256": common.sha256_file(reporter.receipt_path),
        "actual_supervisor_receipt_sha256": admission_hash})
    md = manifest.metadata
    identities = {"executed_long_contract_sha256": args.long_contract_sha256,
        "numerical_template_sha256": args.config_sha256, "conversion_config_sha256": md["config_sha256"],
        "converted_manifest_sha256": manifest.sha256, "raw_sha256": md["raw"]["sha256"],
        "conversion_receipt_sha256": args.conversion_receipt_sha256,
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "node2a_receipt_sha256": args.node2a_receipt_sha256,
        "prefix_reference_sha256": args.prefix_updates_sha256, "tensor_catalog_sha256": codec.CATALOG_SHA256,
        "actual_supervisor_receipt_sha256": admission_hash,
        "source_closure_sha256": digest(canonical(admission["closure_sha256"])),
        "reporting_identity_sha256": reporting_ref["sha256"]}
    write_record(owner.run_dir, "run-identities.json", {"kind": "tgrab_long_fit_identity_v2", **identities,
        "source_commit": args.source_commit, "source_closure": admission["closure_sha256"],
        "data_split_metadata": md["splits"], "converter": md["converter"],
        "import_provenance": modules.provenance_dict()})
    environment = {"python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(), "versions": common.environment_versions(),
        "requirements_lock_sha256": common.ENVIRONMENT_LOCK_SHA256, "wheel_manifest_sha256": common.WHEEL_MANIFEST_SHA256}
    base = {"kind": "tgrab_long_fit_candidate_provenance_v2", "schema_version": 2,
        "run_id": args.run_id, "attempt_id": ATTEMPT_ID, "source_commit": args.source_commit,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "identities": identities,
        "executed_long_contract_sha256": args.long_contract_sha256, "numerical_template_sha256": args.config_sha256,
        "conversion_config_sha256": md["config_sha256"], "initialization": observations,
        "model_seed": 0, "negative_seed": 1, "seed_before_construction": True,
        "construction_order": ["seed_all", "recurrent", "link_head", "optimizer", "negative_generator"],
        "module_modes": mode_ref, "module_state_dict_versions": {component:
            {name: meta["version"] for name, meta in adapter.backend.models[key].state_dict()._metadata.items()}
            for component, key in (("recurrent", "node_emb"), ("link_head", "link_pred"))},
        "optimizer_parameter_order": [name for name, _ in native._ordered_named_parameters(adapter.backend.models)],
        "runtime_environment": environment, "actual_job": admission["job"],
        "runtime_policy": {"device": "cpu", "precision": "float32", "cpu_threads": torch.get_num_threads(),
            "amp": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "work_seconds": WORK_SECONDS, "slurm_seconds": 172800},
        "optimizer_state_included": False, "rng_restore_allowed": False, "test_payload_opened": False}
    candidates = owner.run_dir / "candidates"; candidates.mkdir(mode=0o700)
    journal_root = owner.run_dir / "optimizer-chunks"; journal_root.mkdir(mode=0o700)
    journal = Figure10ChunkJournal(journal_root, ChunkJournalContext(args.run_id, ATTEMPT_ID,
        args.long_contract_sha256, admission_hash))
    selection = PrimitiveSelectionV2(owner.run_dir / "selection", run_id=args.run_id, admission_hash=admission_hash)
    events = DurableEvents(owner.run_dir, reporter)
    epoch_records = []
    accounting = None
    try:
        def attempt(record):
            runtime["attempt_callback_entered"] = record["sequence"]
            journal.on_attempt(record)
        def update(record):
            journal.on_update(record)
            values = {"train/attempted_updates": record["attempted_updates"], "train/completed_updates": record["completed_updates"],
                "train/epoch": record["epoch"], "train/target_timestamp": record["target_timestamp"]}
            if "loss" in record:
                values["train/loss"] = record["loss"]
            events.emit("optimizer_update", record, values)
            parity.on_update(record)
        def complete(evidence):
            if parity.compared != 256:
                raise LongFitError("complete candidate cannot precede accepted native prefix parity")
            epoch = evidence.record.epoch
            record = asdict(evidence.record)
            write_record(owner.run_dir, f"validation-{epoch:03d}.json",
                {"record": record, "rows": [dict(row) for row in evidence.validation_rows]})
            seal = journal.seal_epoch(epoch)
            provenance = {**base, "complete_epoch": epoch, "complete_epoch_counters": {
                "complete_epochs": epoch, "attempted_updates": epoch * 2559, "completed_updates": epoch * 2559,
                "optimizer_steps_confirmed": epoch * 2559, "validation_epochs": epoch},
                "journal_epoch_seal": seal, "created_at_utc": utcnow()}
            reference = write_candidate_v2(candidates, evidence, provenance)
            selection.consider(reference, record["validation_mean_f1"], selected_flag=record["validation_selected"])
            epoch_records.append(record)
            for row in evidence.validation_rows:
                row = dict(row)
                events.emit("validation_row", row, {"validation/" + key: row[key] for key in
                    ("epoch", "target_timestamp", "source_timestamp", "f1", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn")})
            events.emit("complete_epoch", record, {"epoch/index": epoch, "epoch/train_loss": record["mean_training_loss"],
                "epoch/validation_f1": record["validation_mean_f1"], "epoch/selected": record["validation_selected"]})
        def selected(record):
            if not record.validation_selected or selection.last_epoch != record.epoch or selection.selected["epoch"] != record.epoch:
                raise LongFitError("native selection callback differs from committed v2 candidate")
        runtime["training_entered"] = True
        try:
            reason, summary = summarize_native_result(adapter.fit(guard=guard, on_attempt=attempt, on_update=update,
                on_epoch_complete=complete, on_selection=selected), len(epoch_records))
        except FitInterrupted as error:
            if not isinstance(error.__cause__, FitStop):
                raise
            reason = error.__cause__.reason
            summary = {"stop_reason": reason, "epochs_completed": len(epoch_records),
                "attempted_updates": error.attempted_updates, "completed_updates": error.completed_updates,
                "optimizer_steps_confirmed": error.optimizer_steps_confirmed,
                "interrupted_epoch": error.epoch, "partial_epoch": error.epoch > len(epoch_records),
                "interrupted_epoch_validation_completed_at_guard": error.epoch_complete, "history": epoch_records}
        accounting = journal.audit()  # one full history audit, not one per update/epoch
        runtime["accounting"] = accounting
        if any(summary[key] != accounting[key] for key in ("attempted_updates", "completed_updates", "optimizer_steps_confirmed")):
            raise LongFitError("native and durable optimizer accounting disagree")
        finalized = selection.finalize(reason, accounting) if selection.selected is not None else None
        summary.update(kind=PROTOCOL_ID, status="fit_stopped" if reason in {"deadline", "owner_stop"} else "fit_complete",
            stop_reason=reason, selected=selection.selected, selected_validation_f1=selection.best_f1,
            selected_pointer_sha256=selection.pointer_hash, finalized=finalized is not None,
            validation_epochs_completed=len(epoch_records), history=epoch_records,
            journal_accounting=accounting, prefix_exact_loss_matches=parity.compared,
            identities=identities, source_commit=args.source_commit,
            source_shape={"hidden": [1, 100, 128], "cell": [1, 100, 128]},
            test_payload_opened=False, selected_load_attempts=0, scientific_acceptance=False,
            figure10_reproduction=False, reporting_events=events.sequence)
        summary_ref = write_record(owner.run_dir, "fit-summary.json", summary)
        reporter.log_manifest("long-fit-result", metadata={"status": summary["status"], "stop_reason": reason,
            "complete_epochs": len(epoch_records), "completed_updates": accounting["completed_updates"],
            "test_payload_opened": False, "scientific_acceptance": False},
            identities={**identities, "fit_summary_sha256": summary_ref["sha256"]})
        return summary
    except BaseException as error:
        if accounting is None:
            try:
                accounting = journal.audit()
                runtime["accounting"] = accounting
            except BaseException as audit_error:
                # A failed fsync may leave bytes without advancing RAM counters.
                # Never call that zero/unattempted or durably completed.
                runtime["accounting"] = {"durable_accounting_verified": False,
                    "attempted_updates_charged_lower_bound": max(runtime.get("attempt_callback_entered", 0), journal.sequence),
                    "completed_updates": None, "optimizer_steps_confirmed": None,
                    "pending_or_ambiguous_update": True, "audit_error_type": type(audit_error).__name__}
        runtime["in_memory_selection_last_epoch_lower_bound"] = selection.last_epoch
        runtime["selected_candidate_retained"] = selection.selected
        raise
    finally:
        journal.close(); events.close()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-manifest", "environment-lock", "wheel-manifest", "environment-receipt", "node2a-receipt",
                 "source-root", "cell-path", "run-dir", "stop-file", "conversion-receipt", "long-contract", "prefix-updates"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "data-manifest-sha256", "environment-receipt-sha256", "node2a-receipt-sha256",
                 "conversion-receipt-sha256", "long-contract-sha256", "prefix-updates-sha256", "run-id", "source-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--runtime-seconds", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(argv)
    if (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id) is None
            or re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None):
        raise LongFitError("valid immutable run/source identity required")
    contract = load_contract(args.long_contract, args.long_contract_sha256)
    if args.config_sha256 != common.CONFIG_SHA256 or args.prefix_updates_sha256 != PREFIX_SHA256:
        raise LongFitError("exact numerical template and accepted prefix required")
    template_raw = common.read_pinned_bytes(args.config, args.config_sha256, max_bytes=2_000_000)
    common.verify_pin(args.environment_lock, common.ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
    common.verify_pin(args.wheel_manifest, common.WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
    common.validate_environment_receipt(_json(common.read_pinned_bytes(args.environment_receipt, args.environment_receipt_sha256, max_bytes=1_000_000)))
    common.validate_node2a_receipt(_json(common.read_pinned_bytes(args.node2a_receipt, args.node2a_receipt_sha256, max_bytes=1_000_000)))
    common.environment_versions()
    if platform.python_version() != "3.10.19":
        raise LongFitError("exact pinned CPython3.10.19 runtime required")
    for name, expected in contract["input_pins"]["source_files_sha256"].items():
        common.verify_pin(Path(__file__).parent / name, expected, max_bytes=2_000_000)
    for relative, expected in common.WRAPPER_SHA256.items():
        common.verify_pin(args.source_root / relative, expected, max_bytes=64_000)
    common.verify_pin(args.cell_path, common.CELL_SHA256, max_bytes=64_000)
    supervisor_path = os.environ.get("RC_SUPERVISOR_RECEIPT")
    supervisor_hash = os.environ.get("RC_SUPERVISOR_RECEIPT_SHA256")
    if not supervisor_path or not supervisor_hash:
        raise LongFitError("admitted stdlib long-fit supervisor required")
    admission = _json(common.read_pinned_bytes(Path(supervisor_path), supervisor_hash, max_bytes=2_000_000))
    validate_supervisor(admission, args, argv)
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time():
        raise LongFitError("admitted work deadline expired")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise LongFitError("single-thread CPU environment required: " + key)
    owner = common.ProcessOwner.acquire(args.run_dir)
    interrupted = [False]
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    runtime = {"model_constructed": False, "training_entered": False, "attempt_callback_entered": 0}
    def guard():
        owner.check()
        if interrupted[0] or os.path.lexists(args.stop_file):
            raise FitStop("owner_stop")
        if time.time() >= args.deadline_unix:
            raise FitStop("deadline")
    final = {"kind": PROTOCOL_ID, "status": "failed", "run_id": args.run_id, "attempt_id": ATTEMPT_ID,
        "source_commit": args.source_commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "actual_supervisor_receipt_sha256": supervisor_hash, "executed_long_contract_sha256": args.long_contract_sha256,
        "numerical_template_sha256": args.config_sha256, "started_at_utc": utcnow(),
        "test_payload_opened": False, "scientific_acceptance": False, "figure10_reproduction": False}
    try:
        from .reporting import ExperimentReporter
        with ExperimentReporter(owner.run_dir / "wandb", project="representation-collapse", run_id=args.run_id, mode="offline",
            config={"protocol": PROTOCOL_ID, "model_seed": 0, "negative_seed": 1, "device": "cpu", "precision": "float32",
                    "max_complete_epochs": 200, "loss_patience": 100, "work_seconds": WORK_SECONDS, "test_access": False},
            identities={"executed_long_contract_sha256": args.long_contract_sha256, "numerical_template_sha256": args.config_sha256,
                        "data_manifest_sha256": args.data_manifest_sha256, "actual_supervisor_receipt_sha256": supervisor_hash,
                        "prefix_reference_sha256": args.prefix_updates_sha256}) as reporter:
            final.update(run_native(args, owner, guard, reporter, template_raw, contract, admission, supervisor_hash, runtime))
    except BaseException as error:
        final.update(status="failed", error_type=type(error).__name__, runtime_accounting=runtime)
    finally:
        signal.signal(signal.SIGTERM, previous)
        final["finished_at_utc"] = utcnow()
        final["deadline_unix"] = args.deadline_unix
        receipt = owner.run_dir / "wandb/reporting-receipt.json"
        if receipt.exists():
            final["reporting_receipt_sha256"] = common.sha256_file(receipt)
        common.publish_terminal_manifest(owner, final)
    print(json.dumps(final, sort_keys=True))
    return 0 if final["status"] in {"fit_complete", "fit_stopped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
