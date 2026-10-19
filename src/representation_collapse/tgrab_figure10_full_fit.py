"""Single-use admitted CPU Figure 10 fit; validation selection only, no test I/O.

The outer stdlib supervisor authenticates the complete source closure and Slurm
allocation. This worker retains raw optimizer/validation evidence before W&B,
commits complete epoch candidates, and never retries or resumes a partial fit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import sys
import time

from . import tgrab_figure10_canary as common

PROTOCOL_ID = "tgrab_figure10_full_fit_v1"
WORK_SECONDS = 3480
OUTER_SECONDS = 3570
SLURM_SECONDS = 3600
ATTEMPT_ID = "fit01"


class FullFitError(RuntimeError):
    """Admission or fit evidence failed verification."""


class FitStop(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_record(root, name, value, role=None):
    """Create a retained JSON observation and fsync before returning its reference."""
    raw = canonical(value)
    path = root / name
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    result = {"relative_path": name, "bytes": len(raw), "sha256": digest(raw)}
    if role:
        result.update(role=role, encoding="primitive_json")
    return result


class DurableEvents:
    """Raw fsync precedes reporting; event steps never reuse optimizer counts."""
    def __init__(self, root, reporter):
        self.path = root / "fit-events.jsonl"
        self.stream = self.path.open("x", encoding="ascii")
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.reporter = reporter
        self.sequence = 0

    def emit(self, kind, record, metrics):
        event = {"event": self.sequence, "kind": kind, "record": record}
        self.stream.write(canonical(event).decode("ascii") + "\n")
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.reporter.log(metrics, step=self.sequence)
        self.sequence += 1

    def close(self):
        self.stream.close()


def validate_supervisor(receipt, args, argv, environ):
    if (receipt.get("schema") != "tgrab-figure10-supervisor-receipt-v2"
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("run_id") != args.run_id
            or receipt.get("job_id") != environ.get("SLURM_JOB_ID")
            or receipt.get("source_sha256") != common.sha256_file(Path(__file__))
            or receipt.get("canary_args") != list(argv)):
        raise FullFitError("supervisor run/source/argv identity mismatch")
    started = receipt.get("wall_started_epoch_ns")
    if (type(started) is not int or started <= 0 or args.runtime_seconds != WORK_SECONDS
            or receipt.get("work_seconds") != WORK_SECONDS
            or receipt.get("deadline_unix") != args.deadline_unix
            or args.deadline_unix != started / 1e9 + WORK_SECONDS):
        raise FullFitError("supervisor runtime deadline mismatch")
    job = receipt.get("job", {})
    for key, env in {"job_id": "SLURM_JOB_ID", "account": "SLURM_JOB_ACCOUNT",
                     "partition": "SLURM_JOB_PARTITION", "cpus_on_node": "SLURM_CPUS_ON_NODE"}.items():
        if not job.get(key) or (env in environ and str(job[key]) != environ[env]):
            raise FullFitError("Slurm identity mismatch: " + key)
    if (str(job.get("cpus_on_node")) != "1" or job.get("account") != "mila"
            or job.get("partition") != "main-cpu"):
        raise FullFitError("full fit requires the registered one-CPU Mila allocation")
    for key in ("environment_receipt", "node2a_receipt"):
        if receipt.get(key) != {"path": str(getattr(args, key)), "sha256": getattr(args, key + "_sha256")}:
            raise FullFitError("supervisor receipt differs: " + key)
    child = receipt.get("child_argv")
    if (receipt.get("sys_executable") != sys.executable or type(child) is not list
            or child[:3] != [sys.executable, "-B", "-c"]
            or child[-len(argv):] != list(argv)
            or receipt.get("argv_sha256") != digest(json.dumps(child, separators=(",", ":")).encode())):
        raise FullFitError("supervisor bootstrap command mismatch")


def check_conversion(receipt, metadata, manifest_sha):
    if (receipt.get("kind") != "tgrab_primary_conversion_independent_verification_v1"
            or receipt.get("status") != "verified" or receipt.get("primary_eligible") is not True
            or receipt.get("independent_of_producer") is not True
            or receipt.get("manifest_sha256") != manifest_sha
            or receipt.get("raw_sha256") != metadata["raw"]["sha256"]
            or receipt.get("converter_sha256") != metadata["converter"]["sha256"]
            or receipt.get("config_sha256") != common.CONFIG_SHA256
            or receipt.get("snapshot_count") != 3072):
        raise FullFitError("independent conversion receipt does not bind this dataset")


def assemble_identities(*, metadata, config_hash, manifest_hash, conversion_hash,
                        reporting_hash, admission_hash, codec_hash, trainer_hash,
                        environment_hash, schema, catalog_hash, schema_hash,
                        conversion_protocol_hash):
    """Bind actual manifest identities; test references are metadata, never payloads."""
    if (metadata["features"] != "float32_identity"
            or metadata["weights"] != "float32_unit_aligned_with_pairs"
            or metadata["node_order"] != "ascending_integer_ids"):
        raise FullFitError("dataset encoding differs from frozen semantic contract")
    ids = {"executed_config_sha256": config_hash, "schema_sha256": schema_hash,
           "codec_source_sha256": codec_hash, "trainer_source_sha256": trainer_hash,
           "environment_receipt_sha256": environment_hash,
           "converted_manifest_sha256": manifest_hash,
           "conversion_receipt_sha256": conversion_hash, "admission_sha256": admission_hash,
           "reporting_identity_sha256": reporting_hash, "tensor_catalog_sha256": catalog_hash,
           "raw_sha256": metadata["raw"]["sha256"], "reconstruction_config_sha256": config_hash,
           "converter_sha256": metadata["converter"]["sha256"],
           "conversion_protocol_sha256": conversion_protocol_hash}
    # Semantic hashes are versioned contract identifiers. They are valid only
    # after the corresponding actual manifest encodings have passed above.
    contract = schema["$defs"]["manifest"]["properties"]["identities"]["properties"]
    for key in ("node_order_semantic_sha256", "feature_encoding_semantic_sha256", "weight_encoding_semantic_sha256"):
        ids[key] = contract[key]["const"]
    for split, record in metadata["splits"].items():
        ids[split + "_file_sha256"] = record["sha256"]
        ids[split + "_semantic_sha256"] = record["semantic_sha256"]
    return ids


def finalize_selection(ledger, selection, reason):
    if selection.selected is None:
        return None
    def apply(record):
        return selection.finalize()
    def verify(applied, record):
        return (selection.finalized is True and asdict(applied) == record["selected"]
                and selection.best_f1 == record["best_f1"] and selection.last_epoch == record["last_epoch"])
    return ledger.finalize(transaction_id="finalize-fit01", run_stop_reason=reason,
                           apply_selection=apply, verify_selection=verify, created_at_utc=utcnow())


def run_native(args, owner, guard, reporter, config_raw, admission_hash, conversion, report_identity):
    # Numerical/native imports occur only after mandatory reporter admission.
    import numpy as np
    import torch
    from . import graph_data as gd, graph_reconstruction as gr
    from . import tgrab_selected_checkpoint as codec
    from . import tgrab_figure10_native_fit as native
    from .tgrab_figure10_fit import FitInterrupted
    from .tgrab_figure10_attempt_journal import Figure10AttemptJournal
    from .tgrab_figure10_selection_sink import Figure10SelectionSink
    from .tgrab_selection_transaction import SelectionContext, SelectionLedger
    from .tgrab_figure10_bootstrap import actual_rng_snapshot
    from .tgrab_figure10_candidate_encoder import _component_bytes
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256

    guard()
    config = json.loads(config_raw)
    gr.validate_frozen_config(config)
    schema, catalog = codec._schema_and_catalog()
    module_dir = Path(__file__).parent
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    modules = load_pinned_graph_modules(source_root=args.source_root, installed_cell_path=args.cell_path,
        cell_sha256=PINNED_CELL_SHA256, wrapper_sha256=PINNED_WRAPPER_SHA256, guard=guard)
    manifest = gd.read_manifest(args.data_manifest, args.data_manifest_sha256, guard)
    md = manifest.metadata
    check_conversion(conversion, md, manifest.sha256)
    store = gd.open_train_validation(manifest, guard, snapshot_factory=gr.Snapshot)
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
        observations["initial_components"] = {}
        for name, key in (("recurrent", "node_emb"), ("link_head", "link_pred")):
            _, descriptor = _component_bytes(name, models[key].state_dict(), catalog)
            observations["initial_components"][name] = codec.component_semantic_sha256(descriptor["tensors"])
        return models

    def optimizer_factory(models):
        return torch.optim.Adam([p for key in ("node_emb", "link_pred") for _, p in sorted(models[key].named_parameters())],
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)

    adapter = native.build_figure10_fit_adapter(model_seed=0, seed_all=seed_all,
        model_factory=model_factory, optimizer_factory=optimizer_factory,
        backend_factory=lambda models, optimizer: gr.NativeGCLSTMBackend(torch, models, optimizer, device="cpu"),
        training_graphs=store.train(), validation_graphs=store.validation())
    observations["negative_generator_record"] = write_record(owner.run_dir, "negative-generator.json",
        {"seed": 1, "bit_generator": adapter.negative_rng.bit_generator.state}, "negative_generator_initialization")
    mode = {}
    with adapter.backend.training_mode():
        mode["train_flags"] = {key: {name: bool(m.training) for name, m in model.named_modules()}
                               for key, model in adapter.backend.models.items()}
    with adapter.backend.evaluation_mode():
        mode["eval_flags"] = {key: {name: bool(m.training) for name, m in model.named_modules()}
                              for key, model in adapter.backend.models.items()}
        mode["validation_grad_enabled"] = torch.is_grad_enabled()
    if (not all(flag for values in mode["train_flags"].values() for flag in values.values())
            or any(flag for values in mode["eval_flags"].values() for flag in values.values())
            or mode["validation_grad_enabled"]):
        raise FullFitError("actual native train/eval modes disagree with frozen policy")
    mode_ref = write_record(owner.run_dir, "module-modes.json", mode)
    reporting_ref = write_record(owner.run_dir, "reporting-identity.json", report_identity)
    sources = {"trainer": common.sha256_file(Path(native.__file__)),
        "codec": common.sha256_file(Path(codec.__file__)), "reporter": common.REPORTER_SHA256,
        "graph_reconstruction": common.GRAPH_RECONSTRUCTION_SHA256, "graph_data": common.GRAPH_DATA_SHA256,
        "graph_import_helper": common.IMPORT_HELPER_SHA256, "recurrent_wrapper": common.WRAPPER_SHA256["model/gclstm.py"],
        "link_head": common.WRAPPER_SHA256["model/link_predictor.py"], "cell": common.CELL_SHA256,
        "schema": codec.SCHEMA_SHA256}
    protocol_path = Path(__file__).resolve().parents[2] / "doc/auto/tgrab-primary-data-conversion-v1.md"
    protocol_hash = common.sha256_file(protocol_path)
    identities = assemble_identities(metadata=md, config_hash=args.config_sha256, manifest_hash=manifest.sha256,
        conversion_hash=args.conversion_receipt_sha256, reporting_hash=reporting_ref["sha256"],
        admission_hash=admission_hash, codec_hash=sources["codec"], trainer_hash=sources["trainer"],
        environment_hash=args.environment_receipt_sha256, schema=schema, catalog_hash=codec.CATALOG_SHA256,
        schema_hash=codec.SCHEMA_SHA256, conversion_protocol_hash=protocol_hash)
    versions = {"python": platform.python_version(), **{key.replace("-", "_"): value for key, value in common.environment_versions().items()}}
    initialization = {"model_seed": 0, "negative_seed": 1, "seed_before_construction": True,
        "construction_order": ["seed_all", "recurrent", "link_head", "optimizer", "negative_generator"],
        **{key: observations[key] for key in ("seeding_record", "post_model_rng_record", "negative_generator_record")},
        "initial_recurrent_semantic_sha256": observations["initial_components"]["recurrent"],
        "initial_head_semantic_sha256": observations["initial_components"]["link_head"]}
    base = {"kind": "tgrab_selected_inference_provenance_v1", "schema_version": 1,
        "run_id": args.run_id, "attempt_id": ATTEMPT_ID, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source_commit": args.source_commit, "source_sha256": sources, "executed_config_sha256": args.config_sha256,
        "candidate_sha256": digest(canonical(config["candidate"])), "reconstruction_config_sha256": args.config_sha256,
        "environment": {"platform": "linux-x86_64", "implementation": platform.python_implementation(), "versions": versions,
            "environment_receipt_sha256": args.environment_receipt_sha256,
            "wheel_manifest_sha256": common.WHEEL_MANIFEST_SHA256, "requirements_lock_sha256": common.ENVIRONMENT_LOCK_SHA256},
        "initialization": initialization,
        "runtime_policy": {"device": "cpu", "precision": "float32", "amp": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cpu_threads": torch.get_num_threads(), "recurrent_dropout": 0.0, "head_dropout_train": 0.2,
            "head_dropout_eval_active": False, "validation_grad_enabled": mode["validation_grad_enabled"],
            "reset": "zero_hidden_and_cell_each_eval_replay", "module_mode_receipt_sha256": mode_ref["sha256"]},
        "optimizer_parameter_order": [name for name, _ in native._ordered_named_parameters(adapter.backend.models)],
        "module_state_dict_versions": {name:
            {n: meta["version"] for n, meta in adapter.backend.models[key].state_dict()._metadata.items()}
            for name, key in (("recurrent", "node_emb"), ("link_head", "link_pred"))},
        "validation_chronology": {"conditioning_interval": [0,2560], "targets_interval": [2560,2816],
            "last_consumed_t": 2815, "next_target_t": 2816, "score_before_target_consume": True, "weight_epoch_is_fixed": True},
        "admission_sha256": admission_hash, "reporting_identity_sha256": reporting_ref["sha256"],
        "optimizer_state_included": False, "rng_restore_allowed": False}

    def provenance(evidence):
        epoch = evidence.record.epoch
        return {**base, "created_at_utc": utcnow(), "counters": {"complete_epochs": epoch, "validation_epochs": epoch,
            "attempted_updates": epoch * 2559, "completed_updates": epoch * 2559,
            "selected_load_attempts": 0, "test_payload_read_attempts": 0}}
    # Validate the entire provenance before the first optimizer entry, so a
    # shape/version/schema incompatibility cannot waste a complete epoch.
    from types import SimpleNamespace
    probe = provenance(SimpleNamespace(record=SimpleNamespace(epoch=1)))
    codec._schema_validate(probe, schema["$defs"]["provenance"], schema)
    codec._schema_validate(identities, schema["$defs"]["manifest"]["properties"]["identities"], schema)
    selection = gr.ValidationOnlySelection()
    ledger_root = owner.run_dir / "checkpoint-ledger"; ledger_root.mkdir(mode=0o700)
    attempts_root = owner.run_dir / "optimizer-attempts"; attempts_root.mkdir(mode=0o700)
    context = SelectionContext(args.run_id, ATTEMPT_ID, admission_hash, codec.SCHEMA_SHA256, None)
    events = DurableEvents(owner.run_dir, reporter)
    epoch_records = []
    try:
        with SelectionLedger(ledger_root, context) as ledger:
            sink = Figure10SelectionSink(ledger=ledger, selection=selection, identity_map=identities,
                provenance_factory=provenance, transaction_id_factory=lambda e: f"epoch-{e:03d}", timestamp_factory=lambda e: utcnow())
            journal = Figure10AttemptJournal(attempts_root, context, namespace_audit_interval=256)
            def update(record):
                journal.on_update(record)
                metrics = {"train/attempted_updates": record["attempted_updates"], "train/completed_updates": record["completed_updates"],
                           "train/epoch": record["epoch"], "train/target_timestamp": record["target_timestamp"]}
                if "loss" in record:
                    metrics["train/loss"] = record["loss"]
                events.emit("optimizer_update", record, metrics)
            def epoch(evidence):
                # Raw complete epoch and all 256 validation rows are durable
                # before candidate serialization and any reporter side effect.
                record = asdict(evidence.record)
                write_record(owner.run_dir, f"validation-{evidence.record.epoch:03d}.json",
                    {"record": record, "rows": [dict(row) for row in evidence.validation_rows]})
                journal.audit()
                sink.on_epoch_complete(evidence)
                epoch_records.append(record)
                for row in evidence.validation_rows:
                    row = dict(row)
                    events.emit("validation_row", row, {
                        "validation/epoch": row["epoch"],
                        "validation/target_timestamp": row["target_timestamp"],
                        "validation/source_timestamp": row["source_timestamp"],
                        **{"validation/" + key: row[key] for key in
                           ("f1", "precision", "recall", "accuracy", "tp", "fp", "fn", "tn")}})
                events.emit("complete_epoch", record, {"epoch/index": record["epoch"],
                    "epoch/train_loss": record["mean_training_loss"], "epoch/validation_f1": record["validation_mean_f1"],
                    "epoch/selected": record["validation_selected"]})
            try:
                result = adapter.fit(guard=guard, on_attempt=journal.on_attempt, on_update=update,
                    on_epoch_complete=epoch, on_selection=sink.on_selection)
                stop_reason = {"training_loss_patience": "loss_patience", "complete_epoch_cap": "complete_epoch_cap"}[result.stop_reason]
                summary = asdict(result)
                journal.audit()
            except FitInterrupted as exc:
                if not isinstance(exc.__cause__, FitStop):
                    raise
                stop_reason = exc.__cause__.reason
                summary = {"stop_reason": stop_reason, "epochs_completed": len(epoch_records),
                    "attempted_updates": exc.attempted_updates, "completed_updates": exc.completed_updates,
                    "optimizer_steps_confirmed": exc.optimizer_steps_confirmed, "interrupted_epoch": exc.epoch,
                    "partial_epoch": exc.epoch > len(epoch_records),
                    "interrupted_epoch_validation_completed_at_guard": exc.epoch_complete, "history": epoch_records}
                journal.audit()
            finally:
                journal.close()
            finalization = finalize_selection(ledger, selection, stop_reason)
            head = ledger.verify_committed_state()
            summary.update(status="fit_stopped" if stop_reason in {"deadline", "owner_stop"} else "fit_complete",
                stop_reason=stop_reason, test_payload_opened=False, selected_load_attempts=0,
                scientific_acceptance=False, figure10_reproduction=False,
                selected=asdict(selection.selected) if selection.selected else None,
                selected_validation_f1=selection.best_f1, finalized=bool(finalization), ledger_head=head,
                validation_epochs_completed=len(epoch_records), reporting_events=events.sequence,
                identities=identities)
            write_record(owner.run_dir, "fit-summary.json", summary)
            return summary
    finally:
        events.close()


def reconcile_failure(attempts_root, error):
    """Never promote a visible final outcome after a failed journal barrier."""
    from .tgrab_figure10_attempt_journal import AttemptJournalError
    pending, seen, journal_failed = [error], set(), False
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        journal_failed = journal_failed or isinstance(current, AttemptJournalError)
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
    return common.reconcile_attempts(attempts_root, terminal_journal_error=journal_failed)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-manifest", "environment-lock", "wheel-manifest", "environment-receipt", "node2a-receipt",
                 "source-root", "cell-path", "run-dir", "stop-file", "conversion-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "data-manifest-sha256", "environment-receipt-sha256", "node2a-receipt-sha256",
                 "conversion-receipt-sha256", "run-id", "source-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--runtime-seconds", type=int, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id) or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise FullFitError("invalid immutable run/source identity")
    if args.config_sha256 != common.CONFIG_SHA256:
        raise FullFitError("frozen v1 configuration required")
    config_raw = common.read_pinned_bytes(args.config, args.config_sha256, max_bytes=2_000_000)
    common.verify_pin(args.environment_lock, common.ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
    common.verify_pin(args.wheel_manifest, common.WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
    environment = json.loads(common.read_pinned_bytes(args.environment_receipt, args.environment_receipt_sha256, max_bytes=1_000_000))
    common.validate_environment_receipt(environment)
    common.validate_node2a_receipt(json.loads(common.read_pinned_bytes(args.node2a_receipt, args.node2a_receipt_sha256, max_bytes=1_000_000)))
    conversion = json.loads(common.read_pinned_bytes(args.conversion_receipt, args.conversion_receipt_sha256, max_bytes=1_000_000))
    for relative, expected in common.WRAPPER_SHA256.items():
        common.verify_pin(args.source_root / relative, expected, max_bytes=64_000)
    module_dir = Path(__file__).parent
    for name, expected in (("graph_data.py", common.GRAPH_DATA_SHA256), ("graph_reconstruction.py", common.GRAPH_RECONSTRUCTION_SHA256),
                           ("pinned_graph_import.py", common.IMPORT_HELPER_SHA256), ("reporting.py", common.REPORTER_SHA256)):
        common.verify_pin(module_dir / name, expected, max_bytes=100_000)
    common.environment_versions()
    receipt_path = os.environ.get("RC_SUPERVISOR_RECEIPT")
    receipt_hash = os.environ.get("RC_SUPERVISOR_RECEIPT_SHA256")
    if not receipt_path or not receipt_hash:
        raise FullFitError("admitted outer supervisor required")
    admission = json.loads(common.read_pinned_bytes(Path(receipt_path), receipt_hash, max_bytes=2_000_000))
    validate_supervisor(admission, args, argv, os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise FullFitError("single thread environment required: " + key)
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= time.time():
        raise FullFitError("admitted deadline already expired")
    owner = common.ProcessOwner.acquire(args.run_dir)
    interrupted = [False]
    def guard():
        owner.check()
        if os.path.lexists(args.stop_file) or interrupted[0]:
            raise FitStop("owner_stop")
        if not math.isfinite(args.deadline_unix) or time.time() >= args.deadline_unix:
            raise FitStop("deadline")
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    final = {"kind": PROTOCOL_ID, "run_id": args.run_id, "status": "failed", "test_payload_opened": False,
             "scientific_acceptance": False, "figure10_reproduction": False, "started_at_utc": utcnow(),
             "supervisor_receipt_sha256": receipt_hash}
    try:
        from .reporting import ExperimentReporter
        reporting_ids = {"config_sha256": args.config_sha256, "data_manifest_sha256": args.data_manifest_sha256,
                         "admission_sha256": receipt_hash, "worker_source_sha256": common.sha256_file(Path(__file__))}
        reporting_config = {"protocol": PROTOCOL_ID, "device": "cpu", "precision": "float32", "model_seed": 0,
                            "negative_seed": 1, "max_complete_epochs": 200, "loss_patience": 100, "test_access": False}
        reporter = ExperimentReporter(owner.run_dir / "wandb", project="representation-collapse", config=reporting_config,
                                      identities=reporting_ids, mode="offline", run_id=args.run_id)
        with reporter:
            report_identity = {"run_id": args.run_id, "project": "representation-collapse", "mode": "offline",
                "config": reporting_config, "identities": reporting_ids,
                "initial_reporting_receipt_sha256": common.sha256_file(reporter.receipt_path)}
            final.update(run_native(args, owner, guard, reporter, config_raw, receipt_hash, conversion, report_identity))
    except BaseException as exc:
        final.update(status="failed", error_type=type(exc).__name__, error_message=str(exc)[:2000])
        final.update(reconcile_failure(owner.run_dir / "optimizer-attempts", exc))
    finally:
        signal.signal(signal.SIGTERM, previous)
        final["finished_at_utc"] = utcnow()
        final["deadline_unix"] = args.deadline_unix
        report_path = owner.run_dir / "wandb/reporting-receipt.json"
        if report_path.exists():
            final["reporting_receipt_sha256"] = common.sha256_file(report_path)
        common.publish_terminal_manifest(owner, final)
    print(json.dumps(final, sort_keys=True))
    return 0 if final["status"] in {"fit_complete", "fit_stopped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
