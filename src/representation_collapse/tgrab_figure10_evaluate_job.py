"""Single-use offline-reported CPU selected-checkpoint evaluation diagnostic v2.

The submitter independently pins the fit summary, final head and admission.
This CLI does not authenticate a submitter or create a scheduler allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import signal
import sys
import time

from . import tgrab_figure10_canary as common

PROTOCOL = "tgrab_figure10_selected_evaluation_diagnostic_v2"


class EvaluationJobError(RuntimeError):
    pass


def read_json(path, sha256, limit=2_000_000):
    return json.loads(common.read_pinned_bytes(Path(path), sha256, max_bytes=limit))


def verify_external_reference(root, reference, role):
    """Strict fit-root-relative primitive record; no traversal or symlink input."""
    if (type(reference) is not dict
            or set(reference) != {"relative_path", "bytes", "sha256", "role", "encoding"}
            or reference["role"] != role or reference["encoding"] != "primitive_json"):
        raise EvaluationJobError("external reference role/fields mismatch")
    relative = PurePosixPath(reference["relative_path"])
    if (relative.is_absolute() or len(relative.parts) != 1
            or relative.name in ("", ".", "..")):
        raise EvaluationJobError("external reference must name a fit-root record")
    raw = common.read_pinned_bytes(Path(root) / relative.name, reference["sha256"], max_bytes=2_000_000)
    if type(reference["bytes"]) is not int or len(raw) != reference["bytes"]:
        raise EvaluationJobError("external reference byte count mismatch")
    return json.loads(raw)


def verify_selected_external_records(fit_root, verified, *, expected_identities):
    """Resolve all v1 external initialization/reporting/mode dependencies."""
    provenance = verified["provenance"]
    if verified["manifest"]["identities"] != expected_identities:
        raise EvaluationJobError("selected candidate identities differ from pinned fit summary")
    initialization = provenance["initialization"]
    records = {}
    for key, role in (("seeding_record", "seeding_before_models"),
                      ("post_model_rng_record", "post_model_construction_rng"),
                      ("negative_generator_record", "negative_generator_initialization")):
        records[key] = verify_external_reference(fit_root, initialization[key], role)
    records["reporting"] = read_json(Path(fit_root) / "reporting-identity.json", provenance["reporting_identity_sha256"])
    records["modes"] = read_json(Path(fit_root) / "module-modes.json", provenance["runtime_policy"]["module_mode_receipt_sha256"])
    if (records["negative_generator_record"].get("seed") != 1
            or records["reporting"].get("run_id") != provenance["run_id"]
            or records["reporting"].get("mode") != "offline"
            or records["reporting"].get("project") != "representation-collapse"):
        raise EvaluationJobError("external initialization/reporting content mismatch")
    modes = records["modes"]
    if (set(modes.get("eval_flags", {})) != {"node_emb", "link_pred"}
            or set(modes.get("train_flags", {})) != {"node_emb", "link_pred"}
            or any(not values for values in modes["eval_flags"].values())
            or any(not values for values in modes["train_flags"].values())
            or any(flag is not True for values in modes["train_flags"].values() for flag in values.values())
            or any(flag is not False for values in modes["eval_flags"].values() for flag in values.values())
            or modes.get("validation_grad_enabled") is not False):
        raise EvaluationJobError("fit evaluation mode receipt mismatch")
    return records


def charge_fit_evaluation(fit_root, identity):
    """Spend this fit's only selected evaluation across all output run names."""
    from .tgrab_figure10_full_fit import write_record
    return write_record(Path(fit_root), "selected-evaluation-claim.json", {
        "kind": "figure10_fit_bound_evaluation_claim_v2", **identity,
        "selected_load_attempts_charged": 1, "retry_allowed": False})


def run_native(args, owner, guard, reporter, config, conversion, summary):
    # Mandatory offline reporter is already entered; no top-level NumPy/Torch.
    import numpy as np
    import torch
    from . import graph_data as gd, graph_reconstruction as gr
    from . import tgrab_selected_checkpoint as codec
    from .tgrab_selection_transaction import SelectionContext, SelectionLedger
    from .tgrab_figure10_evaluate import run_selected_evaluation
    from .tgrab_figure10_candidate_encoder import _component_bytes
    from .tgrab_figure10_bootstrap import actual_rng_snapshot
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256
    from .tgrab_figure10_full_fit import check_conversion, write_record

    guard()
    gr.validate_frozen_config(config)
    context = SelectionContext(args.fit_run_id, args.fit_attempt_id, args.fit_admission_sha256,
                               codec.SCHEMA_SHA256, args.final_head_sha256)
    with SelectionLedger(args.fit_run_dir / "checkpoint-ledger", context) as ledger:
        head = ledger.verify_committed_state()
        if not head or head["finalized"] is not True or head != summary["ledger_head"]:
            raise EvaluationJobError("pinned fit summary does not match finalized ledger")
        pointer, _ = ledger._read_reference(head["selected_pointer"], "immutable_selected_pointer")
        relative = pointer["manifest_relative_path"]
        if not relative.endswith("/manifest.json"):
            raise EvaluationJobError("selected manifest path mismatch")
        verified = codec.read_candidate(ledger.root / relative[:-len("/manifest.json")])
        records = verify_selected_external_records(args.fit_run_dir, verified,
                                                   expected_identities=summary["identities"])
        p = verified["provenance"]
        ids = verified["manifest"]["identities"]
        for key, expected in (("executed_config_sha256", args.config_sha256),
                              ("converted_manifest_sha256", args.data_manifest_sha256),
                              ("conversion_receipt_sha256", args.conversion_receipt_sha256),
                              ("environment_receipt_sha256", args.environment_receipt_sha256)):
            if ids[key] != expected:
                raise EvaluationJobError("selected runtime input identity mismatch: " + key)
        for key, filename in (("trainer", "tgrab_figure10_native_fit.py"),
                              ("codec", "tgrab_selected_checkpoint.py")):
            common.verify_pin(Path(__file__).parent / filename, p["source_sha256"][key], max_bytes=100_000)
        for key, expected in (("graph_reconstruction", common.GRAPH_RECONSTRUCTION_SHA256),
                              ("graph_data", common.GRAPH_DATA_SHA256), ("reporter", common.REPORTER_SHA256),
                              ("graph_import_helper", common.IMPORT_HELPER_SHA256),
                              ("recurrent_wrapper", common.WRAPPER_SHA256["model/gclstm.py"]),
                              ("link_head", common.WRAPPER_SHA256["model/link_predictor.py"]),
                              ("cell", common.CELL_SHA256)):
            if p["source_sha256"][key] != expected:
                raise EvaluationJobError("selected source differs from admitted runtime: " + key)
        if records["negative_generator_record"]["bit_generator"] != np.random.default_rng(1).bit_generator.state:
            raise EvaluationJobError("negative generator initialization record mismatch")
        guard()
        charge_fit_evaluation(args.fit_run_dir, {
            "fit_run_id": args.fit_run_id, "fit_attempt_id": args.fit_attempt_id,
            "fit_admission_sha256": args.fit_admission_sha256,
            "final_head_sha256": args.final_head_sha256,
            "checkpoint_sha256": pointer["selected"]["checkpoint_sha256"],
            "evaluation_run_id": args.run_id, "evaluation_run_dir": str(owner.run_dir),
            "fit_summary_sha256": args.fit_summary_sha256})
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        modules = load_pinned_graph_modules(source_root=args.source_root, installed_cell_path=args.cell_path,
            cell_sha256=PINNED_CELL_SHA256, wrapper_sha256=PINNED_WRAPPER_SHA256, guard=guard)
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if actual_rng_snapshot(torch) != records["seeding_record"]:
            raise EvaluationJobError("fresh initial RNG differs from pinned seeding record")
        models = {"node_emb": modules.wrapper_type(1, modules.parameter_type(100, 128, 2)).to(device="cpu", dtype=torch.float32),
                  "link_pred": modules.head_type(128, num_layers=2, p_dropout=0.2).to(device="cpu", dtype=torch.float32)}
        if actual_rng_snapshot(torch) != records["post_model_rng_record"]:
            raise EvaluationJobError("fresh construction RNG differs from fit record")
        _, catalog = codec._schema_and_catalog()
        for component, key, initial in (("recurrent", "node_emb", "initial_recurrent_semantic_sha256"),
                                        ("link_head", "link_pred", "initial_head_semantic_sha256")):
            state_dict = models[key].state_dict()
            _, descriptor = _component_bytes(component, state_dict, catalog)
            if codec.component_semantic_sha256(descriptor["tensors"]) != p["initialization"][initial]:
                raise EvaluationJobError("native fresh initialization mismatch: " + component)
            versions = {name: meta["version"] for name, meta in state_dict._metadata.items()}
            if versions != p["module_state_dict_versions"][component]:
                raise EvaluationJobError("native module state metadata version mismatch")
        # Backend receives no optimizer; this worker has no update path.
        backend = gr.NativeGCLSTMBackend(torch, models, None, device="cpu")
        manifest = gd.read_manifest(args.data_manifest, args.data_manifest_sha256, guard)
        check_conversion(conversion, manifest.metadata, manifest.sha256)
        store = gd.open_train_validation(manifest, guard, snapshot_factory=gr.Snapshot)
        reference = gr.CheckpointReference(**pointer["selected"])
        selection = gr.ValidationOnlySelection(reference, pointer["best_f1"], pointer["last_epoch"], True)
        step = [0]
        def report(record):
            metrics = {"evaluation/event": record["event"]}
            for name in ("primary_time_mean_f1", "transition_time_mean_f1", "nontransition_time_mean_f1", "targets"):
                if name in record:
                    metrics["evaluation/" + name] = record[name]
            reporter.log(metrics, step=step[0]); step[0] += 1
        write_record(owner.run_dir, "evaluation-inputs.json", {
            "protocol": PROTOCOL, "fit_summary_sha256": args.fit_summary_sha256,
            "final_head_sha256": args.final_head_sha256, "selected": pointer["selected"],
            "selected_manifest": verified["manifest"], "selected_provenance": p,
            "source_module_sha256": common.sha256_file(Path(__file__)),
            "evaluation_module_sha256": common.sha256_file(Path(__file__).with_name("tgrab_figure10_evaluate.py")),
            "optimizer_created": False, "rng_restored": False, "terminal_state_restored": False})
        result = run_selected_evaluation(ledger=ledger, expected_context=context,
            expected_final_head_sha256=args.final_head_sha256, backend=backend,
            train=store.train(), validation=store.validation(), test_opener=store.sealed_test_opener(selection),
            output_dir=owner.run_dir / "selected-evaluation", report=report, guard=guard)
        for row in result["rows"]:
            reporter.log({"test/" + key: row[key] for key in ("target_timestamp", "source_timestamp", "f1", "precision", "recall", "tp", "fp", "fn", "tn")}, step=step[0])
            step[0] += 1
        write_record(owner.run_dir, "evaluation-result.json", result)
        return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data-manifest", "environment-lock", "wheel-manifest", "environment-receipt",
                 "node2a-receipt", "conversion-receipt", "source-root", "cell-path", "fit-run-dir", "run-dir", "stop-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "data-manifest-sha256", "environment-receipt-sha256", "node2a-receipt-sha256",
                 "conversion-receipt-sha256", "fit-summary-sha256", "final-head-sha256", "fit-run-id",
                 "fit-attempt-id", "fit-admission-sha256", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    return parser


def verify_supervisor(args, argv):
    path = os.environ.get("RC_SUPERVISOR_RECEIPT")
    sha = os.environ.get("RC_SUPERVISOR_RECEIPT_SHA256")
    if not path or not sha:
        raise EvaluationJobError("admitted evaluation supervisor required")
    receipt = read_json(Path(path), sha)
    if (receipt.get("schema") != "tgrab-figure10-evaluation-supervisor-receipt-v2"
            or receipt.get("protocol_id") != PROTOCOL
            or receipt.get("run_id") != args.run_id
            or receipt.get("job_id") != os.environ.get("SLURM_JOB_ID")
            or receipt.get("source_sha256") != common.sha256_file(Path(__file__))
            or receipt.get("canary_args") != argv
            or receipt.get("sys_executable") != sys.executable
            or receipt.get("deadline_unix") != args.deadline_unix):
        raise EvaluationJobError("evaluation supervisor run/source/argv/deadline mismatch")
    return sha


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(argv)
    if args.config_sha256 != common.CONFIG_SHA256:
        raise EvaluationJobError("frozen reconstruction config required")
    config = read_json(args.config, args.config_sha256)
    common.verify_pin(args.environment_lock, common.ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
    common.verify_pin(args.wheel_manifest, common.WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
    common.validate_environment_receipt(read_json(args.environment_receipt, args.environment_receipt_sha256))
    common.validate_node2a_receipt(read_json(args.node2a_receipt, args.node2a_receipt_sha256))
    conversion = read_json(args.conversion_receipt, args.conversion_receipt_sha256)
    summary = read_json(args.fit_run_dir / "fit-summary.json", args.fit_summary_sha256)
    if summary.get("finalized") is not True or summary.get("test_payload_opened") is not False:
        raise EvaluationJobError("finalized, test-unopened fit summary required")
    for relative, expected in common.WRAPPER_SHA256.items():
        common.verify_pin(args.source_root / relative, expected, max_bytes=64_000)
    common.verify_pin(args.cell_path, common.CELL_SHA256, max_bytes=64_000)
    for filename, expected in (("graph_data.py", common.GRAPH_DATA_SHA256), ("graph_reconstruction.py", common.GRAPH_RECONSTRUCTION_SHA256),
                               ("pinned_graph_import.py", common.IMPORT_HELPER_SHA256), ("reporting.py", common.REPORTER_SHA256)):
        common.verify_pin(Path(__file__).parent / filename, expected, max_bytes=100_000)
    common.environment_versions()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise EvaluationJobError("single-thread environment required: " + key)
    if not math.isfinite(args.deadline_unix) or time.time() >= args.deadline_unix:
        raise EvaluationJobError("deadline invalid/expired")
    supervisor_sha256 = verify_supervisor(args, argv)
    owner = common.ProcessOwner.acquire(args.run_dir)
    guard = common.CanaryGuard(args.stop_file, args.deadline_unix, owner)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: setattr(guard, "interrupted", True))
    final = {"protocol": PROTOCOL, "status": "failed", "run_id": args.run_id,
             "supervisor_receipt_sha256": supervisor_sha256,
             "scientific_acceptance": False, "exact_paper_reproduction": False,
             "fit_summary_sha256": args.fit_summary_sha256, "final_head_sha256": args.final_head_sha256}
    try:
        from .reporting import ExperimentReporter
        with ExperimentReporter(owner.run_dir / "wandb", project="representation-collapse", mode="offline", run_id=args.run_id,
                config={"protocol": PROTOCOL, "device": "cpu", "precision": "float32", "optimizer_created": False},
                identities={"config_sha256": args.config_sha256, "data_manifest_sha256": args.data_manifest_sha256,
                            "fit_summary_sha256": args.fit_summary_sha256, "final_head_sha256": args.final_head_sha256}) as reporter:
            result = run_native(args, owner, guard, reporter, config, conversion, summary)
            final.update(status="evaluation_complete", reduction=result["reduction"], checkpoint_sha256=result["checkpoint_sha256"])
    except BaseException as exc:
        final.update(error_type=type(exc).__name__, error=str(exc)[:2000])
    finally:
        signal.signal(signal.SIGTERM, previous)
        receipt = owner.run_dir / "wandb/reporting-receipt.json"
        if receipt.exists():
            final["reporting_receipt_sha256"] = common.sha256_file(receipt)
        common.publish_terminal_manifest(owner, final)
    print(json.dumps(final, sort_keys=True))
    return 0 if final["status"] == "evaluation_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
