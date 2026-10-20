"""Admitted primitive-v2 selected inference, preserving actual long-fit identity.

No v1 manifest, selection head or admission identity is synthesized. Numerical
weight-load/state/metric primitives and passive capture are shared unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import signal
import sys
import time

from . import tgrab_figure10_canary as common

PROTOCOL = "tgrab_figure10_primitive_v2_selected_evaluation_v3"
CONTRACT_SHA256 = "aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381"


class V2EvaluationError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise V2EvaluationError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(raw):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "duplicate JSON metadata key")
            out[key] = value
        return out
    def invalid(_):
        raise V2EvaluationError("nonfinite JSON metadata")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def pinned(path, expected, limit=2_000_000):
    raw = common.read_pinned_bytes(Path(path), expected, max_bytes=limit)
    return decode(raw), raw


def accepted_root_record(root, relative, acceptance):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe accepted metadata reference")
    hashes = acceptance["collected_metadata_sha256"]
    require(relative in hashes, "independent fit acceptance omitted metadata: " + relative)
    return pinned(Path(root).joinpath(*path.parts), hashes[relative])


def verify_selection(root, *, acceptance, summary, pointer_sha256):
    """Verify the real finalized primitive-v2 receipt and intent history."""
    root = Path(root)
    pointer, _ = pinned(root / "selection/selected.json", pointer_sha256, 65536)
    context = {"run_id": acceptance["fit_run_id"], "attempt_id": acceptance["fit_attempt_id"],
               "admission_sha256": acceptance["fit_admission_sha256"],
               "executed_long_contract_sha256": CONTRACT_SHA256}
    require(pointer["kind"] == "tgrab_long_selected_pointer_v2" and pointer["finalized"] is True,
            "finalized primitive-v2 pointer required")
    require(all(pointer.get(k) == v for k, v in context.items()), "pointer execution identity mismatch")
    require(pointer_sha256 == acceptance["selected_pointer_sha256"] == summary["selected_pointer_sha256"],
            "externally authoritative pointer SHA mismatch")
    n = pointer["last_epoch"]
    require(type(n) is int and 1 <= n <= 200, "bounded complete-epoch history required")

    def reference(ref, filename):
        require(type(ref) is dict and set(ref) == {"relative_path", "sha256", "bytes"}
                and ref["relative_path"] == "receipts/" + filename, "unsafe/out-of-order selection reference")
        value, raw = pinned(root / "selection" / ref["relative_path"], ref["sha256"], 65536)
        require(type(ref["bytes"]) is int and len(raw) == ref["bytes"], "selection reference byte count mismatch")
        require(all(value.get(k) == v for k, v in context.items()), "selection receipt context mismatch")
        return value

    final = reference(pointer["receipt"], f"final-{n:03d}.json")
    require(final["kind"] == "tgrab_long_selection_finalized_v2" and final["finalized"] is True
            and final["test_authorized"] is False and final["last_epoch"] == n
            and final["stop_reason"] == summary["stop_reason"]
            and final["accounting"] == summary["journal_accounting"], "primitive-v2 finalization mismatch")
    records = []
    previous = final["previous_receipt"]
    for epoch in range(n, 0, -1):
        decision = reference(previous, f"epoch-{epoch:03d}.json")
        require(decision["kind"] == "tgrab_long_selection_decision_v2"
                and decision["epoch"] == decision["last_epoch"] == epoch
                and decision["test_authorized"] is False and decision["finalized"] is False,
                "complete epoch selection receipt mismatch")
        records.append((decision, previous))
        previous = decision["previous_receipt"]
    require(previous is None, "selection predecessor extends before epoch one")
    selected = best = previous_receipt = previous_pointer = None
    for decision, receipt_ref in reversed(records):
        epoch, score = decision["epoch"], decision["candidate_f1"]
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                "finite validation F1 required")
        choose = best is None or score > best
        require(decision["previous_selected"] == selected and decision["previous_best_f1"] == best
                and decision["previous_receipt"] == previous_receipt and decision["selected_now"] is choose,
                "strict earliest-tie validation selection mismatch")
        candidate = decision["candidate"]
        fields = {"epoch", "manifest_relative_path", "manifest_sha256", "recurrent_sha256", "link_head_sha256", "terminal_state_sha256"}
        require(set(candidate) == fields and type(candidate["epoch"]) is int and candidate["epoch"] == epoch
                and candidate["manifest_relative_path"] == f"candidates/epoch-{epoch:03d}/manifest.json",
                "candidate chronology/path mismatch")
        require(all(type(candidate[key]) is str and re.fullmatch(r"[0-9a-f]{64}", candidate[key])
                    for key in fields - {"epoch", "manifest_relative_path"}), "candidate component digests invalid")
        if choose:
            selected, best = candidate, score
        require(decision["selected"] == selected and decision["best_f1"] == best,
                "selected tuple is not the running validation maximum")
        intent, _ = accepted_root_record(root, f"selection/intents/epoch-{epoch:03d}.json", acceptance)
        require(intent == {"kind": "tgrab_long_selection_intent_v2", **context, "operation": "epoch",
                           "previous_pointer_sha256": previous_pointer, "receipt": decision},
                "epoch intent/predecessor mismatch")
        reconstructed = {"kind": "tgrab_long_selected_pointer_v2", **context, "selected": selected,
                         "best_f1": best, "last_epoch": epoch, "receipt": receipt_ref,
                         "finalized": False, "previous_pointer_sha256": previous_pointer}
        previous_pointer = hashlib.sha256(canonical(reconstructed)).hexdigest()
        previous_receipt = receipt_ref
    intent, _ = accepted_root_record(root, f"selection/intents/final-{n:03d}.json", acceptance)
    require(intent == {"kind": "tgrab_long_selection_intent_v2", **context, "operation": "final",
                       "previous_pointer_sha256": previous_pointer, "receipt": final}, "final intent/predecessor mismatch")
    expected_pointer = {"kind": "tgrab_long_selected_pointer_v2", **context, "selected": selected,
                        "best_f1": best, "last_epoch": n, "receipt": pointer["receipt"],
                        "finalized": True, "previous_pointer_sha256": previous_pointer}
    require(pointer == expected_pointer and final["selected"] == selected and final["best_f1"] == best,
            "authoritative finalized pointer differs from verified history")
    require(summary["selected"] == acceptance["selected"] == selected
            and summary["selected_validation_f1"] == acceptance["selected_validation_f1"] == best
            and summary["finalized"] is True and summary["test_payload_opened"] is False,
            "accepted fit summary/selection mismatch")
    names = {f"epoch-{epoch:03d}.json" for epoch in range(1, n + 1)} | {f"final-{n:03d}.json"}
    for directory in ("intents", "receipts"):
        require(set(os.listdir(root / "selection" / directory)) == names,
                "unexpected primitive-v2 selection namespace entries")
    return pointer


class EvaluationEvents:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(mode=0o700)
        self.identity = dict(identity)
        self.sequence = 0
        self.previous = None
        self.last_event = None

    def emit(self, event, **fields):
        raw = canonical({"kind": "figure10_primitive_v2_evaluation_event_v3", **self.identity,
                         "event": event, "sequence": self.sequence, "previous_event_sha256": self.previous, **fields})
        path = self.root / f"{self.sequence:03d}-{event}.json"
        with path.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.previous = hashlib.sha256(raw).hexdigest()
        self.sequence += 1
        self.last_event = event


def run_verified_candidate(*, verified, selected, pointer_sha256, backend, train, validation,
                           test_opener, events, report, guard=lambda: None, observe_state=None):
    """One genuine v2 load/replay/test pass; caller has charged durable fit claim."""
    from . import graph_reconstruction as gr
    from .tgrab_figure10_evaluate import load_native_weights, verify_terminal
    from .tgrab_figure10_candidate_encoder import _cpu_f32_bytes
    from .tgrab_figure10 import test_reduction

    try:
        require(type(events) is EvaluationEvents and events.sequence == 1 and events.last_event == "charged",
                "one fresh durable primitive-v2 selected-load charge required")
        require(verified["manifest"]["kind"] == "tgrab_complete_epoch_candidate_v2"
                and verified["files_sha256"]["manifest.json"] == selected["manifest_sha256"],
                "actual primitive-v2 candidate identity required")
        guard()
        # The three data-only payload keys are original v2 verified bytes.
        # No v1 manifest, admission or selection schema is manufactured.
        load_native_weights(backend, verified)
        events.emit("loaded", candidate_manifest_sha256=selected["manifest_sha256"])
        prefix = tuple(train) + tuple(validation)
        require(len(prefix) == 2816 and [s.timestamp for s in prefix] == list(range(2816)),
                "exact chronological conditioning prefix required")
        rows = []
        with backend.evaluation_mode():
            state = backend.reset()
            require(state.next_target_t == 0 and state.node_ids == tuple(range(100)), "fresh zero-state identity required")
            for name in ("hidden", "cell"):
                require(_cpu_f32_bytes(getattr(state, name), name=name, shape=(1, 100, 128)) == bytes(51200),
                        "conditioning must begin at fresh positive-zero complete state")
            for snapshot in prefix:
                guard()
                state = backend.detach(backend.consume(state, snapshot))
            verify_terminal(state, verified)
            events.emit("replay_verified", next_target_t=2816,
                        terminal_state_sha256=hashlib.sha256(verified["terminal_state"]).hexdigest())
            guard()
            events.emit("test_read", test_payload_read_attempts=1)
            iterator = iter(test_opener())
            previous = prefix[-1]
            for t in range(2816, 3072):
                guard()
                target = next(iterator)
                require(target.timestamp == t and state.next_target_t == t, "held-out target chronology mismatch")
                if observe_state is not None:
                    observe_state(gr.CompleteState(state.hidden.detach().clone(), state.cell.detach().clone(),
                                                    state.next_target_t, state.node_ids), t)
                probabilities = backend.score_pairs(state, gr.ordered_offdiagonal_pairs(100))
                rows.append(gr.snapshot_metrics(target, probabilities, previous))
                state = backend.detach(backend.consume(state, target))
                previous = target
            sentinel = object()
            require(next(iterator, sentinel) is sentinel, "extra held-out target")
        reduction = test_reduction(rows)
        result = {"kind": PROTOCOL, "rows": rows, "reduction": reduction,
                  "candidate_manifest_sha256": selected["manifest_sha256"],
                  "checkpoint_sha256": selected["manifest_sha256"], "checkpoint_format": "primitive_v2",
                  "executed_long_contract_sha256": verified["manifest"]["executed_long_contract_sha256"],
                  "numerical_template_sha256": verified["manifest"]["numerical_template_sha256"],
                  "selected_pointer_sha256": pointer_sha256,
                  "claim": "descriptive corrected single-seed reconstruction; no causal memory or Floquet claim"}
        report({"event": "evaluation_complete", **reduction})
        events.emit("metrics_complete", **result)
        return result
    except BaseException as exc:
        events.emit("failed", error_type=type(exc).__name__)
        raise


def verify_supervisor(args, argv):
    path, expected = os.environ.get("RC_SUPERVISOR_RECEIPT"), os.environ.get("RC_SUPERVISOR_RECEIPT_SHA256")
    require(path and expected, "admitted primitive-v2 evaluator supervisor required")
    receipt, _ = pinned(path, expected)
    require(receipt.get("schema") == "tgrab-figure10-v2-evaluation-supervisor-receipt-v1"
            and receipt.get("protocol_id") == PROTOCOL and receipt.get("run_id") == args.run_id
            and receipt.get("job_id") == os.environ.get("SLURM_JOB_ID")
            and receipt.get("source_sha256") == common.sha256_file(Path(__file__))
            and receipt.get("canary_args") == argv and receipt.get("sys_executable") == sys.executable
            and receipt.get("deadline_unix") == args.deadline_unix
            and receipt.get("fit_acceptance") == {"path": str(args.fit_acceptance), "sha256": args.fit_acceptance_sha256},
            "primitive-v2 supervisor/source/arguments/acceptance mismatch")
    return expected


def metadata_inputs(args):
    acceptance, _ = pinned(args.fit_acceptance, args.fit_acceptance_sha256)
    require(acceptance.get("kind") == "figure10_long_fit_execution_acceptance_v2"
            and acceptance.get("status") == "accepted" and acceptance.get("test_accessed") is False
            and acceptance.get("source_closure_git_verified") is True,
            "independently accepted actual long fit required")
    require(all(acceptance.get(name) is True for name in ("validation_selection_finalized",
        "source_closure_git_verified", "offline_history_payload_verified", "optimizer_history_verified")),
        "independent long-fit acceptance gates incomplete")
    for name, value in (("fit_run_id", args.fit_run_id), ("fit_attempt_id", args.fit_attempt_id),
                        ("fit_run_dir", str(args.fit_run_dir)), ("fit_admission_sha256", args.fit_admission_sha256),
                        ("fit_summary_sha256", args.fit_summary_sha256),
                        ("selected_pointer_sha256", args.selected_pointer_sha256)):
        require(acceptance[name] == value, "fit acceptance differs from explicit authority: " + name)
    fit_supervisor, _ = pinned(args.fit_supervisor_receipt, args.fit_admission_sha256)
    require(fit_supervisor["schema"] == "tgrab-figure10-supervisor-receipt-v2"
            and fit_supervisor["protocol_id"] == "tgrab_figure10_long_fit_v2"
            and fit_supervisor["run_id"] == args.fit_run_id and fit_supervisor["source_commit"] == acceptance["source_commit"],
            "actual fit supervisor identity mismatch")
    summary, _ = pinned(args.fit_run_dir / "fit-summary.json", args.fit_summary_sha256)
    run, _ = accepted_root_record(args.fit_run_dir, "run-identities.json", acceptance)
    pointer = verify_selection(args.fit_run_dir, acceptance=acceptance, summary=summary,
                               pointer_sha256=args.selected_pointer_sha256)
    ids = summary["identities"]
    for name, expected in (("executed_long_contract_sha256", args.long_contract_sha256),
                           ("numerical_template_sha256", args.config_sha256),
                           ("converted_manifest_sha256", args.data_manifest_sha256),
                           ("environment_receipt_sha256", args.environment_receipt_sha256),
                           ("node2a_receipt_sha256", args.node2a_receipt_sha256),
                           ("conversion_receipt_sha256", args.conversion_receipt_sha256),
                           ("actual_supervisor_receipt_sha256", args.fit_admission_sha256)):
        require(ids[name] == run[name] == expected, "actual v2 input identity mismatch: " + name)
    require(run["source_closure"] == fit_supervisor["closure_sha256"]
            and ids["source_closure_sha256"] == hashlib.sha256(canonical(run["source_closure"])).hexdigest(),
            "original source-long closure mismatch")
    records = {}
    references = {}
    for name, role in (("seeding", "seeding_before_models"), ("post-model-rng", "post_model_construction_rng"),
                       ("negative-generator", "negative_generator_initialization")):
        value, raw = accepted_root_record(args.fit_run_dir, name + ".json", acceptance)
        key = {"seeding": "seeding_record", "post-model-rng": "post_model_rng_record",
               "negative-generator": "negative_generator_record"}[name]
        records[key] = value
        references[key] = {"relative_path": name + ".json", "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest(), "role": role, "encoding": "primitive_json"}
    modes, raw = accepted_root_record(args.fit_run_dir, "module-modes.json", acceptance)
    references["module_modes"] = {"relative_path": "module-modes.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    reporting, raw = accepted_root_record(args.fit_run_dir, "reporting-identity.json", acceptance)
    require(hashlib.sha256(raw).hexdigest() == ids["reporting_identity_sha256"]
            and reporting["kind"] == "tgrab_long_reporting_identity_v2" and reporting["run_id"] == args.fit_run_id
            and reporting["mode"] == "offline" and reporting["project"] == "representation-collapse"
            and reporting["actual_supervisor_receipt_sha256"] == args.fit_admission_sha256,
            "fit reporting identity mismatch")
    require(set(modes["train_flags"]) == set(modes["eval_flags"]) == {"node_emb", "link_pred"}
            and all(flags for flags in modes["train_flags"].values())
            and all(flags for flags in modes["eval_flags"].values())
            and all(flag is True for flags in modes["train_flags"].values() for flag in flags.values())
            and all(flag is False for flags in modes["eval_flags"].values() for flag in flags.values())
            and modes["validation_grad_enabled"] is False, "fit module-mode identity mismatch")
    return acceptance, fit_supervisor, summary, pointer, records, references


def run_native(args, owner, guard, reporter, metadata):
    import numpy as np
    import torch
    from . import graph_data as gd, graph_reconstruction as gr, tgrab_selected_checkpoint as encoding
    from .tgrab_figure10_v2_codec import V2ExpectedIdentity, read_candidate_v2
    from .tgrab_figure10_candidate_encoder import _component_bytes
    from .tgrab_figure10_native_fit import _ordered_named_parameters
    from .tgrab_figure10_bootstrap import actual_rng_snapshot
    from .tgrab_figure10_evaluate_job import TrajectoryWriter
    from .tgrab_figure10_full_fit import write_record, check_conversion
    from .pinned_graph_import import load_pinned_graph_modules, PINNED_CELL_SHA256, PINNED_WRAPPER_SHA256

    acceptance, fit_sr, summary, pointer, records, refs = metadata
    guard()
    # This shared filename blocks a second pass through any evaluator version.
    write_record(args.fit_run_dir, "selected-evaluation-claim.json", {
        "kind": "figure10_primitive_v2_fit_bound_evaluation_claim_v3", "fit_run_id": args.fit_run_id,
        "fit_attempt_id": args.fit_attempt_id, "fit_admission_sha256": args.fit_admission_sha256,
        "fit_summary_sha256": args.fit_summary_sha256, "selected_pointer_sha256": args.selected_pointer_sha256,
        "candidate_manifest_sha256": pointer["selected"]["manifest_sha256"],
        "evaluation_run_id": args.run_id, "evaluation_run_dir": str(owner.run_dir),
        "fit_acceptance_sha256": args.fit_acceptance_sha256, "selected_load_attempts_charged": 1, "retry_allowed": False})
    events = EvaluationEvents(owner.run_dir / "selected-evaluation", {
        "fit_run_id": args.fit_run_id, "fit_attempt_id": args.fit_attempt_id,
        "fit_admission_sha256": args.fit_admission_sha256, "selected_pointer_sha256": args.selected_pointer_sha256,
        "evaluation_run_id": args.run_id})
    events.emit("charged", selected_load_attempts=1, test_payload_read_attempts=0)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    modules = load_pinned_graph_modules(source_root=args.source_root, installed_cell_path=args.cell_path,
        cell_sha256=PINNED_CELL_SHA256, wrapper_sha256=PINNED_WRAPPER_SHA256, guard=guard)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    require(actual_rng_snapshot(torch) == records["seeding_record"], "fresh native seeded RNG differs from accepted fit")
    models = {"node_emb": modules.wrapper_type(1, modules.parameter_type(100, 128, 2)).to(device="cpu", dtype=torch.float32),
              "link_pred": modules.head_type(128, num_layers=2, p_dropout=0.2).to(device="cpu", dtype=torch.float32)}
    require(actual_rng_snapshot(torch) == records["post_model_rng_record"], "fresh model construction RNG differs from fit")
    require(records["negative_generator_record"] == {"seed": 1, "bit_generator": np.random.default_rng(1).bit_generator.state},
            "accepted negative-generator initialization mismatch")
    _, catalog = encoding._schema_and_catalog()  # unchanged real tensor layout; not v1 admission metadata
    initial_hashes, versions = {}, {}
    for component, key in (("recurrent", "node_emb"), ("link_head", "link_pred")):
        state = models[key].state_dict()
        _, descriptor = _component_bytes(component, state, catalog)
        initial_hashes[component] = encoding.component_semantic_sha256(descriptor["tensors"])
        versions[component] = {name: item["version"] for name, item in state._metadata.items()}
    runtime_environment = {"python": platform.python_version(), "implementation": platform.python_implementation(),
        "system": platform.system(), "machine": platform.machine(), "versions": common.environment_versions(),
        "requirements_lock_sha256": common.ENVIRONMENT_LOCK_SHA256, "wheel_manifest_sha256": common.WHEEL_MANIFEST_SHA256}
    expected = V2ExpectedIdentity(run_id=args.fit_run_id, source_commit=acceptance["source_commit"],
        slurm_job_id=str(fit_sr["job_id"]), identities=summary["identities"], source_closure=fit_sr["closure_sha256"],
        runtime_environment=runtime_environment,
        initialization={**{k: refs[k] for k in ("seeding_record", "post_model_rng_record", "negative_generator_record")},
                        "initial_component_semantic_sha256": initial_hashes},
        module_modes=refs["module_modes"], module_state_dict_versions=versions,
        optimizer_parameter_order=[name for name, _ in _ordered_named_parameters(models)], actual_job=fit_sr["job"])
    selected = pointer["selected"]
    bundle = args.fit_run_dir / PurePosixPath(selected["manifest_relative_path"]).parent
    verified = read_candidate_v2(bundle, selected["manifest_sha256"], expected=expected)
    for filename, key in (("recurrent.f32le", "recurrent_sha256"), ("link-head.f32le", "link_head_sha256"),
                          ("validation-terminal.f32le", "terminal_state_sha256")):
        require(verified["files_sha256"][filename] == selected[key], "selected primitive payload identity mismatch")
    backend = gr.NativeGCLSTMBackend(torch, models, None, device="cpu")
    manifest = gd.read_manifest(args.data_manifest, args.data_manifest_sha256, guard)
    conversion, _ = pinned(args.conversion_receipt, args.conversion_receipt_sha256)
    check_conversion(conversion, manifest.metadata, manifest.sha256)
    store = gd.open_train_validation(manifest, guard, snapshot_factory=gr.Snapshot)
    reference = gr.CheckpointReference(selected["epoch"], selected["manifest_sha256"], selected["recurrent_sha256"], selected["link_head_sha256"])
    selection = gr.ValidationOnlySelection(reference, pointer["best_f1"], pointer["last_epoch"], True)
    step = [0]
    def report(record):
        metrics = {"evaluation/event": record["event"]}
        metrics.update({"evaluation/" + key: record[key] for key in
            ("targets", "primary_time_mean_f1", "transition_time_mean_f1", "nontransition_time_mean_f1") if key in record})
        reporter.log(metrics, step=step[0]); step[0] += 1
    report({"event": "evaluation_reporting_initialized"})
    write_record(owner.run_dir, "evaluation-inputs.json", {"protocol": PROTOCOL, "candidate_kind": verified["manifest"]["kind"],
        "fit_acceptance_sha256": args.fit_acceptance_sha256, "fit_summary_sha256": args.fit_summary_sha256,
        "selected_pointer_sha256": args.selected_pointer_sha256, "selected": selected,
        "selected_manifest": verified["manifest"], "selected_provenance": verified["provenance"],
        "optimizer_created": False, "rng_restored": False, "terminal_state_restored": False})
    trajectory = TrajectoryWriter(owner.run_dir)
    try:
        result = run_verified_candidate(verified=verified, selected=selected, pointer_sha256=args.selected_pointer_sha256,
            backend=backend, train=store.train(), validation=store.validation(), test_opener=store.sealed_test_opener(selection),
            events=events, report=report, guard=guard, observe_state=trajectory)
        result["state_trajectory"] = trajectory.finish()
    finally:
        trajectory.close()
    for row in result["rows"]:
        reporter.log({"test/" + key: row[key] for key in
            ("target_timestamp", "source_timestamp", "f1", "precision", "recall", "tp", "fp", "fn", "tn")}, step=step[0])
        step[0] += 1
    result["fit_acceptance_sha256"] = args.fit_acceptance_sha256
    result_ref = write_record(owner.run_dir, "evaluation-result.json", result)
    reporter.log_manifest("figure10-primitive-v2-evaluation-result", metadata={"targets": 256, "state_observations": 256,
        "trajectory_bytes": 26214400, "passive_capture": True, "payload_upload": False, "model_upload": False},
        identities={"evaluation_result_sha256": result_ref["sha256"], "candidate_manifest_sha256": selected["manifest_sha256"],
            "selected_pointer_sha256": args.selected_pointer_sha256, "fit_acceptance_sha256": args.fit_acceptance_sha256,
            "long_contract_sha256": args.long_contract_sha256, "numerical_template_sha256": args.config_sha256,
            "data_manifest_sha256": args.data_manifest_sha256, "trajectory_sha256": result["state_trajectory"]["sha256"],
            "evaluator_source_sha256": common.sha256_file(Path(__file__))})
    events.emit("complete", evaluation_result_record=result_ref, state_trajectory=result["state_trajectory"])
    return {**result, "evaluation_result_record": result_ref}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "long-contract", "data-manifest", "environment-lock", "wheel-manifest", "environment-receipt",
                 "node2a-receipt", "conversion-receipt", "source-root", "cell-path", "fit-run-dir", "fit-acceptance",
                 "fit-supervisor-receipt", "run-dir", "stop-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config-sha256", "long-contract-sha256", "data-manifest-sha256", "environment-receipt-sha256",
                 "node2a-receipt-sha256", "conversion-receipt-sha256", "fit-summary-sha256", "selected-pointer-sha256",
                 "fit-run-id", "fit-attempt-id", "fit-admission-sha256", "fit-acceptance-sha256", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(argv)
    require(args.config_sha256 == common.CONFIG_SHA256 and args.long_contract_sha256 == CONTRACT_SHA256,
            "actual frozen long-v2 contract and numerical template required")
    pinned(args.config, args.config_sha256)
    contract, _ = pinned(args.long_contract, args.long_contract_sha256)
    for name, expected in contract["input_pins"]["source_files_sha256"].items():
        require(Path(name).name == name, "unsafe frozen source filename")
        common.verify_pin(Path(__file__).parent / name, expected, max_bytes=100_000)
    common.verify_pin(args.environment_lock, common.ENVIRONMENT_LOCK_SHA256, max_bytes=1_000_000)
    common.verify_pin(args.wheel_manifest, common.WHEEL_MANIFEST_SHA256, max_bytes=1_000_000)
    environment, _ = pinned(args.environment_receipt, args.environment_receipt_sha256)
    common.validate_environment_receipt(environment)
    node2a, _ = pinned(args.node2a_receipt, args.node2a_receipt_sha256); common.validate_node2a_receipt(node2a)
    for relative, expected in common.WRAPPER_SHA256.items():
        common.verify_pin(args.source_root / relative, expected, max_bytes=64000)
    common.verify_pin(args.cell_path, common.CELL_SHA256, max_bytes=64000)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        require(os.environ.get(name) == "1", "single-thread execution required: " + name)
    require(math.isfinite(args.deadline_unix) and time.time() < args.deadline_unix, "live admitted deadline required")
    supervisor_sha = verify_supervisor(args, argv)
    owner = common.ProcessOwner.acquire(args.run_dir)
    guard = common.CanaryGuard(args.stop_file, args.deadline_unix, owner)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: setattr(guard, "interrupted", True))
    terminal = {"protocol": PROTOCOL, "status": "failed", "run_id": args.run_id,
                "supervisor_receipt_sha256": supervisor_sha, "fit_acceptance_sha256": args.fit_acceptance_sha256,
                "selected_pointer_sha256": args.selected_pointer_sha256, "scientific_acceptance": False,
                "exact_paper_reproduction": False}
    try:
        from .reporting import ExperimentReporter
        with ExperimentReporter(owner.run_dir / "wandb", project="representation-collapse", mode="offline", run_id=args.run_id,
            config={"protocol": PROTOCOL, "candidate_schema": "primitive_v2", "device": "cpu", "precision": "float32", "optimizer_created": False},
            identities={"fit_acceptance_sha256": args.fit_acceptance_sha256, "selected_pointer_sha256": args.selected_pointer_sha256,
                        "long_contract_sha256": args.long_contract_sha256, "numerical_template_sha256": args.config_sha256,
                        "data_manifest_sha256": args.data_manifest_sha256}) as reporter:
            metadata = metadata_inputs(args)
            result = run_native(args, owner, guard, reporter, metadata)
            terminal.update(status="evaluation_complete", reduction=result["reduction"],
                candidate_manifest_sha256=result["candidate_manifest_sha256"], state_trajectory=result["state_trajectory"],
                evaluation_result_record=result["evaluation_result_record"])
    except BaseException as exc:
        terminal.update(error_type=type(exc).__name__, error=str(exc)[:1500])
    finally:
        signal.signal(signal.SIGTERM, previous)
        receipt = owner.run_dir / "wandb/reporting-receipt.json"
        if receipt.exists():
            terminal["reporting_receipt_sha256"] = common.sha256_file(receipt)
        common.publish_terminal_manifest(owner, terminal)
    print(json.dumps(terminal, sort_keys=True))
    return 0 if terminal["status"] == "evaluation_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
