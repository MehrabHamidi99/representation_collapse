#!/usr/bin/env python3
"""Analytic Figure 10 controls from hash-pinned converted manifest metadata only.

No split payloads, native models, checkpoints or training are accessed.
Every CLI run starts mandatory explicit offline W&B before manifest reading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

CONVERTER_SHA256 = "498e60f7be048411c253e4b8cb63e4b907973f969c397bfdfacde949389f057e"


def validate_patterns(patterns, nodes):
    if type(nodes) is not int or not 3 <= nodes <= 100:
        raise ValueError("bounded integer node count required")
    if not isinstance(patterns, (list, tuple)) or len(patterns) != 2:
        raise ValueError("exactly two patterns required")
    sets = []
    for pattern in patterns:
        if not isinstance(pattern, (list, tuple)) or not pattern:
            raise ValueError("nonempty pattern required")
        edges = set()
        for pair in pattern:
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(v) is not int or not 0 <= v < nodes for v in pair)
                    or pair[0] == pair[1]):
                raise ValueError("integer ordered offdiagonal pair required")
            edge = tuple(pair)
            if edge in edges:
                raise ValueError("duplicate pattern edge")
            edges.add(edge)
        if any((v, u) not in edges for u, v in edges):
            raise ValueError("bidirectional undirected patterns required")
        sets.append(edges)
    if sets[0] == sets[1]:
        raise ValueError("distinct pattern sets required")
    return tuple(sets)


def counts(truth, predicted, total):
    tp, fp, fn = len(truth & predicted), len(predicted - truth), len(truth - predicted)
    tn = total - tp - fp - fn
    return {"pairs": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / total}


def calculate(patterns, *, nodes=100):
    """Fixed predeclared target interval, period and three untuned controls."""
    a, b = validate_patterns(patterns, nodes)
    patterns = (a, b)
    result = {}
    for name in ("previous_graph_persistence", "known_generator_phase_counter", "always_negative"):
        rows = []
        for t in range(2816, 3072):
            truth = patterns[(t // 32) % 2]
            previous = patterns[((t - 1) // 32) % 2]
            predicted = previous if name == "previous_graph_persistence" else truth if name == "known_generator_phase_counter" else set()
            transition = t % 64 in (0, 32)
            if (truth != previous) is not transition:
                raise ValueError("pattern/phase transition mismatch")
            rows.append({"target_timestamp": t, "source_timestamp": t - 1,
                         "target_phase": t % 64, "source_phase": (t - 1) % 64,
                         "target_pattern": "A" if (t // 32) % 2 == 0 else "B",
                         "change_point": transition, "phase_transition_expected": transition,
                         **counts(truth, predicted, nodes * (nodes - 1))})
        transitions = [r for r in rows if r["change_point"]]
        nontransitions = [r for r in rows if not r["change_point"]]
        assert len(transitions) == 8 and len(nontransitions) == 248
        mean = lambda values: math.fsum(r["f1"] for r in values) / len(values)
        result[name] = {"rows": rows, "reduction": {
            "targets": 256, "primary_time_mean_f1": mean(rows),
            "transition_time_mean_f1": mean(transitions),
            "nontransition_time_mean_f1": mean(nontransitions),
            "per_target_f1": [r["f1"] for r in rows],
            "blocks": [{"start": 2816 + i * 32, "stop_exclusive": 2848 + i * 32,
                        "pattern": rows[i * 32]["target_pattern"],
                        "targets": 32, "time_mean_f1": mean(rows[i * 32:(i + 1) * 32])}
                       for i in range(8)]}}
    return {"kind": "figure10_deterministic_metadata_controls_v1", "nodes": nodes,
            "ordered_offdiagonal_pairs": nodes * (nodes - 1), "period": 64, "block": 32,
            "targets_interval": [2816, 3072], "pattern_edge_counts": [len(a), len(b)],
            "pattern_intersection_edges": len(a & b), "controls": result,
            "interpretation": "Analytic descriptive controls from known generator metadata; counter oracle is not learned memory.",
            "split_payloads_opened": False, "scientific_acceptance": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    reporting_dir = args.output.with_name(args.output.stem + ".wandb")
    if os.path.lexists(args.output) or os.path.lexists(reporting_dir):
        raise FileExistsError("fresh output and offline reporting directory required")
    if not args.output.parent.is_dir():
        raise ValueError("existing output parent required")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from representation_collapse import graph_data as gd
    from representation_collapse.reporting import ExperimentReporter
    # The pinned converter's strict metadata reader validates primary origin,
    # raw/config identity, encodings, semantic hashes and split declarations.
    if hashlib.sha256(Path(gd.__file__).read_bytes()).hexdigest() != CONVERTER_SHA256:
        raise ValueError("frozen metadata reader source mismatch")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_id = "figure10-controls-" + uuid.uuid4().hex
    with ExperimentReporter(reporting_dir, project="representation-collapse", mode="offline", run_id=run_id,
            config={"protocol": "figure10_deterministic_metadata_controls_v1", "metadata_only": True,
                    "targets": 256, "controls": 3},
            identities={"manifest_sha256": args.manifest_sha256, "baseline_source_sha256": source_hash,
                        "converter_sha256": CONVERTER_SHA256}) as reporter:
        manifest = gd.read_manifest(args.manifest, args.manifest_sha256, lambda: None)
        metadata = manifest.metadata
        if (metadata["kind"] != "tgrab_primary_converted_v1" or metadata["spec"] != gd.PRIMARY_SPEC.as_dict()
                or metadata["converter"]["sha256"] != CONVERTER_SHA256):
            raise ValueError("frozen primary conversion identity required")
        result = calculate(metadata["pattern_pairs"], nodes=100)
        result.update(manifest_sha256=args.manifest_sha256, raw_sha256=metadata["raw"]["sha256"],
                      converter=metadata["converter"], config_sha256=metadata["config_sha256"],
                      baseline_source_sha256=source_hash, reporting_mode="offline", reporting_run_id=run_id,
                      reporting_project="representation-collapse", remote_delivery_verified=False)
        step = 0
        for name, control in result["controls"].items():
            for row in control["rows"]:
                reporter.log({"control": name, **{"control/" + key: row[key] for key in (
                    "target_timestamp", "source_timestamp", "f1", "precision", "recall", "accuracy",
                    "tp", "fp", "fn", "tn", "change_point")}}, step=step)
                step += 1
            reporter.log({"control": name, **{"control/" + key: control["reduction"][key] for key in (
                "primary_time_mean_f1", "transition_time_mean_f1", "nontransition_time_mean_f1")}}, step=step)
            step += 1
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        reporter.log_manifest("figure10-controls-result", metadata={"targets": 256, "controls": 3,
            "split_payloads_opened": False, "reporting_mode": "offline", "output_written_after_finish": True},
            identities={"manifest_sha256": args.manifest_sha256, "result_sha256": hashlib.sha256(raw).hexdigest(),
                        "baseline_source_sha256": source_hash})
    # No result file is published if reporting initialization/logging/finish
    # fails. A retained offline receipt records that failure independently.
    with args.output.open("xb") as handle:
        handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "split_payloads_opened": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
