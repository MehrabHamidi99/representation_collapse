#!/usr/bin/env python3
"""Plot independently accepted Figure 10 behavioral reconstruction metadata.

Reads compact evaluator rows and a separately hash-pinned acceptance record.
Never opens a dataset, model checkpoint or original-paper curve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import stat

MAX_INPUT_BYTES = 2_000_000
EXPECTED_TIMESTAMPS = tuple(range(2816, 3072))
ACCEPTANCE_KIND = "figure10_evaluation_plot_acceptance_v1"
ACCEPTANCE_KIND_V2 = "figure10_evaluation_plot_acceptance_v2"
LONG_CONTRACT_SHA256 = "aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381"


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checkpoint_identity(evaluation):
    """Retain genuine v1 head or primitive-v2 pointer semantics, never both."""
    if not _hash(evaluation.get("checkpoint_sha256")):
        raise ValueError("verified evaluator checkpoint required")
    new_keys = {"selected_pointer_sha256", "checkpoint_format", "executed_long_contract_sha256"}
    if new_keys.intersection(evaluation):
        if ("final_head_sha256" in evaluation
                or evaluation.get("checkpoint_format") != "primitive_v2"
                or evaluation.get("executed_long_contract_sha256") != LONG_CONTRACT_SHA256
                or not _hash(evaluation.get("selected_pointer_sha256"))):
            raise ValueError("unambiguous frozen primitive-v2 checkpoint identity required")
        return {"checkpoint_format": "primitive_v2", "selected_pointer_sha256": evaluation["selected_pointer_sha256"],
                "executed_long_contract_sha256": LONG_CONTRACT_SHA256}
    if not _hash(evaluation.get("final_head_sha256")):
        raise ValueError("verified evaluator final head required")
    return {"final_head_sha256": evaluation["final_head_sha256"]}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_pinned_json(path, expected_hash):
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("explicit SHA-256 input pin required")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_INPUT_BYTES:
        raise ValueError("bounded single-link regular metadata file required")
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("input hash/size mismatch")
    def reject_constant(value):
        raise ValueError("nonfinite JSON token: " + value)
    result = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    if type(result) is not dict:
        raise ValueError("metadata object required")
    return result


def validate_acceptance(record, evaluation_hash):
    if (record.get("kind") not in {ACCEPTANCE_KIND, ACCEPTANCE_KIND_V2} or record.get("status") != "accepted"
            or record.get("evaluation_sha256") != evaluation_hash
            or type(record.get("model_seed")) is not int or record["model_seed"] != 0
            or type(record.get("selected_epoch")) is not int or not 1 <= record["selected_epoch"] <= 200
            or type(record.get("run_id")) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", record["run_id"]) is None
            or record.get("training_stop_reason") not in {"deadline", "owner_stop", "loss_patience", "complete_epoch_cap"}
            or record.get("scientific_acceptance") is not False):
        raise ValueError("accepted seed-0 corrected-reconstruction metadata required")
    for key in ("validation_selection_finalized", "terminal_replay_verified", "test_rows_verified"):
        if record.get(key) is not True:
            raise ValueError("independent acceptance missing gate: " + key)
    if record["kind"] == ACCEPTANCE_KIND_V2:
        if (record.get("checkpoint_format") != "primitive_v2"
                or record.get("executed_long_contract_sha256") != LONG_CONTRACT_SHA256
                or not _hash(record.get("selected_pointer_sha256")) or "final_head_sha256" in record):
            raise ValueError("primitive-v2 acceptance contract/pointer required")
    elif {"selected_pointer_sha256", "checkpoint_format", "executed_long_contract_sha256"}.intersection(record):
        raise ValueError("legacy acceptance cannot carry primitive-v2 identity")


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("finite [0,1] metric required: " + name)
    return float(value)


def _same(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("metric/reduction mismatch: " + name)


def validate_evaluation(evaluation):
    if "event" in evaluation and evaluation["event"] != "complete":
        raise ValueError("only a complete evaluator event may be plotted")
    rows = evaluation.get("rows")
    if type(rows) is not list or len(rows) != 256:
        raise ValueError("exactly 256 held-out target rows required")
    if any(type(row) is not dict for row in rows):
        raise ValueError("metric row object required")
    times = [row.get("target_timestamp") for row in rows]
    if any(type(t) is not int for t in times) or tuple(times) != EXPECTED_TIMESTAMPS:
        raise ValueError("unique chronological target timestamps 2816..3071 required")
    validated = []
    for index, row in enumerate(rows):
        t = EXPECTED_TIMESTAMPS[index]
        expected_transition = t % 64 in (0, 32)
        for key, expected in (("source_timestamp", t - 1), ("target_phase", t % 64),
                              ("source_phase", (t - 1) % 64), ("pairs", 9900)):
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValueError("primary chronology/count mismatch: " + key)
        for key in ("change_point", "phase_transition_expected"):
            if type(row.get(key)) is not bool or row[key] is not expected_transition:
                raise ValueError("transition flag mismatch: " + key)
        counts = {}
        for key in ("tp", "fp", "fn", "tn"):
            value = row.get(key)
            if type(value) is not int or value < 0:
                raise ValueError("nonnegative integer confusion counts required")
            counts[key] = value
        if sum(counts.values()) != 9900:
            raise ValueError("confusion counts must sum to all 9900 ordered pairs")
        tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
        expected = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                    "accuracy": (tp + tn) / 9900}
        for key, value in expected.items():
            _same(_number(row.get(key), key), value, key)
        validated.append({**row, "test_target_index": index, "motif_presentation": index // 32 + 1,
                          "test_cycle": index // 64 + 1, "motif": "A" if t % 64 < 32 else "B"})
    f1 = [float(row["f1"]) for row in validated]
    transition = [row["f1"] for row in validated if row["change_point"]]
    nontransition = [row["f1"] for row in validated if not row["change_point"]]
    reduction = evaluation.get("reduction")
    if type(reduction) is not dict or type(reduction.get("targets")) is not int or reduction["targets"] != 256:
        raise ValueError("complete evaluator reduction required")
    recomputed = {"primary_time_mean_f1": math.fsum(f1) / 256,
                  "transition_time_mean_f1": math.fsum(transition) / 8,
                  "nontransition_time_mean_f1": math.fsum(nontransition) / 248}
    for key, value in recomputed.items():
        _same(_number(reduction.get(key), key), value, key)
    reduced_f1 = reduction.get("per_target_f1")
    if (type(reduced_f1) is not list or len(reduced_f1) != 256
            or [_number(value, "per_target_f1") for value in reduced_f1] != f1):
        raise ValueError("reduction per-target F1 differs from rows")
    checkpoint_identity(evaluation)
    return validated, recomputed



def within_motif_summary(rows):
    """Fixed descriptive windows; cycles/phases are correlated observations."""
    groups = {(motif, phase): [] for motif in ("A", "B") for phase in range(32)}
    for row in rows:
        timestamp = row["target_timestamp"]
        motif = "A" if timestamp % 64 < 32 else "B"
        groups[motif, timestamp % 32].append(row)
    aligned = []
    for (motif, phase), group in groups.items():
        if len(group) != 4 or len({row["target_timestamp"] for row in group}) != 4:
            raise ValueError("four distinct target observations per motif/phase required")
        aligned.append({"motif": motif, "within_motif_phase": phase, "observations": 4,
                        "mean_f1": math.fsum(_number(row["f1"], "f1") for row in group) / 4,
                        "contributing_target_timestamps": [row["target_timestamp"] for row in group]})
    contrast = {"definition": "late_minus_early_f1", "early_phases_inclusive": [1, 8],
                "late_phases_inclusive": [24, 31], "transition_phase_excluded": 0,
                "independent_replicates": False, "uncertainty_intervals": False, "motifs": {}}
    windows = {}
    for motif in ("A", "B"):
        early = [row["f1"] for row in rows if ("A" if row["target_timestamp"] % 64 < 32 else "B") == motif
                 and 1 <= row["target_timestamp"] % 32 <= 8]
        late = [row["f1"] for row in rows if ("A" if row["target_timestamp"] % 64 < 32 else "B") == motif
                and 24 <= row["target_timestamp"] % 32 <= 31]
        if len(early) != 32 or len(late) != 32:
            raise ValueError("fixed per-motif windows require 32 early and 32 late observations")
        early_mean, late_mean = math.fsum(early) / 32, math.fsum(late) / 32
        contrast["motifs"][motif] = {"early_targets": 32, "late_targets": 32,
            "early_mean_f1": early_mean, "late_mean_f1": late_mean,
            "late_minus_early_f1": late_mean - early_mean}
        windows[motif] = (early, late)
    early = windows["A"][0] + windows["B"][0]
    late = windows["A"][1] + windows["B"][1]
    contrast["overall"] = {"early_targets": 64, "late_targets": 64,
        "early_mean_f1": math.fsum(early) / 64, "late_mean_f1": math.fsum(late) / 64,
        "late_minus_early_f1": math.fsum(late) / 64 - math.fsum(early) / 64}
    return aligned, contrast


def render(*, evaluation_path, evaluation_hash, acceptance_path, acceptance_hash, output):
    evaluation = read_pinned_json(evaluation_path, evaluation_hash)
    acceptance = read_pinned_json(acceptance_path, acceptance_hash)
    validate_acceptance(acceptance, evaluation_hash)
    rows, reduction = validate_evaluation(evaluation)
    identity = checkpoint_identity(evaluation)
    is_v2 = acceptance["kind"] == ACCEPTANCE_KIND_V2
    if is_v2:
        if any(acceptance[key] != identity.get(key) for key in
               ("checkpoint_format", "executed_long_contract_sha256", "selected_pointer_sha256")):
            raise ValueError("acceptance/evaluation primitive-v2 identity mismatch")
    elif "final_head_sha256" not in identity:
        raise ValueError("legacy acceptance requires original final-head evaluation")
    elif "final_head_sha256" in acceptance and acceptance["final_head_sha256"] != identity["final_head_sha256"]:
        raise ValueError("acceptance/evaluation final-head identity mismatch")
    aligned, contrast = within_motif_summary(rows)
    if "run_id" in evaluation and evaluation["run_id"] != acceptance["run_id"]:
        raise ValueError("acceptance/evaluation run mismatch")
    protocol_path = Path(__file__).resolve().parents[1] / "doc/auto/figure10-within-motif-descriptive-2026-10-19.md"
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    output = Path(output)
    # Validation precedes any output directory publication or plotting import.
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, (timeline, motifs, phases) = plt.subplots(3, 1, figsize=(7.4, 7.8),
                                         gridspec_kw={"height_ratios": [1.5, .85, 1.1]}, layout="constrained")
    times = [row["target_timestamp"] for row in rows]
    f1 = [row["f1"] for row in rows]
    for block in range(8):
        start = 2816 + 32 * block
        timeline.axvspan(start, start + 32, color="#e7eff8" if block % 2 == 0 else "#f4eee3", alpha=0.8, lw=0)
    timeline.plot(times, f1, color="#215b8f", lw=1.25, label="Per-target F1", zorder=3)
    boundary_rows = [row for row in rows if row["change_point"]]
    timeline.scatter([row["target_timestamp"] for row in boundary_rows], [row["f1"] for row in boundary_rows],
                     s=20, color="#bd4f26", zorder=4, label="Actual motif transition")
    timeline.axhline(reduction["primary_time_mean_f1"], lw=0.9, ls="--", color="#555555", label="Temporal mean")
    timeline.set(xlim=(2815, 3072), ylim=(-0.035, 1.055), ylabel="F1", xlabel="Held-out target timestamp",
                 title="a  Chronological next-graph prediction", yticks=[0, .25, .5, .75, 1],
                 xticks=list(range(2816, 3073, 32)))
    timeline.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.26))
    block_means = [math.fsum(row["f1"] for row in rows if row["motif_presentation"] == block) / 32
                   for block in range(1, 9)]
    motifs.plot(range(1, 9), block_means, color="#215b8f", marker="o", markersize=4, lw=1.2)
    motifs.set(xlim=(.7, 8.3), ylim=(-0.035, 1.055), ylabel="Mean F1 (32 targets)",
               xlabel="Motif presentation within held-out sequence", title="b  Successive repeated motifs",
               xticks=list(range(1, 9)), xticklabels=[f"{i}\n{'A' if i % 2 else 'B'}" for i in range(1, 9)],
               yticks=[0, .5, 1])
    phases.axvspan(.5, 8.5, color="#e4f0e5", lw=0, label="Early: phases 1–8")
    phases.axvspan(23.5, 31.5, color="#f6ecd9", lw=0, label="Late: phases 24–31")
    for motif, color in (("A", "#215b8f"), ("B", "#b56f21")):
        points = [row for row in aligned if row["motif"] == motif]
        delta = contrast["motifs"][motif]["late_minus_early_f1"]
        phases.plot([row["within_motif_phase"] for row in points], [row["mean_f1"] for row in points],
                    color=color, lw=1.3, ls="-" if motif == "A" else "--", label=f"Motif {motif} (late − early: {delta:+.3f})")
    phases.axvline(0, color=".55", ls=":", lw=.8)
    phases.set(xlim=(-.5, 31.5), ylim=(-.035, 1.055), ylabel="Phase-aligned mean F1",
               xlabel="Phase within a 32-step motif (0 = graph transition)",
               title="c  Within-motif behavior (four correlated cycles)",
               xticks=[0, 1, 8, 16, 24, 31], yticks=[0, .5, 1])
    handles, labels = phases.get_legend_handles_labels()
    order = [2, 3, 0, 1]
    phases.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
                  fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.28))
    for axis in (timeline, motifs, phases):
        axis.grid(axis="y", color="0.87", lw=.6)
        axis.set_axisbelow(True)
    fig.suptitle("Seed 0 · capped corrected GC-LSTM reconstruction\n"
                 f"Pdet(2,32) · selected epoch {acceptance['selected_epoch']} · behavioral evidence", fontsize=11)
    paths = []
    for suffix in ("pdf", "png"):
        path = output / ("figure10-reconstruction." + suffix)
        fig.savefig(path, dpi=250, metadata={"Title": "Figure 10 behavioral reconstruction"})
        paths.append(path)
    plt.close(fig)
    csv_path = output / "figure10-reconstruction.csv"
    columns = ["target_timestamp", "source_timestamp", "test_target_index", "target_phase", "source_phase",
               "motif_presentation", "test_cycle", "motif", "change_point", "pairs", "tp", "fp", "fn", "tn",
               "precision", "recall", "f1", "accuracy"]
    with csv_path.open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    paths.append(csv_path)
    aligned_path = output / "figure10-within-motif.csv"
    with aligned_path.open("x", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=["motif", "within_motif_phase", "observations",
                                                   "mean_f1", "contributing_target_timestamps"])
        writer.writeheader()
        writer.writerows({**row, "contributing_target_timestamps": ";".join(map(str, row["contributing_target_timestamps"]))}
                         for row in aligned)
    paths.append(aligned_path)
    manifest = {"kind": "figure10_behavioral_reconstruction_plot_v2" if is_v2 else "figure10_behavioral_reconstruction_plot_v1",
        **({"schema_version": 2} if is_v2 else {}), "evaluation_sha256": evaluation_hash,
        "acceptance_sha256": acceptance_hash, "analysis_protocol_sha256": protocol_sha256, "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": acceptance["run_id"], "model_seed": 0, "selected_epoch": acceptance["selected_epoch"],
        "training_stop_reason": acceptance["training_stop_reason"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"], **identity,
        "reduction": reduction, "within_motif_descriptive": contrast,
        "model_replicates": 1, "targets": 256, "motif_presentations": 8, "independent_motif_replicates": False,
        "scientific_collapse_claim": False, "original_paper_curve_included": False,
        "caption": "Per-target F1 over all 9900 ordered off-diagonal pairs at threshold >=0.5; "
                   "unweighted temporal means. Shading identifies alternating 32-step motifs A and B. "
                   "Orange points mark eight actual adjacent-graph transitions, including the first held-out target. "
                   "The middle panel averages each successive 32-target motif presentation. "
                   "The bottom panel aligns four cycles for each motif, without inferential uncertainty bands. "
                   "Its fixed descriptive contrast compares phases 24–31 with phases 1–8, excluding transition phase zero. "
                   "One seed and one contiguous sequence; motif presentations are not independent replications. "
                   "This capped corrected reconstruction measures behavior and does not establish representation collapse.",
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (output / "figure-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "evaluation-sha256", "acceptance", "acceptance-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    manifest = render(evaluation_path=args.evaluation, evaluation_hash=args.evaluation_sha256,
        acceptance_path=args.acceptance, acceptance_hash=args.acceptance_sha256, output=args.output)
    print(json.dumps({"output": args.output, "targets": manifest["targets"], "scientific_collapse_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
