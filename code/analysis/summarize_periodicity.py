#!/usr/bin/env python3
"""Publish descriptive scalar summaries of the frozen periodicity reduction.

First extraction (the envelope is ignored and contains derived values only)::

    .venv/bin/python code/analysis/summarize_periodicity.py --envelope \
        runs/ops/periodicity-independent-reduction-612745-20261012.json

Recreate the report and figures from the committed summary, without raw arrays::

    .venv/bin/python code/analysis/summarize_periodicity.py

No model, checkpoint, cluster or provider is accessed. This is a descriptive
numerical audit of failed-competence checkpoints, not scientific acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "doc/auto/periodicity-independent-results.json"
REPORT = ROOT / "doc/auto/periodicity-independent-results.md"
FIGURE = ROOT / "doc/auto/figures/periodicity-independent-v1"
HORIZONS = (1, 2, 16, 32)
PRECISIONS = ("float32", "float64")
ARMS = ("periodic", "control")
CELL_ORDER = (("gru", "categorical"), ("gru", "continuous"),
              ("lstm", "categorical"), ("lstm", "continuous"))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/periodicity-summary/mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "runs/periodicity-summary/cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def compact_closure(value: dict | None) -> dict | None:
    if value is None:
        return None
    # Preserve exact terminal-block diagnostics; omit the phase arrays already
    # retained in the full derived report. The norm spans 13 aligned states.
    return {key: value[key] for key in (
        "cycle_pair", "absolute", "relative", "reference_scale", "scale_floor",
        "maximum_coordinate_absolute", "absolute_tolerance", "relative_tolerance",
        "passed", "coordinatewise_passed", "ulp")}


def extract(envelope_path: Path) -> dict:
    envelope = json.loads(envelope_path.read_text())
    report = envelope["artifacts"]["report.json"]
    provenance = envelope["artifacts"]["provenance.json"]
    assert envelope["cluster"] == "tamia" and envelope["job_id"] == 612745
    assert provenance["source_commit"] == "ec1c489fc97c2e6cf94fd58e8a4e04dd3e8ba901"
    assert report["reducer_sha256"] == "8c8acfc68ae82baa7f746673ad56ea47be667a4d53cb015e5171c18ad2708bf4"
    assert envelope["verified_hashes"]["report.json"] == "dd27cf606a2d0a0598d5018983731225aaa7b341ac331ed56fcfac53264b0c1b"
    assert report["independent_raw_reduction_passed"] and not report["errors"]
    assert report["all_frozen_cells_present"] and report["total_dense_windows"] == 64
    assert len(report["cells"]) == 4
    assert not report["scientific_acceptance"]
    assert report["reducer_sha256"] == provenance["source_sha256"][
        "code/analysis/reduce_periodicity.py"]
    assert report["frozen_config_sha256"] == sha256(
        ROOT / "configs/experiments/periodicity-analysis-v1.json")
    cells = []
    for model, task in CELL_ORDER:
        matches = [c for c in report["cells"]
                   if (c["cell"]["model"], c["cell"]["task"]) == (model, task)]
        assert len(matches) == 1
        cell = matches[0]
        assert cell["numerical_contract_passed"] and not cell["numerical_failures"]
        assert cell["independent_raw_reduction_passed"]
        assert not cell["scientific_acceptance"] and not cell["floquet_acceptance"]
        assert cell["cell"]["seed"] == 0 and not cell["cell"]["baseline_competence_passed"]
        assert len(cell["assays"]) == 32
        assays = []
        for panel in (0, 1):
            for precision in PRECISIONS:
                for arm in ARMS:
                    for repeats in HORIZONS:
                        original = cell["assays"][f"{precision}_panel{panel}_R{repeats}_{arm}"]
                        separation = original["paired_cue_separation"]
                        native = separation["native_pre_query"]
                        manual = separation["pre_query"]
                        assert math.isfinite(native) and native >= 0
                        assert native == separation["native_separation_l2"][-1]
                        assert manual == separation["manual_separation_l2"][-1]
                        assert not separation["dimension_estimate_available"]
                        cues = []
                        for cue in original["cues"]:
                            tangent = cue["tangent"]
                            if tangent is not None:
                                assert cue["cue"] == 0 and repeats in (2, 16)
                                assert tangent["window_steps"] == 12
                                assert not tangent["floquet_acceptance"]
                                tangent = {key: tangent[key] for key in (
                                    "window_body_start", "window_steps", "largest_singular_value",
                                    "spectral_radius", "finite_window_log_gain_per_step",
                                    "interpretation", "floquet_acceptance",
                                    "exact_periodic_orbit_verified", "approximate_manual_closure_compatible",
                                    "jacobian_closure", "product_direct_agreement",
                                    "dtype_product_recomputation")}
                            cues.append({
                                "cue": cue["cue"],
                                "terminal_native_cycle": compact_closure(
                                    cue["native_cycles"][-1] if repeats > 1 else None),
                                "terminal_manual_cycle": compact_closure(
                                    cue["manual_cycles"][-1] if repeats > 1 else None),
                                "adapter_max_errors": cue["adapter_max_errors"],
                                "adapter_saved_forward_checks_passed": cue["adapter_saved_forward_checks_passed"],
                                "tangent": tangent,
                            })
                        assays.append({
                            "panel": panel, "precision": precision, "arm": arm,
                            "repeats": repeats, "body_steps": repeats * 12,
                            "driver": original["driver"],
                            "native_after_write_l2": separation["native_after_write"],
                            "native_pre_query_l2": native,
                            "manual_after_write_l2": separation["after_write"],
                            "manual_pre_query_l2": manual,
                            "native_pre_query_ulp": separation["native_pre_query_ulp"],
                            "manual_pre_query_ulp": separation["pre_query_ulp"],
                            "queried_targets_differ": separation["queried_targets_differ"],
                            "query_coordinate": separation["query_coordinate"],
                            "cues": cues,
                        })
        cells.append({
            **cell["cell"], "state_dimension": 128 if model == "gru" else 256,
            "state_order": "h" if model == "gru" else "[h,c]",
            "artifact_manifest_sha256": cell["manifest_sha256"],
            "artifact_hashes": cell["artifact_hashes"],
            "numerical_contract_passed": cell["numerical_contract_passed"],
            "dense_windows": cell["dense_windows"],
            "step_jacobian_evaluations": cell["step_jacobian_evaluations"],
            "r1_anchor_checks": cell["r1_anchor_checks"],
            "query_adapter_validation": cell["query_adapter_validation"],
            "probe_floor_sensitivity": cell["probe_floor_sensitivity"],
            "assays": assays,
        })
    return {
        "schema_version": 1, "kind": "descriptive_independent_periodicity_summary",
        "scientific_acceptance": False, "floquet_acceptance": False,
        "independent_raw_reduction_passed": True,
        "training_source_commit": report["cells"][0]["training_source_commit"],
        "analysis_source_commit": report["cells"][0]["analysis_source_commit"],
        "reduction_source_commit": provenance["source_commit"],
        "reducer_sha256": report["reducer_sha256"],
        "frozen_config_sha256": report["frozen_config_sha256"],
        "source": {
            "envelope_sha256": sha256(envelope_path),
            "original_report_sha256": envelope["verified_hashes"]["report.json"],
            "canonical_embedded_report_sha256": canonical_hash(report),
            "hash_convention": "Original report hash is coordinator-verified remote bytes; canonical embedded hash uses sorted compact JSON.",
        },
        "execution": {
            "cluster": "tamia", "job_id": 612745, "state": "COMPLETED",
            "exit_code": "0:0", "start_utc": "2026-10-12T09:00:41Z",
            "end_utc": "2026-10-12T09:00:55Z", "scheduler_elapsed_seconds": 14,
            "maximum_recorded_step_rss_kib": 684480,
            "requested_resources": provenance["requested_resources"],
            "reducer_runtime_seconds": report["runtime_seconds"],
            "total_dense_windows": 64, "total_step_jacobian_evaluations": 1536,
        },
        "measurement": {
            "horizons": list(HORIZONS), "body_updates_per_repeat": 12,
            "training_seeds_per_cell": 1, "fixed_driver_panels_per_cell": 2,
            "cue_histories_per_panel": 2,
            "pre_query_separation": "Absolute Euclidean norm of paired native one-step complete-state difference; no centering or scale normalization.",
            "complete_state": "GRU h (128); LSTM [h,c] (256). QUERY excluded.",
            "precision": "Fixed weights and literal FP32 inputs; separate native FP32/FP64 dynamics from zero. FP64 is not merely product accumulation.",
            "closure": "Last adjacent pair of 13 phase-matched complete states; global absolute AND relative AND coordinate bounds; R1 unavailable.",
            "ulp": "Absolute coordinate difference divided by maximum absolute local np.spacing of the saved coordinates; not integer representable-value distance.",
            "tangents": "Last 12 body steps at R2/R16, cue zero, differentiated manual adapter; finite-time products only.",
            "sampling": "Within-checkpoint deterministic measurements; driver ranges are not confidence intervals. Control source prefixes are shared across horizons except for forced terminal motifs, so complete bodies need not be nested.",
        },
        "comparison_tolerances": report["comparison_tolerances"],
        "unavailable": [
            "Independent manual/native query-output comparison: manual query arrays absent.",
            "Original probe feature-floor re-reduction: probe train/test clouds absent.",
            "Useful covariance dimension and independent training-seed uncertainty.",
            "Current checkpoint byte rehash by this reducer (frozen identity checked; separate coordinator evidence required).",
        ],
        "cells": cells,
    }


def range_string(values: list[float], digits: int = 3) -> str:
    return f"{min(values):.{digits}g}–{max(values):.{digits}g}"


def make_report(summary: dict) -> None:
    cells = summary["cells"]
    assays = [a for c in cells for a in c["assays"]]
    long = [a for a in assays if a["repeats"] in (16, 32)]
    lines = [
        "# Independent periodicity reduction: descriptive results", "",
        "The frozen [independent protocol](periodicity-independent-reduction-v1.md) passed its raw-array numerical audit for all four seed-0 checkpoints. **Every original native-competence gate failed; harmful collapse and Floquet claims remain unaccepted.** This report contains newly verified measurements, with no retraining or new model selection.", "",
        "The committed [scalar summary](periodicity-independent-results.json) preserves exact per-panel, arm, precision and horizon values, terminal native/manual closure diagnostics and local-spacing ratios. The original arrays and weights remain on cluster scratch.", "",
        "![Native complete-state paired-cue separation](figures/periodicity-independent-v1.png)", "",
        "**Figure.** Absolute pre-QUERY separation of two required-cue histories with shared nuisance and body inputs, using native one-step recurrent updates. GRU uses 128 hidden coordinates; LSTM uses all 256 `[h,c]` coordinates. Color denotes forcing arm; solid/dashed lines denote FP32/FP64; circles/triangles show the two fixed driver panels individually. No centering or scale normalization. All plotted separations are nonzero; zeros would be shown on a separate strip, never replaced by a positive floor. Lines join measured points for readability: control source prefixes are shared except for forced terminal motifs, so complete bodies need not be nested across horizons. These repeated measurements of one seed per cell are not independent replicates or confidence intervals. R1 is an identical-arm anchor; R32 extends the original R16 evaluation horizon and has state measurements only, with no tangent or Jacobian assay.", "",
        "Measured findings:", "",
    ]
    for precision in PRECISIONS:
        values = [a["native_pre_query_l2"] for a in long if a["precision"] == precision]
        changed = [a["native_pre_query_ulp"]["bitwise_changed_coordinates"] for a in long if a["precision"] == precision]
        lines.append(f"- At R16/R32, native {precision.upper().replace('FLOAT', 'FP')} cue separation is {range_string(values)} across both arms and all cells. Every pair remains bitwise different ({min(changed)}–{max(changed)} changed coordinates); small amplitude does not establish an exactly merged state or the absence of decodable information.")
    gains = [cue["tangent"]["largest_singular_value"] for a in assays for cue in a["cues"] if cue["tangent"] is not None]
    lines.append(f"- All 64 observed 12-step manual tangent gains are below one ({range_string(gains)}). Per-cell ranges are " + "; ".join(
        f"{c['model'].upper()} {c['task']}: {range_string([q['tangent']['largest_singular_value'] for a in c['assays'] for q in a['cues'] if q['tangent'] is not None], 5)}"
        for c in cells) + ". These are local finite-time products at cue zero, not cue-channel derivatives, native readout tests or certified monodromy matrices.")
    for precision in PRECISIONS:
        periodic = [q["terminal_native_cycle"] for a in assays if a["precision"] == precision and a["arm"] == "periodic" and a["repeats"] == 32 for q in a["cues"]]
        control = [q["terminal_native_cycle"] for a in assays if a["precision"] == precision and a["arm"] == "control" and a["repeats"] == 32 for q in a["cues"]]
        spacing = [x["ulp"]["maximum_coordinate_spacing_ratio"] for x in periodic]
        lines.append(f"- At R32, all 16 periodic native {precision.upper().replace('FLOAT', 'FP')} last-cycle comparisons pass frozen tolerances: absolute block residual {range_string([x['absolute'] for x in periodic])}. None is bitwise equal; maximal coordinate differences are {range_string(spacing)} local spacings. All 16 control comparisons fail (residual {range_string([x['absolute'] for x in control])}); their adjacent blocks need not have identical inputs. Tolerance-qualified closure does not certify an exact orbit.")
    special = next(c for c in cells if (c["model"], c["task"]) == ("lstm", "categorical"))
    special = next(a for a in special["assays"] if (a["panel"], a["precision"], a["arm"], a["repeats"]) == (1, "float32", "control", 16))
    manual = [q["terminal_manual_cycle"]["absolute"] for q in special["cues"]]
    native = [q["terminal_native_cycle"]["absolute"] for q in special["cues"]]
    lines.append(f"- LSTM categorical, driver panel 1, control R16, FP32: both manual last-cycle checks pass ({range_string(manual)}), while both native checks fail ({range_string(native)}) the unchanged `1e-7` absolute bound. The driver has no proper whole-body period. Adapter tolerance passes do not preserve closure classifications at this scale; no threshold was relaxed.")
    lines += [
        "", "Limits and interpretation. Two histories give at most rank-one centered covariance and cannot estimate representation dimension. Both forcing arms attenuate these particular cue differences, without establishing their population ordering effect or empirical equivalence. One checkpoint per cell, two fixed driver panels, arithmetic dependence, non-nested complete control bodies and failed native competence prevent a learned harmful-collapse conclusion. Manual QUERY arrays and original probe clouds were not saved, so query-adapter validation is partial and probe-floor robustness remains pending. Numerical reduction does not establish a causal mechanism, competent-model equivalence or transfer across modalities.", "",
        "Provenance. Training `9004a0c114a8b7c5335dca3d5247449fd10b5ffa`; diagnostic producer `458bc9cfce474b8e11e433e33bf132faf5032387`; independent reducer `ec1c489fc97c2e6cf94fd58e8a4e04dd3e8ba901`, SHA-256 `" + summary["reducer_sha256"] + "`. Tamia job **612745** completed `0:0`, 2026-10-12 09:00:41–09:00:55 UTC (14 s), maximum recorded step RSS 684,480 KiB, one CPU/2 GiB/zero GPUs. It verified 64 tangent windows and 1,536 step-Jacobian evaluations. The reducer checks frozen checkpoint identity, not checkpoint file bytes; the coordinator's earlier integrity evidence remains separate.", "",
        "Reproduce the vector [PDF](figures/periodicity-independent-v1.pdf), PNG and report from the committed scalar summary (no raw arrays needed):", "",
        "```bash", ".venv/bin/python code/analysis/summarize_periodicity.py", "```", "",
        "The [figure manifest](figures/periodicity-independent-v1.manifest.json) records source/report hashes, exact plotted values and deterministic render outputs. The locked paper dependency group supplies Matplotlib. Scientific acceptance remains the coordinator's separate decision.", "",
    ]
    REPORT.write_text("\n".join(lines))


def make_figure(summary: dict) -> dict:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    colors = {"periodic": "#0072B2", "control": "#D55E00"}
    positive = [a["native_pre_query_l2"] for c in summary["cells"] for a in c["assays"] if a["native_pre_query_l2"] > 0]
    ymin = 10 ** (math.floor(math.log10(min(positive))) - 1)
    ymax = 10 ** math.ceil(math.log10(max(positive)))
    has_zeros = any(a["native_pre_query_l2"] == 0 for c in summary["cells"] for a in c["assays"])
    fig = plt.figure(figsize=(8.0, 7.0 if not has_zeros else 7.7))
    outer = fig.add_gridspec(2, 2, left=.095, right=.985, bottom=.18, top=.87,
                            wspace=.23, hspace=.29)
    plotting_values = []
    for index, cell in enumerate(summary["cells"]):
        zero_ax = None
        if has_zeros:
            inner = outer[index // 2, index % 2].subgridspec(2, 1, height_ratios=(7, 1), hspace=.06)
            ax = fig.add_subplot(inner[0])
            zero_ax = fig.add_subplot(inner[1], sharex=ax)
            zero_ax.set_ylim(-.5, .5)
            zero_ax.set_yticks([0], ["exact 0"])
            zero_ax.tick_params(axis="x", labelbottom=False)
            zero_ax.spines["bottom"].set_visible(False)
        else:
            ax = fig.add_subplot(outer[index // 2, index % 2])
        for panel in (0, 1):
            for precision in PRECISIONS:
                for arm in ARMS:
                    observations = [next(a for a in cell["assays"] if
                        (a["panel"], a["precision"], a["arm"], a["repeats"]) ==
                        (panel, precision, arm, repeats)) for repeats in HORIZONS]
                    values = [a["native_pre_query_l2"] for a in observations]
                    y = [v if v > 0 else float("nan") for v in values]
                    marker = "o" if panel == 0 else "^"
                    ax.plot(HORIZONS, y, color=colors[arm], marker=marker,
                            markersize=4.4, markeredgewidth=.85,
                            markerfacecolor="white" if panel else colors[arm],
                            linestyle="-" if precision == "float32" else "--",
                            alpha=.9)
                    if zero_ax is not None:
                        zeros = [r for r, value in zip(HORIZONS, values) if value == 0]
                        zero_ax.scatter(zeros, [0] * len(zeros), color=colors[arm], marker=marker, s=18)
                    plotting_values.append({
                        "model": cell["model"], "task": cell["task"], "panel": panel,
                        "precision": precision, "arm": arm, "repeats": list(HORIZONS),
                        "native_complete_state_pre_query_separation_l2": values,
                    })
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS, ["1", "2", "16", "32"])
        ax.set_xlim(.83, 38.5)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks([1e-16, 1e-12, 1e-8, 1e-4])
        ax.grid(axis="y", color="#e0e0e0", linewidth=.55)
        ax.set_title(f"{'abcd'[index]}) {cell['model'].upper()} · {cell['task']}", loc="left", pad=8)
        if index % 2 == 0:
            ax.set_ylabel(r"Native paired-cue separation $\|s^{(0)}-s^{(1)}\|_2$")
        if index >= 2:
            ax.set_xlabel("Body repetition count R (12 updates per repeat)")
    handles = [
        Line2D([0], [0], color=colors["periodic"], lw=2, label="Periodic"),
        Line2D([0], [0], color=colors["control"], lw=2, label="Control"),
        Line2D([0], [0], color="#444444", lw=1.5, linestyle="-", label="FP32 dynamics"),
        Line2D([0], [0], color="#444444", lw=1.5, linestyle="--", label="FP64 dynamics"),
        Line2D([0], [0], color="#444444", marker="o", linestyle="none", markersize=5, label="Driver panel 0"),
        Line2D([0], [0], color="#444444", marker="^", markerfacecolor="white", linestyle="none", markersize=5, label="Driver panel 1"),
    ]
    fig.suptitle("Cue separation attenuates in both forcing arms", y=.975, fontsize=12)
    fig.text(.5, .933, "Four seed-0 checkpoints · all original native-competence gates failed",
             ha="center", color="#555555", fontsize=9)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .058),
               ncol=3, frameon=False, fontsize=8.5, columnspacing=2.0)
    fig.text(.5, .025, "Two fixed driver panels shown individually; no confidence intervals. Complete control bodies need not be nested.",
             ha="center", fontsize=8, color="#555555")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".pdf"), metadata={
        "Title": "Native paired-cue separation in failed-competence recurrent checkpoints",
        "Author": "Representation Collapse project", "CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE.with_suffix(".png"), dpi=240,
                metadata={"Software": "summarize_periodicity.py"})
    plt.close(fig)
    return {
        "schema_version": 1, "scientific_acceptance": False,
        "command": ".venv/bin/python code/analysis/summarize_periodicity.py",
        "source_hashes": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
                          str(SUMMARY.relative_to(ROOT)): sha256(SUMMARY)},
        "full_report_hashes": summary["source"],
        "versions": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "plotting_values": plotting_values,
        "scale": {"x": "base-2 log", "y": "base-10 log, absolute unnormalized L2",
                  "y_limits": [ymin, ymax], "zero_count": sum(v == 0 for c in summary["cells"] for a in c["assays"] for v in [a["native_pre_query_l2"]]),
                  "zero_rendering": "Separate exact-zero strip if any; no artificial positive floor."},
        "outputs": {str(path.relative_to(ROOT)): sha256(path)
                    for path in [FIGURE.with_suffix(".pdf"), FIGURE.with_suffix(".png"), REPORT]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", type=Path,
                        help="Extract fresh scalar summary from the ignored verified reduction envelope.")
    args = parser.parse_args()
    if args.envelope:
        atomic_json(SUMMARY, extract(args.envelope))
    summary = json.loads(SUMMARY.read_text())
    assert not summary["scientific_acceptance"] and not summary["floquet_acceptance"]
    make_report(summary)
    manifest = make_figure(summary)
    atomic_json(FIGURE.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"cells": len(summary["cells"]), "plot_series": len(manifest["plotting_values"]),
                      "plotted_values": sum(len(x["repeats"]) for x in manifest["plotting_values"]),
                      "zero_count": manifest["scale"]["zero_count"],
                      "scientific_acceptance": False}, sort_keys=True))


if __name__ == "__main__":
    main()
