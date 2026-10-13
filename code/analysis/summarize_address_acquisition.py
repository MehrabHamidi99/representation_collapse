"""Reproduce the accepted singleton-acquisition endpoints and pinned baselines."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "doc/auto/address-acquisition-v4-results.json"
OUTPUT = ROOT / "doc/auto/figures"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(INPUT.read_text())
    if (data["fit_count"] != 12 or data["total_optimizer_updates"] != 60000
            or data["fixture_only"] or not data["scientific_acceptance"]
            or data["original_eight_coordinate_eligibility"]
            or data["harm_or_floquet_claim_accepted"] or data["selected_checkpoints"]):
        raise ValueError("accepted fixed-endpoint singleton scope required")
    models = ("gru", "lstm")
    tasks = ("categorical", "continuous")
    addresses = (0, 7, 15)
    fits = {(f["model"], f["task"], f["address"]): f for f in data["fits"]}
    baselines = {(f["model"], f["task"], f["address"]): f for f in data["baselines"]}
    if len(data["fits"]) != 12 or set(fits) != {
            (m, t, q) for m in models for t in tasks for q in addresses}:
        raise ValueError("all twelve unique fixed endpoints must be retained")
    if len(data["baselines"]) != 8 or set(baselines) != {
            (m, t, q) for m in models for t in tasks for q in (0, 7)}:
        raise ValueError("baseline comparison must stay within QUERY support 0/7")
    plotted = {}
    for key, fit in fits.items():
        task = key[1]
        value = fit["endpoint_metric"]
        if (fit["updates"] != 5000 or not np.isfinite(value)
                or (task == "categorical" and not 0 <= value <= 1)
                or (task == "continuous" and value <= 0)):
            raise ValueError("finite supported fixed-endpoint metrics required")
        passed = value >= .95 if task == "categorical" else value <= .05
        if passed != fit["endpoint_passed"]:
            raise ValueError("plotted metric disagrees with the recorded gate")
        baseline = baselines.get(key)
        if baseline is not None:
            if (baseline["optimizer_updates"] != 0 or baseline["validation_sampling_seed"] != 20003
                    or not np.isfinite(baseline["metric"])):
                raise ValueError("pinned, reevaluated baseline required")
            contrast = fit["paired_baseline_contrast"]
            expected = ((1 - value) - (1 - baseline["metric"]) if task == "categorical"
                        else value - baseline["metric"])
            if (contrast["paired_episodes"] != 128
                    or not np.isclose(expected, contrast["mean_paired_error_difference"], atol=1e-12, rtol=0)
                    or contrast["confidence_interval"] is not None or contrast["p_value"] is not None):
                raise ValueError("descriptive paired baseline contrast does not reconcile")
        elif fit["paired_baseline_contrast"] is not None:
            raise ValueError("address15 must not acquire an unsupported primary baseline")
        plotted["/".join(map(str, key))] = {
            "singleton": value, "pinned_baseline": None if baseline is None else baseline["metric"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.8), layout="constrained")
    for row, model in enumerate(models):
        for col, task in enumerate(tasks):
            ax = axes[row, col]
            ax.scatter(np.arange(3) - .07, [fits[model, task, q]["endpoint_metric"] for q in addresses],
                       marker="o", s=27, color="#2166AC", label="Fixed-address training")
            ax.scatter(np.arange(2) + .07, [baselines[model, task, q]["metric"] for q in (0, 7)],
                       marker="x", s=32, color="#D6604D", label="Original-query training; reevaluated")
            ax.axhline(.95 if task == "categorical" else .05,
                       color="#444444", linestyle="--", linewidth=1, label="Singleton competence gate")
            ax.axhline(1 / 16 if task == "categorical" else 1,
                       color="#999999", linestyle=":", linewidth=1, label="Population reference")
            ax.set_title(f"{model.upper()}: {task}", loc="left", fontweight="bold")
            ax.set_xticks(range(3), [str(q) for q in addresses])
            ax.set_xlim(-.4, 2.4)
            ax.set_ylabel("Accuracy" if task == "categorical" else "MSE / (1/3); log scale")
            if task == "categorical":
                ax.set_ylim(-.025, 1.05)
            else:
                ax.set_yscale("log"); ax.set_ylim(1e-5, 2)
            ax.grid(axis="y", alpha=.15)
            if row == 1:
                ax.set_xlabel("Queried WRITE address")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Separate singleton models; fixed 5,000-update endpoints; no body", fontsize=10)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / f"address-acquisition-v4-endpoints.{ext}" for ext in ("pdf", "png")]
    fig.savefig(paths[0], bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=220, bbox_inches="tight"); plt.close(fig)
    manifest = {
        "schema_version": 1, "kind": "descriptive_singleton_acquisition_figure",
        "input_sha256": sha(INPUT), "generator_sha256": sha(Path(__file__)),
        "matplotlib": matplotlib.__version__, "outputs": {p.name: sha(p) for p in paths},
        "source_commit": data["source_commit"],
        "independent_result_review_receipt_sha256": data["independent_result_review_receipt_sha256"],
        "claim_scope": data["scientific_acceptance_scope"], "plotted_values": plotted,
        "validation_sampling_seed": 20003, "task_specific_shared_histories": 256,
        "histories_per_task": 128, "training_initialization_seeds": [0],
        "no_trained_seed_uncertainty_estimated": True,
        "baselines": "pinned v3 fresh endpoints, reevaluated without fitting; QUERY15 outside original selection support",
        "limits": "address, lag, known future query and required load vary jointly; separate singleton weights are not one multi-query model",
    }
    (OUTPUT / "address-acquisition-v4-endpoints.manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"input_sha256": manifest["input_sha256"], "outputs": manifest["outputs"]}))


if __name__ == "__main__":
    main()
