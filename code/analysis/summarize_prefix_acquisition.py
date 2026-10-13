"""Reproduce a descriptive acquisition figure from committed scalar evidence."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "doc/auto/prefix-acquisition-v3-results.json"
OUTPUT = ROOT / "doc/auto/figures"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(INPUT.read_text())
    if (data["fit_count"] != 8 or data["total_optimizer_updates"] != 40000
            or data["coordinate_ids"] != list(range(8)) or data["periodic_body_used"]
            or data["trained_seed_ids"] != [0]):
        raise ValueError("summary differs from the frozen descriptive acquisition scope")
    fits = {(f["model"], f["task"], f["condition"]): f for f in data["fits"]}
    conditions = ("fresh_seed0", "immediate_step500_transfer_reset_optimizer")
    expected = {(m, t, c) for m in ("gru", "lstm")
                for t in ("categorical", "continuous") for c in conditions}
    if len(data["fits"]) != 8 or set(fits) != expected:
        raise ValueError("all eight unique fixed endpoints must be retained")
    for (_, task, _), fit in fits.items():
        scores = np.asarray(fit["endpoint_coordinate_metrics"], dtype=float)
        if fit["updates"] != 5000 or scores.shape != (8,) or not np.isfinite(scores).all():
            raise ValueError("finite eight-coordinate fixed endpoints required")
        passed = np.all(scores >= .95) if task == "categorical" else np.all(scores <= .05)
        if bool(passed) != fit["endpoint_coordinate_gate_passed"]:
            raise ValueError("displayed scores disagree with the verified gate")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.25), sharex=True,
                             layout="constrained")
    labels = ("Fresh initialization", "Immediate transfer; AdamW reset")
    for row, model in enumerate(("gru", "lstm")):
        for col, task in enumerate(("categorical", "continuous")):
            ax = axes[row, col]
            for condition, color, label in zip(conditions, ("#2166AC", "#D6604D"), labels):
                fit = fits[model, task, condition]
                ax.plot(range(8), fit["endpoint_coordinate_metrics"], marker="o",
                        markersize=3.5, linewidth=1.4, color=color, label=label)
            ax.axhline(.95 if task == "categorical" else .05,
                       color="#444444", linestyle="--", linewidth=1,
                       label="Coordinate competence gate")
            ax.axhline(1 / 16 if task == "categorical" else 1,
                       color="#999999", linestyle=":", linewidth=1,
                       label="Population reference")
            ax.set_title(f"{model.upper()}: {task}", loc="left", fontweight="bold")
            ax.set_ylabel("Accuracy" if task == "categorical" else "MSE / (1/3)")
            ax.set_ylim(0, 1.05 if task == "categorical" else 1.17)
            ax.set_xticks(range(8)); ax.grid(axis="y", alpha=.15)
            if row == 1:
                ax.set_xlabel("Required WRITE coordinate")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2,
               frameon=False, fontsize=8)
    fig.suptitle("Fixed 5,000-update endpoints; no distractor body", fontsize=10)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [OUTPUT / f"prefix-acquisition-v3-endpoints.{ext}" for ext in ("pdf", "png")]
    fig.savefig(paths[0], bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=220, bbox_inches="tight"); plt.close(fig)
    manifest = {
        "schema_version": 1, "kind": "descriptive_acquisition_endpoint_figure",
        "input_sha256": sha(INPUT), "generator_sha256": sha(Path(__file__)),
        "matplotlib": matplotlib.__version__,
        "outputs": {p.name: sha(p) for p in paths},
        "source_commit": data["source_commit"],
        "independent_review_receipt_sha256": data["independent_review_receipt_sha256"],
        "claim_scope": data["scientific_scope"],
        "no_trained_seed_uncertainty_estimated": True,
        "references": {"categorical": "uniform classifier expected accuracy1/16",
                       "continuous": "population zero-predictor MSE/(1/3)=1; not sample-normalized"},
        "plotted_values": {"/".join(key): fit["endpoint_coordinate_metrics"]
                           for key, fit in sorted(fits.items())},
    }
    target = OUTPUT / "prefix-acquisition-v3-endpoints.manifest.json"
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"input_sha256": manifest["input_sha256"],
                      "outputs": manifest["outputs"]}))


if __name__ == "__main__":
    main()
