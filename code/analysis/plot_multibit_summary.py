"""Plot independently reduced discovery outputs; no new model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    assert not report["failed_checks"] and report["fits_found"] == 4
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), layout="constrained")
    colors = {1: "#315d49", 4: "#b85d3a"}
    for dimension in (1, 4):
        fit = next(f for f in report["fits"] if f["fit"] == f"D{dimension}-seed0")
        rows = [r for r in fit["evaluation"] if r["condition"].endswith("-q1")
                and r["condition"] != "R0-q1"]
        rows.sort(key=lambda r: int(r["condition"].split("-")[0][1:]))
        steps = [12 * int(r["condition"].split("-")[0][1:]) for r in rows]
        for arm, marker, style in (("periodic", "o", "-"), ("control", "x", "--")):
            assert all(r["per_arm"][arm]["whole_vector_exact_accuracy"] == 1 for r in rows)
            for ax, metric in zip(axes, ("mean_participation_ratio", "mean_trace")):
                ax.plot(steps, [r["per_arm"][arm][metric] for r in rows],
                        marker=marker, ls=style, color=colors[dimension], linewidth=2,
                        markersize=5, label=f"{dimension} bit{'s' if dimension > 1 else ''}, {arm}")
    for ax in axes:
        ticks = [12, 24, 48, 96, 192, 384, 768, 1536]
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{x:,}" for x in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Body steps between WRITE and QUERY (dwell = 1)")
        ax.grid(axis="y", alpha=.15)
    axes[0].set(title="Useful memory survives without a falling dimension",
                ylabel="Complete-state participation ratio", ylim=(0, 9))
    axes[0].legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    axes[1].set(yscale="log", ylabel="Complete-state covariance trace",
                title="State variation grows while recall stays perfect")
    fig.savefig(args.output / "repetition-multibit.png", dpi=200)
    plt.close(fig)
    provenance = {"purpose": "Plot of independently reduced exploratory seed-0 outputs",
                  "wandb_required": False,
                  "wandb_reason": "Replotting existing results; no new scientific evaluation",
                  "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "output_sha256": hashlib.sha256((args.output / "repetition-multibit.png").read_bytes()).hexdigest(),
                  "aggregation": "Mean of four within-forcing/arm covariance summaries; H64 then C64",
                  "limitations": ["one fit per load; no population confidence interval",
                                  "validation gates checked from summaries, not raw replay",
                                  "D16 and D64 failed competence and are absent from held-out curves",
                                  "No tangent or Floquet spectrum measured"]}
    (args.output / "multibit-plot-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
