#!/usr/bin/env python3
"""Render independently reconciled archive summaries; never read raw arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pinned_json(path: Path, digest: str) -> dict:
    data = path.read_bytes()
    if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("compact input identity or size mismatch")
    return json.loads(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "audit-sha256", "review", "review-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    audit = pinned_json(Path(args.audit), args.audit_sha256)
    review = pinned_json(Path(args.review), args.review_sha256)
    if not audit.get("integrity_completed") or not review.get("numerical_integrity_accepted"):
        raise ValueError("completed independent numerical reconciliation required")
    if review.get("audit_sha256") != args.audit_sha256:
        raise ValueError("review is not bound to the plotted audit")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Patch

    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "legacy-audit-v1"})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0), layout="constrained")
    spectrum = audit["stored_eigenvalues"]
    axes[0].add_patch(Circle((0, 0), 1, fill=False, color="0.65", linestyle="--"))
    axes[0].scatter(spectrum["real"], spectrum["imag"], s=34, color="#315a9b", zorder=3)
    axes[0].axhline(0, color="0.85", linewidth=0.7)
    axes[0].axvline(0, color="0.85", linewidth=0.7)
    axes[0].set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal",
                xlabel="Real part", ylabel="Imaginary part",
                title="a  Six stored eigenvalues")
    axes[0].text(0.04, 0.04, "Period and residuals\nunknown", transform=axes[0].transAxes, fontsize=9)

    indices = list(range(19))
    norms = np.array([audit["envelopes"][str(i)]["recomputed_norms"][0] for i in indices])
    colors = plt.get_cmap("tab10").colors[:6]
    for mode in range(6):
        axes[1].plot(indices, norms[:, mode], color=colors[mode], linewidth=1.5,
                     label=f"Column {mode + 1}")
    axes[1].set(xlabel="Stored prefix-file index", ylabel="State-axis norm (row 0)",
                yscale="log", title="b  Prefix-file norms")
    axes[1].set_xticks([0, 6, 12, 18])
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=9, ncol=2, frameon=False)

    repeated = np.array([audit["envelopes"][str(i)]["time_rows_exactly_equal_to_row0"]
                         for i in indices], dtype=int).T
    axes[2].imshow(repeated, origin="lower", aspect="auto", interpolation="nearest",
                   vmin=0, vmax=1, cmap=ListedColormap(["#f5ddd7", "#2c8875"]),
                   extent=(-0.5, 18.5, -0.5, 19.5))
    axes[2].set(xlabel="Stored prefix-file index", ylabel="Stored time-row index",
                title="c  Repeated time rows")
    axes[2].set_xticks([0, 6, 12, 18])
    axes[2].set_yticks([0, 5, 10, 15, 19])
    axes[2].legend(handles=[Patch(color="#2c8875", label="Byte-identical to row 0")],
                   fontsize=9, frameon=False, loc="upper center")
    fig.suptitle("Recovered graph-analysis outputs: archive diagnostics", fontsize=13)
    # Titles and labels deliberately preserve stored axes and partial-spectrum scope.
    for suffix in ("png", "svg"):
        fig.savefig(output / ("legacy-array-audit." + suffix), dpi=200)
    plt.close(fig)
    manifest = {"kind": "legacy_array_audit_summary_figure", "audit_sha256": args.audit_sha256,
                "review_sha256": args.review_sha256,
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "raw_array_reads": 0, "new_model_calls": 0,
                "scientific_collapse_claim": False,
                "caption": "Six stored eigenvalues lack product, residual and orbit certification. "
                           "Six state-axis norms vary across nineteen archived prefix files. "
                           "All twenty rows within each file are byte-identical. One archived run; "
                           "prefixes, modes and rows are not independent replications.",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.iterdir())}}
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "raw_array_reads": 0, "panels": 3}))


if __name__ == "__main__":
    main()
