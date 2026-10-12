#!/usr/bin/env python3
"""Reproduce manuscript figures from analytic controls and verified pilot JSON.

Run from the repository root with::

    PYTHONPATH=src .venv/bin/python code/analysis/make_paper_figures.py

No model is trained and no cluster artifacts are downloaded.  Analytic panels
are constructed examples; pilot panels contain one training seed per cell.
The manifest preserves the exact plotted values, formulas and source hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"
# Keep matplotlib's incidental cache outside tracked publication artifacts.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/paper-figures/matplotlib-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from representation_collapse.mechanism_calibration import calibration, linear_recovery_risk


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#606770"
INPUT = ROOT / "doc/auto/initial-phase-results.json"
PILOT_CONFIG = ROOT / "configs/experiments/vector-pilot-v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.titlesize": 9, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.65,
        "lines.linewidth": 1.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05,
                metadata={"Title": name.replace("_", " "),
                          "Author": "Representation Collapse project",
                          "CreationDate": None, "ModDate": None})
    # This preview is also a publication artifact, not a screen capture.
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.05,
                metadata={"Software": "make_paper_figures.py"})
    plt.close(fig)


def analytic_figure() -> tuple[dict, dict]:
    metrics, checks, arrays = calibration(seed=0)
    assert len(checks) == 36 and all(check["passed"] for check in checks.values())
    steps = np.arange(33)
    alpha = 0.8
    normalized_trace = alpha ** (2 * steps)
    normalized_pr = np.ones_like(normalized_trace)
    # Independent closed-form expectations, not a comparison to the plot itself.
    for h, result in zip(metrics["amplitude_only"]["horizons"],
                         metrics["amplitude_only"]["dimensions"]):
        np.testing.assert_allclose(result["total_variance"], 4 * alpha ** (2 * h),
                                   rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(result["participation_ratio"], 4, atol=1e-10)

    trajectories = arrays["categorical_trajectory"]
    labels = arrays["categorical_labels"]
    offsets = arrays["categorical_phase_offsets"]
    phase = offsets[np.arange(len(trajectories)) % 2]
    recovered = np.sign(trajectories - phase[:, None])
    assert np.array_equal(recovered, np.broadcast_to(labels, recovered.shape))
    radius = 0.05
    derivative_bound = 3 * radius + 1.5 * radius**2
    next_error_bound = 1.5 * radius**2 + 0.5 * radius**3
    assert derivative_bound < 1 and next_error_bound < radius

    linear_mse = metrics["affine_ordering_null"]["native_linear_mse"]
    corrected_mse = metrics["affine_ordering_null"]["offset_corrected_linear_mse"]
    assert corrected_mse < 1e-26
    # In this construction the target is the noiseless propagated cue readout,
    # so its residual is zero and the offset contribution is exactly squared.
    projected_offset = arrays["affine_offsets"] @ arrays["affine_readout"]
    np.testing.assert_allclose(linear_mse, projected_offset**2, atol=1e-12)

    noise_variance = np.logspace(-4, 0, 121)
    products = arrays["noncommuting_products"]
    risk = np.array([[linear_recovery_risk(product, n * np.eye(2))
                      for n in noise_variance] for product in products])
    for arm, product in enumerate(products):
        squared_singular_values = np.linalg.svd(product, compute_uv=False)**2
        independent_risk = np.sum(noise_variance[:, None] /
                                  (noise_variance[:, None] + squared_singular_values), axis=1)
        np.testing.assert_allclose(risk[arm], independent_risk, atol=1e-12, rtol=1e-10)
    risk_gap = risk[1] - risk[0]
    assert risk_gap.min() < 0 < risk_gap.max()

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.35))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.115, top=0.93,
                        wspace=0.35, hspace=0.59)
    ax = axes[0, 0]
    ax.plot(steps, normalized_pr, color=BLUE, label=r"PR / $d$")
    ax.plot(steps, normalized_trace, color=ORANGE, label=r"tr$(C)$ / $d$")
    ax.set(yscale="log", ylim=(3e-7, 2.5), xlim=(0, 32),
           xlabel="Retention steps", ylabel="Normalized value",
           title="a  Attenuation need not reduce dimension")
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.legend(frameon=False, loc="lower left")
    ax.text(0.98, 0.88, r"$s_t = 0.8^t z$,  $d=4$", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)

    ax = axes[0, 1]
    for label, color in [(-1, BLUE), (1, ORANGE)]:
        group = trajectories[:, labels == label]
        ax.fill_between(np.arange(len(group)), group.min(axis=1), group.max(axis=1),
                        color=color, alpha=0.3)
        ax.plot(np.arange(len(group)), group.mean(axis=1), color=color,
                label=f"Label {label:+d}")
    ax.set(xlabel="Forced update", ylabel="Complete scalar state",
           xlim=(0, 16), ylim=(-1.5, 1.5),
           title="b  Local contraction can preserve a label")
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.text(0.5, 0.51, "100% retained; cycle derivative = 0",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5)
    ax.legend(frameon=False, loc="upper right", ncol=2,
              bbox_to_anchor=(1, 1.03), handlelength=1.0, columnspacing=0.7)

    ax = axes[1, 0]
    x = np.arange(2)
    ax.bar(x, linear_mse, color=[BLUE, ORANGE], width=0.48)
    ax.scatter(x, [corrected_mse] * 2, marker="D", s=27,
               color=GREEN, zorder=3, label="Offset-corrected error")
    ax.set(xticks=x, xticklabels=["AABB", "ABAB"], ylabel="Native linear MSE",
           xlabel="Four-symbol body order", ylim=(-0.03, 0.5),
           title="c  Raw error can change with equal cue maps")
    ax.text(0.5, 0.89, "Identical centered cue cloud", transform=ax.transAxes,
            ha="center", fontsize=7.7)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.82),
              handlelength=1.0)

    ax = axes[1, 1]
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=":")
    ax.semilogx(noise_variance, risk_gap, color=BLUE)
    ax.scatter([0.04], [metrics["noncommuting_construction"]["population_linear_mse"][1]
                        - metrics["noncommuting_construction"]["population_linear_mse"][0]],
               s=20, color=ORANGE, zorder=4)
    ax.set(xlabel=r"Common terminal noise variance $\sigma^2$",
           ylabel=r"Risk(ABAB) $-$ risk(AABB)", xlim=(1e-4, 1),
           title="d  Noncommuting order effects depend on noise")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.text(0.02, 0.08, "Analytic population linear risk", transform=ax.transAxes,
            fontsize=7.5)
    save_figure(fig, "analytic_controls")

    data = {
        "scope": "constructed analytic controls, not trained-model observations",
        "calibration_seed": 0,
        "checks_passed": len(checks),
        "a": {"formula": "C_t=0.8^(2t) I_4; PR=4 at every finite t",
              "steps": steps.tolist(), "normalized_variance": normalized_trace.tolist(),
              "normalized_participation_ratio": normalized_pr.tolist(),
              "covariance": "unbiased empirical covariance of 128 exactly isotropic cue rows"},
        "b": {"formula": "s_(t+1)=1.5(s_t-b_t)-0.5(s_t-b_t)^3+b_(t+1)",
              "phase_offsets": offsets.tolist(), "labels": [-1, 1],
              "initial_perturbations_per_label": arrays["categorical_initial_perturbations"][:64].tolist(),
              "trajectories": trajectories[:, [0, 63, 64, 127]].tolist(),
              "trajectory_columns": ["negative lower", "negative upper", "positive lower", "positive upper"],
              "plotted_group_mean": {str(label): trajectories[:, labels == label].mean(axis=1).tolist()
                                     for label in [-1, 1]},
              "plotted_group_min": {str(label): trajectories[:, labels == label].min(axis=1).tolist()
                                    for label in [-1, 1]},
              "plotted_group_max": {str(label): trajectories[:, labels == label].max(axis=1).tolist()
                                    for label in [-1, 1]},
              "classification_accuracy_at_every_step": 1.0,
              "cycle_derivative": 0.0, "tube_derivative_bound": derivative_bound,
              "tube_next_error_bound": next_error_bound,
              "shading": "deterministic initial-perturbation envelope, not an uncertainty interval"},
        "c": {"sequence_order": ["AABB", "ABAB"], "native_linear_mse": linear_mse,
              "offset_corrected_linear_mse": corrected_mse,
              "matrix": arrays["affine_matrices"][0].tolist(),
              "forcing_by_arm": arrays["affine_forcing"].tolist(),
              "readout": arrays["affine_readout"].tolist(),
              "target": "w^T A^4 E(z), so native residual is exactly w^T b_arm",
              "encoder": "E(z)=z+0.1*z^3 coordinatewise"},
        "d": {"sequence_order": ["AABB", "ABAB"],
              "symbols": arrays["noncommuting_symbols"].tolist(),
              "column_state_products": products.tolist(),
              "noise_variance": noise_variance.tolist(), "risk_by_arm": risk.tolist(),
              "risk_difference_abab_minus_aabb": risk_gap.tolist(),
              "formula": "tr((I + G^T G / sigma^2)^(-1))",
              "risk": "population sum-squared optimal linear error; independent mean-zero cue covariance I_2 and terminal noise sigma^2 I_2",
              "limitation": "only symbol counts are matched in this analytic example; no universal periodicity effect"},
    }
    return data, checks


def pilot_figure(pilots: list[dict]) -> dict:
    horizons = [1, 2, 16]
    x = np.arange(3)
    categorical = [cell for cell in pilots if cell["task"] == "categorical"]
    continuous = [cell for cell in pilots if cell["task"] == "continuous"]
    assert len(categorical) == len(continuous) == 2
    for cell in pilots:
        assert cell["seed"] == 0 and cell["independent_training_replicates"] == 1
        assert cell["baseline_competence_passed"] is False
    categorical_accuracy = np.array([[100 * (1 - cell["behavior"][str(h)][arm + "_error"])
                                     for h in horizons] for cell in categorical
                                    for arm in ["periodic", "control"]])
    np.testing.assert_array_equal(categorical_accuracy,
                                  np.broadcast_to(categorical_accuracy[0], categorical_accuracy.shape))

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
    fig.subplots_adjust(left=0.085, right=0.987, bottom=0.22, top=0.85, wspace=0.36)
    ax = axes[0]
    ax.axhline(25, color=GRAY, linestyle="--", linewidth=0.9)
    ax.axhline(6.25, color=GRAY, linestyle=":", linewidth=0.9)
    ax.plot(x, categorical_accuracy[0], color=BLUE, marker="o", markersize=7,
            markerfacecolor="white", label="Periodic")
    ax.plot(x, categorical_accuracy[1], linestyle="none", marker="x", markersize=4,
            color=ORANGE, label="Matched control")
    ax.set(xticks=x, xticklabels=horizons, xlabel="Body repeats $R$", ylabel="Native accuracy (%)",
           ylim=(0, 30), xlim=(-0.2, 2.2), title="a  Categorical memory: prerequisite failed")
    ax.text(0.98, 0.85, "25% engineering gate", transform=ax.transAxes,
            ha="right", fontsize=7.2, color=GRAY)
    ax.text(0.98, 0.255, "6.25% chance", transform=ax.transAxes,
            ha="right", fontsize=7.2, color=GRAY)
    ax.text(0.05, 0.61, "GRU = LSTM\nBoth arms overlap exactly", transform=ax.transAxes,
            fontsize=7.7)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              columnspacing=1.0, handlelength=1.3)

    ax = axes[1]
    data = []
    for cell, color in zip(continuous, [BLUE, ORANGE]):
        ratios = {}
        for arm, marker, size in [("periodic", "o", 7), ("control", "x", 4)]:
            ratios[arm] = [cell["behavior"][str(h)][arm + "_error"] /
                           cell["behavior"][str(h)]["zero_predictor_mse_from_saved_targets"]
                           for h in horizons]
            np.testing.assert_allclose(ratios[arm],
                                       [cell["behavior"][str(h)]["native_to_zero_mse_ratio"][arm]
                                        for h in horizons], rtol=1e-14)
            ax.plot(x, ratios[arm], color=color, marker=marker, markersize=size,
                    markerfacecolor="white", linestyle="-" if arm == "periodic" else "none")
        data.append({"model": cell["model"], "ratios": ratios})
    thresholds = [0.25 / continuous[0]["behavior"][str(h)]["zero_predictor_mse_from_saved_targets"]
                  for h in horizons]
    ax.plot(x, thresholds, color=GRAY, linestyle="--", linewidth=0.9)
    ax.axhline(1, color=GRAY, linestyle=":", linewidth=0.9)
    ax.set(xticks=x, xticklabels=horizons, xlabel="Body repeats $R$",
           ylabel="MSE / saved-target zero baseline", ylim=(0.65, 1.07),
           xlim=(-0.2, 2.2), title="b  Continuous memory: near zero predictor")
    ax.text(0.97, 0.30, "MSE = 0.25 engineering gate", transform=ax.transAxes,
            ha="right", fontsize=7.2, color=GRAY)
    ax.text(0.97, 0.71, "Both arms overlap at this scale", transform=ax.transAxes,
            ha="right", fontsize=7.2)
    ax.legend(handles=[Line2D([], [], color=BLUE, label="GRU"),
                       Line2D([], [], color=ORANGE, label="LSTM")], frameon=False,
              ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              columnspacing=1.4, handlelength=1.3)
    save_figure(fig, "pilot_audit")
    return {"training_seed_per_cell": 0, "independent_training_replicates_per_cell": 1,
            "evaluation_pairs_per_horizon": 256,
            "horizons": horizons,
            "categorical_accuracy_percent_all_models_and_arms": categorical_accuracy.tolist(),
            "categorical_gate_percent": 25.0, "categorical_chance_percent": 6.25,
            "continuous": data, "continuous_engineering_gate_in_plotted_units": thresholds,
            "uncertainty": "none displayed; no independent-training uncertainty estimated",
            "overlap": "categorical models and arms exactly equal; continuous arm differences below display resolution"}


def probe_scale_figure() -> dict:
    """An exact scale-floor counterexample, verified through the actual v1 probe."""
    from representation_collapse.pilot import ridge_probe

    floor, alpha = 1e-8, 0.01
    amplitudes = np.logspace(-11, -6, 101)
    train_cues = np.linspace(-1.0, 1.0, 64)[:, None]
    train_cues = (train_cues - train_cues.mean()) / train_cues.std()
    test_cues = np.cos(np.linspace(0, np.pi, 257))[:, None]
    test_cues = (test_cues - test_cues.mean()) / test_cues.std()
    assert not np.array_equal(train_cues, test_cues)
    standardized_gain = amplitudes / np.maximum(amplitudes, floor)
    formula_mse = (alpha / (standardized_gain**2 + alpha))**2
    measured_mse, oracle_mse = [], []
    for amplitude in amplitudes:
        train_state, test_state = amplitude * train_cues, amplitude * test_cues
        measured, _ = ridge_probe(train_state, test_state, train_cues, test_cues,
                                   "continuous", alpha, np.random.default_rng(0))
        measured_mse.append(measured["real"]["mean_mse"])
        oracle_mse.append(float(np.mean((test_state / amplitude - test_cues)**2)))
    np.testing.assert_allclose(measured_mse, formula_mse, rtol=1e-10, atol=1e-13)
    assert max(oracle_mse) < 1e-28
    floor_free_mse = (alpha / (1 + alpha))**2
    # Verify the no-floor reference by solving the unit-standardized design.
    direct_coefficient = np.linalg.solve(train_cues.T @ train_cues +
                                         alpha * len(train_cues) * np.eye(1),
                                         train_cues.T @ train_cues)
    direct_mse = np.mean((test_cues @ direct_coefficient - test_cues)**2)
    np.testing.assert_allclose(direct_mse, floor_free_mse, rtol=1e-11)
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.94)
    ax.axhline(1, color=GRAY, linewidth=0.8, linestyle=":")
    ax.axvline(floor, color=GRAY, linewidth=0.8, linestyle=":")
    ax.semilogx(amplitudes, formula_mse, color=ORANGE, label="Ridge with feature floor")
    ax.semilogx(amplitudes, np.full_like(amplitudes, floor_free_mse), color=BLUE,
                linestyle="--", label="Ridge without floor")
    ax.axhline(0, color=GREEN, linewidth=1.0, label=r"Exact oracle $h/a$")
    ax.set(xlim=(1e-11, 1e-6), ylim=(-0.055, 1.08),
           xlabel=r"Invertible state amplitude $a$ in $h=au$", ylabel="Cue-recovery MSE")
    ax.text(1.3e-8, 0.70, r"Feature floor $10^{-8}$", fontsize=7.8)
    ax.text(1.3e-8, 0.99, "Zero-predictor MSE", fontsize=7.5, va="top", color=GRAY)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.36, 0.48),
              handlelength=1.6, fontsize=7.8)
    save_figure(fig, "probe_scale_artifact")
    return {"scope": "noiseless constructed scalar encoding; not a re-reduction of trained pilot probes",
            "formula": "r=a/max(a,f); MSE=(alpha/(r^2+alpha))^2 for population-variance-one train and test designs",
            "ridge_objective": "sum_squared_error + n*alpha*coefficient_norm_squared",
            "feature_standardization": "training mean and population standard deviation; floor f=1e-8",
            "alpha": alpha, "feature_floor": floor, "amplitudes": amplitudes.tolist(),
            "analytic_mse": formula_mse.tolist(), "actual_v1_probe_mse": measured_mse,
            "floor_free_ridge_mse": floor_free_mse, "oracle_mse": oracle_mse,
            "train_cues": train_cues[:, 0].tolist(), "test_cues": test_cues[:, 0].tolist(),
            "maximum_actual_probe_formula_difference": float(np.max(np.abs(np.array(measured_mse) - formula_mse))),
            "seed": "deterministic cues; seed 0 only for unused label-permutation control",
            "uncertainty": "none; exact finite-design calculation"}


def scientific(value: float, digits: int = 3) -> str:
    coefficient, exponent = f"{value:.{digits}e}".split("e")
    return rf"${coefficient}\!\times\!10^{{{int(exponent)}}}$"


def write_pilot_appendix(pilots: list[dict]) -> None:
    lines = [r"% Generated by code/analysis/make_paper_figures.py; edit that source for tables.",
             r"\section{Verified pilot measurements}", r"\label{app:pilot-details}",
             r"The values below are reductions of \texttt{doc/auto/initial-phase-results.json},",
             r"independently checked against saved predictions and state arrays before this manuscript.",
             r"Each cell has one training seed (0), 1,500 optimization steps and 256 paired",
             r"evaluation episodes per horizon. Checkpoints minimize the same balanced validation loss:",
             r"step 300 for both categorical cells, 400 for continuous GRU and 200 for continuous LSTM.",
             r"Each cue-conditioned cloud fixes the complete driver and terminal phase, with 512",
             r"training histories and 256 held-out probe histories. The GRU state has 128 coordinates;",
             r"the LSTM concatenates all 128 hidden and 128 cell coordinates. Clouds, coordinates and",
             r"horizons are not independent training replicates. No confidence interval is inferred",
             r"from their spread. The query readout uses the model's native hidden state.", "",
             r"\subsection{Native task behavior}",
             r"Both categorical models have identical aggregate accuracies in both arms:",
             r"5.46875\%, 6.25\% and 5.078125\% at $R=1,2,16$, respectively.",
             r"The uniform-classifier reference is 6.25\%. Their paired error contrasts and",
             r"$\DID$ are exactly zero in the saved evaluation samples. All four models fail",
             r"the original engineering criterion at $R=1$ and $R=2$: accuracy at least 25\%",
             r"or continuous MSE at most 0.25 in both arms. Failure therefore precedes the",
             r"stronger competence requirements of the proposed confirmatory design.", "",
             r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Continuous native MSE, computed on paired evaluation episodes. The zero predictor uses the exact same saved targets. Values are rounded for display; contrasts use full-precision reductions.}",
             r"\label{tab:pilot-native}", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Model & $R$ & Periodic & Matched control & Zero predictor \\", r"\midrule"]
    for cell in pilots:
        if cell["task"] != "continuous":
            continue
        for h in [1, 2, 16]:
            b = cell["behavior"][str(h)]
            lines.append(f"{cell['model'].upper()} & {h} & {b['periodic_error']:.9f} & "
                         f"{b['control_error']:.9f} & {b['zero_predictor_mse_from_saved_targets']:.9f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"The continuous periodic-minus-control differences at $R=2$ are",
              r"$1.19058\times10^{-7}$ (GRU) and $7.25557\times10^{-8}$ (LSTM);",
              r"at $R=16$ they are $1.91821\times10^{-8}$ and $1.34424\times10^{-8}$.",
              r"Their corresponding $\DID$ values are $-9.98763\times10^{-8}$ and",
              r"$-5.91133\times10^{-8}$. These seed-0 differences neither establish a",
              r"periodicity effect nor equivalence in competent models.", "",
              r"\subsection{Cue-conditioned probes and geometry}",
              r"Tables~\ref{tab:cloud-gru-categorical}--\ref{tab:cloud-lstm-continuous} retain every",
              r"driver-panel/arm cloud separately. PR is participation ratio; $V=\tr C$ uses",
              r"sample-centered complete states and unbiased covariance. Probe columns report",
              r"the mean over the eight required cue coordinates of held-out accuracy (categorical)",
              r"or MSE (continuous), with training-label permutation as a control. The probes are",
              r"ridge fits with regularization 0.01 after training-cloud feature standardization.",
              r"$q$ is centered-state RMS divided by the RMS coordinate spacing of the stored",
              r"float32 states, a post hoc resolution diagnostic. It is not a numerical error bound",
              r"or a test of float64 dynamics. All displayed clouds have nonzero variance.", "",
              r"Every $R=2$ cloud has a real-label probe advantage over its shuffled-label",
              r"control. At $R=16$ this advantage is inconsistent. The feature-scale floor",
              r"of $10^{-8}$ can alter effective regularization across clouds and horizons;",
              r"floor-hit counts and independently re-reduced probe sensitivities are unavailable.",
              r"The observations therefore support residual short-horizon decodability in these",
              r"stored states, without establishing robust retention, native task use or a",
              r"long-horizon mechanism. The two categorical-LSTM $R=16$ control clouds have",
              r"511 distinct stored rows; each of the other clouds has 512. No geometry below",
              r"is interpreted as a Floquet spectrum or a causal memory estimate."]
    for cell in pilots:
        task, model = cell["task"], cell["model"]
        metric = "accuracy (\\%)" if task == "categorical" else "MSE"
        lines += ["", r"\begin{table}[htbp]", r"\centering\footnotesize",
                  rf"\caption{{{model.upper()} {task}: all eight cue-conditioned clouds. P and C denote periodic and matched-control arms; panel IDs identify different fixed drivers. Probe metric: {metric}.}}",
                  rf"\label{{tab:cloud-{model}-{task}}}", r"\begin{tabular}{lllrrrrr}",
                  r"\toprule", r"$R$ & Panel & Arm & PR & $V$ & Real probe & Shuffled & $q$ \\",
                  r"\midrule"]
        for h in [2, 16]:
            for panel in [0, 1]:
                for arm, short in [("periodic", "P"), ("control", "C")]:
                    c = cell["clouds"][f"R{h}_panel{panel}_{arm}"]
                    key = "mean_accuracy" if task == "categorical" else "mean_mse"
                    factor = 100 if task == "categorical" else 1
                    real = c["probes"]["real"][key] * factor
                    shuffled = c["probes"]["label_control"][key] * factor
                    q = c["precision_audit"]["variation_to_ulp_rms_ratio"]
                    lines.append(f"{h} & {panel} & {short} & {c['participation_ratio']:.3f} & "
                                 f"{scientific(c['total_variance'])} & {real:.4f} & {shuffled:.4f} & {q:.3f} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    lines += ["", r"\subsection{An exact feature-scale counterexample}",
              r"\label{app:probe-scale}",
              r"The v1 probe uses $X^{\mathsf T}X+n\alpha I$ with $\alpha=0.01$ after",
              r"division by $\max(\operatorname{std}(h),f)$, where $f=10^{-8}$ and the",
              r"standard deviation uses divisor $n$. To isolate the resulting scale dependence,",
              r"let a scalar cue $u$ have zero mean and unit second moment in separate",
              r"deterministic training and test designs, and encode it as $h=au$, $a>0$.",
              r"Writing $r=a/\max(a,f)$, ridge predicts",
              r"$\widehat u=r^2u/(r^2+\alpha)$ and its test MSE is exactly",
              r"\begin{equation}",
              r"\label{eq:probe-floor}",
              r"\mathcal E_{\rm ridge}(a)=\left(\frac{\alpha}{r^2+\alpha}\right)^2.",
              r"\end{equation}",
              r"This error approaches the zero-predictor value as $a/f\to0$, although the",
              r"encoding stays invertible and the exact decoder $h/a$ retains the cue.",
              r"Without the floor, ridge error is $(\alpha/(1+\alpha))^2$ at every amplitude.",
              r"The figure script checks 101 amplitudes against the actual v1 probe, using",
              r"64 training and 257 distinct-design test points and float64 arithmetic.",
              r"This construction proves a diagnostic failure mode; it does not quantify its",
              r"contribution to the pilot probe contrasts or establish noise-robust memory.",
              r"\begin{figure}[htbp]", r"\centering",
              r"\includegraphics[width=0.82\linewidth]{figures/probe_scale_artifact.pdf}",
              r"\caption{A feature-standardization floor can manufacture probe deterioration under an invertible rescaling. The exact scalar construction has no noise or training failure. Orange is Eq.~\eqref{eq:probe-floor}, verified through the v1 ridge implementation; blue is floor-free ridge, with MSE $9.80\times10^{-5}$, and green is the exact inverse decoder, with zero mathematical error. These curves are analytic controls, not learned-model observations.}",
              r"\label{fig:probe-scale}", r"\end{figure}"]
    (ROOT / "paper/sections/D-pilot-details.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    result = json.loads(INPUT.read_text())
    config = json.loads(PILOT_CONFIG.read_text())
    assert result["engineering_phase_complete"] and not result["scientific_acceptance"]
    assert config["evaluation_pairs_per_horizon"] == 256
    assert config["competence"]["categorical_min_accuracy"] == 0.25
    assert config["competence"]["continuous_max_mse"] == 0.25
    analytic, checks = analytic_figure()
    pilot = pilot_figure(result["pilots"])
    probe_scale = probe_scale_figure()
    write_pilot_appendix(result["pilots"])
    sources = [INPUT, PILOT_CONFIG, Path(__file__).resolve(),
               ROOT / "src/representation_collapse/mechanism_calibration.py",
               ROOT / "src/representation_collapse/metrics.py",
               ROOT / "src/representation_collapse/pilot.py", ROOT / "uv.lock"]
    products = [OUT / (stem + suffix) for stem in ["analytic_controls", "pilot_audit", "probe_scale_artifact"]
                for suffix in [".pdf", ".png"]]
    products.append(ROOT / "paper/sections/D-pilot-details.tex")
    manifest = {
        "schema_version": 1,
        "command": "PYTHONPATH=src .venv/bin/python code/analysis/make_paper_figures.py",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in products},
        "git_head_at_generation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "pilot_execution_commit": result["source_commit"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "analytic_controls": analytic, "analytic_validation": checks,
        "pilot_audit": pilot, "probe_scale_artifact": probe_scale,
        "scientific_scope": "Analytic identifiability examples and a failed-competence seed-0 pilot audit; no harmful-collapse, cross-modal, causal-memory or Floquet finding.",
        "no_new_training": True,
    }
    atomic_json(OUT / "manifest.json", manifest)
    print(json.dumps({"figures": [p.name for p in products[:6]],
                      "analytic_checks_passed": len(checks),
                      "pilot_training_replicates_per_cell": 1,
                      "manifest": "paper/figures/manifest.json"}))


if __name__ == "__main__":
    main()
