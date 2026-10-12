# Independent periodicity-array reduction v1

Frozen on 2026-10-12 before inspecting reduced real-cell outcomes. This is a
CPU-only audit of the four completed checkpoint diagnostics, not retraining.
The owner's restored-access continuation follows the current tracker and
retains the failed native-competence gate in every cell.

## Inputs and trust roots

Use exactly the four `periodicity/` directories retrieved to Tamia scratch by
native cluv 0.1.5:

- `narval_5254858` and `narval_5254871`;
- `rorqual_22984956` and `rorqual_22984963`.

The producer must be `458bc9cfce474b8e11e433e33bf132faf5032387`, with
`periodicity.py` SHA-256
`5ff03bcddc5b0c3810eec8c02b4efbbd86da370567e876eaa150dd446bcc25e8`.
Frozen configuration and checkpoint hashes come from
`configs/experiments/periodicity-analysis-v1.json`. Verify every manifest
before opening NPZ files and rehash after reduction. Do not load checkpoints.
The independent reducer is `code/analysis/reduce_periodicity.py`; the
coordinator's deployment receipt pins its committed source and byte hash.

## Independent checks

Reconstruct chronological Jacobian products in saved arithmetic and FP64,
compare singular spectra and eigenvalues with multiplicity, and check saved
direct-map, directional and derivative comparisons. Recompute complete-state
phase closure separately for native and manual trajectories, including LSTM
hidden and cell coordinates. Report exact stored differences, coordinate ULP
ratios, absolute/relative residuals and declared tolerance flags separately.

Verify literal cue/driver arrays across precision and horizons, realized
periodic/control inputs, query target identity and recorded replay metadata.
Audit every cell, arm, panel, horizon and precision. This batch contains 64
per-horizon cue trajectories per cell (256 total), and 16 tangent windows per
cell (64 total); these are repeated measurements of four seed-0 checkpoints.

Reported scalar comparisons use zero absolute tolerance and relative tolerance
`1e-9`; spectral comparisons use absolute `1e-12` and relative `1e-8`.
These audit comparisons do not replace the frozen producer's derivative and
closure tolerances. Reject unsupported Floquet/exact-orbit labels, nonfinite
arrays, wrong provenance, missing state components, changed manifests or failed
reported/raw consistency. Preserve unsuccessful reports. No tolerance is
loosened after observing a real result.

## Execution envelope

One Tamia allocation, account `aip-ioannism`, partition `cpubase_bycore_b1`,
QOS `normal`: one CPU, 2 GiB, ten minutes, zero GPUs. Maximum requested
allocation is 1/6 CPU-hour; one submission, no automatic retry or race.
Submission uses native cluv from a clean isolated controller checkout. Existing
controller/helper files and original checkpoints remain preserved. Stage
source, locked caches and all raw artifacts on scratch; no weights or dataset
copies reach the Mac.

The committed `code/ops/periodicity_reduction_job.sh` creates a unique job
snapshot and offline NumPy environment and reduces the four declared inputs.
Its full source SHA, run ID and cluster identity are explicit arguments.
Outputs are under
`$SCRATCH/representation-collapse/logs/cluv/tamia_<job_id>/reduction/`:
atomic status/provenance, `report.json`, and the Slurm log. Stop on the first
integrity, numerical, environment or scheduler failure; reconcile a missing
submission response before considering any new attempt.

## Interpretation and remaining limits

Every spectrum retains the finite-time label, including approximately closed
manual trajectories. A passing adapter tolerance can exceed cue separation;
native and manual differences must remain visible. Two histories cannot estimate
useful covariance dimension or independent training-seed uncertainty.

Manual query logits and the original probe train/test clouds are absent from
this artifact schema. Their independent checks remain explicitly unavailable;
the reducer cannot establish feature-floor robustness. All original native
competence gates failed. Successful reduction establishes numerical audit
evidence, not harmful memory collapse, competent-model equivalence, a certified
Floquet orbit, a causal mechanism or cross-modal transfer. Independent scientific
review follows execution before any manuscript-result change.
