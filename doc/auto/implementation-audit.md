# Implementation audit: measurement code versus the written contract

Date: 2026-10-12. Scope: `src/representation_collapse/`, `tests/`, `pyproject.toml`,
`uv.lock`, `.python-version` at commit `8fce9b064f77ce1616b03e318f83c15d821a8eae`
(branch main; clean at the time the CLIs below ran, `dirty_checkout=false`).
Read-only audit; no repository file other than this one was edited; no cluster,
cluv, `runs/` or external-service access. Scratch outputs: `/tmp/rc-audit/`
(`tests-*.log`, `calib/`, `mech/`, `verify_gaps.py`, `verify_gaps.json`). Each
finding is labelled **verified-by-run**, **read-from-source**, **inferred**, or **unresolved**.

Contracts used: `AGENTS.md`, `skills/floquet-analysis/SKILL.md`, `doc/auto/`
`research-program.md`, `periodicity-research-plan.md` §3–4, `competence-recovery-plan.md`,
`periodicity-analysis-v1.md`, `initial-phase-scientific-review.md`,
`doc/literature/novelty-review.md` §6, `skills/experiment-loop/references/workflow.md`, `configs/experiments/*.json`.

## 1. Environment and test run

**Lock and interpreter (verified-by-run).** `uv.lock` is present (252,755 bytes,
lock revision 3, `requires-python = ">=3.11, <3.15"`). `.python-version` is
`3.11.13`. The lock resolves NumPy `2.4.6` for Python `<3.12` and `2.5.4` for
`>=3.12`, so environment identity depends on the interpreter minor version;
cluster scripts pass `--python` explicitly, and artifacts record the Python and
NumPy versions. Local `uv` is 0.8.13; the `ops` group pins `uv==0.9.18` as a
package, `cluster-uv==0.1.5`, `pyyaml`. Torch is **not** in the default
dependencies: `pyproject.toml` lines 13–17 put `torch==2.6.0` in the
`experiments` dependency group.

**`uv sync --locked` (verified-by-run).** First attempt inside the tool sandbox
failed: `failed to remove file .venv/.../annotated_types-0.8.0.dist-info/INSTALLER:
Operation not permitted (os error 1)` (sandbox restriction). Rerun unsandboxed:
success in 3.3 s, but the default sync **removed** torch 2.6.0, cluster-uv,
milatools, pyyaml, uv and about 25 other packages because they belong to
non-default groups. This is the README's documented command (`README.md:18`),
and in that state the nine torch-dependent tests are skipped
(`tests/test_pilot.py:17`, `tests/test_periodicity.py:14`), so the documented
sequence cannot reproduce the "34 tests pass" statement in `doc/TRACKER.md:20`.
The full environment was restored with `uv sync --locked --all-groups` (11.6 s).

**`uv run` is broken on this Mac (verified-by-run, cause unresolved).**
After every sync, `uv run --locked python -c "import representation_collapse"`
raised `ModuleNotFoundError` although `uv pip list` shows the editable install.
`python -v` reports `Skipping hidden .pth file` for all three `.pth` files.
CPython 3.11.13 `site.addpackage` (`site.py:173–178`) ignores `.pth` files
carrying the macOS `UF_HIDDEN` flag; `ls -lO` shows the `hidden` flag on every
one of the 1,319 entries under `.venv/`. `chflags nohidden` on the `.pth` files
restored the import, but the flag was re-applied within about a minute by an
unidentified process (file mtimes unchanged). Fallback used for all runs below,
as instructed: `PYTHONPATH="$PWD/src" .venv/bin/python`. Cluster scripts are
unaffected (Linux; `prepare_environment.sh:34` smoke-imports the package).

**Test suite (verified-by-run).** `PYTHONPATH="$PWD/src" .venv/bin/python -m
unittest discover -s tests -v`: `Ran 34 tests in 4.445s`, `OK`, 5.23 s wall.
Versions: Python `3.11.13 (main, Aug 18 2025) [Clang 20.1.4]`, NumPy `2.4.6`,
torch `2.6.0` (CPU build, macOS arm64). Count by file: `test_metrics` 8,
`test_calibrate_cli` 1, `test_mechanism_calibration` 13, `test_periodicity` 5,
`test_pilot` 4, `test_vector_task` 3. In the torch-less default environment the
expected result is 25 run and 9 skipped (read-from-source); my attempt there
failed at import because of the hidden-`.pth` problem (`tests-default-group.log`).

**Calibration CLIs (verified-by-run).** Both accept only `--output` (new
directory, refuses to overwrite) and `--seed` (default 0).

- `python -m representation_collapse.calibrate --output /tmp/rc-audit/calib`:
  `{"checks": 10, "passed": true}` in 0.11 s. Files: `arrays.npz`, `config.json`,
  `manifest.json`, `metrics.json`, `status.json` (`state: verified`),
  `verification.json`. Checks: isotropic PR 4, rank-one PR 1, zero-variance
  sentinel, product order, rotation period-4 multiplier `0.8^4`, nonnormal stable
  eigenvalues with transient gain `>1`, forced-orbit periodic/nonconstant/strictly
  contracting.
- `python -m representation_collapse.mechanism_calibration --output /tmp/rc-audit/mech`:
  `{"checks": 36, "passed": true}` in 0.40 s; same six files (`arrays.npz` 9.3 MB).
  Seven families: affine ordering null, commuting null with order-dependent
  process noise, noncommuting construction, amplitude-only decay, categorical
  forced basins, continuous neutral memory, fresh noise without old memory.
- Both manifests list SHA-256 for the five other files (the manifest excludes
  itself); all ten digests were recomputed and match. `config.json` records git
  revision, dirty flag, Python/NumPy versions, host, seed and source hashes:
  `metrics.py` `41a4790b…ebf1e` (identical to the hash cited in
  `initial-phase-scientific-review.md:25`), `calibrate.py` `5de83f69…`,
  `mechanism_calibration.py` `ddb7c370…`, `uv.lock` `a00225f1…`. The 10 + 36
  counts expected by `configs/experiments/initial-phase-v1.json:15–16` are
  confirmed.

## 2. Module map

All findings in this section are read-from-source unless marked.

**`metrics.py` (88 lines; NumPy only).** Dense float64 reference measurements.
`covariance_dimension(samples, relative_tolerance=1e-10)`: casts to float64,
centres across rows, eigenvalues as squared singular values of the centred
matrix over `N-1`; returns participation ratio, covariance-eigenvalue entropy
dimension (`exp(-Σ p log p)`), numerical rank (eigenvalue `> rtol·λ_max`), total
variance, unclipped eigenvalue list, observation/feature counts, and
`normalized_dimension_defined = total > 0`; zero variance yields `0.0`
sentinels (lines 33–36, 47). Raises on non-2D input, `<2` rows, non-finite
input, centring or spectrum overflow. Not provided: D90/D95, stable rank, a
sample-rank ceiling (the pilot computes the ceiling itself). `ordered_product`:
`J[T-1]…J[0]` by left-multiplication, float64, finiteness guard (55–67).
`period_spectrum`: eigenvalues, moduli, singular values, spectral radius,
largest gain of the dense product; the docstring says the caller must establish
periodicity, but the output keys are named `multipliers_*` unconditionally
(78–86), which is a labelling hazard if reused on nonperiodic windows.

**`calibrate.py` (116 lines).** Ten deterministic fixtures from
`numpy.random.default_rng(seed)`; writes `config.json` (git revision, dirty
flag, source hashes, versions, host, `SLURM_JOB_ID`), `status.json`
(`running` → `verified`/`failed`; `slurm_job_id` and `CLUV_CLUSTER_NAME`
injected), `metrics.json`, `verification.json`, `arrays.npz`, `manifest.json`.
Atomic writes by temp-file rename without fsync. `config.json` lacks `argv`.
`mkdir(exist_ok=False)` refuses overwrite (line 80). Exit code 1 on any failed
check. Entry point `collapse-calibrate` (`pyproject.toml:20`).

**`mechanism_calibration.py` (399 lines).** Analytic controls; every check is
tolerance-qualified (`ATOL=1e-11`, `RTOL=1e-10`; prespecified finite-sample
tolerances 0.08/0.12 for the noise family). Public helpers: `isotropic_cues`
(orthogonal design with exactly identity empirical covariance), `propagate_affine`
(row-cue, column-state chronology), `conditioned_dimension` (rejects pooled-body
covariance; labels are a caller contract), `propagated_noise_covariance`,
`linear_recovery_risk` (population linear MSE `tr((I+HᵀΣ⁻¹H)⁻¹)`). CLI records
`argv`, tolerances, sampling unit, scope and `product_label =
finite_time_cue_propagator`; exception path writes failed status, verification
and manifest, then re-raises (lines 385–392). Float64 throughout.

**`vector_task.py` (115 lines).** Five 12-symbol motifs with matched counts and
cyclic bigrams; `proper_period` (exact tiling test), `body_from_ids(ids, q)`
(dwell `q` repeats each symbol), `paired_body_ids(rng, repeats)` (periodic arm =
one motif repeated; control = varying motifs with the same final motif and no
proper whole-string period; **raises for `repeats < 1`**, lines 36–37),
`sample_memory` (categorical int64 0–15 or continuous float32 U[-1,1]),
`render` (38 channels: type[5], coordinate[16], categorical value[16], real
value[1]; **always 16 WRITE rows**, requires `memory.shape == (n,16)`, lines
65 and 77; accepts empty bodies, giving a 17-step episode), `paired_batch`
(exactly paired arms; targets duplicated). Episodes are float32.

**`pilot.py` (373 lines; torch).** `VectorMemoryModel`: `Linear(38→64)+Tanh`
encoder, one-layer `nn.GRU`/`nn.LSTM` (`batch_first`), linear head (16 logits
or 1 value). Complete state: GRU `h[0]` (128); LSTM `cat(h[0], c[0])` (256,
order h then c, line 41–45). `pre_query_state` runs `inputs[:, :-1]`, so the
measured state is immediately before the terminal QUERY; `read_query` applies
the query update then the head; `forward` is the uninterrupted native path.
`ridge_probe`: train-only mean/std standardisation with a `1e-8` scale floor
(line 128), fixed ridge `alpha·N`, closed-form solve, real labels and a seeded
label permutation control, per-coordinate accuracy/MSE; saves means, scales,
permutation, coefficients, predictions. `validate_config` pins `dwell=1`,
`train_repeats=[1,2,4,8,16]`, evaluation `{1,2,16}⊆·⊆{1,2,4,8,16}`,
secondary `⊆{32,64}`, AdamW, 16/8/16 task constants (151–172). `provenance`
records revision/dirty flag, SHA-256 of `pilot.py`, `vector_task.py`,
`metrics.py`, `uv.lock`, `pyproject.toml` and the config file, versions,
accelerator properties, SLURM allocation variables, `CUBLAS_WORKSPACE_CONFIG`,
and the constants `deterministic_algorithms: True` and `precision: "float32
model/state; float64 covariance/probes"`. `run`: seeds `random`, NumPy, torch
(+CUDA) with `seed`; sampling streams `seed`, `+10000` validation, `+20000`
evaluation, `+30000` clouds, `+40000+100R+panel` probe controls; sets
`use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False` (230–232) but **does not set or record TF32 flags**;
trains with full-episode BPTT, gradient clipping, checkpoint on strict
validation improvement; evaluates paired errors at R1/R2/R16 (+ optional
secondary), aggregate competence gate per arm (302–307), cue clouds at R2/R16
for two driver panels with fixed nuisance and fixed body (308–330), probes;
writes `config.json` (effective), `provenance.json`, `status.json`,
`history.json`, `checkpoint.pt`, `paired_errors.npz`, `cue_clouds.npz`,
`metrics.json`, `manifest.json` (excludes itself and `.tmp`). Atomic JSON/NPZ
with fsync; checkpoint via `torch.save` to temp then rename. CLI: `--config
--output --model --task --seed [--device auto|cpu|cuda] [--max-steps]`.
No `verification.json`, no `argv`.

**`periodicity.py` (417 lines; torch).** Checkpoint-only diagnostic.
`unpack_state`/`recurrent_step`: manual PyTorch gate equations for one layer
(`weight_ih_l0`), full-state autodiff with frozen input; LSTM returns
`cat(h, c)`. `dense_jacobian` = `torch.autograd.functional.jacobian(vectorize=True)`;
`step_jacobians` along a replayed window; `tangent_window` multiplies in the
replay dtype (`J[T-1]…J[0]`), also computes the direct 12-step map Jacobian and
a float64 re-accumulation of the saved step Jacobians from which `eigvals` and
`svd` are taken (lines 72–92). `distance`: global L2 absolute **and** relative
(`max(‖a‖,‖b‖,1e-12)`) **and** coordinatewise bounds (95–108). `replay`: zero
state, manual step versus native `model.core` per step, versus the fused
full-sequence native state and hidden trajectory, and manual versus native
query output, all with `torch.allclose`; returns every state from the initial
through the pre-query state and (optionally) the native records; the
post-QUERY state is computed (line 132) but only the head output is returned.
`validate_config` pins horizons `[1,2,16,32]`, derivative horizons `[2,16]`,
`dwell=1`, both precisions and finite positive tolerances. `fixed_panels`:
distinct final motifs, control bodies with no proper period at R2/16/32. `run`:
refuses unless HEAD equals `--source-commit` and the checkout is clean (unless
`verify_source=False` in tests), checks `--checkpoint-sha256`, the frozen cell
registry, the training manifest/provenance/status/config, the saved config's
model/task/seed, the dense-state budget (`hidden ≤ 128`, bottleneck `≤ 64`),
and the full-budget selected step; disables TF32 and sets `highest` matmul
precision (239–244); FP64 assay = `deepcopy(base).to(dtype=float64)` with
inputs cast at line 302 (weights and inputs, from the zero state); per cell ×
precision × panel × horizon × arm × cue: inputs, manual and native trajectories,
adjacent-cycle closure residuals, paired-cue separations (joint and LSTM h/c),
and at derivative horizons the tangent window, previous-window Jacobians,
state/Jacobian closure flags, product-versus-direct agreement, and in FP64 a
one-direction JVP and central finite differences at `eps=[1e-4,1e-5,1e-6]`
(abs `1e-8`, rel `1e-4`). Every tangent packet carries
`interpretation: finite_time_one_motif_propagator`, `exact_orbit_verified:
False`, `floquet_acceptance: False`. Any failed numerical check quarantines the
run (`state: failed`, raw arrays retained, line 390). Internal runtime cap
`min(840 s, walltime−60 s)`. Artifacts: `config.json`, `provenance.json`
(+checkpoint hash, training origin, module hash, TF32 flags), `inputs.npz`,
`trajectories.npz`, `tangents.npz`, `metrics.json`, `status.json`,
`manifest.json`. No `verification.json`, no `argv`.

**`__init__.py`**: docstring only. **`code/ops/*`** (listed, not audited):
`cluv_controller.py` project-local SSH control sockets for cluv;
`monitor_initial_phase.py` / `monitor_periodicity.py` read-only job polling;
`preflight.py` cluv config validation without SSH; `qwen_worker.py` bounded
proposal-only Qwen leaf; `reconcile_initial_phase.py` offline+Slurm roster
acceptance; `stage_periodicity.py` immutable source staging;
`submit_initial_phase.py` / `submit_periodicity.py` single explicit submissions;
`verify_cluv_access.py` read-only access checks; `verify_initial_phase.py`
deterministic raw-array verifier (`atol 1e-7/rtol 1e-5`, variance `atol 0`,
eigenvalue `atol 1e-12·trace`, ULP-ratio audit); `calibration_job.sh`,
`phase1_job.sh`, `periodicity_job.sh` pinned-source Slurm wrappers;
`prepare_environment.sh` locked offline environment; `preflight_remote.sh`
read-only host probe. **`code/literature/build_library.py`**: bibliography and
public-PDF retrieval via `uv run --no-project`.

## 3. Invariant coverage

| Invariant | Enforced in code | Test | Gap or risk |
|---|---|---|---|
| LSTM state includes h and c | `pilot.VectorMemoryModel.pack_state` 41–45; `periodicity.unpack_state`, `recurrent_step` 27–49 | `test_pilot::test_complete_lstm_state_and_query_timing`; `test_periodicity::test_full_lstm_derivative_jvp_and_finite_difference` (h depends on c) | One layer hard-coded (`h[0]`, `weight_ih_l0`); no multi-layer or buffer accounting |
| Graph state includes nodes and ordering | none | none | No graph adapter exists (read-from-source: no `graph`/node code in `src`) |
| Chronological Jacobian ordering | `metrics.ordered_product` 63–65; `periodicity.tangent_window` 76–78, 83–85 | `test_metrics::test_noncommuting_product_and_perturbation`; `test_periodicity::test_noncommuting_chronological_product`; calibrate `product_order` | Covered |
| Finite-time labels on arbitrary products | `periodicity` packets 334–338; `mechanism_calibration` `product_label` | `test_periodicity::test_checkpoint_only_artifacts…` asserts `scientific_acceptance=False` only | `metrics.period_spectrum` emits `multipliers_*` keys regardless of periodicity (78–86) |
| Floquet only after input periodicity and state closure | `periodicity` `driver_closed` 297, `numerical_closure_compatible` 333 requires periodic arm + driver + state + Jacobian closure; `floquet_acceptance` always `False` | no test exercises a closing trajectory | Flag semantics untested on a fixture that actually closes; FP32 absolute tolerance below rounding floor (§5 R1) |
| Seeds, sampling units, phase, centring, normalisation, tolerances, controls reported | `pilot.run` effective config 242–245, metrics 332–343; `periodicity.run` effective 250–259; `metrics.covariance_dimension` fields | `test_pilot::test_tiny_run…` checks artifacts exist | Pilot reports no uncertainty method (single seed, declared); account/partition not in artifacts |
| Synthetic calibration labelled as infrastructure | `kind: synthetic_*`, `scope` strings in both CLIs | `test_mechanism_calibration::test_declared_checks…` | Covered |
| Large Jacobians matrix-free, dense-checked on small cases | dense only: `periodicity.dense_jacobian` 58–60, budget guard 232 | `test_periodicity` dense versus direct map, JVP, FD at state dims 3–8 | No matrix-free path; dense reference exists for small cases only |
| Separate covariance, tangent, behaviour, causal memory | `pilot` behaviour/geometry/probes separate; `periodicity` separation versus tangent | `test_pilot::test_tiny_run…` | No causal intervention code; probes are decodability only |
| Clouds conditioned on fixed driver history | `pilot.state_cloud` 118–120 (same body for every row), nuisance fixed 314–315; `mechanism_calibration.conditioned_dimension` | `test_pilot::test_tiny_run…` (nuisance broadcast); `test_mechanism_calibration::test_wrong_conditioning…` | Covered |
| Train-fit probes with shuffled-label controls | `pilot.ridge_probe` 123–148 | `test_pilot::test_probe_recovers_known_linear_memory_and_label_control` | Scale floor `1e-8` changes effective regularisation at the observed state amplitudes (§4.16, §5 R2) |
| Refuse to overwrite outputs | `mkdir(exist_ok=False)`: calibrate 80, mechanism 351, pilot 247, periodicity 194 | all four CLIs/`run` tested (`FileExistsError`, manifest unchanged) | Covered |
| FD/JVP checks against dense Jacobians | `periodicity.run` 341–359 (FP64 only, one direction) | `test_periodicity::test_full_lstm_derivative_jvp_and_finite_difference` | FP32 windows have no FD check (declared); one direction per window |
| Closure residual reporting with finite-time labels | `periodicity.distance`; residual arrays 311–316; packets | `test_periodicity::test_checkpoint_only_artifacts…` (shapes `(0,)`, `(15,)`) | Only scalar per-cycle residuals saved; raw states allow recomputation |
| Zero-variance sentinel | `metrics.covariance_dimension` 33–36, 47 | `test_metrics::test_zero_variance_is_marked_degenerate`; calibrate `zero_variance_convention` | No resolution-limited flag for ULP-scale nonzero variance |
| Scheduler/job identity in artifacts | `atomic_json`/`_write_json` inject `SLURM_JOB_ID`, `CLUV_CLUSTER_NAME` into `status.json`; `provenance.allocation`; periodicity `--cluster` | none (environment-dependent) | Cluv invocation, account, partition, elapsed allocation not recorded in artifacts |
| Covariance contract (PR, entropy, D90/D95, absolute variance, rank ceilings) | PR/entropy/variance/eigenvalues in `metrics`; ceiling `min(N−1, features)` in `pilot` 325 | `test_metrics::test_known_spectrum` | D90/D95 and stable rank absent; P−1 time-only ceiling never computed because no time-only covariance is measured |

## 4. Coverage gaps for the Gate B mechanism study

Gate B (`novelty-review.md:162`) needs the vector renderer, GRU and full-state
LSTM, both target types, `q=1`, within-support horizons, frozen matching rules,
competence criteria, effect-size threshold and probe/intervention protocol, and
three paired seeds. The competence-recovery ladder (`competence-recovery-plan.md`
§"Diagnostic ladder") needs R0, a single-WRITE/QUERY constructor, a streaming
oracle, coordinate-wise gates and three-phase state capture.

1. **R0 unsupported (verified-by-run).** `paired_body_ids(rng, 0)` raises
   `repeats must be positive` (`vector_task.py:36–37`); `validate_config`
   rejects `train_repeats` containing 0 and `evaluation_repeats` containing 0
   (`pilot.py:158–161`); `periodicity.validate_config` pins horizons
   (`periodicity.py:153`). `render` with empty bodies does work: shape
   `(4, 17, 38)`.
2. **No single-WRITE immediate-QUERY constructor (verified-by-run).** `render`
   requires `memory.shape == (n, 16)` and writes 16 coordinates
   (`vector_task.py:65, 77`); a `(n,1)` memory raises `inconsistent episode shapes`.
3. **No streaming register oracle in the repository (verified-by-run).** `rg -i
   oracle src tests code` returns nothing. `competence-recovery-plan.md:108–112`
   reports an oracle check on 128 zero-body episodes; that code is not committed.
4. **No intervention or sham code (verified-by-run).** No swap, transplant,
   sham, subspace-perturbation or equal-norm random control in `src`.
5. **No statistics module (verified-by-run).** No paired seed-level effects,
   bootstrap, confidence interval or Holm code; `pilot` metrics declare
   `statistical_unit: one trained seed`.
6. **Three-phase capture incomplete (verified-by-run + read-from-source).**
   Pilot clouds are pre-query only (`pre_query_state`, `pilot.py:47–49`;
   `measurement_phase: before terminal query`, line 340). `periodicity.replay`
   returns 41 states for a 24-step body (initial, 16 WRITE, 24 body), so
   post-WRITE (index 16, used at line 370) and pre-QUERY (index −1) exist; the
   post-QUERY state is computed at line 132 but not returned or saved.
7. **FP64 fixed-weight replay exists and is labelled (read-from-source).**
   `periodicity.py:283–284` casts a deep copy of the weights, line 302 casts
   inputs, zero state inherits the dtype; provenance `precision` string and
   `float64_` key prefixes label it as a cast-copy assay, not native behaviour.
   Derivatives are taken on the manual-equation adapter, not on native kernels.
8. **Dwell `q>1` untested at model level (read-from-source).** `body_from_ids`
   and `paired_batch` accept `q`, and string-level tests cover `q∈{1,2}`
   (`test_vector_task.py:21–26`), but `pilot.run` calls `paired_batch` without
   `q` (line 263) and hard-codes `q=1` in `state_cloud` calls (320–321);
   both validators reject `dwell≠1`.
9. **CUDA determinism unverified (verified-by-run on CPU only).** Two identical
   tiny CPU runs produce bitwise-identical checkpoints (SHA-256 of concatenated
   weights equal) and equal `delta2`/`delta16`; metrics differ only in
   `runtime_seconds`. No test asserts this, and no GPU repeat run exists.
   `provenance.deterministic_algorithms: True` (`pilot.py:205`) is a constant,
   not a measurement. TF32 flags are neither set nor recorded in `pilot.py`;
   PyTorch's default `cudnn.allow_tf32=True` (inferred) means the A100/H100
   pilots may have used TF32 inside cuDNN RNN kernels.
10. **Single seed (read-from-source).** `vector-pilot-v1.json` lists seeds
    `[0,1,2]`; `run` handles one seed; no reducer across seeds.
11. **Coordinate-wise competence gates absent (read-from-source).** Gates are
    aggregate per arm with config thresholds 0.25 (`pilot.py:302–307`); the plan
    requires ≥0.95 accuracy or normalised MSE ≤0.05 **per required coordinate**
    (`competence-recovery-plan.md:168–172`). Raw per-episode errors, queries and
    targets are saved (`pilot.py:296–301`), so stratification is possible post hoc.
12. **Graph complete-state adapter absent (verified-by-run).**
13. **Dense Jacobians only (read-from-source).** `dense_jacobian` is `O(N)`
    backward passes with `vectorize=True`; state ≤256 enforced; JVP is used only
    for the FD direction check (line 345). No QR/Lyapunov or matrix-free product.
14. **Eigen/SVD choices (read-from-source).** `np.linalg.eigvals`/`svd` on a
    dense 12-step product re-accumulated in float64 (`periodicity.py:83–88`);
    adequate for 12 steps at ≤256 dims; no stabilisation for longer windows.
15. **Float32 ULP (read-from-source).** States are float32; `covariance_dimension`
    casts to float64 (`metrics.py:19`); `normalized_dimension_defined` is `True`
    for any nonzero variance; `numerical_rank` uses `rtol=1e-10` relative to
    `λ_max`, which can count float32 roundoff modes as rank. ULP ratios are
    computed only by the ops verifier (`verify_initial_phase.py:261–266`).
16. **Probe feature floor (verified-by-run).** With noise-free exactly linear
    memory in 128 features, `ridge_probe` gives MSE 0 at per-feature std 1 and
    `1e-6`; at std `≈1.5e-8`, 11/128 features hit the floor, MSE still 0; at
    `1e-9`, 128/128 at the floor and MSE `2.7e-4`; at `1e-10`, MSE `0.175` versus
    control `0.33`. The reported R16 cloud variances (`~1e-16` over 128–256
    coordinates, `initial-phase-scientific-review.md:113–120`) imply per-feature
    std `≈1e-9`, so the floor was active for essentially all features at R16 and
    for many at R2 (inferred), making the ridge penalty ~100× stronger relative
    to feature variance at R16 than at R2. `probe_feature_scale` arrays are saved
    in `cue_clouds.npz`, so floor-hit counts can be recovered from the artifacts.

## 5. Numerical risk register

| Risk | Trigger | Symptom | Current mitigation | Proposed acceptance test |
|---|---|---|---|---|
| R1 FP32 closure `atol=1e-7` below rounding floor (inferred, arithmetic) | 13×128 state block at scale 1 has half-ULP L2 floor `2.4e-6`; at scale 0.1, `1.5e-7` (`verify_gaps.json` D) | FP32 closure passes only for bitwise-repeated states; near-closure reads as failure | Relative bound, raw residuals saved, FP64 assay | Float32 forced contracting map that converges to an exact float32 cycle must pass; same map with injected 1-ULP jitter must fail with residual ≈ floor |
| R2 Probe scale floor | Per-feature std `<1e-8` | Horizon-dependent effective ridge; weak signals shrunk | Floors saved | Scale-equivariance: probe results invariant under uniform state rescaling `1e-10…1e2` for noise-free linear memory; floor-hit count in metrics |
| R3 ULP-scale nonzero variance | Variance `~1e-16` on float32 states | `normalized_dimension_defined=True` and PR reported for roundoff-dominated clouds | Review notes; verifier ULP ratio | `resolution_limited` flag when centred RMS `<` k·ULP RMS; test with float32-quantised constant cloud plus 1-ULP noise |
| R4 Numerical rank at `rtol=1e-10` | float32-origin matrices | Roundoff modes counted as rank | None | Known-rank fixture cast through float32: rank correct only with dtype-aware tolerance |
| R5 Long products | Windows `>12` steps or expanding Jacobians | Overflow/underflow; `ordered_product` raises | Finiteness guard | QR-stabilised log-singular-values agree with dense log-SVD on 3–8-dim fixtures over 384 steps |
| R6 Nonnormal eigenvalue sensitivity | Defective/near-defective products | Eigenvalues unstable; spectral radius misread | Singular values reported alongside | Jordan-block fixture: eigenvalue perturbation `O(ε^{1/k})` while singular values stable |
| R7 Manual adapter versus fused kernel | GPU cuDNN kernels, float32 | Adapter check fails or passes with `atol 1e-6` exceeding cue separation | Native and manual trajectories both saved | Assert both arrays present and report separation on both (already partly done at 361–369) |
| R8 TF32 in training (inferred) | Default `cudnn.allow_tf32=True` on Ampere/Hopper | Pilot arithmetic not pure float32; unrecorded | None in `pilot.py` | Record both TF32 flags and `float32_matmul_precision` in provenance; CUDA smoke test asserts recorded values |
| R9 Lock resolves NumPy by interpreter minor | Python ≥3.12 | Different NumPy wheel than cluster | `--python` in scripts; versions recorded | Environment check asserts `python==3.11.*` and `numpy==2.4.6` before running protocols |
| R10 Local `uv run` unusable (hidden `.pth`) | macOS `UF_HIDDEN` on `.venv` | Import failures; silent test skips | None | Documented fallback and an import smoke step in the test command |

## 6. Proposed target layout for Gate B (names and responsibilities only)

- `representation_collapse/tasks/`: `vector_episodes.py` (motifs, bodies incl.
  R0 and dwell, paired arms, renderer); `diagnostics.py` (single-WRITE/QUERY,
  zero-body, corruption controls); `oracle.py` (streaming register oracle).
- `models/`: `adapter.py` (uniform `init_state / step / pack / unpack / readout /
  metadata` contract); `torch_rnn.py` (one-layer GRU/LSTM native + manual
  equations); `graph/` (GC-LSTM full-state adapter, later).
- `measure/`: `geometry.py` (covariance contract incl. D90/D95, stable rank,
  resolution flags, h/c separate and joint); `tangent.py` (dense, JVP/VJP,
  QR-stabilised windows, FD sweeps); `probes.py` (ridge + controls + floor
  accounting); `closure.py` (driver periodicity, cycle residuals).
- `intervene/`: `swaps.py`, `transplants.py`, `subspace.py`, `shams.py`.
- `stats/`: `paired.py` (seed-level paired effects, bootstrap CIs), `multiplicity.py`
  (Holm), `reducers.py` (missing-seed retention).
- `artifacts/`: `atomic.py`, `provenance.py`, `manifest.py`, `status.py`,
  `schemas.py`.
- `protocols/`: thin frozen entry points `calibrate_v1`, `mechanism_v1`,
  `pilot_v1`, `periodicity_v1`.

Migration note: frozen configurations (`vector-pilot-v1.json`,
`periodicity-analysis-v1.json`) run against pinned commits (`9004a0c…`,
`458bc9c…`), and `periodicity.run` refuses a mismatched or dirty source, so the
existing modules must stay importable and behaviourally unchanged. Add the new
packages alongside; let `pilot.py`, `periodicity.py`, `vector_task.py`,
`metrics.py` remain as-is or become re-export shims only when a byte-identical
behaviour test guards them. Version new validators separately (`schema_version
2`), keep `tests/` for v1 untouched and add new test modules.

## 7. Prioritised backlog

| # | Item | Acceptance test | Before new compute? |
|---|---|---|---|
| 1 | Document and script the test environment (`uv sync --locked --group experiments`; import smoke; hidden-`.pth` note) | Fresh clone: 34 tests run, 0 skipped | Yes |
| 2 | Record and pin TF32/matmul precision in `pilot` provenance; decide pure-float32 training for Gate B | Provenance keys present; CUDA smoke asserts values | Yes |
| 3 | Diagnostic constructors: R0 bodies, single-WRITE/QUERY; versioned validator | Shapes `(n,17,38)`, `(n,2,38)`; no-leak tests | Yes |
| 4 | Committed streaming oracle with corruption controls | Exact categorical recovery, continuous error `<1e-12`; shuffled memory/query break agreement | Yes |
| 5 | Coordinate-wise competence gates, normalised MSE `MSE/(1/3)`, 128 episodes per coordinate | Synthetic predictions: gate fails if any coordinate below threshold | Yes |
| 6 | Three-phase batched state capture (post-WRITE, pre-QUERY, post-QUERY) with forked-versus-uninterrupted check | Batched equality within declared tolerance for GRU/LSTM | Yes |
| 7 | Probe floor accounting and scale-equivariance; report floor hits and effective gains | §5 R2 test | Yes (also reinterprets v1 probes) |
| 8 | Deterministic repeat-run test (CPU in `tests/`; CUDA pair in the job script) | Identical weight hashes | CUDA: yes |
| 9 | Geometry contract completion (D90/D95, stable rank, resolution flag, h/c separate) | Known-spectrum tests | Before interpretation |
| 10 | Multi-seed reducer and paired statistics (bootstrap, Holm, missing-seed retention) | Synthetic seed table with known effect | Before confirmation |
| 11 | Interventions and shams | Linear fixture where only the task subspace matters | Before mechanism claims |
| 12 | QR-stabilised/matrix-free tangent path cross-checked against dense | §5 R5 test | Before states `>256` |
| 13 | Artifact contract: `argv`, account/partition, `verification.json` for pilot and periodicity | Schema test on tiny runs | Before next release of results |
| 14 | Graph complete-state adapter | Node-ordering and `2NH` state tests | Before Gate D |

## 8. Reproducibility-appendix paragraph

The analysis code is a small Python package (`representation_collapse`,
~1,500 source lines) under a `uv` lockfile (Python 3.11.13, NumPy 2.4.6,
torch 2.6.0 in an optional `experiments` group). It provides a dense float64
reference for centred-covariance dimension (participation ratio, covariance
entropy dimension, numerical rank, absolute variance, explicit zero-variance
sentinel), chronologically ordered Jacobian products with eigenvalues and
singular values, two deterministic synthetic calibration CLIs (10 and 36
tolerance-qualified checks covering known-rank clouds, noncommuting order,
rotating/nonnormal/contracting maps, affine and commuting ordering nulls,
categorical basins, continuous neutral memory and fresh-noise geometry), a
paired periodic-versus-matched-control vector memory task, a torch GRU/LSTM
pilot that records complete pre-query state (hidden and cell for LSTM),
held-out ridge probes with shuffled-label controls, and a checkpoint-only
periodicity diagnostic with manual-equation full-state Jacobians, dense
12-step finite-time propagators, direct-map/JVP/finite-difference checks,
adjacent-cycle closure residuals and a labelled fixed-weight float64 replay.
Every CLI refuses to overwrite an output directory, writes JSON and NPZ
artifacts atomically, records source hashes, git revision, versions, seeds and
Slurm identity, and emits a SHA-256 manifest; all 34 unit tests pass locally
(4.4 s) and both calibrations reproduce their declared check counts.
Verification status: synthetic calibrations are infrastructure evidence only;
no Floquet multiplier, causal intervention, multi-seed statistic, graph adapter,
matrix-free tangent method or coordinate-wise competence gate is implemented,
and GPU-level determinism and TF32 usage of the executed pilots are unverified.

## 9. Statements that contradict or exceed the written protocols

1. `README.md:18–19` documents `uv sync --locked` then `uv run python -m unittest
   …`; that environment lacks torch and skips 9 tests, so it cannot reproduce
   `doc/TRACKER.md:20` ("All 34 local tests … pass"). Verified-by-run.
2. `competence-recovery-plan.md:108–112` reports an oracle verification; no
   oracle code exists in `src`, `tests` or `code`, contrary to `AGENTS.md`
   "Keep source … in git". Verified-by-run.
3. `workflow.md:73–76, 87–88` require `config.json` with exact `argv` and a
   `verification.json` per run; `pilot.py` and `periodicity.py` emit neither
   (`argv` only in `mechanism_calibration.py:361`). `workflow.md:90–92` allows
   initial shortfalls but asks that missing fields be recorded; they are not.
   Read-from-source.
4. `research-program.md:136–139` requires D90/D95 and stable rank; `metrics.py`
   returns neither (eigenvalues are retained). `research-program.md:155`
   requires hidden/cell recorded separately and jointly; pilot clouds are joint
   only (`pilot.py:41–45, 320–324`). Read-from-source.
5. `periodicity-analysis-v1.md:122–126` says "Save the raw phase-wise
   differences"; `periodicity.py:311–316` saves scalar per-cycle residuals only
   (states are saved, so differences are recomputable). Read-from-source.
6. `pilot.py:205` writes `deterministic_algorithms: True` as a constant and
   omits TF32 flags, while `periodicity-analysis-v1.md:102–103` records TF32
   control only for the diagnostic. Read-from-source / inferred.
7. `periodicity-research-plan.md:444–446` asks for ≥1,024 paired episodes and
   eight driver pairs × 512 cues; `vector-pilot-v1.json` uses 256 pairs and two
   driver pairs. Declared engineering scope, but the gap should be stated when
   v1 geometry is cited. Read-from-source.

## 10. Unresolved

- Which process re-applies `UF_HIDDEN` to `.venv/` on this Mac, and whether the
  Oct 11 local test runs used `uv run` or a `PYTHONPATH` fallback.
- Whether the four executed GPU pilots ran with cuDNN TF32 enabled (needs the
  cluster torch defaults or a provenance patch; not checkable locally).
- Floor-hit counts and effective probe gains in the real R2/R16 cue clouds
  (requires the remote `cue_clouds.npz`; `probe_feature_scale` is saved).
- Whether reported numerical ranks (114–172) include float32 roundoff modes.
- CUDA repeat-run determinism; independent reduction of the periodicity raw
  arrays (pending per `doc/TRACKER.md`).
