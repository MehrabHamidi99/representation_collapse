# 2026-10-11 — project bootstrap

## Scope and preserved evidence

Set up recurrent representation dimension-collapse research around exact discrete
Floquet and finite-time analysis, starting from temporal graphs. Adapted the
Skill Inference project's canonical skills, month/day notebook, tracker,
scratch-first storage and evidence DAG. Its spending approvals, model-provider
configuration and GPU reservations were not imported.

Initial local tree contained manuscript/figure snapshots under `floquet_analysis/`,
`ACM___TGRAB/`, and `ACM___TGRAB_KDD_2026/`, with no git metadata or experiment
implementation. These sources were preserved in place. The appendix identifies
public T-GRAB code/data; source revision, checkpoint provenance, raw trajectories
and dimension-analysis implementation still require recovery. Scientific
milestones and controls are in `doc/auto/`.

## Implementation and delegated work

The coordinator integrated root guidance, package/lockfile, local NumPy reference
measurements and calibration, artifact checks, discovery links and the notebook.
Three bounded agents owned disjoint scientific documentation, cluv compute,
and research workflow scopes. A separate forward review checked numerical
semantics and compute readiness; findings and corrections are appended below.

Canonical skills: `floquet-analysis`, `experiment-loop`, `project-ops`, and
`model-routing`. Discovery symlinks under `.agents/skills/` point back to these
files. `configs/research-plan.json` defines seven acyclic dependency nodes;
remote submissions are disabled and resource caps unset until a concrete
approval. This is an active-session workflow, not an unattended controller.

## Local evidence

Environment: CPython 3.11.13, NumPy 2.4.6, cluster-uv 0.1.5 in the `ops` group;
full dependency provenance is `uv.lock`. Commands:

```bash
uv sync --locked --group ops
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m representation_collapse.calibrate --output runs/calibration/setup-2026-10-11-v2
uv run --locked --group ops python code/ops/preflight.py
```

The calibration passed ten checks. Isotropic four-feature covariance has PR 4;
rank-one covariance has PR 1; a contracted rotation has four-step multiplier
modulus `0.8^4 = 0.4096`; a nonnormal matrix has spectral radius 0.8 with
singular gain greater than one. A stable periodically forced response closes
its four-step orbit while its multiplier modulus is `0.5^4 = 0.0625`.
These are synthetic numerical/analytic controls, not findings about trained
graph models. Zero-variance normalized dimensions are undefined scientifically;
the API encodes zero sentinels with `normalized_dimension_defined=false`.

Artifacts include source/lock hashes, configuration, raw arrays, metrics,
verification, atomic status and SHA-256 manifest. New attempts refuse to
overwrite existing outputs. The final verification count is appended below.

## Compute preparation and authorization

Read-only probes verified Mila and Tamia via Mila; direct laptop Tamia access
timed out. Tamia account/partition for the proposed CPU smoke:
`aip-ioannism`, QOS `normal`, `cpubase_bycore_b1`. Request: one node, one task,
one CPU, 2 GB, ten minutes, zero GPUs, maximum one submission, no retry.
Additional Alliance profiles are examples with unverified project entitlements.

cluv 0.1.5 supports `submit first` and multiple `sbatch_args` configurations;
both can create several candidates. Race scope must match the approved actual
connected set. No race is part of the CPU smoke. New environment variables in
this release's generated sbatch shell are not all exported; the wrapper receives
the immutable source SHA as an explicit argument and exports job-local cache
settings itself. Queued jobs use a detached source snapshot and unique outputs.

The native cluv config is in `pyproject.toml`; its default preflight validates
without SSH or remote side effects. Cluster staging, source access and exact
submission commands are specified in `doc/auto/compute-plan.md`. No remote
allocation, environment preparation, or project-file copy was performed during
bootstrap. Approval is requested only for the concrete bounded next step.

## Scientific decisions and next work

Use exact discrete updates rather than assuming the draft's `F-s` vector field
embeds the recurrence. Verify orbit/Jacobian periodicity before interpreting
products as Floquet maps. Treat covariance dimension, perturbation contraction,
task information and causal memory as separate estimands. Periodic output can
persist under contracting dynamics; repeated-period covariance has a sampling
rank ceiling. The inherited AB/AABB task notation requires resolution.

Next: pin T-GRAB source/data; recover checkpoints and producing analysis;
validate complete graph hidden/cell state and output chronology on a tiny graph;
only then prepare a reviewed GPU reproduction pilot. Cross-modal matched
tasks and independent natural-data pilots follow graph evidence review.

## Integration validation

Final local checks passed: nine unit/integration tests, ten synthetic calibration
checks, all four skill validators, all shell syntax checks, repository-relative
document links, seven-node acyclic plan, and source-audit artifact hashes.
Final pre-commit calibration is `runs/calibration/setup-2026-10-11-v4/`; earlier
attempts are retained. The saved arrays reproduce reported dimensions and all
manifest hashes verify; duplicate-output attempts are rejected.

Independent review found and corrected covariance overflow, ambiguous degenerate
rank reporting, nonexistent source SHA acceptance, alternative-config preview
mismatch, and incomplete offline Python staging. Preparation now explicitly
installs the pinned Python into scratch and checks a fresh offline environment.

Remote uv is 0.9.18; the Mac's global uv is 0.8.13. cluv can automatically
align remote uv to the client version. The ops group therefore pins uv 0.9.18,
confirmed through `uv run --locked --group ops uv --version`, avoiding an
unintended downgrade of shared cluster tooling. Remote GitHub SSH access to
this initially empty repository was verified on both Mila and Tamia. Local
GitHub SSH failed; HTTPS listing succeeded. Git retains the requested SSH
origin and uses a separate HTTPS push URL for local publication.

No cluster job was submitted. A committed-source Tamia preview and publication
verification follow this integration record.
