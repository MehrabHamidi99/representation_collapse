# Representation Collapse

When does repetition erase what a recurrent network needs? This project studies
useful-history loss under controlled periodic forcing in fixed-weight recurrent
models: a competent network writes task-required values into its complete state,
receives a distractor body that is either an exact repetition of one motif or a
matched non-periodic motif sequence of equal length, and is then queried. We ask
whether the periodic body causes excess loss of the written information, through
which state-transition mechanism, and when stable attractor memory prevents the
loss. Covariance geometry, tangent propagation and behavior/information are
measured as separate estimands; analytic nulls fix what ordinary stability can
and cannot explain. "Dimension collapse across modalities" is the motivating
umbrella, not the claim. The program began with temporal graphs (GC-LSTM on the
public T-GRAB benchmark) and currently runs on controlled vector memory tasks.

**Status (2026-10-12):** no accepted scientific claim. Measurement and mechanism
calibration pass (infrastructure evidence). Four seed-0 GRU/LSTM pilots failed
short-horizon competence, so the harmful-collapse decision is inconclusive.
Checkpoint-only periodicity diagnostics now pass independent numerical reduction
and review; their finite-time measurements remain descriptive. The competence
recovery trainer, pinned-validation selector and complete native phase capture
are implemented and locally reviewed, with new training still pending. Upstream
graph sources are pinned; legacy checkpoint/data payload provenance remains
missing. Current Qwen liveness is unverified after an SSH master expired.
See [doc/TRACKER.md](doc/TRACKER.md).

## Start here

| Topic | Canonical document |
| --- | --- |
| Question, hypotheses, claims, antecedent deltas, publication rule | [doc/auto/research-charter.md](doc/auto/research-charter.md) |
| Theory: propositions P1–P12 with proofs, fixtures, limits, predictions; corrections to the inherited draft | [doc/auto/theory-propositions.md](doc/auto/theory-propositions.md) |
| Experimental design: task, intervention, Gates A–D, measurement battery, statistics, evidence ledger | [doc/auto/experimental-design-v2.md](doc/auto/experimental-design-v2.md) |
| Measurement contract | [doc/auto/research-program.md](doc/auto/research-program.md), [skills/floquet-analysis/SKILL.md](skills/floquet-analysis/SKILL.md) |
| Evidence and current state | [doc/TRACKER.md](doc/TRACKER.md), [doc/auto/initial-phase-scientific-review.md](doc/auto/initial-phase-scientific-review.md), dated notebook under `doc/october/` |
| Literature and novelty audit | [doc/literature/novelty-review.md](doc/literature/novelty-review.md), [doc/literature/README.md](doc/literature/README.md) |
| Code status versus the contract | [doc/auto/implementation-audit.md](doc/auto/implementation-audit.md) |
| Diagnostic manuscript | [paper/README.md](paper/README.md) |
| Operations and compute | [doc/auto/compute-plan.md](doc/auto/compute-plan.md), [skills/project-ops/SKILL.md](skills/project-ops/SKILL.md) |

Agents start at [AGENTS.md](AGENTS.md). The declarative dependency plan is
[configs/research-plan.json](configs/research-plan.json); the
[agent protocol](doc/auto/agent-protocol.md) defines bounded subagent handoffs.

## Repository layout

| Path | Role |
| --- | --- |
| `src/representation_collapse/` | Measurement, calibration, vector task, pilot training and checkpoint periodicity analysis |
| `tests/` | 128 passing unit and integration tests, no skips |
| `configs/` | Research plan DAG, frozen experiment configurations, cluv examples |
| `doc/auto/` | Scientific protocols, reviews, evidence records and operations notes |
| `doc/literature/` | Bibliography, novelty review, reference records |
| `doc/<month>/` | Dated progress reports (append-only) |
| `paper/` | Focused manuscript with reproducible analytic and pilot figures |
| `skills/` | Canonical operating skills; regular discovery entries in `.agents/skills/` |
| `code/ops/`, `code/literature/` | Cluster submission/monitoring helpers and the literature builder |
| `floquet_analysis/`, `ACM___TGRAB/`, `ACM___TGRAB_KDD_2026/` | Inherited manuscript snapshots, preserved unchanged |

## Local checks

```bash
uv sync --locked --all-groups
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m representation_collapse.calibrate --output runs/calibration/local
uv run --locked python -m representation_collapse.mechanism_calibration --output runs/calibration/local-mechanism
```

`torch` is in the `experiments` dependency group; a plain `uv sync --locked`
installs a torch-less environment in which model-dependent tests are skipped.
If `uv run` cannot import the package (observed on one macOS checkout where `.venv` files
carry the hidden flag and CPython skips hidden `.pth` files), use
`PYTHONPATH="$PWD/src" .venv/bin/python` in place of `uv run --locked python`.
The [diagnostic constructor/oracle guide](doc/auto/diagnostics-implementation.md)
explains the additive R0, single-WRITE and corruption-control fixtures.

The calibrations check participation ratio and entropy dimension on known
covariances, ordered Jacobian products, multipliers of known periodic maps,
nonnormal transient gain, forced periodic responses, and the seven analytic
mechanism families (10 + 36 checks). Choose a new output directory for every
attempt. These calibrate analysis code; they are not trained-model findings.

## Research and compute

```bash
uv run --group ops cluv --help
uv run --group ops python code/ops/preflight.py
```

Follow the [compute plan](doc/auto/compute-plan.md) and
[cluster access notes](doc/auto/cluster-access.md). Cluster jobs require a
concrete approved resource envelope recorded in ignored
`configs/approval.local.json`; both envelopes used so far are exhausted and no
new allocation is authorized by any document in this repository. Graph
code/data leads from the manuscripts must be pinned and audited before training
or reproduction.

## Overleaf

Import the GitHub repository and select `paper/main.tex` as the main document
for the current manuscript. The inherited manuscript snapshots are separate
entrypoints. Skill discovery uses regular files; the repository contains no
tracked symbolic links, which would block Overleaf's GitHub import.
