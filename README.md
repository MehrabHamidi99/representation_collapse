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

**Status (2026-10-14):** harmful collapse remains inconclusive. The original
four pilots failed competence; subsequent acquisition work completed 32 local
fits and stopped one fit after 114,200 total updates. No checkpoint has passed
the original eight-coordinate gate. A 120,000-call numerical audit is independently
verified, with strict execution-path agreement failing at a trained GRU checkpoint.
A new label-free, complete-state calibration completed 32 conditions and 736
forward calls in 50 seconds. Independent review accepts the audit's integrity:
fixed-precision checks pass, while four FP32/FP64 perturbation-logit comparisons
fail at one independent history. See the [reviewed results](doc/auto/cached-forward-v2-results-independent-review-2026-10-14.md).
The experiment and reduction have verified offline W&B records; cloud sync is
pending.
Legacy graph-analysis source and numerical candidates have now been located in
the Mila TSA/Floquet archive; checkpoint and producer linkage remain unverified.
[Original graph reproduction is the next priority](doc/auto/tgrab-reproduction-audit-2026-10-14.md).

Qwen and its monitor were verified running at October 14 05:42 UTC. The worker
has nine verified replies and is waiting for evidence; no additional reply has
appeared overnight. Its five-minute monitor has 106 observations with no recorded
metadata issues in existing Mila allocation 11228391. Monitoring continues
through October 14 08:50 UTC;
allocation expiry is 08:55:56 UTC. Native coordination has had gaps, and worker
liveness alone does not establish continuous research. The new
[W&B project](https://wandb.ai/mehrab-hamidi-milaquebec5431/representation-collapse)
has a verified logging-only transport run. Future numerical runs use the
[reporting contract](doc/auto/wandb-reporting.md); historical runs were not
tracked from their beginning. See [doc/TRACKER.md](doc/TRACKER.md) for evidence
and limits.

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
| `tests/` | Measurement and operations tests; versioned verification receipts in the tracker |
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
Numerical experiment entry points additionally require their reviewed protocol,
source/config freeze and W&B initialization before work; the legacy calibration
modules have not yet been integrated with the new reporting helper.

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
