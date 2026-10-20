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

**Status (2026-10-20):** the corrected seed-0 GC-LSTM long fit completed
**200 epochs / 511,800 updates**; validation selected epoch **160**. Its single
held-out pass gives mean F1 **0.891900** and a frozen early-to-late decline
**0.959297 → 0.818771** (14.05 percentage points) within repeated motifs.
[Verified report, plots, compact spectra and provenance](doc/auto/figure10-gclstm-longfit-results-2026-10-20.md).

Complete hidden-plus-cell variance falls sharply within both motifs, while
participation ratio decreases modestly. Hidden-only dimension moves in different
directions between motifs. Pooling motifs produces a much larger PR decrease
while variance increases, so conditioning changes the interpretation. States
are not exactly periodic; no Floquet mechanism or causal memory loss is established.
This is one corrected model/graph, below Persistence F1 **0.968750**, rather than
the full multi-method/five-seed Figure 10. The earlier
[seven-epoch pilot](doc/auto/figure10-gclstm-pilot-results-2026-10-19.md)
is preserved; the two budgets are not independent replications. Controlled-memory
pilots still fail their joint competence gate.

Mila jobs **11322268** (fit), **11330003** (selected replay/evaluation), and
**11330070** (state analysis) completed successfully. Independent audits verified
source identities, exact selected complete-state replay, numerical summaries
and actual offline W&B history; cloud delivery remains unverified. The previous
monitor had an overnight connection-error gap despite training continuing.
The local proposal-only Qwen worker is live with 44 completed reviews. Native agents implement and independently review changes.
[Current tracker and stop controls](doc/TRACKER.md).

The [manuscript](paper/README.md) separates the behavioral reconstruction from
its geometric and causal limits. The
[coverage plan](doc/auto/figure10-coverage-and-next-fit-2026-10-19.md) and
[T-GCN readiness audit](doc/auto/tgcn-reconstruction-readiness-2026-10-20.md)
identify remaining models, seeds and source-parity gates. T-GCN now has an accepted
[native calibration](doc/auto/tgcn-native-calibration-results-2026-10-20.md) and
a verified [256-update production training prefix](doc/auto/tgcn-production-prefix-results-2026-10-20.md).
Its full fit, checkpoint selection/replay and held-out curve remain pending. Source snapshots remain
preserved. Numerical runs follow the [reporting contract](doc/auto/wandb-reporting.md).

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
