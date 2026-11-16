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

**Status refreshed 2026-11-16; latest verified experiments: October 23.**
Read the public research note:
[Does repetition make a recurrent model forget?](https://mehrabhamidi99.github.io/blog/repetition-and-memory/).

The corrected seed-0 GC-LSTM long fit completed
**200 epochs / 511,800 updates**; validation selected epoch **160**. Its single
held-out pass gives mean F1 **0.891900** and a frozen early-to-late decline
**0.959297 → 0.818771** (14.05 percentage points) within repeated motifs.
[Verified report, plots, compact spectra and provenance](docs/October/2026-10-20/local/auto-research/figure10-gclstm-longfit-results-2026-10-20.md).

Complete hidden-plus-cell variance falls sharply within both motifs, while
participation ratio decreases modestly. Hidden-only dimension moves in different
directions between motifs. Pooling motifs produces a much larger PR decrease
while variance increases, so conditioning changes the interpretation. States
are not exactly periodic; no Floquet mechanism or causal memory loss is established.
This is one corrected model/graph, below Persistence F1 **0.968750**, rather than
the full multi-method/five-seed Figure 10. The earlier
[seven-epoch pilot](docs/October/2026-10-19/local/auto-research/figure10-gclstm-pilot-results-2026-10-19.md)
is preserved; the two budgets are not independent replications.

A fixed-checkpoint graph sweep subsequently found mean F1 falling from **0.8894
at dwell32 to 0.6565 at dwell256**, while persistence improved. Changed timing
relative to training is a competing explanation; this is not a memory-capacity
measurement. [Sweep report](docs/October/2026-10-20/local/auto-research/repetition-dwell-results-2026-10-20.md).

The simpler one-bit LSTM passed its competence gates and retained the bit with
**100% accuracy in periodic and matched-control arms through 6,144 body steps**.
Low pooled H+C participation ratio coexists with useful memory in this setting.
This is one seed/bit, not a capacity curve or a Floquet finding.
[One-bit report](docs/October/2026-10-23/local/auto-research/onebit-long-horizon-results-2026-10-23.md).
Earlier harder memory pilots failed competence; they remain distinct outcomes.

Mila jobs **11322268** (fit), **11330003** (selected replay/evaluation), and
**11330070** (state analysis) completed successfully. Independent audits verified
source identities, exact selected complete-state replay, numerical summaries
and actual offline W&B history; cloud delivery remains unverified. The previous
monitor had an overnight connection-error gap despite training continuing.
The local Qwen worker was subsequently stopped at the owner's request. Current
cluster queues and worker liveness were not inspected during the November16
publication update; no new experiment was launched. The next scientific node
is competent multi-bit address-query retention, followed by timing controls,
independent training seeds and complete-state tangent analysis.
[Current tracker and stop controls](docs/October/2026-10-26/local/report/TRACKER.md).

The [manuscript](paper/README.md) separates the behavioral reconstruction from
its geometric and causal limits. The
[coverage plan](docs/October/2026-10-19/local/auto-research/figure10-coverage-and-next-fit-2026-10-19.md) and
[T-GCN readiness audit](docs/October/2026-10-20/local/auto-research/tgcn-reconstruction-readiness-2026-10-20.md)
identify remaining models, seeds and source-parity gates. T-GCN now has an accepted
[native calibration](docs/October/2026-10-20/local/auto-research/tgcn-native-calibration-results-2026-10-20.md) and
a verified [256-update production training prefix](docs/October/2026-10-20/local/auto-research/tgcn-production-prefix-results-2026-10-20.md).
Its full fit, checkpoint selection/replay and held-out curve remain pending. Source snapshots remain
preserved. Numerical runs follow the [reporting contract](docs/October/2026-10-26/local/auto-research/wandb-reporting.md).

## Start here

| Topic | Canonical document |
| --- | --- |
| Question, hypotheses, claims, antecedent deltas, publication rule | [docs/October/2026-10-26/local/auto-research/research-charter.md](docs/October/2026-10-26/local/auto-research/research-charter.md) |
| Theory: propositions P1–P12 with proofs, fixtures, limits, predictions; corrections to the inherited draft | [docs/October/2026-10-26/local/auto-research/theory-propositions.md](docs/October/2026-10-26/local/auto-research/theory-propositions.md) |
| Experimental design: task, intervention, Gates A–D, measurement battery, statistics, evidence ledger | [docs/October/2026-10-26/local/auto-research/experimental-design-v2.md](docs/October/2026-10-26/local/auto-research/experimental-design-v2.md) |
| Measurement contract | [docs/October/2026-10-26/local/auto-research/research-program.md](docs/October/2026-10-26/local/auto-research/research-program.md), [skills/floquet-analysis/SKILL.md](skills/floquet-analysis/SKILL.md) |
| Evidence and current state | [docs/October/2026-10-26/local/report/TRACKER.md](docs/October/2026-10-26/local/report/TRACKER.md), dated reports under `docs/MONTH/YYYY-MM-DD/CLUSTER/report/` |
| Literature and novelty audit | [docs/October/2026-10-26/local/literature/novelty-review.md](docs/October/2026-10-26/local/literature/novelty-review.md), [docs/October/2026-10-26/local/literature/README.md](docs/October/2026-10-26/local/literature/README.md) |
| Code status versus the contract | [docs/October/2026-10-26/local/auto-research/implementation-audit.md](docs/October/2026-10-26/local/auto-research/implementation-audit.md) |
| Diagnostic manuscript | [paper/README.md](paper/README.md) |
| Operations and compute | [docs/October/2026-10-26/local/auto-research/compute-plan.md](docs/October/2026-10-26/local/auto-research/compute-plan.md), [skills/project-ops/SKILL.md](skills/project-ops/SKILL.md) |

Agents start at [AGENTS.md](AGENTS.md). The declarative dependency plan is
[configs/research-plan.json](configs/research-plan.json); the
[agent protocol](docs/October/2026-10-26/local/auto-research/agent-protocol.md) defines bounded subagent handoffs.

## Repository layout

| Path | Role |
| --- | --- |
| `src/representation_collapse/` | Measurement, calibration, vector task, pilot training and checkpoint periodicity analysis |
| `tests/` | Measurement and operations tests; versioned verification receipts in the tracker |
| `configs/` | Research plan DAG, frozen experiment configurations, cluv examples |
| `docs/MONTH/YYYY-MM-DD/CLUSTER/` | Current protocols, reviews, literature, planning and progress reports by type |
| `doc/auto/` | Legacy redirects, checksum-pinned protocols and immutable evidence |
| `documentation-index.json` | Canonical path mapping for migrated documents |
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
The [diagnostic constructor/oracle guide](docs/October/2026-10-26/local/auto-research/diagnostics-implementation.md)
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

Follow the [compute plan](docs/October/2026-10-26/local/auto-research/compute-plan.md) and
[cluster access notes](docs/October/2026-10-26/cross-cluster/auto-research/cluster-access.md). Cluster jobs require a
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
