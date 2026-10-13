# Documentation map and lab notebook

The canonical current state is [TRACKER.md](TRACKER.md); reusable protocols,
reviews and evidence records are under `auto/`; the literature library is under
`literature/`; dated reports are under `<month>/<month><day>/`.

## Canonical documents (read in this order)

1. [Research charter](auto/research-charter.md): question, sub-questions mapped
   to Gates A–D, hypotheses with falsifiers, estimands, primary contrast,
   claims and non-claims, antecedent deltas, publication threshold.
2. [Theory propositions](auto/theory-propositions.md): P1–P12 with assumptions,
   proofs, fixtures, limits and predictions; corrections to the inherited draft.
3. [Experimental design v2](auto/experimental-design-v2.md): task and
   intervention checked against the code, staged gates, measurement battery,
   interpretation matrix, statistics, evidence ledger, implementation gaps.
4. [Implementation audit](auto/implementation-audit.md): what the code enforces,
   what the tests cover, numerical risks, backlog for Gate B.
5. [Novelty review](literature/novelty-review.md) and the
   [literature library](literature/README.md).

Supporting protocols: [measurement contract](auto/research-program.md),
[competence-recovery ladder](auto/competence-recovery-plan.md),
[bounded training integration](auto/competence-training-v2.md),
[checkpoint periodicity protocol v1](auto/periodicity-analysis-v1.md),
[graph reproduction](auto/graph-reproduction.md),
[pinned upstream graph audit](auto/tgrab-upstream-pin-2026-10-12.md),
[source inventory](auto/source-inventory.md). The
[periodicity research plan](auto/periodicity-research-plan.md) records the
theoretical and experimental path developed with the owner; its broad
351-checkpoint screen is deferred coverage under the v2 gate order.

Evidence records: [initial-phase report](auto/initial-phase-report.md),
[independent scientific review](auto/initial-phase-scientific-review.md),
[compact results](auto/initial-phase-results.json),
[periodicity launch](auto/periodicity-analysis-launch.json) and
[original completion status](auto/periodicity-analysis-status.json),
[independent numerical reduction and plot](auto/periodicity-independent-results.md),
[frozen reduction protocol](auto/periodicity-independent-reduction-v1.md),
[local competence grid](auto/local-competence-grid-2026-10-13.md),
and [current research/worker status](auto/research-status-2026-10-13.json).

Operations: [compute plan](auto/compute-plan.md),
[cluster access](auto/cluster-access.md), [agent protocol](auto/agent-protocol.md),
[Qwen worker](auto/qwen-worker.md),
[daytime research plan](auto/day-research-2026-10-13.md).

## Notebook convention

Write one report per workday:
`doc/<lowercase-month>/<month><day>/YYYY-MM-DD-progress-report.md`.
For example, `doc/october/october12/2026-10-12-progress-report.md`.
Append timestamped sections when continuing the same day. Reports retain
history; the tracker is edited to reflect current state.

Every report records purpose, changes, commands, evidence/artifact paths,
results with their scope, unresolved issues, and the next bounded tasks.
Distinguish proposed, implemented, tested locally, tested on cluster,
reproduced, and scientifically accepted. A successful job alone is insufficient.
Link scratch artifacts rather than copying model weights or datasets into git.

Record protocol changes in a dated decision note and link the new frozen
protocol. Analysis references source hashes, environment lock, run ID,
cluster/job ID if applicable, seeds, controls, and uncertainty method.
Speculative ideas remain labeled hypotheses until measured.

- [GPU efficiency adoption and evidence](auto/gpu-efficiency-adoption-2026-10-12.md)
- [Prospective statistics proposal](auto/statistics-freeze-proposal-v2.md) — reviewed, not frozen.
- [Overnight continuation](auto/overnight-research-2026-10-12.md)
