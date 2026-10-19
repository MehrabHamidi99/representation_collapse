# Documentation map and lab notebook

The canonical current state is [TRACKER.md](TRACKER.md); reusable protocols,
reviews and evidence records are under `auto/`; the literature library is under
`literature/`; dated reports are under `<month>/<month><day>/`.

Current objective: [active research goal](auto/active-research-goal.md), following
the charter and novelty audit. Figure 10 work now includes a verified native
[256-update prefix](auto/figure10-prefix-results-2026-10-19.md), a running
[complete-epoch GC-LSTM fit](auto/figure10-full-fit-2026-10-19.md), independently
verified [analytic controls](auto/figure10-deterministic-controls-results-2026-10-19.md),
and an explicit [coverage and next-fit audit](auto/figure10-coverage-and-next-fit-2026-10-19.md).
The tracker records live outcomes; older review documents describe their
dated state. Relevant engineering reviews: [primary converter/store
review](auto/tgrab-primary-data-conversion-independent-review-2026-10-16.md),
[input diagnostics review](auto/matched-input-statistics-independent-review-2026-10-16.md),
and [conversion-plus-verification job review](auto/graph-data-conversion-verification-job-independent-review-2026-10-16.md).
The [independent raw-to-artifact verifier](auto/tgrab-primary-data-verifier-independent-review-2026-10-16.md)
and [source-derived checkpoint catalog](auto/tgrab-selected-tensor-catalog-independent-review-2026-10-16.md)
have also passed review. Primary conversion and the small native node2a
fixture have since been independently verified. Selected trained-checkpoint
loading, held-out reconstruction and trained collapse evidence remain pending.
The [selected checkpoint proposal](auto/tgrab-selected-checkpoint-codec-proposal-2026-10-16.md)
records the original format decisions; current implementations and execution
evidence are indexed by the tracker.

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
[prefix acquisition results/plot](auto/prefix-acquisition-v3-results-2026-10-13.md),
[addressed diagnostic and results](auto/address-acquisition-v4-results-2026-10-13.md),
[query-averaging proposal](auto/query-averaged-acquisition-proposal-v5.md),
[stopped v5 campaign and independent review](auto/query-averaged-acquisition-v5-results-2026-10-13.md),
[prospective numerical chronology calibration](auto/native-chronology-calibration-proposal-2026-10-13.md),
[reviewed calibration implementation](auto/native-chronology-calibration-v1.md),
[independent full calibration results](auto/native-chronology-calibration-v1-independent-results-review-2026-10-13.md),
[reviewed next forward-only contract](auto/post-calibration-measurement-contract-v2-2026-10-14.md),
[contract review](auto/post-calibration-contract-v2-independent-review-2026-10-14.md),
[forward implementation review](auto/cached-forward-v2-implementation-independent-review-2026-10-14.md),
[independent forward results](auto/cached-forward-v2-results-independent-review-2026-10-14.md),
[T-GRAB archive recovery and debugging](auto/tgrab-reproduction-audit-2026-10-14.md),
[verified legacy arrays and figure](auto/legacy-floquet-array-audit-results-2026-10-14.md),
[independent raw-array closeout](auto/legacy-floquet-array-results-independent-review-2026-10-14.md),
[corrected graph component review](auto/tgrab-reconstruction-implementation-independent-review-2026-10-14.md),
[causal-bridge scope audit](auto/causal-bridge-audit-2026-10-14.md),
[conditional approximate-bridge bounds](auto/approximate-bridge-audit-2026-10-14.md),
and [current research/worker status](auto/research-status-2026-10-16.json).

Operations: [compute plan](auto/compute-plan.md),
[cluster access](auto/cluster-access.md), [agent protocol](auto/agent-protocol.md),
[Qwen worker](auto/qwen-worker.md),
[active local continuation](auto/local-autoresearch-continuation-2026-10-14.md),
[daytime research plan](auto/day-research-2026-10-13.md),
[continuous monitor](auto/qwen-continuous-monitor-deployment-2026-10-13.md),
[proposal-quality audit](auto/qwen-proposal-quality-2026-10-13.md),
[W&B reporting and verification](auto/wandb-reporting.md).

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
