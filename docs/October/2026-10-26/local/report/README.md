# Documentation map and lab notebook

The canonical current state is [TRACKER.md](TRACKER.md); reusable protocols,
reviews and evidence records are under `auto/`; the literature library is under
`literature/`; dated reports are under `<month>/<month><day>/`.

Current objective: [active research goal](../auto-research/active-research-goal.md), following
the charter and novelty audit. Figure 10 work now includes a verified native
[256-update prefix](../../../2026-10-19/local/auto-research/figure10-prefix-results-2026-10-19.md), a verified
[GC-LSTM behavioral reconstruction and state analysis](../../../2026-10-19/local/auto-research/figure10-gclstm-pilot-results-2026-10-19.md), independently
verified [analytic controls](../../../2026-10-19/local/auto-research/figure10-deterministic-controls-results-2026-10-19.md),
and an explicit [coverage and next-fit audit](../../../2026-10-19/local/auto-research/figure10-coverage-and-next-fit-2026-10-19.md).
The tracker records live outcomes; older review documents describe their
dated state. Relevant engineering reviews: [primary converter/store
review](../../../2026-10-16/local/auto-research/tgrab-primary-data-conversion-independent-review-2026-10-16.md),
[input diagnostics review](../../../2026-10-16/local/auto-research/matched-input-statistics-independent-review-2026-10-16.md),
and [conversion-plus-verification job review](../../../2026-10-16/local/auto-research/graph-data-conversion-verification-job-independent-review-2026-10-16.md).
The [independent raw-to-artifact verifier](../../../2026-10-16/local/auto-research/tgrab-primary-data-verifier-independent-review-2026-10-16.md)
and [source-derived checkpoint catalog](../../../2026-10-16/local/auto-research/tgrab-selected-tensor-catalog-independent-review-2026-10-16.md)
have also passed review. Primary conversion and the small native node2a
fixture have since been independently verified. Selected trained-checkpoint loading and the seed-0 held-out reconstruction are
verified. Harmful useful-memory erasure and the full multi-method Figure 10
remain unresolved. The longer cluv fit and its selected evaluation/state analysis completed; see the
[long-fit report](../../../2026-10-20/local/auto-research/figure10-gclstm-longfit-results-2026-10-20.md) and tracker.
The [selected checkpoint proposal](../../../2026-10-16/local/auto-research/tgrab-selected-checkpoint-codec-proposal-2026-10-16.md)
records the original format decisions; current implementations and execution
evidence are indexed by the tracker.

## Canonical documents (read in this order)

1. [Research charter](../auto-research/research-charter.md): question, sub-questions mapped
   to Gates A–D, hypotheses with falsifiers, estimands, primary contrast,
   claims and non-claims, antecedent deltas, publication threshold.
2. [Theory propositions](../auto-research/theory-propositions.md): P1–P12 with assumptions,
   proofs, fixtures, limits and predictions; corrections to the inherited draft.
3. [Experimental design v2](../auto-research/experimental-design-v2.md): task and
   intervention checked against the code, staged gates, measurement battery,
   interpretation matrix, statistics, evidence ledger, implementation gaps.
4. [Implementation audit](../auto-research/implementation-audit.md): what the code enforces,
   what the tests cover, numerical risks, backlog for Gate B.
5. [Novelty review](../literature/novelty-review.md) and the
   [literature library](../literature/README.md).

Supporting protocols: [measurement contract](../auto-research/research-program.md),
[competence-recovery ladder](../auto-research/competence-recovery-plan.md),
[bounded training integration](../auto-research/competence-training-v2.md),
[checkpoint periodicity protocol v1](../auto-research/periodicity-analysis-v1.md),
[graph reproduction](../auto-research/graph-reproduction.md),
[pinned upstream graph audit](../../../2026-10-12/local/auto-research/tgrab-upstream-pin-2026-10-12.md),
[source inventory](../auto-research/source-inventory.md). The
[periodicity research plan](../auto-research/periodicity-research-plan.md) records the
theoretical and experimental path developed with the owner; its broad
351-checkpoint screen is deferred coverage under the v2 gate order.

Evidence records: [initial-phase report](../auto-research/initial-phase-report.md),
[independent scientific review](../auto-research/initial-phase-scientific-review.md),
[compact results](../../../../../doc/auto/initial-phase-results.json),
[periodicity launch](../../../../../doc/auto/periodicity-analysis-launch.json) and
[original completion status](../../../../../doc/auto/periodicity-analysis-status.json),
[independent numerical reduction and plot](../auto-research/periodicity-independent-results.md),
[frozen reduction protocol](../auto-research/periodicity-independent-reduction-v1.md),
[local competence grid](../../../2026-10-13/local/auto-research/local-competence-grid-2026-10-13.md),
[prefix acquisition results/plot](../../../2026-10-13/local/auto-research/prefix-acquisition-v3-results-2026-10-13.md),
[addressed diagnostic and results](../../../2026-10-13/local/auto-research/address-acquisition-v4-results-2026-10-13.md),
[query-averaging proposal](../auto-research/query-averaged-acquisition-proposal-v5.md),
[stopped v5 campaign and independent review](../../../2026-10-13/local/auto-research/query-averaged-acquisition-v5-results-2026-10-13.md),
[prospective numerical chronology calibration](../../../2026-10-13/local/auto-research/native-chronology-calibration-proposal-2026-10-13.md),
[reviewed calibration implementation](../auto-research/native-chronology-calibration-v1.md),
[independent full calibration results](../../../2026-10-13/local/auto-research/native-chronology-calibration-v1-independent-results-review-2026-10-13.md),
[reviewed next forward-only contract](../../../2026-10-14/local/auto-research/post-calibration-measurement-contract-v2-2026-10-14.md),
[contract review](../../../2026-10-14/local/auto-research/post-calibration-contract-v2-independent-review-2026-10-14.md),
[forward implementation review](../../../2026-10-14/local/auto-research/cached-forward-v2-implementation-independent-review-2026-10-14.md),
[independent forward results](../../../2026-10-14/local/auto-research/cached-forward-v2-results-independent-review-2026-10-14.md),
[T-GRAB archive recovery and debugging](../../../2026-10-14/local/auto-research/tgrab-reproduction-audit-2026-10-14.md),
[verified legacy arrays and figure](../../../2026-10-14/local/auto-research/legacy-floquet-array-audit-results-2026-10-14.md),
[independent raw-array closeout](../../../2026-10-14/local/auto-research/legacy-floquet-array-results-independent-review-2026-10-14.md),
[corrected graph component review](../../../2026-10-14/local/auto-research/tgrab-reconstruction-implementation-independent-review-2026-10-14.md),
[causal-bridge scope audit](../../../2026-10-14/local/auto-research/causal-bridge-audit-2026-10-14.md),
[conditional approximate-bridge bounds](../../../2026-10-14/local/auto-research/approximate-bridge-audit-2026-10-14.md),
and [current research/worker status](../../../../../doc/auto/research-status-2026-10-16.json).

Operations: [compute plan](../auto-research/compute-plan.md),
[cluster access](../../cross-cluster/auto-research/cluster-access.md), [agent protocol](../auto-research/agent-protocol.md),
[Qwen worker](../auto-research/qwen-worker.md),
[active local continuation](../../../2026-10-14/local/auto-research/local-autoresearch-continuation-2026-10-14.md),
[daytime research plan](../../../2026-10-13/local/auto-research/day-research-2026-10-13.md),
[continuous monitor](../../../2026-10-13/local/auto-research/qwen-continuous-monitor-deployment-2026-10-13.md),
[proposal-quality audit](../../../2026-10-13/local/auto-research/qwen-proposal-quality-2026-10-13.md),
[W&B reporting and verification](../auto-research/wandb-reporting.md).

## Notebook convention

Write one report per workday:
`doc/<lowercase-month>/<month><day>/YYYY-MM-DD-progress-report.md`.
For example, `docs/October/2026-10-12/local/report/2026-10-12-progress-report.md`.
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

- [GPU efficiency adoption and evidence](../../../2026-10-12/local/auto-research/gpu-efficiency-adoption-2026-10-12.md)
- [Prospective statistics proposal](../auto-research/statistics-freeze-proposal-v2.md) — reviewed, not frozen.
- [Overnight continuation](../../../2026-10-12/local/auto-research/overnight-research-2026-10-12.md)

T-GCN implementation now has an independently accepted
[native synthetic calibration](../../../2026-10-20/local/auto-research/tgcn-native-calibration-results-2026-10-20.md).
Its separate [production training-prefix protocol](../../../2026-10-20/local/auto-research/tgcn-production-prefix-protocol-2026-10-20.md)
has a verified [first256-update training result](../../../2026-10-20/local/auto-research/tgcn-production-prefix-results-2026-10-20.md)
without validation/test access. Actual execution status belongs to the tracker,
not the prospective protocol.
