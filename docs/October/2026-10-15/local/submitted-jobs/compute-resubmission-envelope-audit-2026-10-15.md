# Initial CPU resubmission envelope audit

2026-10-15. The recorded initial CPU envelope has **one submission remaining**:
one CPU, 2 GiB, ten minutes, zero GPUs. This is the fourth and last initial
analytic-calibration slot, not a renewed allocation. No submission or remote
query was performed in this independent review.

The ignored approval record, SHA-256
`5e7abb34ac18c933cd47f066b844462cc255f6b1bc270e6b8057f4eb2a77d1bc`,
fixes maximum four CPU submissions, one CPU/2 GiB/ten minutes each and maximum
2/3 requested CPU-hour. Its exact CPU purpose is “existing and additional
analytic calibration; no requeue.” Only public resource and purpose fields
were inspected; credentials, endpoints and unrelated worker fields were not
printed or exported.

Three distinct initial submissions are recorded:

- Mila `11190579`, source `7ca2be4c7827d0644b0216cb86bec1dbb855f5a4`,
  one CPU/2 GiB/ten minutes. The native submission ID is in
  `runs/ops/phase1-bootstrap-launch.log`. It failed before calibration at the
  prepared-Python check; the dated October 11 report records `FAILED/2:0`.
  A failed submission consumes its slot.
- Tamia `536757`, the same bootstrap source and resource limit, separately
  submitted in that log. The October 11 report records `COMPLETED/0:0`.
- Narval `5234514`, source `9004a0c114a8b7c5335dca3d5247449fd10b5ffa`,
  independently recorded in `runs/ops/phase1-launches-20261011.json`. Saved
  native UTC accounting in `phase1-status-20261011.json` verifies
  `COMPLETED/0:0`, 32 seconds elapsed, one CPU/2 GiB and a ten-minute limit,
  account `def-ioannism_cpu`, QOS `normal`. The scheduler reconciliation and
  final initial-phase report agree on that CPU roster.

Thus 3/4 slots and 1/2 requested CPU-hour are consumed; one slot and 1/6
requested CPU-hour remain. Short actual runtimes do not refund submission
counts or enlarge the requested-hour ceiling. The original bootstrap failed
attempt remains counted; it is not treated as an unsubmitted command.

Tamia `612745` used the distinct `continuation_20261012.cpu` envelope:
maximum one submission, one CPU/2 GiB/ten minutes, no GPU/retry/race. Its saved
launch/accounting/artifact receipt verifies completion and that separate cap
is consumed. Charging it again to the original maximum-four roster would mix
two explicitly separate approvals. The shared Mila hosting allocation
`11228391` is also separate, belongs to its hosting owner and ended; no initial
slot or GPU allowance transfers from it. Four original GPU pilots and four
later periodicity GPU diagnostics remain consumed within their own envelopes.

The coordinator supplied the latest explicit user directive “continue ...
resubmit.” Together with the existing additional-analytic-calibration purpose,
it supports one further CPU-only native graph analytic calibration within the
remaining limits. This inference depends on the actual packet remaining an
untrained, bounded instrumentation calibration: complete H+C state, native
cell/derivative checks, no optimizer, datasets, checkpoints or task-training
pilot. Calling that calibration “graph reproduction” or adding GPU/training
work would change the scope. The directive does not renew GPU budgets, permit
another automatic retry, authorize a race, or increase CPU/memory/time caps.
The `source_commit_cpu` field identifies bootstrap execution provenance; the
same envelope already supported the later separately frozen Narval source.
Every new attempt still needs its own exact source/configuration admission.

The coordinator can complete preparation and use the remaining recorded slot
without another permission question once the exact native-calibration packet
passes independent engineering review and is published. Admission must name
the actual cluster/account/partition/QOS, exact command and source/configuration
hashes, one unique scratch output, one submission, physical one-CPU/2-GiB/
600-second limits, no requeue/race/retry and its stopping rule. Current cluv,
target availability, offline environment and owned project jobs must be checked
before submission; this local evidence audit does not refresh scheduler state.
An ambiguous submission must be reconciled before another command is issued.

Use pinned SDK 0.30.0 offline reporting before native numerical work. Preserve
the complete H+C ordering, frozen derivative tolerances, physical Slurm caps,
work deadline and detached SDK-child ownership/cleanup. Infrastructure
acceptance will not accept trained-graph, orbit/Floquet, memory or collapse
claims. The native author has been asked to return immutable source, protocol
and configuration hashes before independent code review.

The exact evidence hashes, public approval projection, arithmetic, execution
conditions and pending native review checklist are preserved in
`runs/review/compute-resubmission-audit-20261015/capacity-review.json` and
`native-calibration-checklist.json`. No remote, native, provider, model or
compute execution was performed for this audit. The coordinator owns final
integration, exact admission and scheduler execution.
