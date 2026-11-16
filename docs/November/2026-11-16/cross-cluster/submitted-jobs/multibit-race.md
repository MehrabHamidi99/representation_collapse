# Higher-dimensional memory: actual cluv execution

Recorded November 16, 2026. This continues the owner's explicit instruction to
run the higher-dimensional follow-up with native `cluv sync` and
`cluv submit first job.sh`. It does not infer authority from another project.

## Frozen experiment and launch

Source: `5e432540731a2b1c6df82dad73da63bd110dc53e`. The
[prospective protocol](../experiment/multibit-memory-v1.md), configuration,
worker, wrapper and CPU environment lock were committed before submission.
The launch review passed 11 focused tests locally; all 11 also passed in the
staged Rorqual production environment, Torch 2.0.0+cpu, in 13.161 seconds.
This resolves the production-compatibility item left pending in the immutable
[prelaunch review](multibit-launch-review.json).

From `launch/multibit-20261116`, with the repository's locked operations
environment on PATH, native cluv 0.1.5 executed:

```bash
cluv submit first job.sh --no-sync-datasets -- \
  5e432540731a2b1c6df82dad73da63bd110dc53e multibit-20261116T2243Z-s0
```

Six candidates were enabled: Mila, Tamia, Rorqual, Fir, Killarney and Vulcan.
Each requested one CPU, 8 GiB, zero GPUs and three hours, with no requeue.
Five were submitted before the winner began; cluv skipped Tamia before
submission. There was one race and no retry or subsequent training campaign.

## Scheduler reconciliation

- Mila **11741707** won, ran on `cn-h002`, and completed `0:0` in **314 seconds**.
- Rorqual **26966492** was cancelled after six allocated seconds.
- Fir **4458321** was cancelled before allocation.
- Killarney **7678414** was cancelled before allocation.
- Vulcan **2388345** was cancelled after eight allocated seconds; the final
  accounting reports the expected termination signal, `0:15`.

The parent accounting rows total 328 allocated CPU-seconds and zero GPUs.
Overlapping batch/extern rows are not additional CPU allocations; cluster
billing weights are separate from CPU time. See the exact
[launch receipt](multibit-race-launch.json) and
[terminal accounting](multibit-race-accounting.json).
The receipt's nonzero composite SSH status comes from `squeue` rejecting job
IDs already removed from its live queue; the preceding `sacct` rows establish
each terminal state.

Cluv's bare `scancel` was unavailable in Killarney's nonlogin shell. The
coordinator cancelled only owned job 7678414 through a login shell, then
verified its terminal state independently. Cluv subsequently exited normally
with Mila as winner. No unrelated job was changed.

## Cluster coverage and exclusions

Native cluv synchronized the exact final source and installed the locked CPU
environment on the six enabled candidates. Nine targets were reachable for
initial Git staging; this does not mean nine targets ran the experiment.
The [source verification](multibit-source-sync.json) distinguishes final source
from preparation commit `6a78814`:

- Nibi: reachable and scheduler-eligible, but its initial dependency staging
  remained incomplete when dispatch occurred; excluded from this race.
- Trillium: the CPU scheduler requires a 192-core whole-node allocation.
  Trillium-GPU requires a GPU allocation. Both retained preparation source and
  staged dependencies; neither was admitted for this one-thread CPU workload.
- Narval: the direct probe timed out and the Tamia route rejected batch
  authentication. No submission or verified synchronization is claimed.

Disabled candidates were scoped to this launch project's cluv cache. Other
projects' settings, sources and jobs were preserved. The winner's source was
not changed during execution.

## Outputs, stopping and reporting

Winner outputs remain on Mila scratch:

```text
/network/scratch/m/mehrab.hamidi/representation-collapse/runs/multibit-memory/multibit-20261116T2243Z-s0
```

The sibling `multibit-20261116T2243Z-s0.STOP`, worker deadline of 10,500 seconds,
outer timeout of 10,620 seconds, and Slurm limit of 10,800 seconds were configured.
None was needed. The worker finished all four fit attempts at 22:48:21 UTC.
Its statuses report competence passed for D=1 and D=4 after 10,000 updates each,
and competence failed at stage 2 for D=16 and D=64 after 6,000 updates each.
Failed fits did not open held-out evaluations.

Every file listed in the four producer inventories, including the cluster-only
checkpoints, was hashed again on Mila; all match. The
[verification receipt](multibit-remote-inventory.json) preserves those checks.
Only derived JSON/NPZ artifacts were retrieved locally for independent reduction;
no model checkpoints or primary datasets were downloaded to the laptop.

W&B is required because this is monitored training and scientific evaluation.
Each fit wrote an offline history under project `representation-collapse`.
Cloud synchronization and independent numerical reduction are recorded in the
result report separately from scheduler success. No persistent Qwen service
was started by this launch; the Slurm process was a deterministic experiment
worker. The automatic goal service did not accept a new goal because the thread
already has an unfinished goal.
