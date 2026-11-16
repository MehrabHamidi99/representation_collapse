# Load ladder v2: startup recovery

Prepared November 16, 2026, before recovery dispatch. Owner: Mehrab Hamidi.
Authority: the owner's request to increase the bit load and use `cluv sync`
followed by `cluv submit first`, together with the explicit instruction to run
the reproduction/follow-up work without another compute approval question.

## Original attempt and reason

Source `be71bd3df72aede3726b2696e28af1ecef2ce276` was synchronized to seven
eligible targets. Native Cluv selected Mila job **11742215**. Narval 9559340,
Rorqual 26970360, Fir 4467821, Killarney 7679342 and Vulcan 2389183 are
terminally cancelled; Tamia was skipped before submission. Killarney required
an explicit cancellation of this owned losing job after Cluv's cancellation
command failed in its non-login shell.

Mila seed 0 failed importing NumPy before CLI initialization; seed 2 failed
importing Torch before any fit. Both report missing package files. The files
were subsequently visible on the login node. Filesystem visibility is a
candidate explanation, not an established diagnosis. Seed 1 remains running;
at the recovery preparation check it reached D4 stage 3, update 32,100.
Preserve that worker, its checkout and all original failure logs.

## Separate recovery attempt

Recover **seeds 0 and 2 only**, using exactly the same scientific worker,
configuration, protocol, RNG streams, gates and evaluation conditions. No
result-dependent hyperparameter change or additional seed is introduced.
Cancelled race candidates may have performed partial work; their outputs are
retained and excluded from the selected attempt's scientific results.

The new launch directory is `launch/multibit-load-repair-20261116/`. It uses a
separate source checkout and publishes a runtime snapshot outside Cluv's
reinstalled `.venv` directory. `prepare_runtime.sh` runs on login nodes after
Cluv sync, verifies imports and versions, then atomically publishes the snapshot.
The worker runs a bounded import check before any model work. No dependency
installation occurs in the allocation. This isolates environment directory
entries; it does not assert that the underlying filesystem cannot fail.

Eligible candidates: Tamia, Narval, Rorqual, Fir, Killarney and Vulcan, using
the explicit accounts/partitions in the launch profile. Mila is excluded from
this recovery to avoid the observed startup failure while preserving its
healthy seed. Nibi's dependency preparation remains incomplete. Trillium CPU
requires a larger allocation and Trillium GPU requires a GPU, so neither is
appropriate for this CPU workload.

Per candidate: **2 CPUs, 8 GiB total RAM, zero GPUs, 3 hours**, one allocation,
two exclusive 1-CPU/3-GiB worker steps. At most six submissions in one native
first-started race; losing candidates are cancelled and independently reconciled.
Worst case before cancellation is 36 requested CPU-hours; selected attempt is
at most six CPU-hours. No requeue or implicit retry. Global STOP, a 10,500-second
worker deadline and Slurm's three-hour time limit bound execution.

After committing/pushing, run `cluv sync tamia narval rorqual fir killarney
vulcan --no-sync-datasets`, prepare and verify each runtime snapshot, record the
source SHA and new run ID, then run from the launch directory:

```sh
cluv submit first job.sh --no-sync-datasets -- SOURCE_SHA RUN_ID
```

Outputs: `$SCRATCH/representation-collapse/runs/multibit-load-ladder/RUN_ID/`.
Source: `$SCRATCH/representation-collapse/sources/multibit-load-repair-20261116/`.
Runtime: `$SCRATCH/representation-collapse/envs/load-v2-repair-20261116/`.
The original Mila source/run remains unchanged.

W&B is required for the scientific fits; use the existing offline per-fit
histories and later sync and verify the original runs. Runtime preparation,
polling and scheduler accounting do not create W&B runs. Failed pre-training
imports provide infrastructure evidence only. Scientific acceptance still
requires raw-array verification and separation of failed acquisition from
retention loss and representation geometry.

Actual source, run ID, submissions, winner and observations are appended in the
recovery receipt after dispatch. This document records a prospective operational
recovery, not a completed scientific result.

## Dispatch and verified progress at 23:44 UTC

Recovery source is `31dd7537e9d090697c133822b42d186314290547`; run ID is
`load-v2-repair-20261116T234042Z`. Native Cluv sync completed successfully on
all six intended candidates. Tamia, Fir, Killarney and Vulcan passed snapshot
publication and import verification. Narval's snapshot copy and Rorqual's import
check exceeded the 180-second local verification deadline; neither snapshot
was published at reconciliation, so both were excluded from the recovery race.
Their owned preparation process groups were identified and sent SIGTERM at
23:44 UTC; no duplicate preparation or submission was started. Nibi's earlier
Cluv staging command exited successfully, but a fresh import check timed out,
so it remains unverified. These exclusions concern this recovery attempt;
Narval and Rorqual participated in the original race.

Actual candidate list was **Tamia, Fir, Killarney, Vulcan**. The exact reserved
command and resources are in the [reservation](load-ladder-v2-recovery-reservation.json).
`cluv submit first job.sh --no-sync-datasets -- SOURCE_SHA RUN_ID` exited zero.
Vulcan **2389302** won on `compute1`; Fir **4470814** and Killarney **7679777**
are independently confirmed CANCELLED. Tamia was skipped before submission.
Killarney again required explicit owned-job cancellation through a login shell.
Three actual submissions requested at most 18 CPU-hours before cancellation;
only Vulcan received a recovery allocation, with two CPUs and 8 GiB.

At 23:44 UTC, Vulcan's two seed workers were RUNNING, at D4 stage 2 updates
12,600 and 12,700. Mila's preserved seed 1 had completed the 4- and 8-bit fits
with worker-reported competence passes and moved to 12-bit training. These are
execution observations, not independently accepted scientific results. The
original Mila wrapper will eventually retain its failed-seed exit codes;
scientific seed 1 artifacts must be reconciled separately from that aggregate
launcher failure. Combine seed 1 from the original Mila lineage with seeds
0/2 from this Vulcan lineage only after independent reduction. Preserve all
cancelled and failed attempts, and never pool duplicate seed attempts.

See [runtime verification](load-ladder-v2-recovery-runtime.json),
[Killarney verification](load-ladder-v2-recovery-runtime-killarney.json),
[first successful worker observations](load-ladder-v2-recovery-first-progress.json)
and [scheduler/progress receipt](load-ladder-v2-recovery-accounting.json).
W&B histories are initialized offline; cloud delivery for v2 remains pending.
