# Coherent blog and higher-load memory campaign

November 16, 2026. This continuation follows the owner's request to integrate
rather than append a dated blog follow-up, increase memory load beyond four
bits, and execute `cluv sync` before `cluv submit first job.sh`.

## Published article

Website commit `e27f12c` integrates the one-bit and four-bit findings into the
main narrative, removes the separate dated follow-up heading, and describes
capacity-boundary experiments as the next question. The Floquet multiplier
analysis, leading-eigenvalue/singular-value measurement methods, figures and
citation remain. The public article matches the built HTML exactly; see
[publication verification](publication-verification-v5.json).

## Scientific work completed

The frozen [v2 protocol](../../cross-cluster/experiment/multibit-load-ladder-v2.md)
uses independent 4/8/12/16/24/32/48/64-bit fits and three training seeds. It fixes
34,000 maximum optimizer updates per fit, stage-wise competence gates, exact
saved-checkpoint validation replay, raw validation logits and integer counts,
and fresh held-out horizons through 6,144 body steps. Complete hidden-plus-cell
geometry is conditioned on forcing and arm. Failed acquisition, retention loss,
ordering effects and dimension change remain separate outcomes. The experiment
cannot guarantee collapse or establish a universal capacity bound.

Eight focused tests pass locally and in the pinned Rorqual CPU runtime. The
independent launch review accepted the scientific implementation. The final
original resource profile was increased from 8 to 12 GiB for three workers,
with all seven scheduler candidates rechecked. Source `be71bd3` was committed,
pushed and verified on every admitted target before submission.

## Actual execution and startup recovery

The first native race selected Mila **11742215**. Five losing allocations were
cancelled; Tamia was skipped before submission. Seed 1 entered scientific
training. Seeds 0 and 2 failed during NumPy/Torch imports before fitting, with
missing package files. These infrastructure failures are retained and do not
count as model failures. The healthy seed and its source checkout are preserved.
See the [original accounting](../../cross-cluster/submitted-jobs/load-ladder-v2-first-accounting.json)
and [failure observation](../../cross-cluster/submitted-jobs/load-ladder-v2-startup-failures.json).

A separate [operational recovery](../../cross-cluster/submitted-jobs/load-ladder-v2-recovery.md)
uses unchanged scientific code/configuration for seeds 0 and 2, a distinct
source checkout, and runtime directory snapshots outside Cluv's reinstalled
environment. Two exclusive CPU workers request 8 GiB total and three hours.
Independent review finds no material blocker; hardlinked directory snapshots
protect against unlink-based reinstalls but are not protection against in-place
package writes or a failing filesystem. Actual runtime imports are checked on
each admitted candidate before dispatch.

W&B is required for the scientific fits. Their original offline histories must
be synced and remotely verified; preparation and scheduler checks create no
W&B runs. No persistent Qwen service or unobserved all-day research process is
claimed. Scientific results still require independent reduction, and the new
campaign does not yet change the accepted four-bit negative result.

## Confirmed continuation

At 23:44 UTC, Mila **11742215** and Vulcan **2389302** were RUNNING. Mila seed 1
had finished D4/D8 with reported competence passes and entered D12; Vulcan
seeds 0/2 were training D4 beyond 12,000 updates. The recovery Cluv command
exited zero; Fir 4470814 and Killarney 7679777 are CANCELLED, and Tamia was
skipped before submission. Four recovery targets passed runtime verification;
Narval/Rorqual runtime preparation timed out and Nibi's import check remained
unverified. Source and failed preparation are preserved. Exact observations
and the two run lineages are recorded in the
[recovery ledger](../../cross-cluster/submitted-jobs/load-ladder-v2-recovery.md).

The autonomous work is the running, predetermined load ladder. A separate
independent reducer is being reviewed; its completion and use on real output
must be reported separately. No new collapse conclusion is accepted from
these progress observations.
