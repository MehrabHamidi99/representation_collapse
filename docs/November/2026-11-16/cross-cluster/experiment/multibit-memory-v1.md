# Higher-dimensional memory under repetition — discovery v1

Prospective protocol, 2026-11-16. The owner requests higher-dimensional inputs,
explicitly authorizes follow-up experiments and `cluv sync` / `cluv submit first
job.sh` across available Alliance clusters. Freeze this protocol, configuration,
worker, environment lock and shell script in Git before dispatch. This is a new
experiment, not a reinterpretation of the accepted one-bit result.

## Question and scope

At a fixed trained checkpoint and equal retention horizon, does periodic ordering
cause more loss of a previously written vector than a matched varying-order body?
Does increasing independent memory load expose a decrease in useful retained
dimension? Accept preservation, generic forgetting, failed acquisition and
opposite ordering effects as possible outcomes. Graph prediction deterioration
motivates the study but is not a presumed mechanism or guaranteed outcome.

The first campaign fits seed 0 separately at D=1,4,16,64 independent binary input
coordinates. All models have the same architecture: 69 input channels, native
LSTM hidden size 64 (complete H+C state dimension 128), and 64 output logits.
The loss and scores use only the first D outputs. Single WRITE presents the D
signed values; inactive coordinates are zero. BODY and QUERY carry no payload,
labels, arm indicator or clock. The five type channels encode WRITE/A/B/X/QUERY.
QUERY requests the entire vector. This is simultaneous vector recall; late
addressed queries and continuous analog inputs remain later experiments.

Loss is averaged across active bits, so per-coordinate supervision changes
with D. Load comparisons describe this acquisition recipe; they do not establish
a capacity threshold. Per-coordinate input magnitude is fixed at one, so total WRITE energy grows
with D. This is a declared load manipulation, not an energy-matched comparison.
Separate fits across D cannot isolate dynamics of one checkpoint across loads.
The D=1 fit is a new anchor with hidden size64, not an exact repeat of the older
hidden32 one-bit experiment. Binary D64 comprises 64 independent source bits,
not a continuously valued 64-dimensional memory manifold.

## Matched forcing and acquisition

Reuse `vector_task.paired_body_ids` and `body_from_ids`: five length12 motifs
with matched counts, transitions, run lengths and boundary context. Periodic
repeats one motif; controls vary motif order with a matching terminal motif,
rejecting proper whole-body tilings for R>1. R0 has no body, R1 identical arms.
Dwell q repeats each symbol q times; H=12Rq is the body horizon. Higher-order
predictability and motif diversity also change: interpret the specified
ordering intervention, not an isolated universal periodicity treatment.

Training batch: 32 cue vectors duplicated across arms (64 rows), independently
balanced bit columns, same cue within each pair. Adam lr0.001, no weight decay,
mean active-bit BCE, global gradient clipping1, complete-episode backpropagation.
CPU FP32, deterministic algorithms, one intra/inter-op thread, no dropout,
auxiliary geometry loss, test-driven tuning or checkpoint selection.

Fixed stages per fit:

1. 2,000 updates at R0; validate R0.
2. 4,000 updates cycling R0/R1/R2; validate all three.
3. 4,000 updates cycling R0/R1/R2/R4/R8; validate all five.

Each required bit and forcing arm must reach accuracy >=0.95 at every required
validation horizon. Preserve correct counts and denominators; do not round
before comparison. Gate is bitwise, not whole-vector accuracy; report whole-vector
accuracy separately, since these differ at large D. Validation uses256 balanced
cue vectors per panel and a separate fixed RNG stream. Record actual stream
seeds and panel identities. Failure stops that fit without held-out evaluation;
other preregistered loads may still run. Save failed endpoints and diagnostics.
Only the endpoint after every gate passes is eligible. No rescue or extra
training follows from a failed boundary in this campaign.

## Held-out measurements

Only after competence, draw the independent test cue panel:256 vectors, balanced
within each bit. Reuse this panel across every frozen test condition. Four
independent forcing draws per condition are each shared across all256 cues;
both arms receive the same cue. Save body/panel identities and raw arrays.
Test conditions (R,q) are (0,1),(1,1),(2,1),(4,1),(8,1),(16,1),(32,1),(64,1),
(128,1),(32,4). The last condition and (128,1) both have H=1536; comparisons
across different R at q1 also change elapsed time. No weights change at test.

Report each bit's accuracy, mean bit error, whole-vector exact-match accuracy,
paired periodic-minus-control bit error, per-forcing outcomes, oracle extraction
of WRITE and zero-state QUERY scores. Primary descriptive contrast is the paired
error difference at R32q1; report R2q1 anchor and their difference in differences.
Do not select a favorable horizon afterward. Positive difference in differences
without positive primary harm is not excess periodic harm at the primary horizon.

Capture complete H64 then C64 after WRITE and immediately before QUERY. Center
the256 cue states within each forcing draw and arm separately; compute FP64
covariance eigenvalues, trace, participation ratio and explicit sample/rank
ceilings. Do not pool forcing or phase means into a memory dimension estimate.
Rank ceiling is min(255,128,2^D-1); for small D repeated cue vectors cannot
increase it. A nonlinear embedding of D bits can exceed D in covariance rank;
PR is not bits stored. Preserve undefined zero-variance PR as null, roundoff
tolerances and unclamped spectral minima. Absolute scale and error accompany PR.

Cue copies across arms, bits within a vector, repeated panels across horizons,
and time steps are dependent. Four forcing draws and one trained seed are not
population replication. This discovery reports exact descriptive results and
does not claim a seed-population confidence interval. Confirmation seeds1/2,
decoder recovery probes, continuous-valued cues, energy controls, causal
directional interventions and complete-state tangent analysis are follow-ups.
Native readout failure alone does not establish irrecoverable information loss.
No Floquet spectrum, basin certification or harmful collapse acceptance is
automatic. Require independent raw-array reduction before scientific acceptance.

## Execution and stopping

Use the isolated `launch/multibit-20261116` uv project with locked CPU Torch2.0.0,
NumPy1.24.4 and W&B0.30.0. Cluv sync stages Git source and installs dependencies
on login nodes under scratch; jobs never download packages. Verify actual
accounts/partitions, current owned jobs and scheduler admission before the race.
Preserve unrelated jobs and dirty historical project checkouts.

One race, at most one candidate per configured cluster: Mila plus Tamia, Narval,
Rorqual, Fir, Nibi, Trillium, Trillium-GPU, Killarney and Vulcan. Include only
currently connected targets whose scheduler accepts the requested CPU resources;
record unavailable/ineligible candidates rather than claiming they ran. Each
candidate requests one CPU,8GiB,zero GPUs,three hours. This small recurrent workload
uses CPU following the measured earlier one-bit CPU runs. No idle GPU request is
added to make a GPU-only target eligible. Cluv cancels race losers; independently
reconcile their IDs and terminal state. No retry, array, requeue or dependent
campaign is part of this initial launch. Worst case ten overlapping accepted
candidates is30 requested CPU-hours before cancellation; record actual allocated
TRES if scheduler policy enlarges a request.

Run IDs, source SHA, exact command, all candidate job IDs, scheduler outcomes and
outputs belong in the dated submission ledger. Outputs are under each target's
`$SCRATCH/representation-collapse/runs/multibit-memory/RUN_ID`. STOP is the sibling
`RUN_ID.STOP`. Worker checks STOP/deadline between bounded chunks; internal work
deadline is10500seconds, outer timeout10620seconds, Slurm10800seconds. A failed
fit is distinct from interrupted/incomplete work and an engineering error.

W&B required: true — monitored training and scientific evaluation. Record live
loss/update counts, competence, final interpretable metrics and durable offline
history from work start. Project `representation-collapse`; cloud delivery remains
pending until verified. No datasets, state arrays or model weights are uploaded
by the reporting helper. Artifacts remain on scratch; compact verified summaries
and source/protocol/environment identities return to Git.

## Candidate verification before staging

Read-only scheduler tests on November16 accept the one-CPU request on Mila,
Tamia, Rorqual, Fir, Nibi, Killarney and Vulcan. Killarney routes CPU-only work
to its L40S partition without a GPU request; Vulcan selects the partition from
resources. These seven are the first race's eligible candidates (maximum21
requested CPU-hours before race cancellation). Narval's direct probe timed out
and the Tamia route rejected batch authentication. Trillium accepts only a
whole192-core CPU node; Trillium-GPU requires a GPU. Sync reachable source and
environment to these latter systems but disable their candidates in this
launch project's local cluv cache. Do not modify another project's cluv state.
Numbers printed by `sbatch --test-only` are estimates, not submitted job IDs.
