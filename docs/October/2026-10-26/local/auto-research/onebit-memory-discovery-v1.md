# One-bit retention under repetition: discovery protocol v1

2026-10-20. Prospective seed0 discovery protocol; freeze with executable source
before submission. This implements the [candidate](../../../2026-10-20/local/auto-research/repetition-memory-next-experiment-2026-10-20.md),
with explicit scope reductions below. It is not a confirmation study or result.

## Task and model

WRITE(bit), body, QUERY from zero H and C. Six FP32 input channels are one-hot
WRITE/A/B/X/QUERY and signed bit −1/+1 at WRITE only. No body/QUERY bit, clock,
arm or target channel. Native stepwise LSTMCell(input6,hidden32) and Linear(32,2)
head; score after consuming QUERY. No encoder, dropout or normalization.
Complete persistent state is H32 followed by C32; pre-QUERY capture excludes the
QUERY update. Use this same native cell path in training, evaluation and
interventions: CPU FP32, deterministic algorithms, one intra/inter-op thread.

Motifs are AXAXAXBXBXBX, AXAXBXAXBXBX, AXAXBXBXAXBX, AXBXAXAXBXBX,
AXBXBXAXAXBX. Periodic bodies repeat one uniformly sampled motif R times;
controls draw uniform motif IDs, fix their final motif to the periodic motif,
and reject proper whole-body tilings using `vector_task.paired_body_ids`.
R1 has identical arms and R0 no body. Dwell q repeats each symbol q times.
Arms match counts, bigrams, runs, endpoints and total updates, but higher-order
ordering, predictability and motif diversity may differ. This estimates the
specified ordering intervention, not periodicity isolated from all properties.

## Units and frozen streams

Model Python/NumPy/Torch seeds are0. NumPy default_rng seeds1000,2000,3000
respectively govern training, validation and held-out panels. Record actual
panel hashes and seed mapping with source/environment identity. Validation
panels are generated and hashed before update1; held-out panels are not generated
or opened until all competence gates pass. Separate streams do not guarantee
unique or unseen body strings: sampling is with replacement from a finite
population (only20 matched motif-ID pairs at R2).

Training uses32 independently sampled schedule pairs and a shuffled exactly
balanced vector of16 zeros/16 ones; duplicate each cue/schedule across arms to
obtain batch64. Validation uses512 independently drawn schedule pairs per
R0/R1/R2, rendering both cue polarities and both arms:1024 episodes per arm.
The same fixed validation panel is reused at the declared stage boundaries.
Evaluation uses1024 schedule pairs per condition, both cues and arms:2048
episodes per arm. Each schedule draw is the within-seed unit; its cue flips,
arm copies and recurrent time steps are dependent. Identical draws can recur.
There is one trained seed, not1024 independent models. R0/R1 arm duplicates do
not increase the independent sample count.

## Acquisition, selection and stopping

Adam lr0.001, default betas/epsilon, weight decay0; global gradient clipping1,
full-episode backpropagation, mean cross-entropy at QUERY only. No geometry loss.
At most5000 updates:

1. Updates1–1000: R0.
2. Updates1001–3000: repeat R0,R1,R2, starting at R0.
3. Updates3001–5000: repeat R0,R1,R2,R4,R8, starting at R0.

Stage1 gates R0; stages2/3 gate R0/R1/R2. Every gated horizon, cue and arm must
reach accuracy≥0.95, i.e. at least487/512 correct per cue; compare exact counts,
not rounded percentages. Preserve validation logits/labels and all gate counts
for independent reduction. Identical arms must agree. R4/R8 are training
exposures, not eligibility gates. Only the final checkpoint after all gates
pass is eligible; no retrospective best-checkpoint selection or geometry-based
selection. A boundary failure ends the fit as `competence_failed` without
held-out evaluation, rescue, extended training or automatic confirmation runs.
Nonfinite quantities, integrity failures and numerical-control failures end as
engineering failures, distinct from competence failure. Failed discovery is a
valid outcome, not evidence of harmful repetition.

## Frozen evaluation and contrasts

Primary q1 grid R={2,4,8,16,32} has body lengths24,48,96,192,384; add R0/R1
anchors. Secondary (q,R)=(2,16),(4,8), together with primary(1,32), holds body
length384 fixed while changing both dwell and repeat count. Include WRITE and
QUERY in every episode; do not conflate body length with total updates.
Separate condition panels need not be nested. Freeze checkpoint before testing
and verify its hash unchanged afterward; no test-driven model update.

For schedule i and arm a, let e[i,a] average the two cue-specific error
indicators. Report accuracy separately by arm/cue, balanced cross-entropy, and delta[R]=mean_i(e[i,periodic]−e[i,control]). Primary descriptive
contrast delta16; anchor delta2; DID=delta16−delta2. A positive DID without
positive delta16 is not excess periodic harm at R16. Cross-horizon contrasts
compare panel means rather than paired trajectory changes. Preserve per-schedule
outcomes and report all conditions. No seed-population confidence interval,
p-value, post hoc meaningful-effect threshold or automatic confirmation campaign.
The exactly balanced constant-class reference has accuracy0.5. Independent
RNG streams do not establish generalization to body strings absent in training.

## Complete-state interventions and scientific boundary

Capture complete H+C after WRITE and before QUERY. Compare opposite-cue states
within a fixed schedule/arm using raw L2 distance, maximum absolute difference and exact equality; the
initial summary averages over both arms. Raw logits permit later native QUERY
margin reduction. Preserve raw separation scale. A numerical zero in FP32 is
not proof of erasure at all precision. No tangent spectrum or certified
Floquet analysis is performed here.

Exploratory geometry centers the2048 pre-QUERY H+C states separately per arm,
pooling both cues and all schedule draws, and computes FP64 sample covariance
with denominator N−1, eigenvalues, trace and participation ratio
(sum eigenvalues)^2/sum(squared eigenvalues). Raw H then C coordinates are
unrescaled. These dependent cue pairs do not provide2048 independent units.
Rank ceiling is min(N−1,64); zero trace has undefined PR (null). Reject
nonfinite spectra or eigenvalues below minus
64×eps64×max(N,64)×max(max(abs(eigenvalues)),tiny64); clip only remaining
negative roundoff eigenvalues to zero, preserving unclamped minimum/tolerance.
This pooled geometry is descriptive, not task-information dimension; no
geometry statistic selects weights or establishes collapse.

Evaluate interventions on every evaluation schedule, batched in fixed chunks32:
untouched QUERY, exact self-copy, entire opposite-cue H+C donor, same-cue donor
from the previous schedule in the chunk (circular shift), and zero H+C. Preserve
arm, horizon and cue for same-cue donors. Independent schedule draws can produce
identical forcing strings: call this a same-cue different-draw copy, not a
guaranteed different-body copy. Exact self-copy logits must be byte-identical
to native logits; opposite-cue and same-cue donor logits must also exactly
match their donor native logits. No relaxed tolerance. Opposite-cue donor state is scored both
against donor and recipient labels. With identical QUERY this reproduces the
donor's native readout by construction, demonstrating state dependence but not
a localized mechanism, information-erasure rescue or a novel memory subspace.
Zero-state corruption is not a matched-magnitude sham. Record actual completed
intervention count; no post hoc choice of a favorable subset.

The native readout alone cannot separate erasure from inaccessible/misread
information. Learned linear/nonlinear probes, all-X controls, shuffled-label
readouts, an executed streaming-register oracle, FP64 sensitivity and
perturbation-scale assays are deferred in this discovery execution. The
constant-class reference is0.5 by exact balancing. The implemented register
reference extracts the rendered WRITE sign and compares it with targets; this
checks input-target integrity and should equal1.0. It is not a streaming
implementation with independently tested recurrent register dynamics. No claim
may rely on an unexecuted control.
One bit yields at most a one-dimensional contrast of cue means: this experiment
cannot establish broad dimension or memorization-capacity collapse. Both-arm
loss can be generic forgetting; preserved retention and null/opposite ordering
effects are admissible results. Load K>1 and confirmation seeds require later
protocols, not implicit extensions of this run.

## Execution envelope and acceptance

One cluv Tamia submission, account aip-ioannism, normal QOS/cpubase_bycore_b1:1 CPU,4GiB, zero
GPUs, maximum2h, no retry/requeue/race. Exact worker CLI, immutable source and
protocol hashes, environment, unique scratch run directory, STOP path and
internal deadline are recorded by the outer launcher before submission. Stop
on STOP, deadline, failed competence or integrity; never schedule follow-up work.
Offline W&B actual history and completed reporting receipt are required; online
delivery is unknown until verified. Checkpoint/state payloads stay on scratch.

Independently verify scheduler outcome, source/panel/checkpoint hashes, actual
update counts, all stage gates, episode chronology, paired reductions and
reporting history before scientific acceptance. A Slurm exit0 is insufficient.
Publish failed outcomes and scope reductions alongside successes. Distinguish
engineering validity, task-retention findings and causal/mechanistic claims.

Prelaunch execution amendment: Mila authentication is unavailable. Tamia access
is verified; use a dedicated hash-locked CPU Torch2.0.0+cpu/NumPy1.24.4/W&B0.30.0
environment on project scratch, with exact interpreter/package/lock identity
recorded before submission. This changes the execution host before any run,
not the scientific protocol or a failed experiment outcome.
