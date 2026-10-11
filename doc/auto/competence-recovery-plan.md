# Competence-first recovery of vector memory training

Date: 2026-10-11. Status: proposed diagnostic protocol; no new training or
submission is authorized by this document. Preserve the frozen v1 runs,
checkpoints, manifests and failed competence outcomes. This is a new training
protocol, not a repair of their reported results.

The initial acquisition concern is now supported by **all four artifact-verified
full-budget seed-0 GRU/LSTM × categorical/continuous runs**, source commit
`9004a0c114a8b7c5335dca3d5247449fd10b5ffa`. Narval GRU jobs `5234860` and
`5235508`, and Rorqual LSTM jobs `22980985` and `22980998`, all completed with
ExitCode `0:0` in the
[persisted scheduler snapshot](../../runs/ops/phase1-status-20261011.json).
The [final verifier](../../runs/ops/phase1-final-verification-20261011.json)
independently validates saved artifacts and raw-output reductions for all four
GPU pilots plus two synthetic calibration directories (6/6 directories, not
six trained models). Verification JSON SHA-256:
`4a71636fa22baba0cc4b20c1b0b8ff0ca339278e159854fe7966b5159e01a4a0`.
Verifier SHA-256:
`5f092ee5b87c5e407b5271eee97d5a59c66264570d2862df3cc672de0cb8d22b`.

All four selected checkpoints fail even the frozen engineering competence
thresholds at R1/R2. Both categorical cells have 5.47%/6.25% accuracy and
DID zero. Continuous GRU R1/R2 MSE is 0.35293/0.36423; LSTM is
0.35247/0.36561. Their raw-recomputed DID values are approximately `-9.99e-8`
and `-5.91e-8`. Saved-target zero-predictor baselines put native/zero MSE ratios
near one (0.99766–1.01810). Near-zero contrasts at failed acquisition do not
identify a harmful-collapse mechanism. Uniform categorical prediction has
population cross-entropy `log(16)=2.772588722`; near-chance validation loss
fits this acquisition concern without locating its cause.

Stored-state precision audits place much residual variation near float32
spacing. R2 standardized probes nevertheless outperform shuffled labels in
both architectures/tasks, so lack of all cue information is not established.
Higher-precision dynamical replay and noise robustness remain untested. The
LSTMs' actual 40 GiB H100 MIG slices are recorded allocation deviations on
the same job IDs, not extra submissions. See the
[independent scientific review](initial-phase-scientific-review.md) for the
complete four-cell behavior, precision flags and probe counterevidence.

## Diagnosis supported by the frozen source

The model must learn addressed writing, retention and query selection jointly.
The sixteen WRITE observations contain eight required values first, followed by
eight nuisance values. The selector appears only at the terminal QUERY. Even
with zero distractors, each queried value is followed by 9–16 recurrent updates
before its answer is read. These distances become 21–28 at R1, 33–40 at R2,
and 201–208 at R16, including the query update.

The frozen curriculum cycles R1, R2, R4, R8 and R16 immediately; it never trains
the complete zero-body task. Each of 1,500 updates supervises one randomly
queried value per episode. Batch size 32 consists of 16 paired prefix/query
examples, giving 4,800 distinct prefix/query examples per horizon and about
600 queried examples per coordinate per horizon in expectation. Counts are
not training replicates. Balanced long-horizon exposure is appropriate for
the comparison, but does not establish that acquisition succeeded.

The jointly trained encoder maps 38 input channels to 64 features with tanh;
64 is larger than the raw input dimension, so its name does not establish a
dimensional bottleneck. Saturation, a learned deficient encoding, weak credit
assignment and an ineffective query/readout are hypotheses to test, not
observed causes. Source review found no answer leakage or omitted LSTM cell
state. References: [task renderer](../../src/representation_collapse/vector_task.py),
[model and training loop](../../src/representation_collapse/pilot.py), and
[frozen v1 configuration](../../configs/experiments/vector-pilot-v1.json).

## Diagnostic ladder

Freeze a separately named task/training configuration and record every variant.
Keep 16 equiprobable categories or continuous Uniform[-1,1] values, the original
38 input channels, query indices 0–7, and the native GRU/LSTM query step. Never
supply the selected coordinate during the full WRITE prefix.

1. **Renderer and streaming oracle.** Decode each WRITE's coordinate and value
   into eight explicit registers; ignore distractors and nuisance writes; look
   up the QUERY address. Require exact categorical recovery and continuous
   error below 1e-12 on bounded CPU fixtures for zero body and R1/R2/R16. This
   oracle is an explicit-memory reference with counted storage, not evidence
   that the learned recurrent architecture works. Compare with uniform
   categorical prediction and continuous prediction zero. The latter has
   population MSE 1/3 and normalized MSE one. Shuffled memories must destroy
   oracle agreement with the original targets; shuffled query indices must
   select the changed coordinate rather than the old answer.
2. **Single WRITE, immediate QUERY.** Use one required value followed by its
   matching QUERY, with no distractor or nuisance. This two-update task requires
   a separate diagnostic constructor; the original renderer always writes
   sixteen coordinates. Check a fixed small-batch overfit and then untouched
   IID value generalization, with gradients and actual parameter changes.
   Failure here localizes value encoding, target/loss, optimization or native
   query handling before the original storage demand. Success establishes only
   this simplified task, not eight-coordinate memory.
3. **All sixteen WRITEs, zero body.** Use the original renderer with empty
   body strings and the original hidden QUERY selector. `render` supports this
   17-observation episode; `paired_body_ids` and the frozen v1 validator do not
   support R0. Add R0 only in the separately versioned diagnostic configuration.
   Require acquisition of all eight coordinates, including those written
   earliest, despite the later nuisance writes. A model that passes step 2 but
   fails here has not demonstrated original-task acquisition.
4. **R1 then R2 competence.** Add distractors only after zero-body acquisition.
   Preserve balanced matched arms and the shared R1 anchor. Inspect losses and
   coordinate-specific performance at every gate; never replace this with a
   global score that two retained coordinates could satisfy.
5. **Long-horizon exposure.** Introduce R4/R8/R16 while preserving short-horizon
   rehearsal. Compare with the unchanged v1 training as an exploratory
   optimization comparison, acknowledging the different curriculum and budget.
   Keep the selected weights fixed for subsequent memory evaluation.

A bounded local source check already verified that 128 zero-body episodes for
each task render with shape `(128,17,38)` and the explicit streaming register
oracle recovers every target exactly. This verifies the zero-body renderer and
oracle only; no learned model was trained by this reviewer. Full diagnostic
fixtures, including corruption controls, remain to be implemented.

## Native-state localization

For the same held-out memory and fixed driver, capture the complete native
state after the WRITE prefix, immediately before QUERY at R0/R1/R2/R16, and
immediately after QUERY. For LSTM retain both h and c, and also report their
separate measurements. Fork each query evaluation from the same pre-query
state; include the query update before applying the original task head. Verify
forked evaluation against uninterrupted native forward execution. Never use a
decoder's access to labels or original cues as a substitute for the native
model's memory.

Fit train-only probes to fresh cue clouds at these phases. A strong prefix
probe followed by weak pre-query recovery motivates a retention hypothesis;
strong pre-query recovery with weak native behavior motivates a query/head
hypothesis. Linear probe failure cannot certify information absence. Native
cue interventions, nonlinear probes and noise-resolved tests remain separate
follow-up evidence. Report total variance as well as normalized geometry;
standardized probes can hide amplitude decay.

Record gradient norms at queried WRITE observations, per-module parameter
gradient/update norms, clipping frequency, and activation/gate saturation on
fixed diagnostic examples. Nonzero gradients alone do not prove acquisition,
and small loss gradients can be appropriate after correct prediction. If the
encoder cannot support held-out latent-observation decoding, test encoder
pretraining in an explicitly labeled branch. If state probes succeed but the
query fails, test an independent query-conditioned decoder in a labeled
readout diagnostic. Neither alteration can be silently folded into the native
v1 result. Auxiliary prefix-memory supervision is likewise a separate
objective, with its supervision and budget recorded.

## Frozen discovery protocol and stopping rules

The proposed first discovery grid retains hidden size 128, encoder size 64,
AdamW, weight decay 0.01, clipping 1.0 and batch size 32. Freeze learning rates
`[0.0005, 0.001, 0.003]` before inspecting new outcomes. Use discovery seed 0
for each of four model/task cells: at most twelve fits. Each fit has this
fixed maximum schedule:

- 500 updates: single-WRITE immediate-query diagnostic.
- 500 updates: full-prefix zero body.
- 500 updates: deterministic equal R0/R1 exposure.
- 500 updates: deterministic equal R0/R1/R2 exposure.
- 1,500 updates: deterministic equal R1/R2/R4/R8/R16 exposure, balanced arms.

Evaluate fixed validation manifests every 100 updates; stage transitions occur
only at the declared boundaries and only if the preceding competence gate
passes. No extra steps or relaxed thresholds after a failure. This yields at
most 3,500 updates per fit, 42,000 across the grid, plus fixed diagnostic
evaluation costs. These are proposed ceilings, not approved compute. Derive
the concrete runtime/allocation envelope from the verified v1 runs before
requesting execution. Stop a numerical failure immediately and retain it.

Use 128 independent validation episodes per query coordinate, per evaluated
horizon/arm. Define continuous normalized MSE as `MSE/(1/3)` using the fixed
population variance, not the variance of the observed test sample. Require
categorical accuracy at least 0.95, or continuous normalized MSE at most 0.05,
on **each** required coordinate for R0, R1 and R2 in both arms. The continuous
threshold is raw MSE at most 1/60. The engineering v1 thresholds of 0.25
accuracy/raw MSE cannot substitute for these gates.

For every candidate, save best stage-specific checkpoints as diagnostics.
Select the final eligible candidate by minimum balanced validation task loss
over the full trained horizons, retaining the earliest checkpoint on ties;
eligibility requires the short-horizon gates at that checkpoint. Select learning
rate using the same rule, breaking remaining ties by the frozen grid order.
Neither covariance rank, DID nor the sign of an arm contrast enters selection.
If no candidate qualifies, report acquisition/optimization unresolved and stop
this grid. Propose a new diagnosed branch rather than expanding it implicitly.

After discovery, freeze the chosen protocol before rerunning fresh seeds 1 and
2 for feasibility variability; those remain pilots, not confirmation. If
scientific confirmation becomes justified, use ten untouched seed blocks and
test manifests under a separately frozen effect-size/uncertainty protocol.
The old seed-0 results and recovery grid are discovery evidence throughout.

## Long-horizon interpretation and evidence release

Evaluate R16 with the **same** coordinate-wise thresholds, reporting both arms
whether they pass or fail. Short-horizon competence is the prerequisite for
calling deterioration a retention problem. Long-horizon failure in both arms
can be shared horizon forgetting or unresolved long-horizon optimization. If
the R16 control is competent and the periodic arm fails, a relative periodic
penalty becomes an actionable hypothesis; it still needs replicated frozen
evaluation and mechanism controls. Do not require the periodic arm to pass in
order to retain a forgetting observation, and do not select runs for failing it.

If both long-horizon arms are competent, retain that valid outcome. If both
fail, retain it without declaring a universal recurrent-memory limitation.
Always report delta2, delta16 and DID separately: zero DID at chance performance
is uninformative about a harmful-collapse mechanism. Categorical attractor
memory and useful continuous compression remain valid alternatives. This
protocol measures finite-time retention; Floquet terminology requires separate
state/forcing closure and tangent verification.

Release competence evidence only with frozen/effective configuration, source
and environment hashes, native checkpoint, complete-state chronology checks,
raw coordinate-stratified errors, validation history, failed branches, and
independent artifact verification. All four existing v1 results have now been verified and interpreted. The
next proposed step is bounded competence discovery; implementation and
execution of this recovery protocol require their own recorded handoff and
compute envelope.
