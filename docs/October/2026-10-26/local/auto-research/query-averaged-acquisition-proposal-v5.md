# Query-averaged acquisition v5 — prospective proposal

2026-10-13. **Proposal and tiny numerical calibration only. No v5 production
module, committed configuration, execution freeze or acquisition fit exists.**
The coordinator owns admission and scientific acceptance. The calibration below
checks an unclipped-gradient identity at fixed untrained weights; it cannot
establish learned competence, retention, a mechanism or a periodicity effect.

## Question and reason to change the estimator

The [v3 random-QUERY experiment](../../../2026-10-13/local/auto-research/prefix-acquisition-v3-results-2026-10-13.md)
failed complete eight-coordinate R0 acquisition. The completed
[v4 singleton experiment](../../../2026-10-13/local/auto-research/address-acquisition-v4-results-2026-10-13.md) reports
eight endpoint passes at addresses 7/15 and four failures at address 0 after
fixed 5,000-update fits. V4 changed task-required load and future-selector
uncertainty together; its separate successful weights cannot supply original
eight-coordinate eligibility. This proposal preserves the original task and
asks whether removing conditional QUERY-sampling noise from the training
gradient helps acquisition under a separately declared fixed schedule.

For every training prefix, average the loss of **eight independent alternative
terminal QUERY continuations**, one for each required address 0–7, from its
shared complete post-WRITE state. They are independent *branches*, not eight
independent data histories or eight QUERY updates along one trajectory. Each
branch remains a native sixteen-WRITE plus one-QUERY episode. The evaluation
task still reveals a randomly selected required address only at its terminal
QUERY. All sixteen original payloads remain in the WRITE prefix; addresses
8–15 remain nuisance. No body, new head, extra input labels, external memory,
architecture change, gradient truncation or predictable singleton task enters.

The operational question is whether the new fixed endpoint passes **every**
original coordinate gate. This is a change in the gradient estimator and finite
supervision exposure. It does not guarantee success or isolate architecture,
capacity, overwriting, addressing, lag or credit assignment as a pure cause.

## Exact objective and gradient identity

Let a history `P` contain sixteen ordered WRITE observations and their original
values. For fixed parameters `θ`, write `Sθ(P)` for the complete state obtained
from zero native initial state after all sixteen updates. For a GRU this is
`h`; for an LSTM this is the ordered pair `(h,c)`. Let `Qq` be an ordinary
value-free QUERY observation at address `q`, and let

`fθ(P,q) = headθ(native_queryθ(Sθ(P), encodeθ(Qq)))`.

The target `yq(P)` is the original addressed WRITE value. Use the existing
sixteen-class cross-entropy, or **raw** continuous squared error; continuous
population normalization by `1/3` belongs to evaluation, not training. With
uniform QUERY selection independent of the prefix,

`E[q | P] ℓ(fθ(P,q), yq(P)) = (1/8) Σq=0..7 ℓ(fθ(P,q), yq(P))`.

This is a finite sum. Differentiating the complete differentiable computation
at fixed `θ` gives

`E[q | P] ∇θ ℓ(fθ(P,q), yq(P)) = (1/8) Σq=0..7 ∇θ ℓ(fθ(P,q), yq(P))`.

The right-hand gradients may each be computed by a separate native full
17-update forward trajectory. Sharing the sixteen-update prefix is only common
subexpression reuse: the reverse pass sums the eight QUERY contributions into
the same prefix graph, then propagates through all sixteen WRITEs. It includes
all encoder, recurrent-core and query-head parameters. Detaching the prefix,
dropping LSTM `c`, mutating the fork origin or feeding one branch's QUERY state
into the next breaks this identity. The existing native capture helper is
decorated `no_grad` and copies detached states; it must **not** supply training
fork states.

For batch `B=32`, the loss is exactly

`Laverage = (1/8) Σq [(1/32) Σi ℓ(fθ(Pi,q), yiq)]`.

There are sixteen independent histories duplicated into identical paired R0
arms. The original estimator samples one QUERY per independent history and
duplicates it for its two arms. Linearity gives the same conditional mean as
the formula above, even though the eight reference batches each use a common
address. The full normalization is `1/(32*8)` over arm-instance/query losses.
The two identical arm copies do not double effective sample size. Averaging
does not change the population expected task loss at fixed weights; it does
change the realized per-batch loss and removes the sampled-selector noise.

Under finite second moments, for the *unclipped* gradient at fixed weights,
`Cov(gsample) = Cov(E[gsample | P]) + E[Cov(gsample | P)]`.
Thus the conditional-mean estimator removes that nonnegative conditional-noise
term. This statement assumes the same prefix law, deterministic native model
and uniform QUERY law. It says nothing about the covariance of clipped updates
or competence after optimization.

Clip **once after** averaging/backpropagation, then make **one** AdamW update.
Clipping and adaptive optimizer updates are nonlinear: generally
`clip(mean gq) != mean clip(gq)` and
`AdamW(mean gq) != mean AdamW(gq)`. For scalar gradients 2 and 0, ideal unit
clipping gives 1 for the mean versus 0.5 for the mean of the clipped gradients.
At a zero-parameter, fresh first AdamW step with epsilon `1e-8`, the normalized
directions are approximately 1 versus 0.5 as well. A succession of eight
optimizer steps is a different experiment. Neither update parity nor equal
optimization trajectories follows from the expected-loss identity.

## Proposed fits, exposure and stopping rule

Prepare four **new fresh seed-0** fits, in declared order: GRU categorical,
GRU continuous, LSTM categorical, LSTM continuous. Start from the exact pinned
v3 fresh initial parameter tensors, verify their hashes and tensor equality,
and initialize empty AdamW state. Architecture remains hidden 128/encoder 64;
one CPU thread, native Torch FP32, deterministic algorithms, TF32 off and
highest matmul precision. Batch stays 32 paired instances/16 independent prefix
histories. AdamW stays LR `0.001`, weight decay `0.01`, gradient clipping `1`.

Each fit receives exactly 5,000 new optimizer updates, evaluates at
500/1,000/…/5,000, and retains every scheduled checkpoint and current endpoint.
The proposed envelope is four fits, **20,000 total optimizer updates and 1,200
seconds overall** from engine preparation through baseline reevaluation,
training and evaluation. No early-success stopping, best-checkpoint selection,
replacement, extension or resume is granted. Nonfinite values, identity changes
or the deadline stop execution; preserve realized updates/current weights and
mark censored endpoint competence unknown. Passing an intermediate gate is
descriptive and cannot replace the step-5,000 endpoint.

This is **equal optimizer-update and prefix-history exposure**, not equal
QUERY-loss exposure, walltime or compute. Each averaged update evaluates 256
arm-instance/query losses instead of 32: 128 distinct history/query combinations
instead of sixteen sampled combinations. Over four fits this is 5,120,000
arm-instance/query loss evaluations versus 640,000 for the four v3 fresh fits.
Both process 320,000 prefix-history instances: 80,000 per fit. The task-specific
streams are shared across architectures, so there are 160,000 distinct
task/step/history draw IDs across the four fits, not 320,000 independent draws
or additional independent training seeds.

The shared fork nominally executes `32*(16+8)=768` native recurrent row-updates
per optimizer step, compared with `32*17=544` in v3, a ratio `24/17`. Eight
separate full trajectories would execute 4,352 row-updates. These are operation
counts, not FLOPs or throughput guarantees: query-head work, eight recurrent
calls, autograd/storage overhead and kernel splitting also differ. Record
actual per-fit CPU walltime, useful updates/second, branch-loss counts and
recurrence counts. No GPU allocation or empirical performance claim follows
from this CPU fixture.

## RNG matching and immutable references

Regenerate the original v3 task-specific base stream using seed `2026101301`
and `paired_batch_with_zero(rng,16,0,task)` at every step. Consume **all** its
original RNG draws, including the random QUERY selectors, even though those
selectors are unused in the averaged loss. Do not skip or replace the selector
draws: doing so changes later WRITE payloads. Verify all original base batch
and rolling hashes against the existing v3 5,000-update manifests. The sixteen
WRITE rows must be byte-identical. Derive eight value-free one-hot QUERY rows
deterministically, with targets from the original memory/payload rows, and
persist a distinct all-query target/continuation manifest before fitting.
Preserve the original sampled-query metadata for provenance; do not silently
reuse the old sampled target as the eight-query target array.

Model/data source is pinned to v3
`999d88d8679c0f353c6bb7fb4f5ab8433a7e730c`. Current fixture imports of
`pilot.py`, `diagnostics.py` and `vector_task.py` were byte-compared with that
preserved v3 source and agree. The new training module/configuration would have
their own committed source and independent-review pins before execution.

Baseline files are the **fresh** v3 current step-5,000 endpoints, never transfer,
historical-best or newly optimized weights. For each model/task the files are
under `runs/prefix-acquisition-v3-20261013/experiment/<model>-<task>-fresh_seed0/`:

- GRU categorical: `checkpoint-step-005000.pt` SHA-256
  `710e470c8a3c55ba4db993f108cacb362a19198721c05d7a81d0d3578cfb23fe`;
  `initial-checkpoint.pt` SHA-256
  `1d865f50723ee4388a06a7c16937cbb5929e5dfefd19f3afb6b146d220e03904`.
- GRU continuous: endpoint
  `0cb44d7e7b143fc18c037c0a22a4e4e38b5a4d803307666a3266784c4a448b96`;
  initial `ea2e3c827b44991d23709240a986091ecf28b256a2d98b59ad84913dca82ea79`.
- LSTM categorical: endpoint
  `47c44d1d0fa0bfdd7f88e750bca750bc6d229c21f4c8cd63d1e4f73c4005a9ca`;
  initial `c95e01cf440bf24cb4016093f949a4f7a8e1b165ba27c8edd4739504e61cc57e`.
- LSTM continuous: endpoint
  `31a8ef0449284f3c072b68ead74fea73ae16fe16bd9f871dbe75cf95e602b693`;
  initial `c31ae7cab4a5dc67847b95b291a5e6e83b4c261cd8766a30843c411c204babf6`.

These eight file hashes were checked locally during proposal preparation.
Adopt the original provenance/metric/stream pins from the v4 configuration's
`v3_fresh_references` and `v3_base_training_streams` into the eventual v5
configuration, checking them again rather than assuming availability.

## Fresh validation and paired conditional errors

Propose new validation sampling seed **20005**, distinct from v3 seed 20001
and v4 seed 20003. Reuse the existing R0 coordinate-stratified constructor:
128 independent full-prefix histories per required coordinate per arm, eight
strata, 1,024 independent episode pairs/2,048 arm instances. Histories are
sampled separately across coordinate strata; the same manifest is used for
both architectures, every scheduled checkpoint and its pinned baseline. This
choice preserves the existing scorer and sample geometry. It does not turn
the two arms, coordinates or models into independent trained replications.

Persist input/target/query/pair-ID arrays and hashes before optimizer step 0.
Reevaluate every pinned baseline on **those exact new full histories and all
eight supported terminal selectors**, without optimization. Native terminal
QUERY predictions are the behavioral reference. Retain full post-WRITE,
pre-QUERY and post-QUERY `h` or `(h,c)` with independent complete-state
fork-versus-uninterrupted replay. Use the unmodified native-coordinate scorer:
all eight accuracies must be ≥0.95 for categorical tasks; all eight
MSE/(1/3) values must be ≤0.05 for continuous tasks, with the existing fixed
four-ULP float64 threshold allowance. No coordinate omission, average-score
substitution, weakened gate or q15 evaluation is allowed.

At the fixed endpoint, retain per-episode paired native-error differences
`averaged-trained minus pinned sampled-query-trained`: categorical mismatch
indicator differences, or raw squared-error differences divided by `1/3`.
Positive means worse. Reduce in float64 and report every coordinate, its
128-pair count and descriptive mean. The pairing concerns new evaluation
histories; old seed-20001/20003 scores remain historical and are never pooled
or used as an unpaired primary contrast. Baseline reevaluation adds zero fits
and zero optimizer updates. It is a schedule comparison with unequal QUERY
supervision exposure, not a rescue or exposure-equivalence test.

There is only one independent initialization/discovery seed. No confidence
interval, p-value, equivalence decision, population competence guarantee,
periodicity body, covariance/tangent result or Floquet claim is enabled.
Even full R0 success releases only an acquisition prerequisite: later body
training, forcing/chronology/numerical controls and confirmation remain separate.

## Completed tiny native calibration

Ignored artifacts: `runs/query-averaging-calibration-20261013/`. This is a
standalone fixture and verifier, not production implementation. Native Torch
2.6.0 on CPU/one thread, hidden 5/encoder 4, three independent prefixes duplicated
into six arm instances; fixture initialization seeds 0/17; GRU/LSTM ×
categorical/continuous × FP32/FP64: **16 cases, zero optimizer updates**.
Each FP64 case uses the same exactly representable FP32-generated initial
weights and input payloads as its FP32 counterpart, then evaluates in FP64.

The fixture compares the differentiable shared-prefix fork against the mean
of eight separately generated full native 17-update gradients. It preserves
all eight query logits, complete post-QUERY states, losses, individual full
native gradients, averaged gradients and the parameter ordering/shapes. Every
encoder/core/head parameter tensor has a nonzero reference gradient and is
checked separately. Predeclared elementwise tolerances are
`atol=1e-6, rtol=1e-5` for FP32 and `atol=1e-12, rtol=1e-10` for FP64.

All comparisons pass. Maximum absolute full-gradient error is
**2.9802322387695312e-08 (FP32)** and **2.7755575615628914e-17 (FP64)**.
Maximum logit difference is `2.9802322387695312e-08` and
`1.1102230246251565e-16`, respectively. Full-native FP32 versus FP64 gradient
error is at most `3.61830024275811e-08`. Agreement is within declared tolerance,
not a promise of bitwise production identity.

A separately written verifier checks **336** raw-array/target/normalization/
gradient conditions. It independently computes cross-entropy/MSE from saved
logits and uses scalar central finite differences of the eight full native
losses for every parameter tensor in all eight FP64 cases: **64 checks**, step
`1e-5`, tolerance `2e-9 + 2e-7*abs(reference)`, maximum absolute derivative
error **5.276529560505061e-11**. This verifier is separate deterministic code
by the same implementation agent, not independent scientific review.

Negative controls detect a detached prefix, sequential QUERY chaining, omitted
LSTM cell and missing `1/8` normalization. Their smallest observed gradient
discrepancies are respectively `0.0042526`, `0.0034920`, `0.0159115` and
`0.6335757`; missing normalization multiplies loss by eight. An analytic scalar
clipping/AdamW noncommutation counterexample is included. These are intentionally
wrong fixture branches only and never enter model fitting.

Reproduction commands, each with a fresh create-only artifact directory or
preserved original output:

```bash
PYTHONPATH=src .venv/bin/python runs/query-averaging-calibration-20261013/calibrate.py
PYTHONPATH=src .venv/bin/python runs/query-averaging-calibration-20261013/verify.py
```

The scripts reject existing result/array outputs; copy the scripts into a new
sibling directory beneath `runs/` to reproduce without replacing this evidence.
The calibration's original `manifest.json` predates the verifier; the final
`complete-manifest.json` binds both scripts and all numerical artifacts.

Key SHA-256 identities:

- Calibration script: `165e6869cb6d992d4bb29f1a32ec7f82f9a3a3bfd68e49d334824f8ffdce5bda`.
- Verifier: `044374af4665f2f717b942e3b773fb6ce06fd2ff97e4ee7ff2ef5f01e19cf432`.
- Numerical results: `90e67e219560d372f2be3d2ce5cefa828b10fb2fab8248355c6d0ffea772264b`.
- Verification receipt: `0c822e749fa255093b42759b4cdabe828cbe2a81cb6d7ff5f623a48f78541c91`.
- Native model source: `0aa1c337ffd0ab576056e70b4ecb0dfeee073c1d3453e87389447f77f1dfbbd3`.
- Task renderer: `df3843330ac9c59205823c8d55868e7878e903c0065c5af8bc89d1308243a297`.
- R0 constructor: `076ef9d9600f4b73bbf2b92d4ead7802452a51fd3492b55828c34f2eaf6027c4`.
- Environment lock: `588147aa66724c075fd44754db3ddf757b249eefcaf94c9154ddc49578c016dc`.

## Smallest implementation and remaining admission gaps

Implement one additive, separately named `query_acquisition.py` module and v5
configuration, preserving old modules/guards and the native model class. A small
training helper can call the existing differentiable `pre_query_state(inputs)`
once, then `read_query(query,state)` eight times from that unchanged complete
state, average the eight standard batch losses and backpropagate once. Prefer
an explicit eight-branch loop first; vectorized state expansion is optional
later work that needs its own native gradient and ordering checks. Do not
introduce cell equations, an adapter model, global optimizer changes, training
through detached diagnostic captures or an external memory buffer.

Reuse the validated model/data/source-pin checks, paired-error conventions,
deadline/current-weight handling and artifact contract from v3/v4 only where
their protocol identifiers and assumptions apply. Bind the new all-query
target/branch manifest separately. Training states retain autograd; diagnostic
state snapshots detach only **after** producing the training graph. Evaluation
still performs ordinary uninterrupted full 17-update native episodes, and its
complete-state fork check is an adapter assay, not a replacement behavioral
measurement.

Before any fit, resolve these precise gaps:

1. Coordinator approves this estimator/validation/sample-geometry proposal and
   four-fit local envelope; no authority is inferred from this document.
2. Additive production code/configuration receive independent source,
   state-fork/gradient/oracle/deadline review at their final exact hashes.
   Extend the tiny calibration to the actual 128/64/batch-32 FP32 training
   helper without fitting, checking both state components and all parameters.
3. Pin all original v3 initial/endpoint/provenance/base-stream references and
   regenerate their base manifests; persist seed-20005 validation and derived
   all-query identities before optimization. No new validation scores have
   been inspected for this proposal.
4. Commit clean immutable source/configuration, create an external coordinator
   freeze receipt binding the accepted review and source, then admit the fixed
   four-fit run. Require independent raw-loss/native-checkpoint reduction
   before accepting any new competence finding.

Remaining scientific uncertainties are whether query-gradient averaging helps
this optimization at all, which coordinates change, and whether any success
survives independent trained seeds and later body exposure. A negative fixed
endpoint is valid evidence about the declared schedule, not architectural
impossibility. A positive endpoint is scoped acquisition evidence, not a pure
capacity mechanism or harmful representation collapse.
