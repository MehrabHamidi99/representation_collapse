# T-GRAB reconstruction runner readiness and next bounded envelope

2026-10-15. **Planning only: runner blocked, original configuration disabled,
new training budget absent, scientific acceptance pending.** This record owns
no execution authority. Preparation here consisted of local source/document
reads and SHA-256 checks; no model, optimizer, generator, dataset/checkpoint
payload, SDK, provider, network, cluster or numerical campaign was invoked.
The coordinator reports that the last original CPU slot was consumed by Mila
job 11256293, which failed with 1:0 after 23 s at package import:
`No module named torch_geometric.utils.to_dense_adj`. All constructor, native
cell and AD counts were zero; no GC-LSTM mathematics ran. Temporal 0.54.0's
umbrella import is incompatible with the pinned PyG 2.6.1 API. Numerical native
calibration therefore remains unverified; remaining original CPU submissions
are **zero**. Installation/submission/outcome reconciliation belongs to root
and must be bound to its actual receipts. Failure does not refund the slot.

The next concrete proposal is one CPU **engineering training-throughput
canary**, followed only conditionally by a separately admitted seed-0 fit.
Neither a tiny calibration pass nor a throughput canary establishes learned
task competence, original checkpoint parity or harmful collapse. AB (period 2),
AABB (period 4) and primary Pdet(2,32) (period 64) remain distinct conditions.
This plan neither adds conditions nor tunes towards a desired change-point
decline, dimension or multiplier.

## Frozen inputs and unresolved provenance

These SHA-256 identities were checked locally for this planning handoff:

- Accepted contract module `src/representation_collapse/graph_reconstruction.py`:
  `89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.
  Its synthetic tests are
  `9b1f55a5f3e57ac8d48914500eada783071f7820b41a63f06a52c766eba89eb2`.
  [Independent component review](tgrab-reconstruction-implementation-independent-review-2026-10-14.md):
  `9cc61e8a9436867ec118d74792b708219fadc05d53c532017bd7e0b8fb7f729b`.
- [Reconstruction specification](tgrab-checkpoint-reconstruction-v1.md):
  `698a3150566e47a1866308f7b9782ab2b2b1f4878b6592acb978972cb9f1107b`;
  `configs/graph/tgrab-reconstruction-v1.json`:
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
  Frozen canonical candidate SHA:
  `dea0e7684f52a0a53cba5072e77778d22995cb0b0e30f3352a6e349b89628092`.
  Launch remains false; executable, resources and approval envelope remain null.
- [Graph source/recovery audit](tgrab-reproduction-audit-2026-10-14.md):
  `96dc51d23d3345ab314649760e8a6337a6eb2d5c3e25427a01bb31b1ddc06741`;
  `configs/graph/tgrab-upstream-pin.json`:
  `784e7fd60463dbdd92bce15ab1aee80e4dd22bcd794401fcba9e9c0b964ca812`.
  The primary behavioral target is the active KDD2026 appendix
  `ACM___TGRAB_KDD_2026/kdd/tex/100appendix.tex:319–338`, included at
  that snapshot's `main.tex:353`. AABB six-mode analysis is a separate draft.
- Pending [native calibration](native-graph-calibration-v1.md): protocol
  `49010335090b22052b38ec153aa378b4f5aef8672cdfd5f97123581db0665a2e`,
  module `3cf9136240d6fa30d1aabaffd15aaf971acea328cb117364bf0cb2b6091c0d51`,
  config `785225b573c243b914eb82a5bebd71794ff8ac5fc3b86fd6a4b18a71245a0735`.
  N=3/H=2/K=2, full H+C dimension 12, FP32 wrapper wiring and FP64 derivatives
  are engineering scope. Its finite successful schedule is 50 cell forwards,
  5 head forwards, 4 dense-Jacobian, 9 JVP and 5 VJP API entries.
  These immutable v1 identities identify preparation and the failed import
  attempt, not successful numerical runtime evidence. Root is preparing a
  separately reviewed isolated exact-cell loader; its new packet hashes are
  unresolved in this planning record and must not be substituted for v1.
- Separate legacy CPU wheel manifest
  `environments/graph-native-cpu/wheel-manifest.json`:
  `67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842`;
  lock `ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06`.
  [Environment preparation review](native-graph-environment-independent-review-2026-10-15.md):
  `15f61ece605671ae9619da73e79ae37ca9f1eb872b61ab3a8a7e2d42bf6097ca`.
  Metadata/pure-wheel source acceptance does not establish installation,
  ABI/import, actual physical-memory behavior or reporting success. Root's
  subsequent import failure demonstrates that metadata installation success
  is insufficient to establish native package compatibility.

The public repository commit is
`31293102b065feef7817ff86211427d7594edfdb`; the actual Temporal 0.54.0
GC-LSTM commit is `46abd39abec0cd8de23c0e489a617c320c3fc5e8`.
The source fixtures are regular files under
`runs/graph/upstream-pin-20261012/source/`, with cell reference
`runs/graph/upstream-pin-20261012/gclstm-v0.54.0.py`. Exact cell SHA is
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
At import time, check installed cell path and bytes and the actual class used by
the wrapper; no replacement recurrent cell or alternate convolution is allowed.
Wrapper SHA pins are:

- `model/gclstm.py`: `86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`.
- `model/link_predictor.py`: `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`.
- `model/node_emb.py`: `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.

Data/conversion and trainer sources are also pinned, but their execution and
old-run linkage remain unresolved:

- `dataset/DTDG/graph_generation/periodicity.py`:
  `754cd90de5a3ba0887d73df29391cc62933c479e89f5e5d6960d07b6bd542e54`;
  `graph_generator.py` in that directory:
  `447028d7c436bcc160c294ed10febe7dd372465d042531335b123c17aaf5e1f0`.
- `dataset/DTDG/torch_dataset/link_pred/node_feat_static.py`:
  `de5cbc2996c797df5982ff3e6ba6cd94c59c4da63921f3b8d8185e9b6f602234`;
  `utils/node_feat.py`:
  `c26c33871382cfd4128b452cc764239ca23a1d9f0c08c7819cb454b5314a9c26`.
- `train/DTDG/link_pred/trainer.py`:
  `beada76c5761e771167247afed94ec273dba927ba1bd6167d081b394e14151a7`;
  `train/DTDG/link_pred/periodicity/trainer.py`:
  `e25be9b55f4e1f54535e6a4ca7f05232d0236f50c0c5ab3eed0bfdb501c575ec`;
  `train/trainer.py`:
  `de11f276e58ea09f489b59e6f08e089e277fdbe8f42d3296b21dbf1256257bd2`.
- Follow-up fixture `runs/graph/upstream-followup-20261014/source/train/run.py`:
  `0307e50958e3cca25675886721e90de2535248add0bfdde545a50f3e870f716a`.

Hugging Face revision `603388f332c7756f90bc1702b26053c2bc834654` is metadata
provenance; the earlier receipt verified AB/AABB metadata only. It does not
provide a verified primary Pdet(2,32) payload/hash or CSV-to-NPZ conversion.
Acquire the exact primary file under separately concrete acquisition authority,
or freeze a source-generated prospective dataset with generator/dependency/RNG
identities and call it newly generated. The task here generated neither.
Generation is never evidence of equality to an unidentified historical payload.

Archived TSA HEAD `403bd6f47e42f5fd98c1cf366da6891b955fc208` is a recovery
reference, with working-tree equality unknown. Archived analysis SHA
`6f5b539648162a5abf258514c639c365b238b199dd656f8752e7bb74744e77d0`
is not bound to the old arrays or checkpoint. The bounded owner-TSA search found
no recurrent/head checkpoint in its enumerated scopes; global absence is not
established. Legacy checkpoint/head weights, dataset, command, seeds, optimizer,
runtime and RNG history remain null. Byte-identical old checkpoint regeneration
cannot be promised without the original weights and histories. A new fit is an
explicitly corrected reconstruction with deviations, even if its F1 resembles
an inherited picture.

## Missing implementation and readiness gates

The accepted module supplies snapshot/state contracts, integer-mask negative
sampling, metrics, loss-patience and validation-selection helpers, a causal
controller and a guarded native adapter. It deliberately has no CLI, converter,
checkpoint reader/writer or launch supervisor. `preflight()` always appends
`native_checkpoint_and_reporting_runner_not_implemented`; `load_selected()`
raises. The controller accepts tiny test sequences and is not the full split
orchestration. The 28 synthetic tests and lifecycle repairs remain component
evidence only. Do not remove blockers by asserting readiness booleans.

The following bounded components must be implemented/reviewed before asking
the owner to approve a runnable envelope:

1. A primary-data manifest and deterministic converter that verify all 3072
   timestamps, fixed node IDs 0..99, distinct A/B patterns, repetitions,
   bidirectional unique integer edges, weight-one alignment, features and split
   masks. Raw payload, conversion source, snapshot/order/pattern hashes and
   provenance must be independently checked. Resolve source empty-graph loop
   markers explicitly; reject empty off-diagonal positives and saturated
   negative rows. Preserve missing timestamps rather than deleting them.
   Pattern IDs 1/2 reset Python/NumPy to 12346/12347
   (`periodicity.py:137–148`); outer seed 12345 and model seed 0 have different
   roles. Public CSV and generated `data.npz` are not interchangeable by name.
2. A locked-environment receipt for this actual runner, including Python,
   interpreter/wheel/extension hashes and native import outcome. Use the
   separately declared CPython 3.10 Linux-x86_64 fixture environment via explicit
   source/PYTHONPATH imports; do not claim mainproject Python >=3.11 installation.
   The literal original metadata probe rejects Torch 2.0.0+cpu and extension
   +pt20cpu versions. A reviewed runner must allow **only the exact full build
   identities in the immutable CPU manifest**, distinguish them from base
   source versions, and retain the exact cell check. Arbitrary version stripping
   or substitution is not a fix. Main lock and old environments stay unchanged.
   If root's isolated cell adapter imports the exact installed `gc_lstm.py`
   without the incompatible Temporal umbrella, freeze its helper/namespace/
   dependency/wrapper import-path hashes and label that import-path deviation.
   Review must prove unchanged cell mathematics, source bytes and versions.
   Adapter acceptance does not establish the full Temporal API, the original
   public umbrella import or benchmark replication. Its runtime numerical
   evidence and any new allocation need a fresh, separately admitted packet.
3. Actual native initialization, complete-state, head, dropout, loss-gradient
   and checkpoint/RNG round-trip evidence. Tiny N=3 calibration can establish
   only its declared wiring/AD scope; it has no training loss/Adam/dropout or
   checkpoint round trip. Freeze a finite engineering fixture with manual
   positive-mean-plus-negative-mean BCE and parameter/RNG/state round trips;
   admit its native operations separately before running it. Then add the real
   split/loader integration, without modifying public snapshots.
4. The full epoch/patience/selection state machine; trusted, hash-checked dual
   recurrent/head serialization; atomic selected/resume manifests; test opener
   guarded by finalized validation selection; and terminal test lifecycle.
   Maintain the repaired post-load budget/STOP check before test opens and
   reject subsequent training/validation. Partial epochs and canary artifacts
   must never be presented to checkpoint selection.
5. A stdlib outer launcher and reporting integration. Start the monotonic wall
   clock and owned-child journal before numeric/SDK imports; initialize pinned
   offline W&B successfully before native constructors, data/model/array payload
   hashing or numerical work. Configure single numerical/BLAS threads before
   imports. Bind source/config/data/env/admission/argv/STOP/output/job identities.
   Enforce actual Slurm/cgroup **physical memory** and an import-inclusive
   physical deadline, including owned SDK-descendant cleanup. RLIMIT_AS is
   virtual address space and cannot substitute for the physical cap. Boundary
   checks alone cannot preempt a hung native call or checkpoint write.
6. Independent engineering and scientific protocol reviews and six correctly
   typed/config-bound readiness receipts: environment, conversion, engineering
   review, scientific review, reporting and admission. The calibration receipts
   and the original analytic budget do not satisfy reconstruction admission.
   Freeze a new runner/launch packet with exact argv only once it exists; this
   document invents no executable training command.

## Frozen fit contract that integration must preserve

The candidate remains L=1/N=100/input=100/H=128/K=2 with exact native symmetric
ChebConv and the source head. Features are the fixed float32 identity. The
ordered pair head is concatenation, ReLU, Linear 256→128, ReLU, Linear 128→128,
Dropout 0.2, Linear 128→1, sigmoid (`model/link_predictor.py:4–25`). Train mode
enables dropout; conditioning/validation/test use eval mode and no gradients.
Public wrappers are reused byte-for-byte. There is no substitute dense LSTM,
pooling, extra recurrent dropout, AMP, clipping or increased effective batch.

Before either model component or Adam construction, seed Python, NumPy and
Torch with model seed 0 and record deterministic backend settings. Create new
recurrent/head/optimizer objects and named recurrent-then-head parameter order;
never reuse a previous experiment's objects. Record Python hash seed at process
start. Negatives use a separate RNG seeded 1: one uniformly sampled admissible
off-diagonal nonedge with the same source per positive, replacement permitted,
against the entire current integer positive mask. Assert counts/collisions and
finite termination before loss. The source membership bug and diagonal exclusion
are explicit prospective deviations, not historical parity.

Training uses FP32 Adam lr 1e-4, betas (0.9,0.999), eps 1e-8, decay 0,
amsgrad=false. Loss is mean positive BCE plus mean negative BCE, rather than
their pooled mean or MSE. Reset zero H/C each epoch; t=0 is conditioning only.
For t=1..2559, detach the predecessor state, consume G[t−1] with its own weights,
score G[t] from that resulting state, compute loss/backward/update, then proceed.
The target input is consumed only as the next target's predecessor. This is
one-step truncated backpropagation; the final training target is not needlessly
consumed. Each complete epoch has exactly 2559 updates. Record optimizer attempt,
successful return and committed-step counters separately: the current helper
increments after backend return, so a post-step error cannot be assumed to mean
the optimizer did nothing. No retry can hide an extra step.

Train interval is [0,2560), validation [2560,2816), closed test [2816,3072).
For every complete epoch, freeze its final weights, freshly reset H/C and replay
all 2560 training inputs in eval mode. At each validation target score first,
then consume once. Never use validation state as the next training epoch's
initial state. Loss stopping is the mean of 2559 completed update losses;
improvement is `(best-current)/max(abs(best),1e-12)>1e-6`, else patience
increments. Stop at 100 consecutive nonimproving complete epochs, owner STOP,
nonfinite/integrity failure or the first cap: 200 complete epochs,
511800 optimizer updates,3600 **entire-pilot** seconds. A partial epoch cannot
select. Validate after every complete epoch; select strict maximal validation
time-mean F1, earliest ties. Selection is independent of loss stopping.

Only a finalized, integrity-verified selection reloads both components and
opens test once. Freshly reset and replay train+validation 2816 inputs at those
selected weights, then score and consume all 256 test targets once. Preserve
the first boundary transition. Test data/metrics cannot affect threshold,
hyperparameters, patience, selection, continuation, competence thresholds or
extra-run decisions. If the wall cap/STOP/failure prevents finalization or the
final pass, test remains closed or terminally partial; no automatic resume or
retry follows. The 200-epoch cap is a prospective pilot, not full source/paper
convergence. CPU 1 may finish substantially fewer epochs; N=3 calibration timing
cannot predict N=100 backward/Adam/evaluation cost.

## Checkpoint, monitoring, metrics and reporting contract

Selected weights and last resumable state are different artifacts. The selected
manifest needs recurrent and head bytes/hashes, named shapes/parameter order,
source/config/environment/data identities, selected epoch and validation
metrics. Resume additionally needs Adam state bound to named parameters,
all RNG states (Python, global NumPy, dedicated negative RNG, Torch CPU and any
declared CUDA stream), dropout/module modes, complete H/C, update/epoch/loss
patience/selection counters, next target/last consumed input and remaining
deadline lineage. Captured H/C is not a replacement for either weight component.
Atomic writes require durable completion/hash checks; failed partial writes
cannot become selected. Inspect schema/provenance before trusted weight
deserialization; no filename-only cache, implicit `torch.load` or old-run reuse.
Any future resume needs a separately explicit admission and strict round-trip
evidence. This planning document authorizes neither serialization nor loading.

Both H and C have shape [1,100,128], joint D=25600, flatten hidden then cell,
layer/node/feature ascending. Track finiteness and absolute norms for both and
joint state without averaging nodes away. `z=h_last` is derived, with timestamp
and equality assertion. Record teacher forcing, fixed one-hot X, consumed
graph/edge-weight identity and next target for every retained state. Monitoring
must include the reporting/native synchronization overhead in phase timings.
For the proposed canary, retain at most eight full states immediately after
updates 32,64,...,256, together with their exact predecessor-state chronology;
eight FP32 joint payloads occupy 819200 bytes before headers. This is monitoring,
not an orbit or covariance measurement.

For a future full fit, the prospective bounded raw capture is 65 pre/post-input
states s[2560]..s[2624] for each complete-epoch validation, at most 200×65 states
(1331200000 FP32 payload bytes), plus at most 257 final test states
s[2816]..s[3072] (26316800 bytes). These are captured during existing forwards,
with no additional model evaluation. Bind every trajectory to its actual epoch
checkpoint; older epoch trajectories cannot be relabeled as selected. Whole
scratch allocated-byte census, environment/cache overhead and checkpoint files
must fit the project's20 GiB cap before admitting this output plan. Storage
forecasts are ceilings, not current measurements. Covariance/PCA/Floquet
analysis remains outside the training pilot. A time-centered64-state periodic
record has rank≤63 by construction; zero variance has undefined normalized
dimension and does not establish harmful collapse.

Every validation/test target uses all 9900 lexicographically ordered off-diagonal
pairs, probability>=0.5 and zero_division0. Store per-target TP/FP/FN/TN,
precision/recall/F1/accuracy, t, target phase t mod 64, source phase(t−1) mod 64,
actual adjacent-graph transition flag and graph hashes. Average F1 across time,
not confusion counts. Report all-target, nontransition, transition and
direction/phase-specific means. Verified primary test has 256 targets, eight
transitions (target phases 0/32; source phases 63/31),248 nontransitions.
Persistence uses G[t−1] under this identical mask/chronology; EdgeBank remains
blocked until its exact variant/window/source definition is pinned. Neither
control entails another learned fit.

The reporting pin is `src/representation_collapse/reporting.py`, SHA
`443c9e89d2eba954ee60cc35e015fe61196fbec2f50331067540b4a8259e8184`;
offline SDK wandb 0.30.0 is mandatory. Reusing deadline/child ownership helper
`code/ops/legacy_floquet_audit.py` requires SHA
`8e1f388d4f5ce72e0bfd9a4b5a4b1f8c98f66f2512841d96875822bfd9bffe09`
and a reviewed launcher integration. Raw JSON/arrays/checkpoints, SHA manifest,
offline history/summary/finish receipt, supervisor stdout/status, child cleanup,
actual allocation/cgroup/RSS and scheduler outcome must survive failure.
Offline durability is separate from remote W&B delivery. Reporting failure
stops the worker; model numerical work cannot continue with tracking disabled.

## Smallest next concrete envelope: CPU training-throughput canary

**Prospective, disabled, not admitted.** This is a new engineering task with
real training semantics, outside the exhausted analytic envelope. It needs a
separate reviewed disabled config and runnable runner; the original candidate
and launch configuration remain byte-identical. Source/config/argv/admission
hashes cannot be finalized before that implementation exists.

- One job/one attempt on Mila, account mila, QOS normal, partition main-cpu;
  one CPU thread, physical 2 GiB, 600 s total, zero GPUs, no array, race, requeue,
  retry or automatic resume. Requested ceiling is 1/6 CPU-hour; failures and
  startup count. Coordinator must freshly verify target/profile/queues and exact
  cluv 0.1.5 arguments before admission; the October 15 verified CPU profile is
  a preparation lead, not a new scheduler observation in this task.
- Same pinned primary N=100/H=128/K=2/one-hot/FP32/head/Adam/seed 0 and negative
  seed 1. At most 256 optimizer **attempts**, scoring t=1..256 from the real train
  prefix, with one conditioning consume and correct detach/update chronology.
  This is a deliberately partial 40-period epoch: zero complete epochs,
  zero selectable checkpoints and zero test access. No early success on F1.
- After the last admitted update, freeze those partial-fit weights and, only
  if the work deadline permits, perform one fresh full train-prefix replay and
  one 256-target validation pass with fixed weights. Validation here measures
  forward/metric cost and integrity; it cannot select or certify task competence.
  Partial replay/metrics remain explicitly incomplete. Stop before any second
  replay or new optimizer attempt beyond the finite schedule.
- Maximum successful native schedule: 256 training recurrent/cell forwards and
  512 head forwards (two loss-head calls/update), plus 2816 evaluation cell
  forwards and 256 evaluation head forwards: **3072 cell and 768 head entries**.
  One wrapper/cell/head construction and one Adam construction; at most 256
  backward/update attempts. Dense-Jacobian/JVP/VJP/FD APIs: **zero**; native
  calibration calls are not part of these counts. Counters count entries before
  invocation and preserve partial counts on failure. Internal backward kernels
  are not extra forward/API counts. Evaluation emits at most 2534400 ordered
  pair probabilities. Actual positive counts remain data-dependent and must be
  recorded rather than guessed from pER.
- Arm import-inclusive 600 s supervisor scope; 590 s work deadline and 10 s owned
  reporting/cleanup reserve. Record every phase and first useful update. First 32
  updates are the declared warmup interval; the remaining 224 form at most 14
  consecutive 16-update timing windows. Report useful updates/s, actual ordered
  positive/negative pairs/s, time in negative sampling/data transfer, backward,
  evaluation/pair scoring/checkpoint/reporting and physical RSS. A short/failed
  run cannot fabricate steady-state windows. There are no GPU utilization fields
  for this CPU job; GPU readiness remains unknown.
- Suggested unique scratch root, to be frozen by the coordinator:
  `/network/scratch/m/mehrab.hamidi/representation-collapse/runs/tgrab-reconstruction-throughput-20261015/attempt001/`.
  Bind sibling owner STOP, new source checkout, verified input manifests and the
  isolated environment; refuse existing output. Emit only an explicitly
  **unselected partial-fit** artifact if the reviewed runner includes durable
  checkpoint I/O, never a selected reconstruction checkpoint.
- Stop on first count/work/wall/memory bound, STOP, nonfinite/integrity error,
  reporting failure or unsupported deterministic operation; preserve diagnostic
  failure. No automatic resource increase or rerun. Submission ambiguity is
  reconciled before another command. Acceptance is engineering only: source,
  chronology, loss/sampler, physical bounds/counters and raw timing integrity.

## Conditional competence-first fit and CPU/GPU decision

If canary/environment/data/native-loss/checkpoint/reporting/watchdog gates pass,
use the measured N=100 phase timings to prepare the next exact owner request.
The smallest existing algorithmic candidate is **one fresh seed-0 CPU fit**:
Mila/account mila/QOS normal/main-cpu, 1 CPU, 2 GiB, 3600 s total, zero GPUs,
one submission/no retry/requeue/race/resume, 200 complete epochs and 511800
attempted optimizer-step ceiling. This would request at most 1 CPU-hour and
requires its own approval; neither canary nor calibration authority covers it.
Physical memory fit must be supported by canary RSS plus input/checkpoint/output
headroom. If 2 GiB is inadequate, stop preparation and propose a concrete measured
revision; do not silently enlarge resources. The work deadline is 3590 s with 10 s
cleanup inside the 3600 s cap. A cutoff may leave no complete epoch or no finalized
test pass; mark that outcome partial/capped/ineligible, without claiming a
failed paper replication. The compute envelope is not a promise of convergence.

For transparent cost accounting, a full epoch has 2559 training cell forwards,
5118 training head forwards, 2816 validation cell forwards and 256 validation
head forwards: 5375 cells/5374 heads. At 200 epochs plus the single final test
replay, upper bounds are 1078072 cells and 1075056 heads, with 511800
optimizer attempts. These are much larger than the N=3 engineering schedule.
Add a **single initial untrained validation reference**, from the same fresh
seed-0 model before optimization, only in the separately frozen fit reporting
protocol: 2816 cells/256 heads, never epoch 0 selection. With that prospective
control, maxima become 1080888 cells/1075312 heads. Its eval/reset must not
advance dropout/negative RNG streams or training continuation state. No second
model fit or initialization search is introduced. Record controls as extra
forward work rather than burying them in update counts.

Forecast allocation-wide cost from the measured canary update windows and
full validation replay, including setup/checkpoint/reporting; do not scale the
tiny AD calibration's duration. Show optimistic/pessimistic forecasts and the
number of complete epochs plausibly fitting the wall cap. Those forecasts are
engineering estimates from one process, not convergence or seed uncertainty.
No full-fit request should imply 200 epochs are achievable on CPU 1.

**GPU jobs proposed for immediate admission: zero.** There is no comparable
N=100 GPU throughput/device telemetry or locked CUDA graph environment in this
packet. CPU+CUDA wheels cannot be inferred equivalent from source-version
labels. If CPU measurements motivate acceleration, first freeze the exact
GPU/partition/account, immutable CUDA/extension environment and numerical
equivalence fixture, then prepare **one** independently budgeted single-GPU
throughput canary with the same 256-update/3072-cell/768-head schedule and a
600 s total ceiling, no test/selection and no concurrent CPU fit. Its minimum
CPU/RAM/typed-device request must follow the actual profile and measurements;
those fields are unresolved here, so it is not submission-ready. Record per
assigned device activity provider/interval, VRAM/power, update throughput and
allocation-wide cost; preserve missing telemetry as unknown. Matched workload,
FP32/batch 1/chronology/loss/seed/dropout are fixed. No AMP, batch enlargement,
parallel time steps or additional seeds are justified by utilization alone.
Only a reviewed useful-throughput/cost gain or a documented owning-operator
exception can support a later long GPU fit and its fresh resource envelope.

## Prospective scientific eligibility and interpretation

These proposed operational gates need independent scientific review and an
immutable reporting-protocol hash **before** any canary/fit validation outcomes
are inspected. They are new reconstruction criteria, not recovered paper
cutoffs. They do not change validation-only maximal all-time F1 selection or
the source threshold 0.5. The canary has no scientific competence verdict.

For the later fit, call **pattern/readout competence** eligible only if the
selected checkpoint has validation nontransition time-mean F1≥0.95 for each
of the two patterns and its overall nontransition mean exceeds the one initial
untrained reference by ≥0.10. This establishes useful learned pattern prediction
on this fixed benchmark, conditional on valid artifacts; it does not establish
counting/memory. Report full values and the untrained/persistence controls even
when the gate fails. Persistence can have excellent interior F1, which is why
that label cannot certify history retention.

Separately report **transition competence** if validation change-point F1≥0.90
for each A→B and B→A direction. Failure here does not discard a pattern-competent
checkpoint or prompt retraining: poor change-point performance is a possible
primary long-repeat behavioral outcome. Thresholds are never selected to match
the old forgetting curve. Test confirms or contradicts the frozen labels once;
it never selects an epoch, adjusts a threshold or creates another seed request.
If insufficient training/validation completion prevents selection, classify
optimization/competence as inconclusive rather than attribute failure to
dimension collapse. No seed, condition, cutoff or sign retuning follows a result.

Single seed 0 on two fixed patterns supplies one training realization, no
seed-level confidence interval, and no generalization across graph draws.
Correlated time steps/phases are not independent replicas. Report phase curves,
all failures and prospective deviations, without calling visual agreement
checkpoint parity. A bounded competent reconstruction can release a separately
reviewed measurement node; it cannot establish the original paper result or
H1 harmful collapse. H2 sufficient compression, H3 synchronization, H4 numerical
artifact and optimization/readout failure remain competing explanations.

## Finite tangent validation after competence, with separate admission

The first training canary/fit has zero tangent/covariance/eigensolver work.
Before trained-model derivatives, independently verify the saved tiny native
calibration raw matrices/vectors/counts/source/env/reporting and all frozen
tolerances, including noncommuting chronology, node permutation, complex
real/imaginary extension and nonnormal controls. The tiny dense Jacobian is
12×12; a full D=25600 dense matrix alone is 2621440000 FP32 or 5242880000
FP64 bytes, already beyond the proposed2 GiB physical envelope. Never construct
it there, or substitute independent node blocks/pooling and call them the full
graph derivative.

A minimal **future** full-state integration assay can be frozen separately:
selected checkpoint only, validation state s[2591] (fixed phase 31), inputs
G[2591] then G[2592] across A→B, all H+C coordinates, fixed external X/weights,
eval mode. Use a separate read-only FP64 shadow copied from the verified FP32
checkpoint, explicitly labeled a numerical shadow map rather than the exact
FP32 training map. Do not mutate the selected modules or silently claim GPU
or FP32 derivative equivalence. Require a separately frozen paired forward
consistency check before interpreting that shadow; its calls need their own
ledger rather than being included in the38-call schedule below.

For that prospective two-step shadow assay, take two fixed normalized full-state
directions proportional to(1,2,...,D) and(+1,−1,+1,−1,...), and one cotangent
proportional to(D,D−1,...,1). Freeze flattening order and normalization before
measurement. Six JVP entries (two direct two-step, four stepwise), three VJP
entries (one direct two-step, two stepwise),12 central-FD function evaluations
(two directions × three epsilons × two signs), and two primal cell forwards
give a maximum **38 cell forwards**, zero head/optimizer/dense-full-state
entries. JVP/VJP API counts differ from internal reverse sweeps. Require direct
versus chronological step composition, transpose/adjoint identity and all
fixed FD epsilons 1e-3/1e-4/1e-5 to pass the previously calibrated FP64 AD
atol 1e-10/rtol 1e-8 and FD atol 5e-6/rtol 5e-5; retain failures without selecting
epsilon/direction or loosening tolerance. This finite proposed assay requires
its own source/config/environment/admission and physical resource review.
It is not admitted by either training proposal and is not a scientific spectrum.

Longer products must be ordered J[t+L−1]...J[t], evaluated along the actual
trajectory; repeated input does not imply equal Jacobians. Any 64-step study
needs an independently frozen finite JVP/VJP/solver and closure budget, phases,
directions/converged-pair counts and residual tolerances. Until complete-state,
input and phase-wise Jacobian closure pass calibrated gates, call them
finite-time propagators. A partial spectrum cannot prove all directions
contract; covariance dimension, absolute variance, singular amplification and
task retention are separate measurements. Independently varied required-history
and matched sham/causal interventions are later scientific tasks with their
own freeze, replication set and compute authority.

## Next implementation handoff

Proceed locally with disjoint bounded components: primary conversion/provenance
manifest and full split validator; trusted dual-component/RNG checkpoint codec
and native round-trip fixture; complete causal fit/canary orchestration with
attempt counters, loss patience and validation-only selection; fail-closed
reporting plus physical supervisor; then independent engineering/scientific
review of the immutable runnable packet. Actual native loss/checkpoint fixtures
and data acquisition are not dependency-free unit tests and need separately
defined authority. None may consume another project's allocation or replace
the already submitted analytic calibration.

After these components and new admitted native-calibration/environment evidence
are verified, root can prepare the exact single CPU canary command, source hashes,
new disabled config, scratch inputs/outputs, 600 s resources and stopping rules
for owner approval. Later full-fit or GPU requests depend on measured useful
throughput and fresh approval. This document changes only its own planning
file; original reconstruction/config/source snapshots remain unchanged.
