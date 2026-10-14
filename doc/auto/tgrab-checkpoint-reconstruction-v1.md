# T-GRAB checkpoint replay and prospective reconstruction v1

2026-10-14. **Prepared specification; implementation and admission blocked.**
The paired configuration is `configs/graph/tgrab-reconstruction-v1.json`.
It describes one prospective seed-0 GC-LSTM pilot, not an executable launcher,
full paper reproduction, expected convergence, or a recovered checkpoint.
No model, optimizer, data/weight payload, native job, download, provider call,
or new engineering test was executed for this preparation.

## Evidence and the two paths

Public source is T-GRAB commit `31293102b065feef7817ff86211427d7594edfdb`.
References below use paths relative to that repository, inspected under
`runs/graph/upstream-pin-20261012/source/`; `train/run.py` is additionally
preserved under `runs/graph/upstream-followup-20261014/source/` at that same
commit. Dependency cell source is `torch_geometric_temporal` v0.54.0, commit
`46abd39abec0cd8de23c0e489a617c320c3fc5e8`, SHA-256
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
These are source pins, not proof of the legacy installed environment.
Dataset repository metadata is pinned to Hugging Face revision
`603388f332c7756f90bc1702b26053c2bc834654`; the existing pin verifies AB/AABB
file metadata only. No local dataset payload checksum or CSV-to-NPZ conversion
has been verified, including the primary long-repetition condition.

Archived TSA HEAD is `403bd6f47e42f5fd98c1cf366da6891b955fc208` on branch
`appendix`; working-tree equality remains unknown. Individual text identities
and limited search scopes are in `runs/review/mila-tsa-recovery-20261014/`.
The June-1 archived analysis source has SHA-256
`6f5b539648162a5abf258514c639c365b238b199dd656f8752e7bb74744e77d0` and
is not bound to the April-25 numerical candidates. The final conditions
receipt identifies an archived script with `N=8` and an active
`METHODS_TO_RUN=("CTDG/_gclstm")` spelling, not proof that it launched the
DTDG model for AB, AABB, or the primary condition. The inspected `O0.txt`
contains module-load messages, not a recovered training configuration.

**Historical replay if recovered:** require immutable recurrent **and link-head**
weights, checkpoint schema, exact task/data bytes and conversion, node order,
selected epoch, source/env identities, chronology/reset/cache state, and an
explicit binding to the original output. Record source behavior faithfully,
including any target warm-start and test-dependent stopping that actually
occurred. Do not silently apply the prospective corrections below to a replay.
Recovered weights may permit an inference replay without recovering training
RNG; regenerating those same weight bytes cannot be promised without the
complete RNG, initialization, optimization and environment history.

**Reconstruction if not recovered:** run only the prospective candidate below
after engineering review and a concrete fresh compute envelope. This is a new
training trajectory. All unidentified legacy settings and legacy head-weight
identity remain null in the config. The bounded search found no checkpoint in
enumerated TSA scopes; that is not a global absence claim. An independently
admitted archival-array audit is a separate workflow and does not admit this
pilot or establish checkpoint linkage.

## Registered conditions and behavioral hypothesis

Primary is **Pdet(2,32)**: A repeated 32 times, then B repeated 32 times,
period 64. The behavioral forgetting plot is in
`ACM___TGRAB_KDD_2026/kdd/tex/100appendix.tex:319–338`, actively included by
that snapshot's `main.tex:353`. The older `ACM___TGRAB/kdd/main.tex` excludes
its appendix. **AB=Pdet(2,1), period 2**, and **AABB=Pdet(2,2), period 4**, are
separate registered conditions with no execution in this candidate.
`floquet_analysis/main.tex:69` names AB; `:436–510` treats AABB and the six
reported magnitudes. They are not interchangeable checkpoints or targets.

The prospective behavioral question is whether this bounded GC-LSTM pilot
learns the primary next-graph task and how per-step F1 differs at transitions
and within repeated blocks. A failed competence outcome is a pilot result,
not a failed replication of the original paper. No result can establish
harmful representation collapse by F1 alone. Competing explanations include
sufficient compression, input synchronization, optimization/readout failure,
and measurement artifacts. A single initialization on two fixed patterns
supports no seed-level uncertainty estimate or generalization across graphs.

## Source-backed candidate architecture and data

The conditional public script branch specifies `NUM_UNITS=1`,
`OUT_CHANNELS=128` (`scripts/task/link_pred/periodicity/slurm.sh:223–256`),
one-hot features (`:66–71`), and `K=2` in
`scripts/task/link_pred/periodicity/DTDG/_gclstm.sh:29,37–48`.
That script's currently active methods select DyGFormer and its loop uses
AB; this supports available settings, not evidence of old GC-LSTM execution.
The CLI defaults instead are H=100 and K=1
(`train/DTDG/gclstm.py:11–26`), so the candidate explicitly overrides them.

Use `MultiLayerGCLSTM` with L=1, input 100, hidden 128, K=2
(`model/gclstm.py:15–51`). The pinned cell uses symmetric-normalized
ChebConv on previous hidden state in each gate, bias enabled, affine node
features, sigmoid input/forget/output gates, tanh candidate and hidden
activation; it has no peephole cell contribution in those gate equations.
See `gclstm-v0.54.0.py:40–124,136–167,196–203` in the source-pin evidence.
Retain the exact pinned cell/graph-convolution implementation; do not replace
it with a generic dense LSTM or an independently chosen convolution.

The source definitely constructs a separate `LinkPredictor(hidden_dim=128,
num_layers=2)` (`train/DTDG/link_pred/trainer.py:71–80`). Its ordered pair head
is concatenation of source/destination last-layer hidden vectors, then
ReLU → Linear(256,128) → ReLU → Linear(128,128) → Dropout(0.2) →
Linear(128,1) → sigmoid (`model/link_predictor.py:4–25`). Preserve this
unusual ReLU-before-linear order, biases and pair orientation. Train enables
dropout; validation/test/conditioning replay use `.eval()` and disabled
gradients. No recurrent dropout is present in the inspected cell. Both model
components belong in every checkpoint; hidden-only recovery is insufficient.

Data: 100 labeled nodes, fixed undirected ER patterns with p=0.01, every edge
stored in both directions with weight one, no evaluation self loops. Node
features are the 100×100 float32 identity (`utils/node_feat.py:20–21`);
the CLI `node-feat-dim=1` does not make these one-dimensional. Pattern IDs
1 and 2 reset Python/NumPy RNG to 12346 and 12347 independently of the outer
generation seed (`dataset/DTDG/graph_generation/periodicity.py:137–148`).
Declare outer generation seed 12345 for bookkeeping; model seed 0 is not
an independent graph seed. Bidirectional expansion is
`graph_generator.py:17–39`; periodic pattern ordering is
`periodicity.py:56–60`. If raw source generates an empty graph, it adds a
node-0 self-loop marker (`:224–228`); record and resolve that marker explicitly
in the conversion gate. The candidate requires nonempty off-diagonal positive
sets and a valid negative destination for every positive source before launch;
it does not silently reinterpret marker loops as true links.

Use 40/4/4 whole periods, chronological, unshuffled one-snapshot batches
(`periodicity.py:44–70,125–134`; `link_pred/trainer.py:103–137`). For the
primary condition: train timestamps 0..2559, validation 2560..2815, test
2816..3071. Preserve all timestamps, including empty ones, integer node order
0..99, duplicate removal, and a conversion manifest binding raw and dense
snapshot hashes, pattern IDs, edge weights and split masks. The trainer's
first split-local predecessor equals its first current graph
(`dataset/DTDG/torch_dataset/link_pred/node_feat_static.py:53–65`); the
corrected chronology must not use that value as conditioning at a boundary.

## Prospective chronology and complete state

Define `s[t+1]=F(s[t], G[t], X)` and `p[t]=head(h_last[t])`, predicting
**G[t] before consuming G[t]**. The hidden component of s[t] has seen only
graphs 0..t−1; labels, edges, weights, features and caches for G[t] may not
enter its construction. Training t=0 is unscored conditioning. Each epoch
resets h/c to zero, consumes G[0], then scores targets t=1..2559 in order;
each target's optimizer step follows its score/loss, before the next graph
is consumed. Detach h/c at each next update, with a one-step backpropagation
window and `backward_only_last`. Use the predecessor's matching weights,
not the split's first target weights, and no random pseudo-predecessor.

For each validation, freeze the epoch's final weights, reset zero state, and
replay G[0]..G[2559] in eval mode as conditioning only. Score each validation
target before consuming it, then consume once. Validation state never becomes
training continuation state; the next epoch resets independently. After
validation-only selection is finalized, reload both selected model components,
reset, replay training and validation graphs once under frozen selected
weights, and score the closed test sequence once in the same order. Test is
not read, evaluated, used for reset/cache continuation, stopping, selection,
thresholds or hyperparameter decisions before that final pass.

Core h and c are each `[1,100,128]`; joint dimension is **25600**, flattened
as component (hidden then cell), layer, integer node ID, feature, all axes
ascending. Store complete per-node/per-layer state. `z=h_last` is a derived
output with an explicit timestamp and equality assertion, not an independently
advancing cache. The end-to-end prediction manifest also records last consumed
graph/weights, next target timestamp and reset mode; these external inputs are
fixed when taking state derivatives. Pooling nodes or using only 2H is a
different measurement. No derivative/covariance/orbit analysis is executed by
this candidate. Future discrete products are finite-time propagators unless
input, complete-state and Jacobian periodicity/closure all pass calibrated gates.

## Loss, optimizer, metrics and bounded stopping

Source loss is **BCELoss**, not MSE: mean positive BCE plus mean sampled
negative BCE (`link_pred/trainer.py:68–69,287–301`). Source Adam applies to
recurrent and head parameters with learning rate 1e−4 (`:58–66`;
`train/trainer.py:174`). Prospective Adam explicitly fixes betas (0.9,0.999),
epsilon 1e−8, weight decay 0, amsgrad false, with a deterministic named
parameter order instead of the source's unordered set. These optimizer-default
choices are explicit candidate settings, not recovered optimizer history.
Use float32 training, no AMP or gradient clipping. Record deterministic-backend
flags and fail/report unsupported deterministic operations without silently
switching algorithms.

For each off-diagonal positive ordered edge (u,v), sample one destination
uniformly from all off-diagonal nonedges with that same source u, with
replacement, using a dedicated negative RNG seeded 1. Sample against the
**full current positive mask** using integer IDs/boolean membership; duplicates
among negatives remain permitted and counted. Assert every negative is a
nonedge, diagonal exclusion, source alignment and equal positive/negative
counts; reject an empty admissible row rather than recurse forever. This
corrects the source's Tensor-object membership collision bug
(`link_pred/trainer.py:194–219`) and additionally excludes source-permitted
diagonal negatives. These are declared prospective deviations. Generation,
initialization, dropout and sampling RNG states must be recorded separately.

Evaluation enumerates all **9900 ordered off-diagonal pairs**, including both
orientations, with threshold **probability >=0.5**, fixed from source
(`periodicity/trainer.py:27–59`). Compute one F1 per timestamp with
`zero_division=0`, then arithmetic time mean; do not pool confusion counts or
sample negatives for this metric. Store per-step TP/FP/FN, precision, recall,
F1, accuracy, target/source phases and transition flags. Target phase is
`phi=t mod T`; source phase is `(t−1) mod T`. A change point means
`G[t] != G[t−1]`, equivalently target phases **0 and 32** for the validated
primary two-distinct-pattern manifest. Source phases are **63 and 31**.
Test has 256 scored targets, eight transitions and 248 nontransitions; include
the first test boundary. Report time means for all targets and transitions,
nontransitions, and each target phase. Distinct-pattern/data validation is a
gate, not assumed numerical evidence. Persistence predicts G[t−1] under the
same pair/phase contract; EdgeBank needs a separately pinned exact definition
before being called source matched. Neither baseline grants another fit.

Stopping and selection are separate. Epoch training loss is the arithmetic
mean of 2559 completed update losses. Initialize best loss on the first finite
epoch; improvement requires `(best−current)/max(abs(best),1e−12)>1e−6`.
Only then replace best/reset patience; otherwise increment. Stop at **100
consecutive nonimproving epochs**, any nonfinite quantity/integrity error,
owner stop, or first hard cap: **200 complete epochs, 511800 optimizer updates,
3600 elapsed seconds for the entire pilot**. Stop at a safe update boundary;
an interrupted epoch is incomplete and cannot select a checkpoint.
Validate after every complete epoch; select maximal validation time-mean F1,
strict improvement only, earliest epoch on ties. Retain last resumable state
separately from the selected weights; no automatic resume/retry. Test remains
closed if failure/timeout prevents finalized integrity and selection.
The caps are prospective pilot limits, not authorized resources or a claim
that source patience/paper convergence will be reached. The source wrapper has
100000 max epochs, and its loss tolerance resets on a plateau; the paper
describes patience without an epoch cap. A cap-bound exit is reported as capped.

## Corrections, acceptance checks and readiness

Do not run the public trainer unchanged: construction precedes `run()` seeding
(`train/run.py:28–29`; `train/DTDG/trainer.py:16–25`;
`train/trainer.py:465–479`); repeated runs reuse model/optimizer objects.
Its first training predecessor can be random or a prior evaluation cache
(`link_pred/trainer.py:252–257`), cold evaluation consumes the target twice
(`:387–406`), test follows every validation improvement
(`train/trainer.py:347–371`), perfect cached test F1 stops training
(`periodicity/trainer.py:74–78`), and test state can seed the next epoch.
Loss patience uses the wrong sign test for decreasing loss
(`utils/early_stopping.py:56–64`). Historical replay records these branches;
prospective implementation corrects them explicitly.

Minimal engineering acceptance, not yet performed:

1. Verify pinned source/dependency identities, supported locked environment,
   installed cell equality, parameter names/counts/shapes and complete head
   round trip; preserve exact ReLU order and train/eval dropout behavior.
2. Tiny symbolic chronology checks prove no current-target input before score,
   exactly one consumption per graph, correct split predecessor/weight matching,
   independent reset epochs, and no test access before finalized selection.
3. Seed before constructing either module or optimizer; fresh objects and
   dedicated RNG streams, repeatable initialization, ordered optimizer params,
   no implicit checkpoint load or prior-run/cache reuse.
4. Tiny adversarial negative sampler checks cover forced positive collision,
   same-source multiple positives, diagonal exclusion, duplicates policy,
   saturated rows and finite termination; compare loss/gradient to a manual
   positive-plus-negative mean BCE reference.
5. Verify 40/4/4 exact periods, source pattern seeds/order, directed expansion,
   duplicate/self-loop/empty-step handling, node mapping and snapshot hashes;
   compare conversion against pinned generator semantics before training.
6. Metric fixtures enumerate 9900 pairs, threshold equality at 0.5,
   zero-positive convention, time mean versus pooled F1, all split-boundary
   transitions, target/source phase offset and the eight primary test transitions.
7. Stop/selection fixtures cover strict loss improvement, zero-best-loss floor,
   plateau patience, validation-only earliest-tie selection, all hard caps,
   partial epochs, atomic selected/resume manifests and complete h/c state.

Before model execution, freeze/review the corrected implementation under `src/`
and its checks; acquire/verify actual data only under its own authority; resolve
CSV/NPZ conversion; bind environment lock, Python/Torch/PyG/compiled extension
identities and sklearn metric version. The public requirements request Torch
2.0.0/PyG2.6.1/Temporal0.54.0 plus compiled extensions but omit sklearn and do
not pin Python/CUDA/toolchain; they are not proof of a buildable environment.
An existing vector environment is not automatically graph compatible. Plan
fail-closed W&B reporting, scratch paths, hashes, stop controls and useful
throughput/per-device evidence before requesting any long GPU work.

Concrete next implementation tasks: implement the explicit recurrence/head
adapter and complete-state manifest; implement the corrected causal trainer,
integer-mask sampler and deterministic parameter/RNG order; implement the
metric/phase/stopping/checkpoint writers; add the seven minimal fixture groups;
obtain independent engineering and scientific protocol review; resolve the
environment/data gates; then prepare exact executable commands, cluster/account,
resources, walltime, number of submissions, output paths and stop rule for a
fresh approval envelope. **No corrected trainer/launcher exists yet; command,
cluster, account and admitted resource fields remain null and launch disabled.**
Array-audit results or the expired shared allocation cannot bypass these gates.
