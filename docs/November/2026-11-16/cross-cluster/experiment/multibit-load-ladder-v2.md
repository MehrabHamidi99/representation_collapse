# Increasing memory load: acquisition and retention boundaries

Prospective protocol, November 16, 2026. The owner requests increasing the
number of bits beyond four and explicitly directs `cluv sync` followed by
`cluv submit first`. This campaign fixes the tested load range before results
are inspected; it does not stop at the first favorable sign or treat failed
acquisition as representation collapse.

## Question and fixed design

At fixed model size, which independent cue loads are learned reliably, and
among competent fits, where does retention fail as the distractor horizon grows?
Does periodic ordering cause excess error relative to a matched varying-order
body at equal elapsed time? Preserve generic forgetting, opposite ordering
effects, perfect retention and failed acquisition as distinct outcomes.

Loads are **4, 8, 12, 16, 24, 32, 48 and 64 binary coordinates**, each fitted
separately with seeds **0, 1 and 2**. The fixed architecture is native LSTM,
69 input channels, hidden 64/cell 64, and a 64-logit linear head. Only the first
D outputs contribute to mean binary cross entropy. WRITE presents all D signed
bits simultaneously; BODY and QUERY carry no payload. QUERY asks for the entire
vector. This is finite categorical recall, not continuous D-dimensional memory,
a graph benchmark or a late addressed-query task. Input energy grows with D;
the per-bit gradient scaling and independently trained weights also vary with
load. Thus this is a boundary for this acquisition recipe, not a universal
architectural capacity limit. Sixty-four is the fixed input/head payload limit;
larger payloads would require a new architecture contract and low-load controls.

Keep v1's paired length12 motif construction: equal symbol counts, adjacent
transitions, run lengths and terminal context, periodic versus varying motif
order. Higher-order predictability/motif diversity still differ. We interpret
this specified ordering intervention, not periodicity isolated from all other
statistics. R0 has no body; R1 arms are identical. The body length is H=12Rq.

## Acquisition, selection and raw evidence

CPU FP32, deterministic algorithms and one thread per seed worker. Each update
uses 32 independently balanced cue vectors duplicated across arms (64 rows),
Adam lr0.001, no weight decay, mean active-bit BCE, gradient norm clipping 1,
and complete-episode backpropagation. Initialize at 7000+seed, identically
across loads. Training, validation and test RNG streams are separated by a
new `multibit-load-ladder-v2` namespace, dimension and seed. They do not reuse
v1's held-out cue/body panels.

Fixed endpoints, without interim selection or rescue training:

1. **6,000 updates**, R0 only; validate R0.
2. **12,000 further updates**, cycling R0/R1/R2; validate R0/R1/R2.
3. **16,000 further updates**, cycling R0/R1/R2/R4/R8; validate all five.

Each gate requires >=0.95 accuracy for every active bit, each cue polarity,
each forcing arm and each required validation horizon. Validation has 256
balanced cue vectors, 128 per polarity. Whole-vector exact accuracy is reported
separately; passing a bitwise gate does not imply 95% full-vector recall at large
D. A failed gate stops that fit without opening its held-out test. Other
preregistered loads/seeds continue. All endpoints, including failures, remain.

Save each stage checkpoint separately, raw validation logits/labels/correct
arrays, integer correct counts/denominators, the cue/body design, and threshold
decisions. Reload the final saved checkpoint and replay every final validation
horizon before opening test data; require exact logits in the pinned CPU
runtime. This is an execution integrity check, not an independent implementation
of the LSTM. Independent reduction must verify raw counts and gate decisions.
This directly repairs v1's missing validation raw/count record; v1 remains
historical and is not retrospectively upgraded.

## Frozen retention measurements

Only fully competent, replay-verified endpoints see the fresh test panel.
Use 256 balanced cue vectors, four forcing draws shared across all cues,
and paired arms. The same panel is reused across test horizons:

- R=0,1,2,4,8,16,32,64,128,256,512 with q1.
- R128q4 and R256q2, matching R512q1 at 6,144 body steps.

There are 13 conditions and 2,048 episode rows per fit/condition. Preserve logits,
labels, errors, forcing/arm identities, complete H64+C64 state after WRITE and
before QUERY, executed WRITE oracle and zero-state QUERY scores. Report per-bit
and whole-vector accuracy, periodic-minus-control bit error, and per-forcing
contrasts. The fixed primary is R32q1 minus-control error, R2q1 is the anchor,
and their difference in differences is secondary. All longer-horizon and load
curves are exploratory, not a newly selected primary endpoint.

Center covariance within forcing draw and arm across cue states; use FP64
sample covariance, raw H+C units, trace, spectra and PR with rank ceiling
min(255,128,2^D-1). Never pool forcing means into a memory dimension. Undefined
zero-variance PR stays null. A nonlinear encoding of D bits can have covariance
rank greater than D; PR is not information capacity.

Do not count repeated cue panels, time steps, bits or neurons as independent
training seeds. Report all three seeds, missing/failed fits and paired seed
differences. With three seeds, uncertainty remains weak and must not support
a broad population claim. Describe any first failing load/horizon as an
observed boundary on this grid. Native readout error does not establish lost
decodability; geometric contraction does not establish causal memory loss.
No Floquet spectrum or stable basin is inferred from this campaign. Subsequent
decoder/intervention and tangent studies need their own frozen contracts.

## Resource and execution contract

One native cluv first race; at most one candidate per currently ready CPU
cluster. Target Mila, Tamia, Narval, Rorqual, Fir, Killarney and Vulcan after fresh
scheduler/access checks. Nibi joins only if its dependency staging finishes
and the new source/environment can be verified; otherwise preserve pending
status. Narval access and three-CPU scheduler admission were freshly verified
at 23:12 UTC after the earlier authentication failure. Trillium's 192-core CPU
minimum and Trillium-GPU's mandatory accelerator are unsuitable for this
three-thread CPU experiment; neither is included merely to fill a cluster list.

Each candidate requests one node, three tasks, one CPU per task, 8GiB total,
zero GPUs and three hours, no requeue. The single winning allocation starts
three exclusive one-CPU/2GiB srun steps, one per seed, with unique seed output
directories. The previous four-fit CPU campaign completed in 314 seconds; its
measured performance supports CPU use. No GPU efficiency exemption is needed
because no GPU is allocated. The final candidate set is recorded before dispatch;
seven candidates imply 63 worst-case requested CPU-hours before cancellation
(eight imply 72). Actual race overlap and loser cleanup are reconciled separately.

All source/config/protocol/lock bytes are committed before `cluv sync`, and
verified on admitted targets before `cluv submit first job.sh`. New source
paths preserve the earlier campaign's checkout and artifacts. No dependency
download occurs inside the Slurm allocation. Every run gets a unique path
under `$SCRATCH/representation-collapse/runs/multibit-load-ladder/RUN_ID`.
The sibling `RUN_ID.STOP` stops all three seed workers. Internal deadline is
10,500 seconds, outer timeout 10,620 seconds, Slurm 10,800 seconds. Interruption
preserves checkpoints, partial results and truthful fit status; it is not
competence failure. No job retry or automatic extension is part of this race.

W&B required: true — monitored training/evaluation. Each fit logs interpretable
metrics from work start to the existing `representation-collapse` project,
initially offline. Synchronize the original histories and verify cloud values;
retain pending status until verified. Do not upload raw cues/states or weights.
Stage/scheduler checks and independent verification are not separate W&B runs.
