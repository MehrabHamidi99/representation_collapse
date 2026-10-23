# Frozen one-bit long-horizon extension v1

2026-10-23, prospective protocol. Use the accepted seed-0 checkpoint from
Tamia job 633041 exactly once, without retraining or checkpoint selection.
The accepted fit and first held-out result are documented in
`onebit-memory-results-2026-10-23.md`. This extension asks whether its perfect
one-bit retention continues beyond 384 body updates, and whether repetition
ordering causes excess error relative to matched nonperiodic order.

Use `onebit_memory.make_model` and `onebit_memory.evaluate` unchanged with
CPU/FP32 deterministic Torch 2.0.0+cpu, NumPy 1.24.4, one thread. Verify the
trusted checkpoint SHA-256
`1af080a2cef09b68d627ef4389f2b2a1d03983098b10f8e08d38063d2e281c9e`,
prior result SHA-256
`ab36f689f337d57ebfe165a523c772c4a6dd7404155781dadf3b5e92270886ba`,
prior status `competence_passed`, update 5000, and both pre/post load model
digest. The checkpoint and state arrays remain on Tamia scratch. New held-out
stream: NumPy default_rng(4000), separate from all prior streams.

Freeze six q/R conditions, in order: `(64,1), (128,1), (256,1), (512,1),
(256,2), (128,4)`. The latter three have equal body length 6144 steps. Each
condition draws 512 forcing schedules, with both bit polarities and both arms,
so 2048 episode rows per condition. Arms match counts, bigrams, runs,
endpoints, final motif and duration under `vector_task.paired_body_ids`.
Sample with replacement; identical motif patterns across draws remain possible.
Keep the exact native complete H+C path, row-level logits/interventions, and
raw pre-QUERY state capture from the accepted first run. Check exact self-copy
and donor replay. Score both arms by cue and pooled accuracy/CE; report paired
periodic-minus-control error for **all six conditions** and all native/control
shams. Primary contrast is R512/q1, with R64/q1 anchor. No favorable subset,
model update or retrospective horizon selection.

The sample unit is one forcing draw; bit flips and arm copies are dependent.
This is one trained seed and a single stored bit. Report full H+C cue
separation, raw pooled covariance trace and PR only as descriptive geometry.
If either arm fails, do not infer erasure from native readout without probes
and precision controls. If all arms remain perfect, call it a finite tested
negative result, not unbounded retention. No Floquet computation is included.

One cluv Tamia CPU submission: account `aip-ioannism`, QOS `normal`, partition
`cpubase_bycore_b1`, one CPU, 4 GiB, zero GPUs, two-hour cap, no race/requeue
or automatic retry. The worker uses the existing hash-locked scratch CPU
environment, a fresh immutable source checkout, a unique output directory,
an internal 7080-second deadline and STOP file. Require frozen source identity,
all expected rows, inventory hashes, scheduler `COMPLETED 0:0`, actual offline
W&B history/receipt, and independent arithmetic before interpreting any result.
