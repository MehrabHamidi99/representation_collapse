# Frozen long fit v2: corrected seed-0 GC-LSTM

Frozen 2026-10-19 before examining the first pilot's actual held-out curve.
The owner explicitly authorized retraining without the earlier compute budget
limit. This is a new fresh fit, not a modification or extension of the already
launched one-hour v1 attempt. Its executed contract is
`configs/graph/tgrab-long-fit-v2.json`; the unchanged reconstruction-v1 config
is a **numerical and conversion template**, never the executed v2 training
contract. New primitive bundles identify version 2 and both identities.

Keep the pinned public GC-LSTM wrapper/cell, native fit adapter, Adam update,
model seed 0, negative seed 1, CPU FP32, single CPU thread, one snapshot per
batch, predecessor chronology, exact accepted graph bytes, node ordering,
40/4/4 periods and validation-only strict earliest-tie selection unchanged.
Stop at 200 complete epochs, corrected training-loss patience 100 with
relative improvement tolerance 1e-6, any integrity/numerical/reporting failure,
owner STOP, or the new 48-hour operational ceiling. Exactly 2,559 successful
updates constitute an epoch. Work ends by 172,680 seconds; shell ends by
172,770 within 172,800 Slurm seconds. The root coordinator will bind the
actual immutable source, inputs and allocation through its stdlib supervisor
and cluv submission on Mila: account mila, main-cpu, QOS normal, one CPU,
2 GiB, zero GPUs, no retry/requeue/race. No new allocation is launched by this
protocol file or the worker's local tests.

The first 256 native update losses must exactly equal the independently
accepted v1 prefix reference in the same pinned CPU runtime. Pin the reference
JSONL and its hash in admission. Compare each confirmed update before allowing
the next one; retain durable failure evidence and stop on the first mismatch.
Publish a prefix-parity receipt only after all 256 comparisons pass. This is
an engineering equivalence gate, not another experiment chosen from test data.
No reporting, RNG observation, mode probe or checkpoint operation may consume
additional model RNG values before these updates.

Mandatory offline W&B initialization precedes native imports, graph payload
reads and training. Each optimizer intent is fsynced before the step; each
outcome is fsynced after it. A fresh append-only epoch chunk journal with
hash-chained records and complete-epoch seals bounds file count to about two
files per epoch. Persist raw loss, every validation row and epoch summary
before reporting them. Training, validation and complete-epoch reporting use
a separate strictly increasing event counter. Never automatically retry a
failed callback, reopen the same run or resume a partially updated model.

After a complete training epoch, reset to fresh zero hidden/cell state and
replay the full training prefix under fixed epoch-final weights in evaluation
mode, then score all 256 validation targets before consuming each target.
Publish only complete candidates: both model components, exact terminal H/C
at s[2816], all validation rows and actual provenance. Use the unchanged
source-derived finite FP32 tensor catalog for serialization, but a **new
versioned primitive-v2 manifest**, not a v1 schema with false cap/config claims.
The manifest records actual executed-v2 contract hash, unchanged numerical
and conversion template identity, pinned dataset/environment/source/admission,
initial RNG observations and weight semantic hashes, actual modes, journal
seal and complete-epoch counters. Stage create-only, verify every file SHA
and byte count, then atomically publish a complete candidate directory. The immediate readback
checker verifies file inventory, sizes and hashes; it is not an independent
selected-inference semantic decoder or admission receipt. The future v2
loader must separately validate the catalog, metadata cross-identities,
chronology and actual native tensor load/replay.

Select only the maximal complete-epoch validation time mean F1, choosing the
first epoch on a tie. Persist immutable decision receipts and an atomic
selected pointer; verify their predecessor and candidate identities. A later
partial epoch cannot replace a complete selected candidate. Finalization
records the actual stop reason, complete epochs and all charged partial
updates. Failed or unmatched optimizer outcomes remain charged with unknown
mutation status. No optimizer or RNG resume state is claimed or written.
The selected bundle contains inference weights and a diagnostic terminal
state, not a resumable checkpoint.

This worker has no held-out opener or evaluation path. A later v2 evaluator
will require independently accepted fit artifacts, strict dual-component
load, independently pinned selection head, actual initialization/source
checks and fresh complete-state prefix replay before a separately admitted
single test pass. Do not relabel these v2 candidates as v1 to reuse the old
loader. The user asks to reproduce Figure 10, but this 200-epoch corrected
seed-0 attempt still differs from the manuscript's uncapped patience rule,
historical device/seeds and potential implementation bugs. Report cap-bound
outcomes and valid negative findings. F1 alone never proves representation
collapse, and no held-out curve will choose this fit's duration or checkpoint.
