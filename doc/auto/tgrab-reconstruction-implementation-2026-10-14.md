# T-GRAB reconstruction: local implementation and explicit integration gaps

2026-10-14. Status: **local contract implementation complete; independent
review pending; native parity, runnable training and scientific acceptance
absent**. This task was bounded to four exclusive paths: the new module, its
focused test file, this report, and ignored preparation evidence. Root owns
tracker/notebook integration. No source snapshot, frozen protocol/config,
environment, other module or cluster state was modified.

The implementation follows the reconstruction specification/config published
at `351fa688f3333f0e44701d15d2acf9a2553a4b98`:

- `doc/auto/tgrab-checkpoint-reconstruction-v1.md`, SHA-256
  `698a3150566e47a1866308f7b9782ab2b2b1f4878b6592acb978972cb9f1107b`.
- `configs/graph/tgrab-reconstruction-v1.json`, SHA-256
  `22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.

Primary remains Pdet(2,32), period 64; AB and AABB remain distinct unexecuted
conditions. The candidate is one new seed-0 reconstruction with L1/H128/K2,
one-hot100, and both recurrent and link-head weights. Limits remain prospective
pilot bounds: 200 complete epochs, 511800 updates and 3600 seconds for the
entire pilot. No configuration or source identification reproduces the old
checkpoint bytes without its actual checkpoint/RNG/environment provenance.

## What is implemented

`src/representation_collapse/graph_reconstruction.py` is importable using the
project's NumPy dependency without importing Torch, PyG or Temporal. It has no
CLI, scheduler command, launcher, dataset converter or implicit checkpoint load.
The complete canonical candidate JSON is SHA-256 checked against its frozen
content, alongside explicit architecture, source and finite-limit checks.

The state contract includes hidden and cell `[layers,nodes,hidden]`, canonical
ascending integer node IDs and the next target timestamp. The synthetic
flattening reference uses hidden then cell, layer, node, feature; its full
candidate dimension is 25600. Immutable snapshot edge tuples reject duplicate,
diagonal, out-of-range and noninteger positive pairs. Conversion must resolve
source self-loop markers explicitly before constructing these snapshots.

Negative sampling enumerates admissible off-diagonal destinations for each
positive source against the **entire** positive edge set. It samples uniformly
with replacement using a dedicated NumPy RNG, permits repeated negatives,
checks source alignment/collisions, and fails finitely on saturated rows or
empty positive training targets. The dedicated stream is independent of global
NumPy draws. This implements the declared prospective collision/diagonal
correction, not a replay of the faulty public negative sampler.

The causal controller resets every training epoch, detaches h/c before each
one-step consume, consumes t0 as conditioning, and scores/updates each target
before that target can be consumed. Validation freshly resets and replays the
training prefix under the backend's evaluation mode, then scores each target
before consuming it exactly once. It retains no evaluation continuation for
the next training epoch. The test opener is called only after finalized
validation-only selection and selected-weight loading, and opens at most once.
An evaluation failure does not permit another test opening.

Metric code enumerates all ordered off-diagonal pairs, including 9900 for 100
nodes, threshold >=0.5, zero-division=0, per-time confusion counts and F1,
arithmetic time-mean F1, source/target phases and actual graph-equality
transition flags. Expected target transition phases are 0/32 and source
phases 63/31; the primary test timestamp interval contains eight transitions
among 256 targets. A distinct-pattern/data manifest must establish that the
phase prediction agrees with actual graph equality; a phase flag alone is
not that evidence.

Selection rejects test metrics, partial epochs, nonfinite values and
nonincreasing epoch numbers; strict maximal validation F1 keeps the earliest
tie. Checkpoint references require hashes for recurrent and link-head
components. Strict relative training-loss improvement, a zero-loss denominator
floor, plateau patience, owner stops and finite epoch/update/walltime bounds
are separate components. Partial epochs never increment the complete-epoch
count. The controller and policy components do not constitute an experiment
runner: integration must enforce full native split counts and connect the
epoch loss, selection and patience policies to durable artifacts.

## Actual native source is preserved; native execution is blocked

The lazy factory verifies then imports the actual public `MultiLayerGCLSTM`,
`LinkPredictor` and `NodeEmbeddingModel` source files from pinned T-GRAB commit
`31293102b065feef7817ff86211427d7594edfdb`, with exact SHA-256 identities:

- `model/gclstm.py`: `86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609`.
- `model/link_predictor.py`: `cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86`.
- `model/node_emb.py`: `4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29`.

The installed GC-LSTM cell must equal Temporal v0.54.0, commit
`46abd39abec0cd8de23c0e489a617c320c3fc5e8`, with cell source SHA-256
`c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af`.
It imports the genuine cell through the exact source wrapper, rechecks the
imported cell, and provides no alternate recurrent implementation. Cached
wrapper imports must match their exact path and source hash. No upstream
trainer or source module was imported/executed during this implementation.

The prepared native backend delegates recurrence and the precise source head
to those modules. It preserves h/c and verifies `z == hidden[-1]`, generates
matching unit edge weights and identity features, applies real train/eval
module flags, uses positive-mean BCE plus negative-mean BCE, and prepares
fresh deterministic named Adam parameters. Seeding precedes recurrent/head
construction and Adam construction. Nonfinite state/loss/gradient/parameter
checks fail rather than silently changing precision or algorithms. These native
methods have **not executed**, and their graph-convolution, gradient, head
dropout, numerical compatibility and checkpoint parity remain unverified.

Preflight uses distribution metadata and source text hashes only. It requires
pinned independent receipts binding the exact config for environment,
conversion, engineering review, scientific review, W&B reporting and fresh
admission. Public source requirements are treated as an explicit initial
environment gate; changing dependency versions requires a recorded reviewed
deviation. The native factory always fails closed on the deliberate
`native_checkpoint_and_reporting_runner_not_implemented` blocker, even if
someone supplies all other receipts. Removing that blocker is a future reviewed
implementation change. `load_selected` raises instead of calling `torch.load`.
The existing frozen config still disables launch.

## Focused synthetic checks and dependency evidence

Ran once successfully:

```text
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_graph_reconstruction.py' -v
```

**25 tests passed in 0.139 seconds**. Fixtures use three-node arrays, event
spies and small integer pair/phase contracts; no native recurrent module,
link head, optimizer or training-data generator was constructed or called.
A fresh-interpreter import check confirmed no Torch/PyG/Temporal import.
The native factory refusal check confirms failure before any native import.
Checks cover complete h/c flattening, node ordering, positive/negative
collisions, duplicates/saturation, RNG independence, metric threshold/pair
counts/time averaging, phase offsets, training/evaluation chronology,
fresh construction call order, test closure, selection, strict plateau
patience, hard caps and partial epochs. They are **synthetic engineering
evidence**, not native graph parity evidence, a trained-model result, or a
Floquet/collapse measurement. Native loss-gradient/source-wrapper equivalence
and native complete-state round trips remain acceptance gates.

The earlier invocation omitted `PYTHONPATH=src` and failed test discovery with
`ModuleNotFoundError` before the new module loaded. It is preserved in
`runs/review/tgrab-reconstruction-implementation-20261014/test-attempt-1.json`;
the corrected outcome is `test-attempt-2.json`. No failed native model run
or optimization attempt occurred. Source syntax/import was exercised by the
focused suite; whitespace inspection reported no issues.

Read-only dependency metadata, captured in `dependency-probe.json`, found
Torch **2.6.0**, NumPy **2.4.6**, and missing PyG, Temporal, torch-cluster,
torch-scatter, torch-sparse and torch-spline-conv. Installed cell identity is
unknown. Public source requests Torch2.0.0/NumPy1.24.4/PyG2.6.1/Temporal0.54.0
plus those extensions. The project manifest requires NumPy>=2.2 and its
experiment group pins Torch2.6.0. A graph environment therefore needs its own
locked, reviewed setup or an explicit prospective protocol deviation; the
current vector lock is not an accepted graph environment.

`preflight.json` records `ready=false`, disabled launch, eight dependency
mismatches/missing distributions, unverified installed cell, six missing pinned
readiness receipts, and the unfinished native runner. No native packages were
imported by that metadata pass and no data payload was read. The W&B skill
requires reporting from the first numerical/training step and exempts unit
tests from creating experiment runs. No SDK smoke/experiment ran here; reporting
integration and its verified online/offline receipt are required before later
native execution. There was no provider call or cluster operation.

## Handoff and next bounded task

Frozen module SHA-256:
`b61f32bd674055032cf22c30803b1537b84cb1a0583260f560289418af93c6d8`.
Frozen focused-test SHA-256:
`654a55e60a03e597605870912647087f7d562b50a98e7ac39d116afe4db9b63f`.
Ignored receipt binds these, the frozen input identities, all three wrapper
pins, the cell source, fixture names, execution outcomes and readiness metadata.

Next: independent static/engineering review of this coherent implementation;
resolve the graph environment and data-conversion manifest separately; then
implement/review native checkpoint schema/hash handling, per-run RNG state and
named-parameter manifests, atomic selected/resume/failure/status writers,
W&B reporting and the full epoch/validation/patience orchestration with exact
native split enforcement. Only after appropriate separate admission can native
tiny parity/gradient/dropout/round-trip checks run. Training/hosting subsequently
requires exact executable commands and a concrete fresh envelope. The expired
shared allocation and the separate archival-array audit supply no authority
for those operations. Root config/protocol remain unchanged and disabled.

## Addendum: bounded lifecycle repair 1

Independent review found two defects in the initial frozen component snapshot:
`training_epoch` permitted additional updates after `test_opened`, and
`test_once` did not recheck walltime/owner STOP after selected-checkpoint loading
before calling the test opener. Root authorized only these guard repairs and
necessary focused spy checks. The initial source, tests and report were copied
byte-for-byte into
`runs/review/tgrab-reconstruction-implementation-20261014/repair-1/prior/`;
its `identities.json` binds the original SHA-256 identities above. The initial
main receipt and both previous invocation records remain unchanged.

The repair makes test opening terminal for training as it already was for
validation. It checks the pilot budget/STOP again after `load_selected` and
before `open_test`, and after the opener returns before any state reset or
conditioning replay. Test access already begun remains terminal even if that
last check stops the evaluation. Three focused spy regressions demonstrate no
post-test training events, no test opener after load-induced walltime/STOP, and
no reset/replay after opener-induced walltime/STOP. There were no other source
changes, no new configuration decisions and no added native operations.

The same focused command passed **28 tests in 0.101 seconds**, including the
original 25 and three added methods. The actual tool result and full stdout/
stderr stream are saved as `repair-1/test-tool-result.json` and
`repair-1/test-output.txt`. No broad suite or native experiment ran. Native
construction remains fail-closed; environment, conversion, checkpoint/reporting
runner, native parity and execution admission gaps are unchanged.

Repaired source SHA-256:
`89894f8bd3e8574b5b57c777353fe1aac3a1f123f792467a009b04b8a6f9bcdb`.
Repaired test SHA-256:
`9b1f55a5f3e57ac8d48914500eada783071f7820b41a63f06a52c766eba89eb2`.
The additive `repair-1/receipt.json` binds old preserved snapshots, these new
files, the updated report, focused check evidence and unchanged frozen inputs.
