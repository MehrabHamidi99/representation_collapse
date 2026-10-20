# Primitive-v2 selected inference, evaluation protocol v3

Frozen execution contract, 2026-10-20, after independent implementation
review and before the long-fit candidate's held-out evaluation. The separately
recorded admission binds its exact source and CPU allocation. This evaluates one corrected seed-0 reconstruction;
even a fit reaching its 200-epoch cap is not the original paper's complete
multi-seed Figure 10 or historical-run reproduction. It adds no causal memory,
representation-collapse or Floquet claim and changes no training policy.

## Actual identity and authority

The worker is
`python -m representation_collapse.tgrab_figure10_v2_evaluate_job` with protocol
`tgrab_figure10_primitive_v2_selected_evaluation_v3`. It consumes the original
`tgrab_complete_epoch_candidate_v2` six-file bundle from immutable long-fit
source `1891bdd59bf47b944c8799ae345ebb26b9a4e497`. Its execution contract is
the actual long-fit v2 contract, SHA
`aec185fe544bf94a1dc0ea5e952d27b0d335c3653d815f7c0796d97b747ea381`.
The unchanged numerical template remains a separate identity, SHA
`22bef6bf781c5bfb5493dd5e03169ea0e658b2463ccaa9ba09189af454508f1f`.
Template identity does not assert v1 launch/admission compatibility.

The caller supplies the independently accepted long-fit receipt and its SHA,
actual fit run/attempt/admission identities, fit summary SHA, authoritative
`selection/selected.json` SHA, actual fit supervisor receipt path, and native
source/environment/data/conversion pins. Acceptance kind is
`figure10_long_fit_execution_acceptance_v2`; the four independent gates must
be true: finalized validation selection, Git source-closure authentication,
actual offline history verification and optimizer-history verification.
Both the actual long contract and the numerical template are explicit inputs.
Every runtime module named by that contract is rehashed before execution.

`selected_pointer_sha256` is the actual primitive-v2 pointer. No v1 selection
head, v1 admission, v1 provenance manifest or replacement hash is constructed.
The worker reopens the real immutable decision/finalization receipts and
intents, verifies every predecessor and execution identity, reconstructs the
strict running validation maximum with earliest ties, and compares the entire
final pointer to its external SHA and accepted fit summary. Unexpected
selection namespace entries stop execution. This check performs no training
or test access.

The independent acceptance's compact-file SHA map authenticates run identities,
seed/post-construction/negative-generator records, module-mode and reporting
records. The original supervisor authenticates source-long closure and actual
job. Freshly constructed models independently provide native state-dictionary
versions, parameter order and initial component semantic hashes. These values
form the strict primitive-v2 decoder's external expected identity. Candidate
self-assertions are not copied into that authority.

The data-only v2 decoder verifies all six original files, original manifest and
provenance semantics, both native components, complete terminal H/C bytes,
validation chronology and counters. The shared tensor catalog is the actual
unchanged encoding layout; using it does not relabel a v2 candidate as v1.
Only the shared numerical loader and complete-state comparator are reused
from the earlier evaluator. The v1 ledger/orchestration and v1 candidate reader
are not used. The generic GR checkpoint reference is used only to reserve its
cooperative sealed test capability and contains the real v2 component hashes.

## Reporting, one attempt, and numerical invariants

Mandatory explicit offline W&B is entered before numerical imports, native
construction or graph-data opening. Project is `representation-collapse`, with
a fresh unique run ID/output directory. Missing or failed reporting stops the
run; no online credential lookup or sync is attempted. The outer launcher owns
source closure, real Slurm allocation and deadline/STOP enforcement.
The proposed envelope is one CPU, 2 GiB, zero GPUs and ten minutes; admitting
that exact job is a separate recorded operation, not authority from this file.

Before model construction, bundle payload reading or native selected loading,
an fsynced create-only `selected-evaluation-claim.json` is written under the
actual fit root. This common filename applies across evaluator versions and
binds fit acceptance, actual admission, pointer, candidate and evaluation run.
It prevents a second pass under a different output name. Failures after that
barrier remain spent; no repair, retry, alternate candidate, state resume or
second test opener is allowed. A fresh evaluation event records one selected
load charge and zero test-read charges before any selected bytes are loaded.

Fresh native construction repeats seed 0 and verifies the exact initial and
post-construction Python/NumPy/Torch RNG observations, initial recurrent/head
semantic weights and module metadata. The seed-1 negative-generator record is
checked against a fresh generator solely as initialization evidence. Saved
RNG, optimizer state and terminal H/C are never restored. No optimizer is
created. Shared strict loading prechecks both complete CPU FP32 catalogs and
storage separation, loads both recurrent and head weights, verifies every
post-load byte and verifies evaluation flags.

Under evaluation/no-grad mode the complete positive-zero H/C state consumes
train 0..2559 then validation 2560..2815 exactly once at fixed selected weights.
All hidden and cell bytes at s[2816], node IDs 0..99 and next-target 2816 must
match the original selected terminal artifact exactly, including signed zero.
No tolerance, hidden-only comparison, saved-state substitution or extra replay
is allowed. Only after the durable replay-verification record may test access
be charged, then the one lazy sealed opener invoked.

For each t=2816..3071, observe a detached independent clone of s[t], score all
9,900 ordered off-diagonal pairs at threshold >=0.5, compute the existing
predecessor/phase/confusion/F1 metrics, then consume G[t]. The passive observer
is unchanged: 256 complete pre-score H/C states, shape [256,2,1,100,128],
FP32 little endian, 26,214,400 bytes. It preserves partial output on failure
and publishes only after exact count/order, fsync, stable original inode and
independent actual-byte SHA readback. No extra model, graph or test calls arise.

Reductions remain the unweighted mean of 256 target F1 values and separate
8 transition/248 nontransition means; expected and actual changes must agree.
No new posthoc windows, statistical tests, training adjustments or causal
interpretation are introduced.

## Retained outputs and acceptance boundary

New hash-chained event kind is `figure10_primitive_v2_evaluation_event_v3`.
The successful sequence is charged -> loaded -> replay_verified -> test_read
-> metrics_complete -> complete. Events bind the actual fit admission/pointer
and evaluation identity. Decoder/initialization failure may leave only the
charged prefix plus a failed outer manifest. A load/replay/test/observer failure
records failed. A later capture/publication/report failure leaves retained
partial evidence and a failed outer manifest; it cannot become accepted by
the presence of a metrics-complete event alone.

`evaluation-inputs.json` retains the original v2 manifest/provenance and actual
accepted selection identities. `evaluation-result.json` retains all rows,
reductions, real candidate SHA and pointer SHA, explicit
`checkpoint_format="primitive_v2"`, actual long-contract/template hashes, and
the complete-state trajectory descriptor. `checkpoint_sha256` is a generic
content-ID alias of the original v2 manifest's `candidate_manifest_sha256`;
it is not a v1 manifest identity. No `final_head_sha256` alias is generated.

All 256 raw metric rows, initialization/completion events and reductions are
logged offline with monotonic steps. A compact result artifact binds the actual
fsynced result SHA, candidate, pointer, accepted fit, long contract, numerical
template, data manifest, evaluator source and state-trajectory SHA. Neither
weights, graph payloads nor binary states are uploaded. The outer success
manifest is produced only after reporter finish and records actual reporting,
result and trajectory identities. Offline completion does not verify dashboard
delivery.

Independent v3 execution verification and a version-aware plotting acceptance
are still required. Old v1 evaluation receipts are not manufactured or reused
as authority for this new execution. The v3 result supports the existing
descriptive state reader through its real rows/state descriptor; a new outer
analysis admission must explicitly accept its versioned execution receipt.
The one retained trajectory has rank ceiling 255 before any tighter phase
ceiling. Timesteps/cycles are not independent seeds; scientific acceptance,
historical reproduction, causal erasure and Floquet certification remain false
at this execution layer.
