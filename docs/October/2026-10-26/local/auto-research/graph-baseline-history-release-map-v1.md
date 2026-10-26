# Graph baseline and required-history release map

**Version 1 — 2026-10-16.** This map implements the staged separation requested
by the [Gate B readiness review](../../../2026-10-16/local/auto-research/gate-b-readiness-independent-review-2026-10-16.md).
It is a dependency and claim map, not an execution authorization. The original
T-GRAB checkpoint was not found in the enumerated TSA archive scope; this is a
bounded-search result, not proof of global absence. Exact reproduction is
therefore not currently available. Any new fit is a **fresh reconstruction**
unless its original source, data, run and checkpoint lineage is recovered and
verified. The graph reproduction audit records the scoped search and the
distinction ([audit](../../../2026-10-14/local/auto-research/tgrab-reproduction-audit-2026-10-14.md)).

The sequence below keeps ordinary next-graph prediction, runtime parity, the
unchanged vector Gate B mechanism study, and a future graph required-history
intervention as distinct research nodes. A pass at one node never silently
releases the scientific claims of another.

## 1. Original next-graph baseline: reproduction or reconstruction

**Question.** Can the reported T-GRAB GC-LSTM per-time and change-point F1
behavior be reproduced on the identified periodic graph conditions, especially
`Pdet(2,32)`, with the historical evaluation chronology faithfully represented?
AB (`Pdet(2,1)`), AABB (`Pdet(2,2)`), and `Pdet(2,32)` are separate conditions;
the draft’s AABB Floquet claim and the paper’s long-repetition behavioral curve
must not be conflated ([graph protocol](graph-reproduction.md),
[T-GRAB audit](../../../2026-10-14/local/auto-research/tgrab-reproduction-audit-2026-10-14.md)).

**Inputs and dependencies.** Pin the upstream T-GRAB source/data revisions,
environment, preprocessing, graph/task generation, seed roles, architecture,
training and selection rules, prediction chronology, candidate-pair mask and
metrics. For *exact reproduction*, also establish provenance linking the
original run, data, checkpoint and reported result. The archived TSA numerical
candidates are not yet linked to these identities; no original recurrent/link
head checkpoint was found in the enumerated TSA scope. If that lineage remains
unavailable, specify and label every irrecoverable choice as a fresh
reconstruction. Follow the historical and prospective chronology distinction
in the [graph reproduction protocol](graph-reproduction.md).

Only source/data/protocol audits and provenance searches may proceed before
node2a. Any new fit and validation-only checkpoint selection wait for the
deterministic pre-fit bootstrap parity fixture in node2a. Test targets stay
closed until node2b validates the actual selected checkpoint. Exact
reproduction can skip a new fit only when the original checkpoint and its run
lineage are recovered and pass node2b. A fresh reconstruction also requires a
separately frozen multi-seed replication plan; matching one inherited summary
is not replication.

**Acceptance evidence.** Immutable source/config/data/checkpoint identities (or
an explicit reconstruction declaration); an independent reproduction of
predeclared per-time and change-point F1 under the pinned chronology; persistence
and EdgeBank comparisons under the same pair mask; complete logs, seeds,
selection evidence, hashes and independently recalculable outputs. Exact
reproduction additionally requires verified original-run lineage, not merely
similar scores. Treat numerical archive arrays as separate candidates until
their producer and capture chronology are established.

**Stop/failure states.** Stop exact-reproduction labeling if the checkpoint or
producer linkage cannot be verified. Stop interpretation on target leakage,
incorrect period/phase, missing output provenance, unverified data conversion,
or a mismatch in candidate-pair/metric definitions. Preserve discrepancies and
valid failures; do not tune toward inherited figures. Such a stop may still
leave a clearly labeled reconstruction or a provenance finding.

**Claim boundary.** Successful next-graph F1 reproduction establishes behavioral
baseline agreement for the tested conditions and protocol only. It does not
establish arbitrary-prefix retention, causal memory loss, covariance dimension
collapse, tangent contraction, a Floquet result, or the vector Gate B estimand.
The T-GRAB audit explicitly reports no graph reproduction and no negative
replication to date.

## 2. Finite native checkpoint-load, replay and loss/gradient parity

**Question.** Does the object-free checkpoint path round-trip a deterministic
bootstrap fixture and later a selected trained checkpoint into the pinned
native model with the exact catalog, full-state chronology and numerically
matched replay outputs/loss and gradients required by the trainer contract?

**Inputs and dependencies.** The accepted selected-checkpoint schema is a
specification only ([schema](tgrab-selected-checkpoint-schema-v1.md),
[revision review](../../../2026-10-16/local/auto-research/tgrab-selected-checkpoint-schema-independent-review-2026-10-16.md));
the accepted codec review is design-direction only
([codec review](../../../2026-10-16/local/auto-research/tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md)).
Implement and independently verify the bounded encoder/decoder, nested artifact
semantics, trust-root and immutable selection-history checks, exact native
parameter/buffer/alias catalog, selected-load attempt accounting, observer
ownership, and pre-test replay-parity gate. The fixture must compare both hidden
and cell states for all nodes in canonical order and preserve the pinned source,
environment, initialization, input, and chronology identities. Use the
trainer/checkpoint contract and its required native call accounting; no test
target may be exposed before parity passes. Current conversion and S04 results
cover data conversion/provenance and serialized validation-row consistency;
they do **not** implement or verify this codec, native load, replay, loss, or
gradient parity ([status](../../../../../doc/auto/research-status-2026-10-16.json)).

**Node2a — pre-fit bootstrap.** Before any fit, create a deterministic
synthetic initialized state for the pinned architecture and fixed synthetic
inputs/targets. Check serialization, observed native state-dictionary catalog,
load, complete H/C capture, forward/loss/gradient equivalence and transaction
crash/reopen behavior without training, primary-data access or held-out targets.
Pin its seed, source/environment, initialization, expected arrays and hashes.
This tests the implementation path only; it does not validate selection or a
trained checkpoint.

**Node2b — selected-checkpoint parity.** After a separately authorized fit and
validation-only selection—or recovery of the original checkpoint with verified
run lineage—validate that exact immutable selected candidate through the same
native path. Compare saved/reloaded H/C state, predictions, loss and gradients
at the frozen boundary. Keep all test targets closed until node2b passes. Node2a
cannot substitute for this instance-specific check.

**Acceptance evidence.** Node2a requires independently reviewed codec/schema
and transaction fixtures, including malformed/truncated/swapped-component and
failed-publication cases; externally anchored hashes; observed native
state-dictionary catalog and two-component load; exact initialized/replayed
complete-state comparison on synthetic inputs; and predeclared
forward/loss/gradient parity checks and tolerances. Node2b binds these checks to
the actual selected checkpoint, its source/data/config/environment identity and
validation-only selection history, and verifies no test access before parity.
Record every failure attempt and terminal state. Schema validity or synthetic
fixtures alone do not pass node2b.

**Stop/failure states.** Stop before fitting if node2a fails on missing trust
identity, schema/catalog mismatch, unsafe or partial load, invalid transaction
history, hidden/cell or chronology mismatch, parity failure or missing
independent verification. Stop before test access if node2b fails or a selected
load attempt was not charged. No automatic retry after a terminal selected-load
or parity attempt. Keep failed artifacts and report the specific gate that
failed.

**Claim boundary.** Passing is engineering evidence that this pinned checkpoint
path and finite replay/loss-gradient fixture behave consistently. It does not
show model competence, reproduced paper scores, useful-history loss, collapse,
or a periodic orbit. It is a prerequisite for interpreting later trained graph
results, not a scientific result by itself.

## 3. Unchanged vector Gate B: assigned-order mechanism study

**Question.** At fixed weights and equal retention horizon, does the frozen
higher-order periodic distractor-body ordering cause excess loss of previously
written, task-required information versus its matched nonperiodic ordering, and
under what tested conditions is that information preserved?

**Inputs and dependencies.** Use the registered vector task, matched-body
construction, estimand, competence ladder, acquisition/readout controls,
selection rule, and measurement protocol without substituting a graph task or
altering the treatment. Preserve the accepted limitation: this estimates the
total effect of the specified assigned body ordering; current controls do not
isolate a pure periodicity mechanism because registered final-motif exposure
and possibly other higher-order statistics differ
([Gate B review](../../../2026-10-16/local/auto-research/gate-b-readiness-independent-review-2026-10-16.md),
[scope review](../../../2026-10-16/local/auto-research/ordering-mechanism-scope-independent-review-2026-10-16.md)).
Separate discovery/competence diagnosis from independently frozen confirmation
inference as specified in the
[experimental design](experimental-design-v2.md) and
[competence protocol](competence-training-v2.md). Required-history access,
complete state, causal corruption/rescue with shams, uncertainty and failure
handling remain distinct evidence requirements.

**Acceptance evidence.** Frozen protocol and source/config identity before
held-out inspection; competence on the required task; valid independent seed
blocks and paired uncertainty; all three estimands kept separate (state
geometry, tangent behavior, and task-relevant memory/performance); primary
named-horizon contrast with its frozen inference and multiplicity/equivalence
rules; causal interventions and matched shams for mechanism claims; preserved
failures and valid negative outcomes. Discovery results cannot be relabeled as
confirmation.

**Stop/failure states.** Stop scientific interpretation if acquisition,
short-retention access, competence, required-history dependence, state
accounting, probe-floor/precision robustness, or frozen confirmation validity
fails. Chance-level or ceiling-level forgetting is not evidence of intact
storage; a confidence interval containing zero is inconclusive unless the
predeclared equivalence conditions pass. No positive result is required to
complete the node.

**Claim boundary.** Results concern the vector task, registered assigned-order
contrast, tested population and horizon. They do not establish pure periodicity
as the isolated cause, graph-model behavior, universal recurrent collapse, or
cross-modal transfer. This node remains unchanged by the graph baseline work.

## 4. Later graph required-history intervention

**Question.** In a competent temporal graph recurrent model, does a controlled
distractor ordering impair a previously written graph-dependent cue that is
needed for a later query, and can a specified state-transition mechanism
predict loss and preservation under causal intervention?

**Inputs and dependencies.** This is a new, separately versioned protocol, not
ordinary next-graph F1 and not an assumed translation of vector Gate B. Define
the cue/write operation, graph distractor sequence and matched controls, query
and target such that identical current observations can require different
outputs depending on the designated earlier history. Verify immediate
acquisition, short-retention readout and native query access before long-delay
interpretation. Freeze motif exposure and all other treatment differences,
seeds, independent sampling units, phase alignment, node correspondence and
canonical ordering. Track every layer’s node-wise hidden **and cell** state;
separate covariance, finite-time tangent propagation, performance and causal
recoverability. Establish input, state and Jacobian periodicity/closure before
using Floquet terminology; otherwise label propagators finite-time. Require
the complete checkpoint-parity stage (node 2 in this map), including native
checkpoint/load/replay/loss-gradient parity, before trained results are
interpreted. See the
[active goal](active-research-goal.md), [graph protocol](graph-reproduction.md),
and [workflow contract](../../../../../skills/experiment-loop/references/workflow.md).

**Acceptance evidence.** A preregistered graph task with demonstrated dependence
on the varied history; competent acquisition and readout; matched controls and
the stated query/target chronology; complete-state artifacts and independent
recalculation; paired uncertainty over valid independent units; causal
corruption/rescue and matched shams for mechanism claims; robust measurement
calibration and frozen confirmation rules. Report preservation boundaries,
nulls, failures and limitations as first-class outcomes.

**Stop/failure states.** Stop if the task does not require the cue, the current
input or target leaks the answer, controls fail matching, competence or short
retention fails, complete-state capture is missing, parity is unverified, or
numerical measurements fail calibration. If periodic trajectory/Jacobian
closure fails, Floquet claims stop while correctly labeled finite-time analysis
may remain possible. Do not infer graph-memory effects from a next-graph score,
input spectrum, low-rank trajectory, or unpaired run.

**Claim boundary.** A valid result is conditional on the graph model, task,
ordering contrast, horizon and tested seeds. It cannot establish that all
temporal graph models or modalities collapse, that periodicity alone caused an
effect where controls leave other ordering differences, or that local tangent
contraction is necessary for memory loss.

## Current release order and actionable work

The next scientific release is not an experiment submission. The conversion
closeout and S04 checker do not release checkpoint parity: typed checkpoint
artifact semantics/codec and observed native parity remain absent. The most
useful single next local task is to implement the selected-checkpoint object-free
codec and transaction/semantic validation against the accepted schema, with
adversarial fixtures and independent review. Then complete node2a before any
new fit. A fit still requires a frozen protocol and separate owner compute
authorization; node2b must pass before held-out evaluation. This map itself
grants no model, data, compute, or cluster authorization.

## Evidence references

- [Active research goal](active-research-goal.md)
- [Gate B readiness review](../../../2026-10-16/local/auto-research/gate-b-readiness-independent-review-2026-10-16.md)
- [T-GRAB recovery and reproduction audit](../../../2026-10-14/local/auto-research/tgrab-reproduction-audit-2026-10-14.md)
- [Graph-first reproduction protocol](graph-reproduction.md)
- [Research charter](research-charter.md)
- [Vector competence protocol](competence-training-v2.md)
- [Frozen experimental design v2](experimental-design-v2.md)
- [Accepted selected-checkpoint schema specification](tgrab-selected-checkpoint-schema-v1.md)
- [Independent schema revision review](../../../2026-10-16/local/auto-research/tgrab-selected-checkpoint-schema-independent-review-2026-10-16.md)
- [Independent codec design review](../../../2026-10-16/local/auto-research/tgrab-selected-checkpoint-codec-independent-review-2026-10-16.md)
- [Research DAG and artifact contract](../../../../../skills/experiment-loop/references/workflow.md)
- [Current structured status](../../../../../doc/auto/research-status-2026-10-16.json)
