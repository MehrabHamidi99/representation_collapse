# Graph-first reproduction protocol

Version 1, 2026-10-11. Status: protocol ready; empirical reproduction not run.

Source-pin follow-up, 2026-10-12: the
[bounded upstream audit](tgrab-upstream-pin-2026-10-12.md) and
[machine-readable record](../../configs/graph/tgrab-upstream-pin.json) now fix
code, dataset metadata and dependency revisions, and verify AB/AABB generator
semantics. Dataset payload hashes and the legacy checkpoint remain unverified.
The audit identifies conditional eval-only current-target warm-start chronology,
prediction caches beyond the core h/c state, and cached test-F1 participation in
stopping. New scientific training must freeze validation-only stopping and
native chronology; a historical reproduction records the original behavior.
The preparation text below retains the prior source-audit snapshot.
The goal is to reconstruct the inherited GC-LSTM behavior and then test
dimension loss with calibrated measurements. Failure to recover a draft's
numbers is a reportable result, not permission to tune against those numbers.

## Sources and naming

The benchmark manuscripts define `Pdet(k,n)`: `k` distinct snapshots, each
repeated `n` times, period `T=k*n`. The Floquet manuscript instead uses `N`
patterns and `k` repetitions. Use config fields `num_patterns`,
`block_length`, and `period_steps`, and preserve both legacy notations in
reports. AABB is `Pdet(2,2)`, while `Pdet(2,1)` is AB. The preliminary text
mentions both; identifying the exact checkpoint/task is a recovery prerequisite.
Sources: `ACM___TGRAB/kdd/tex/050periodic.tex:18–32` and
`floquet_analysis/main.tex:69,102–107,443–455`.

The local appendix identifies [T-GRAB code](https://github.com/alirezadizaji/T-GRAB)
and [dataset](https://huggingface.co/datasets/Gilestel/T-GRAB).
As checked on 2026-10-11, upstream advertises dataset generation, DTDG
GC-LSTM training and `scratch/res` outputs. This is a recovery lead; no
upstream commit or dataset revision is pinned here yet and no upstream result
has been verified. Do not run its batch submission scripts during acquisition.

## Recovery gate before training

- Obtain an immutable upstream commit and dataset revision/checksum manifest;
  inspect the GC-LSTM cell, graph preprocessing, temporal trainer, checkpoint
  schema and exact dependency versions. Preserve upstream license/attribution.
- Search the authorized source/scratch locations for the checkpoint, training
  command/config, data split/seeds, state dumps and analysis code producing
  `floquet_analysis/figs/interpret/`. The inherited manuscript contains pictures,
  not numerical provenance. Do not invent a checkpoint identity.
- Map manuscript sources to active LaTeX includes. Both `kdd/main.tex` files
  select `_ai` section variants; the original sections and appendix remain
  protocol evidence. Resolve discrepancies in a versioned decision record.
- Record hidden width, layers, full graph state shape, directed/undirected
  convention, edge-pair candidate set, self-loop treatment, graph convolution,
  feature encoding, activation/head, loss, optimizer and truncation/detachment.
  Published parameter counts are a cross-check, not a full architecture spec.

If artifacts are absent, label the work **reconstruction** and list choices
that cannot be recovered. Do not call it exact checkpoint reproduction.

## Task and training reconstruction

The detailed inherited sections specify 100-node ER deterministic patterns at
edge probability 0.01, one-hot node identities, and edge features of one.
Periodicity uses 40/4/4 complete periods for train/validation/test; other task
families use chronological 80/10/10 splits. The appendix reports one message
passing layer, Adam at `1e-4`, one-snapshot batches, five random seeds and
training-loss patience 100 without a fixed epoch cap. See
`ACM___TGRAB/kdd/tex/050periodic.tex:76–80` and
`ACM___TGRAB/kdd/tex/100appendix.tex:28–37,105–106`.

Resolve the ambiguous seed list, width, GC-LSTM implementation version,
initial/reset state, validation carry-over, gradient clipping, detach horizon,
negative sampling for training, link threshold and checkpoint-selection rules
from pinned code or recovered configuration. Record whether seed changes
affect graph generation, initialization, data order, or all three. A new list
chosen for reconstruction is explicitly a new list.

For reproduction, retain the recovered training-loss stopping behavior and
validation-selected checkpoint, but impose a declared maximum walltime or
update count for each authorized pilot. A capped pilot is not full training
parity. Keep test data closed until selection. The legacy text evaluates test
at each validation improvement; reconstruct its numbers if required while
using validation alone for new decisions and reporting the distinction.

## Evaluation chronology and behavioral controls

Make the prediction target explicit: observing `G_t` must not expose the
target `G_{t+1}` through features, labels, graph edges, cached neighborhoods,
or state advancement. Unit-check output-before/after-update timing. Confirm
the persistence baseline under the same chronology; overlapping patterns
need not produce zero change-point F1.

Enumerate the declared possible node pairs at every test step, compute F1 per
step, and average over time as in `040method.tex:36–37` (also `_ai` Evaluation
Protocols). Report all steps, change points, and per-phase/time curves. Freeze
prediction threshold using training/validation only. Report denominator,
pair direction, duplicate removal, self-loops and empty-positive convention.
Do not replace this with sampled-negative F1 or pooled-count F1 silently.

Start with persistence, EdgeBank, a memoryless graph encoder/readout, untrained
GC-LSTM and a known sufficient-state control. Preserve each baseline's exact
definition; a phase oracle is an upper-bound control, not a learned recurrent
baseline. Node-ID features allow graph-pattern recognition and therefore do
not isolate autonomous counting. Include history-gap or phase-ambiguity tasks
where the same current input requires different outputs depending on history.

## Measurement and calibration gate

Implement the exact complete-state recurrence. For one-layer node-wise LSTM
hidden/cell tensors of shape `[nodes,hidden]`, joint tangent dimension is
`2*nodes*hidden`, unless recovered architecture proves a different shape.
Graph convolutions can couple nodes. Per-node Jacobians and pooled-state
covariances are explicitly separate analyses. Retain node order and flattening
manifest with every trajectory.

Validate analytic linear products (including noncommuting steps), rotations,
uniform contraction, singular maps and a nonnormal transient-amplification
fixture on CPU. Confirm full-period autodiff, stepwise products, JVP and
finite-difference directional derivatives agree within precision-aware
tolerances. For larger states use composed JVP/VJP, but compare a tiny graph
against dense products first. Report eigensolver residuals and the number of
requested/converged pairs; an incomplete spectrum does not support a global
contraction conclusion.

Replay deterministic ground truth with frozen weights in evaluation mode;
record dropout/numerical settings. Burn in and measure closure over multiple
cycles and initial states. If state/Jacobian periodicity fails, use finite-window
products. Fresh stochastic SBM sequences are not deterministic periodic orbits.
Archive absolute/relative closure residuals and all numerical diagnostics.

Measure hidden, cell and joint activity spectra, PR, covariance entropy,
variance-explained dimensions, absolute variance, and finite-window tangent
singular spectra per [research program](research-program.md). A four-state
closed orbit bounds centered time-only rank by three; that bound alone is not
harmful dimension collapse. Compare matched sampling across independent
graph sequences and required-memory variables.

## Pilot and expansion

1. Freeze sources/config and pass CPU numerical/chronology checks.
2. Run one bounded seed on AB and AABB to resolve the legacy mismatch, using
   recovered settings where available. Explicitly distinguish smoke, pilot
   and finished training in artifact labels.
3. Freeze the primary condition before independent replications. Default
   target is five seed blocks to match the manuscript's replication count;
   specific IDs and graph/model seed factorial remain a protocol decision.
4. Expand one dimension at a time: block length for counting, pattern count
   for memorization, and delayed cause/effect lag for retention. The inherited
   cause/effect family uses lags `{1,4,16,64,256}` and predictable memory-node
   edges (`060cause.tex:23,31` and active `060cause_ai.tex:39`); use a bounded subset for a pilot with deviations
   declared. Stochastic tasks enter only after deterministic measurement works.
5. Analyze paired seeds and controls, then review whether the behavioral and
   representational evidence support contraction, synchronization, sufficient
   compression or harmful collapse. Null/contradictory results stop unsupported
   claims and still release the next analysis milestone.

Each cluster plan states exact command/config, dataset/checkpoint paths,
environment identity, outputs, expected state shape, seeds, CPU/GPU count,
memory, walltime, total reservation cap and retry policy. Source readiness and
scientific calibration precede submission. Submit through the repository's
cluv workflow; external template GPU/wake quotas do not transfer. Requested
Compute Canada allocation approval belongs to the concrete compute plan,
not to this scientific protocol. Preserve failed attempts and partial files.

## Acceptance

**Artifact acceptance:** complete provenance; correct prediction chronology;
finite expected metrics; splits and pair counts match; trajectory shape is
verified; cycle/numerical tests passed or fallback clearly labeled; hashes and
run/job identity available.

**Reproduction acceptance:** the recovered source settings and reported legacy
condition are identifiable, and reconstructed task metrics/spectra are compared
with the inherited figures and numbers with uncertainty and all deviations.
Report discrepancies without rewriting inherited values.

**Scientific acceptance:** independent reviewer checks required-information
controls, dimension sampling/normalization, seed uncertainty and intervention
interpretation. The draft's `|mu|≈0.41–0.44` for six pairs
(`floquet_analysis/main.tex:502–510`) is **unverified preliminary evidence**.
Even reproducing it would establish only the measured modes' local cycle
contraction; it would not establish loss of a clock, lost memory or universal
representation collapse.
