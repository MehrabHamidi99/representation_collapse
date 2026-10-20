# Fixed-checkpoint repetition-length diagnostic

Protocol `gclstm_repetition_dwell_v1`, frozen 2026-10-20 before running this
sweep or inspecting its outputs. This follows the accepted
[epoch-160 GC-LSTM reconstruction](figure10-gclstm-longfit-results-2026-10-20.md).
It is an exploratory, generated-forcing diagnostic motivated by already observed
within-motif decline. It is not an untouched confirmatory test, a retraining
comparison, a memory-capacity experiment, or a Floquet computation.

## Question and intervention

At fixed selected weights, does replacing `(A^32 B^32)` with longer or shorter
constant-motif blocks change native link-prediction performance and temporal
state geometry? Prespecified dwell lengths are `L = 8, 16, 32, 64, 128, 256`.
For each length, run both `(A^L B^L)` and `(B^L A^L)` starts, independently from
fresh positive-zero hidden and cell state. All 12 conditions have exactly
**2048 scored targets**, so elapsed horizon is matched. This gives respectively
128, 64, 32, 16, 8, and 4 complete input cycles per condition. There is no
additional burn-in, hidden-state reset at transitions, or optimizer update.

A and B are taken only from the authenticated training split: snapshot 0 and
snapshot 32, with canonical nodes 0–99 and the original graph edge convention.
Verify their identities against the pinned converted data and that they are
distinct; record hashes and edge counts. Construct every subsequent snapshot
from these two motifs. Do not open validation or sealed test payloads.
Authenticate and strictly load both components of the actual selected epoch-160
checkpoint. The separate generated-forcing loader is authorized for this
intervention; it does not bypass or reuse a held-out test opener.

Changing L changes transition frequency, phase frequencies and the temporal
input distribution relative to training; L≠32 is a schedule shift. Equal total
horizon controls elapsed time, not all of these changes. A new zero-start L32
condition is the internal reference. It is not the historical held-out replay,
which started from the exact state after 2816 consumed training/validation
snapshots. Both start orders are sensitivity analyses of the same trained model,
not independent model seeds or graph realizations.

## Execution and chronology

Use the accepted native GC-LSTM/head, fixed CPU FP32 parameters, evaluation
mode, deterministic settings and one intra/inter-op thread. Complete state is
H+C, with each component `[1,100,128]`; preserve component and canonical node
order. Score target G[t] from pre-consumption state s[t], then consume G[t] to
obtain s[t+1]. Score all 9900 ordered off-diagonal node pairs at threshold
`>=0.5`, exactly as in the accepted evaluation. Record TP/FP/FN/TN, F1 and
precision/recall with the existing zero-denominator conventions. Target t=0
is explicitly a zero-state startup prediction and must not be silently omitted.
Transition phase is q=t mod L=0; full-cycle phase is t mod (2L). Record actual
motif identity independently of start order.

No graph training, validation selection, model comparison or held-out test
inference occurs here. Retain all per-target scores, state norms, phase indices
and derived compact summaries. Large binary state trajectories remain scratch
only; do not copy weights, graph payloads or binary states into git.

## Behavioral estimands and plots

Report each start order separately before balanced descriptive averaging.

1. **Equal-horizon performance:** mean F1 over all 2048 targets; also report
   startup, transition and nontransition means with exact counts. Plot F1
   versus absolute target index and report mean F1 relative to the zero-start
   L32 reference. Averaging per-target F1 is distinct from pooling confusions;
   if pooled-confusion F1 is included, label it separately.
2. **Comparable late-window drop:** in the final four complete cycles, for each
   motif compare phases **q=1,2** with **q=L−2,L−1**. Each window has exactly
   eight correlated targets per motif and start order at every L. Define
   `drop = early mean F1 − late mean F1`; positive means deterioration.
   Report per-motif, per-start values and their balanced pooled means. Plot
   these against log2 L, including all negative and nonmonotone outcomes.
3. **Relative-phase behavior:** supplementary windows use
   `w=floor((L−1)/4)`, early q=1..w and late q=L−w..L−1, again in the final
   four cycles. These exclude transitions and do not overlap, but sample
   counts vary with L. Plot phase-averaged curves against q/L and retain
   absolute phase counts. Do not claim their contrast estimates the same
   absolute-age effect as the two-step windows.
4. **Within-condition settling:** per-cycle mean F1 and first-four versus
   final-four-cycle behavior, where reported, distinguish startup from later
   response. At L256 these windows coincide; do not count them twice as
   independent evidence. No monotone-trend claim without inspecting all
   prespecified L, motif and start-order contrasts.

Include the deterministic persistence control: at nonstartup targets predict
exactly the preceding graph. Compute its confusion/F1 using the same graph and
pair convention, rather than assuming disjoint motif edge sets. Mark startup
baseline undefined unless an explicit empty-graph startup rule is reported.
Its changing transition penalty reveals why overall F1 alone is insufficient.
Do not tune prediction thresholds after viewing these results.

## Geometry, closure and limitations

For final-four-cycle **geometry**, use three fixed phases per window: early
q=1,2,3 and late q=L−3,L−2,L−1. Each cloud has 12 rows per motif/start/length,
with centered rank ceiling11. Analyze H, C and concatenated H+C separately,
without coordinate rescaling. Use the existing frozen FP64 centered Gram
implementation (`tgrab_figure10_trajectory_analysis.covariance_cloud`): own-cloud
mean, denominator n−1, raw eigenvalues, tolerance
`64*eps64*max(n,d)*largest_eigenvalue`, explicit numerical-rank ceiling, trace,
PR, natural-log entropy rank, mean-state norm, and undefined normalized
metrics for exactly zero variance. Preserve the tolerance and removed positive
trace. Do not silently clip material negative eigenvalues. These geometry
windows deliberately differ from the two-phase behavioral windows and must be
labeled in figures and metadata.

All clouds have equal counts, but repeated cycles may be nearly identical and
provide little independent variation. With an exactly repeated orbit, three
sampled phases yield centered rank at most two regardless of the 12 rows.
A changing PR therefore measures temporal path geometry over those windows,
not retained bits, causal sensitivity or population dimension. Joint H+C
geometry depends on units and may be dominated by C. Report absolute variance
alongside normalized dimension; never infer information loss from PR alone.

Record per-target H/C norms and successive-state differences. Where available,
compare complete states at lag 2L using absolute and relative L2 residuals,
with zero denominator handled explicitly, and report the number and phases of
comparisons. This is descriptive closure, not certified Jacobian periodicity,
Floquet multipliers or contraction. No Jacobian is computed in this sweep.

One selected seed and one graph realization support no seed confidence interval,
bootstrap treating time steps as independent, p-value or generalization claim.
A monotone worsening with L would support a repetition-length-dependent failure
for this checkpoint under changed forcing. It would not by itself establish
reduced memorization capacity: predicting the next repeated graph often needs
only the most recent graph, not independently written older information.

## Execution provenance and acceptance

One cluv CPU job on Mila, account `mila`, partition `main-cpu`, QOS `normal`,
**1 CPU, 4 GiB, zero GPUs, two hours walltime**. Exact deadlines and argv are
frozen in the create-only admission record after source review. The initial
30-minute planning estimate was increased before launch to accommodate 24576
full-pair scores; it is not a numerical retry. Never truncate conditions silently
or selectively interpret a partial sweep. No automatic retry,
requeue or race. Unique scratch output and STOP path; fail on a STOP file,
deadline, nonfinite state, integrity mismatch or missing condition. Final source
revision, exact argv, checkpoint/data hashes, runtime identity, scheduler outcome
and elapsed time must accompany outputs.

Enter the mandatory offline ExperimentReporter before numerical payload access.
Verify actual finished W&B histories and result artifact; do not equate local
history with online delivery. Acceptance independently checks all 12 conditions,
24576 target rows, chronology, phase identities, confusion arithmetic, window
counts and independently recomputed summaries, source/input identities and
successful termination. State spectral checks are compact arithmetic checks
unless native states are actually reread and recomputed. Record this distinction.
Plot provenance must bind exact accepted results and CSVs, with no hand-edited
curves. Check both PNG and PDF rendering before publication.

## Separate next experiment: useful memory under repetition

A direct memory experiment must independently write information before the
repeated body and require it at the final query. Freeze a competence-first task
with multiple independently sampled cue values/bit strings, independent body
motifs and queries, held-out cue/body combinations, and a trained readout whose
query cannot be solved from the final body input alone. Demonstrate reliable
short-delay performance before interpreting long-delay loss.

At fixed weights and **equal retention horizon**, compare a periodic body with
matched nonperiodic orders of the identical token/graph multiset and matched
query timing. Vary motif dwell and total delay factorially so their effects are
not conflated. Include no-distractor and controlled-delay references, multiple
model seeds and independent examples. Measure query accuracy/bit recovery,
held-out cue decodability, covariance across independent histories at fixed
phase, and complete-state cue separation, keeping these distinct.

Test causal use with task-aligned state interventions and matched-norm/phase
shams, including H and C jointly. Stable attractor memory is an explicit
alternative: states may converge within each cue while retaining separation
between cues and correct query answers. Only loss of task-required information
relative to matched retention controls, with appropriate intervention evidence,
would justify a harmful memory-collapse claim. Protocol, competence threshold,
seed/sample counts and analysis must be frozen before that new task's held-out
results; this graph sweep cannot substitute for those gates.
