# Frozen descriptive within-motif analysis for the first Figure 10 reconstruction

Frozen 2026-10-19 before examining an actual reconstructed model per-phase
validation curve or any held-out model curve. The coordinator has seen an
aggregate first-epoch validation mean, which is acknowledged. The windows below
are fixed now and will not be chosen using the held-out trajectory. This
protocol describes the seed-0, one-graph, capped corrected GC-LSTM run; it
neither changes fitting nor permits another checkpoint selection/test access.

Use the independently accepted 256 held-out per-target F1 rows, absolute
timestamps 2816..3071, chronological score-before-consume evaluation, and all
9900 ordered off-diagonal pairs at probability >=0.5. Preserve all raw rows and
recompute F1 from their confusion counts. Include actual transition targets
at 2816+32j for j=0..7 in the full chronological plot and the means of eight
successive 32-target motif presentations.

For an additional phase-aligned panel, define within-motif phase
`q = target_timestamp mod 32` and motif A when `target_timestamp mod 64 < 32`,
otherwise B. For each motif and each phase q=0..31, average the four observed
F1 values at that aligned phase across the four test cycles. Plot the two
32-point descriptive curves and identify phase zero as the graph transition.
Retain these aligned means, counts and contributing absolute timestamps in a
separate CSV. There are four correlated repetitions of each motif, not four
independent trained models or graph replicates. Draw no uncertainty band and
perform no inferential test on those four cycles.

The fixed descriptive contrast is **late minus early F1**, where early phases
are q=1..8 inclusive and late phases are q=24..31 inclusive. Exclude transition
phase zero from both windows. For each motif, each window therefore contains
4 cycles x 8 phases = 32 target observations. Compute its arithmetic mean, then
subtract early mean from late mean. Also compute the overall contrast by
pooling the two balanced motif windows: 64 early and 64 late target values.
The overall contrast equals the arithmetic mean of the A and B contrasts
because their sample counts are equal. Retain exact window definitions,
counts, means and contrasts in the plot manifest. Preserve negative, zero or
positive contrasts without relabelling the windows or selecting cycles.

A negative contrast describes a within-motif decline in predictive F1 under
this fixed model and chronology. It does not establish representation-dimension
loss, useful-memory erasure, a periodic hidden orbit, a Floquet mechanism, or a
causal effect of repetition. Early/late differences can also reflect temporal
adaptation, calibration, graph-specific prediction difficulty and other state
dynamics. The two motifs can have different baseline F1; report both instead
of hiding their difference in an overall average. The eight presentation means
answer a different question—change across successive presentations—and must
not be substituted for this phase-aligned within-motif contrast.

The numerical deliverables are the complete per-target CSV, the phase-aligned
CSV, chronological and presentation-mean panels plus the phase-aligned panel,
and hash-bound metadata including the selected checkpoint and final ledger
head. Label the output as a single-seed capped corrected behavioral
reconstruction. Do not overlay an invented original curve, claim the original
seed roster or uncertainty convention, or count correlated timesteps as
independent replications. Subsequent independent model seeds require their own
frozen aggregation protocol; this analysis does not grant new fitting scope.
