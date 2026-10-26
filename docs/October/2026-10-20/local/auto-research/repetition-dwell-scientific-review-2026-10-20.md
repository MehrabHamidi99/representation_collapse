# Scientific review of the repetition-length experiment

**Accept a descriptive failure of persistent graph tracking under extended repetition for the selected GC-LSTM checkpoint. Do not accept a representation-dimension collapse, reduced memorization-capacity, or Floquet-mechanism claim.** The question produced a useful positive behavioral result and a mixed, partly contrary geometric result.

This review read the completed compact outputs and inspected the absolute-phase/error PNG. It did not rerun native inference, recalculate spectra from binary states, or independently rerun the reporting verifier. The coordinator's audited execution is Mila job **11332452**, completed `0:0` in **11:26**, twelve conditions and 24576 targets, with 384 verified actual offline W&B histories. Online delivery remains unverified. Executed source: `3541ae02441357664a9328fcfebd515a472d9668`; actual result SHA: `92242b8ad720a3e42b512eceb64f75e4798d0a79a1969d180e932787c517a60f`; independently reduced result SHA: `7b9d0d432c8c50e627de12377c1852916eb54c23afdaf7b2c14ca9912a9bb8d5`. Inputs were the selected epoch-160 model and authenticated training motifs, with fresh zero states per condition and unchanged weights.

## Accepted behavioral finding

Mean target F1 improves from approximately 0.808 at dwell8 to 0.854 at16 and 0.889–0.890 at the training dwell32. It then declines for both start orders: **0.766–0.767 at64, 0.697–0.698 at128, and 0.656–0.657 at256**. Thus the complete six-point curve is not monotonically declining; the decline applies to extending repetition beyond the trained dwell32. The observed 32-to256 difference is approximately23 percentage points, not a seed-estimated effect size or confidence interval.

This decrease is not explained by a larger fraction of unavoidable graph transitions: persistence F1 excluding startup rises from approximately0.969 at32 to0.997 at256 as transitions become rarer. On nontransition targets persistence is perfect under this data convention. The trained recurrent model increasingly fails to track a graph that continues to be directly observed. Persistence and model all-horizon means have a one-target startup denominator difference, which is explicitly labeled and far smaller than the observed pattern.

The fixed two-step within-block contrast shows late improvement at dwell8/16 and deterioration at32 and longer. At32 the final-four-cycle A/B drops are approximately27.03/20.15–20.21 percentage points; at64 they are33.37/22.56; at128 approximately34.36–34.39/24.75–24.92. The magnitude broadly saturates rather than increasing strictly at every length. It must not be conflated with the historical14.05-point contrast, which used different early/late windows and a different initial replay state.

## Startup sensitivity at the longest dwell

At dwell256, four cycles span the entire2048-target run. Consequently the frozen final-four-cycle statistic includes startup. In startB, motifB's primary drop is6.254 points, versus24.643 in startA. The stored last1024-target exploratory supplement gives **24.496 points for B in both start orders**; its early/late F1 values are0.908613/0.663652. This supports startup sensitivity of the primary contrast. Retain the6.254-point primary result and report the supplement alongside it rather than silently replacing it or declaring a protocol exception.

The supplement has only four targets per early/late motif window at256, rather than eight in the primary windows. Across lengths its counts vary. It also shows nearly identical start-order means over the common final1024 targets: approximately0.892,0.769,0.700,0.660 for dwells32,64,128,256. This makes a purely initial-zero-state explanation of the overall extended-dwell deterioration implausible, while providing no independent-seed uncertainty. These supplementary results are exploratory; the frozen primary comparison is unchanged.

## Geometry does not establish dimension collapse

In these three-phase temporal clouds, **complete H+C participation ratio increases from early to late in every motif/start condition at dwells8 through128**. For example, at32/startA, A increases1.052→1.419 and B1.051→1.576 while covariance traces fall sharply. At64/startA, A increases1.247→1.788 and B1.071→1.407. Large loss of absolute trajectory variance therefore coexists with increased normalized dimension in these measurements.

At256, H+C participation ratios decrease slightly or substantially depending on motif/start, but the twelve-row clouds contain startup as well as later cycles. Their early covariance traces are strongly start-sensitive, including approximately112908 for the initial A motif in startA and74266 for initial B in startB. These pooled early/late clouds are not a clean isolated comparison of settled dynamics. Most numerical ranks reach the twelve-row centered ceiling11. No cross-history population dimension or retained-information capacity was measured.

These facts preclude a uniform geometric-collapse interpretation. Temporal variance, participation ratio, numerical rank and predictive usefulness must remain separate. The prior graph study used other windows; different estimates are not evidence of a software contradiction, but underscore window dependence. Native states were not independently reanalyzed in this review; the accepted reducer checked compact spectrum arithmetic only.

## Mechanistic reading and next discriminating tests

The absolute-age plot shows deterioration near the learned32-step scale and sustained lower performance during prolonged constant forcing. Much of the visible error growth is false positives, with relatively few false negatives. A learned transition expectation, a changing readout bias, or state evolution toward an unsuitable attractor are plausible alternatives. The graph alone does not identify which is responsible. The late response fluctuates rather than decaying smoothly to zero; exact instantaneous monotonic decay is unsupported.

Lag-period complete-state residuals also do not certify an orbit: their reported final-cycle maxima range roughly0.012–0.196 across conditions. No Jacobians or Floquet multipliers were computed. Neither contraction nor failure to preserve causal memory follows from these closure numbers.

The next mechanism experiment should freeze comparisons at matched absolute repetition ages, retain pre-threshold probabilities and graph-conditional error types, and test whether a deliberately early/late motif switch or a phase-aligned state intervention moves the error onset. Such tests can discriminate a learned timing/readout effect from a more general loss of useful state sensitivity. Multiple independently trained seeds are needed before generalizing beyond this checkpoint.

For the requested memorization claim, introduce independently written information that is absent from the repeated body yet required at a later query. Establish short-delay competence, compare periodic and matched nonperiodic bodies at equal retention horizons, vary dwell and delay separately, and measure cue recovery, complete-state history separation and causal interventions with shams. Stable cue-specific attractors are a valid memory-preserving alternative. A representation-collapse claim requires this useful-memory evidence; the present experiment supplies a strong behavioral motivation, not that proof.
