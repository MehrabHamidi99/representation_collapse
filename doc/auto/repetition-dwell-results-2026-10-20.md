# Longer repetition: verified GC-LSTM results

**Longer blocks beyond the training dwell of32 substantially impair this
checkpoint's prediction performance. This experiment does not establish loss
of memorization capacity or a general decline in representation dimension.**

Cluv Mila job **11332452** completed with exit`0:0` in **11min26s** oncn-f003.
All twelve conditions and **24,576 scored targets** were independently reduced.
The actual offline W&B stream contains **384 verified history records** and the
correct result artifact. Online delivery is not verified.

## Experiment

The [frozen protocol](repetition-dwell-protocol-2026-10-20.md) varies the lengthL
of constant graph blocks in `(A^L B^L)` and `(B^L A^L)`. L is8,16,32,64,128,256.
Every condition starts from zero H+C and runs2048 targets with the same
validation-selected epoch160 GC-LSTM and head. Neither weights nor thresholds
change. A/B come from the authenticated training split; this new generated
schedule diagnostic opens no validation/test payload and performs no training.
All9900 ordered off-diagonal node pairs are scored before each target is consumed.

There is one trained seed and one graph realization. Starting order and repeated
cycles are sensitivity conditions, not independent model replications. L≠32
changes the temporal distribution relative to training. The zero-startL32
reference differs from the original Figure10 replay initialization.

## Behavioral findings

Balanced averages over the two starting orders, using all2048 targets:

- L8: mean F1 **0.807702**; persistence **0.875427**.
- L16: mean F1 **0.854149**; persistence **0.937958**.
- L32: mean F1 **0.889429**; persistence **0.969223**.
- L64: mean F1 **0.766452**; persistence **0.984856**.
- L128: mean F1 **0.697467**; persistence **0.992672**.
- L256: mean F1 **0.656541**; persistence **0.996580**.

Persistence excludes its undefined startup prediction; model means include it,
as labeled. The model loses about **23.29 F1 percentage points** fromL32 toL256,
while persistence improves. Thus the deterioration cannot be attributed simply
to more transitions or an inherently harder graph prediction schedule.
Performance is best near the trained length, not monotone over the entire grid.

The prespecified within-block contrast compares phases1–2 withL−2–L−1 in the
final four cycles; positive early-minus-late means deterioration. Balanced over
both motifs and starting orders, the drops are respectively **−6.13, −4.36,
23.61, 27.97, 29.61 and24.97 percentage points**. The first two negative values
mean improvement within short blocks. For longer blocks, deterioration grows
throughL128 and then the primary pooled contrast decreases atL256. **Do not
report a strictly monotone increase in the drop.** These two-step contrasts
are not directly comparable to the original Figure10 report's eight-step
windows and replay initialization.

L256 has only four cycles total, so its primary final-four-cycle estimate
includes startup. MotifB/startB has a primary drop of6.25pp, versus24.64pp for
motifB/startA. In the supplementary last1024-step window, both give about24.50pp.
The balanced last1024 drop is29.82pp. This supports startup dilution as an
explanation for the primary discrepancy, but does not replace the registered
primary result. The last1024 calculation existed in the executed worker; it is
an exploratory elapsed-window supplement, not a newly substituted primary
estimand. Overall last1024 F1 also decreases beyondL32: **0.891894,0.769074,
0.700298,0.659890** at32,64,128,256.

[Behavior figure](figures/repetition-dwell-2026-10-20/dwell-behavior.png) ·
[normalized-phase curves](figures/repetition-dwell-2026-10-20/dwell-phase.png) ·
[absolute-phase errors](figures/repetition-dwell-2026-10-20/dwell-absolute-phase-errors.png).

The absolute-phase curves show substantial false-positive growth around the
trained32-step timescale, followed by lower performance with oscillations in
longer blocks. False negatives are generally much smaller outside transitions.
A learned timing/readout bias is a plausible competing explanation; these plots
do not prove an internal clock, attractor, memory erasure or causal mechanism.

## State geometry does not supply the missing memory claim

Each state cloud uses twelve correlated observations: three phases across four
cycles, analyzed separately for hidden, cell and complete H+C. Coordinates are
unrescaled and centered within the cloud. The centered rank ceiling is11;
exactly repeating three phases would have rank at most2.

Complete-state participation ratio **increases** from early to late in every
motif/start condition forL8 throughL128. AtL256 it decreases, with substantial
start-order sensitivity in motifB. Within-window covariance trace is lower late
than early across the sweep, but its ratio is not monotone inL. Less absolute
variation therefore coexists with increased normalized dimension in much of the
grid. This does not support a uniform dimension-collapse interpretation of the
performance loss. The window definition also differs from earlier Figure10
state measurements, so their PR signs need not agree.

[Complete-state geometry figure](figures/repetition-dwell-2026-10-20/dwell-geometry.png).
Full H/C/HC spectral summaries are retained. Independent verification here checks
compact spectral arithmetic; the saved binary state clouds have not been
independently recomputed. No Jacobian, Floquet spectrum, independently written
cue, causal state intervention, or memory-load capacity curve was measured.

## Accepted conclusion and next tests

Accepted: **a fixed GC-LSTM trained at dwell32 tracks longer repeated graph
blocks increasingly poorly in mean F1 over32–256, despite an improving simple
persistence control; late-block deterioration is substantial but its primary
size is not uniformly monotone.** This is descriptive evidence for repetition-
length-dependent predictive failure under changed forcing in one checkpoint.

Open: whether old task-required information is lost, whether the readout instead
anticipates a transition, and whether any effect generalizes across training
seeds or schedules. Predicting a repeated graph can be solved from the most
recent graph; it is not a direct test of older-memory capacity.

Next discriminating steps are:

1. Train matched checkpoints at different dwell lengths and variable dwell,
   then repeat this fixed inference grid. Test whether the deterioration
   timescale tracks the training period rather than an intrinsic memory limit.
2. Validate tangent measurements on complete H+C and relate finite-time
   perturbation propagation to the error onset, without calling them Floquet
   multipliers before trajectory/Jacobian periodicity checks.
3. Implement the [competence-first independent-WRITE memory candidate](repetition-memory-next-experiment-2026-10-20.md):
   retain a bit through matched periodic/nonperiodic bodies at equal horizon,
   require native QUERY recovery, and use complete-state swaps with shams.
   Only after reliable one-bit competence should load be varied to measure
   operational capacity. This candidate is documented, not yet executed.

## Provenance and verification

Execution source`3541ae02441357664a9328fcfebd515a472d9668`; accepted checkpoint
manifest`4abeba2b64261468c8844ee302570d6c5d48cec873a51f9eafd1e6010f75db61`;
result`92242b8ad720a3e42b512eceb64f75e4798d0a79a1969d180e932787c517a60f`.
OneCPU/4GiB/zeroGPU, two-hour maximum, no retry/requeue/race. MaxRSS614352K.
The immutable launch and actual scheduler receipt accompany the public figures.
Original checkpoint, graph payloads and state clouds remain on cluster scratch.

The first local reduction attempt failed during W&B initialization because the
sandbox prohibited its IPC socket, before numerical reduction. That failed
receipt is preserved. A distinct permitted offline reduction completed; the
cluster experiment was not rerun. Independent reporting acceptance was written
only after its own reporting finish succeeded. Thirteen worker/reducer tests
passed. All four PNGs and all four PDF renderings were visually inspected.

[Scientific review](repetition-dwell-scientific-review-2026-10-20.md) ·
[verified numerical summaries](figures/repetition-dwell-2026-10-20/results.json) ·
[all target metrics](figures/repetition-dwell-2026-10-20/target-metrics.csv) ·
[artifact hashes](figures/repetition-dwell-2026-10-20/inventory.json).
