# Prelaunch review: frozen one-bit long-horizon extension

2026-10-23. The extension retains the accepted seed-0 model and native evaluator
from job 633041. It checks the previous checkpoint, result and evaluator hashes,
competence status, update count and loaded parameter digest before generating a
new held-out stream. It verifies the checkpoint/model hashes again after all
conditions. No optimizer or training call occurs. The six horizons and seed
4000 are frozen in [protocol v1](../../../2026-10-26/local/auto-research/onebit-long-horizon-v1.md).

The paired generator and native evaluator are reused without source changes.
One forcing draw crosses two cue polarities and both arms. At R512/q1, total
body duration is 6144 updates; R256/q2 and R128/q4 share that duration.
Controls match counts, bigrams, runs, endpoints and final motif. The exact
motif generator still changes higher-order ordering and predictability as a
bundle, so the result estimates that intervention rather than pure spectral
periodicity. Full H+C donor and self-copy checks are inherited. The output
keeps raw logits, schedule designs, state arrays, summary spectra, hashes and
offline W&B metadata on scratch; only compact evidence is fetched for review.

Twelve focused native/design tests pass. A full runner smoke with a synthetic
checkpoint and two forcing draws passed, including result creation and a
completed offline reporter. That smoke caught a non-scalar reporter config;
the grid is now bound by source/protocol hash and condition count is logged as
a scalar. The smoke is infrastructure calibration, not model evidence. The
actual Tamia environment, accepted checkpoint and scheduler still need live
verification. No result is accepted solely from a successful shell exit.

Scientific limitations are explicit: one trained seed, one stored bit,
dependent cue/arm rows, finite evaluated horizons, no independent state-array
reduction yet, no tangent/Floquet computation, and no memory-capacity test.
Perfect memory at R512 would remain a finite negative finding; loss in both
arms would be generic long-horizon forgetting until further controls.
