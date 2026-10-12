# Manuscript revision and scientific readiness

Post-manuscript continuation, 2026-10-12: the
[independent periodicity reduction](periodicity-independent-results.md) is now
verified and published separately with a reproducible figure. Its finite-time
contraction and arithmetic-sensitive cue separation remain descriptive, with
all original competence failures retained. Probe-floor checks and manual QUERY
arrays are still unavailable. Additive coordinate-wise competence utilities
are tested; training integration and checkpoint selection remain incomplete.
The 15-page manuscript below retains its earlier evidence cutoff; its appendix's
"reduction pending" statement describes that cutoff. No new periodicity result
has been inserted into the paper or promoted to a Floquet/collapse claim.

The owner requested a more focused, precise conference paper with plots and
selected a venue-neutral ML format. The new manuscript is **When Does
Repetition Erase Memory? Identifiability Limits for Recurrent-State Collapse**.
It replaces the 35-page research-program skeleton with a seven-page main
argument, references, and detailed appendices (15 pages total). Complete prose and explicit
scope replace anticipated-result placeholders.

## What the paper now establishes

The contribution is a reproducible diagnostic framework, assembled from known
mathematical ingredients and explicit constructions. Fixed affine dynamics
cannot produce a body-order effect on the conditioned cue channel at equal
horizon; commuting switched-affine transitions share that null under matched
counts. Native error can nevertheless differ through offsets. Covariance
participation ratio, local contraction and cue recoverability can dissociate.
A controlled ordering design specifies the comparison a competent trained
model would need to pass. The recorded pilots instead establish failure of
native competence, without localizing acquisition, retention or readout.

“Identifiability limits” is used in the defined diagnostic sense: a particular
proxy need not determine whether task information survives. It is not presented
as a new formal statistical-identification theorem. The elementary swap bound
is a standard telescoping upper bound, not a positive prediction of periodic
harm or a solution of the broader Gate C problem.

## Four figures and their evidence

1. WRITE/body/QUERY schematic: matching and timing, with an explicitly
   illustrative four-motif body.
2. Four analytic controls: isotropic attenuation, contracting categorical
   basins, affine readout offsets, and noise-dependent ordering risk. These are
   deterministic constructed systems. The ABAB/AABB example does not satisfy
   the tighter experimental motif-matching rules and is labelled accordingly.
3. Native pilot behavior: exact committed reductions of four seed-0 cells.
   Categorical arm/model overlaps are explicit; continuous MSE is divided by
   the zero-predictor MSE on those same saved targets. No seed uncertainty is
   inferred from episodes, horizons, coordinates or driver panels.
4. Probe-floor artifact: an invertible scalar code with declared noiseless
   train/test designs. The ridge error curve is derived analytically and
   checked against the actual v1 implementation at 101 amplitudes. It identifies
   a possible estimator confound, not its magnitude in the pilot results.

The generator, `code/analysis/make_paper_figures.py`, records all plotted values,
source/output hashes, formulas, package versions and checks in
`paper/figures/manifest.json`. Generated Appendix C (source filename
`D-pilot-details.tex`) includes all 32 pilot clouds rather than hiding
unfavorable cells. PDFs and PNG previews reproduce byte-for-byte in the locked
paper environment. Model weights and raw cluster arrays remain on scratch.

## Independent review and corrections

Separate workers owned narrative, theory/proofs, and figure/results files.
The narrative worker then reviewed the theory and figure assumptions without
editing the other worker's files. The coordinator integrated and accepted the
following corrections:

- Unrestricted decoding-risk invariance under rotation needs a rotationally
  invariant full noise law; the example now specifies independent isotropic
  Gaussian noise, not merely isotropic covariance.
- The linear-risk construction explicitly states zero-mean noise independent
  of the cue, and records its variance sweep and possible sign reversal.
- The offset-loss formula retains the residual/offset cross term; a random
  paired offset stays inside the expectation. Compensation is body-specific.
- A small mean reconstruction error does not guarantee positive local gain;
  exact smooth-family neutral modes retain their stronger hypotheses.
- Native competence failure is not called evidence of failed acquisition;
  the R2 probes already show surviving cue information.
- Generated table rows use valid TeX terminators. No draft evidence badges,
  TODO result slots, unverified legacy figures or scheduler narrative remain
  in the main paper.

Known antecedents remain prominent. Stable phase-locked memory was already
analyzed in [Pals, Macke and Barak (2024)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011852).
Repetitions and multiple stable driven responses are central to
[Ashwin and Ceni (2023)](https://arxiv.org/abs/2309.04728), while information
relative to noise is central to [Ganguli, Huh and Sompolinsky (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2596211/).
These primary sources were spot-checked during revision; this is not a new
exhaustive literature search. The paper removes the bibliography's internal
“venue not verified in this audit” wording without claiming a publication venue.

## What still prevents a strong conference submission

Editorial readiness and scientific readiness are separate. All four trained
cells fail competence and have one seed; the analytic ingredients substantially
overlap prior work. Accordingly, the current manuscript is a coherent
methodological draft, not a verified learned-collapse result. It should not be
represented as meeting a top-conference novelty/evidence threshold merely
because it has polished figures and proofs.

For the intended empirical mechanism paper, the decisive missing work is:

1. Competent GRU/LSTM baselines in both arms under a frozen curriculum and
   coordinate-wise rule, including an explicit policy for failed seeds.
2. Independent periodicity-array reduction with complete-state, precision and
   probe-floor checks; no empirical Floquet spectrum is currently accepted.
3. Fresh-seed confirmation with a frozen effect-size/uncertainty procedure;
   a near-zero contrast from failed baselines is not an equivalence result.
4. A prediction or boundary beyond the assembled classical controls, supported
   by direction-specific intervention/rescue with shams and appropriate
   held-out or native-task transfer.

A methodology-focused venue may value the current diagnostic framework, but
its novelty must still be assessed against the cited antecedents. No new
compute, model-service call, or scientific claim acceptance follows from this
manuscript revision. The broader research charter retains its publication gates.
