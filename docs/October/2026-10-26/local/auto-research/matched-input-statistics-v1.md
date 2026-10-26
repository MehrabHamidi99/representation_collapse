# Realized input-order diagnostics

2026-10-16. Additive implementation of the descriptive input/encoder-statistics
requirement in [experimental design v2](experimental-design-v2.md), sections
1.2–1.3 and 3B. This changes no frozen task, control, competence rule, model,
training schedule, confirmation statistic or compute allocation.

`src/representation_collapse/input_diagnostics.py` provides:

- `body_statistics(body, q=1, max_lag=24)`: validate registered dwell-expanded
  A/B/X motifs; retain symbol counts, cyclic bigrams/trigrams, X-skipping cyclic
  letter bigrams, linear run-length histogram, proper whole-body tiling period
  and number of distinct motifs; compute centered feature statistics for the
  ordered A/B/X one-hot inputs. WRITE and QUERY are excluded.
- `matched_body_audit(periodic, control, repeats=..., q=1, max_lag=24)`:
  require the declared equal horizon, repeated periodic motif, non-tiling
  control at R>1, identical R0/R1 anchors, matched low-order statistics,
  start/end context and final motif. Retain realized higher-order differences
  and the number of final-motif exposures in each arm.
- `sequence_statistics(features, max_lag=24)`: descriptive measurements of
  caller-supplied, explicitly typed NumPy input/encoder features, with no model invocation.
  Future orchestration must bind their checkpoint/encoder source and verify
  capture identity. Mixed Python sequences are refused before array coercion,
  because that coercion can round a large integer before a precision check.
  Caller-owned construction of an already rounded array cannot be detected.
  This helper does not independently verify an encoder.

For a finite feature matrix X with T updates, center each feature by the
whole-body time mean. Cyclic cross-covariance at lag k is
`centered.T @ roll(centered, -k) / T`, with the lag direction and denominator
explicit in the result. The boundary wraps last to first by definition; it is
not a recorded extra recurrent update. Lags stop at min(max_lag,T−1); an empty
R0 sequence has no lags or mean. These are input-time statistics, not
cue-conditioned recurrent-state covariance or independent sample estimates.

The summed one-sided power uses squared rFFT magnitudes divided by T², doubling
interior positive-frequency bins while retaining DC and, for even T, the
Nyquist bin once. Its sum equals mean squared centered feature norm. Frequency
is cycles per update. Normalized power divides by that measured energy; zero
energy has `normalized_power_defined=false` and null normalized power, including
empty inputs and constant features. No noise/variance floor is fitted. Values
are computed in float64 with input dtype recorded; extended floating precision
and integers outside the exact float64 range are refused, as are arithmetic
overflow, nonfinite values, complex state features and excessive work requests.
Floating underflow/roundoff can limit measured energy; zero measured energy is
not a proof of mathematical constancy or memory loss.

Bounds are 4096 updates,128 feature dimensions,65536 feature scalars, maxlag128,
at most1e6 reported covariance entries and128e6 nominal covariance product
terms; dwell expansion is at most64. Bounds are implementation limits rather
than allocation authority. Full output is an artifact, not a W&B scalar map.
Caller-owned numerical runs initialize reporting and bind source/config before
using the helper. Synthetic unit tests require no experiment SDK run.

Eight focused tests passed after the final integer-precision guard, including
all20 distinct R2 motif pairs, R0/R1 anchors, dwell2, exact cyclic counting,
literal wrapped-covariance checks, scale/offset controls, odd/even Parseval and
Nyquist weighting, undefined normalization and invalid inputs. One registered
R2 pair has identical cyclic trigrams and another differs. Thus the older
design table's shorthand “arms differ at trigram order by design” must not be
interpreted as a guarantee for every realized pair; section1.2 already records
this qualification. These are synthetic engineering checks, not new trained
model results or an empirical survey.

Input periodicity, FFT lines, unequal high-order statistics and validated
matched counts do not prove periodic hidden trajectories, Floquet stability,
retained cue loss, causal use or a collapse mechanism. The helper is not yet
integrated into a new admitted training/evaluation campaign. Independent review
of its implementation remains a separate handoff step.
