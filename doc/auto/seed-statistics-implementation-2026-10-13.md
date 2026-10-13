# Prospective seed-statistics reducer — 2026-10-13

This is locally tested implementation infrastructure for the
[statistics proposal v2](statistics-freeze-proposal-v2.md). The proposal remains
unfrozen. No confirmation observations were read, no checkpoint was selected,
no compute/model service was invoked, and no learned-model claim was accepted.
Independent scientific and engineering review remain coordinator decisions.

## Implemented contract

[seed_statistics.py](../../src/representation_collapse/seed_statistics.py) has
three analysis entry points:

- `paired_behavior_effects` accepts R2/R16 loss and pair-ID arrays in fixed
  `[query coordinate 0..7, episode]` order. It checks elementwise arm pairing,
  unique IDs within each horizon, finite nonnegative losses, binary categorical
  indicators, and raw continuous squared-error units. It averages paired raw
  loss differences, then divides continuous effects by the fixed population
  variance `1/3`. Coordinates receive equal weights. The usual 128 episodes
  per coordinate is the default; smaller fixtures are explicitly labelled as
  differing from the proposal. R2/R16 are paired by training seed without an
  assumption of episode pairing across horizons.
- `seed_t_summary` reports the mean, sample SD with denominator `n-1`, SE,
  ordinary 95% and Bonferroni-per-eight-slot 99.375% t intervals, and the
  upper-tail t probability at the proposed SEI of `0.02`. These are explicitly
  approximate reports with unverified calibration assumptions. Zero seed SD,
  an underflowed tail, or an unrepresentable statistic/interval makes numerical
  inference unavailable; a finite descriptive mean is retained.
- `reduce_seed_family` requires the caller's prospective seed list and fixed
  GRU/LSTM × categorical/continuous identities. It calculates `delta16` and
  `DID = delta16-delta2` within each seed, retaining all eight family slots.
  Holm uses the full eight-slot family, stable ties in family order, and the
  proposal's alpha `0.05`. Its outputs are named *approximate candidate*
  calculations and never accept harm.

Each record has `model`, `task`, `trained_seed`, boolean `validation_eligible`,
an optional explicit `unavailable_reason`, and an optional `outcome` packet.
An available packet contains task-matched `loss_units`, R2/R16
`arm_mean_losses` for `periodic`/`matched_control`, and R2/R16
`paired_mean_raw_loss_differences`. The episode reducer produces this packet.
The family reducer checks the paired differences against the separately
reported arm means within eight floating-point ULPs of their scale, and uses
the paired differences for effects. This is an arithmetic consistency check,
not verification of raw arrays, source chronology or genuine pairing.
Optional supplied `effects` are checked against the reconstructed effects.

Explicit paired differences preserve a small contrast when separately
averaging very large continuous losses loses it to rounding. A numerical
fixture with pair differences `[2,2,0,0]` around arm losses `1e16` has raw mean
difference one and normalized effect three; subtracting its rounded arm means
gives zero. This fixture verifies arithmetic behavior only. It does not endorse
these losses as a competent model or identify a trained-model mechanism.

Missing records, validation ineligibility and explicit unavailable outcomes
make both primary slots of that cell inconclusive with `decision_p=1`. No
available-case primary mean/interval is created. Finite observed outcomes of
an ineligible seed remain in its ledger as `observed_outcome_effects`, without
primary eligibility or subset averaging. Other complete cells still receive
the original eight-slot correction. Duplicate records, unknown replacement
seeds, malformed units and contradictory effects raise errors. The caller
must preserve raw failed artifacts and attempt history outside this reducer.

## Numerical calculation and independent checks

SciPy is neither declared in `pyproject.toml`/`uv.lock` nor installed in the
project virtual environment. Dependencies were unchanged. Student-t survival
probabilities use the incomplete-beta identity with a safeguarded continued
fraction; at `|t|<1`, a complementary small-argument formula preserves the
CDF slope. Outside that genuinely central range, the direct upper-tail beta
formula prevents subtraction from erasing small tails at large degrees of
freedom. Positive critical values use bracketed bisection. The inverse API
supports upper-tail probabilities in `[1e-100,0.5)` and checks its relative
survival residual within `2e-11`; smaller positive tails raise `ArithmeticError`
rather than returning an underflow boundary. These numerical limits do not
affect the proposed reporting probabilities `0.025` and `0.003125`. The
[SciPy Student-t integral documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.stdtr.html)
and [NIST incomplete-beta definitions and identities](https://dlmf.nist.gov/8.17)
provide mathematical reference points. Numerical evaluation of the specified
t distribution does not establish validity of a seed-effect t inference.

The command
`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_seed_statistics.py -v`
passes **21 tests**, zero skips; the final focused run took 0.067 seconds.
[Tests](../../tests/test_seed_statistics.py) include:

- Analytic df=1 Cauchy and df=2 CDFs, including near-zero inputs; independent
  192-node Gauss-Legendre integration of the t density at df=3/9/29; the
  official SciPy df=3 reference value; and analytic Cauchy inverse quantiles.
- Independent review found an inverse-tail failure at df=1 and `p=1e-200`:
  squaring the beta argument underflowed, giving approximately `6.36e161`
  instead of the analytic `3.18e199`. Regression checks now verify the analytic
  inverse through `1e-100` and explicit unsupported-tail errors through
  `1e-300`. The reviewer also identified central-mass cancellation at `t=10`,
  df=1000; its repaired direct tail matches an independently calculated
  80-digit reference `8.335351479300033e-23` within `2e-12` relative tolerance.
- The reviewed ten-seed interval coefficients `1.1215044595072001` and
  `0.7153569059706628`, checked within `2e-12`, plus a rationally calculated
  fixture mean and sample variance, tested against independent density
  integration for its one-sided probability.
- Pair mismatch/duplicate/missing checks, binary-indicator and raw-MSE unit
  checks, continuous population normalization, and the large-arm-loss
  cancellation fixture.
- A within-seed DID fixture with varying endpoints and exactly constant DID,
  so subtracting unrelated endpoint intervals would lose the pairing.
- A p-value between `0.05/8` and `0.05/2`: six missing family slots must prevent
  the rejection that a reduced two-slot family would produce.
- Missing/ineligible/numerically failed seed handling, retained ineligible
  observations, no seed replacement or duplicate rescue, fixed tie order,
  explicit zero-SD/underflow behavior, and finite JSON output.

Source SHA-256:
`257d00e8f8b8416e5a062adac21ac9c80a396b17cd054844829eeaabdd314774`.
Test SHA-256:
`6796b740245da20f5e37c6bc5aedf83caa7bd028f5bab23afa57575ee9bb9001`.

## Acceptance limits and remaining decisions

Independent Gaussian seed effects in the **validation-eligibility conditional
population** yield the usual finite-sample t calibration. That assumption is
unverified here; nondegenerate bounded categorical effects cannot be exactly
Gaussian. Ten seeds do not validate a normal approximation. Consequently the
reducer claims neither exact coverage nor an unconditional familywise-error
guarantee. Holm cannot repair invalid local p-values.

The coordinator must still accept or revise primary mean inference, the
substantive SEI, seed set, hardware assignment, immutable held-out manifests,
artifact/eligibility verification and all other measurement choices before
confirmation. No changed seed set is silently accepted: the output records
whether the declared order equals the proposed IDs 3–12. Independence and
source/seed identities require checks by the ingestion layer; a boolean
eligibility assertion is insufficient scientific evidence.

The proposed joint seed-bootstrap and its simultaneous/ordinary reports are
**not implemented** in this bounded task. The original claim rule requiring
agreement of both reporting intervals is therefore still unsatisfied even
when a candidate t/Holm calculation rejects. No exact sign flip, bootstrap
coverage claim, equivalence test, covariance/tangent combination or causal
mechanism acceptance is performed. This infrastructure does not remove any
existing competence or confirmation gate.
