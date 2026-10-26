# Prospective statistics specification — proposal v2

Prepared 2026-10-12 for coordinator review. **Proposal only: not frozen,
preregistered, accepted, or a compute approval.** The companion
[machine-readable proposal](../../../../../configs/experiments/statistics-freeze-proposal-v2.json)
records the unresolved decisions. It implements the intent of
[experimental design v2 §§2–5](experimental-design-v2.md),
[the research charter §§4–5](research-charter.md), and
[competence training v2](competence-training-v2.md); it does not amend those
documents or accept a scientific claim.

Existing seed-0 pilots and periodicity results have already been observed.
They remain discovery/descriptive evidence. Seeds 0–2 are reserved for
discovery and feasibility. The proposed confirmation set is exactly
`[3,4,5,6,7,8,9,10,11,12]`, with new untouched evaluation manifests. This set
requires coordinator acceptance, manifest construction and a separate compute
envelope. Preparation is not execution. No confirmation contrasts were read
to prepare this proposal.

## 1. Population, pairing and competence

The four cells are GRU/categorical, GRU/continuous, LSTM/categorical and
LSTM/continuous. The unit of replication is an independent training seed,
including its frozen validation/checkpoint-selection procedure. Conditions,
learning rate and schedule are fixed per cell after the complete discovery
grid and feasibility checks. Checkpoint selection within each new seed follows
the same validation-only rule. Architecture, GPU model, cluster, coordinate,
episode, driver pair, time step and horizon are not interchangeable seed
replicates. Assignments to hardware must be recorded; a hardware-confounded
comparison is not an architecture effect.

Within a seed and horizon, construct both forcing arms from the same cue,
query coordinate, nuisance values and matched driver-pair record. Save a
pair ID and both raw outcomes. There are 128 independent base episodes per
queried coordinate, eight coordinates, hence 1,024 episode pairs per horizon.
Episodes have distinct independent generator draws; the two arms are paired.
The proposed R2 and R16 manifests use distinct streams, so the proposal does
not claim episode-level pairing across horizons. Their seed-level effects are
always paired when computing DID, intervals and resamples. R0/R1 are shared
anchors with identical arms; their duplicated arm records are not independent
observations. Extrapolation at R32/R64 is reported separately.

A selected checkpoint is eligible only if its pinned native validation
manifest passes **every coordinate** at R0/R1/R2 in both arms:

- Categorical accuracy is at least 0.95. With 128 episodes this means at
  least 122 correct, not a rounded aggregate of 95%.
- Continuous `raw MSE / (1/3)` is at most 0.05, equivalently raw MSE at most
  `1/60`; the denominator is the fixed target population variance, never an
  arm-specific or sample-specific variance.

The four-example fixed-batch check is an optimization diagnostic; it requires
four correct categorical examples per coordinate and does not replace fresh
128-episode validation. Use uninterrupted native QUERY predictions, complete
LSTM hidden and cell state, and the exact saved checkpoint. Threshold passes
are empirical sample checks, not confidence bounds certifying true accuracy
or MSE. Report untouched short-horizon test scores for generalization, but
never select or replace a checkpoint using them.

The proposed retention population is **checkpoints produced by this fixed
algorithm conditional on validation eligibility**, in the stated synthetic
task/forcing distribution. It is not every initialization or every model
which can be made competent after arbitrary retries. Report acquisition
reliability and all seed failures alongside this conditional scope. Any
distributional assumption used for inference must apply to this conditional
seed-effect distribution; a normality/symmetry assertion about unselected
initializations is insufficient.

No confirmation seed is removed or replaced. If any declared seed in a cell
has no eligible checkpoint, a numerical failure, missing artifacts or an
unresolved deviation, that cell's confirmatory retention result is
**inconclusive**, not a smaller successful-seed analysis. Its two primary
family slots remain present with `decision_p=1` and an explicit unavailable
reason; there is no fabricated effect or interval. Available outcomes and
failures remain in the ledger. Other complete cells retain the original
eight-slot multiplicity correction. An exploratory successful-subset summary
must say it changes the sampled population and cannot rescue the primary
decision. Mechanical retries require reconciliation and authorization and
must never select the better scientific outcome.

## 2. Three separate estimands

### Native behavior and information

For seed `s`, coordinate `q`, arm `a`, horizon `R`, and episode `i`, use
`l_s,q,a,R,i = 1{predicted class != target}` for categorical behavior and
`l_s,q,a,R,i = (prediction-target)^2/(1/3)` for continuous behavior. Continuous
raw MSE is also saved, but all primary contrasts use normalized MSE. Define

```text
E_s,a(R) = (1/8) sum_q [(1/128) sum_i l_s,q,a,R,i]
delta_s(R) = (1/8) sum_q [(1/128) sum_i
                         (l_s,q,P,R,i - l_s,q,C,R,i)]
DID_s = delta_s(16) - delta_s(2)
theta_delta16 = expectation over eligible training seeds of delta_s(16)
theta_DID = expectation over eligible training seeds of DID_s
```

Positive values mean excess periodic-arm error. DID is an arm-by-horizon
interaction, **not** observational difference-in-differences identification
requiring a parallel-trends claim. Positive DID alone does not show harm at
R16; both arms could favor periodic forcing at both horizons. Compute DID per
seed first. Do not obtain its uncertainty by subtracting unrelated endpoint
intervals. Matched ordering identifies the total effect of the specified
ordering intervention at fixed weights; it does not isolate geometric
mediation, current readout offsets or a universal property of periodicity.

Held-out probe recovery, shuffled-label controls, counterfactual cue
separation and native intervention effects are distinct information/use
measurements within this estimand family. They do not replace native error.
The primary family below contains only native `delta16` and DID.

### Cue-conditioned covariance geometry

For each seed, horizon, arm and state phase, hold the complete body driver,
nuisance and phase fixed and vary cue histories. Center each cloud by its own
training-cloud mean and use sample covariance with denominator `N-1`. Primary
coordinate convention is raw complete state (`h` for GRU, `[h,c]` for LSTM);
report `h` and `c` separately as decompositions, not independent replicates.
No pooled time/driver cloud substitutes for the conditional cloud.

With covariance eigenvalues `lambda_j >= 0`, save the full spectrum,
`trace C`, `PR=(sum lambda)^2/sum lambda^2`, entropy dimension
`exp(-sum p_j log p_j)` with `p_j=lambda_j/trace C`, smallest D90/D95 reaching
the stated explained-variance fraction, numerical rank at
`1e-10*lambda_max`, and covariance stable rank `trace C/lambda_max`.
Here “covariance stable rank” is also the stable rank of the centered data
matrix; it is not `||C||_F^2/||C||_2^2`. If variance is zero, normalized
dimensions are undefined, with a reason flag and no invented zero dimension.
All ranks respect `min(N-1,d)`. The finite-time tangent dimension below is a
different quantity.

Use eight paired driver records. For scalar metric `m`, save each
`m_s,P,g(R)-m_s,C,g(R)` and its equal-driver average per seed. There are eight
driver observations nested in a seed, not eight additional training
replications. Subsampling at N=64/128/256/512 uses declared nested cue subsets;
it is a convergence assay, not four independent results. Three state phases
(post-WRITE/pre-QUERY/post-QUERY), driver components and precision labels stay
separate. Geometry is secondary descriptive evidence in this proposal.

### Tangent propagation

For the declared final 12-update window, use complete persistent state and
`P=J_11 ... J_0`. Save all singular values, eigenvalues, largest singular gain,
`log(sigma_j)/12` per update and tangent PR of squared singular values. Exact
zero singular values have rate `-infinity` represented by a zero/undefined
reason field compatible with finite JSON; they are not silently clipped into
finite contraction estimates. A partial spectrum is labelled partial.

Eight driver pairs, two arms, R2/R16 and FP32/FP64 give 64 windows per seed.
The representative cue history is the first history in each predeclared driver
manifest, never one chosen for its effect. Average paired scalar driver
differences within seed; retain raw spectra. Dense/reference agreement,
JVP/finite-difference checks, nonnormal transient gain and every required
closure check accompany interpretation. Periodic input alone cannot authorize
the Floquet label. Failure of complete-state/input/Jacobian periodicity leaves
finite-time propagators. Tangent quantities are secondary descriptive evidence.

No scalar score combines these three estimands. A behavior effect plus
covariance contraction is not proof that the contraction caused the effect.

## 3. Primary family, margins and uncertainty

The proposed harm family is the Cartesian product of the four cells and
`[delta16,DID]`: **eight** one-sided comparisons, including unavailable cells.
For task-specific margin `epsilon=0.02`, the null is `theta <= epsilon`
and the alternative is `theta > epsilon`. Categorical units are probability
of error; continuous units are normalized MSE (`0.02` equals raw MSE `1/150`).
`delta2`, individual coordinates, R4/R8, R32/R64, geometry, probes, tangent
spectra and interventions are secondary. They receive no confirmatory
significance claim under this eight-test family.

Apply Holm at family alpha 0.05 to **valid** raw p-values. Sort ascending,
breaking exact ties by the frozen family order; compare `p_(k)` to
`0.05/(8-k+1)` until the first failure. Adjusted values are
`min(1,max_{j<=k}[(8-j+1)*p_(j)])`. Cross-cell dependence is allowed;
invalid individual p-values are not repaired by multiplicity correction.
The sequential control argument depends on valid local tests.
[Holm (1979)](https://www.jstor.org/stable/4615733);
[Goeman and Solari (2010)](https://arxiv.org/html/1211.3313v1).

Preserve the v2 proposal's paired seed-bootstrap and t reports. Resample the
**joint vector** of each seed's R2/R16 arm outcomes and derived effects, using
the same seed index draw for all effects in that cell. For the proposed shared
seed blocks, reuse index draws across cells to preserve cross-cell pairing.
Use 10,000 resamples, NumPy PCG64 seed 2026101201, equal seed weights and
explicit linear-interpolation empirical quantiles. A two-sided per-effect
report is at `1-0.05/8 = 0.99375`, using endpoints `0.003125,0.996875`;
also display ordinary 95% intervals labelled nonsimultaneous.
At 10,000 draws each extreme endpoint uses about 31 tail draws: deterministic
RNG reproducibility does not eliminate Monte Carlo quantile error or the
small-n bootstrap coverage error. A degenerate bootstrap is flagged;
additional episode resampling cannot manufacture new training seeds.
[Efron (1979)](https://doi.org/10.1214/aos/1176344552).

For `n=10`, let `d_s` denote one primary seed effect, `bar d` its mean and
`s_d` its sample SD with denominator nine. The paired t reporting interval is

```text
bar d +/- t_(9, 1 - 0.05/(2*8)) * s_d / sqrt(10)
        = bar d +/- 1.1215044595 * s_d.
Ordinary 95%: bar d +/- 0.7153569060 * s_d.
```

The t formula has exact finite-n coverage **under independent Gaussian seed
effects in the specified population**; it is approximate otherwise.
Nondegenerate bounded categorical seed effects cannot be exactly Gaussian,
so their t report necessarily uses an approximation. Ten seeds and three
discovery seeds do not establish a good Gaussian approximation. Zero seed SD
is recorded explicitly; no divide-by-zero t p-value is silently accepted.
For a proposed t test, boundary statistic would be
`sqrt(10)*(bar d-epsilon)/s_d`, with upper-tail Student-t p-value; choosing it
as primary is a protocol revision requiring acceptance before confirmation.
[Student (1908)](https://doi.org/10.1093/biomet/6.1.1).

The existing design requires a valid primary decision **and** both reporting
intervals to exclude the SEI in the harmful direction; disagreement is
inconclusive. Extra interval requirements can reduce rejections but do not
make invalid primary p-values valid. If the inference model is not justified,
retain descriptive estimates and approximate intervals without a
confirmatory harm declaration.

The proposed margins remain substantive choices awaiting acceptance. The
existing assertion that 1,024 paired episodes generically resolve an error
difference near 0.01 at the 5% level is not a power guarantee: at coordinate
`q`, categorical pair differences have variance `v_q=p_discordant,q-delta_q^2`.
For independent stratified episodes the conditional episode-mean variance is
`sum_q v_q/(8^2*128)`, which depends on arm discordance; training-seed
variability is a separate source of uncertainty. The margin is two percentage
points for categorical error and 40% of the continuous competence margin;
accept or revise that practical justification before confirmation. Never
reduce the margin after seeing a held-out contrast. Discovery SD is a rough
precision diagnostic, not a precise power estimate from three seeds. Assess
**both** primary effects in all four cells; `1.1215*s_d > epsilon` warns that
the proposed reporting interval is wider than the SEI, not a formal power
calculation. Any larger seed set needs an accepted plan and compute envelope.

## 4. Exact sign flipping: a conditional option, not an established guarantee

For a complete ten-seed vector, shift by the tested boundary,
`z_s=d_s-epsilon`. Enumerate all 1,024 signs, including the observed identity:

```text
T(z) = mean(z)
p_plus = #{b in {-1,+1}^10 : mean(b*z) >= mean(z)} / 1024.
```

This is the proposed one-sided harm statistic; no studentization and no
Monte Carlo approximation. Equality is included. A numerical implementation
must record its summation and conservative tie tolerance; the proposed
absolute statistic tolerance is `1e-12*max(1,max(abs(z)))` in FP64. Counting
values down to `T_observed-tolerance` can enlarge the p-value. Do not use
mid-p, omit the identity or count only strictly larger transformations.
The minimum one-sided p-value is 1/1,024, so the eight-family first Holm
threshold 0.00625 is attainable with n=10. With n=7 even the minimum
1/128 exceeds it; small-n enumeration cannot justify reducing the family.

At the boundary, this test is valid if the joint shifted-effect distribution
is invariant to the sign group. Independent, symmetric centered residuals
in a common location model provide one sufficient assumption. For the
composite null `theta<=epsilon`, the location model and monotone upper-tail
test make the boundary least favorable; symmetry merely at an unrelated
zero mean does not establish this. At most `alpha` rejection probability
means an exact/conservative level, not guaranteed equality to nominal alpha
with discrete ties. Group invariance is the relevant requirement.
[Hemerik and Goeman (2018)](https://link.springer.com/article/10.1007/s11749-017-0571-1).

Neither matched episodes, a random nonperiodic input generator, nor arbitrary
renaming of P/C supplies randomized treatment labels at the training-seed
unit. The two forcing conditions differ by construction and were both
evaluated deterministically at the same checkpoint. A zero-mean null with
skewed seed effects does not give sign exchangeability. Nor does a symmetry
test on three discovery seeds prove it. A binomial sign test instead targets
a sign probability/median and cannot silently replace the mean estimand.

A bounded analytic counterexample makes the limitation concrete: let each
independent categorical seed effect be `1/32` with probability 0.91 and
`-3/32` with probability 0.09. Its mean is exactly the proposed boundary
0.02. All ten effects equal `1/32` with probability `0.91^10`, about 0.3894;
on that event the shifted one-sided sign-flip p-value is 1/1,024, below even
the first eight-family Holm threshold. Thus a mean-null assertion alone can
have very large rejection probability. This is a constructed mathematical
example, not a distribution estimated from this project's models.

Under the common symmetric location model, inversion of a two-sided
enumerated test at level 0.00625 gives a valid per-effect 99.375% confidence
**set** for the location parameter; simultaneous coverage then follows by
Bonferroni. Save the accepted set or a conservative enclosing interval, not
an assumed smooth t-shaped interval. This option is distinct from the
approximate percentile bootstrap and the Gaussian-exact t interval.
Its assumptions are **not currently justified** for trained-model errors.

As a limitation check, categorical `delta16` is bounded in `[-1,1]` and DID
in `[-2,2]`. For independent seed effects, Hoeffding gives a conservative
simultaneous mean interval radius
`(b-a)*sqrt(log(2*8/0.05)/(2*10))`: about 1.07409 for `delta16` and 2.14818
for DID, intersected with their support. These wide bounds illustrate the
cost of a distribution-free ten-seed mean guarantee; they are not a useful
replacement claimed by this proposal. Native continuous predictions are
unbounded, so bounded targets alone do not justify the same bound for MSE.
Post hoc clipping would change the estimand.
[Hoeffding (1963)](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).

**Primary-test decision is a freeze blocker.** The coordinator must either
justify the sign-invariant location model, accept an explicitly approximate
t-based primary revision with its limitations, or adopt a different validated
design/test before confirmation. This proposal cannot retrospectively supply
exactness. Until then the primary method field stays unresolved and no
confirmatory decision is authorized.

## 5. Equivalence, probes, precision and causal follow-up

An interval containing zero is inconclusive, never equivalence. If equivalence
is later authorized, use null `theta<=-epsilon OR theta>=epsilon` and
alternative `-epsilon<theta<epsilon`. With a valid location-model test,
`p_TOST=max(p_lower,p_upper)`; both one-sided components must reject. A 90%
interval wholly inside the margins is the unadjusted alpha-0.05 rule only.
The harm and equivalence families cannot each claim an unstated combined
0.05 guarantee. This proposal leaves equivalence **disabled**. To enable it,
freeze a joint 16-comparison family or a justified gatekeeping/alpha-split
scheme before inspecting confirmation outcomes. Its test assumptions remain
the same substantive issue as harm inference.
[Schuirmann (1987)](https://pubmed.ncbi.nlm.nih.gov/3450848/).

Probe clouds use the design's 512 train cues and 256 untouched test cues per
driver/arm/horizon/phase. Fit preprocessing, task directions, regularization
and any probe selection using training/validation cue histories only; keep
all coordinates of the same cue history in one split. Save raw features,
labels, predictions, feature SD, floor, gains and floor-hit counts. The
proposed scale-relative floor is `1e-6*median(positive train feature SD)`;
exactly constant features map to zero after centering, and an all-constant
cloud has undefined normalized geometry. This floor coefficient and probe
ridge/selection policy remain acceptance blockers. Record per-component h/c
scales. An unstandardized sensitivity analysis and a scale-equivariance
fixture at amplitudes `1e-10` through `1e2` are required. A shuffled baseline
uses its frozen permutation and identical fit pipeline, never the best
shuffle. Probe failure is not information absence, and probe success is not
native task use.

FP32 closure tolerances must be fixed with ULP calibration before confirmation:
save coordinate spacings, absolute/relative residuals, changed-coordinate
counts and bitwise equality. A passed tolerance is not equality; an aggregate
half-ULP norm is not a universal minimum separation. FP64 fixed-weight replay
is a different numerical assay, not exact arithmetic or new training. Never
floor small cue distances/singular values to zero to obtain the desired
conclusion. Geometry below calibrated resolution is labelled
`resolution_limited`. Exact closure/noise tolerances and intervention/noise
scales remain blockers, to be chosen using Gate A/discovery only. Neither
current FP32 tolerances nor the historical probe floor transfer automatically.

For an intervention `I` and matched sham `S`, the per-seed causal contrast is
the equal-coordinate mean of `loss(I)-loss(S)` on the same 256 base episodes
(32 per coordinate), with equal-size random-subspace controls. Complete-state
swap, post-WRITE transplant, rank/norm-matched task versus random corruption,
and rescue are distinct manipulations. All shams rerun the same chronology;
intervention rank, norm, phase, direction fitting and targets must be frozen.
Noise assays use a discovery-calibrated scale, fixed before confirmation,
not a scale selected by the held-out effect. Report seed estimates and
explicitly descriptive intervals. A confirmatory mechanistic claim needs a
separate preregistered causal family and acceptance checks; this proposal's
behavior family does not confer it.

## 6. Freeze ledger and remaining decisions

The proposed behavior manifest seeds are
`1_000_000 + 10_000*cell_index + 100*training_seed + horizon_index`, with cell
order as listed in §1 and horizon order `[0,1,2,4,8,16,32,64]`. Manifest
identity includes task, generator revision, stratum, RNG algorithm, seed and
content SHA-256. Disjoint named domains are required for training, validation,
probe training/testing, driver, intervention, shuffle and noise streams;
their offsets/hashes are still to be specified and collision-tested. The
formula alone is not a saved immutable manifest. No held-out loss, arm
contrast or geometry is inspected before an accepted freeze receipt exists.

The final accepted receipt must hash source commit, uv.lock/environment,
training/task/measurement protocols, dimensions/chronology, all validation
and test manifests, exact confirmation set, selection rule, estimand units,
family order, margins, inference assumptions and implementation, bootstrap
seed/quantiles, missing-seed policy, probe/tolerance/noise/intervention settings,
acceptance reviewer and timestamp. A changed scientific setting creates a
new version and states whether any earlier outcomes were seen. Execution
approval is separate. Missing hashes remain null, never a fake frozen hash.

Unresolved before acceptance: primary mean-test justification/revision;
substantive SEI justification; exact manifest domains and generator hashes;
probe ridge/floor policy; ULP/finite-difference resolution decisions; causal
intervention/sham/noise contracts; sufficient seed precision; and the tested
statistics implementation. The canary is discovery only and cannot unblock
these by retrospective relabelling.

Acceptance checks for a future reducer: planted effects; explicit skewed
mean-zero counterexample to unjustified sign flipping; full 1,024 enumeration
with ties/zeros/SEI shifting; Holm monotonicity and unavailable slots; DID
covariance preservation; coordinate weighting and continuous normalization;
failed-seed retention; bootstrap deterministic indices and quantile convention;
zero-variance/undefined cases; conditional coverage simulations labelled
simulation rather than proof; and exclusion of held-out results from selection.
This task wrote a proposal only, with no test implementation, cluster access,
provider calls, new data or model execution.

Primary statistical sources consulted: Holm (1979), *Scandinavian Journal of
Statistics* 6(2), 65–70; Goeman and Solari (2010), *Annals of Statistics* 38(6),
3782–3810, DOI `10.1214/10-AOS829`; Efron (1979), *Annals of Statistics* 7(1),
1–26, DOI `10.1214/aos/1176344552`; Student (1908), *Biometrika* 6(1), 1–25,
DOI `10.1093/biomet/6.1.1`; Hemerik and Goeman (2018), *TEST* 27, 811–825,
DOI `10.1007/s11749-017-0571-1`; Schuirmann (1987), *Journal of
Pharmacokinetics and Biopharmaceutics* 15(6), 657–680, DOI
`10.1007/BF01068419`; Hoeffding (1963), *JASA* 58(301), 13–30, DOI
`10.1080/01621459.1963.10500830`. Publisher metadata/primary papers are linked
near the corresponding method; the t/Hoeffding numeric checks here are local
mathematical calculations, not model evidence.
