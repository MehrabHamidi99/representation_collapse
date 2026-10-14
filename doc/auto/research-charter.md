# Research charter: useful-history loss under controlled periodic forcing

Version 1, 2026-10-12. This is the canonical statement of the program's
question, hypotheses, estimands, claims and publication rule. Every other
document in `doc/auto/`, `paper/` and the skills defers to it; where an older
document disagrees, this charter governs and the older text is deferred
coverage, not a competing plan. It records no new empirical result. Evidence
status follows [the tracker](../TRACKER.md) and the dated notebook.

## Current manuscript scope — revision of 2026-10-12

At the owner's request, the paper is now a compact, venue-neutral methodological
study titled **When Does Repetition Erase Memory? Identifiability Limits for
Recurrent-State Collapse**. It presents established analytic nulls and explicit
counterexamples, a matched-order experiment, and the verified pilot audit of
failed native competence. It does not claim a learned erasure mechanism, an empirical null,
a new general Floquet theorem or cross-modal transfer. Diagnostic
non-identifiability means that individual geometric/stability measurements do
not determine retained task information; it is not a new formal statistical
identification theorem.

This presentation scope is distinct from the broader mechanism program below.
The scientific publication requirements in section 11 remain in force for
mechanism or empirical-null claims. A polished theory/methodology manuscript
also needs a compelling original contribution relative to the cited antecedents;
reorganizing known results and illustrating failed pilots does not itself
satisfy a top-conference novelty threshold. The manuscript's readiness review
records these limitations separately from editorial completeness.

## 1. Framing and abstract

**Working title.** When does repetition erase what a recurrent network needs?
Useful-history loss under controlled periodic forcing in fixed-weight
recurrent models.

"Dimension collapse across modalities" is the motivating umbrella, not the
claim. The [novelty audit](../literature/novelty-review.md) found that
periodic/Floquet memory, echo-state washout, recurrent and graph collapse,
noise-maintained geometry and contracting loops with input reinjection all have
close antecedents. The narrower program is: a competent recurrent network has
written task-required information into its complete state; a distractor body of
equal length then follows, either as exact repetitions of one motif or as a
matched non-periodic motif sequence; we ask whether the periodic body causes
excess loss of the written information, by which state-transition mechanism,
and when stable attractor memory prevents the loss. Three estimands
(covariance geometry, tangent propagation, behavior/information) are measured
separately; analytic nulls fix what ordinary stability can and cannot explain;
causal interventions with shams decide use. As of this version no trained-model
claim is accepted: four seed-0 pilots failed short-horizon competence, so the
harmful-collapse decision is inconclusive, and the checkpoint-only periodicity
diagnostics have now passed independent numerical reduction and review. Their
[descriptive measurements](periodicity-independent-results.md) retain the
competence failures and do not establish harmful collapse or certified Floquet
dynamics. Subsequent competence-recovery integration is prospective discovery.

## 2. Research question, sub-questions and gates

**RQ.** At fixed weights and equal retention horizon, does higher-order
repetitive (periodic) forcing of a distractor body cause excess loss of
previously written, task-required information in a recurrent model, relative to
a matched non-periodic body (same symbol counts, cyclic bigrams, run lengths,
start/end context, total updates, final motif, cue/query/nuisance)? Through
which state-transition mechanism (input/state-dependent gating; noncommuting
propagation; cue-dependent basin selection or merging; saturation; specified
noise/precision effects)? And when does stable attractor memory prevent that
loss?

The audit's five open questions map onto the staged gates. Gate order
supersedes the 351-checkpoint screen of
[periodicity-research-plan.md §3E](periodicity-research-plan.md), which is
deferred coverage.

| Sub-question | Gate | Decisive test | Falsifier |
| --- | --- | --- | --- |
| OQ1 Does higher-order repetitive ordering cause excess loss of required history? | B (after A) | Paired arms, both target types, `δR` at `R∈{2,16}`, `DID=δ16−δ2`, conditioned recoverability | Equal forgetting in both arms; CI excludes the preregistered smallest effect; covariance/predictability baselines explain the effect |
| OQ2 Which mechanism separates useful attractor compression from destructive synchronization? | B, C | Basin membership, finite perturbation radii, precision/noise/dwell/order sweeps; localize encoder, body, query/readout | Preserved categorical decoding with contracting local spectra refutes a local-spectrum explanation |
| OQ3 Can task-sensitive dynamics predict and causally explain failures beyond simpler diagnostics? | C | Held-out comparison of horizon, norm, PR, gate saturation, autocovariance, readout baselines against a declared cue-to-output sensitivity/noise model; direction-specific corruption and rescue with shams | No held-out improvement or no direction-specific effect: descriptive diagnostic only |
| OQ4 Does a verified mechanism transfer beyond the synthetic vector task? | D | Full-state GC-LSTM (state `2NH`) on a reproduced T-GRAB setting, then one native non-graph task | Effect confined to one renderer, a faulty adapter or unused prefix information |
| OQ5 Which bounded interventions preserve required history? | after D | Budget-matched intervention versus orthogonal/skip/reinjection/buffer controls, all persistent state counted | Improvement explained by more bytes, retained prompts or extra tuning |

**Gate A** — analytic and measurement falsification: affine ordering null,
commuting null, noncommuting positive construction, readout-offset control,
stable categorical basins, continuous neutral memory with contracting nuisance,
amplitude-only contraction, fresh noise without cue retention; complete-state
products, chronology and conditioning validated. **Gate B** — minimal trained
mechanism study: GRU and full-state LSTM, vector renderer, categorical and
continuous targets, `q=1`, within-support horizons, competence first.
**Gate C** — theory that adds a prediction beyond the affine null. **Gate D** —
graph (T-GRAB/GC-LSTM full-state adapter) and one native non-graph task.

## 3. Hypotheses, alternatives and falsifiers

| Hypothesis | Claim | Decided by | Falsified when |
| --- | --- | --- | --- |
| H1 harmful collapse | Fewer reliably measured task-relevant directions accompany excess loss of written information under periodic forcing, and intervening on those directions changes behavior | All three estimands plus causal interventions with shams | Behavior loss without geometric loss; geometric loss without behavior loss; intervention effect matched by equal-energy shams |
| H2 sufficient compression | Dimension falls while required information and performance remain intact | Behavior and probes at competence; absolute variance | Native error rises with dimension loss and probes lose the cue |
| H3 input synchronization | Forcing removes initial-state sensitivity while a persistent, contracting periodic response continues to carry current-input information | Closure residuals, tangent contraction, paired cue separation, native behavior | Cue separation persists; closure fails; or behavior fails despite synchronized state |
| H4 measurement artifact | Apparent collapse follows short windows, repeated identical inputs, pooling, sampling rank ceilings, float32 resolution or normalization | Rank ceilings `P−1` and `N_cue−1`, FP64 replay, subsampling curves, absolute variance | Effect survives matched sampling, declared precision and normalization controls |

Named alternatives that must be reported as legitimate outcomes: the
**readout-offset effect** (an arm-dependent additive output shift changes raw
error while the cue-conditioned channel is unchanged; Proposition P1 corollary)
and **stable categorical attractor memory**
([Pals, Macke and Barak 2024](../literature/bibliography.md#pals2024-phase-locked-memory)):
a cue selects a basin whose identity survives while perturbations inside the
basin contract. A theory that predicts only failures is insufficient; it must
predict preservation too.

## 4. Three estimands

1. **Covariance geometry** of cue-conditioned complete-state clouds: vary the
   required cues while holding the whole body history, nuisance values, phase
   and horizon fixed; report participation ratio `PR=(Σλ)²/Σλ²`, covariance
   entropy dimension `exp(−Σp_i log p_i)`, `D90`/`D95`, absolute variance
   `trace C`, spectra, sample counts and the centering/normalization used.
2. **Tangent propagation**: finite-time propagators `P_{t,L}=J_{t+L−1}⋯J_t`
   with singular values, finite-time rates `log σ_i / L` and eigenvalues; call
   eigenvalues Floquet multipliers only when input periodicity and complete-state
   orbit closure are verified at declared tolerances. Otherwise every product is
   a finite-time propagator.
3. **Behavior and information**: native task error, held-out train-fit probes
   with shuffled-label controls, counterfactual cue separation, and causal
   state interventions (complete-state swaps, direction-specific corruption and
   rescue) with equal-energy shams.

No single proxy decides. Isotropic attenuation changes amplitude and not PR;
fresh noise raises rank without preserving the cue; nuisance compression lowers
PR while improving the task; contracting local spectra coexist with categorical
memory; probe failure is not information absence and probe success is not
native use. For LSTM dynamics the state is `[h,c]`; for graph models the state
includes every node and its ordering.

## 5. Primary contrast and competence prerequisite

Primary preregistered behavioral contrast: horizon-specific arm differences
`δR = E_periodic(R) − E_control(R)` for `R∈{2,16}` and `DID = δ16 − δ2`, with
`R=1` the shared anchor (arms identical) and `R=32/64` extrapolation. A positive
`DID` with both `δ` negative indicates relative deterioration, not harm at
either horizon; excess harm is claimed only at a horizon whose own `δR`
supports it.

Competence gate, prerequisite to any retention interpretation: coordinate-wise
categorical accuracy at least 95%, or continuous normalized MSE
(`MSE/(1/3)`) at most 0.05, at `R0`, `R1` and `R2` in both arms. The v1
engineering gates (25% accuracy or raw MSE 0.25 at `R1/R2`) are weaker and do
not substitute. Replication unit is the independent training seed; report
paired seed-level effects with confidence intervals; Holm correction across
registered primary comparisons; declare the smallest effect of interest before
inspecting held-out results; episodes, cues, coordinates and time steps are not
training replicates. The concrete values (seed counts, interval method,
registered family, smallest effect of interest) are proposed in
[experimental-design-v2.md §5](experimental-design-v2.md) and must be frozen,
with a hashed freeze record, before any confirmation seed is inspected.

## 6. Claims

**Safe now.** We study whether matched higher-order repetitive forcing impairs
required-history retention in fixed-weight recurrent models, and test when
geometric and local-stability diagnostics do and do not predict that
impairment. We provide analytic nulls (P1–P11 in `paper/sections/04-theory.tex`)
and executable fixtures (10 measurement and 36 mechanism checks) that any
proposed mechanism must respect. Four seed-0 vector pilots failed short-horizon
competence, so the v1 protocol establishes an acquisition problem, not a
collapse result.

**Conditional, brackets filled only from accepted evidence.** "In [models] on
[tasks] at horizons [R], periodic forcing produced `δR=[…]` (CI `[…]`) and
`DID=[…]`, explained by [mechanism] under [conditions], and causal
[intervention] changed behavior by [effect] against [sham] while [alternatives]
did not account for it." "Stable categorical memory survived periodic forcing in
[conditions], bounding the mechanism."

**Non-claims, never written as claims.** "First Floquet analysis of RNN
memory"; "universal law of representation collapse"; "negative exponents imply
lost memory"; "periodic input implies periodic hidden orbit"; "PSD peak implies
neutral mode"; "AABB predicts fourth-root-of-unity multipliers";
"modality-general collapse" from isomorphic renderers; "the `F−s` Euler
surrogate is an exact continuous embedding".

## 7. Antecedent deltas

| Work | What it established | What we reuse | What remains open | Our distinguishing move |
| --- | --- | --- | --- | --- |
| [T-GRAB, Dizaji et al. 2025](../literature/bibliography.md#dizaji2025tgrab) | Periodic counting/memorizing, delayed cause–effect and long-range temporal-graph tasks; change-point F1 | Task definitions `Pdet(k,n)`, GC-LSTM baseline, public code/data leads | Whether next-graph success reflects retained prefix information | Independently cued, task-required memory with a full-state adapter |
| [Arroyo et al. 2025](../literature/bibliography.md#arroyo2025bridging) | GNN Jacobian/recurrence bridge; contraction, oversmoothing and oversquashing | Vectorized Jacobian viewpoint | Forced temporal dynamics and useful-history interventions | Equal-horizon ordering intervention, not depth |
| [Pals, Macke and Barak 2024](../literature/bibliography.md#pals2024-phase-locked-memory) | Cue-selected stable phase-locked cycles retain memory under periodic drive | Mandatory successful categorical control | Boundary between basin stabilization and merging | Predict both preservation and loss |
| [Ashwin and Ceni 2023](../literature/bibliography.md#ashwin2023-echo-repetitions) | Input repetitions change echo index and attractor multiplicity | Repetition-sensitive multistability framing | Whole-motif repetition with matched counts and run lengths | Matched motif bodies at fixed weights |
| [Manjunath and Jaeger 2013](../literature/bibliography.md#manjunath2013) | Input-specific echo state; periodic driver gives response whose period divides the input period | Washout versus history retention distinction; H3 | None claimed; foundation | Separate initial-state washout from written-cue loss |
| [Ganguli, Huh and Sompolinsky 2008](../literature/bibliography.md#ganguli2008-memory-traces) | Noise-resolved Fisher memory; normal/nonnormal distinction | Task-channel and declared-noise viewpoint (P9) | Nonlinear, periodically driven, gated models | Held-out predictive test against simpler diagnostics |
| [Sharma et al. ICLR 2026](../literature/bibliography.md#sharma2026-temporal-superposition) | Delayed-recall feature geometry, interference, useful forgetting | Geometry–recall framing as a baseline | Matched periodic versus varying driver contrast | Ordering intervention at frozen weights |
| [Fokam et al. 2026](../literature/bibliography.md#fokam2026-temporal-correlations) | Temporal correlations change learned memory tradeoffs in linear RNNs | Training-correlation effects as a named alternative | Higher-lag and inference-time effects | Frozen checkpoints and controlled training distributions |
| [Labovich 2026](../literature/bibliography.md#looped_stability_recall) | Autonomous versus recall sensitivity; `(I−J)⁻¹B` reinjection | P10 and the depth-loop separation | Designed neutral manifolds, categorical basins | Streaming once-written memory as primary setting |
| [MeSH, ICLR 2026](../literature/bibliography.md#mesh) | Recursive hidden-state spectral collapse and explicit-memory mitigation | Collapse diagnostic comparison | Full-state clouds across cue histories | Cue-conditioned complete-state geometry |
| [Viakhirev et al. 2026](../literature/bibliography.md#think_shallow_solve_deep) | Task outcomes with QR Lyapunov, dimension proxies, nonnormality in recurrent reasoners | Diagnostic battery as a bar to exceed | Once-written streaming memory | Separated estimands plus causal interventions |
| [Balasubramanian 2026](../literature/bibliography.md#all_routes_collapse) | Claims shared routing-collapse signature across architectures | Scope warning | Null validity (Appendix B.4 permutation null is spectrum-invariant as written) | Key-cloud rank is not task memory; measure required information |
| [Chehreghani 2026](../literature/bibliography.md#chehreghani2026noise) | Persistent Gaussian perturbations sustain graph Dirichlet energy | Fresh-noise fixture (geometry without memory) | Downstream utility | Preserve the cue, not just covariance |
| [Seung 1996](../literature/bibliography.md#seung1996) | Line attractors; continuous memory needs neutral directions | P4 antecedent | Robustness to noise and mistuning | Scope P4 to exact smooth memory only |
| [Vogt et al. 2020](../literature/bibliography.md#vogt2020) | Lyapunov analysis of input-driven RNNs | Finite-time and conditional exponent methodology | Exponent sign versus task memory | Report singular values and rates per window, never sign alone |
| [Hu and Sompolinsky 2022](../literature/bibliography.md#hu2022) | Covariance spectra of linear recurrent networks | Covariance versus recurrence spectrum distinction | Finite required-history clouds | Cue-conditioned covariance with absolute variance |
| [Roy and Vetterli 2007](../literature/bibliography.md#roy2007effective) | Entropy effective rank | Naming convention; P7 | Not an information theorem | Report PR, entropy dimension, `D90/D95` and variance together |
| Cvitanović et al., ChaosBook, cycle stability | Period-map Jacobians and Floquet multipliers | Standard construction | Not in the library yet | Verified closure before the label; freeze edition when cited |

## 8. Relation to T-GRAB

Mehrab Hamidi is a coauthor of T-GRAB (MLoG-GenAI Workshop at KDD 2025). Reused
from it: the periodic task family `Pdet(k,n)` (`k` patterns, block length `n`,
period `kn`; AABB is `Pdet(2,2)`, AB is `Pdet(2,1)`), the delayed cause–effect
and long-range families, the GC-LSTM baseline, the per-time exhaustive-pair F1
protocol and the public code/data leads. Reused inherited manuscripts are
preserved unchanged (`ACM___TGRAB/`, `ACM___TGRAB_KDD_2026/`; consulted
entrypoint `ACM___TGRAB_KDD_2026/kdd/main.tex:327–354`, active includes
`040method_ai`, `050periodic_ai`, `060cause_ai`, `070long_ai`,
`080discussion_ai`, `100appendix`). The May 2025 Floquet note
(`floquet_analysis/main.tex`) is an inherited draft whose invalid inferences are
replaced in `paper/sections/B-inherited-draft.tex`. New here: the equal-horizon
matched-body intervention, complete-state (`2NH`) measurement, separated
estimands, analytic nulls, causal interventions and the competence prerequisite.
October 14 follow-up recovered archived graph-analysis source and numerical
candidates on Mila. Their checkpoint/data/producer linkage and numerical values
remain unverified; the six inherited multipliers `|λ|≈0.41–0.44` remain
attributed observations. See the [recovery and debugging audit](tgrab-reproduction-audit-2026-10-14.md).

## 9. Evidence state, 2026-10-12

- Measurement calibration (10 checks) and mechanism calibration (36 checks)
  pass locally and on Narval CPU job 5234514: infrastructure evidence.
  [Initial-phase report](initial-phase-report.md).
- Four seed-0 vector pilots (training source `9004a0c114a8b7c5335dca3d5247449fd10b5ffa`,
  1,500 steps; Narval GRU 5234860/5235508 on A100; Rorqual LSTM
  22980985/22980998 on H100 MIG `3g.40gb`) are executed and verified from raw
  arrays. All four failed short-horizon competence: categorical accuracy
  5.47%/6.25%/5.08% at R1/R2/R16 against 6.25% chance; continuous MSE near the
  saved-target zero-predictor baseline (0.3533/0.3591/0.3341). Harmful collapse
  is **inconclusive**. R2 held-out probes beat shuffled labels; R16 probes do
  not, although the probe's `1e-8` feature-scale floor (`pilot.py:128`) is
  active at the observed per-feature amplitudes (~1e-9 std), so the R2-versus-R16
  probe comparison carries a horizon-dependent effective ridge strength and is
  confounded until re-reduced with the saved `probe_feature_scale` arrays
  ([implementation audit](implementation-audit.md) §4–5); the R2 advantage over
  shuffled labels stands. PR is not monotone across horizons while absolute
  variance falls to float32 scale.
  [Scientific review](initial-phase-scientific-review.md),
  [compact results](initial-phase-results.json).
- Checkpoint-only periodicity diagnostics (analysis source
  `458bc9cfce474b8e11e433e33bf132faf5032387`; Narval 5254858/5254871, Rorqual
  22984956/22984963) completed `0:0`; worker checks pass over 64 dense tangent
  windows and 1,536 step-Jacobians; 28 artifact hashes verified. Independent
  raw-array reduction and scientific review are **pending**; no number from
  them is reportable. [Protocol](periodicity-analysis-v1.md),
  [status](periodicity-analysis-status.json).
- Verified-infrastructure (2026-10-12, [implementation audit](implementation-audit.md)
  §1): the local test suite (34/34, 4.4 s; Python 3.11.13, NumPy 2.4.6,
  torch 2.6.0) and both calibration CLIs (10 and 36 checks, all manifest hashes
  recomputed and matching) were rerun locally at commit `8fce9b06`. `uv run` was
  broken on that machine by hidden (`UF_HIDDEN`) `.pth` files, so a
  `PYTHONPATH=src` fallback was used; the audit also records that `pilot.py`
  neither set nor recorded TF32 flags, that its `deterministic_algorithms: True`
  is a constant, that CPU repeat runs are bitwise identical, and that CUDA
  determinism is unverified. Infrastructure evidence only.
- Legacy graph assets not recovered; [reproduction protocol](graph-reproduction.md).
- The Mila Qwen worker is proposal-only; its proposals are not evidence.
- No accepted scientific claim exists.

## 10. Contribution language

**Safe:** "We investigate whether matched higher-order repetitive forcing
impairs required-history retention in fixed-weight recurrent models, and test
when geometric or local-stability diagnostics do and do not predict that
impairment." "We state and verify analytic nulls that a mechanism must respect."

**Conditional:** "We identify [mechanism] under [conditions], establish
[boundary/prediction], and confirm its causal effect on [tasks/models] against
delay, covariance, readout and attractor alternatives."

**Avoid:** the non-claims of §6; also "new theory that stable loops can recall
injected input" and any priority word ("novel", "first") not supported by the
audit. Call the treatment a specified higher-order ordering intervention, not
"periodicity" in isolation; report input- and encoder-level statistics that the
arms do not match.

## 11. Publication threshold and decision rule

A **mechanism paper** requires: a prespecified excess-retention effect at a
named horizon in competent models; a model of why it occurs and when it should
not; independent fresh-seed confirmation under a frozen effect-size and
uncertainty rule; causal evidence stronger than a geometry correlation; and a
consequential native transfer (Gate D) or a sufficiently sharp theoretical
result (Gate C).

A **null/counterexample paper** requires: competent models in both arms; a
confidence interval excluding the preregistered smallest effect at the
within-support horizons; systematic disproof of common collapse inferences with
the analytic nulls and fixtures; and predictive boundaries (for instance,
categorical attractor memory surviving forcing while continuous memory
degrades, or vice versa), each independently replicated.

Decision rule after Gate B confirmation: if the paired-seed `δ16` CI excludes
the smallest effect of interest in the harmful direction and `δ2` does not
contradict it, pursue the mechanism paper through Gate C; if the CI excludes the
smallest effect in both directions, pursue the null/counterexample paper; if the
CI is inconclusive, report the bounded result in the notebook and request a
power-justified confirmation envelope rather than widening the screen. Failed
competence in either arm returns to the competence ladder and publishes nothing
about collapse.

## 12. Reading map

| Topic | Canonical document |
| --- | --- |
| Question, hypotheses, claims, publication rule | this charter |
| Theory and proposition table | [theory-propositions.md](theory-propositions.md) (P1–P11 plus the open Gate C target P12; its §3 is the inherited-draft correction table); `paper/sections/04-theory.tex` restates P1–P11 |
| Experimental design v2 | [experimental-design-v2.md](experimental-design-v2.md) (competence ladder, battery, proposed statistics in its §5, compute estimate in its §8 with no authorization); predecessors [competence-recovery-plan.md](competence-recovery-plan.md) and [periodicity-analysis-v1.md](periodicity-analysis-v1.md) |
| Measurement contract | [research-program.md](research-program.md), [floquet-analysis skill](../../skills/floquet-analysis/SKILL.md) |
| Code status versus the contract | [implementation-audit.md](implementation-audit.md) (module map, Gate B coverage gaps, numerical risk register, local reruns) |
| Evidence | [TRACKER.md](../TRACKER.md), [initial-phase-report.md](initial-phase-report.md), [initial-phase-scientific-review.md](initial-phase-scientific-review.md), dated notebook under `doc/october/` |
| Literature and novelty | [novelty-review.md](../literature/novelty-review.md), [bibliography.md](../literature/bibliography.md) |
| Graph reproduction and provenance | [graph-reproduction.md](graph-reproduction.md), [source-inventory.md](source-inventory.md) |
| Operations and compute | [compute-plan.md](compute-plan.md), [cluster-access.md](cluster-access.md), [project-ops skill](../../skills/project-ops/SKILL.md) |
| Manuscript | `paper/` (focused diagnostic manuscript; plots trace to committed evidence or declared analytic constructions) |
| Manuscript build, layout and placeholder policy | [paper/README.md](../../paper/README.md) |
