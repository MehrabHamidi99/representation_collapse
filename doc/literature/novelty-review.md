# Novelty audit: periodic forcing and useful-memory collapse

**Audit date: 2026-10-11. Verdict: reject the broad novelty claim; retain a narrower, falsifiable research question.** This is a critical assessment of the [research plan](../auto/periodicity-research-plan.md), not a review of completed new experiments. The repository has measurement calibration and inherited graph manuscripts, but no recovered legacy checkpoint/raw-evidence chain and no new trained-model results establishing the proposed mechanism.

A defensible question is: **At fixed weights and equal retention horizon, when does higher-order repetitive forcing reduce recoverability of specified, previously written information, through which state-transition mechanism, and when does stable attractor memory prevent that loss?** The proposed matched-motif experiment is a candidate way to answer this. We found no exact counterpart combining all its controls in the inspected literature; that is a bounded search observation, not a priority certificate.

The broad statements “recurrent representations become low-dimensional,” “periodic inputs create attracting responses,” “contracting dynamics forget initial conditions,” and “looped transformers can collapse” already have close antecedents. A Floquet calculation, an effective-rank curve, and more architectures do not jointly become a novel explanation merely by appearing in the same paper. Strong categorical-memory counterexamples also make a universal repetition-to-forgetting claim false.

## 1. What was reviewed, and what the labels mean

The [annotated bibliography](bibliography.md), [BibTeX](references.bib), [machine-readable records](references.json), and [search log](search-log.md) are the source ledger. Three independent bounded reviews cover [theory](evidence/theory-review.md), [memory and experiments](evidence/memory-review.md), and [architectures](evidence/architectures-review.md); the coordinator audited graph antecedents and integrated their objections. The [download manifest](download-manifest.json) distinguishes acquired files from unavailable copies.

`full_text` means the relevant substantive sections identified in the record were read. It does **not** mean every appendix, proof, dataset or implementation was independently verified. `abstract` and `metadata` records supply context or follow-up leads, not theorem evidence. Downloading a PDF never changes that evidence level. Recent preprints receive novelty attention without being treated as settled empirical findings. No cited experiment was reproduced during this audit.

“Representation collapse” needs an operational definition. We distinguish: reduced covariance effective dimension; reduced absolute signal; local tangent contraction; synchronization across initial histories; reduced recoverability of required targets under specified precision/noise; and impaired native task performance. These can disagree. Neural collapse during supervised training, graph node homogenization, token homogenization with depth, and loss of a streaming cue are related literatures with different measured ensembles and causal questions.

## 2. Closest prior work and the actual overlap

### The project already has a public graph predecessor

[T-GRAB, Dizaji et al. (2025)](https://arxiv.org/abs/2507.10183) includes Mehrab Hamidi as coauthor and is public workshop work. Sections 3.1–3.3 cover periodic counting/memorizing, delayed cause–effect, and long-range tasks; changepoint F1 is already part of the evaluation. The new paper must clearly identify reused tasks, assets and findings. A new stability explanation or independent replication can extend it; periodic graph diagnostics alone cannot be presented as new. Good next-graph prediction also does not establish retention of arbitrary prefix information irrelevant to the next graph.

[Arroyo et al. (2025), *Bridging Recurrent and Graph Learning*](https://arxiv.org/abs/2502.10818), Sections 3–4, explicitly connect vectorized GNN Jacobians, recurrent/state-space perspectives, gradients, oversmoothing and oversquashing. Propositions 4.3–4.4 use restrictive contraction/common-zero assumptions. This is a major obstacle to claiming a new generic Jacobian explanation for graph collapse. It leaves room for forced temporal dynamics and useful-history interventions, but the difference must be demonstrated.

[Chehreghani (2026)](https://arxiv.org/abs/2607.28185), Sections 3–4, studies recurrent graph updates with fresh isotropic Gaussian perturbations and a geometric anti-oversmoothing result. Its definition explicitly does not guarantee downstream utility. Adding noise to maintain covariance or graph energy is therefore prior art; preserving a cue despite noise is a distinct question. Treat its results as a preprint's conditional claims, not a reproduced remedy.

### Floquet memory and repetition-sensitive attractors are direct antecedents

[Pals, Macke and Barak (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/), Figure 3 and the Poincaré/Floquet methods, show memory-selective stable phase-locked cycles. This is both a direct periodic recurrent-memory antecedent and a counterexample to “all local multipliers below one implies forgetting.” A cue can select a basin whose identity survives even while perturbations within that basin decay.

[Ashwin and Ceni (2023)](https://arxiv.org/abs/2309.04728), Assumption 2.1, Lemma 2.2 and Theorems 2.3–2.4, already relate input repetitions to echo-index/attractor multiplicity. Their symbol-block dwell conditions differ from repeating whole motifs with matched run lengths. That difference could matter experimentally; it does not license a generic first claim about repetition changing memory.

[Manjunath and Jaeger (2013)](https://www.ai.rug.nl/minds/uploads/ESP_NC.pdf) provide input-specific echo-state theory. Uniqueness of a driven response and forgetting initial conditions have established formal treatments. A periodic affine orbit under a stable period map is a useful calibration case, not a new theory of collapse.

### Geometry and memory already have task-sensitive theories

[Ganguli, Huh and Sompolinsky (2008)](https://ganguli-gang.stanford.edu/pdf/08.Memory.pdf), Equations 1–5, connect recurrent sensitivity and noise-whitened Fisher memory. [Sharma et al. (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html), Sections 3–4, analyze temporal feature geometry, delayed recall, interference and useful forgetting. A task-channel spectrum or a geometry–recall correlation is consequently insufficient novelty on its own.

[Fokam et al. (2026)](https://arxiv.org/html/2609.00420v1) are especially close: temporal correlations change memory tradeoffs in aligned linear learning. Lemma 2 concerns a **local** release boundary; Appendix E explicitly qualifies global transitions under strong periodicity. Matching lag-one statistics does not exclude their entire mechanism. Our primary intervention instead freezes weights and changes the body ordering; training-correlation effects remain an alternative explanation unless checkpoints and training distributions are controlled.

[Guan et al.](https://arxiv.org/abs/2409.03187) further make noise structure a required part of a memory claim. Biological dimensionality theory also predates the project: [Gao et al. (2017)](https://doi.org/10.1101/214262) address task/sampling constraints, [Recanatesi et al. (2019)](https://doi.org/10.1371/journal.pcbi.1006446) relate covariance dimension to recurrent connectivity, and [Dahmen et al. (2026)](https://doi.org/10.1038/s41593-026-02395-w) extend recurrence–dimension evidence across brain areas. These are antecedents, not direct demonstrations of the proposed AI memory effect.

### Recurrent depth and broad architectural collapse are crowded territory

[MeSH (ICLR 2026)](https://arxiv.org/abs/2510.07739), Sections 2, 3.2 and 4.1, diagnoses recursive hidden-state spectral collapse and introduces explicit memory. Its token-by-feature spectrum differs from a full-state cloud across cue histories; iteration-specific routers and a buffer also change resources and dynamics. Those distinctions motivate fair comparisons, not a claim that recursive collapse or explicit-memory mitigation is new.

[Labovich (2026)](https://arxiv.org/html/2604.15259v1), Sections 4–5 and Appendix A.6, already separates autonomous iteration from input recall and derives equilibrium sensitivity `(I−J)⁻¹B`. This directly overlaps a theoretical component of our plan. Its smooth transversality assumptions do not rule out designed neutral manifolds or categorical basin memory.

[Viakhirev et al. (2026), *Think Shallow, Solve Deep*](https://arxiv.org/html/2608.18222v1), Sections 3.4–3.5 and 6, already compare recurrent reasoners using task outcomes, QR Lyapunov estimates, dimension proxies, nonnormality and objective ablations. A broad spectral survey must advance beyond these diagnostics. Their input-conditioned depth loops differ from once-written streaming memory, which is the more promising primary setting here.

[Balasubramanian (2026), *All Routes Lead to Collapse*](https://arxiv.org/html/2606.22325v1) claims shared routing-collapse behavior across attention, graphs and recurrent/state-space architectures. It is a serious scope overlap even if its mechanism remains disputable. Our mathematical concern with Appendix B.4 is conditional: **if** its token-shuffle null is a common row permutation of a centered Gram/distance matrix, then `G′=PGPᵀ` has the same singular spectrum, so spectral gaps cannot arise from that null alone. Code and the precise shuffle need checking; this is not a verified implementation defect or grounds to ignore the paper.

[Huginn](https://arxiv.org/abs/2502.05171), [selective-SSM token dynamics](https://arxiv.org/abs/2410.03292), [LoopUS](https://arxiv.org/abs/2605.11011), and [Fixed-Point Reasoners](https://arxiv.org/abs/2606.18206) supply additional competing mechanisms and stabilization comparisons. Pure-attention rank-loss results and initialization analyses have narrower assumptions than a full modern transformer. The bibliography records that boundary rather than merging all “collapse” titles into one theorem.

## 3. Claims to reject, claims to reuse, and claims still open

**Already established foundations:** periodic linear stability, input-specific echo states, fading memory under suitable global contraction, continuous-attractor neutrality, noise-resolved linear memory, and input reinjection into stable equilibria. Cite and reuse them. A clear synthesis can be valuable, but label it as synthesis.

**False without extra assumptions:** local contraction always destroys categorical memory; lower PR implies lower task information; a stable spectral radius excludes finite-time amplification; periodic input guarantees a periodic hidden orbit; a top-k tangent spectrum identifies full-state dimension; and all bounded-state architectures must fail the fixed-size tag task as horizon grows. Finite categorical memories can persist in robust attractors without storing an ever-growing history.

**Conditional statements that need precise scope:** if a smooth encoder `E` and decoder `D` satisfy `D(E(u))=u` on an open subset of `R^d`, differentiating gives `rank DE≥d`. If one common period map `P` fixes that smooth memory family, `DP(E(u))DE(u)=DE(u)`, so it has at least `d` independent +1 eigenvectors. These are elementary continuous-memory consequences, not new universal finite-precision bounds. They exclude categorical coding, approximate finite-horizon memory, and retained external input. [Seung (1996)](https://neurophysics.ucsd.edu/courses/physics_171/seung_pnas_line.pdf) is a foundational antecedent.

**Not established in this repository:** a positive periodic-versus-control retention effect; a causal link between that effect and loss of task-relevant state directions; transfer to native graph/audio/vision/text tasks; and a mitigation outperforming matched state/input-access controls. These are hypotheses, even if inherited plots are suggestive.

### An exact null the planned mechanism must respect

Let the entire body be the fixed affine recurrence

\[
s_{t+1}=As_t+Bx_t+b,\qquad s_0=E(u).
\]

For any fixed body history `h` of length `H`,

\[
s_H(u,h)=A^H E(u)+k(h),\qquad
\Delta s_H=A^H\big(E(u)-E(u')\big).
\]

Thus the cue-conditioned centered covariance and cue derivative are independent of body ordering at equal `H`. This holds for a nonlinear initial encoder too. Additive process noise whose entire sequence has the same joint conditional law, independent of cue and body ordering, gives the same propagated noise law; common independent identically distributed noise is sufficient. Matching only per-step marginals is insufficient if temporal correlations differ. With a common linear readout, the **cue-dependent component** is unchanged; an arm-dependent additive output bias can still affect raw error. A nonlinear query/readout can additionally respond differently to the shifted operating point. A decoder informed of the body and allowed to compensate its deterministic offset has identical recoverability across arms under these assumptions.

Therefore ordinary stability of `A` cannot explain periodicity-specific erasure of the conditioned cue channel. This is an elementary derivation and a mandatory analytic control, not a claimed new theorem. A full S4-style network can contain nonlinear blocks; only the genuinely affine adapter is covered by this null.

For switched affine updates with matrices `A_x` depending only on the current body symbol, pairwise commutativity plus matched symbol counts likewise fixes the product acting on cue differences. Noncommutativity removes this obstruction but does **not** guarantee an effect. This extension concerns the noiseless cue channel (or matched terminal noise); intermediate process-noise covariance can depend on ordering even when the final cue product agrees.

Possible mechanisms left for the main study include input/state-dependent gating, noncommuting propagation, cue-dependent basin selection or merging, saturation, and specified noise/precision effects. A readout-offset effect is a scientifically legitimate alternative outcome and must be named as such.

## 4. Reviewer 2: major reasons to reject the present broad pitch

1. **The proposed contribution is an assembly of existing ingredients.** Much of the original theory list is classical or directly anticipated by recent work. A new benchmark spelling, architecture registry, or modal renderer is not a mechanism. The paper needs one result that changes a prediction beyond these antecedents.

2. **“Periodicity” is underidentified.** Counts, cyclic bigrams, endpoints and elapsed updates control useful confounds, but not every higher-order subsequence, autocovariance, spectrum or predictability statistic. The arms intentionally differ on some of these. Call the treatment a specified higher-order ordering intervention; report input- and encoder-level statistics and competing covariance predictors. No control can both destroy exact periodicity and preserve every statistic that identifies it.

3. **The proposed period sweep confounds dwell and period.** Doubling each symbol changes `q`, period and exposure structure together. It is a dwell/period condition, not an isolated frequency test. Separate motif length, symbol dwell, repeated-cycle count, total updates and physical sampling interval before making a general period law.

4. **The measured cloud may answer the wrong question.** Pooling time points from one settled period creates a centered sample-rank ceiling of at most `P−1`; pooling driver histories measures driver diversity. The primary cloud must vary required cue values while holding the *whole body history*, nuisance settings, phase and horizon fixed. Retain separate within-driver and between-driver components. A cloud of `Ncue` independently varied cues also has centered sample rank at most `Ncue−1` (511 for 512 cues), regardless of state size; check sample-size convergence and estimator bias before extrapolating a dimension claim.

5. **Dimension is neither necessary nor sufficient for harmful forgetting.** Isotropic contraction preserves normalized PR while reducing signal-to-noise. Fresh noise can increase rank without preserving the old cue. Nuisance compression can lower PR while improving the task. PR also changes under invertible anisotropic coordinate rescaling: fix the metric/units, give absolute variance, and avoid direct numerical comparisons across incompatible state encodings.

6. **Local stability is being asked a global question.** Stable cue-selected cycles can preserve labels. Measure finite inter-cue differences, basin identities, separations and perturbation radii as well as local Jacobians. Distinguish a transition between basins from contraction inside one basin. Algebraic rank can remain full while all usable singular directions fall below the declared noise floor.

7. **The Floquet label may be unjustified.** Frozen periodic input is insufficient. Verify complete-state closure and phase consistency; test whether the response has the input period, a multiple, or no detected cycle. A numerically small residual is a tolerance-qualified observation, not a proof of an exact orbit. Use finite-time tangent/conditional Lyapunov terminology otherwise. A discrete forced orbit does not automatically have the neutral phase multiplier of an autonomous continuous-time limit cycle.

8. **Behavioral failure can be access failure.** Use native error, controlled held-out probes, counterfactual cue separation, and causal state interventions jointly. Probe failure alone is weak evidence; probe success alone does not show native use. Full-state transplantation establishes causal relevance of state, not mediation by a PR change. Target-direction ablations/rescues need equal-energy random/nuisance shams and tests for off-manifold disruption.

9. **The state budget is incomplete unless every memory path is counted.** Include LSTM hidden/cell state, graph-node ordering, matrix memories, convolution buffers, recurrent blocks, adaptive controllers, explicit buffers and accessible caches. Persistent prompts in a depth loop are external memory resources. Parameter count, persistent bytes, FLOPs and input access cannot be replaced by one “same size” number.

10. **Training failure is a competing explanation.** Require baseline competence and analyze the prespecified within-support `R=2` versus `R=16` contrast first. Mixed-arm training, frozen checkpoints, matched tuning budgets and identical queried information matter. Treat `R=32/64` separately as extrapolation. If initial acquisition differs, locate that before attributing later errors to retention.

11. **Five renderings may be one task repeated five times.** Matched synthetic modalities are an excellent control for the recurrence mechanism, but not five independent demonstrations of real-world importance. Encoders may erase the difference or introduce it. Transfer a frozen mechanistic prediction to at least one native task where retaining earlier information is actually necessary, with encoder and external-memory audits.

12. **The current breadth risks weak evidence.** The 351-checkpoint screen precedes tuning and confirmation, while the key interaction is unverified. Three seeds are diagnostic pilots, not strong population inference. Histories within one checkpoint are not independent training replicates. Use paired seed-level uncertainty, independent confirmation seeds, prespecified smallest effects of interest and correction/qualification for architecture selection. Do not choose the best rank–error correlation after seeing the test set.

## 5. Open questions worth pursuing

### OQ1 — Does higher-order repetitive ordering cause excess loss of required history?

**Status:** specific empirical question unresolved by this audit, not a demonstrated effect. Closest antecedents are Ashwin–Ceni, Fokam, T-GRAB and delayed-recall work. The distinguishing setting freezes a competent network and compares equal-horizon bodies with declared matching constraints.

**Decisive test:** paired cue/query/final-motif histories, periodic and varying-motif bodies, categorical and continuous targets; primary error difference-in-differences between `R=16` and `R=2`. Also report each horizon-specific arm difference `δR = error_periodic(R) − error_control(R)` and conditioned cue recoverability. A positive difference-in-differences `δ16−δ2` can occur while both `δ` values are negative; it indicates relative deterioration, not necessarily harm at either horizon. Claim excess harm at a named horizon only when that arm contrast supports it. Compare nonlinear/gated models against affine and commuting controls.

**Falsifier / decision:** if both arms forget equally, conclude ordinary horizon-limited memory. If a confidence interval excludes the preregistered meaningful excess, do not claim that excess in the tested models, tasks, horizons and dwell condition; this does not establish general harmlessness. If covariance/predictability baselines explain the effect and interventions add no explanatory power, favor that simpler predictive account, while leaving its causal mechanism unproven without interventions.

### OQ2 — Which mechanism separates useful attractor compression from destructive synchronization?

**Status:** the distinction is established; a predictive boundary for these trained tasks could be new. Pals supplies the indispensable successful categorical control. The open target is when a driver merges cue distinctions or erodes noise-resolved separation rather than stabilizing them.

**Decisive test:** estimate basin membership and finite perturbation robustness; sweep label precision, noise, dwell and motif order. Localize the cue encoder, body propagation and query/readout contributions. Test whether a controlled gate/state perturbation changes the predicted boundary without simply changing state capacity.

**Falsifier / decision:** preserved categorical decoding with contracting local spectra refutes a universal local-spectrum explanation and becomes a boundary result. A useful theory must predict both preservation and failure, not fit only failures retrospectively.

### OQ3 — Can task-sensitive dynamics predict and causally explain failures beyond simpler diagnostics?

**Status:** generic noise-whitened channels are prior art. The open contribution would be an independently validated predictive and intervention result in nonlinear, periodically driven models.

**Decisive test:** compare horizon, norm, PR, gate saturation, autocovariance and simple readout baselines against a prespecified cue-to-output sensitivity/noise model on held-out histories and models. Declare the noise location. For `y=Hu+ε` with `u~N(0,I)`, independent `ε~N(0,Σ)` and `Σ` positive definite, posterior covariance is `(I+HᵀΣ⁻¹H)⁻¹`. A positive-definite Gaussian prior covariance `Λ` changes this to `(Λ⁻¹+HᵀΣ⁻¹H)⁻¹`. The proposed uniform/categorical cues do not have this exact Bayesian posterior: use the formula as a declared Gaussian calibration or justified surrogate, and measure actual recovery separately. Nonlinear Jacobians supply local approximations. For categorical basins, include finite differences rather than replacing them with derivatives. Use direction-specific corruption and rescue with shams.

**Falsifier / decision:** no held-out improvement or no direction-specific intervention effect means a descriptive diagnostic, not a new causal theory. A recoverable cue with failing native readout supports an access mechanism instead of erased storage.

### OQ4 — Does a verified mechanism transfer beyond a synthetic vector task?

**Status:** open external-validity question. Recurrent architecture breadth and prior graph benchmark performance do not settle it.

**Decisive test:** transfer a frozen prediction first to a full-state GC-LSTM or another native temporal graph model, then a native sequence task in a different modality. Match or measure encoder signal-to-noise, query requirements, state budgets and accessible inputs. Reproduce the relevant predecessor task before claiming improvement over it.

**Falsifier / decision:** effects confined to one renderer, faulty adapter, or unused prefix information fail the modality-general claim. A negative transfer should identify the protective mechanism or scope limit; adding more similar renderers is not a substitute.

### OQ5 — Which bounded interventions preserve required history under repeated computation?

**Status:** explicit memory, input reinjection and stable-loop controls already exist. Their generic usefulness is not open. A possible extension is a budget-matched intervention targeting the verified streaming failure while preserving ordinary performance.

**Decisive test:** compare task-direction preservation, gate adjustment or a justified memory path with simple orthogonal/skip/reinjection/buffer controls. Count all persistent state and task access. Depth loops should be a separate branch: one whole-motif loop has its own forcing unit and cannot inherit the streaming period mechanically.

**Falsifier / decision:** improvement explained by more bytes, retained prompts or extra tuning is an engineering tradeoff, not proof of the proposed mechanism. Do not launch this branch before OQ1–OQ3 produce a result worth mitigating.

## 6. Revised research priority and publication threshold

This audit should change execution order, not merely append citations to the existing broad plan. The full architecture registry remains a coverage target; it is not the first compute request.

**Gate A — analytic and measurement falsification.** Add the exact affine ordering null, the commuting switched null, a nonlinear-readout offset control, stable categorical basins, continuous neutral memory with contracting nuisance directions, amplitude-only contraction, and noise-maintained covariance without cue retention. Validate complete-state products, chronology and conditioning. Existing numerical calibration does not yet certify all these cases.

**Gate B — minimal trained mechanism study.** Start with the vector renderer, GRU and full-state LSTM; use the existing fixed-size cue task, both target types, `q=1`, and the within-support horizons. Freeze matching rules, input statistics, competence criteria, effect-size threshold and probe/intervention protocol before held-out evaluation. Three paired training seeds diagnose implementation and effect scale; confirm a selected claim with fresh seeds and an uncertainty/power rule set before inspection. A null or opposite effect is an admissible outcome.

**Gate C — theory that adds a prediction.** Move from the affine null to switched/nonlinear propagation. Seek an explicit sufficient condition or counterexample connecting ordered products, task directions and finite-noise recoverability, with assumptions visible. Compare against echo-index, covariance and Fisher-memory accounts. A commutator plot or a restated neutral-eigenvector argument is insufficient on its own.

**Gate D — graph and cross-modal confirmation.** Recover and pin legacy assets in parallel. Replicate one graph setting with a valid complete-state adapter, then test one native non-graph task. Only after these gates should the wide 351-checkpoint screen, modern-architecture ranking, or recurrent-depth mitigation consume substantial cluster budget. Compute remains subject to the existing bounded cluv approval process; this review submits no jobs.

A publishable mechanism paper needs: a prespecified excess-retention effect; a model of why it occurs and when it should not; independent confirmation; causal evidence stronger than a geometry correlation; and a consequential native transfer or a sufficiently sharp theoretical result. A careful null/counterexample paper is possible if it systematically disproves common collapse inferences and establishes useful predictive boundaries. Neither outcome is guaranteed by the plan.

## 7. Contribution language and remaining novelty risk

Safe now: “We investigate whether matched higher-order repetitive forcing impairs required-history retention in fixed-weight recurrent models, and test when geometric or local-stability diagnostics do and do not predict that impairment.” This states a research program.

Conditional on successful evidence: “We identify [specified mechanism] under [specified conditions], establish [precise boundary/prediction], and confirm its causal effect on [specified tasks/models] against delay, covariance, readout and attractor alternatives.” Fill every bracket from accepted evidence.

Avoid: “the first Floquet analysis of RNN memory,” “a universal law of representation collapse,” “negative exponents imply lost memory,” “new theory that stable loops can recall injected input,” and “modality-general collapse” based only on isomorphic renderers.

The search is broad but bounded. Several classical records remain abstract/metadata-level, recent preprint code has not been audited, and citation-chain expansion around the closest 2026 papers can reveal further overlap. [The search log](search-log.md) identifies those gaps explicitly. The strongest current recommendation is to make **useful-history loss under controlled forcing**, including its analytic nulls and categorical counterexamples, the center of the project. “Dimension collapse across modalities” should remain a motivating umbrella until the narrower claim survives these tests.
