# Theory novelty review

Audit date: 2026-10-11. Reviewed plan: [periodicity-research-plan.md](../../auto-research/periodicity-research-plan.md). This is a bounded primary-source audit, not an exhaustive priority certification. The 24 bibliographic records and passage locators are in [theory-references.json](../../../../../../doc/literature/evidence/theory-references.json).

The broad story that forcing changes recurrent stability, dimensionality and memory is established. A defensible contribution would identify **when higher-order repetition at fixed weights removes required history**, under controlled history, input statistics, duration and noise, and distinguish that effect from useful compression or selection among stable memory attractors. This is a candidate contribution, not an established novelty claim or an empirical finding.

## Evidence fidelity and scope

Seven entries have `full_text` evidence: six closest papers were read in substantive main-text/method/proof passages (Pals, Ashwin–Ceni, Manjunath–Jaeger, Ganguli, Sharma et al., Fokam et al.), plus the classical Seung line-attractor discussion. This label means the cited full-text passages were inspected; it does not claim every appendix was checked or every theorem independently proved. Fifteen entries are abstract-level leads and two are metadata-level foundations. Do not cite uninspected theorems from the latter groups.

An article year is its verified publication year when available; arXiv version dates are recorded separately where material. Missing identifiers mean unverified, not nonexistent. Names use expanded first names from primary records where available and retain published middle initials rather than invent expansions. The Jaeger short-term-memory report has an explicit 2001/2002 index/title-page conflict; its record states the evidence for choosing 2002. Jaeger's ESP PDF is the corrected 2010 version of a 2001 report.

No experiments were run for this audit. No local PDFs were downloaded. Accessible PDFs were read through the browser. Inherited manuscripts and the plan are not evidence of a replicated result.

## Direct anticipation: five strongest threats

1. **Periodic Floquet working memory is already demonstrated.** Pals et al. locate two stable memory-carrying cycles with a Poincare map and Floquet analysis. Cite Fig. 3 and Methods “Poincare maps and linear stability analysis.” An analysis that merely detects contracting phase-locked recurrent cycles would substantially reproduce this result. The extension must explain task-relevant loss versus retained attractor identity. [Pals et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/)

2. **Repetition-dependent attractor multiplicity has theorems.** Ashwin–Ceni's Assumption 2.1, Lemma 2.2 and Theorems 2.3–2.4 treat long symbol blocks and echo-index transitions; Section 2.3 gives obstructions. The plan's matched whole-motif repetitions differ from symbol-run repetition, but an “input repetition changes memory” headline alone is anticipated. [Ashwin and Ceni, 2023](https://arxiv.org/abs/2309.04728)

3. **Memory geometry already has a mechanistic loss decomposition.** Sharma et al. relate task-relevant temporal feature directions, interference and forgetting: Sections 3–4, Eq. (5), Appendices A–B. Comparing PR against recall error is insufficient novelty without an additional controlled periodicity mechanism. [Sharma et al., ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html)

4. **Correlated-input memory transitions are unusually close.** Fokam et al.'s Lemmas 1–2 and Appendix C concern aligned linear learning. Crucially, Appendix E “Scope condition”/Table S4 distinguish the local memoryless release boundary from first-order global crossings with strong periodicity. A matched lag-one control does not rule out their higher-order covariance effects. [Fokam et al., 2026 preprint](https://arxiv.org/html/2609.00420v1)

5. **Noise-resolved sensitivity is established Fisher-memory theory.** Ganguli et al., pp. 18970–18971, Eqs. (1)–(5), already connect input sensitivity, recurrent dynamics and noise whitening. A task-channel singular spectrum needs a specific target and causal validation; renaming that channel is not a new theory. [Ganguli et al., 2008](https://ganguli-gang.stanford.edu/pdf/08.Memory.pdf)

The parent audit separately covers the 2026 graph perturbation paper [arXiv:2607.28185](https://arxiv.org/abs/2607.28185) and the recent dimensionality paper [DOI:10.1038/s41593-026-02395-w](https://doi.org/10.1038/s41593-026-02395-w). Compare these before claiming novelty for the forced-covariance/noise proposition or a broad recurrence-dimensionality mechanism; this review does not independently certify their findings.

## Established foundations versus possible extensions

The unique forced-response argument is already input-specific ESP theory. Manjunath–Jaeger Definition 2.2, Lemma 2.2 and Proposition 2.3 distinguish uniqueness from dependence on past inputs and imply a periodic response under periodic input plus ESP. Our affine solution is a transparent special case and calibration foundation. Local contraction alone supplies no global echo-index theorem. [Manjunath and Jaeger, 2013](https://www.ai.rug.nl/minds/uploads/ESP_NC.pdf)

The neutral-direction argument is standard continuous-attractor geometry. Seung's pp. 13342–13343 discuss tuned line memory, slow drift and discrete point-attractor alternatives. The cycle-map version follows by differentiation: if smooth `E:U→R^n` has rank `DE=d` and `P(E(u))=E(u)`, then `DP(E(u))DE(u)=DE(u)`, giving at least `d` independent +1 eigenvectors. Exact continuous memory, smooth immersion and a common phase are essential; categorical basins and approximate finite-horizon memory are excluded. [Seung, 1996](https://neurophysics.ucsd.edu/courses/physics_171/seung_pnas_line.pdf)

The full-state/input distinction is essential for depth loops. With `z_(l+1)=G(z_l,x)`, input sensitivity obeys `S_(l+1)=J_l S_l+B_l`, even if same-input state perturbations contract. Equilibrium representations can remain input-dependent. Depth/position schedules or growing caches may make a fixed-dimensional periodic map inapplicable; an append-only cache also retains an identity derivative on stored coordinates. These are measurement requirements, not a new general collapse theorem. [Bai et al., 2019](https://arxiv.org/abs/1909.01377)

The closest **possible** extension is fixed-weight, history-conditioned, higher-order matched forcing with intervention-validated channel loss across native modalities and architectures. Architecture breadth by itself is replication scope, not theoretical novelty. A modality renderer must not reintroduce the query target, change cue SNR, or provide external cache access that defeats the intended memory manipulation. Confirm this by decoding targets from the body-only input and from every exposed cache.

## Strongest rejection arguments

- The observed phenomenon is ordinary delay forgetting. Longer repeated streams also lengthen the retention interval; compare repetition against a same-length nonperiodic arm and analyze the prespecified interaction, not only a within-arm decline.
- Temporal PR saturates at the period-imposed sample-rank limit. A centered cloud of only a single period has rank at most period minus one. Measure independent histories at fixed phase and driver; retain absolute variance and amplitude alongside normalized geometry.
- Geometry is a nuisance-direction statistic. A task-preserving low-rank attractor can retain all finite labels; a scalar contraction can attenuate every task direction without changing normalized PR. Required-target decoding and causal task use must arbitrate the claim.
- Negative exponents are only local evidence. Cue-induced histories can occupy different attracting cycles. Estimate basin identity and perturbation radius instead of assuming global synchronization.
- “Matched bigrams” is stronger language than the actual control. Matching symbol counts and adjacent transitions does not match the full autocovariance, spectrum, higher-order predictability or gate trajectories. Report those differences as designed interventions or residual confounds; fit appropriate covariance/spectrum baselines.
- A Gaussian-linear result is applied to a whole nonlinear representation. For `y=Hu+ε`, `u~N(0,I)`, independent `ε~N(0,Σ)`, `Σ>0`, the posterior covariance is `(I+HᵀΣ⁻¹H)⁻¹`. A target `Tu` has MMSE `tr[T(I+HᵀΣ⁻¹H)⁻¹Tᵀ]`. This is exact only for that channel. State-noise injection, readout-noise injection and dynamical noise lead to different effective `Σ`; a nonlinear local Jacobian is only a local approximation.
- A task-observable output need not reveal full-state instability: `W=diag(a,b)`, `|a|<1<|b|`, input/readout `(1,0)`, gives stable observed output. Full-spectral claims require reachability/observability or genericity. Sharma Appendix D.1 excludes degenerate parameter choices; do not promote it to a universal ESP theorem.
- A stable affine covariance theorem is called new. The periodic affine orbit, discrete Lyapunov covariance and reachable-span rank are classical linear-system foundations. Their value here is establishing a falsifiable baseline and exposing the difference between state noise, activity variance and retained history.

## Concrete falsification and acceptance tests

1. **Stable categorical memory control.** Train or construct two attracting cue-selected cycles under a common periodic driver. If all local multipliers are below one while labels remain robustly separable, reject any universal multiplier-to-forgetting inference. Report inter-attractor separation and within-basin stability separately.
2. **Amplitude-only control.** Use an isotropic `aI` contraction on an isotropic required-history cloud. Normalized participation ratio stays constant while signal amplitude and noise-limited recovery decrease. The measurement pipeline must classify this without declaring geometric rank reduction.
3. **Covariance versus memory control.** Add independent dynamical noise to a stable full-rank affine system. Its stationary covariance can have full rank while old cue influence decays. Require both stationary covariance predictions and cue-channel recovery predictions to agree with analytic values.
4. **Input reinjection control.** Compare hidden-state perturbations against derivatives of the persistently accessible input in a contracting depth loop. Convergence plus retained input sensitivity falsifies any inference that loop-state contraction removes the external evidence.
5. **Higher-order matching test.** Freeze one checkpoint and cue set. Pair same-horizon periodic and nonperiodic streams with matched low-order statistics and final block; quantify full covariance and spectrum. If the proposed repetition effect disappears under better covariance matching, favor a correlation explanation over a new periodicity-specific mechanism.
6. **Task-channel intervention.** At held-out phase/horizon, remove a predicted required-memory direction and compare equal-energy random and nuisance-direction shams. Loss changes must track required-target corruption, not generic state disturbance. Fit probes on disjoint histories and report nonlinear-probe recovery before asserting information loss.
7. **Continuous versus categorical contrast.** Sweep label count/precision, injected noise and delay. Exact continuum neutrality predicts different failure modes from basin-coded discrete retention; finite-label success cannot establish a dimensional lower bound proportional to label count.

An accepted scientific packet requires frozen checkpoint/configuration, complete state definition, independently varied required and nuisance histories, closure diagnostics before Floquet naming, task-noise location and units, untouched evaluation histories, uncertainty across independent seeds/checkpoints, and raw spectra/probe predictions/intervention effects. Accept a mechanistic claim only if the competing amplitude, covariance, delay and multistability explanations have been tested. A null repetition interaction or useful compression is a valid result.

## Search log and remaining gaps

Sources searched: arXiv abstracts/HTML/PDFs, official ICLR and PMLR proceedings, author-hosted papers, PLOS, PNAS/PMC, eLife, University of Groningen and institutional repositories. Discovery results from secondary indexes were used only to locate primary material, not as theorem evidence.

Representative queries, issued during this audit:

- “Trained recurrent neural networks develop phase-locked limit cycles working memory Pals”
- “Transitions echo index dependence input repetitions Ashwin Ceni”
- “Temporal superposition feature geometry RNNs memory demands”
- “How Temporal Correlations Shape Memory Linear Recurrent Neural Networks”
- “Memory traces dynamical systems Ganguli Huh Sompolinsky Fisher”
- “Echo State Property Linked to an Input Manjunath Jaeger”
- “Short-Term Memory Orthogonal Neural Networks White”
- “Information Processing Capacity Dynamical Systems Dambre”
- “Learning dynamics linear recurrent neural networks Proca”
- “Dynamics learning integrate linear recurrent Bordelon”
- “Optimal sequence memory driven random networks Schuecker”
- “Lyapunov spectra chaotic recurrent neural networks Engelken”
- “How brain keeps eyes still Seung”
- “Fundamental limits persistent activity noisy neurons Burak”
- “Fundamental bound persistence capacity short-term memory graded persistent activity”

Access failures: Nature's Dambre page redirected to inaccessible cookie authorization; Europe PMC full text failed; one Fraunhofer report PDF timed out. These entries remain abstract/metadata level. Broad 2025–2026 search was bounded, so no “first” or exhaustive absence claim is licensed. Required follow-up is a verified Dambre theorem reading, inspection of Proca/Bordelon full proofs, and integration with the parent's newest graph/recurrence findings before freezing the paper's contribution statement.

