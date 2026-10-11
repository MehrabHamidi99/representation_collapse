# Theory propositions: useful-history loss under controlled periodic forcing

Version 1, 2026-10-12. Status: consolidated theory document. Everything below is a derivation, a restatement of an established
result, or an explicitly open target; nothing here is an empirical finding about a trained model (see [the tracker](../TRACKER.md)).

## 0. Scope, status header and notation

This document consolidates the theory scattered across the [research program](research-program.md) (discrete framework, Floquet
scope, dimension contract), [periodicity research plan, section 2](periodicity-research-plan.md) (theoretical programme A–E) and the
[novelty audit, section 3](../literature/novelty-review.md) (affine and commuting nulls, continuous-memory consequences, Gaussian
channel). It also corrects the inherited draft [`floquet_analysis/main.tex`](../../floquet_analysis/main.tex) without editing it
(section 3). The paper's theory section and proof appendix are lifted from here; protocol stays in the plan documents.

Working framing: when does repetition erase what a recurrent network needs? The primary question is useful-history loss under
controlled periodic forcing in fixed-weight recurrent models, at equal retention horizon, relative to a matched non-periodic body.
"Dimension collapse across modalities" is the motivating umbrella, not the claim.

**Status labels used throughout.** `established` (textbook or cited result), `derived` (elementary consequence proved here),
`verified-infrastructure` (fixture passes locally; evidence about code, not trained models), `executed-unverified` (job completed,
independent reduction pending), `preliminary-attributed` (inherited draft statement without artifacts), `planned`, `open`.

**Notation.**

- Complete persistent state $s_t\in\mathbb R^n$ immediately before input $x_t$. GRU: $s=h$, $n=128$ in the pilots. LSTM: $s=(h,c)$
  packed in that order, $n=256$ (hidden size 128). A one-layer node-state graph LSTM with $N$ nodes has $n=2NH$, not $2H$; node
  ordering is part of the state.
- Step convention $s_{t+1}=F_\theta(s_t,x_t)$. Jacobian $J_t=\partial_sF_\theta(s_t,x_t)\in\mathbb R^{n\times n}$ at fixed input and
  parameters, acting on column perturbations.
- Finite-time propagator $P_{t,L}=J_{t+L-1}\cdots J_t$ (the earliest factor is applied before later ones; the convention of
  `ordered_product` and `tangent_window`). Singular values $\sigma_i(P_{t,L})$ and finite-time rates
  $\ell_i=\log\sigma_i(P_{t,L})/L$ per update.
- One-period map $M_t=P_{t,P}$, used only when input periodicity $x_{t+P}=x_t$ and complete-state closure $s_{t+P}=s_t$ are verified
  at declared tolerance.
- Episode: sixteen WRITE observations ($t=0,\dots,15$; coordinate $i$ carries value $u_i$; coordinates $0$–$7$ are the required cue
  $u$, $8$–$15$ the nuisance $\nu$), a distractor body of $R$ twelve-symbol motifs over $\{A,B,X\}$ with dwell $q$ (period $P=12q$,
  body length $H=12qR$), and a terminal QUERY revealing one required coordinate. Input dimension 38 (`type[5]`, `coordinate[16]`,
  `categorical value[16]`, `real value[1]`). Categorical values are sixteen equiprobable symbols; continuous values are
  $\mathrm{Uniform}[-1,1]$ (population variance $1/3$).
- Arms: periodic (one motif repeated $R$ times) versus matched varying ($R$ motif identifiers with the same final motif and no
  proper whole-body period). Matched by construction: symbol counts, cyclic bigrams, run lengths, start/end context, total updates,
  final motif, cue/query/nuisance. $R=1$ is a shared anchor; $R\in\{2,4,8,16\}$ are training horizons; $R=32,64$ are extrapolation.
- Cue-conditioned cloud $\mathcal C(h,\nu,R)=\{s_{\text{pre-query}}(u,\nu,h)\}_{u}$: vary the required cue over $N_{\text{cue}}$
  draws at one fixed body history $h$, nuisance $\nu$ and phase. Pooled-body clouds are not cue-channel clouds.
- Hypotheses: H1 harmful collapse, H2 sufficient compression, H3 input synchronization, H4 measurement artifact; named alternatives:
  readout-offset effect and stable categorical attractor memory.

Estimands kept separate: (i) covariance geometry of cue-conditioned clouds (PR, covariance entropy dimension, $D_{90}/D_{95}$,
absolute variance, spectrum); (ii) tangent propagation ($P_{t,L}$, singular values, rates; multipliers only under verified closure);
(iii) behavior and information (native error, held-out probes, counterfactual cue separation, causal interventions with shams). No
single proxy proves collapse.

## 1. Discrete framework and when the word Floquet applies

**Proposition 1.1 (discrete monodromy; established).** Let $x_{t+P}=x_t$ and let a complete-state orbit satisfy $s^*_{t+P}=s^*_t$
exactly. Then $J_{t+P}=J_t$ and the one-period map $M_t=J_{t+P-1}\cdots J_t$ is the Jacobian of the $P$-step return map at $s^*_t$;
its eigenvalues are the discrete Floquet multipliers. *Proof.* Chain rule over $P$ steps; $J_{t+P}=J_t$ because state and input both
repeat. $\square$ The nonzero multipliers are phase-independent: with $X=J_{t+P-1}\cdots J_{t+1}$, $M_t=XJ_t$ and $M_{t+1}=J_tX$
share their nonzero spectrum. Singular values of $M_t$ do depend on the phase.

**Proposition 1.2 (observed period; established).** If the orbit closes only after $kP$ steps ($k>1$, a subharmonic response) then
the $P$-step product maps $s^*_t$ to $s^*_{t+P}\neq s^*_t$ and is not a return-map derivative; the monodromy is the $kP$-step
product. Use the observed orbit period, never the input period alone. Multiple attracting responses can coexist ([Ashwin and Ceni
2023](../literature/bibliography.md#ashwin2023-echo-repetitions); [Ceni et al. 2020](../literature/bibliography.md#ceni2020)).

**Proposition 1.3 (no automatic neutral multiplier; established).** A forced discrete periodic orbit need not have a multiplier of
modulus one. *Proof.* P5 below exhibits $s_{t+1}=as_t+bx_t$ with periodic nonconstant $x$: the response is a nonconstant periodic
orbit with the single multiplier $a^T$, $|a^T|<1$. $\square$ The neutral direction of an autonomous continuous-time limit cycle
comes from time-translation symmetry; a recurrence with an external phase has none. In the suspended autonomous system $(s,\phi)$,
$\phi_{t+1}=\phi_t+1 \bmod P$, the trivial multiplier $1$ lives in the phase coordinate, which is not a recurrent-state direction.

**Proposition 1.4 (finite-time propagator otherwise; established).** Without verified input periodicity and closure, the object
$P_{t,L}$ is a finite-time propagator. Report $\sigma_i(P_{t,L})$, the largest gain $\sigma_1$, and rates $\log\sigma_i/L$; call
long-horizon QR estimates conditional Lyapunov exponents under replayed forcing ([Vogt et al.
2020](../literature/bibliography.md#vogt2020)). Eigenvalues of a non-return product have no orbit-stability meaning.

**Remark 1.5 (Euler surrogate; derived).** Writing $G(s,x)=F(s,x)-s$ and $\dot s=G(s,x(t))$ defines an Euler surrogate whose
unit-time flow is not $F$. Scalar check: $F(s)=as$ gives $G(s)=(a-1)s$ and flow $\varphi_1(s)=e^{a-1}s$; $e^{a-1}=a$ only at $a=1$,
and no scalar autonomous flow realizes $a<0$ (one-dimensional flows preserve orientation). An exact embedding needs a logarithm of
the step map; for invertible linear $J$ without eigenvalues on $(-\infty,0]$ one may take $\dot z=(\log J)z$ piecewise, as the
draft's commented-out linear section does. Nothing guarantees this for the nonlinear GRU/LSTM cell, and nothing requires it: the
discrete product in 1.1 is the correct object.

## 2. Propositions P1–P12

Each proposition lists assumptions, statement, proof (full when elementary), numerical fixture, counterexample or limits, and the
empirical prediction for the vector task with its refuting observation. Fixtures are `checks` blocks in
[`mechanism_calibration.py`](../../src/representation_collapse/mechanism_calibration.py) (36 checks) and
[`calibrate.py`](../../src/representation_collapse/calibrate.py) (10 checks); tests live under [`tests/`](../../tests/); check names
are in section 4.

### P1 — Affine ordering null (derived; fixture verified-infrastructure)
- **Assumptions.** During the body the update is affine with a fixed state matrix, $s_{t+1}=As_t+Bx_t+b$ for $t=0,\dots,H-1$ (body
  time), with $A$ independent of $x_t$. The post-WRITE state is $s_0=E(u,\nu)$ for an arbitrary, possibly nonlinear encoder $E$.
  Optional additive process noise $\xi_t$ independent of $u$. Two bodies $h,h'$ of equal length $H$.
- **Statement.** $s_H(u,h)=A^HE(u,\nu)+k(h)+\sum_{j<H}A^{H-1-j}\xi_j$ with offset $k(h)=\sum_{j<H}A^{H-1-j}(Bx_j+b)$. Hence the cue
  difference $\Delta s_H=A^H\big(E(u)-E(u')\big)$, the cue derivative $\partial s_H/\partial u=A^H\,D_uE(u)$ and the centered
  cue-conditioned covariance $A^H\,\mathrm{Cov}(E(u))\,A^{H\top}$ are independent of body ordering at equal $H$; only $k(h)$
  changes. If the joint law of $(\xi_0,\dots,\xi_{H-1})$ is the same for both arms (common iid noise suffices), the propagated noise
  law $\sum_j A^{H-1-j}\xi_j$ is identical too. *Readout-offset corollary.* For a common linear readout $y=Ws+c$,
  $y=WA^HE(u)+\big(Wk(h)+c\big)$: the cue-dependent component is unchanged while an arm-dependent additive bias
  $W\big(k(h)-k(h')\big)$ can change raw error; a decoder allowed to compensate the deterministic offset has identical risk in both
  arms. A nonlinear readout $g(Ws+c)$ has cue sensitivity $g'(Ws_H+c)\,WA^HD_uE$, which does respond to the shifted operating point.
- **Proof.** Induction on $t$: $s_{t+1}=A s_t+Bx_t+b+\xi_t$ unrolls to the displayed formula. Ordering enters only through $k(h)$
  and the noise term; the noise term's law depends on the joint law of the $\xi_j$ because the cross terms
  $A^{H-1-j}\mathrm{Cov}(\xi_j,\xi_l)A^{(H-1-l)\top}$ change with temporal correlations even at equal per-step marginals. The
  corollary is substitution. $\square$
- **Numerical fixture.** `affine_ordering_null` block of `mechanism_calibration.calibration` (nine checks); `conditioned_dimension`
  refuses pooled-body clouds; test `test_affine_body_order_changes_offset_not_conditioned_cue` in
  [`tests/test_mechanism_calibration.py`](../../tests/test_mechanism_calibration.py). Scratch check in section 4.
- **Counterexample / limits.** Fails as soon as $A$ depends on the symbol (P2–P3), the body update is nonlinear (gates, saturation),
  the noise joint law differs by arm, or the readout is nonlinear. It says nothing about absolute retention: $A^H$ can be tiny in
  both arms (ordinary horizon forgetting). Only the genuinely affine part of a network is covered; a diagonal state-space layer with
  input-independent transition is covered, its surrounding nonlinearities are not.
- **Empirical prediction.** If a model's body dynamics are effectively affine on the reached states, the conditioned cue derivative
  and the centered cue-conditioned clouds agree across arms at equal $H$ (same cue sample, same $\nu$), and any arm difference in
  raw error is removed by an offset-aware decoder. The null is rejected for that model when the conditioned cue derivative or
  centered cloud differs between arms beyond the replay tolerance; this is what the gated mechanisms must show.

### P2 — Commuting switched-affine null (derived; fixture verified-infrastructure)
- **Assumptions.** $s_{t+1}=A_{x_t}s_t+b_{x_t}$ with $A_x$ depending only on the current body symbol; the matrices $\{A_x\}$ commute
  pairwise; both arms have the same symbol counts $n_x$.
- **Statement.** The product acting on cue differences is $\Pi_{\text{arm}}=\prod_tA_{x_t}=\prod_xA_x^{\,n_x}$, identical across
  arms; so $\Delta s_H$, the cue derivative and the centered conditioned covariance coincide. The offset
  $k(h)=\sum_j\big(\prod_{i>j}A_{x_i}\big)b_{x_j}$ and the propagated process-noise covariance
  $\sum_j\big(\prod_{i>j}A_{x_i}\big)\Sigma\big(\prod_{i>j}A_{x_i}\big)^{\top}$ depend on the partial products and therefore on
  order. The null covers the noiseless cue channel or common terminal noise, not intermediate noise.
- **Proof.** Commuting factors may be reordered freely, and matched counts give the same multiset of factors. The fixture example
  $A=\mathrm{diag}(0.5,0.8)$, $B=\mathrm{diag}(0.9,0.4)$ with $\Sigma=I$ added after each update gives $\mathrm{diag}(1.81,1.16)$
  when $A$ acts before $B$ and $\mathrm{diag}(1.25,1.64)$ when $B$ acts before $A$, although $AB=BA$. $\square$
- **Numerical fixture.** `commuting_ordering_null` block (three checks) and `propagated_noise_covariance`; test
  `test_commuting_product_null_does_not_imply_process_noise_null`.
- **Counterexample / limits.** Cyclic-bigram matching does not imply commutation. Simultaneous diagonalizability is strong, but it
  holds for one diagonal selective state-space recurrence whose transition depends only on the current body symbol, so for a single
  such layer P2 predicts no cue-channel ordering effect at matched counts; stacked layers break the assumption because later layers
  receive cue-dependent inputs.
- **Empirical prediction.** For architectures with commuting symbol-dependent transitions, conditioned cue differences are
  order-independent at matched counts, while FP32/FP64 replay differences and offsets may still differ by arm. Refuted for a given
  layer if its conditioned cue difference depends on order with the transitions verified to commute.

### P3 — Noncommutativity is necessary but not sufficient (derived)
- **Assumptions.** Switched-affine body as in P2 without the commutation assumption; matched symbol counts.
- **Statement.** (a) If $\Pi_{\text{arm}}\neq\Pi_{\text{arm}'}$ then some pair $A_x,A_{x'}$ fails to commute. (b) Noncommutativity
  does not imply a cue-channel ordering effect, in three senses: (i) products may differ yet agree on the task directions; (ii)
  products may differ by an isometry, leaving singular values and isotropic-noise recoverability unchanged; (iii) products may
  differ by less than the finite noise floor. (c) Ordering-specific loss can still occur; the fixture's positive construction shows
  it.
- **Proof.** (a) is the contrapositive of P2. (b-i) Upper-triangular $A=\begin{pmatrix}0.9&0.4\\0&0.6\end{pmatrix}$,
  $B=\begin{pmatrix}0.7&-0.5\\0&0.8\end{pmatrix}$ share the eigenvector $e_1$; $AB\neq BA$ and $B^2A^2\neq BABA$, but every ordered
  product acts on $e_1$ as $0.9^{n_A}0.7^{n_B}$. Cue differences in $\mathrm{span}(e_1)$ feel no ordering. (b-ii) Rotations about
  different axes in $SO(3)$ do not commute, their ordered products differ, yet every product is orthogonal: equal singular values,
  equal PR of a propagated cloud and equal Gaussian risk $\mathrm{tr}(I+\Pi^{\top}\Pi/\sigma^2)^{-1}$. (c) The fixture pair
  $A=\begin{pmatrix}0.8&0.3\\0&0.7\end{pmatrix}$, $B=\begin{pmatrix}0.7&0\\0.3&0.8\end{pmatrix}$ has $B^2A^2\neq BABA$ with
  different linear recovery risks at terminal noise $0.04I$. $\square$
- **Numerical fixture.** `noncommuting_construction` block (four checks); test
  `test_noncommuting_positive_construction_has_order_specific_risk`; chronology is checked by
  `test_noncommuting_product_and_perturbation` in [`tests/test_metrics.py`](../../tests/test_metrics.py) and
  `test_noncommuting_chronological_product` in [`tests/test_periodicity.py`](../../tests/test_periodicity.py). The (b-i) and (b-ii)
  counterexamples are scratch-verified only (section 4).
- **Counterexample / limits.** A commutator norm is not an effect size. The relevant quantity is the ordered product restricted to
  the cue directions, compared with the noise floor (P9, P12).
- **Empirical prediction.** Finding $J_t$ that do not commute along a trajectory (generic for gated cells) predicts nothing by
  itself. The prediction with content is a measured arm difference of $\Pi$ on the cue directions,
  $\|(\Pi_{\text{per}}-\Pi_{\text{con}})\,\partial s_0/\partial u\|$, exceeding the replay tolerance. Refuted if products differ
  only off cue directions.

### P4 — Continuous memory requires dimension (established; Seung 1996)
- **Assumptions.** $U\subset\mathbb R^d$ open; encoder $E:U\to\mathbb R^n$ and decoder $g:\mathbb R^n\to\mathbb R^d$ of class $C^1$
  with exact recovery $g(E(u))=u$ on $U$. For the second part, a $C^1$ map $\mathcal P$ (a period map at fixed phase, or any fixed
  body map) with $\mathcal P(E(u))=E(u)$ on $U$.
- **Statement.** (a) $\operatorname{rank}DE(u)=d$ for all $u\in U$. (b) $D\mathcal P(E(u))\,DE(u)=DE(u)$, so the $d$-dimensional
  column space of $DE(u)$ lies in the eigenspace of $D\mathcal P(E(u))$ for eigenvalue $+1$: at least $d$ linearly independent unit
  multipliers. A map with all multipliers of modulus below one cannot preserve an exact continuous memory family.
- **Proof.** (a) Chain rule: $Dg(E(u))\,DE(u)=I_d$, so $DE(u)$ has a left inverse and is injective; $DE(u)$ is $n\times d$, hence
  rank exactly $d$. (b) Differentiate $\mathcal P\circ E=E$ and apply (a). $\square$ Antecedent: line attractors for analog
  persistent memory, [Seung 1996](../literature/bibliography.md#seung1996).
- **Numerical fixture.** `continuous_neutral_memory` block (four checks): product $\mathrm{diag}(1,1,2^{-16},2^{-16})$ keeps two
  required coordinates exactly (two unit multipliers) while PR falls from 4 to 2; test
  `test_continuous_required_memory_survives_nuisance_compression`.
- **Counterexample / limits.** Excluded: categorical coding (finitely many states can be strictly attracting, P11); approximate
  storage ($(1-\delta)^R$ contraction over $R$ periods is tolerable while $R\delta$ is below readout resolution); finite horizon (no
  invariant family); retained external input (re-presenting the cue gives $D_s\mathcal P\,DE+D_u\mathcal P=DE$, no constraint on
  $D_s\mathcal P$). The bound is on algebraic rank, not PR: an anisotropic $d$-dimensional cloud may have PR far below $d$ (P7).
- **Empirical prediction.** A competent continuous model (raw MSE $\le1/60$ on each of eight coordinates) must have a conditioned
  cue Jacobian with eight singular values above the readout's resolution on the tested cue region, hence body-propagator gains on
  the cue directions above that resolution; exact unit multipliers are not required at finite $R$. Competence together with fewer
  than eight such singular values would indicate quantized (categorical) coding and remove the model from P4's scope, not refute P4.

### P5 — Forced scalar contracting recurrence (established)
- **Assumptions.** $s_{t+1}=as_t+bx_t$, $0<|a|<1$, $b\neq0$, $x$ periodic with period $T$ and nonconstant.
- **Statement.** There is a unique, globally attracting $T$-periodic response $s^*_t=b\sum_{j\ge0}a^jx_{t-1-j}$; at phase $0$,
  $s^*_0=b(1-a^T)^{-1}\sum_{k=0}^{T-1}a^{T-1-k}x_k$. The response is nonconstant. The variational equation $\delta_{t+1}=a\delta_t$
  is input-independent, so the one-period multiplier is $a^T$ with $|a^T|<1$. The power spectrum of $s^*$ is a line spectrum at
  multiples of $1/T$ with the harmonic support of $x$, because $s^*$ is $x$ passed through the LTI filter
  $b\,e^{-i\omega}/(1-ae^{-i\omega})$, whose gain never vanishes. A PSD peak at the forcing frequency thus coexists with a strictly
  contracting multiplier. Vector form: $s_{t+1}=As_t+Bx_t$, $\rho(A)<1$, has $s^*_0=(I-A^T)^{-1}\sum_kA^{T-1-k}Bx_k$ and $M=A^T$.
- **Proof.** The geometric series converges since $|a|<1$; periodicity of $x$ makes $s^*$ periodic; any other solution differs by
  $a^t\delta_0\to0$. If $s^*$ were constant, $s^*_{t+1}-s^*_t=a(s^*_t-s^*_{t-1})+b(x_t-x_{t-1})$ would force $x$ constant. The
  filter statement is the convolution theorem for the periodic steady state. $\square$
- **Numerical fixture.** `calibrate.calibration`: checks `forced_orbit_periodic`, `forced_orbit_nonconstant`,
  `forced_orbit_strict_contraction` ($a=1/2$, $T=4$, two-dimensional forcing) and `rotation_period_multiplier`; test
  `test_calibration_controls`. `metrics.period_spectrum` documents that it does not verify periodicity. Scratch check in section 4
  ($a=0.7$, $T=12$, motif-coded forcing).
- **Counterexample / limits.** Linear and scalar. Nonlinear forced systems can have several coexisting periodic responses,
  subharmonic or aperiodic responses; then uniqueness fails and the echo index exceeds one. For nonnormal $A$, $\rho(A^T)<1$ need
  not bound $\|A^T\|$ (P8).
- **Empirical prediction.** A periodic-arm complete-state trajectory that approximately closes, with all finite-window singular
  values below one and a PSD line at $1/12$, is consistent with input-driven computation (H3) and says nothing about cue retention,
  which must be read from the conditioned cue difference and behavior. The inherited inference "PSD peak implies neutral mode" is
  refuted analytically; the checkpoint diagnostics should show the peak with $\rho(P)<1$ (executed-unverified).

### P6 — Sampling rank ceilings (derived)
- **Assumptions.** Rows of $Z\in\mathbb R^{N\times n}$ take at most $m$ distinct values, each with positive frequency.
- **Statement.** The centered matrix has rank at most $m-1$. Consequently: (a) the time-only centered covariance of any window of an
  exact period-$P$ orbit has rank $\le P-1$ ($\le11$ for $q=1$, $\le23$ for $q=2$), regardless of $n$; (b) a cloud of
  $N_{\text{cue}}$ cues has centered rank $\le N_{\text{cue}}-1$ ($\le511$ for 512 cues, non-binding for $n=128$ or $256$); (c) two
  histories give rank $\le1$ and PR identically one whenever variance is nonzero, so the two-history periodicity diagnostic is not a
  dimension estimate. Finite-sample bias is separate from the ceiling: for an isotropic $d$-dimensional cloud with $N$ samples the
  expected sample PR is approximately $d/\big(1+(d+1)/(N-1)\big)$ (Wishart second moments), about $102$ for $d=128$, $N=512$.
- **Proof.** Distinct rows $v_1,\dots,v_m$ with frequencies $w_k>0$, $\sum w_k=1$, mean $\bar z=\sum w_kv_k$; the centered distinct
  rows $c_k=v_k-\bar z$ satisfy $\sum_kw_kc_k=0$, a nontrivial relation, so their span has dimension $\le m-1$, and every centered
  row equals some $c_k$. (a)–(c) substitute $m=P$, $m=N_{\text{cue}}$, $m=2$. The bias formula follows from
  $\mathbb E\,\mathrm{tr}\,\hat C^2=\mathrm{tr}\,C^2(1+1/m)+(\mathrm{tr}\,C)^2/m$ with $m=N-1$ for Gaussian samples. $\square$
- **Numerical fixture.** Two-row conditioned cloud with numerical rank 1 in
  `test_wrong_conditioning_is_rejected_and_creates_spurious_variance`; the two-history limit is stated in [periodicity analysis
  v1](periodicity-analysis-v1.md). No repository fixture yet exercises (a) on a repeated orbit; scratch check in section 4.
- **Counterexample / limits.** Upper bounds only; rank can be lower (orbit in a subspace). No information content is implied.
- **Empirical prediction.** A time-pooled "collapse to at most eleven dimensions" inside one periodic episode is a sampling artifact
  (H4), and time-pooled PR comparisons between arms are confounded because the control body visits more inputs; hence clouds vary
  cues at a fixed body. Refuted only by a time-pooled centered rank above $P-1$ on a verified exact orbit, which cannot occur.

### P7 — Participation-ratio invariances and non-invariances (established)
- **Assumptions.** Centered covariance $C$ with eigenvalues $\lambda_i\ge0$,
  $\mathrm{PR}=(\sum\lambda_i)^2/\sum\lambda_i^2\in[1,\operatorname{rank}C]$; likewise the covariance entropy dimension
  $\exp(-\sum p_i\log p_i)$, $p_i=\lambda_i/\sum\lambda$, and $D_{90}/D_{95}$.
- **Statement.** All three normalized dimensions are invariant to translation, orthogonal rotation and isotropic scaling, and change
  under anisotropic rescaling. Isotropic attenuation $s\mapsto\alpha^Hs$ preserves them while total variance falls as $\alpha^{2H}$.
  At zero variance they are undefined ($0/0$); the implementation returns a sentinel with `normalized_dimension_defined=false`. The
  covariance spectrum is not the recurrence spectrum: for $s_{t+1}=As_t+\xi_t$ the stationary $C$ solves $C=ACA^{\top}+\Sigma$, and
  matrices with equal eigenvalues ($\mathrm{diag}(0.9,0.1)$ versus $\begin{pmatrix}0.9&5\\0&0.1\end{pmatrix}$) give distinct $C$
  spectra.
- **Proof.** Translation is removed by centering; $C\mapsto Q^{\top}CQ$ preserves eigenvalues; $\lambda_i\mapsto\alpha^2\lambda_i$
  leaves $p_i$ unchanged; $C\mapsto DCD$ with $D=\mathrm{diag}(1,10)$ sends $I_2$ to eigenvalues $(1,100)$ and PR from $2$ to
  $\approx1.02$. The Lyapunov statement is direct substitution. $\square$ Antecedents: [Hu and Sompolinsky
  2022](../literature/bibliography.md#hu2022), [Roy and Vetterli 2007](../literature/bibliography.md#roy2007effective).
- **Numerical fixture.** `metrics.covariance_dimension` with `test_dimension_translation_rotation_and_scale_invariance`,
  `test_known_spectrum`, `test_zero_variance_is_marked_degenerate`; `amplitude_only` block (two checks) with
  `test_isotropic_attenuation_changes_amplitude_not_normalized_dimension`; three `calibrate` checks (isotropic, rank-one, zero).
- **Counterexample / limits.** PR summarizes variance concentration, not the number of task variables (P4) and not information
  (fresh-noise fixture: `fresh_noise_pr_remains_high` with `fresh_noise_linear_cue_recovery_deteriorates`). Joint LSTM $(h,c)$ PR
  depends on the relative scales of $h$ and $c$; report them separately as well.
- **Empirical prediction.** Lower absolute variance with equal normalized PR across arms at equal $H$ is attenuation, not
  concentration; equal variance with lower PR is concentration. Refuted descriptions: calling attenuation "dimension collapse", or
  comparing PR across models with different state coordinates without fixing units.

### P8 — Nonnormal transient amplification (established)
- **Assumptions.** Square $A$ or a time-varying sequence $J_t$.
- **Statement.** $\rho(A)<1$ does not bound $\|A^k\|_2$ for finite $k$: $A=\begin{pmatrix}0.8&4\\0&0.8\end{pmatrix}$ has
  $\sigma_1(A)>1$. For time-varying products the spectral radii of the factors do not bound the product at all:
  $A_1=\begin{pmatrix}0&2\\0&0\end{pmatrix}$, $A_2=\begin{pmatrix}0&0\\2&0\end{pmatrix}$ are nilpotent yet
  $A_2A_1=\mathrm{diag}(0,4)$. Even with every multiplier of a verified $M$ inside the unit circle, within-period perturbations can
  grow; a multiplier of modulus $0.42$ per period does not imply monotone within-period decay.
- **Proof.** Direct computation; the general fact is the gap between spectral radius and operator norm for nonnormal matrices
  (pseudospectral theory). $\square$
- **Numerical fixture.** `calibrate` checks `nonnormal_stable_eigenvalues`, `nonnormal_transient_gain`;
  `test_nonnormal_gain_is_distinct_from_eigenvalues`; `periodicity.tangent_window` reports eigenvalues and singular values apart.
- **Counterexample / limits.** Transient growth followed by decay still forgets asymptotically; the amplified direction may be
  irrelevant to the task, so $\sigma_1$ alone is not a memory statement.
- **Empirical prediction.** Gated cells have nonnormal $J_t$, so one-motif products with $\rho(P)<1$ and $\sigma_1(P)>1$ are
  expected; measure $\|P_{t,\ell}v\|$ for $\ell=1,\dots,12$ along cue directions $v$ and apply direct kicks. Refuted for a window if
  $\sigma_1(P)\approx\rho(P)$ (near-normal product).

### P9 — Gaussian channel recoverability (established; declared surrogate)
- **Assumptions.** $z=Hu+\eta$, $u\sim\mathcal N(0,I_d)$, $\eta\sim\mathcal N(0,\Sigma)$ independent, $\Sigma\succ0$; optionally
  $u\sim\mathcal N(0,\Lambda)$.
- **Statement.** $\mathrm{Cov}(u\mid z)=(I+H^{\top}\Sigma^{-1}H)^{-1}$, and with prior $\Lambda$,
  $(\Lambda^{-1}+H^{\top}\Sigma^{-1}H)^{-1}$; the Bayes risk is the trace. For any centered cue with identity covariance (Gaussian
  or not) the same trace is the population risk of the best *linear* decoder. Factor the channel at the pre-query state as
  $H=C\Phi G$: $G=D_uE$ (cue injection by WRITE), $\Phi$ the body propagator on the cue directions, $C$ the readout Jacobian ($C=I$
  for the state channel). Under P1–P2 $\Phi$ is arm-independent, so equal terminal noise gives equal risk; under P3(c) it differs.
  The real cues are uniform or categorical, so this is a declared calibration surrogate and linear-decoder risk, not the task's
  Bayes risk.
- **Proof.** Joint Gaussian conditioning gives $\mathrm{Cov}(u\mid z)=I-H^{\top}(HH^{\top}+\Sigma)^{-1}H$, equal to
  $(I+H^{\top}\Sigma^{-1}H)^{-1}$ by the Woodbury identity; the prior case replaces $I$ by $\Lambda$. Linear MMSE depends only on
  second moments. $\square$ Antecedent: Fisher memory of linear networks, [Ganguli, Huh and Sompolinsky
  2008](../literature/bibliography.md#ganguli2008-memory-traces).
- **Numerical fixture.** `linear_recovery_risk`; tests `test_noise_location_and_whitening_in_linear_channel` (whitening invariance;
  zero channel gives risk $d$), `test_noncommuting_positive_construction_has_order_specific_risk`;
  `amplitude_only.terminal_noise_linear_mse`; `fresh_noise_linear_decoder_risk_matches`.
- **Counterexample / limits.** Noise location matters (terminal versus process, P2); nonlinear models give local approximations;
  categorical cues need finite differences and basin membership (P11), not derivatives.
- **Empirical prediction.** For a competent continuous model, held-out probe MSE across arms and drivers should track
  $\mathrm{tr}(I+H_{\text{arm}}^{\top}\Sigma^{-1}H_{\text{arm}})^{-1}$ with $H_{\text{arm}}$ the conditioned cue Jacobian and
  $\Sigma$ a declared noise model (injected noise or the FP32 floor). Refuted if measured recovery is unrelated to the predicted
  risk on held-out drivers, which would mean the local linear channel is the wrong description.

### P10 — Input reinjection in depth loops (established; Labovich 2026, DEQ)
- **Assumptions.** Depth recurrence $z_{\ell+1}=G_\ell(z_\ell,x)$ with the same $x$ available at every loop;
  $J_\ell=\partial_zG_\ell$, $B_\ell=\partial_xG_\ell$.
- **Statement.** Input sensitivity obeys $S_{\ell+1}=J_\ell S_\ell+B_\ell$ while initial-state sensitivity is
  $J_{\ell-1}\cdots J_0$. Under contraction the latter vanishes while $S_\ell$ stays of order one; at a tied fixed point with
  $\rho(J_*)<1$, $S_*=(I-J_*)^{-1}B_*$. Calibration $z_{\ell+1}=az_\ell+(1-a)x$, $|a|<1$: $z_\ell=a^\ell z_0+(1-a^\ell)x$, so the
  initial state is forgotten and $x$ is preserved exactly in the limit.
- **Proof.** Chain rule for $S$; geometric (Neumann) series for $S_*$, equivalently the implicit function theorem at $z_*=G(z_*,x)$.
  $\square$ Antecedents: [Labovich 2026](../literature/bibliography.md#looped_stability_recall) (equilibrium sensitivity
  $(I-J)^{-1}B$), [Deep Equilibrium Models](../literature/bibliography.md#deep_equilibrium).
- **Numerical fixture.** None yet in `src/`; planned under [periodicity research plan, sections 2E and
  3D](periodicity-research-plan.md).
- **Counterexample / limits.** Requires convergence; non-tied layers change the formula; a growing cache changes the state dimension
  and stores old entries explicitly. Sensitivity to a re-presented $x$ is not recall of a prefix written once. One loop is one
  forcing unit: a fixed context repeated across loops has period one loop, so the twelve-step streaming period does not transfer.
- **Empirical prediction.** In the persistent-access control, contraction of $\partial z_\ell/\partial z_0$ coexists with order-one
  $\partial z_\ell/\partial x$ on tag directions and intact recall; in the state-carried regime the same contraction erases the tag.
  Refuted if persistent-access loops lose the tag with $S_*$ of full rank on tag directions (access failure), or if the
  state-carried regime retains it under uniform contraction (non-contracting directions exist, P4/P11).

### P11 — Stable categorical basins under local contraction (established; Pals-type control)
- **Assumptions.** A period-$P$ forced map with $K$ distinct periodic orbits, one per category; each orbit has all multipliers of
  modulus below one and a forward-invariant neighborhood of radius $r$ on which the $P$-step map is a contraction; the WRITE phase
  places each cue inside the neighborhood of its category's orbit; orbit separation exceeds $2r$.
- **Statement.** The category is preserved at every horizon and recovered by basin membership. Within each category the cloud
  contracts toward its orbit, so within-category variance tends to zero and normalized PR becomes undefined, while between-category
  separation stays above the orbit separation minus $2r$. Local contraction, lost within-category dimension and retained finite
  memory are compatible. Measure basin identity separately from within-basin contraction: finite inter-cue differences, basin
  membership across $R$, the largest label-preserving perturbation radius, and the number of attracting responses per driver (echo
  index), not local multipliers alone.
- **Proof.** Contraction mapping on each invariant neighborhood gives convergence to the orbit and label preservation by invariance;
  the separation bound is the triangle inequality. $\square$ Empirical antecedent: memory-selective stable phase-locked cycles,
  [Pals, Macke and Barak 2024](../literature/bibliography.md#pals2024-phase-locked-memory).
- **Numerical fixture.** `categorical_forced_basins` block (eight checks): scalar map $r\mapsto1.5r-0.5r^3$ relative to alternating
  phase offsets $\pm0.2$, fixed points $r=\pm1$ with derivative $0$, invariant radius $0.05$ (derivative bound $0.154$, next-error
  bound $0.0038$), 128 perturbed initial states, labels retained with accuracy $1$, separation above $1.9$; test
  `test_local_contracting_forced_basins_preserve_finite_memory`.
- **Counterexample / limits.** Basins must exist and WRITE must route cues into them; eight required coordinates with sixteen values
  need $16^8$ distinguishable terminal states (32 bits), a large but finite memory. Basin merging under one driver but not the other
  is itself an H1 mechanism (cue-dependent basin selection/merging). Basins narrower than FP32 spacing are not robust.
- **Empirical prediction.** A competent categorical model can show small or contracting within-cloud variance with accuracy above
  95% and between-label separation above the perturbation radius. The basin account is refuted for a model if labels are recoverable
  but same-label states do not cluster (continuous coding), or if sub-tolerance perturbations flip labels (no robust basins).

### P12 — Candidate sufficient condition for ordering-specific loss (open; Gate C target)
- **Status.** Open; no theorem is claimed. This records what would have to be proved, in which quantities, what a counterexample
  looks like, and which accounts it must be compared against.
- **Setting.** Gated or otherwise nonlinear switched propagation $s_{t+1}=F(s_t,x_t)$ (GRU: $(1-z_t)\odot n_t+z_t\odot s_t$ with
  input- and state-dependent gates). Body inputs are cue-independent, but the states are cue-dependent, so $J_t=J(s_t(u),x_t)$
  depends on $u$; this cue dependence of the propagator is impossible in P1–P3 and is the primary signature to measure. Task
  directions $T_t(u)=\operatorname{col}\,\partial s_t/\partial u$. The conditioned body propagator on task directions is
  $\Phi_{\text{arm}}(u)=\partial s_H/\partial s_0$ restricted to $T_0(u)$, with
  $\partial s_H/\partial u=\Phi_{\text{arm}}(u)\,\partial s_0/\partial u$.
- **Target statement (to be proved or refuted).** Under matched counts, give verifiable conditions on gate statistics along each arm
  (for example an upper bound $z_{\max}<1$ on the retained-state gate along the periodic orbit, or a bound on
  $\|\partial n/\partial s\|$ in the reached region) implying a per-motif contraction factor on task directions
  $\kappa_{\text{per}}=\sup_u\|\Phi_{\text{motif}}(u)|_{T(u)}\|<1$ and a larger lower bound $\kappa_{\text{con}}$ for the control
  arm, so that for a declared noise floor $\varepsilon$ and initial separation $\Delta_0$ the periodic arm's conditioned cue
  separation falls below $\varepsilon$ at $R^*=\log(\Delta_0/\varepsilon)/\log(1/\kappa_{\text{per}})$ while the control arm's does
  not within the tested $R$. Categorical version: the periodic driver merges cue-selected basins (echo index one) while the varying
  driver keeps them separated. Quantities: ordered products on task directions, finite noise floor, gate saturation, basin counts.
- **What a counterexample looks like.** A trained model in which periodic forcing stabilizes cue-dependent basins
  ($\kappa_{\text{per}}\ge\kappa_{\text{con}}$ on task directions, P11), or in which both arms synchronize at the same rate because
  matched bodies are also repetitive at the bigram level; both predict a null or reversed DID and are admissible outcomes.
- **Accounts to compare against.** Echo index and repetition theorems ([Ashwin and Ceni
  2023](../literature/bibliography.md#ashwin2023-echo-repetitions)): ordering-specific loss would correspond to a periodic driver
  with a single entrained response and a varying driver with several. Covariance accounts (P7): PR is neither necessary nor
  sufficient. Fisher-memory accounts ([Ganguli, Huh and Sompolinsky 2008](../literature/bibliography.md#ganguli2008-memory-traces);
  P9): under P1 the memory curve is order-independent, so ordering dependence at equal counts in gated models is the content the
  result must add. Training correlation accounts ([Fokam et al.
  2026](../literature/bibliography.md#fokam2026-temporal-correlations)) are controlled by freezing weights; they remain an
  alternative for how the weights arose.
- **Numerical fixture.** The switched-affine positive construction (P3) shows existence in an affine setting only; no gated fixture.

## 3. Corrections to the inherited draft `floquet_analysis/main.tex`

Line numbers refer to the preserved file; the draft is not edited. Its empirical statements are `preliminary-attributed`.

| Location | Draft asserts | Why invalid or unsupported | Correct statement | Replacing item |
| --- | --- | --- | --- | --- |
| 160–210 (theorem 175, proof 183–210; 200–201, 209) | GCLSTM has a continuous-time limit $\dot s=F(s,A)-s$ whose solution interpolates the discrete sequence | The "limit $\Delta\tau\to0$" is taken after fixing $\Delta\tau=1$; the unit-time flow of $F-s$ is not $F$ (Remark 1.5), and sign-reversing or non-injective steps have no autonomous flow | Analyze the discrete recurrence directly; a continuous model needs its own embedding assumptions | Remark 1.5, Prop. 1.1 |
| 214, 217 | "Since the task is periodic, we assume there exists a [stable] periodic solution" | Periodic input does not imply a periodic complete-state orbit; the orbit may have period $kP$, be one of several, or not exist; closure must be verified at declared tolerances | Verify input periodicity and complete-state closure; otherwise use finite-time propagators | Props. 1.1–1.4 |
| 432–433 | $\lvert\lambda\rvert=1$ iff undamped oscillation; $\lvert\lambda\rvert<1$ iff exponential decay by $\lvert\lambda\rvert$ per period | Defective unit-modulus blocks grow polynomially; a multiplier $+1$ does not oscillate; $\lvert\lambda\rvert<1$ describes decay per period of a modal component only at the orbit, with possible within-period growth | Modulus one is neutral per period only for semisimple blocks; modulus below one is per-period modal contraction, not monotone decay | P8, Prop. 1.1 |
| 437–455 (447, 449–455) | AABB gives $M=J_B^2J_A^2$ and predicts four undamped multipliers $e^{2\pi im/4}$ | Repeated input does not make the Jacobian constant because the state differs between repeats; a one-step rotation at frequency $m/4$ raised to the period gives multiplier $1$, so fourth roots of unity are not predicted for the one-period map; the forced orbit needs no neutral mode | Use $J_3J_2J_1J_0$; substitute $J_B^2J_A^2$ only after verifying the Jacobian equalities | Props. 1.1, 1.3, P5 |
| 495–510 (504, 510) | Six multipliers $\lvert\lambda\rvert\approx0.41$–$0.44$ show each mode decays by $\approx0.42$ per cycle | No checkpoint, raw arrays, closure residuals, complete-state definition or code are recovered; top-$K$ eigenpairs are a partial spectrum | Treat as preliminary-attributed values pending reproduction under [graph reproduction](graph-reproduction.md) | Section 0 labels; Prop. 1.4 |
| 514–530 (530) | Envelope norms show strictly monotone within-cycle decrease "as the result of $\lvert\lambda\rvert<1$" | Multiplier modulus constrains the per-period map, not the within-period path; nonnormal factors allow transient growth | Report $\|P_{t,\ell}v\|$ per step and singular values; monotonicity is an observation, not a consequence | P8 |
| 563 | Low linearization error means the network "converge[s] to a linear dynamics" | A small linearization residual at a sampled state measures the local accuracy of the tangent map there; it does not make the global dynamics linear | Report linearization error as a tangent-validity diagnostic only | Prop. 1.4 |
| 568–676 (616, 642, 651, 666) | Theorems that GCLSTM hidden sequences have finite energy, bounded variation and step-wise continuity, licensing Fourier analysis | Every finite sequence has finite energy and bounded variation and a DFT; the statements are vacuous; the proof at 651 claims continuity on a compact set implies Lipschitz, which is false in general | The DFT of a finite sequence always exists; interpretation, not existence, is the issue | P5 |
| 679–716 (711–716) | A sharp PSD peak at $f=1/4$ is "a unit-modulus Floquet mode in disguise"; absence of peaks "confirms" no undamped clock | A contracting forced linear system has a line spectrum at the input harmonics with multiplier $a^T$ inside the unit circle; spectral peaks of a trajectory are not tangent modes; line 716 contradicts its own observation | Spectra describe the trajectory; multipliers describe perturbations; measure both and do not infer one from the other | P5, Prop. 1.3 |
| 730–747 (732–733, 747) | A closed PCA loop indicates a smooth periodic orbit; a zig-zag indicates discrete re-encoding; the analyses "demonstrate" re-encoding | A three-dimensional projection does not identify a mechanism or exclude a clock; time-pooled PCA over one episode is rank-limited (P6) | PCA is a visualization; mechanisms require conditioned cue clouds, tangent spectra and interventions | P6, Section 5 |
| 478, 485, 492, 583 (commented 353, 757) | State $(h,c)\in\mathbb R^{2H}$, Jacobians $2H\times2H$ | A graph LSTM carries node-wise hidden and cell states; the complete state is $2NH$ with node ordering, and a node-wise block is not the full transition | Define the complete graph state and its flattening before differentiating | Section 0 notation |
| 140 | Perfect accuracy implies an internal modulo-$\tau$ counter; absence of undamped modes implies re-encoding | Input-driven computation needs no autonomous clock: the input supplies the phase; P5 computes a periodic output with no neutral mode | Correct periodic prediction is compatible with contracting input-driven dynamics; retention must be tested with independently varied history | P5, Prop. 1.3 |
| 123–138; 77 | "Equivalence" of two task specifications; $x\in\mathbb R$ | The forward direction assumes contiguous equal-length blocks (an extra hypothesis) and both proof parts are labelled $(1)\Rightarrow(2)$; minor typo for $x\in\mathbb R^n$ | A periodic signal has a block form only under that hypothesis | — |

## 4. Mapping table and local verification record

| Proposition | Fixture / function | Test | Local status 2026-10-12 |
| --- | --- | --- | --- |
| 1.1–1.5 | `metrics.ordered_product`, `metrics.period_spectrum` (does not verify periodicity); `periodicity.tangent_window` labels products finite-time; no fixture for Remark 1.5 | `test_noncommuting_product_and_perturbation`; `test_full_lstm_derivative_jvp_and_finite_difference`; `test_checkpoint_only_artifacts_and_fail_closed_registry` | verified-infrastructure (pass) |
| P1 | `affine_ordering_null`: `affine_equal_centered_cue_cloud`, `affine_exact_cue_component`, `affine_offset_explains_linear_error`, `affine_offset_corrected_native_error`, `affine_offset_corrected_nonlinear_error`, `affine_native_linear_error_can_differ`, `affine_native_nonlinear_error_can_differ`, `pooled_body_conditioning_is_rejected`, `pooled_body_variance_contains_offset` | `test_affine_body_order_changes_offset_not_conditioned_cue`, `test_wrong_conditioning_is_rejected_and_creates_spurious_variance` | verified-infrastructure (pass); scratch 9/9 |
| P2 | `commuting_ordering_null`: `commuting_matched_counts_equal_product`, `commuting_equal_cue_cloud`, `commuting_process_noise_can_depend_on_order`; `propagated_noise_covariance` | `test_commuting_product_null_does_not_imply_process_noise_null` | verified-infrastructure (pass) |
| P3 | `noncommuting_construction`: `noncommuting_products_differ`, `noncommuting_conditioned_clouds_differ`, `noncommuting_terminal_noise_recovery_differs`, `noncommuting_explicit_chronology` | `test_noncommuting_positive_construction_has_order_specific_risk`, `test_noncommuting_chronological_product` | verified-infrastructure (pass); (b-i),(b-ii) scratch only |
| P4 | `continuous_neutral_memory`: `continuous_required_coordinates_retained`, `continuous_neutral_required_directions`, `continuous_nuisance_contraction`, `continuous_dimension_reduction_is_task_preserving` | `test_continuous_required_memory_survives_nuisance_compression` | verified-infrastructure (pass) |
| P5 | `calibrate`: `forced_orbit_periodic`, `forced_orbit_nonconstant`, `forced_orbit_strict_contraction`, `rotation_period_multiplier` | `test_calibration_controls` | verified-infrastructure (pass); scratch 5/5 |
| P6 | two-row rank in `conditioned_dimension` test; no repeated-orbit fixture | `test_wrong_conditioning_is_rejected_and_creates_spurious_variance` | partial; scratch 7/7 (orbit ceiling, cue ceiling, two histories, finite-sample bias) |
| P7 | `metrics.covariance_dimension`; `amplitude_only`: `isotropic_decay_preserves_pr`, `isotropic_variance_decays_exactly`; `calibrate`: `isotropic_dimension_four`, `rank_one_dimension_one`, `zero_variance_convention` | `test_dimension_translation_rotation_and_scale_invariance`, `test_known_spectrum`, `test_zero_variance_is_marked_degenerate`, `test_isotropic_attenuation_changes_amplitude_not_normalized_dimension` | verified-infrastructure (pass); scratch 5/5 |
| P8 | `calibrate`: `nonnormal_stable_eigenvalues`, `nonnormal_transient_gain`; `periodicity.tangent_window` spectra | `test_nonnormal_gain_is_distinct_from_eigenvalues` | verified-infrastructure (pass) |
| P9 | `linear_recovery_risk`; `fresh_noise_not_old_memory`: `fresh_noise_covariance_sustained`, `fresh_noise_linear_decoder_risk_matches`, `fresh_noise_linear_cue_recovery_deteriorates`, `fresh_noise_pr_remains_high` | `test_noise_location_and_whitening_in_linear_channel`, `test_fresh_noise_keeps_geometry_without_old_linear_memory` | verified-infrastructure (pass) |
| P10 | none yet | — | planned |
| P11 | `categorical_forced_basins`: `categorical_labels_retained`, `categorical_within_basin_convergence`, `categorical_closed_two_step_reference`, `categorical_reference_local_derivatives_zero`, `categorical_neighborhood_is_contracting`, `categorical_neighborhood_is_invariant`, `categorical_terminal_cycle_closure`, `categorical_basins_remain_separated` | `test_local_contracting_forced_basins_preserve_finite_memory` | verified-infrastructure (pass) |
| P12 | none (P3 affine existence only) | — | open |

**Commands run on 2026-10-12 (local Mac, no `uv sync`, nothing installed).**

```text
cd /Users/mehrabhamidi/Desktop/phd/paper/representation_collapse   # .venv/bin/python present: CPython 3.11.13, NumPy 2.4.6
PYTHONPATH="$PWD/src" .venv/bin/python -m unittest discover -s tests -v        -> Ran 34 tests in 3.280s  OK
   (an initial sandboxed attempt aborted with "OMP: Error #179"; rerun outside the sandbox; not a repository problem)
PYTHONPATH="$PWD/src" .venv/bin/python -m representation_collapse.calibrate --output /tmp/rc-theory/calib-measurement
                                                                               -> checks: 10, passed: true
PYTHONPATH="$PWD/src" .venv/bin/python -m representation_collapse.mechanism_calibration --output /tmp/rc-theory/calib-mechanism
                                                                               -> checks: 36, passed: true
.venv/bin/python /tmp/rc-theory/check_props.py                                 -> 29/29 scratch checks passed
```

The two calibration outputs were written under `/tmp`, not `runs/`, and the working tree stayed clean. These runs are
verified-infrastructure evidence about the fixtures; they are not findings about trained models. Note that the tracker's line "Full
source suite passes 29 tests" predates the five periodicity tests; the current count is 34.

**Scratch verification, not a repository artifact** (`/tmp/rc-theory/check_props.py`, independent NumPy implementation, seed
20261012; abridged output):

```text
P1 (random nonnormal A, rho~0.71, H=48, tanh encoder, 64 cues): centered clouds equal, max|diff| 1.78e-15; pairwise cue
   differences equal, 1.44e-15; cue component = A^H E(u) in both arms while ||k_per - k_con|| = 0.0420; FD cue Jacobian
   order-independent, max|dJ| 8.88e-10; PR 1.1242 = 1.1242 and variance 3.57e-13 equal; linear raw MSE 0.1055 / 0.1097
   equals bias^2; offset-aware decoder error 0 in both arms; tanh readout gain 0.9015 (per) vs 0.8979 (con)
P3a SO(3) pair: ||[Ra,Rb]|| 0.997, ||P1-P2|| 0.997, isotropic risks 0.11538 = 0.11538   P3b common eigenvector: ||Q1-Q2||
   0.0462, Q1 e1 = Q2 e1 = (0.3969, 0)   P3c repository pair: ||P1-P2|| 0.0744, linear risks 0.74065 / 0.74353
P5 (a=0.7, T=12): two initial conditions converge (difference 0.0); closed form s*_0 = -0.495352 matches; peak-to-peak
   3.096 with a^T = 1.38e-02; PSD peak at 1.00/T with the input's harmonic support
P6a exact period-12 orbit, 247 rows in 128 dims: centered rank 11 (PR 9.99)   P6b 6/100/512 cues in 50/128/128 dims:
   rank 5/99/128; 512 cues in 256 dims: rank 256   P6c two histories: rank 1, PR 1.000000   P6d isotropic 128-dim
   cloud, 512 samples: mean sample PR 102.24 (approximation 102.20)
P7a translation/rotation/isotropic scaling: PR 3.587628 = 3.587628, variance x13.69   P7b anisotropic: 3.5876 -> 1.0297
P7c isotropic attenuation: PR constant, variance 9.03e+01 -> 5.67e-05   P7d zero variance: PR nan   P7e equal eig(A) =
   (0.1, 0.9), covariance PR 1.3702 (normal) vs 1.0123 (nonnormal)
29/29 scratch checks passed
```

## 5. Predictions that distinguish hypotheses

Each item names a measurable quantity, the gated prediction (H1 variants, P12) beyond the affine and commuting nulls (P1–P2), and
what refutes it. Comparisons are at equal $R$, same cue sample, nuisance and final motif; the replication unit is the training seed.

1. **Cue dependence of the propagator.** Gated: $\partial P/\partial u\neq0$, so products at two cue histories differ; nulls: $P$ is
   cue-independent. Measurable with `derivative_cues=2` in the periodicity contract (the completed batch used one). Refuted if
   products at distinct cues agree within the product tolerance.
2. **Order dependence of the conditioned cue derivative.** Gated: $\partial s_H/\partial u$ differs between arms on some required
   coordinate; nulls: identical. Refuted if the two Jacobians agree within replay tolerance across panels and seeds.
3. **Finite-difference cue separation by arm.** Gated: $\|s_H(u)-s_H(u')\|$ (and its $h$/$c$ parts) differs by arm, identifying the
   deficient arm; H3 predicts the periodic arm's separation falling with $R$ together with closure residuals; nulls: equal. Refuted
   if separations agree or both fall identically (ordinary horizon forgetting).
4. **Singular values of $P$ on task directions.** Gated: $\sigma_i(\Phi_{\text{arm}}\,\partial s_0/\partial u)$ differs by arm; H1
   predicts the periodic arm's values below the noise floor at $R=16$ but not $R=2$ in a competent model; nulls: equal spectra.
   Refuted if spectra coincide or differ only off the task directions.
5. **Basin counts (categorical).** H1 via merging: fewer distinct terminal clusters per label set in the periodic arm; P11: equal
   counts with contracting within-basin variance; nulls: no clustering beyond the encoder's. Refuted if cluster counts and
   membership are stable across arms and $R$.
6. **Readout-offset signature.** Null-consistent: equal centered clouds, arm-dependent mean shift $k(h)$, raw-error difference
   removed by an offset-aware decoder; gated: a residual arm difference after offset correction. Refuted (for the gated account) if
   offset correction equalizes arms.
7. **Attenuation versus concentration (P7).** H3/attenuation: equal normalized PR, lower absolute variance in the periodic arm. H1
   concentration: lower PR at equal variance on task directions. Refuted by the opposite pattern.
8. **Precision dependence (P2 caveat).** Order-dependent propagated roundoff predicts FP32/FP64 replay differences that differ by
   arm even when cue products agree. Refuted if replay differences match across arms.
9. **Behavioral contrast.** H1: $\delta_{16}>\delta_2$ with $\delta_{16}>0$ in a model passing the $R_0/R_1/R_2$ competence gate in
   both arms; nulls and P11: $\delta_{16}\approx\delta_2\approx0$ or offset-only shifts. Refuted by an interval excluding the
   preregistered excess; positive DID with both $\delta$ negative is relative deterioration, not harm at a named horizon.

## 6. Open theoretical questions

1. **P12 sufficient condition.** Which gate-statistic bounds along an orbit imply $\kappa_{\text{per}}<1$ on task directions, and
   can a matched varying body be shown to keep $\kappa_{\text{con}}$ larger? Settled by a proof with explicit constants or a trained
   counterexample with verified basins.
2. **Formalization of ordering specificity.** Is "echo index one under the periodic driver, greater than one under the varying
   driver" the right formal notion, given that the control bodies are also locally repetitive? Settled by computing
   entrained-response counts for both drivers.
3. **Noise-floor horizon.** Does $R^*=\log(\Delta_0/\varepsilon)/\log(1/\kappa)$ with $\varepsilon$ the FP32 spacing predict where
   held-out probes lose the cue (the R2-versus-R16 probe pattern of the failed pilots is only descriptive)? Settled by FP64 replay
   with declared injected noise.
4. **Commuting architectures.** For a single diagonal selective state-space layer P2 predicts no cue-channel ordering effect; where
   does stacking or state-dependent selectivity break it? Settled by the architecture registry with per-layer transition audits.
5. **Graph state symmetry.** Does node permutation symmetry in a $2NH$ graph state create covariance dimension unrelated to memory,
   and how should the complete-state adapter fix node ordering? Settled during Gate D adapter validation.
6. **Categorical capacity.** How many robust basins can a 128- or 256-dimensional gated state hold at a given perturbation radius,
   against the $16^8$ required combinations? Settled by basin counting with calibrated perturbations on a competent model.
