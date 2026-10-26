# Adversarial memory and experimental novelty audit

Reviewed 2026-10-11 against `docs/October/2026-10-26/local/auto-research/periodicity-research-plan.md`.
This is a literature-grounded design review, not a trained-model result.
The accompanying [reference records](../../../../../../doc/literature/evidence/memory-references.json) contain 21 primary
references with explicit evidence levels. Six closest papers were read beyond
their abstracts: Pals, Ashwin/Ceni, Sharma, Fokam, Guan and Henaff. Selected
sections of the other full-text records were also inspected. `full_text` means
the relevant sections were read, not that every appendix was verified.
No PDFs were downloaded by this reviewer. No cluster work was performed.
Names follow verified primary-source spelling; published initials remain when
their expansion could not be verified. Null identifiers mean unverified or
unavailable, rather than a claim that no identifier exists.

## Decision

The broad story “repeated inputs contract recurrent representations and destroy
memory” is not defensible as a new general result. Existing theory and trained
network counterexamples already separate repetition, stable memory, useful
compression, interference and noise-resolved recoverability. The proposed
matched-driver experiment is a reasonable vehicle for a narrower contribution:
identify when **input/state-dependent propagation removes specified useful cue
information**, while controlling elapsed updates and a declared set of input
statistics. A benchmark string, architecture count or modality count cannot
substitute for that mechanism.

I found no directly matching experiment combining this motif intervention,
counterfactual cue histories, task-conditioned state geometry and causal state
controls. This bounded search does not establish priority. The exact claim
must survive full-text expansion, forward citations, and negative controls.

## Six closest comparisons: what is already established

**Pals et al. (2024), Results/Figs. 1–3 and Floquet methods:** cue identity selects
distinct stable phase-locked cycles in a forced RNN. Local attraction coexists
with retained categorical memory. This defeats “all multipliers below one
implies forgetting.” Their two-identity oscillator task does not implement our
higher-dimensional matched motif/query contrast.
[Primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/).

**Ashwin and Ceni (2023), §§2.1–2.3, PDF pp. 6–10, and §§3.1–3.3, pp. 11–13:**
symbol dwell constraints can change the multiplicity of attracting driven
responses. Basin and hyperbolicity assumptions matter; finite-horizon
clustering can misestimate multiplicity. Repetition-dependent loss of
initial-state distinctions is prior art. Task-relevant cue-channel impairment
under matched horizons remains a different experimental question.
[Primary preprint](https://arxiv.org/pdf/2309.04728).

**Sharma et al. (ICLR 2026), §§3–3.4, pp. 3–5, and §4.1, pp. 5–6:** delayed
recall is explained through age-indexed features, readout geometry, and
projection/composition interference. Forgetting obsolete features can be
useful. Their specified recall models and temporally independent-input loss
analysis are not our driver intervention. A generic geometry–memory tradeoff
or temporal-superposition explanation cannot be claimed anew.
[Author-hosted conference paper](https://aproca.github.io/assets/pdf/Temporal_superposition_in_RNNs.pdf).

**Fokam et al. (2026), Mathematical Setup, Findings and Appendix C step 6:**
correlated inputs alter learned memory costs and recurrent/feedthrough
tradeoffs. Their lag-one boundary result uses stationarity, feature whitening,
aligned modes and a geometric target profile. It is not a theorem that all
higher-order input statistics are irrelevant. Our fixed-weight cue-retention
study must state exactly how its assumptions differ.
[Primary HTML v1](https://arxiv.org/html/2609.00420v1).

**Guan et al. (2025), main paper pp. 1–2 and covariance/noise-spectrum analysis:**
memory reconstruction depends on noise covariance and injection structure,
not merely raw noise amplitude. Shared signal/noise injection and asymptotic
qualifications constrain their conclusions. Our measurements must specify
where noise enters and its covariance; an unqualified “noise-aware memory
channel” is not a novelty claim.
[Primary paper](https://arxiv.org/pdf/2409.03187).

**Henaff et al. (2016), §3.1, pp. 3–4:** delayed copying already has an explicit
orthogonal rotation/clock construction; adding requires a different solution.
Cue–delay–query architecture and recurrent clocks are therefore controls and
antecedents. The distinguishing proposed intervention is a structured body
with a matched varying-driver comparison, rather than a blank delay.
[Official conference paper](https://proceedings.mlr.press/v48/henaff16.pdf).

## Rejection arguments a reviewer can make

1. **The headline has published counterexamples.** Categorical memory may live
   in separate attracting basins. Contracting within a basin does not merge
   basins. Continuous smooth memory and finite categorical coding need separate
   propositions and experiments. The plan correctly distinguishes these, but
   the paper title and abstract must too.
2. **Dimension alone does not measure useful information.** Noise-relative
   memory was already formalized by
   [Ganguli et al. (2008)](https://pubmed.ncbi.nlm.nih.gov/19020074/);
   sparse reconstructible memory by
   [Ganguli and Sompolinsky (2010)](https://proceedings.neurips.cc/paper/2010/hash/0f2c9a93eea6f38fabb3acb1c31488c6-Abstract.html);
   target-function capacity by
   [Dambre et al. (2012)](https://www.nature.com/articles/srep00514).
   Conditional covariance PR and task-channel quantities can disagree. Showing
   disagreement validates an interpretation; it does not invent the distinction.
3. **The intervention changes higher-order statistics.** The motif library
   matches declared counts, cyclic bigrams, endpoints and horizon, but changes
   long-lag predictability, higher-order subsequences and arm detectability.
   Equal symbol counts are not equal temporal covariance for every renderer or
   encoder. Describe a specific higher-order ordering intervention, not a
   universal effect of repetition independent of all input statistics.
4. **The native readout may fail while stored information survives.**
   [Hewitt and Liang (2019)](https://aclanthology.org/D19-1275/) and
   [Ravichander et al. (2021)](https://aclanthology.org/2021.eacl-main.295/)
   motivate probe controls and separation of decodability from task use.
   Conversely, failure of one fitted probe does not establish information loss.
   Compare native readout, held-out controlled probes, finite cue perturbations,
   and causal state interventions before describing forgetting.
5. **Training failure or horizon shift explains the effect.** Classical
   [gradient difficulty](https://proceedings.mlr.press/v28/pascanu13.html)
   and modern
   [linear learning dynamics](https://proceedings.mlr.press/v267/proca25a.html)
   require separating acquired competence from fixed-weight inference.
   R=2 versus R=16 is within the mixed training support; R=32/64 is an explicit
   extrapolation analysis. An OOD-only failure cannot establish the primary
   inference mechanism.
6. **An attractive prior title uses a different causal setting.**
   [Howlett (2025)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1601641/full)
   changes Hebbian weights during repeated presentations. Cite it as a
   plasticity/basin antecedent and distinguish frozen-weight inference.
7. **Precision assumptions have been hidden.** The
   [finite-precision language-recognition literature](https://aclanthology.org/P18-2117/)
   and [Korsky/Berwick](https://arxiv.org/abs/1906.06349) warn against mixing
   unbounded real arithmetic with bounded hardware state. The differentiable
   d-dimensional exact-memory argument is not a finite categorical storage
   theorem. FP32/bfloat16 robustness must be measured with declared arithmetic,
   ranges and noise; these abstract-only records do not settle formal bounds.

## Exact affine null: a necessary new design check

This is an independent derivation, not a result attributed to a source above.
For a fixed affine body update

\[
s_{t+1}=A s_t+B x_t+b,\qquad s_0=E(m),
\]

after H updates with body history h,

\[
s_H(m,h)=A^H E(m)+k(h).
\]

Holding h fixed and comparing two cue vectors eliminates the body offset:

\[
\Delta s_H=A^H[E(m)-E(m')].
\]

Thus, at matched H, cue differences, centered cue-conditioned covariance and
the cue-to-state derivative are identical in both arms. This holds for any
fixed encoder E, not just a linear one. Additive process noise with the same joint sequence law, independent of cue
and body ordering (for example common iid noise), also yields the same
state-noise covariance. Equal per-step marginals alone are insufficient.
With a fixed linear readout C, the channel is C A^H D E and is arm-independent.

A nonlinear native readout/query update can nevertheless give different
errors because k(h) changes its operating point. That is a readout-offset
mechanism, not destruction of the conditioned cue representation. Conditional
centering does not remove this offset from the model's actual query operation.

Consequences:

- Spectral contraction by itself predicts horizon-dependent forgetting in this
  null, but zero periodic-versus-control cue-channel contrast at a common H.
- Any periodic-arm covariance difference in this implementation reveals a
  sampling/centering bug, unequal H, state/noise mismatch, or a violated model
  assumption.
- A mechanism-specific positive result needs input/state-dependent propagation
  or a separately identified readout effect. A switched affine model with
  noncommuting input-selected matrices is a minimal analytical extension.

## Genuine unresolved questions, ordered by decision value

1. **Does higher-order ordering impair useful memory in already competent,
   fixed-weight gated networks, at matched elapsed updates?** Predeclare the
   within-support error difference-in-differences. A valid null result is a
   valuable boundary condition and should stop causal overinterpretation.
2. **Where does the effect occur?** Distinguish cue encoding G, body propagation
   Φ, query/readout C, state-dependent noise, and attractor switching. An error
   increase alongside unchanged recoverability favors readout failure;
   diminished noise-resolved cue sensitivity plus controlled rescue supports a
   stronger storage-loss interpretation.
3. **Does categorical robustness come from multistability while continuous
   memory needs marginal directions or finite-horizon protection?** Compare
   finite perturbations and basin identity against local derivatives. Failure
   of infinitesimal sensitivity around a categorical attractor can be expected,
   even when identity is perfectly recoverable.
4. **Does useful forgetting differ from harmful forgetting?** Vary queried
   versus irrelevant tags without changing the driver's statistics. Measure
   each separately; compressing unqueried information is not collapse of the
   task-demanded representation.
5. **Which effects transfer across renderings after mechanism verification?**
   Keep the broad architecture/modalities plan as staged coverage. Renderer
   artifacts, encoder nonlinearities and different state arithmetic can create
   apparent cross-modal differences without different recurrent mechanisms.
6. **What changes under persistent input access in depth recurrence?** Separate
   the state-carried regime from repeated access to the original cues. Recovery
   from retained input is not autonomous memory preservation in the recurrent
   state. One whole-motif transformer loop also changes the forcing unit.

## Minimum decisive experiment before the full screening matrix

This is a recommended gate, not a replacement for the owner's required family
coverage. Use the vector renderer and GRU/LSTM first, with the full-state LSTM
adapter. Keep the 16 independent write tags and eight queried tags; test both
16-level categorical and continuous values. Begin at q=1 and R=2/16, with equal
mixed-arm training and equal valid readout/training selection budgets. Three
pilot seeds diagnose implementation and effect scale; a preregistered
confirmation uses fresh seeds and paired evaluation histories. The current
plan's ten confirmation seeds are a reasonable fixed starting rule, not a
guarantee of power.

Before trained-model interpretation, pass four analytical controls:

- The affine null above, including a deliberately nonlinear readout that can
  change behavior through a body offset without changing cue covariance.
- Stable, separated categorical memory attractors under a periodic driver.
- A continuous memory manifold with neutral retained directions and a decaying
  orthogonal nuisance subspace.
- Isotropic contraction with unchanged normalized PR, and a readout rotation
  that separates state storage from access.

For each fixed body history, sample independent cue counterfactuals and center
within that history. Never pool different nonperiodic body histories before
centering. Pair arms on cue vector, queried coordinate, final motif, q and H;
report inference over independent training seeds, not thousands of correlated
state samples as if they were training replicates. Diagnostic intervals over
histories are distinct from seed-level uncertainty. Freeze driver panels and
keep fresh cue histories for probing.

Measure native query error, within-history cue covariance PR and amplitude,
held-out tag probes with randomized-label controls, finite cue perturbations,
complete-state swaps, and a specified noise-resolved cue-to-output channel.
For categorical values, include finite differences between categories instead
of relying only on derivatives. State swaps should occur at an identical driver
phase before the query, and transfer all persistent state. Whole-state swaps
show the state's causal relevance, not that a PR change mediates the effect.
A claimed direction-specific mechanism requires interventions restricted to
the proposed directions plus matched nuisance/energy controls.

Acceptance for a memory-loss claim requires all of: competent within-support
baseline, paired periodic-specific behavioral impairment, diminished cue
recoverability or a clearly specified task-channel effect, and intervention
evidence locating the relevant failure. If only PR falls, report geometry. If
only native error rises while tag information survives, report access/readout
failure. If both arms deteriorate equally, report horizon-limited memory. If
the categorical model preserves tags despite contraction, report a positive
counterexample to the proposed generic mechanism.

Period and dwell are coupled in this q-only pilot. The q=2 follow-up tests a
changed dwell/period condition, not an isolated period effect. Claim separation
of period, dwell and frequency only after independently varying motif length,
symbol-run lengths and physical sample interval with specified controls.

## Search log and coverage limits

Searches used primary conference proceedings, publishers, author-hosted papers,
arXiv and official indexing records; secondary summaries were not treated as
paper evidence. The recent-work check included the following actual query
families, followed by direct full-text retrieval of the closest results:

- `site.arxiv.org "2609.00420"`
- `"Temporal Superposition" "2026" recurrent`
- `site.arxiv.org "Temporal superposition and feature geometry"`
- `Pals Macke Barak 2024 periodic working memory oscillations recurrent neural networks`
- `Ashwin Ceni 2023 repetition recurrent memory attractors`
- `"Sudden restructuring of memory representations" recurrent authors`
- `"How noise affects memory in linear recurrent networks" authors`
- `"Information Processing Capacity of Dynamical Systems" Dambre`
- `"Learning long-term dependencies with gradient descent is difficult" Bengio Simard Frasconi DOI`
- `"Long Short-Term Memory" Hochreiter Schmidhuber 1997 DOI`
- `"On the difficulty of training recurrent neural networks" Pascanu 2013 PMLR`
- `"Unitary Evolution Recurrent Neural Networks" Arjovsky Shah Bengio PMLR`
- `"Opening the Black Box" "Slow Points" Sussillo Barak 2013`
- `"Recurrent Orthogonal Networks and Long-Memory Tasks" Henaff Szlam LeCun`
- `Orhan Ma 2019 "working memory" "recurrent neural networks" representations`
- `"A diverse range of factors" Orhan Ma arxiv`
- `site.proceedings.mlr.press "Learning dynamics in linear recurrent neural networks"`
- `site.arxiv.org "Learning dynamics in linear recurrent" Proca`
- `site.aclanthology.org "Designing and Interpreting Probes with Control Tasks"`
- `site.aclanthology.org "Probing the Probing Paradigm"`
- `site.aclanthology.org "On the Practical Computational Power of Finite Precision"`
- `site.arxiv.org "On the Computational Power of RNNs" Korsky`

The query `"Simple recurrent networks are not obsolete" Orhan Ma` did not yield
a verified matching title and was excluded. Initial Ganguli PDF retrieval was
followed by timeouts; its record remains abstract-level, rather than implying
an audited proof. Publisher barriers similarly limited some older records.
The 2026 correlation work is a preprint, not a verified peer-reviewed result.
This review does not fully cover pretrained state-space/native-modal models,
all causal-ablation methods, biological repetition suppression, or every
post-2025 follow-up. Those are next audit scopes, not evidence of an empty field.
