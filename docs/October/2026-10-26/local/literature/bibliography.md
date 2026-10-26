# Annotated bibliography

Search cutoff: 2026-10-11. This is a scoped research bibliography, not a proof of exhaustive coverage.

Evidence levels describe what was inspected for this audit; downloading a PDF does not upgrade its evidence level.
Publication years and venues follow the verified source; preprint and publication dates can differ. Null DOI fields are intentional.

See [novelty review](novelty-review.md), [search method](search-log.md), [BibTeX](../../../../../doc/literature/references.bib), and [download manifest](../../../../../doc/literature/download-manifest.json).

<a id="floquet1883"></a>
## Sur les équations différentielles linéaires à coefficients périodiques

**floquet1883** — Gaston Floquet. 1883. Annales scientifiques de l’École Normale Supérieure.

[Primary source](https://www.numdam.org/item/?id=ASENS_1883_2_12__47_0) · [DOI](https://doi.org/10.24033/asens.220)

**Themes:** Floquet, foundations. **Evidence:** metadata; Numdam bibliographic record..

**Relevance:** Original continuous-time periodic linear-system foundation.

**Novelty boundary:** Floquet analysis itself is classical; applying a period map needs a distinct scientific question.

**Limits:** Historical source; the project uses discrete updates, and no claim is made to have audited this French proof.

**Local PDF:** [floquet1883.pdf](../../../../../literature/papers/floquet1883.pdf) — 43 pages; SHA-256 `c86ff33e6217f13e9e0ce513070bbdba5b7d77ee7e38030bd9c0beab4e2f230c`.

<a id="bengio1994-long-term-dependencies"></a>
## Learning long-term dependencies with gradient descent is difficult

**bengio1994-long-term-dependencies** — Yoshua Bengio; Patrice Simard; Paolo Frasconi. 1994. IEEE Transactions on Neural Networks 5(2):157–166.

[Primary source](https://ieeexplore.ieee.org/document/279181) · [DOI](https://doi.org/10.1109/72.279181)

**Themes:** vanishing gradients, long-term dependencies. **Evidence:** abstract; Original-paper first-page search extract and indexed abstract; publisher access blocked..

**Relevance:** Classical explanation of why learning distant dependencies is difficult.

**Novelty boundary:** Long-horizon failure and gradient decay are established phenomena, not sufficient novelty.

**Limits:** Optimization difficulty is not a theorem that a competent fixed-weight network must forget under repeated inference.

**Copy/version note:** Institutional teaching copy of the author manuscript (title page says to appear); publication metadata refers to the final 1994 article.

**Local PDF:** [bengio1994-long-term-dependencies.pdf](../../../../../literature/papers/bengio1994-long-term-dependencies.pdf) — 35 pages; SHA-256 `bab3b7f441358173e55a307ed3c7471d9f6c9344d14afed0ca651cf153a6968b`.

<a id="seung1996"></a>
## How the brain keeps the eyes still

**seung1996** — H. Sebastian Seung. 1996. Proceedings of the National Academy of Sciences 93(23): 13339-13344.

[Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC24094/) · [DOI](https://doi.org/10.1073/pnas.93.23.13339)

**Themes:** continuous memory, line attractor, neutral direction. **Evidence:** full_text; Author-paper scan, pp. 13342-13343: Eq. (4)-(5), linear tuning/eigenvalue discussion, nonlinear approximation, and explicit line-versus-point-attractor contrast..

**Relevance:** Treats analog persistent memory as an attracting line of equilibria, explains tuned feedback and slow drift, and contrasts discrete noise-resistant point attractors.

**Novelty boundary:** Continuous-memory neutral directions and the categorical/continuous distinction are classical foundations, not candidate novelty.

**Limits:** Exact manifold neutrality does not guarantee robustness to noise or mistuning. Expanded author name retains published H. initial; no unverified expansion.

**Local PDF:** [seung1996.pdf](../../../../../literature/papers/seung1996.pdf) — 7 pages; SHA-256 `b3937d74b62abfdcdf2dea3b19d04202b7a2f12289391d029c666af2d013e92e`.

<a id="hochreiter1997-lstm"></a>
## Long Short-Term Memory

**hochreiter1997-lstm** — Sepp Hochreiter; Jürgen Schmidhuber. 1997. Neural Computation 9(8):1735–1780.

[Primary source](https://doi.org/10.1162/neco.1997.9.8.1735) · [DOI](https://doi.org/10.1162/neco.1997.9.8.1735)

**Themes:** gated memory, gradient preservation, delayed recall. **Evidence:** full_text; Author-hosted PDF, pp. 1–2, introduction and architectural motivation; relevant introductory sections inspected..

**Relevance:** Introduces gated access to a persistent internal memory path and long-delay artificial tasks.

**Novelty boundary:** Gating, persistent state, and delayed cue retention are foundational prior art.

**Limits:** Original architecture and learning algorithm differ from modern forget-gate LSTMs; retention depends on the learned complete state.

**Local PDF:** [hochreiter1997-lstm.pdf](../../../../../literature/papers/hochreiter1997-lstm.pdf) — 32 pages; SHA-256 `ceb9e53dbc0493f5b3bf5520ed940f3e6b526064d17b2118d77e51f79c0edcc6`.

<a id="jaeger2001"></a>
## The echo state approach to analysing and training recurrent neural networks - with an Erratum note

**jaeger2001** — Herbert Jaeger. 2001. GMD Report 148; corrected author version dated 2010.

[Primary source](https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf)

**Themes:** echo state, initial-state forgetting, periodic generation. **Evidence:** metadata; Corrected PDF title page/erratum notice; Introduction and Definition 1, pp. 3-5..

**Relevance:** Introduces echo states as functions of left-infinite input history and readout training; includes periodic generators and multiple-switch examples.

**Novelty boundary:** Initial-state forgetting while retaining input-history information is foundational, so these must not be conflated.

**Limits:** 2010 author version corrects a flaw in Definition 3 and Proposition 1 of the 2001 report. Original theorem equivalences were not independently audited.

**Local PDF:** [jaeger2001.pdf](../../../../../literature/papers/jaeger2001.pdf) — 47 pages; SHA-256 `0a0296f586f86a0e49e651e4581ce2aef3a926ca9cb3d57653c8963b7b83e625`.

<a id="jaeger2002"></a>
## Short term memory in echo state networks

**jaeger2002** — Herbert Jaeger. 2002. GMD Report 152.

[Primary source](https://www.ai.rug.nl/minds/pubs/)

**Themes:** linear readout, memory capacity. **Evidence:** metadata; Author-uploaded report title page: May 28, 2002; footnote: GMD Report 152, publication March 28, 2002. Author publication index labels the item 2001..

**Relevance:** Foundational short-term-memory capacity reference for reconstruction of past inputs from recurrent states.

**Novelty boundary:** A memory-capacity versus state-dimension statement needs this antecedent and its task/readout assumptions.

**Limits:** Year conflict in author index retained explicitly; 2002 follows report title-page publication note. Capacity theorem not read in this audit; Fraunhofer PDF endpoint timed out.

**Local PDF:** [jaeger2002.pdf](../../../../../literature/papers/jaeger2002.pdf) — 60 pages; SHA-256 `441da98b9e1c6e9b9f67f86bd74df63f2ed8c00f258f24a469812bd1d0880feb`.

<a id="white2004"></a>
## Short-Term Memory in Orthogonal Neural Networks

**white2004** — Olivia L. White; Daniel D. Lee; Haim Sompolinsky. 2004. Physical Review Letters 92: 148102.

[Primary source](https://arxiv.org/abs/cond-mat/0402452) · [DOI](https://doi.org/10.1103/PhysRevLett.92.148102)

**Themes:** orthogonal recurrence, sequence memory, capacity. **Evidence:** abstract; arXiv abstract, journal-reference and DOI fields..

**Relevance:** Studies retrieval of temporal sequences from the instantaneous state in linear discrete networks; orthogonal and distributed shift-register capacity scales with size.

**Novelty boundary:** Preservation controls and claims connecting effective dimension to memory must compare established orthogonal-memory results.

**Limits:** Specific linear/noise/capacity normalization assumptions not audited from full text.

**Local PDF:** [white2004.pdf](../../../../../literature/papers/white2004.pdf) — 4 pages; SHA-256 `45600bf7d57b273e92d05163ec41eaa43704bd6d91aae11a090cac4890102c4c`.

<a id="roy2007effective"></a>
## The Effective Rank: a Measure of Effective Dimensionality

**roy2007effective** — Olivier Roy; Martin Vetterli. 2007. 15th European Signal Processing Conference (EUSIPCO).

[Primary source](https://zenodo.org/records/40328) · [DOI](https://doi.org/10.5281/zenodo.40328)

**Themes:** measurement, effective rank. **Evidence:** metadata; Zenodo proceedings record..

**Relevance:** Entropy-based effective rank provides a historical measurement reference.

**Novelty boundary:** Normalized spectral dimension is an established diagnostic, not an information-retention theorem.

**Limits:** Repository DOI; distinguish normalized singular-value entropy from covariance-eigenvalue entropy.

**Local PDF:** [roy2007effective.pdf](../../../../../literature/papers/roy2007effective.pdf) — 5 pages; SHA-256 `6c8c6a03c61e7fbbcb291ec5737299313af1d259563a0ea2352b738427d289ab`.

<a id="ganguli2008-memory-traces"></a>
## Memory traces in dynamical systems

**ganguli2008-memory-traces** — Surya Ganguli; Dongsung Huh; Haim Sompolinsky. 2008. Proceedings of the National Academy of Sciences 105(48):18970–18975.

[Primary source](https://www.pnas.org/doi/10.1073/pnas.0804451105) · [DOI](https://doi.org/10.1073/pnas.0804451105)

**Themes:** Fisher memory, noise-resolved memory, nonnormal dynamics, noise. **Evidence:** full_text; Author PDF pp. 18970-18971: Fisher Memory Matrix, Eq. (1)-(5); normal-network memory curve and sum rule..

**Relevance:** Measures retained input information relative to noise with a Fisher memory curve.

**Novelty boundary:** Noise-aware dynamical memory measures and normal/nonnormal distinctions long predate participation-ratio comparisons.

**Limits:** Local input sensitivity and explicitly specified dynamical noise; bounds depend on input/noise/weight normalization. Does not equate covariance rank with nonlinear global memory.

**Local PDF:** [ganguli2008-memory-traces.pdf](../../../../../literature/papers/ganguli2008-memory-traces.pdf) — 6 pages; SHA-256 `3c2f1e87df2c6699ae70d765652482d63b2444e6dcebf1648660a2cd480c6e6e`.

<a id="scarselli2009gnn"></a>
## The Graph Neural Network Model

**scarselli2009gnn** — Franco Scarselli; Marco Gori; Ah Chung Tsoi; Markus Hagenbuchner; Gabriele Monfardini. 2009. IEEE Transactions on Neural Networks.

[Primary source](https://doi.org/10.1109/TNN.2008.2005605) · [DOI](https://doi.org/10.1109/TNN.2008.2005605)

**Themes:** graph, contraction, foundations. **Evidence:** abstract; Publisher-indexed abstract and Crossref metadata..

**Relevance:** Early graph-state recurrence and contraction-based computation.

**Novelty boundary:** Treating GNN computation as recurrent state dynamics is foundational prior art.

**Limits:** Historical graph-state architecture, not a periodic temporal-graph cue-retention experiment.

**Local PDF:** [scarselli2009gnn.pdf](../../../../../literature/papers/scarselli2009gnn.pdf) — 20 pages; SHA-256 `85cb7e0388177b51dd024dad04399e65d4ee35096bcc66568276870ef6b1e0a9`.

<a id="ganguli2010-compressed-memory"></a>
## Short-term memory in neuronal networks through dynamical compressed sensing

**ganguli2010-compressed-memory** — Surya Ganguli; Haim Sompolinsky. 2010. Advances in Neural Information Processing Systems 23.

[Primary source](https://proceedings.neurips.cc/paper/2010/hash/0f2c9a93eea6f38fabb3acb1c31488c6-Abstract.html)

**Themes:** compressed sensing, sparse memory, orthogonal recurrence. **Evidence:** abstract; Official NeurIPS abstract, including sparsity and orthogonal-network qualifications..

**Relevance:** Sparse signal structure allows memory reconstruction beyond a naive neuron-count bound.

**Novelty boundary:** Dimension reduction can preserve reconstructible structured memory; storage dimension is not an unconditional capacity proxy.

**Limits:** Sparse-sequence reconstruction uses assumptions different from smoothly storing independent continuous tags.

**Local PDF:** [ganguli2010-compressed-memory.pdf](../../../../../literature/papers/ganguli2010-compressed-memory.pdf) — 9 pages; SHA-256 `14f7c6541a4f8436e408716d2edb609cd6f20c6beebddb00d73ecaabfbb7ad96`.

<a id="rajan2010"></a>
## Stimulus-dependent suppression of chaos in recurrent neural networks

**rajan2010** — Kanaka Rajan; Larry F. Abbott; Haim Sompolinsky. 2010. Physical Review E 82: 011903.

[Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10683875/) · [DOI](https://doi.org/10.1103/PhysRevE.82.011903)

**Themes:** periodic forcing, chaos suppression, frequency response. **Evidence:** abstract; Abstract; bibliographic header; Eq. (1)-(3) and Fig. 1 inspected to verify model/driver definition..

**Relevance:** Mean-field theory links oscillatory drive amplitude and frequency to suppression of chaotic fluctuations and phase locking.

**Novelty boundary:** Input-induced stabilization and frequency-selective chaos suppression precede a generic Floquet-collapse claim.

**Limits:** Random continuous-time rate circuits; no required-history recall endpoint. Deep mean-field derivation not reviewed. Abbott's expanded first name verified from Columbia.

**Local PDF:** [rajan2010.pdf](../../../../../literature/papers/rajan2010.pdf) — 5 pages; SHA-256 `3054fad04b97e9a71178489c6555bd82eaf82dabc52c3e5b670d4a1efa1be672`.

<a id="halko2011randomized"></a>
## Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions

**halko2011randomized** — Halko, Nathan; Martinsson, Per-Gunnar; Tropp, Joel A.. 2011. SIAM Review.

[Primary source](https://arxiv.org/abs/0909.4061)

**Themes:** numerical methods. **Evidence:** abstract; Primary arXiv abstract and Caltech publication record.

**Relevance:** Randomized methods approximate dominant matrix decompositions.

**Novelty boundary:** Matrix-free spectral estimation is an implementation tool, not scientific novelty here.

**Limits:** Partial decompositions require residual and tail checks.

**Local PDF:** [halko2011randomized.pdf](../../../../../literature/papers/halko2011randomized.pdf) — 74 pages; SHA-256 `d5d99d95b3d9657de6413f4c26d635238103ef9cf1a28ff1b7fde531cf11f6be`.

<a id="burak2012"></a>
## Fundamental limits on persistent activity in networks of noisy neurons

**burak2012** — Yoram Burak; Ila R. Fiete. 2012. Proceedings of the National Academy of Sciences 109(43): 17645-17650.

[Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC3491496/) · [DOI](https://doi.org/10.1073/pnas.1117386109)

**Themes:** continuous attractor, noise diffusion, Fisher information. **Evidence:** abstract; Abstract; Results model assumptions and Diffusive Drift due to Ongoing Neural Noise inspected..

**Relevance:** Bounds noise-induced diffusion of continuously stored memory using an information-diffusion relation.

**Novelty boundary:** Neutral tangent directions alone cannot support claims of robust memory; continuous-attractor noise limits are prior theory.

**Limits:** Poisson spiking/weak-fluctuation approximation and tuned-away systematic drift; supporting derivations not deeply audited.

**Local PDF:** [burak2012.pdf](../../../../../literature/papers/burak2012.pdf) — 6 pages; SHA-256 `6ea4f05bf1d0c8870baf8f042e9fc4392c1806e9100d94d2b0d3ab35f345adea`.

<a id="dambre2012-information-capacity"></a>
## Information Processing Capacity of Dynamical Systems

**dambre2012-information-capacity** — Joni Dambre; David Verstraeten; Benjamin Schrauwen; Serge Massar. 2012. Scientific Reports 2:514.

[Primary source](https://www.nature.com/articles/srep00514) · [DOI](https://doi.org/10.1038/srep00514)

**Themes:** information processing capacity, linear readouts, memory-computation tradeoff, orthogonal tasks, linear readout. **Evidence:** full_text; PMC full text, definitions of capacity, orthogonal target functions, and maximum-capacity discussion..

**Relevance:** Defines target-function capacities and a qualified total-capacity relation to independent state variables.

**Novelty boundary:** Task-aligned capacity, state dimension, and computation/memory tradeoffs have an established quantitative framework.

**Limits:** Bounds rely on target/input assumptions and linear readouts; covariance participation ratio is not their capacity.

**Local PDF:** [dambre2012-information-capacity.pdf](../../../../../literature/papers/dambre2012-information-capacity.pdf) — 7 pages; SHA-256 `2136b7f85e908fa38f570088a7530ab6cc9b24d8e5c4ae106ff3c6899c6ff1a6`.

<a id="manjunath2013"></a>
## Echo State Property Linked to an Input: Exploring a Fundamental Characteristic of Recurrent Neural Networks

**manjunath2013** — Gandhi Manjunath; Herbert Jaeger. 2013. Neural Computation 25(3): 671-696.

[Primary source](https://www.ai.rug.nl/minds/uploads/ESP_NC.pdf) · [DOI](https://doi.org/10.1162/NECO_a_00411)

**Themes:** input-specific synchronization, periodic response, echo state. **Evidence:** full_text; Definition 2.2, author-PDF p. 5; Lemma 2.2, pp. 7-8; Proposition 2.3 and proof, p. 9; Section 2.1 cautions on folklore..

**Relevance:** Input-specific ESP distinguishes dependence on input history from arbitrary initial conditions. Uniform shrinking bounds imply uniqueness; a periodic driver with ESP has a response whose period divides the input period.

**Novelty boundary:** The proposed unique forced response and history-washout foundations are standard input-dependent synchronization theory.

**Limits:** Compact state space and entire-solution definitions. Contraction bound is sufficient; negative local exponents alone do not establish global uniqueness. First name expanded using university author record.

**Local PDF:** [manjunath2013.pdf](../../../../../literature/papers/manjunath2013.pdf) — 27 pages; SHA-256 `bcf2d219e1e4e9e4ade7b55138668d0c3ae037d753af301d689d2c56f81fdc3e`.

<a id="pascanu2013-training-rnns"></a>
## On the difficulty of training recurrent neural networks

**pascanu2013-training-rnns** — Razvan Pascanu; Tomas Mikolov; Yoshua Bengio. 2013. ICML; Proceedings of Machine Learning Research 28(3):1310–1318.

[Primary source](https://proceedings.mlr.press/v28/pascanu13.html)

**Themes:** exploding gradients, vanishing gradients, training confounds. **Evidence:** abstract; Official PMLR abstract and publication metadata..

**Relevance:** Separates numerical and optimization obstacles associated with recurrent gradient propagation.

**Novelty boundary:** A collapse experiment must distinguish failed learning from inference-time loss in a trained competent model.

**Limits:** Gradient clipping and optimization findings do not establish periodic-driver-specific memory loss.

**Local PDF:** [pascanu2013-training-rnns.pdf](../../../../../literature/papers/pascanu2013-training-rnns.pdf) — 9 pages; SHA-256 `d786c18c0e1ed58aee30ce78b5163a9f771978c3146b68737528e1cda6ac1e74`.

<a id="sussillo2013-slow-points"></a>
## Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks

**sussillo2013-slow-points** — David Sussillo; Omri Barak. 2013. Neural Computation 25(3):626–649.

[Primary source](https://doi.org/10.1162/NECO_a_00409) · [DOI](https://doi.org/10.1162/NECO_a_00409)

**Themes:** fixed points, slow points, low-dimensional computation. **Evidence:** abstract; Author-institution publication abstract and bibliographic record..

**Relevance:** Uses fixed and slow points to explain computations in trained high-dimensional networks.

**Novelty boundary:** Low-dimensional dynamics can implement useful memory and computation; low geometry is not inherently pathological.

**Limits:** Only abstract inspected here; detailed dynamical analyses require full-text verification before reuse.

**Local PDF:** [sussillo2013-slow-points.pdf](../../../../../literature/papers/sussillo2013-slow-points.pdf) — 24 pages; SHA-256 `e31f67fcac51c35dad221b79460956eb485cac8173093f2c89ac450c9874f599`.

<a id="shi2015convlstm"></a>
## Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

**shi2015convlstm** — Shi, Xingjian; Chen, Zhourong; Wang, Hao; Yeung, Dit-Yan; Wong, Wai-kin; Woo, Wang-chun. 2015. NeurIPS 2015.

[Primary source](https://arxiv.org/abs/1506.04214)

**Themes:** vision architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Convolutional input and recurrent transitions support spatiotemporal forecasting.

**Novelty boundary:** Established modality-native recurrent baseline.

**Limits:** Baseline reference; no claim of demonstrated repetition-induced collapse.

**Local PDF:** [shi2015convlstm.pdf](../../../../../literature/papers/shi2015convlstm.pdf) — 12 pages; SHA-256 `6b83643820fa049a9cdf30621385bd5d33c67408d348a93edb6c59ed247986ef`.

<a id="arjovsky2016-unitary-rnn"></a>
## Unitary Evolution Recurrent Neural Networks

**arjovsky2016-unitary-rnn** — Martin Arjovsky; Amar Shah; Yoshua Bengio. 2016. ICML; Proceedings of Machine Learning Research 48.

[Primary source](https://proceedings.mlr.press/v48/arjovsky16.html)

**Themes:** unitary recurrence, gradient preservation, long memory. **Evidence:** abstract; Official PMLR abstract and arXiv metadata..

**Relevance:** A principled long-memory architecture control based on unitary recurrent weights.

**Novelty boundary:** Orthogonal/unitary memory preservation is established; including such models is a control, not a mechanism claim.

**Limits:** A unitary weight matrix alone does not make the entire nonlinear recurrent-state Jacobian unitary.

**Local PDF:** [arjovsky2016-unitary-rnn.pdf](../../../../../literature/papers/arjovsky2016-unitary-rnn.pdf) — 9 pages; SHA-256 `9ec5bf8f3f7fd0ffbd8c220b8263bd2da27608edf6075d545376b9a6e04952a8`.

<a id="ballas2016convgru"></a>
## Delving Deeper into Convolutional Networks for Learning Video Representations

**ballas2016convgru** — Ballas, Nicolas; Yao, Li; Pal, Chris; Courville, Aaron. 2016. ICLR 2016.

[Primary source](https://arxiv.org/abs/1511.06432)

**Themes:** vision architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Convolutional GRU feature processing for video representations.

**Novelty boundary:** Established convolutional gated recurrence.

**Limits:** Visual encoder and spatial state create additional comparison factors.

**Local PDF:** [ballas2016convgru.pdf](../../../../../literature/papers/ballas2016convgru.pdf) — 11 pages; SHA-256 `6462761ca9eb929b837869e2196d21c842d695d93ecfe9a8eca00bed4d251e2a`.

<a id="henaff2016-orthogonal-memory"></a>
## Recurrent Orthogonal Networks and Long-Memory Tasks

**henaff2016-orthogonal-memory** — Mikael Henaff; Arthur Szlam; Yann LeCun. 2016. ICML; Proceedings of Machine Learning Research 48:2034–2042.

[Primary source](https://proceedings.mlr.press/v48/henaff16.html)

**Themes:** copy task, orthogonal recurrence, clock mechanism. **Evidence:** full_text; Official PDF, section 3.1, pp. 3–4; copying construction and comparison with adding..

**Relevance:** Constructs an orthogonal rotational solution to delayed copying and contrasts it with an adding task.

**Novelty boundary:** Cue–delay–query tasks and periodic internal clocks already have analytical recurrent solutions.

**Limits:** Blank-delay copying does not test the proposed structured periodic versus matched varying-driver intervention.

**Local PDF:** [henaff2016-orthogonal-memory.pdf](../../../../../literature/papers/henaff2016-orthogonal-memory.pdf) — 9 pages; SHA-256 `fffada2890349e404e2be4478881c9d351a243271b23602e57e43520997ef507`.

<a id="gao2017dimensionality"></a>
## A theory of multineuronal dimensionality, dynamics and measurement

**gao2017dimensionality** — Peiran Gao; Eric Trautmann; Byron Yu; Gopal Santhanam; Stephen Ryu; Krishna Shenoy; Surya Ganguli. 2017. bioRxiv preprint.

[Primary source](https://doi.org/10.1101/214262) · [DOI](https://doi.org/10.1101/214262)

**Themes:** neuroscience, measurement, task complexity. **Evidence:** full_text; Main introduction and theory overview; Supplement II.II participation-ratio definition..

**Relevance:** Task complexity and trajectory smoothness constrain observed neural dimension.

**Novelty boundary:** Dimension changing with task or temporal sampling is not by itself a novel recurrent-memory result.

**Limits:** Selected sections only; trial-averaged neural activity and sampling theory differ from conditioned cue retention.

**Local PDF:** [gao2017dimensionality.pdf](../../../../../literature/papers/gao2017dimensionality.pdf) — 50 pages; SHA-256 `bda3c93b81acd67891203d9d9eb194899670cd8df7d3c2d398b33a2fddc6b2c7`.

<a id="koyluoglu2017"></a>
## Fundamental bound on the persistence and capacity of short-term memory stored as graded persistent activity

**koyluoglu2017** — Onur Ozan Koyluoglu; Yoni Pertzov; Sanjay Manohar; Masud Husain; Ila R. Fiete. 2017. eLife 6: e22225.

[Primary source](https://elifesciences.org/articles/22225) · [DOI](https://doi.org/10.7554/eLife.22225)

**Themes:** graded memory, noisy channels, coding bounds. **Evidence:** abstract; Publisher abstract and article citation..

**Relevance:** Derives recall-precision limits for continuous persistent-memory channels and considers protective encoding.

**Novelty boundary:** Noise-aware graded-memory bounds must be positioned against this prior channel-based framework.

**Limits:** Model-dependent graded activity/noise assumptions; does not bound arbitrary nonlinear finite-category codes.

**Local PDF:** [koyluoglu2017.pdf](../../../../../literature/papers/koyluoglu2017.pdf) — 29 pages; SHA-256 `869be3b6e8c7848e5c352b121306bf6a4b1343ee837583b7043159f0f7da4ad1`.

<a id="wang2017predrnn"></a>
## PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs

**wang2017predrnn** — Yunbo Wang; Mingsheng Long; Jianmin Wang; Zhifeng Gao; Philip S. Yu. 2017. Advances in Neural Information Processing Systems 30.

[Primary source](https://papers.nips.cc/paper_files/paper/2017/hash/e5f6ad6ce374177eef023bf5d0c018b6-Abstract.html)

**Themes:** vision, architecture, spatiotemporal memory. **Evidence:** abstract; Official NeurIPS abstract and author list..

**Relevance:** Native video-prediction baseline with temporal and spatial memory paths.

**Novelty boundary:** Architecture coverage is replication scope; adapter must include both memories.

**Limits:** No claim about periodicity-specific collapse in this source.

**Local PDF:** [wang2017predrnn.pdf](../../../../../literature/papers/wang2017predrnn.pdf) — 10 pages; SHA-256 `f280ba06d6a50e6622b440cffe76af56598715ef130266d36025c8b04a8821e0`.

<a id="li2018dcrnn"></a>
## Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

**li2018dcrnn** — Li, Yaguang; Yu, Rose; Shahabi, Cyrus; Liu, Yan. 2018. ICLR 2018.

[Primary source](https://arxiv.org/abs/1707.01926)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Diffusion graph convolution and recurrent forecasting for road sensors.

**Novelty boundary:** Native traffic recurrence baseline and METR-LA provenance.

**Limits:** Forecasting success does not certify arbitrary-cue retention.

**Local PDF:** [li2018dcrnn.pdf](../../../../../literature/papers/li2018dcrnn.pdf) — 16 pages; SHA-256 `032c978ebeb9af9d06b90dfa2715d32918d93c35308a35f73389e0ff76fb0703`.

<a id="mastrogiuseppe2018"></a>
## Linking connectivity, dynamics and computations in low-rank recurrent neural networks

**mastrogiuseppe2018** — Francesca Mastrogiuseppe; Srdjan Ostojic. 2018. Neuron.

[Primary source](https://arxiv.org/abs/1711.09672) · [DOI](https://doi.org/10.1016/j.neuron.2018.07.003)

**Themes:** low-rank connectivity, useful low-dimensional dynamics. **Evidence:** abstract; arXiv abstract and journal metadata..

**Relevance:** Relates structured low-rank connectivity to low-dimensional computations in recurrent networks.

**Novelty boundary:** Low dimension is often a computational solution; the project must show loss of required information rather than treating low rank as damage.

**Limits:** Connectivity rank, activity dimensionality and task-channel rank are different quantities; full derivations not audited.

**Local PDF:** [mastrogiuseppe2018.pdf](../../../../../literature/papers/mastrogiuseppe2018.pdf) — 69 pages; SHA-256 `bbfb508f2a3de506cf4c094bc52a63e76e79965ad8509f730ef3fbd53173b608`.

<a id="schuecker2018"></a>
## Optimal sequence memory in driven random networks

**schuecker2018** — Jannis Schuecker; Sven Goedeke; Moritz Helias. 2018. Physical Review X 8: 041029.

[Primary source](https://arxiv.org/abs/1603.01880) · [DOI](https://doi.org/10.1103/PhysRevX.8.041029)

**Themes:** driven chaos, Lyapunov stability, memory capacity. **Evidence:** abstract; arXiv abstract; publisher journal reference and DOI..

**Relevance:** Dynamic mean-field theory connects input, Lyapunov stability and sequence memory; locally expansive but nonchaotic dynamics can improve memory.

**Novelty boundary:** A single exponent threshold or edge-of-chaos narrative cannot replace task-specific retention measurements.

**Limits:** Random-network mean-field regime, not a general gated-model theorem; local expansion and asymptotic stability must be distinguished.

**Local PDF:** [schuecker2018.pdf](../../../../../literature/papers/schuecker2018.pdf) — 18 pages; SHA-256 `167e641e54aa8d48f70027a660be691d20c43ef5ec19bbdee00462e3e577ebbb`.

<a id="universal_transformer"></a>
## Universal Transformers

**universal_transformer** — Mostafa Dehghani; Stephan Gouws; Oriol Vinyals; Jakob Uszkoreit; Łukasz Kaiser. 2018. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/1807.03819)

**Themes:** depth_recurrence, expressivity. **Evidence:** abstract; Primary abstract.

**Relevance:** Recurrent attention and adaptive per-position halting predate current looped models.

**Novelty boundary:** Low for periodic memory; high for novelty of depth recurrence.

**Limits:** Step encoding and halting can make the effective iteration nonautonomous.

**Local PDF:** [universal_transformer.pdf](../../../../../literature/papers/universal_transformer.pdf) — 23 pages; SHA-256 `d4b8c752b7ad28b6df67bab764ba6744c575ffe8e2f09879c750b05c7daf3e90`.

<a id="weiss2018-finite-precision"></a>
## On the Practical Computational Power of Finite Precision RNNs for Language Recognition

**weiss2018-finite-precision** — Gail Weiss; Yoav Goldberg; Eran Yahav. 2018. ACL, Short Papers:740–745.

[Primary source](https://aclanthology.org/P18-2117/) · [DOI](https://doi.org/10.18653/v1/P18-2117)

**Themes:** finite precision, counting, architecture expressivity. **Evidence:** abstract; Official ACL Anthology abstract, author list, DOI and publication metadata..

**Relevance:** Studies practical numerical restrictions and differences between gated recurrent families.

**Novelty boundary:** Categorical capacity and architecture differences cannot be inferred from differentiable dimension arguments alone.

**Limits:** Formal precision/range assumptions must be read before applying results to bounded FP32 state; no hardware-capacity theorem asserted here.

**Local PDF:** [weiss2018-finite-precision.pdf](../../../../../literature/papers/weiss2018-finite-precision.pdf) — 6 pages; SHA-256 `11d564a7c259380e4fdedccab1e8f0252c563846583abecbd0820e2305d5e2d6`.

<a id="albert"></a>
## ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

**albert** — Zhenzhong Lan; Mingda Chen; Sebastian Goodman; Kevin Gimpel; Piyush Sharma; Radu Soricut. 2019. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/1909.11942)

**Themes:** weight_sharing. **Evidence:** abstract; Primary abstract.

**Relevance:** Parameter sharing is architectural context for tied-depth controls.

**Novelty boundary:** Low: architecture context.

**Limits:** Sharing alone does not establish arbitrary test-time looping or periodic memory dynamics.

**Local PDF:** [albert.pdf](../../../../../literature/papers/albert.pdf) — 17 pages; SHA-256 `d631c30f3f41684862e854ad1cb379e9dbf9b1c1fa1510ab618f45b8dfc07e2c`.

<a id="deep_equilibrium"></a>
## Deep Equilibrium Models

**deep_equilibrium** — Shaojie Bai; J. Zico Kolter; Vladlen Koltun. 2019. Advances in Neural Information Processing Systems 32 (NeurIPS 2019).

[Primary source](https://arxiv.org/abs/1909.01377)

**Themes:** equilibrium, depth_recurrence, depth recurrence, input reinjection, equilibrium inference. **Evidence:** abstract; Primary abstract.

**Relevance:** Input-conditioned equilibrium models establish useful convergent tied-depth computation.

**Novelty boundary:** High against contraction necessarily erasing input.

**Limits:** Root-finding and implicit differentiation are not temporal periodic forcing; full implementation was not audited.

**Local PDF:** [deep_equilibrium.pdf](../../../../../literature/papers/deep_equilibrium.pdf) — 16 pages; SHA-256 `731b37714d475497f02af65f85024bed2deb733b4aaecb757179909fd147916c`.

<a id="hewitt2019-control-probes"></a>
## Designing and Interpreting Probes with Control Tasks

**hewitt2019-control-probes** — John Hewitt; Percy Liang. 2019. EMNLP-IJCNLP:2733–2743.

[Primary source](https://aclanthology.org/D19-1275/) · [DOI](https://doi.org/10.18653/v1/D19-1275)

**Themes:** probing, selectivity, control tasks. **Evidence:** abstract; Official ACL Anthology abstract and metadata..

**Relevance:** Uses control tasks to distinguish representation evidence from probe capacity.

**Novelty boundary:** High decoding accuracy alone is not an uncontested representation interpretation.

**Limits:** Linguistic probing setting; adapt controls to randomized independent memory tags and held-out cue histories.

**Local PDF:** [hewitt2019-control-probes.pdf](../../../../../literature/papers/hewitt2019-control-probes.pdf) — 11 pages; SHA-256 `557f5d7dc18b17d641c9e54c91c0071b00612f9ec051fc788bcc7117f4d194d0`.

<a id="korsky2019-rnn-power"></a>
## On the Computational Power of RNNs

**korsky2019-rnn-power** — Samuel A. Korsky; Robert C. Berwick. 2019. arXiv preprint.

[Primary source](https://arxiv.org/abs/1906.06349)

**Themes:** finite precision, formal languages, expressivity assumptions. **Evidence:** abstract; Primary arXiv abstract and author metadata..

**Relevance:** Shows why numerical assumptions matter in comparisons of recurrent computational power.

**Novelty boundary:** Infinite-real and finite-state memory claims describe different resources and must not be mixed.

**Limits:** Detailed definitions were not audited; do not treat differences from Weiss et al. as a contradiction without aligning precision models.

**Local PDF:** [korsky2019-rnn-power.pdf](../../../../../literature/papers/korsky2019-rnn-power.pdf) — 25 pages; SHA-256 `11f76f9cbd0a3cf6a898924f67561e96e956221a6ad8ec48b63b6551a1fd5946`.

<a id="orhan2019-memory-representations"></a>
## A diverse range of factors affect the nature of neural representations underlying short-term memory

**orhan2019-memory-representations** — A. Emin Orhan; Wei Ji Ma. 2019. Nature Neuroscience 22:275–283.

[Primary source](https://www.nature.com/articles/s41593-018-0314-y) · [DOI](https://doi.org/10.1038/s41593-018-0314-y)

**Themes:** working-memory geometry, dynamic representations, task dependence. **Evidence:** abstract; Publisher abstract and bibliographic record; cookie barrier prevented final full-text retrieval..

**Relevance:** Memory representation form varies with task and circuit factors.

**Novelty boundary:** Observed dynamics or apparent coding differences alone do not establish memory absence.

**Limits:** No detailed methods or theorem claims audited here. Author's published initial retained without speculative expansion.

**Copy/version note:** Author-hosted published PDF marked Corrected: Publisher Correction.

**Local PDF:** [orhan2019-memory-representations.pdf](../../../../../literature/papers/orhan2019-memory-representations.pdf) — 12 pages; SHA-256 `29b4171db80c4b9624a33a9c16e71366cbb5c501cbd87d22ece32380f8f5ac91`.

<a id="recanatesi2019dimensionality"></a>
## Dimensionality in recurrent spiking networks: Global trends in activity and local origins in connectivity

**recanatesi2019dimensionality** — Stefano Recanatesi; Gabriel Koch Ocker; Michael A. Buice; Eric Shea-Brown. 2019. PLOS Computational Biology.

[Primary source](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006446) · [DOI](https://doi.org/10.1371/journal.pcbi.1006446)

**Themes:** neuroscience, covariance, recurrent dynamics. **Evidence:** full_text; Introduction and Results dimensionality definitions/connectivity expansion..

**Relevance:** Relates recurrent connectivity and external input to covariance dimensionality.

**Novelty boundary:** Generic recurrence-to-dimension mechanisms have substantial neuroscience antecedents.

**Limits:** Selected sections; spiking/Hawkes and linearized population models do not directly prove useful-history loss in trained gated networks.

**Local PDF:** [recanatesi2019dimensionality.pdf](../../../../../literature/papers/recanatesi2019dimensionality.pdf) — 29 pages; SHA-256 `4bebfb2c7afcfabcd3dc914a7fe2fff3437595dc643c9dd2d67a12c580c11828`.

<a id="ceni2020"></a>
## The Echo Index and multistability in input-driven recurrent neural networks

**ceni2020** — Andrea Ceni; Peter Ashwin; Lorenzo Livi; Claire Postlethwaite. 2020. Physica D: Nonlinear Phenomena.

[Primary source](https://arxiv.org/abs/2001.07694) · [DOI](https://doi.org/10.1016/j.physd.2020.132609)

**Themes:** echo index, multistability, input-driven dynamics. **Evidence:** abstract; arXiv abstract and related DOI metadata, version 2..

**Relevance:** Defines input-dependent multiplicity of stable recurrent responses and explains reliable computation beyond global echo-state uniqueness.

**Novelty boundary:** Multiple retained memories under a common driver are an established phenomenon, not a new exception discovered by this project.

**Limits:** Full theorem assumptions and numerical protocols not audited here.

**Local PDF:** [ceni2020.pdf](../../../../../literature/papers/ceni2020.pdf) — 43 pages; SHA-256 `fb9852fa5b986c36bc35b0aa9a2935d9466b1e51f5175c1c4629345d20c6a142`.

<a id="oono2020exponential"></a>
## Graph Neural Networks Exponentially Lose Expressive Power for Node Classification

**oono2020exponential** — Oono, Kenta; Suzuki, Taiji. 2020. ICLR 2020.

[Primary source](https://arxiv.org/abs/1905.10947)

**Themes:** graph collapse. **Evidence:** abstract; Primary arXiv abstract; ICLR proceedings PDF located.

**Relevance:** Spectral sufficient conditions yield asymptotic information loss in deep GCNs.

**Novelty boundary:** Dynamical explanations of repeated graph propagation are established.

**Limits:** Static graph depth and norm conditions; not a blanket theorem about trained temporal models.

**Local PDF:** [oono2020exponential.pdf](../../../../../literature/papers/oono2020exponential.pdf) — 37 pages; SHA-256 `2448b3525b81d87a65c6c28c775241358ebedb0c6ad555f1ef0a443b3885faf7`.

<a id="papyan2020neuralcollapse"></a>
## Prevalence of Neural Collapse during the terminal phase of deep learning training

**papyan2020neuralcollapse** — Papyan, Vardan; Han, X. Y.; Donoho, David L.. 2020. Proceedings of the National Academy of Sciences.

[Primary source](https://arxiv.org/abs/2008.08186) · [DOI](https://doi.org/10.1073/pnas.2015509117)

**Themes:** collapse terminology. **Evidence:** abstract; Primary abstract and PMC article record.

**Relevance:** Terminal training exhibits class-conditioned neural-collapse geometry.

**Novelty boundary:** The phrase neural collapse has a precise existing supervised-learning meaning.

**Limits:** Training-phase class geometry is not fixed-weight repetition-induced forgetting.

**Local PDF:** [papyan2020neuralcollapse.pdf](../../../../../literature/papers/papyan2020neuralcollapse.pdf) — 17 pages; SHA-256 `559951b1b46e55f9c305a127f2f6f9790152848ea636cf99e430d71c543ef28e`.

<a id="pareja2020evolvegcn"></a>
## EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs

**pareja2020evolvegcn** — Pareja, Aldo; Domeniconi, Giacomo; Chen, Jie; Ma, Tengfei; Suzumura, Toyotaro; Kanezashi, Hiroki; Kaler, Tim; Schardl, Tao B.; Leiserson, Charles E.. 2020. AAAI 2020.

[Primary source](https://arxiv.org/abs/1902.10191)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Recurrent evolution of graph convolution parameters.

**Novelty boundary:** Established model family; evolving weights belong in the persistent state.

**Limits:** State is not interchangeable with node-level hidden activations.

**Local PDF:** [pareja2020evolvegcn.pdf](../../../../../literature/papers/pareja2020evolvegcn.pdf) — 9 pages; SHA-256 `e15af7dc50a4eb89c343d36b489d3589c4bddf7d1d888cb63073b445036d2642`.

<a id="rossi2020tgn"></a>
## Temporal Graph Networks for Deep Learning on Dynamic Graphs

**rossi2020tgn** — Rossi, Emanuele; Chamberlain, Ben; Frasca, Fabrizio; Eynard, Davide; Monti, Federico; Bronstein, Michael. 2020. arXiv preprint.

[Primary source](https://arxiv.org/abs/2006.10637)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Event-based temporal graph memory and message updates.

**Novelty boundary:** Explicit graph memory is longstanding; relevant comparison to snapshot recurrence.

**Limits:** Event timestamps, stored messages and update chronology affect state completeness.

**Local PDF:** [rossi2020tgn.pdf](../../../../../literature/papers/rossi2020tgn.pdf) — 16 pages; SHA-256 `580113858f363b80388bbf7c1b2a42482cfb9b6debdb3db2cdde8cb6db8166c0`.

<a id="schuessler2020"></a>
## Dynamics of random recurrent networks with correlated low-rank structure

**schuessler2020** — Friedrich Schuessler; Alexis Dubreuil; Francesca Mastrogiuseppe; Srdjan Ostojic; Omri Barak. 2020. Physical Review Research 2: 013111.

[Primary source](https://arxiv.org/abs/1909.04358) · [DOI](https://doi.org/10.1103/PhysRevResearch.2.013111)

**Themes:** low-rank structure, spectral outliers, stability. **Evidence:** abstract; arXiv abstract; published author-PDF title/DOI metadata..

**Relevance:** Connects correlations between random and low-rank connectivity to dynamical regimes and spectral outliers.

**Novelty boundary:** Structured low-dimensional dynamical mechanisms have mature analytic antecedents.

**Limits:** Autonomous/random structured networks are not equivalent to trained periodic full-state Jacobian products.

**Local PDF:** [schuessler2020.pdf](../../../../../literature/papers/schuessler2020.pdf) — 18 pages; SHA-256 `b7df8cf0dc9a06be754233787397d380d5bba92697bbe32db6080507c1130ed0`.

<a id="vogt2020"></a>
## On Lyapunov Exponents for RNNs: Understanding Information Propagation Using Dynamical Systems Tools

**vogt2020** — Ryan Vogt; Maximilian Puelma Touzel; Eli Shlizerman; Guillaume Lajoie. 2020. arXiv preprint.

[Primary source](https://arxiv.org/abs/2006.14123)

**Themes:** RNN Lyapunov analysis, information propagation. **Evidence:** abstract; arXiv abstract and author metadata..

**Relevance:** Uses Lyapunov analysis to study propagation and performance in recurrent networks.

**Novelty boundary:** Applying conditional exponents to learned RNNs is established methodology rather than sufficient novelty.

**Limits:** No claim that exponent sign proves global history forgetting; journal/workshop venue and DOI not verified here.

**Local PDF:** [vogt2020.pdf](../../../../../literature/papers/vogt2020.pdf) — 15 pages; SHA-256 `3cce81c0aba8240edade9b491669971bbd21d1658c9c438aecc426080e86eb56`.

<a id="zhao2020tgcn"></a>
## T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction

**zhao2020tgcn** — Zhao, Ling; Song, Yujiao; Zhang, Chao; Liu, Yu; Wang, Pu; Lin, Tao; Deng, Min; Li, Haifeng. 2020. IEEE Transactions on Intelligent Transportation Systems.

[Primary source](https://arxiv.org/abs/1811.05320) · [DOI](https://doi.org/10.1109/TITS.2019.2935152)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Temporal graph convolution combined with gated recurrence for traffic prediction.

**Novelty boundary:** Established modality-native recurrent baseline.

**Limits:** Need actual implementation state contract before dynamical comparison.

**Local PDF:** [zhao2020tgcn.pdf](../../../../../literature/papers/zhao2020tgcn.pdf) — 11 pages; SHA-256 `20517b9c3ad7900e8cebb619b72d2015c9098d73971af161d8ca0550323f1307`.

<a id="meyer2021hutchpp"></a>
## Hutch++: Optimal Stochastic Trace Estimation

**meyer2021hutchpp** — Meyer, Raphael A.; Musco, Cameron; Musco, Christopher; Woodruff, David P.. 2021. Symposium on Simplicity in Algorithms.

[Primary source](https://arxiv.org/abs/2010.09649)

**Themes:** numerical methods. **Evidence:** abstract; Primary arXiv abstract and PMC publication record.

**Relevance:** Variance-reduced stochastic trace estimation for PSD operators.

**Novelty boundary:** Scalable trace-based effective dimensions build on established estimators.

**Limits:** Ratios of estimated traces need uncertainty analysis.

**Local PDF:** [meyer2021hutchpp.pdf](../../../../../literature/papers/meyer2021hutchpp.pdf) — 24 pages; SHA-256 `42131d8143809fe493dcd81f310c5f8017cba2a6ddbcf7f3fb6bbd46340ebd94`.

<a id="pure_attention_rank"></a>
## Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth

**pure_attention_rank** — Yihe Dong; Jean-Baptiste Cordonnier; Andreas Loukas. 2021. ICML 2021.

[Primary source](https://arxiv.org/abs/2103.03404)

**Themes:** attention, rank. **Evidence:** abstract; Primary abstract.

**Relevance:** Pure-attention depth-collapse theorem is foundational overlap.

**Novelty boundary:** High for a generic repeated-attention collapse claim.

**Limits:** Pure attention omits residual and MLP mechanisms; token rank is not retained-history dimension.

**Local PDF:** [pure_attention_rank.pdf](../../../../../literature/papers/pure_attention_rank.pdf) — 22 pages; SHA-256 `6e996091c58b96f1324e54a322e513e33ff8c2b8b76aef5d2e02bd6e0a5a9e22`.

<a id="ravichander2021-probe-relevance"></a>
## Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?

**ravichander2021-probe-relevance** — Abhilasha Ravichander; Yonatan Belinkov; Eduard Hovy. 2021. EACL:3363–3377.

[Primary source](https://aclanthology.org/2021.eacl-main.295/) · [DOI](https://doi.org/10.18653/v1/2021.eacl-main.295)

**Themes:** task relevance, causal interpretation, probing confounds. **Evidence:** abstract; Official ACL Anthology abstract and publication record; primary arXiv metadata..

**Relevance:** Controlled experiments separate decodability from information relevant to task decisions.

**Novelty boundary:** Probe recovery of tags is not evidence that the trained model uses them.

**Limits:** Causal use requires additional interventions; state swaps establish a stronger but still intervention-specific claim.

**Local PDF:** [ravichander2021-probe-relevance.pdf](../../../../../literature/papers/ravichander2021-probe-relevance.pdf) — 15 pages; SHA-256 `918432ffa2b09d6be28747bec99f6e38d4765141e5e3094201aefe499179d2d6`.

<a id="s4"></a>
## Efficiently Modeling Long Sequences with Structured State Spaces

**s4** — Albert Gu; Karan Goel; Christopher Ré. 2021. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2111.00396)

**Themes:** ssm, memory, cross_modal. **Evidence:** abstract; Primary abstract.

**Relevance:** Structured linear state spaces supply cross-modal long-memory baselines.

**Novelty boundary:** Low for periodic novelty; important analytical control.

**Limits:** Linear transition submodule differs from a stacked nonlinear model; frozen full-state tests are needed.

**Local PDF:** [s4.pdf](../../../../../literature/papers/s4.pdf) — 32 pages; SHA-256 `872b4ef302312d04ef4d55610613c0e59da32c443a2e38552d7cebb7dc3636fa`.

<a id="block_recurrent_transformer"></a>
## Block-Recurrent Transformers

**block_recurrent_transformer** — DeLesley Hutchins; Imanol Schlag; Yuhuai Wu; Ethan Dyer; Behnam Neyshabur. 2022. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2203.07852)

**Themes:** temporal_recurrence, attention, memory. **Evidence:** abstract; Primary abstract.

**Relevance:** Block-wise temporal attention recurrence provides the bounded-state comparator.

**Novelty boundary:** Medium for architecture scope.

**Limits:** Persistent vectors and any local cache must both count as state; blocks determine the recurrence clock.

**Local PDF:** [block_recurrent_transformer.pdf](../../../../../literature/papers/block_recurrent_transformer.pdf) — 25 pages; SHA-256 `a9e62ba73566215946c8122a1679454a1e80a35e6242115fa3af85362fe11449`.

<a id="chen2022gclstm"></a>
## GC-LSTM: Graph Convolution Embedded LSTM for Dynamic Link Prediction

**chen2022gclstm** — Chen, Jinyin; Wang, Xueke; Xu, Xuanheng. 2022. Applied Intelligence; preprint first posted 2018.

[Primary source](https://arxiv.org/abs/1812.04206)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract and journal reference.

**Relevance:** Graph-convolution embedded LSTM used in the inherited preliminary experiments.

**Novelty boundary:** Baseline architecture, not a new contribution of this project.

**Limits:** Publication year is journal year; analyse all node hidden and cell state.

**Local PDF:** [chen2022gclstm.pdf](../../../../../literature/papers/chen2022gclstm.pdf) — 21 pages; SHA-256 `eb2a0a49c9f1c66a1ace16f45598ce92ceb03d0b2218fef525a3b8d9f2f97b42`.

<a id="hu2022"></a>
## The spectrum of covariance matrices of randomly connected recurrent neuronal networks with linear dynamics

**hu2022** — Yu Hu; Haim Sompolinsky. 2022. PLOS Computational Biology 18(7): e1010327.

[Primary source](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010327) · [DOI](https://doi.org/10.1371/journal.pcbi.1010327)

**Themes:** covariance spectrum, linear recurrent dynamics, participation ratio. **Evidence:** abstract; Publisher abstract, authors and citation header..

**Relevance:** Analyzes recurrent connectivity and covariance spectral structure in linear networks.

**Novelty boundary:** A covariance-concentration mechanism needs comparison to established random-network spectra, not just visualization.

**Limits:** The covariance/noise model differs from finite required-history clouds; current audit does not reproduce equations.

**Local PDF:** [hu2022.pdf](../../../../../literature/papers/hu2022.pdf) — 27 pages; SHA-256 `1cf5988fc87b9702462b1b573356a1f221ad0590b4642014df6c3a105ea4f0cb`.

<a id="jing2022dimensional"></a>
## Understanding Dimensional Collapse in Contrastive Self-supervised Learning

**jing2022dimensional** — Jing, Li; Vincent, Pascal; LeCun, Yann; Tian, Yuandong. 2022. ICLR 2022.

[Primary source](https://arxiv.org/abs/2110.09348)

**Themes:** collapse terminology. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Dimensional collapse in contrastive self-supervised learning.

**Novelty boundary:** Dimension collapse is established terminology and objective-dependent phenomenon.

**Limits:** Training objectives and data augmentations differ from inference dynamics.

**Local PDF:** [jing2022dimensional.pdf](../../../../../literature/papers/jing2022dimensional.pdf) — 17 pages; SHA-256 `575847f8de13f88d1814e74566a5148ac30a6c40429e67c223c22761a5785d6f`.

<a id="rusch2022graphcon"></a>
## Graph-Coupled Oscillator Networks

**rusch2022graphcon** — Rusch, T. Konstantin; Chamberlain, Benjamin P.; Rowbottom, James; Mishra, Siddhartha; Bronstein, Michael M.. 2022. ICML 2022.

[Primary source](https://arxiv.org/abs/2202.02296)

**Themes:** graph dynamics. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Oscillator-based graph propagation links oversmoothing to stability of homogeneous states.

**Novelty boundary:** Oscillatory dynamics as an anti-oversmoothing mechanism are prior art.

**Limits:** Continuous-depth oscillator mechanism is distinct from forced periodic memory.

**Local PDF:** [rusch2022graphcon.pdf](../../../../../literature/papers/rusch2022graphcon.pdf) — 22 pages; SHA-256 `b09433b0bd4a2673320194e58980f9b480f916dbe66fc11dc397fe375d3ad996`.

<a id="transformer_signal_rank"></a>
## Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse

**transformer_signal_rank** — Lorenzo Noci; Sotiris Anagnostidis; Luca Biggio; Antonio Orvieto; Sidak Pal Singh; Aurelien Lucchi. 2022. NeurIPS 2022.

[Primary source](https://arxiv.org/abs/2206.03126) · [DOI](https://doi.org/10.52202/068431-1972)

**Themes:** attention, rank, initialization. **Evidence:** abstract; NeurIPS abstract.

**Relevance:** Rank collapse and gradient propagation connect spectra to transformer design.

**Novelty boundary:** Medium: spectral diagnosis is established.

**Limits:** Initialization and training signal propagation differ from fixed-weight periodic inference.

**Local PDF:** [transformer_signal_rank.pdf](../../../../../literature/papers/transformer_signal_rank.pdf) — 35 pages; SHA-256 `999e1d38f45e86f5f368cf8c8622a95b90b342af0d788ef90deffc5676e7d5b1`.

<a id="ashwin2023-echo-repetitions"></a>
## Transitions in echo index and dependence on input repetitions

**ashwin2023-echo-repetitions** — Peter Ashwin; Andrea Ceni. 2023. arXiv preprint; published venue not verified in this audit.

[Primary source](https://arxiv.org/abs/2309.04728)

**Themes:** echo index, input repetitions, multistability, dwell time. **Evidence:** full_text; Sections 2.1–2.3, PDF pp. 6–10; Assumption 2.1 and Theorems 2.3–2.4; sections 3.1–3.3, pp. 11–13..

**Relevance:** Relates symbol dwell/repetition constraints to multiplicity of attracting input-driven responses.

**Novelty boundary:** Repetition-induced retention or loss of initial-state distinctions is already an explicit theoretical topic.

**Limits:** Theorems impose basin/hyperbolicity conditions; finite-horizon clustering estimates are not task-specific memory measures.

**Reviewed version:** `2309.04728v1` (the first PDF candidate is pinned to this version).

**Local PDF:** [ashwin2023-echo-repetitions.pdf](../../../../../literature/papers/ashwin2023-echo-repetitions.pdf) — 17 pages; SHA-256 `4f77592018e8bf0cfed46d5ea659d7a004add8cc452c6709debeafe0ec6c3e4f`.

<a id="engelken2023"></a>
## Lyapunov spectra of chaotic recurrent neural networks

**engelken2023** — Rainer Engelken; Fred Wolf; Larry F. Abbott. 2023. Physical Review Research 5: 043044.

[Primary source](https://arxiv.org/abs/2006.02427) · [DOI](https://doi.org/10.1103/PhysRevResearch.5.043044)

**Themes:** Lyapunov spectra, attractor dimension, input suppression. **Evidence:** abstract; arXiv abstract, journal reference, DOI; MPI publication metadata..

**Relevance:** Analyzes full Lyapunov spectra of recurrent dynamics and effects of input fluctuations on chaotic entropy and attractor dimension.

**Novelty boundary:** Reduced dynamical dimension with forcing is established; connecting it to required-task memory must add evidence.

**Limits:** 2020 preprint versus 2023 journal year distinguished. Attractor dimension is not covariance PR or target decodability.

**Local PDF:** [engelken2023.pdf](../../../../../literature/papers/engelken2023.pdf) — 28 pages; SHA-256 `3c0d3c1fc07ef9b18fd837320ab33afdee7dbbeca6df81a342bc52aa659058e2`.

<a id="looped_programmable"></a>
## Looped Transformers as Programmable Computers

**looped_programmable** — Angeliki Giannou; Shashank Rajput; Jy-yong Sohn; Kangwook Lee; Jason D. Lee; Dimitris Papailiopoulos. 2023. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2301.13196)

**Themes:** depth_recurrence, expressivity. **Evidence:** abstract; Primary abstract.

**Relevance:** Constructed looped transformers execute programs and iterative algorithms.

**Novelty boundary:** High against inevitable failure of all repeated blocks.

**Limits:** Hand-programmed expressivity is not a prediction of trained models or finite-precision memory.

**Local PDF:** [looped_programmable.pdf](../../../../../literature/papers/looped_programmable.pdf) — 64 pages; SHA-256 `e3f11f6989757a0924d574a2c11d1a6afd21394111449ae27654dbd0c313b3ab`.

<a id="mamba"></a>
## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**mamba** — Albert Gu; Tri Dao. 2023. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2312.00752)

**Themes:** ssm, selectivity, cross_modal. **Evidence:** abstract; Primary abstract.

**Relevance:** Input-dependent selection explicitly regulates retention and forgetting.

**Novelty boundary:** Medium for a general cross-modal recurrence claim.

**Limits:** Token-conditioned coefficients and convolution history are part of complete dynamics.

**Local PDF:** [mamba.pdf](../../../../../literature/papers/mamba.pdf) — 36 pages; SHA-256 `adf70ed1803c85b1899dec3e21f3af0b124411439e8654b840ea65f7b9f52b2e`.

<a id="wu2023attention"></a>
## Demystifying Oversmoothing in Attention-Based Graph Neural Networks

**wu2023attention** — Wu, Xinyi; Ajorlou, Amir; Wu, Zihui; Jadbabaie, Ali. 2023. NeurIPS 2023.

[Primary source](https://arxiv.org/abs/2305.16102)

**Themes:** graph collapse. **Evidence:** abstract; Primary abstract and proceedings record.

**Relevance:** Time-varying nonlinear graph attention is analysed with matrix products and joint spectral radius.

**Novelty boundary:** Time-varying attention propagation and exponential oversmoothing already have theory.

**Limits:** Requires paper-specific mixing and boundedness assumptions; state axis differs from cue-conditioned memory.

**Local PDF:** [wu2023attention.pdf](../../../../../literature/papers/wu2023attention.pdf) — 24 pages; SHA-256 `e0e5ea3b94cfb811fa21aa7410cf02c06576525820d441edcbd575a045384b91`.

<a id="gravina2024ctan"></a>
## Long Range Propagation on Continuous-Time Dynamic Graphs

**gravina2024ctan** — Gravina, Alessio; Lovisotto, Giulio; Gallicchio, Claudio; Bacciu, Davide; Grohnfeldt, Claas. 2024. ICML 2024.

[Primary source](https://arxiv.org/abs/2406.02740)

**Themes:** graph architecture. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Continuous-time dynamic graph models seek long-range propagation.

**Novelty boundary:** Long-term temporal graph memory is an established architectural objective.

**Limits:** Architectural context; no verified anticipation of the exact matched-body experiment.

**Local PDF:** [gravina2024ctan.pdf](../../../../../literature/papers/gravina2024ctan.pdf) — 20 pages; SHA-256 `706737f2f677eb13e288e4fc3919ba3e1fd3b695078c0c72d69585183cb1b20b`.

<a id="mamba2_ssd"></a>
## Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**mamba2_ssd** — Tri Dao; Albert Gu. 2024. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2405.21060)

**Themes:** ssm, attention, duality. **Evidence:** abstract; Primary abstract.

**Relevance:** Structured state-space duality connects attention-like operators and Mamba-2.

**Novelty boundary:** Medium: attention-versus-SSM separation is already qualified.

**Limits:** Operator equivalence does not make all architectures share the same nonlinear full-state Jacobian.

**Local PDF:** [mamba2_ssd.pdf](../../../../../literature/papers/mamba2_ssd.pdf) — 52 pages; SHA-256 `6fe16414fa471a5ab150cddcba5840d3a94e347e96992c50c12c1fb8b919479f`.

<a id="pals2024-phase-locked-memory"></a>
## Trained recurrent neural networks develop phase-locked limit cycles in a working memory task

**pals2024-phase-locked-memory** — Matthijs Pals; Jakob H. Macke; Omri Barak. 2024. PLOS Computational Biology 20(2):e1011852.

[Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/) · [DOI](https://doi.org/10.1371/journal.pcbi.1011852)

**Themes:** periodic forcing, working memory, Floquet stability, categorical attractors, Floquet, low-rank dynamics, categorical memory. **Evidence:** full_text; Results, Figs. 1–3; Fig. 3 stable memory cycles; Methods Poincare maps and linear stability analysis; frequency/amplitude manipulation..

**Relevance:** Transient cue identities select stable phase-locked memory cycles in a periodically driven trained RNN.

**Novelty boundary:** Direct counterexample to interpreting local contraction or repeated forcing as inevitable categorical forgetting.

**Limits:** Two cue identities and oscillator-driven output; not the proposed higher-dimensional matched-driver query task. Published middle initial retained without inventing its expansion.

**Local PDF:** [pals2024-phase-locked-memory.pdf](../../../../../literature/papers/pals2024-phase-locked-memory.pdf) — 23 pages; SHA-256 `baa0b9a13fd26fe171639b6558d361cbe61dd7eba3fddddc9b59567fbfa20fd0`.

<a id="repeat_after_me"></a>
## Repeat After Me: Transformers are Better than State Space Models at Copying

**repeat_after_me** — Samy Jelassi; David Brandfonbrener; Sham M. Kakade; Eran Malach. 2024. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2402.01032)

**Themes:** memory, capacity, copying, length. **Evidence:** abstract; Primary abstract.

**Relevance:** Copying lower bounds separate fixed-state capacity from context access.

**Novelty boundary:** High competing explanation for long-horizon failure.

**Limits:** Arbitrary string copying exceeds the proposed fixed-number-of-tags task; precision and cache resources matter.

**Local PDF:** [repeat_after_me.pdf](../../../../../literature/papers/repeat_after_me.pdf) — 20 pages; SHA-256 `1e2fdb93eca16f338264f6a330f1866ebd062523d48ed1f38973f694a20fbaf2`.

<a id="roth2024rank"></a>
## Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks

**roth2024rank** — Roth, Andreas; Liebig, Thomas. 2024. Proceedings of the Second Learning on Graphs Conference, PMLR 231.

[Primary source](https://arxiv.org/abs/2308.16800)

**Themes:** graph collapse. **Evidence:** abstract; PMLR abstract and bibliographic record.

**Relevance:** Rank collapse unifies graph oversmoothing and feature over-correlation.

**Novelty boundary:** Using rank instead of node similarity alone is established.

**Limits:** Results concern graph propagation operators, not arbitrary nonlinear recurrent states.

**Local PDF:** [roth2024rank.pdf](../../../../../literature/papers/roth2024rank.pdf) — 23 pages; SHA-256 `d7c6f0d80f7dbb9363b36f6cb372fcd920527178246fea0726300edadb8f9a3e`.

<a id="roth2024splitting"></a>
## Preventing Representational Rank Collapse in MPNNs by Splitting the Computational Graph

**roth2024splitting** — Roth, Andreas; Bause, Franka; Kriege, Nils M.; Liebig, Thomas. 2024. Learning on Graphs 2024.

[Primary source](https://arxiv.org/abs/2409.11504)

**Themes:** graph collapse. **Evidence:** abstract; arXiv abstract; publication comment.

**Relevance:** Multi-relational message passing is designed to preserve independent representation directions.

**Novelty boundary:** Anti-collapse graph architectural interventions already exist.

**Limits:** An architecture intervention does not isolate temporal periodic forcing.

**Local PDF:** [roth2024splitting.pdf](../../../../../literature/papers/roth2024splitting.pdf) — 24 pages; SHA-256 `d45a39dd9831661bc9817504440b10258aa8970871231d19f725630536da3b3a`.

<a id="transformer_oversmoothing_record"></a>
## Setting the Record Straight on Transformer Oversmoothing

**transformer_oversmoothing_record** — Gbètondji J-S Dovonon; Michael M. Bronstein; Matt J. Kusner. 2024. arXiv preprint; conference status not verified.

[Primary source](https://arxiv.org/abs/2401.04301)

**Themes:** attention, oversmoothing, spectrum. **Evidence:** full_text; arXiv v2 Secs.2–4, Assumption3.1, Theorems3.3–3.4.

**Relevance:** Value/projection spectra distinguish smoothing from sharpening.

**Novelty boundary:** High against universal transformer smoothing.

**Limits:** Sec.3 analyzes fixed attention and weight matrices; dynamic attention and task-memory loss require separate tests.

**Local PDF:** [transformer_oversmoothing_record.pdf](../../../../../literature/papers/transformer_oversmoothing_record.pdf) — 19 pages; SHA-256 `7a139274e66ac97f7a664d0c5124001fc13f03326420c8475b4b148ab2cc70fb`.

<a id="xlstm"></a>
## xLSTM: Extended Long Short-Term Memory

**xlstm** — Maximilian Beck; Korbinian Pöppel; Markus Spanring; Andreas Auer; Oleksandra Prudnikova; Michael Kopp; Günter Klambauer; Johannes Brandstetter; Sepp Hochreiter. 2024. arXiv preprint; conference venue not reverified.

[Primary source](https://arxiv.org/abs/2405.04517)

**Themes:** recurrence, gating, matrix_memory. **Evidence:** abstract; Primary abstract.

**Relevance:** Scalar and matrix memories provide contrasting gated memory mechanisms.

**Novelty boundary:** Medium for recurrence-wide claims.

**Limits:** Cell, hidden, normalizer and stabilization states cannot be reduced to output activations.

**Local PDF:** [xlstm.pdf](../../../../../literature/papers/xlstm.pdf) — 55 pages; SHA-256 `4ff5c189ebae4b8d1df59435654434a1993b7de3924708609fce1dd086cb8da4`.

<a id="arroyo2025bridging"></a>
## On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning

**arroyo2025bridging** — Arroyo, Álvaro; Gravina, Alessio; Gutteridge, Benjamin; Barbero, Federico; Gallicchio, Claudio; Dong, Xiaowen; Bronstein, Michael; Vandergheynst, Pierre. 2025. arXiv preprint, v2.

[Primary source](https://arxiv.org/abs/2502.10818)

**Themes:** graph collapse. **Evidence:** full_text; Sections 3–4; Lemma 3.1; Propositions 4.3–4.4; Appendix E.

**Relevance:** Connects graph propagation, recurrent models, Jacobian contraction and feature collapse.

**Novelty boundary:** A general Jacobian explanation of graph collapse is already explicit prior work.

**Limits:** Depth, common-zero-map and contraction assumptions differ from input-forced temporal memory.

**Local PDF:** [arroyo2025bridging.pdf](../../../../../literature/papers/arroyo2025bridging.pdf) — 32 pages; SHA-256 `85109ec389dd25440e498ca3d34953f45c6393b6a6fcd3044ae918be34e2c8aa`.

<a id="bordelon2025"></a>
## Dynamics of learning to integrate in linear recurrent neural networks

**bordelon2025** — Blake Bordelon; Jordan Cotler; Cengiz Pehlevan; Jacob A. Zavatone-Veth. 2025. arXiv preprint; version 2 revised June 2026.

[Primary source](https://arxiv.org/abs/2503.18754)

**Themes:** learning slow modes, integrators, oscillatory filters. **Evidence:** abstract; arXiv abstract and version history (24 March 2025; 8 June 2026)..

**Relevance:** Analyzes learning of long integration timescales through recurrent spectral outliers and extends to damped oscillatory filters.

**Novelty boundary:** Learning neutral/slow or oscillatory memory modes has recent analytic treatment; the project should add inference-specific causal evidence.

**Limits:** Analytically restricted linear networks; DOI/journal venue not verified. Full derivations not reviewed.

**Local PDF:** [bordelon2025.pdf](../../../../../literature/papers/bordelon2025.pdf) — 26 pages; SHA-256 `f8693715c6c760f5fc65489b0a0762fb4cab15400072470e37aa1ed539fd101f`.

<a id="dizaji2025tgrab"></a>
## T-GRAB: A Synthetic Diagnostic Benchmark for Learning on Temporal Graphs

**dizaji2025tgrab** — Dizaji, Alireza; Tjandra, Benedict Aaron; Hamidi, Mehrab; Huang, Shenyang; Rabusseau, Guillaume. 2025. MLoG-GenAI Workshop at KDD 2025; arXiv v2.

[Primary source](https://arxiv.org/abs/2507.10183)

**Themes:** graph benchmark. **Evidence:** full_text; Sections 3.1–3.3; Figures 3–7; Appendix E.

**Relevance:** Direct project predecessor: periodic counting, delayed causal and long-range graph tasks.

**Novelty boundary:** A new periodic graph benchmark or GC-LSTM periodicity observation is already covered.

**Limits:** Next-graph performance does not establish independently cued memory loss.

**Local PDF:** [dizaji2025tgrab.pdf](../../../../../literature/papers/dizaji2025tgrab.pdf) — 19 pages; SHA-256 `16e117520457a0541efa729cc4cbc06e851833191b72f4f896b011c0e265ff25`.

<a id="gated_delta"></a>
## Gated Delta Networks: Improving Mamba2 with Delta Rule

**gated_delta** — Songlin Yang; Jan Kautz; Ali Hatamizadeh. 2025. ICLR 2025.

[Primary source](https://arxiv.org/abs/2412.06464)

**Themes:** recurrence, gating, matrix_memory, length. **Evidence:** abstract; Primary abstract; author publication page.

**Relevance:** Combines adaptive erasure and targeted delta updates, with retrieval and extrapolation tests.

**Novelty boundary:** Medium for memory loss mechanisms.

**Limits:** Gated DeltaNet, hybrid variants and ungated DeltaNet must not be conflated.

**Local PDF:** [gated_delta.pdf](../../../../../literature/papers/gated_delta.pdf) — 22 pages; SHA-256 `55f84f2ae9c4e52ff494bfa699499867f5e9e17514994ac71de43888363a5fb9`.

<a id="guan2025-noise-memory"></a>
## How noise affects memory in linear recurrent networks

**guan2025-noise-memory** — JingChuan Guan; Tomoyuki Kubota; Yasuo Kuniyoshi; Kohei Nakajima. 2025. Physical Review Research 7:023049.

[Primary source](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.7.023049) · [DOI](https://doi.org/10.1103/PhysRevResearch.7.023049)

**Themes:** noise covariance, linear readout memory, correlated noise. **Evidence:** full_text; Main paper pp. 1–2, linear ESN and memory-function definitions; covariance derivation and large-dimension noise-spectrum discussion..

**Relevance:** Derives memory effects of noise spectra and injection structure in linear recurrent systems.

**Novelty boundary:** Noise-resolved memory geometry is established; equal raw noise amplitude need not mean equal memory impairment.

**Limits:** Assumptions include independent signal/noise and shared injection weights; asymptotic claims are not universal finite-state or gated-network guarantees.

**Reviewed version:** `2409.03187v1` (the first PDF candidate is pinned to this version).

**Local PDF:** [guan2025-noise-memory.pdf](../../../../../literature/papers/guan2025-noise-memory.pdf) — 28 pages; SHA-256 `b1832bdb03a49a9e057606a8ef1327a3ce33f08dd44ca56bd500535741c46e43`.

<a id="howlett2025-repeated-presentations"></a>
## Sudden restructuring of memory representations in recurrent neural networks with repeated stimulus presentations

**howlett2025-repeated-presentations** — Jonathon R. Howlett. 2025. Frontiers in Computational Neuroscience 19:1601641.

[Primary source](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1601641/full) · [DOI](https://doi.org/10.3389/fncom.2025.1601641)

**Themes:** repeated stimuli, attractor basins, Hebbian plasticity. **Evidence:** full_text; Publisher full text, Methods: Hebbian training, 10-neuron model, enumeration of 1024 states and repeated presentations..

**Relevance:** Repeated presentations alter basin organization and memory representations in a binary recurrent network.

**Novelty boundary:** A title-level repetition/restructuring claim overlaps, but the causal setting differs materially.

**Limits:** Weights change during repeated training; this is not fixed-weight inference-time periodic forgetting. Published middle initial not expanded without verification.

**Local PDF:** [howlett2025-repeated-presentations.pdf](../../../../../literature/papers/howlett2025-repeated-presentations.pdf) — 9 pages; SHA-256 `e8dfbaca90d19b2017061cb2be8ded832541b372b330304bda859b286cae8cf7`.

<a id="huginn"></a>
## Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach

**huginn** — Jonas Geiping; Sean McLeish; Neel Jain; John Kirchenbauer; Siddharth Singh; Brian R. Bartoldson; Bhavya Kailkhura; Abhinav Bhatele; Tom Goldstein. 2025. arXiv preprint; final venue not reverified.

[Primary source](https://arxiv.org/abs/2502.05171)

**Themes:** depth_recurrence, collapse, reinjection, cache. **Evidence:** full_text; arXiv v1 Secs.3.1–3.3,4.3,6.3–7; Fig.5.

**Relevance:** Huginn documents recurrence-induced token collapse and state-ignoring training failures.

**Novelty boundary:** High for generic recurrent-transformer collapse diagnosis.

**Limits:** Failure evidence concerns training/initialization; per-loop cache policy and repeated input change state and sensitivity.

**Reviewed version:** `2502.05171v1` (the first PDF candidate is pinned to this version).

**Copy/version note:** Inspected/downloaded arXiv v1 is pinned; its PDF title page agrees with the canonical title. Current unversioned arXiv URLs resolve to v2.

**Local PDF:** [huginn.pdf](../../../../../literature/papers/huginn.pdf) — 38 pages; SHA-256 `fff93f44e72cfda55f89fc0424b5e0fe297ef9522efecb7562894bbd544ce49e`.

<a id="looped_expressive_timestep"></a>
## On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

**looped_expressive_timestep** — Kevin Xu; Issei Sato. 2025. ICML 2025.

[Primary source](https://arxiv.org/abs/2410.01405)

**Themes:** depth_recurrence, expressivity, step_encoding. **Evidence:** abstract; ICML proceedings abstract.

**Relevance:** Approximation analysis motivates loop-dependent scaling through timestep encoding.

**Novelty boundary:** Medium for interpreting loop-count effects.

**Limits:** Loop-index conditioning changes the map; approximation power does not prove memory preservation.

**Local PDF:** [looped_expressive_timestep.pdf](../../../../../literature/papers/looped_expressive_timestep.pdf) — 34 pages; SHA-256 `4293225dd214f98f0afebf55e52b426dd9af5544100ac4834de9376190872fbf`.

<a id="mamba_token_dynamics"></a>
## Demystifying the Token Dynamics of Deep Selective State Space Models

**mamba_token_dynamics** — Thieu N. Vo; Duy-Tung Pham; Xin T. Tong; Tan Minh Nguyen. 2025. ICLR 2025.

[Primary source](https://arxiv.org/abs/2410.03292)

**Themes:** ssm, depth_dynamics, collapse. **Evidence:** full_text; arXiv v2 Secs.3–5, Theorems4.1/4.3/4.4; ICLR paper.

**Relevance:** Analyzes deep selective-state-space token convergence and performance interventions.

**Novelty boundary:** High for new SSM token-dynamics claims.

**Limits:** Main theory is a scalar S6 depth-continuum surrogate; not exact full-state temporal periodic recurrence. Primary sources use author-name variants.

**Local PDF:** [mamba_token_dynamics.pdf](../../../../../literature/papers/mamba_token_dynamics.pdf) — 27 pages; SHA-256 `38bac41fcc904e6e7394c3a0209f00079b774ed00c4a292b23cdf6db78ba42fd`.

<a id="proca2025-linear-learning"></a>
## Learning dynamics in linear recurrent neural networks

**proca2025-linear-learning** — Alexandra Maria Proca; Clémentine Carla Juliette Dominé; Murray Shanahan; Pedro A. M. Mediano. 2025. ICML; Proceedings of Machine Learning Research 267:49860–49902.

[Primary source](https://proceedings.mlr.press/v267/proca25a.html)

**Themes:** linear recurrent learning, temporal features, training confounds, extrapolation, learning dynamics, recency bias, memory regularization. **Evidence:** abstract; Official PMLR abstract and complete author metadata..

**Relevance:** Analyzes learning dynamics, temporal structure, and recurrence/feedthrough tradeoffs.

**Novelty boundary:** Differences in acquired recurrence or learned memory are already studied separately from inference trajectories.

**Limits:** Linear learning setting; relevance to fixed-weight nonlinear matched-driver effects requires a separate derivation. Published initials preserved where expansion unverified.

**Local PDF:** [proca2025-linear-learning.pdf](../../../../../literature/papers/proca2025-linear-learning.pdf) — 43 pages; SHA-256 `3d146fe63cb060af168030d1c69d660db1d0890886ff5b602d6b1ea195deb96e`.

<a id="rwkv7"></a>
## RWKV-7 "Goose" with Expressive Dynamic State Evolution

**rwkv7** — Bo Peng; Ruichong Zhang; Daniel Goldstein; Eric Alcaide; Xingjian Du; Haowen Hou; Jiaju Lin; Jiaxing Liu; Janna Lu; William Merrill; Guangyu Song; Kaifeng Tan; Saiteja Utpala; Nathan Wilce; Johan S. Wind; Tianyi Wu; Daniel Wuttke; Christian Zhou-Zheng. 2025. arXiv preprint.

[Primary source](https://arxiv.org/abs/2503.14456)

**Themes:** recurrence, matrix_memory, state_tracking. **Evidence:** abstract; Primary abstract.

**Relevance:** Dynamic delta-rule state evolution is relevant to persistent categorical memory.

**Novelty boundary:** Medium against blanket recurrent incapacity.

**Limits:** RWKV-7 differs from RWKV-4 used in routing-collapse work; implementation and state accounting need separate review.

**Local PDF:** [rwkv7.pdf](../../../../../literature/papers/rwkv7.pdf) — 56 pages; SHA-256 `f01a33393cb2cbd3a0147047d221aa304036e56b047990225539774af89d7418`.

<a id="yi2025tgbseq"></a>
## TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics

**yi2025tgbseq** — Yi, Lu; Peng, Jie; Zheng, Yanping; Mo, Fengran; Wei, Zhewei; Ye, Yuhang; Zixuan, Yue; Huang, Zengfeng. 2025. arXiv preprint; publication venue not independently checked.

[Primary source](https://arxiv.org/abs/2502.02975)

**Themes:** graph benchmark. **Evidence:** abstract; Primary arXiv abstract.

**Relevance:** Sequential temporal graph benchmark reduces easy edge-repeatability shortcuts.

**Novelty boundary:** Long-history evaluation and repetition confounds have benchmark precedents.

**Limits:** Natural sequential prediction differs from controlled cue retention.

**Local PDF:** [yi2025tgbseq.pdf](../../../../../literature/papers/yi2025tgbseq.pdf) — 20 pages; SHA-256 `2c03907ddb588ebd6453adf42091f8f779cc3c1478d4fedb381fd4766b64bcfe`.

<a id="all_routes_collapse"></a>
## All Routes Lead to Collapse

**all_routes_collapse** — K. R. Balasubramanian. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2606.22325)

**Themes:** attention, ssm, recurrence, graph, collapse. **Evidence:** full_text; arXiv v1 Secs.3–7; Appendices B.2–B.5,C.

**Relevance:** Claims shared low-rank routing signature with Mamba/RWKV interventions.

**Novelty boundary:** Very high for architecture-general collapse.

**Limits:** Key-cloud rank is not task memory. Appendix B.4 token-permutation null is spectrum-invariant as written; code unverified. Source author given only as K. R. Balasubramanian.

**Local PDF:** [all_routes_collapse.pdf](../../../../../literature/papers/all_routes_collapse.pdf) — 20 pages; SHA-256 `913eb70fa4b1fe0a0e48d4264cc9786c57c59bb7bba2f76b868b9cb09ebbce1d`.

<a id="chehreghani2026noise"></a>
## Persistent Gaussian Perturbations Prevent Oversmoothing in Recurrent Graph Neural Networks

**chehreghani2026noise** — Chehreghani, Mostafa Haghir. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2607.28185)

**Themes:** graph collapse. **Evidence:** full_text; Definitions 3.2–3.3; Sections 4–7.

**Relevance:** Persistent additive Gaussian perturbations sustain nonzero stationary graph Dirichlet energy.

**Novelty boundary:** Noise-maintained geometric diversity under contraction is not a new theorem direction.

**Limits:** The stated guarantee is geometric; it explicitly does not establish downstream memory or accuracy.

**Local PDF:** [chehreghani2026noise.pdf](../../../../../literature/papers/chehreghani2026noise.pdf) — 28 pages; SHA-256 `9c0d5ada45b58bdce2337f234df75d70bb6e8ac1c89d393ef5fa11e497abdf56`.

<a id="collapse_symmetry_rank"></a>
## Rank, Head-Channel Non-Identifiability, and Symmetry Breaking: A Precise Analysis of Representational Collapse in Transformers

**collapse_symmetry_rank** — Giansalvo Cirrincione. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2604.23681)

**Themes:** attention, rank, residual, normalization. **Evidence:** abstract; Primary abstract and introduction.

**Relevance:** Recent preprint distinguishes algebraic rank, residual effects and head identifiability.

**Novelty boundary:** Medium: challenges loose collapse definitions.

**Limits:** Abstract/intro claims only checked; exact-rank and LayerNorm assertions require proof audit and must not be treated as effective-rank guarantees.

**Local PDF:** [collapse_symmetry_rank.pdf](../../../../../literature/papers/collapse_symmetry_rank.pdf) — 36 pages; SHA-256 `a8ae734f7918f65ac3d45731b8289acf80f4adf848e7bf48fbee30407bfa11aa`.

<a id="dahmen2026dimensionality"></a>
## Strong and localized recurrence controls the dimensionality of neural activity across brain areas

**dahmen2026dimensionality** — David Dahmen; Stefano Recanatesi; Xiaoxuan Jia; Gabriel K. Ocker; Nilufar Lahiji; Simon Musall; Luke Campagnola; Stephanie Seeman; Tim Jarsky; Moritz Helias; Eric Shea-Brown. 2026. Nature Neuroscience.

[Primary source](https://www.nature.com/articles/s41593-026-02395-w) · [DOI](https://doi.org/10.1038/s41593-026-02395-w)

**Themes:** neuroscience, recurrence, dimensionality. **Evidence:** abstract; Publisher/PubMed abstract and Crossref metadata; online 2026-08-19..

**Relevance:** Recent biological recurrence-dimensionality comparison across brain areas.

**Novelty boundary:** Broad recurrence controls dimension framing is already active and needs acknowledgment.

**Limits:** No audited proof or trained-AI memory result; publisher access can be restricted.

**Copy/version note:** Author-hosted 34-page manuscript dated 2026-08-13, labelled Nature Neuroscience submission; metadata refers to published article 2026-08-19. Not the publisher typeset PDF.

**Local PDF:** [dahmen2026dimensionality.pdf](../../../../../literature/papers/dahmen2026dimensionality.pdf) — 34 pages; SHA-256 `17b3e7120f874fdbbc30f3b579d0b6623e4f22f8c3a3e8386ddfbe52dc3e0bce`.

<a id="fixed_point_reasoners"></a>
## Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers

**fixed_point_reasoners** — Sajad Movahedi; Vera Milovanović; Shlomo Libo Feigin; Alexander Theus; Thomas Hofmann; Valentina Boeva; T. Konstantin Rusch; Antonio Orvieto. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2606.18206)

**Themes:** depth_recurrence, equilibrium, normalization. **Evidence:** full_text; arXiv v1 Introduction and Secs.2–3.

**Relevance:** Convergence-based adaptive halting and residual scaling give competing successful dynamics.

**Novelty boundary:** High against useful computation requiring nonconvergence.

**Limits:** Input-conditioned puzzle reasoning is not delayed streaming recall; main design sections only, not full theorem audit.

**Local PDF:** [fixed_point_reasoners.pdf](../../../../../literature/papers/fixed_point_reasoners.pdf) — 32 pages; SHA-256 `169dc66aca853f5e8d922a170eac8421e78e1829d0d39025976a01915ff11c3d`.

<a id="fokam2026-temporal-correlations"></a>
## How Temporal Correlations Shape Memory in Linear Recurrent Neural Networks

**fokam2026-temporal-correlations** — Arnol Manuel Fokam; Fasseu Sieyondji Akpevwoghene; Edem Fiifi Dawson. 2026. arXiv preprint, v1 submitted 2026-08-31.

[Primary source](https://arxiv.org/abs/2609.00420)

**Themes:** temporal correlation, linear recurrent learning, memory cost, feedthrough, correlated inputs, learning dynamics, memory transition, periodicity. **Evidence:** full_text; HTML v1 Mathematical Setup and Findings; Appendix C, step 6, lag-one boundary derivation..

**Relevance:** Studies gradient-flow learning and memory/shortcut tradeoffs with stationary temporally correlated inputs.

**Novelty boundary:** Temporal correlations changing learned recurrence and memory cost are current explicit prior art.

**Limits:** Whitened feature covariance, aligned singular modes, geometric target profiles and linear learning dynamics; the local boundary result does not eliminate all higher-lag effects.

**Reviewed version:** `2609.00420v1` (the first PDF candidate is pinned to this version).

**Local PDF:** [fokam2026-temporal-correlations.pdf](../../../../../literature/papers/fokam2026-temporal-correlations.pdf) — 21 pages; SHA-256 `0233261ff348a80ce30bd88c8fa87fc402728516c0097ccc771f80c667668432`.

<a id="looped_stability_recall"></a>
## Stability and Generalization in Looped Transformers

**looped_stability_recall** — Asher Labovich. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2604.15259)

**Themes:** depth_recurrence, fixed_points, reinjection. **Evidence:** full_text; arXiv v1 Secs.3–5, Props.1–5; Appendices A.3,A.6–A.8.

**Relevance:** Autonomous versus recall analysis includes fixed-point input sensitivity.

**Novelty boundary:** Very high for proposed continuous-memory or reinjection theory.

**Limits:** Smooth transversality assumptions exclude designed neutral manifolds; isolated fixed points can retain categorical labels by basin.

**Reviewed version:** `2604.15259v1` (the first PDF candidate is pinned to this version).

**Local PDF:** [looped_stability_recall.pdf](../../../../../literature/papers/looped_stability_recall.pdf) — 27 pages; SHA-256 `cc4ad4b1c112dcc3dac4ff30fb9516dcc106229795a6faa38ec4b7d9e58671da`.

<a id="loopus"></a>
## LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models

**loopus** — Taekhyun Park; Yongjae Lee; Dohee Kim; Hyerim Bae. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2605.11011)

**Themes:** depth_recurrence, collapse, gating. **Evidence:** full_text; arXiv v1 Secs.3.1,4.5–4.6; Eqs.3–8.

**Relevance:** Retrofitted latent loops use gating, detached supervision and adaptive stopping.

**Novelty boundary:** High for gated loop stabilization.

**Limits:** Convex interpolation does not prove contraction; inference cache semantics need implementation verification.

**Local PDF:** [loopus.pdf](../../../../../literature/papers/loopus.pdf) — 30 pages; SHA-256 `7ac2312be838315b4c5486cebaa61cd0063374afd7eccb58a3f9ebdc13944a36`.

<a id="mesh"></a>
## MeSH: Memory-as-State-Highways for Recursive Transformers

**mesh** — Chengting Yu; Xiaobo Shu; Yadao Wang; Yizhen Zhang; Haoyi Wu; Jiaang Li; Rujiao Long; Ziheng Chen; Yuchi Xu; Wenbo Su; Bo Zheng. 2026. ICLR 2026.

[Primary source](https://arxiv.org/abs/2510.07739)

**Themes:** depth_recurrence, memory, collapse, step_routers. **Evidence:** full_text; arXiv v1 Secs.2,3.2,4.1; Figs.1,5. Acceptance verified by arXiv v2 comment; v2 Sec.3.2/Figs.1,5 also checked..

**Relevance:** Direct recursive hidden-state singular-spectrum collapse diagnosis and explicit-memory mitigation.

**Novelty boundary:** Very high; closest architecture-specific dimensional-collapse prior.

**Limits:** Top-50 normalized token singular values omit full-state history geometry; step-specific routers change the map and buffer changes state budget.

**Reviewed version:** `2510.07739v1` (the first PDF candidate is pinned to this version).

**Local PDF:** [mesh.pdf](../../../../../literature/papers/mesh.pdf) — 24 pages; SHA-256 `68ef08381f33a314ecef83baa439de58b106f9c5ae4e4ea32cd48811a0b7d1d2`.

<a id="sharma2026-temporal-superposition"></a>
## Temporal superposition and feature geometry of RNNs under memory demands

**sharma2026-temporal-superposition** — Pratyaksh Sharma; Alexandra Maria Proca; Lucas Prieto; Pedro A. M. Mediano. 2026. International Conference on Learning Representations.

[Primary source](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bfcb583d225b1db8d3ca2331f6785774-Abstract-Conference.html)

**Themes:** temporal superposition, feature geometry, interference, useful forgetting, memory geometry, forgetting. **Evidence:** full_text; Sections 3–3.4, pp. 3–5; section 4.1, pp. 5–6; Appendix I.2 for untied-readout qualification..

**Relevance:** Analyzes delayed recall using age-indexed features, readout geometry, and interference costs.

**Novelty boundary:** Memory-demand-dependent geometry and beneficial removal of obsolete information are already directly analyzed.

**Limits:** Specified recall/input models, including temporally independent inputs in the loss decomposition; does not establish the matched periodic-driver contrast. Unverified middle-name expansions retained as published initials.

**Local PDF:** [sharma2026-temporal-superposition.pdf](../../../../../literature/papers/sharma2026-temporal-superposition.pdf) — 42 pages; SHA-256 `855119eb39ef3a4820e2c9d2512d68ed1322b0699301933d6d85f13eb5ac3ba5`.

<a id="think_shallow_solve_deep"></a>
## Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth

**think_shallow_solve_deep** — Ivan Viakhirev; Kirill Borodin; Amirah Almutairi; Serguei Barannikov; Maxim Abramov; Grach Mkrtchian. 2026. arXiv preprint.

[Primary source](https://arxiv.org/abs/2608.18222)

**Themes:** depth_recurrence, lyapunov, dimension, non_normality. **Evidence:** full_text; arXiv v1 Secs.3.4–3.5,4,6; Appendices E,G,N.

**Relevance:** Controlled recurrent reasoners relate finite-time dynamics and task extrapolation.

**Novelty boundary:** Very high for spectra-plus-task generality.

**Limits:** Top-eight exponents and projected non-normality are partial estimates; input remains available each step.

**Local PDF:** [think_shallow_solve_deep.pdf](../../../../../literature/papers/think_shallow_solve_deep.pdf) — 16 pages; SHA-256 `1616b7bd075b8001dc4cbdc80012599f0aadf536a86f6c97d2ba81dae3a2467d`.
