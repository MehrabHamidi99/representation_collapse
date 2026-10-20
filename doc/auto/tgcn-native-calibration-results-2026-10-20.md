# T-GCN native synthetic calibration — 2026-10-20

**The dedicated hidden-only T-GCN backend passed native synthetic engineering
calibration.** Mila job **11332092** completed `0:0` in **38 seconds** on
`cn-f002`, using one CPU, 2 GiB and zero GPUs through cluv. Its immutable source
was `e0e6c20a72a98134aa1f2a2450f9eeb011718ab2`.

Independent compact verification accepted **248 checks** and **161 exact tensor
comparisons**, with reported maximum absolute error **0.0** and byte equality.
These checks establish implementation readiness in the pinned runtime. They do
not constitute a trained graph benchmark, recovered historical checkpoint,
Figure 10 behavioral reproduction, representation-collapse finding, causal
memory test or Floquet analysis. No empirical dataset or held-out split was
opened by this calibration.

The [machine-readable summary](tgcn-native-calibration-results-2026-10-20.json)
and [exact-byte evidence inventory](evidence/tgcn-native-calibration-2026-10-20/inventory.json)
preserve the original compact result, acceptance, scheduler, supervisor,
reporting and source-review records. The
[frozen protocol](tgcn-native-calibration-protocol-2026-10-20.md) and
[source-readiness audit](tgcn-reconstruction-readiness-2026-10-20.md) define the
engineering and historical-provenance boundaries.

## Native checks that passed

The tiny fixture uses four nodes, one-hot features and hidden width three.
Actual upstream `MultiLayerTGCN` and the dedicated backend were compared with
independent direct calls to the pinned dependency `TGCN`; this is not the
batched `TGCN2` class. Topology changes A → B → A, nonuniform symmetric C edge
weights, and an empty-edge feature-permutation control checked graph and weight
wiring. Explicit flags preserve `add_self_loops=False`, `cached=False`,
`improved=False`, normalized source-to-target graph convolution, and empty
normalized-adjacency caches.

Complete persistent state is **hidden only**, with canonical node ordering.
Checks covered positive-zero bytes, carried state, detached independent storage,
unchanged caller inputs, and passive clone mutation leaving live state and
scores unchanged. No artificial LSTM cell tensor was introduced.

The training calibration first warms on A at time 0 without gradients, detaches,
then consumes B at time 1 with a tracked nonzero hidden predecessor. It scores
target D at time 2 before consumption, using fixed source-preserving negatives.
Both branches used the real Adam optimizer and matched loss, gradients,
parameter updates, parameter groups and optimizer states exactly. **Four reset-
gate gradient tensors were nonzero**, so the reset path was exercised.

The training branch deliberately fills both heads' parameters with **0.1** to
keep the tiny ReLU path active and pairs dropout with Torch seed **73**. This
predeclared synthetic override is separate from earlier evaluation parity with
the untouched seed-0 head; it is not a proposed empirical training setting.
Python, NumPy and Torch were seeded at zero for two independent constructions;
initial component bytes and post-construction Torch RNG matched. The seeds were
then reset before the main fixture. Six model sets were constructed, including
the independent reference, fresh strict-load recipient and production inventory.

Both native parameter components passed strict preflight, missing-head rejection
without partial mutation, exact readback, and fresh zero-state prefix replay.
The actual production **100-node, width-128** inventory contains **137,472
recurrent parameters** and **49,537 link-head parameters**. Names, shapes,
CPU FP32 dtype, tensor sizes and native module-state versions are retained in
the compact result. This is native inventory evidence, extending the earlier
source-derived parameter count.

## Runtime, verification and reporting

The run used CPython **3.10.19**, Torch **2.0.0+cpu**, PyG **2.6.1** and
PyG Temporal **0.54.0**, with one intra-op thread, one inter-op thread and
deterministic algorithms enabled. The exact dependency and extension versions,
wrapper/cell hashes, initial component hashes and RNG hash are recorded in the
[result](evidence/tgcn-native-calibration-2026-10-20/calibration-result.json).
The installed `GCNConv` source matched a separately verified public wheel member
at SHA-256 `036b464769ad1d98501c9fc18ff3132b507890b2c5cb41b3e6f9e1f22d4f57f6`;
the [wheel proof](evidence/tgcn-native-calibration-2026-10-20/gcn-wheel-verification.json)
retains the wheel identity and immutable download URL.

Mandatory W&B reporting began before native construction and mathematics in
explicit offline mode. Independent verification decoded **one actual history
row**, checked its reported metrics, result artifact and successful terminal
framing, and verified source closure against Git. **Cloud delivery and remote
history remain unverified.** The independent verifier did not execute the
native models again: acceptance covers retained compact arithmetic/provenance
and the actual offline stream, not a second native numerical run.

The public evidence directory contains compact JSON only. It excludes W&B
streams, model weights, raw tensor/state arrays, datasets, optimizer payloads,
credentials and provider configuration. The prelaunch review remains an
unchanged record of what was known before this execution.

## Next step

Prepare a separately admitted **256-update empirical T-GCN prefix** using the
accepted conversion, corrected chronology, deterministic parameter order and
validation-only selection. Engineering calibration permits that preparation;
it does not show empirical competence or performance. A frozen fit/checkpoint
contract, accepted empirical prefix and separately controlled held-out
evaluation remain necessary before reporting another Figure 10 model curve.
