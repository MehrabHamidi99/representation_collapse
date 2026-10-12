# Competence diagnostic constructors and oracle

The additive `src/representation_collapse/diagnostics.py` module supplies
the local fixtures needed to start the competence-recovery ladder. It does
not change the frozen v1 training or periodicity configurations.

- `paired_batch_with_zero` adds R0: sixteen WRITE observations and one QUERY,
  identical paired arms and no distractor body. At R≥1 it delegates to the
  existing constructor, preserving its outputs and random-number consumption.
- `single_write_episode` emits the addressed WRITE followed immediately by
  its QUERY, with either a specified required coordinate or a sampled one.
- `streaming_register_oracle` stores eight required-coordinate values, ignores
  nuisance writes and distractors, and applies last-WRITE-wins semantics.
  It rejects unwritten targets and malformed or leaking observations.
- `shuffle_memories` permutes WRITE payloads; `shuffle_queries` changes each
  terminal query to a different required address. Both preserve input dtype,
  leave the caller's array unchanged and return new oracle-consistent targets.
  Query shuffling requires all eight required registers to have been written.
- Baseline helpers report uniform categorical cross-entropy, sample
  zero-predictor MSE and MSE normalized by the fixed population variance 1/3.

Example, using the repository's existing environment:

```python
import numpy as np
from representation_collapse.diagnostics import (
    paired_batch_with_zero, streaming_register_oracle,
)

batch = paired_batch_with_zero(np.random.default_rng(17), 128, 0, "continuous")
predictions = streaming_register_oracle(batch.inputs, "continuous")
np.testing.assert_array_equal(predictions, batch.targets)
```

Run the focused regression tests with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_diagnostics.py -v
```

These are explicit-memory reference fixtures, not learned-model competence.
For corruption controls compare the new predictions with the **original**
targets; compare with the returned new targets to check oracle consistency.
A random permutation can be the identity, repeated categorical values can
match, and different query addresses can hold equal values. Therefore no
per-episode disagreement guarantee is valid. Regression fixtures use known
distinct values and fixed seeds to test the intended changes deterministically.

The module validates exact type/address flags, payload encoding and a single
terminal QUERY. It supports repeated and interleaved WRITEs for streaming
checks, but is not a training-protocol validator. Stage-aware training config,
coordinate-wise competence gates, three-phase state capture, causal
interventions/shams and seed-level statistical analysis remain separate
Gate B implementation work. See the [design](experimental-design-v2.md) and
[implementation audit](implementation-audit.md).
