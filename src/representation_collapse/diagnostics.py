"""Competence-first ladder stages 0–2: R0 batches, single WRITE, streaming oracle.

Estimands here are renderer-faithful fixtures and an explicit-memory reference
with counted storage, not evidence about learned recurrent models. Thresholds
and episode geometry come from experimental-design-v2.md section 2 B.0 and
competence-recovery-plan.md steps 1–3. Additive wrapper: `vector_task.py` is
not modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vector_task import (
    INPUT_DIM,
    QUERY,
    WRITE,
    PairedBatch,
    paired_batch,
    paired_body_ids,
    render,
    sample_memory,
)

N_CATEGORICAL_CLASSES = 16
N_REQUIRED_COORDINATES = 8
N_WRITE_COORDINATES = 16
# Uniform[-1, 1] population second moment / variance. Design v2 B.0 and
# competence-recovery-plan.md fix this population value, not the sample variance.
POPULATION_VALUE_VARIANCE = 1.0 / 3.0
# Stage-0 continuous oracle tolerance. Design v2 B.0; competence-recovery-plan step 1.
ORACLE_CONTINUOUS_ATOL = 1e-12
REQUIRED_COORDINATES = tuple(range(N_REQUIRED_COORDINATES))


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _task(task: str) -> None:
    if task not in ("categorical", "continuous"):
        raise ValueError("task must be categorical or continuous")


@dataclass
class SingleWriteBatch:
    """Two-observation WRITE→QUERY diagnostic (design v2 B.0 stage 1).

    `inputs` has shape `(count, 2, 38)` in the `vector_task.render` layout.
    `targets` and `queries` are length `count`; the queried coordinate is the
    written required coordinate. No distractor body and no nuisance writes.
    """

    inputs: np.ndarray
    targets: np.ndarray
    queries: np.ndarray


def paired_body_ids_with_zero(
    rng: np.random.Generator, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    """Paired motif-id sequences, including the R0 empty-body case.

    For `repeats >= 1` this delegates to `vector_task.paired_body_ids` (same
    structure: two length-`repeats` int64 arrays). For `repeats == 0` it returns
    two empty int64 arrays of shape `(0,)`, which stack to `(pairs, 0)`.
    `vector_task.paired_body_ids` raises on `repeats < 1`; this wrapper is the
    additive R0 path required by design v2 section 7 item 1 and
    competence-recovery-plan.md step 3.
    """
    repeats = _integer(repeats, "repeats", 0)
    if repeats >= 1:
        return paired_body_ids(rng, repeats)
    empty = np.empty(0, dtype=np.int64)
    return empty, empty.copy()


def paired_batch_with_zero(
    rng: np.random.Generator, pairs: int, repeats: int, task: str, q: int = 1
) -> PairedBatch:
    """`PairedBatch` including R0 (empty bodies, 17-observation episodes).

    For `repeats >= 1` this is exactly `vector_task.paired_batch`. For
    `repeats == 0` the R0 fields that differ from the v1 path are:

    - `repeats` is 0; v1 `paired_body_ids` / `paired_batch` cannot produce this.
    - `periodic_ids` and `control_ids` have shape `(pairs, 0)` rather than
      `(pairs, repeats)`. No motif identities are drawn, so the RNG stream
      after `sample_memory` and the query draw is not consumed for bodies.
    - Bodies are empty strings, so `inputs` has shape `(2 * pairs, 17, 38)`
      (16 WRITE + terminal QUERY; competence-recovery-plan.md verified this
      shape). There are no A/B/X rows. `q` is recorded but stretches no motif.
    - Arms are identical by construction (shared empty body), as with the R1
      shared anchor. `memory`, `queries`, and duplicated `targets` keep v1
      semantics. Gate B uses `q=1` only (design v2 section 1.3).
    """
    pairs = _integer(pairs, "pairs", 1)
    repeats = _integer(repeats, "repeats", 0)
    q = _integer(q, "q", 1)
    _task(task)
    if repeats >= 1:
        return paired_batch(rng, pairs, repeats, task, q)
    memory = sample_memory(rng, pairs, task)
    queries = rng.integers(N_REQUIRED_COORDINATES, size=pairs, dtype=np.int64)
    ids = [paired_body_ids_with_zero(rng, 0) for _ in range(pairs)]
    periodic_ids = np.stack([item[0] for item in ids])
    control_ids = np.stack([item[1] for item in ids])
    bodies = [""] * (2 * pairs)
    inputs = render(
        np.concatenate([memory, memory]),
        np.concatenate([queries, queries]),
        bodies,
        task,
    )
    values = memory[np.arange(pairs), queries]
    return PairedBatch(
        inputs,
        np.concatenate([values, values]),
        memory,
        queries,
        periodic_ids,
        control_ids,
        repeats,
        q,
    )


def single_write_episode(
    rng: np.random.Generator,
    count: int,
    task: str,
    coordinate: int | None = None,
) -> SingleWriteBatch:
    """One required WRITE and its matching QUERY; no distractor, no nuisance.

    Design v2 B.0 stage 1 / competence-recovery-plan.md step 2. The original
    renderer always writes sixteen coordinates (`vector_task.render`); this
    diagnostic keeps the same 38-channel layout (type[5], coordinate[16],
    categorical value[16], real value[1]; QUERY is type+coordinate only) but
    emits two observations. WRITE/QUERY rows are taken from an equivalent
    `render` episode so they match that constructor exactly. The QUERY row
    must not carry a memory value (the v1 no-leakage invariant).
    """
    count = _integer(count, "count", 1)
    _task(task)
    if coordinate is not None:
        coordinate = _integer(coordinate, "coordinate", 0)
        if coordinate not in REQUIRED_COORDINATES:
            raise ValueError("coordinate must be a required index 0..7")
    memory = sample_memory(rng, count, task)
    if coordinate is None:
        queries = rng.integers(N_REQUIRED_COORDINATES, size=count, dtype=np.int64)
    else:
        queries = np.full(count, int(coordinate), dtype=np.int64)
    rendered = render(memory, queries, [""] * count, task)
    rows = np.arange(count)
    # Renderer writes coordinate k at time index k, so the matching WRITE row
    # is `rendered[i, queries[i]]`; the QUERY is the terminal row.
    inputs = np.stack([rendered[rows, queries], rendered[rows, -1]], axis=1)
    if np.any(inputs[:, 1, 21:] != 0):
        raise AssertionError("QUERY row leaked a memory value")
    if not np.all(inputs[:, 1, QUERY] == 1) or np.any(inputs[:, 1, :4] != 0):
        raise AssertionError("QUERY row must be type QUERY with no other type flags")
    return SingleWriteBatch(inputs, memory[rows, queries], queries)


def _one_hot(rows: np.ndarray) -> bool:
    return bool(np.all((rows == 0) | (rows == 1)) and np.all(rows.sum(axis=-1) == 1))


def _validated_layout(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact renderer channels before decoding discrete addresses.

    Streaming episodes may interleave WRITE and body rows, but must finish in
    one QUERY. Validation uses exact one-hot flags, not approximate argmax.
    """
    x = np.asarray(inputs)
    if x.ndim != 3 or x.shape[-1] != INPUT_DIM or min(x.shape[:2]) < 1:
        raise ValueError("inputs must have nonempty shape (episodes, time, 38)")
    if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
        raise ValueError("inputs must contain finite real numeric values")
    if not _one_hot(x[..., :5]):
        raise ValueError("each observation must have exactly one type flag")
    types = x[..., :5].argmax(axis=-1)
    if np.any(types[:, -1] != QUERY) or np.any(types[:, :-1] == QUERY):
        raise ValueError("each episode must contain exactly one terminal QUERY")
    addressed = (types == WRITE) | (types == QUERY)
    if not _one_hot(x[..., 5:21][addressed]):
        raise ValueError("WRITE and QUERY require exactly one coordinate flag")
    if np.any(x[..., 5:][~addressed] != 0):
        raise ValueError("body observations must not carry coordinates or values")
    if np.any(x[..., 21:][types == QUERY] != 0):
        raise ValueError("QUERY must not carry a memory value")
    coordinates = x[..., 5:21].argmax(axis=-1)
    if np.any(coordinates[:, -1] >= N_REQUIRED_COORDINATES):
        raise ValueError("QUERY coordinate must be a required index 0..7")
    return x, types, coordinates


def _validate_payloads(x: np.ndarray, types: np.ndarray, task: str) -> None:
    _task(task)
    write = types == WRITE
    if not np.any(write):
        raise ValueError("at least one WRITE observation required")
    categorical = x[..., 21:37][write]
    real = x[..., 37][write]
    if task == "categorical":
        if not _one_hot(categorical) or np.any(real != 0):
            raise ValueError("categorical WRITEs require one-hot values and zero real payload")
    elif np.any(categorical != 0) or np.any(np.abs(real) > 1):
        raise ValueError("continuous WRITEs require real values in [-1,1] and zero categorical payload")


def infer_task(inputs: np.ndarray) -> str:
    """Infer a homogeneous valid WRITE encoding, including all-zero real values.

    Mixed categorical/continuous payloads and malformed layouts are rejected.
    As with an explicit task argument, the oracle separately checks that the
    terminal queried register has actually been written.
    """
    x, types, _ = _validated_layout(inputs)
    write = types == WRITE
    task = "categorical" if np.any(x[..., 21:37][write] != 0) else "continuous"
    _validate_payloads(x, types, task)
    return task


def streaming_register_oracle(inputs: np.ndarray, task: str) -> np.ndarray:
    """Look up the QUERY address in eight explicit required-coordinate registers.

    Estimand: the value written to the queried required coordinate, ignoring
    distractor body observations (types A/B/X) and nuisance writes to
    coordinates 8–15. Storage is eight registers (last WRITE wins). This is
    an explicit-memory reference with counted storage, not evidence that a
    learned recurrent model acquired the task (design v2 B.0 stage 0;
    competence-recovery-plan.md step 1). On valid R0/R1/R2/R16 paired
    episodes and single-WRITE episodes it must recover categorical targets
    exactly and continuous targets with error below `ORACLE_CONTINUOUS_ATOL`.
    """
    x, types, coordinates = _validated_layout(inputs)
    _validate_payloads(x, types, task)
    n_episodes = x.shape[0]
    if task == "categorical":
        predictions = np.empty(n_episodes, dtype=np.int64)
    else:
        predictions = np.empty(n_episodes, dtype=np.float64)
    for i in range(n_episodes):
        registers: list[object] = [None] * N_REQUIRED_COORDINATES
        for t in range(x.shape[1]):
            kind = int(types[i, t])
            if kind == WRITE:
                coordinate = int(coordinates[i, t])
                if coordinate >= N_REQUIRED_COORDINATES:
                    continue
                if task == "categorical":
                    registers[coordinate] = int(x[i, t, 21:37].argmax())
                else:
                    registers[coordinate] = float(x[i, t, 37])
            elif kind == QUERY:
                coordinate = int(coordinates[i, t])
                if coordinate not in REQUIRED_COORDINATES:
                    raise ValueError("QUERY coordinate must be a required index 0..7")
                value = registers[coordinate]
                if value is None:
                    raise ValueError("QUERY addressed an unwritten required register")
                predictions[i] = value
    return predictions


def shuffle_memories(
    inputs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Permute WRITE value payloads across coordinates and episodes.

    Coordinate and type channels stay put; only channels `[21:38]` on WRITE
    rows are permuted. The oracle on the corrupted inputs is then consistent
    with the shuffled payloads and must disagree with the original targets
    whenever the permutation moves a distinct value onto a queried register
    (competence-recovery-plan.md step 1; design v2 B.0 stage 0). Returns the
    corrupted inputs and the new oracle-consistent targets.
    """
    task = infer_task(inputs)
    streaming_register_oracle(inputs, task)  # Reject unwritten targets before RNG draws.
    out = np.array(inputs, copy=True)
    write = out[..., :5].argmax(axis=-1) == WRITE
    payloads = out[..., 21:][write]
    # Include the identity permutation: forcing it away biases this control.
    order = rng.permutation(payloads.shape[0])
    out[..., 21:][write] = payloads[order]
    return out, streaming_register_oracle(out, task)


def shuffle_queries(
    inputs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Change each QUERY coordinate so the oracle reads a different register.

    The new address is drawn from the other required coordinates `0..7`. The
    oracle therefore selects the changed coordinate, not the old answer
    (competence-recovery-plan.md step 1). Returns the corrupted inputs and
    the new oracle-consistent targets. Limitation: the new coordinate must
    already have been written, so this control is for full-prefix episodes
    (all eight required WRITEs), not single-WRITE diagnostics.
    """
    task = infer_task(inputs)
    _, types, coordinates = _validated_layout(inputs)
    # This control is defined on full required-register prefixes; fail before
    # consuming RNG if even an unselected alternative register is unwritten.
    for i in range(types.shape[0]):
        written = set(coordinates[i, types[i] == WRITE])
        if not set(REQUIRED_COORDINATES).issubset(written):
            raise ValueError("query shuffle requires all eight required registers written")
    out = np.array(inputs, copy=True)
    for i in range(out.shape[0]):
        old = int(coordinates[i, -1])
        choices = np.asarray([c for c in REQUIRED_COORDINATES if c != old], dtype=np.int64)
        new = int(rng.choice(choices))
        out[i, -1, 5:21] = 0
        out[i, -1, 5 + new] = 1
    return out, streaming_register_oracle(out, task)


def uniform_categorical_cross_entropy(n_classes: int = N_CATEGORICAL_CLASSES) -> float:
    """Population cross-entropy of the uniform classifier, `log n_classes`.

    For 16 equiprobable categories this is `log 16` (natural log, matching
    softmax cross-entropy). Named reference from competence-recovery-plan.md
    and design v2 B.3 item 1; not a trained-model result.
    """
    n_classes = _integer(n_classes, "n_classes", 2)
    return float(np.log(n_classes))


def zero_predictor_mse(targets: np.ndarray) -> float:
    """Mean squared error of predicting 0, computed from the saved targets.

    For Uniform[-1, 1] the population value is `POPULATION_VALUE_VARIANCE`
    (`1/3`). The sample value is `mean(targets^2)` from the provided targets
    (design v2 B.3 item 1; competence-recovery-plan.md). This is a reference
    baseline, not evidence about a learned model.
    """
    values = np.asarray(targets)
    if values.dtype.kind not in "fiu":
        raise ValueError("targets must be real numeric values")
    values = values.astype(np.float64)
    if values.size < 1 or not np.isfinite(values).all():
        raise ValueError("finite nonempty targets required")
    return float(np.square(values).mean())


def normalized_mse(mse: float | np.ndarray) -> float:
    """Normalize raw MSE by the Uniform[-1, 1] population variance `1/3`.

    Design v2 B.0 and competence-recovery-plan.md require `MSE / (1/3)`, not
    division by the observed sample variance. The continuous competence
    threshold `0.05` is therefore raw MSE `1/60`.
    """
    scalar = np.asarray(mse)
    if scalar.ndim != 0 or scalar.dtype.kind not in "fiu":
        raise ValueError("mse must be a real numeric scalar")
    value = float(scalar)
    if not np.isfinite(value) or value < 0:
        raise ValueError("mse must be a finite nonnegative number")
    return value / POPULATION_VALUE_VARIANCE
