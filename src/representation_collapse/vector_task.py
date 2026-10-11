"""Latent WRITE/body/QUERY task. Body and query contain no memory values."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

MOTIFS = (
    "AXAXAXBXBXBX", "AXAXBXAXBXBX", "AXAXBXBXAXBX",
    "AXBXAXAXBXBX", "AXBXBXAXAXBX",
)
INPUT_DIM = 38  # type[5], coordinate[16], categorical value[16], real value[1]
WRITE, A, B, X, QUERY = range(5)
BODY_TYPES = {"A": A, "B": B, "X": X}


def proper_period(text: str) -> int | None:
    """Return a proper whole-string period (an exact tiling), if present."""
    for p in range(1, len(text) // 2 + 1):
        if len(text) % p == 0 and text == text[:p] * (len(text) // p):
            return p
    return None


def cyclic_bigrams(text: str) -> Counter:
    return Counter(zip(text, text[1:] + text[:1]))


def body_from_ids(ids: np.ndarray, q: int = 1) -> str:
    return "".join(c * q for i in ids for c in MOTIFS[int(i)])


def paired_body_ids(rng: np.random.Generator, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    final = int(rng.integers(len(MOTIFS)))
    periodic = np.full(repeats, final, dtype=np.int64)
    if repeats == 1:
        return periodic, periodic.copy()
    for _ in range(10000):
        control = rng.integers(len(MOTIFS), size=repeats, dtype=np.int64)
        control[-1] = final
        if proper_period(body_from_ids(control)) is None:
            return periodic, control
    raise RuntimeError("failed to construct varying nonperiodic motif body")


def sample_memory(rng: np.random.Generator, count: int, task: str) -> np.ndarray:
    if task == "categorical":
        return rng.integers(16, size=(count, 16), dtype=np.int64)
    if task == "continuous":
        return rng.uniform(-1, 1, size=(count, 16)).astype(np.float32)
    raise ValueError("task must be categorical or continuous")


def render(memory: np.ndarray, queries: np.ndarray, bodies: list[str], task: str) -> np.ndarray:
    """Render equal-length episodes; selector is revealed only at terminal query."""
    memory = np.asarray(memory)
    queries = np.asarray(queries)
    n = len(memory)
    if task not in {"categorical", "continuous"}:
        raise ValueError("unknown task")
    if memory.shape != (n, 16) or queries.shape != (n,) or len(bodies) != n:
        raise ValueError("inconsistent episode shapes")
    if not n or len(set(map(len, bodies))) != 1 or np.any((queries < 0) | (queries >= 8)):
        raise ValueError("nonempty equal-horizon bodies and query coordinates 0..7 required")
    if not np.isfinite(memory).all():
        raise ValueError("finite memory values required")
    if task == "categorical" and (np.any(memory != memory.astype(np.int64)) or np.any((memory < 0) | (memory >= 16))):
        raise ValueError("categorical memory values must be integers 0..15")
    if task == "continuous" and np.any(np.abs(memory) > 1):
        raise ValueError("continuous memory values must lie in [-1,1]")
    inputs = np.zeros((n, 17 + len(bodies[0]), INPUT_DIM), dtype=np.float32)
    rows = np.arange(n)
    for coordinate in range(16):
        inputs[:, coordinate, WRITE] = 1
        inputs[:, coordinate, 5 + coordinate] = 1
        if task == "categorical":
            inputs[rows, coordinate, 21 + memory[:, coordinate].astype(np.int64)] = 1
        else:
            inputs[:, coordinate, 37] = memory[:, coordinate]
    for row, body in enumerate(bodies):
        for t, symbol in enumerate(body):
            inputs[row, 16 + t, BODY_TYPES[symbol]] = 1
    inputs[:, -1, QUERY] = 1
    inputs[rows, -1, 5 + queries] = 1
    return inputs


@dataclass
class PairedBatch:
    inputs: np.ndarray
    targets: np.ndarray
    memory: np.ndarray
    queries: np.ndarray
    periodic_ids: np.ndarray
    control_ids: np.ndarray
    repeats: int
    q: int


def paired_batch(rng: np.random.Generator, pairs: int, repeats: int, task: str, q: int = 1) -> PairedBatch:
    if pairs < 1 or q < 1:
        raise ValueError("positive pair count and dwell required")
    memory = sample_memory(rng, pairs, task)
    queries = rng.integers(8, size=pairs, dtype=np.int64)
    ids = [paired_body_ids(rng, repeats) for _ in range(pairs)]
    periodic_ids = np.stack([x[0] for x in ids])
    control_ids = np.stack([x[1] for x in ids])
    bodies = [body_from_ids(x, q) for x in periodic_ids] + [body_from_ids(x, q) for x in control_ids]
    inputs = render(np.concatenate([memory, memory]), np.concatenate([queries, queries]), bodies, task)
    values = memory[np.arange(pairs), queries]
    return PairedBatch(inputs, np.concatenate([values, values]), memory, queries, periodic_ids, control_ids, repeats, q)
