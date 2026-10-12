"""Additive, stage-aware native competence assays for the proposed Gate B.

No fitting or checkpoint selection happens here. Scores use decoded native
predictions *after* the terminal QUERY update. Full-prefix R0 still has sixteen
WRITEs and is distinct from the two-observation immediate-WRITE assay. Synthetic
oracle predictions validate this implementation, not learned-model competence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from .diagnostics import (
    N_CATEGORICAL_CLASSES, N_REQUIRED_COORDINATES, POPULATION_VALUE_VARIANCE,
    REQUIRED_COORDINATES, infer_task, paired_batch_with_zero, single_write_episode,
    streaming_register_oracle,
)
from .vector_task import A, B, X, WRITE, MOTIFS, proper_period

IMMEDIATE = "single_write_immediate_query"
R0 = "full_prefix_r0"
R1 = "full_prefix_r1"
R2 = "full_prefix_r2"
ASSAY_REPEATS = {IMMEDIATE: None, R0: 0, R1: 1, R2: 2}
NATIVE_PHASE = "after_terminal_query"


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _task(task: str) -> None:
    if task not in ("categorical", "continuous"):
        raise ValueError("task must be categorical or continuous")


@dataclass(frozen=True)
class FrozenCompetenceProtocol:
    """Fixed prospective contract; changing it requires a separately named module.

    A four-ULP float64 comparison allowance at the declared metric threshold
    handles arithmetic rounding at an inclusive boundary. It is fixed before
    scoring, reported explicitly and is unrelated to model/state tolerances.
    """

    protocol_id: str = field(default="gate-b-native-coordinate-v2", init=False)
    categorical_min_accuracy: float = field(default=0.95, init=False)
    continuous_max_normalized_mse: float = field(default=0.05, init=False)
    episodes_per_coordinate: int = field(default=128, init=False)
    comparison_ulps: int = field(default=4, init=False)

    def record(self) -> dict:
        return {
            "protocol_id": self.protocol_id,
            "categorical_min_accuracy": self.categorical_min_accuracy,
            "continuous_max_normalized_mse": self.continuous_max_normalized_mse,
            "continuous_raw_mse_threshold": self.continuous_max_normalized_mse * POPULATION_VALUE_VARIANCE,
            "normalization": "Uniform[-1,1] population variance; never observed sample variance",
            "population_value_variance": POPULATION_VALUE_VARIANCE,
            "episodes_per_coordinate_per_arm": self.episodes_per_coordinate,
            "comparison_ulps_float64": self.comparison_ulps,
            "phase": NATIVE_PHASE,
            "selection": "validation task loss only; short-horizon eligibility; earliest checkpoint then frozen LR order on ties",
            "selection_excludes": ["geometry", "arm contrasts", "held-out test scores"],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True, allow_nan=False).encode()).hexdigest()


PROTOCOL = FrozenCompetenceProtocol()


@dataclass(frozen=True)
class CompetenceBatch:
    """One assay manifest; paired arrays are periodic rows then control rows.

    Sampling seed identifies the evaluation stream, not an independent trained
    seed. Each full-task pair has the same memory and QUERY selector. R0/R1
    have identical arms. Immediate episodes have one arm and no arm contrast.
    """

    assay: str
    task: str
    inputs: np.ndarray
    targets: np.ndarray
    queries: np.ndarray
    pair_ids: np.ndarray
    sampling_seed: int


def make_competence_batch(assay: str, task: str, sampling_seed: int) -> CompetenceBatch:
    """Freeze 128 independent episode draws for each of the eight coordinates.

    The coordinate-stratified draws reuse the existing diagnostic constructors;
    full-prefix draws replace their random selectors with prespecified strata.
    The same stratum and memory draw are used in both arms. Different assay
    streams should be assigned and recorded by the eventual training protocol.
    """
    if assay not in ASSAY_REPEATS:
        raise ValueError("unknown competence assay")
    _task(task)
    sampling_seed = _integer(sampling_seed, "sampling_seed")
    rng = np.random.default_rng(sampling_seed)
    count = PROTOCOL.episodes_per_coordinate
    inputs, targets, queries = [], [], []
    controls = []
    for coordinate in REQUIRED_COORDINATES:
        if assay == IMMEDIATE:
            batch = single_write_episode(rng, count, task, coordinate)
            inputs.append(batch.inputs)
            targets.append(batch.targets)
            queries.append(batch.queries)
        else:
            batch = paired_batch_with_zero(rng, count, ASSAY_REPEATS[assay], task, q=1)
            x = batch.inputs.copy()
            x[:, -1, 5:21] = 0
            x[:, -1, 5 + coordinate] = 1
            inputs.append(x[:count])
            controls.append(x[count:])
            targets.append(batch.memory[:, coordinate])
            queries.append(np.full(count, coordinate, dtype=np.int64))
    x, y, q = np.concatenate(inputs), np.concatenate(targets), np.concatenate(queries)
    pairs = np.arange(len(y), dtype=np.int64)
    if assay != IMMEDIATE:
        x = np.concatenate((x, np.concatenate(controls)))
        y, q, pairs = np.tile(y, 2), np.tile(q, 2), np.tile(pairs, 2)
    for array in (x, y, q, pairs):
        array.setflags(write=False)
    result = CompetenceBatch(assay, task, x, y, q, pairs, sampling_seed)
    validate_competence_batch(result)
    return result


def validate_competence_batch(batch: CompetenceBatch) -> None:
    """Validate task, chronology, targets, paired identities and motif matching.

    Sample-count mismatch is a *failed gate*, not a malformed task: the scorer
    reports missing/undersampled coordinates explicitly instead of dropping them.
    """
    if not isinstance(batch, CompetenceBatch) or batch.assay not in ASSAY_REPEATS:
        raise ValueError("a known CompetenceBatch assay is required")
    _task(batch.task)
    _integer(batch.sampling_seed, "sampling_seed")
    oracle = streaming_register_oracle(batch.inputs, batch.task)
    if infer_task(batch.inputs) != batch.task:
        raise ValueError("task does not match WRITE payload encoding")
    x = np.asarray(batch.inputs)
    total = len(x)
    q, pair_ids, targets = (np.asarray(v) for v in (batch.queries, batch.pair_ids, batch.targets))
    for values, name in ((q, "queries"), (pair_ids, "pair_ids")):
        if values.shape != (total,) or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer vector matching episodes")
    if targets.shape != (total,) or targets.dtype.kind not in ("iu" if batch.task == "categorical" else "fiu"):
        raise ValueError("targets must be a task-compatible vector matching episodes")
    if not np.isfinite(targets).all() or not np.array_equal(targets, oracle):
        raise ValueError("targets must equal the renderer-faithful register oracle")
    coordinates = x[..., 5:21].argmax(axis=-1)
    if not np.array_equal(q, coordinates[:, -1]):
        raise ValueError("recorded queries must match terminal QUERY addresses")
    types = x[..., :5].argmax(axis=-1)
    if batch.assay == IMMEDIATE:
        if x.shape[1] != 2 or np.any(types[:, 0] != WRITE) or not np.array_equal(coordinates[:, 0], q):
            raise ValueError("immediate assay requires one matching WRITE followed immediately by QUERY")
        if not np.array_equal(pair_ids, np.arange(total)):
            raise ValueError("immediate episode IDs must be unique and ordered")
        return
    repeats = ASSAY_REPEATS[batch.assay]
    if total % 2 or x.shape[1] != 17 + 12 * repeats:
        raise ValueError("full-prefix assay must have two equal arms and its exact declared horizon")
    if np.any(types[:, :16] != WRITE) or not np.all(coordinates[:, :16] == np.arange(16)):
        raise ValueError("full-prefix assay requires the sixteen ordered WRITE coordinates")
    if np.any(~np.isin(types[:, 16:-1], [A, B, X])):
        raise ValueError("body may contain only A/B/X distractors")
    n = total // 2
    if not np.array_equal(pair_ids, np.tile(np.arange(n), 2)):
        raise ValueError("paired episode IDs must align periodic and control rows")
    if not np.array_equal(x[:n, :16], x[n:, :16]) or not np.array_equal(x[:n, -1], x[n:, -1]):
        raise ValueError("paired arms must share the complete WRITE prefix and terminal QUERY")
    if not np.array_equal(q[:n], q[n:]) or not np.array_equal(targets[:n], targets[n:]):
        raise ValueError("paired arms must share selectors and targets")
    if repeats == 0:
        return
    symbols = {A: "A", B: "B", X: "X"}
    for i in range(n):
        left = "".join(symbols[int(k)] for k in types[i, 16:-1])
        right = "".join(symbols[int(k)] for k in types[n + i, 16:-1])
        motif = left[:12]
        if motif not in MOTIFS or left != motif * repeats:
            raise ValueError("periodic arm must repeat a registered twelve-symbol motif")
        if repeats == 1:
            if right != left:
                raise ValueError("R1 is an identical-arm anchor")
        elif any(right[j:j + 12] not in MOTIFS for j in range(0, len(right), 12)) or right[-12:] != motif or proper_period(right) is not None:
            raise ValueError("control must use registered motifs, match the terminal motif and have no proper whole-body period")


def _batch_hash(batch: CompetenceBatch) -> str:
    digest = hashlib.sha256(json.dumps({"assay": batch.assay, "task": batch.task, "sampling_seed": int(batch.sampling_seed)}, sort_keys=True).encode())
    for value in (batch.inputs, batch.targets, batch.queries, batch.pair_ids):
        array = np.asarray(value)
        digest.update(json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _finite_statistic(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} overflowed or is nonfinite")
    return result


def _finite_mean(values: np.ndarray, name: str) -> float:
    # Finite individual errors do not guarantee a finite float64 reduction.
    # Reject unresolved overflow instead of returning Inf or clamping a score.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=np.float64).mean()
    return _finite_statistic(result, name)


def evaluate_competence(batch: CompetenceBatch, predictions: np.ndarray, *,
                        trained_seed: int, checkpoint_id: str,
                        measurement_phase: str, split: str = "validation") -> dict:
    """Score decoded native predictions with every required coordinate retained.

    Predictions are integer classes (not logits/probabilities) or finite real
    values. The supplied phase is an explicit caller assertion; equivalence of
    forked and uninterrupted native execution needs separate instrumentation.
    Test records are reportable but cannot make checkpoint selection eligible.
    """
    validate_competence_batch(batch)
    trained_seed = _integer(trained_seed, "trained_seed")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("nonempty immutable checkpoint identity is required")
    if measurement_phase != NATIVE_PHASE:
        raise ValueError("native competence requires predictions after the terminal QUERY update")
    if split not in ("validation", "test", "fixture"):
        raise ValueError("split must be validation, test or fixture")
    prediction = np.asarray(predictions)
    if prediction.shape != np.asarray(batch.targets).shape or prediction.dtype.kind not in ("iu" if batch.task == "categorical" else "fiu"):
        raise ValueError("predictions must be a task-compatible decoded vector matching targets")
    if not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite")
    if batch.task == "categorical" and np.any((prediction < 0) | (prediction >= N_CATEGORICAL_CLASSES)):
        raise ValueError("categorical predictions must be integer classes 0..15")
    threshold = PROTOCOL.categorical_min_accuracy if batch.task == "categorical" else PROTOCOL.continuous_max_normalized_mse
    allowance = float(PROTOCOL.comparison_ulps * np.spacing(np.float64(threshold)))
    if batch.task == "categorical":
        error = (prediction != batch.targets).astype(np.float64)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            error = np.square(prediction.astype(np.float64) - np.asarray(batch.targets, dtype=np.float64))
        if not np.isfinite(error).all():
            raise ValueError("squared prediction errors must be finite")
    labels = ("immediate",) if batch.assay == IMMEDIATE else ("periodic", "control")
    count = len(error) // len(labels)
    arm_records = {}
    for j, arm in enumerate(labels):
        start, end = j * count, (j + 1) * count
        rows = []
        for coordinate in REQUIRED_COORDINATES:
            mask = np.asarray(batch.queries)[start:end] == coordinate
            n = int(mask.sum())
            reasons = [] if n == PROTOCOL.episodes_per_coordinate else ["sample_count_mismatch"]
            mse_or_error = _finite_mean(error[start:end][mask], "coordinate mean error") if n else None
            metric = _finite_statistic(1 - mse_or_error if batch.task == "categorical" else mse_or_error / POPULATION_VALUE_VARIANCE,
                                       "coordinate metric") if n else None
            metric_passed = metric is not None and (metric >= threshold - allowance if batch.task == "categorical" else metric <= threshold + allowance)
            if not metric_passed:
                reasons.append("missing_coordinate" if not n else "below_accuracy_threshold" if batch.task == "categorical" else "above_normalized_mse_threshold")
            row = {"coordinate": coordinate, "episodes": n, "metric": metric, "metric_passed": bool(metric_passed),
                   "passed": not reasons, "failure_reasons": reasons}
            if batch.task == "continuous":
                row["raw_mse"] = mse_or_error
                target = np.asarray(batch.targets)[start:end][mask].astype(np.float64)
                row["sample_zero_predictor_mse"] = _finite_mean(np.square(target), "sample zero-predictor MSE") if n else None
            rows.append(row)
        aggregate_error = _finite_mean(error[start:end], "aggregate mean error")
        aggregate_metric = _finite_statistic(1 - aggregate_error if batch.task == "categorical" else aggregate_error / POPULATION_VALUE_VARIANCE,
                                             "aggregate metric")
        arm_records[arm] = {"episodes": count, "coordinates": rows,
            "aggregate_metric": aggregate_metric,
            "passed": all(row["passed"] for row in rows),
            "failed_coordinates": [row["coordinate"] for row in rows if not row["passed"]]}
    passed = all(row["passed"] for row in arm_records.values())
    paired_differences = None
    if len(labels) == 2:
        paired_differences = []
        for coordinate in REQUIRED_COORDINATES:
            mask = np.asarray(batch.queries)[:count] == coordinate
            paired_differences.append(_finite_mean((error[:count] - error[count:])[mask], "paired mean error difference")
                                      if np.any(mask) else None)
    return {"schema_version": 1, "kind": "native_coordinate_competence", "scientific_acceptance": False,
        "protocol": PROTOCOL.record(), "protocol_sha256": PROTOCOL.sha256, "assay": batch.assay, "task": batch.task,
        "trained_seed": trained_seed, "sampling_seed": int(batch.sampling_seed), "checkpoint_id": checkpoint_id,
        "batch_sha256": _batch_hash(batch), "split": split, "measurement_phase": measurement_phase,
        "metric_name": "coordinate_accuracy" if batch.task == "categorical" else "coordinate_population_normalized_mse",
        "comparison_allowance": allowance, "arms": arm_records, "coordinate_gate_passed": passed,
        "validation_eligibility_met": passed and split == "validation",
        "paired_episode_units": count if len(labels) == 2 else 0,
        "unique_episode_units": count, "trained_seed_units": 0 if split == "fixture" else 1,
        "fixture_only": split == "fixture",
        "statistical_unit": "synthetic reference fixture; no trained-seed evidence" if split == "fixture" else "one trained seed; paired episodes are within-seed evaluation draws",
        "paired_coordinate_error_differences": paired_differences,
        "paired_difference_units": "periodic minus control 0/1 error" if batch.task == "categorical" else "periodic minus control raw squared error; not normalized MSE",
        "aggregate_is_diagnostic_only": True,
        "interpretation": "R0 includes later required/nuisance WRITE updates; immediate acquisition is a separate assay" if batch.assay == R0 else "native behavior only; no geometry, tangent or causal-memory conclusion",
        "checkpoint_selection_implemented": False}


def assess_stage_competence(records: list[dict], ladder_stage: int) -> dict:
    """Check the coordinate-score portion of a declared ladder stage.

    All records must be validation results from the same trained seed, task and
    checkpoint under the frozen protocol. This does not certify training update
    boundaries, optimizer diagnostics, fixed-batch overfit or native chronology.
    R16 is deliberately outside eligibility; its failure must remain reportable.
    """
    ladder_stage = _integer(ladder_stage, "ladder_stage")
    requirements = {1: (IMMEDIATE,), 2: (R0,), 3: (R0, R1), 4: (R0, R1, R2), 5: (R0, R1, R2)}
    if ladder_stage not in requirements:
        raise ValueError("learned ladder stages are 1..5")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of competence result dictionaries")
    by_assay = {}
    identity = None
    for record in records:
        if record.get("kind") != "native_coordinate_competence" or record.get("protocol_sha256") != PROTOCOL.sha256 or record.get("protocol") != PROTOCOL.record():
            raise ValueError("records must use the frozen competence protocol")
        if record.get("split") != "validation" or record.get("measurement_phase") != NATIVE_PHASE:
            raise ValueError("ladder eligibility must use native validation predictions")
        current = (record.get("task"), record.get("trained_seed"), record.get("checkpoint_id"))
        _task(current[0])
        _integer(current[1], "trained_seed")
        if not isinstance(current[2], str) or not current[2].strip():
            raise ValueError("checkpoint identity required")
        if identity is not None and current != identity:
            raise ValueError("stage assays must share trained seed, task and checkpoint")
        identity = current
        assay = record.get("assay")
        if assay not in ASSAY_REPEATS or assay in by_assay:
            raise ValueError("one record per known assay is required")
        arms = record.get("arms", {})
        expected_arms = {"immediate"} if assay == IMMEDIATE else {"periodic", "control"}
        if set(arms) != expected_arms:
            raise ValueError("record arms do not match assay")
        # Recheck coordinates and thresholds; do not trust an aggregate/pass flag.
        for arm in arms.values():
            coordinates = arm.get("coordinates", [])
            if len(coordinates) != N_REQUIRED_COORDINATES or [r.get("coordinate") for r in coordinates] != list(REQUIRED_COORDINATES):
                raise ValueError("record must retain every required coordinate")
            threshold = PROTOCOL.categorical_min_accuracy if current[0] == "categorical" else PROTOCOL.continuous_max_normalized_mse
            tolerance = PROTOCOL.comparison_ulps * np.spacing(np.float64(threshold))
            for row in coordinates:
                _integer(row.get("coordinate"), "coordinate")
                _integer(row.get("episodes"), "episodes")
                metric = row.get("metric")
                if metric is not None and (isinstance(metric, bool) or not isinstance(metric, (int, float)) or not np.isfinite(metric) or metric < 0 or (current[0] == "categorical" and metric > 1)):
                    raise ValueError("record has invalid coordinate metric")
                expected_pass = row["episodes"] == PROTOCOL.episodes_per_coordinate and metric is not None and (metric >= threshold - tolerance if current[0] == "categorical" else metric <= threshold + tolerance)
                if row.get("passed") is not bool(expected_pass):
                    raise ValueError("record coordinate pass flag disagrees with frozen gate")
            if arm.get("passed") is not all(r["passed"] for r in coordinates):
                raise ValueError("record arm pass flag disagrees with coordinate scores")
        calculated_pass = all(r["passed"] for arm in arms.values() for r in arm["coordinates"])
        if record.get("coordinate_gate_passed") is not calculated_pass:
            raise ValueError("record gate flag disagrees with coordinate scores")
        by_assay[assay] = record
    missing = [assay for assay in requirements[ladder_stage] if assay not in by_assay]
    failed = [assay for assay in requirements[ladder_stage] if assay in by_assay and not by_assay[assay]["coordinate_gate_passed"]]
    return {"ladder_stage": ladder_stage, "required_assays": list(requirements[ladder_stage]),
        "missing_assays": missing, "failed_assays": failed, "coordinate_requirement_met": not missing and not failed,
        "scientific_acceptance": False, "protocol_sha256": PROTOCOL.sha256,
        "remaining_training_requirements": ["declared stage-boundary update count", "verified native query chronology", "recorded gradient and parameter-change norms"] + (["fixed-batch overfit and untouched IID generalization"] if ladder_stage == 1 else []),
        "selection_requirements": "minimum balanced validation task loss among eligible checkpoints; earliest checkpoint and frozen LR-order tie breaks; not implemented here"}
