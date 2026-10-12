"""Validation-only checkpoint selection for the bounded Gate B discovery grid.

The input is a task-loss selection packet, not a general metrics dictionary.
Geometry, arm contrasts and held-out outcomes belong in separate reports. A
``native_validation`` assertion records the caller's contract; it does not
independently establish native execution or accept a scientific claim.
"""

from __future__ import annotations

import math
import hashlib
import json
from numbers import Integral, Real
import re

from .competence import R0, R1, R2, assess_stage_competence

HORIZONS = ("R1", "R2", "R4", "R8", "R16")
VALIDATION_HORIZONS = ("R0",) + HORIZONS
_ASSAY_HORIZONS = {R0: "R0", R1: "R1", R2: "R2"}
ARMS = ("periodic", "control")
_FIELDS = {
    "model", "task", "trained_seed", "learning_rate", "step", "stage",
    "checkpoint_id", "split", "short_horizon_records",
    "validation_loss_by_horizon", "native_validation", "checkpoint_path",
    "validation_task_loss", "validation_manifest",
}


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _freeze_validation_manifest(value):
    if not isinstance(value, dict) or set(value) != set(VALIDATION_HORIZONS):
        raise ValueError("frozen validation manifest requires exactly R0/R1/R2/R4/R8/R16")
    frozen = {}
    for horizon in VALIDATION_HORIZONS:
        binding = value[horizon]
        if not isinstance(binding, dict) or set(binding) != {"batch_sha256", "sampling_seed"}:
            raise ValueError("each frozen validation horizon requires batch_sha256 and sampling_seed")
        digest = binding["batch_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("validation batch_sha256 must be a canonical lowercase 32-byte SHA-256 hex digest")
        if not _integer(binding["sampling_seed"]):
            raise ValueError("validation sampling_seed must be a nonnegative integer")
        frozen[horizon] = {"batch_sha256": digest, "sampling_seed": int(binding["sampling_seed"])}
    return frozen


def select_checkpoint(candidates: list[dict], lr_grid: list[float], *, validation_manifest: dict | None = None) -> dict:
    """Choose eligible stage-5 weights by equal-arm/equal-horizon native loss.

    Candidates use ``R1/R2/R4/R8/R16 -> {periodic, control}`` loss mappings.
    Categorical losses are cross-entropy; continuous losses are *raw* MSE.
    Every candidate's R0/R1/R2 coordinate records must share its model, task,
    trained seed and immutable checkpoint ID. They are validated by the frozen
    scorer, including all eight coordinates and both arms. No R16 competence
    gate is imposed: either long-horizon outcome must remain reportable.

    ``validation_manifest`` is the caller's prospective fixed mapping for all
    six horizons: ``{R0,R1,R2,R4,R8,R16: {batch_sha256,sampling_seed}}``.
    Freeze it from saved batch identities before optimizing any candidate.
    Every candidate must carry that same mapping; all R0/R1/R2 score records
    must also match their frozen batch hash and sampling seed. Selection never
    infers the frozen manifest from a candidate. Long-horizon bindings assert
    the loss source; independent artifact verification must establish that the
    declared losses were actually computed from those batches.

    Malformed/failed individual candidates receive explicit reasons. A mixed
    model/task/seed population, unknown learning rate, duplicate checkpoint ID,
    empty input or malformed grid raises ValueError rather than combining
    incomparable evidence. Exact loss ties prefer the earliest update and then
    the prespecified LR order. Changing the grid order is a protocol change.
    """
    frozen_manifest = _freeze_validation_manifest(validation_manifest)
    if not isinstance(candidates, list) or not candidates or any(not isinstance(c, dict) for c in candidates):
        raise ValueError("nonempty candidate dictionary list required")
    if not isinstance(lr_grid, (list, tuple)) or not lr_grid:
        raise ValueError("nonempty frozen learning-rate grid required")
    grid = [_positive_number(v, "learning rate") for v in lr_grid]
    if len(set(grid)) != len(grid):
        raise ValueError("frozen learning-rate grid must have unique entries")
    identity = None
    checkpoint_ids = set()
    for candidate in candidates:
        current = (candidate.get("model"), candidate.get("task"), candidate.get("trained_seed"))
        if current[0] not in ("gru", "lstm") or current[1] not in ("categorical", "continuous") or not _integer(current[2]):
            raise ValueError("candidate requires a valid model/task/trained_seed identity")
        if identity is not None and current != identity:
            raise ValueError("checkpoint selection cannot mix model/task/trained_seed cells")
        identity = current
        rate = _positive_number(candidate.get("learning_rate"), "candidate learning rate")
        if rate not in grid:
            raise ValueError("candidate learning rate is outside the frozen grid")
        checkpoint_id = candidate.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip() or checkpoint_id in checkpoint_ids:
            raise ValueError("nonempty unique immutable checkpoint IDs required")
        checkpoint_ids.add(checkpoint_id)

    reviews = []
    for index, candidate in enumerate(candidates):
        reasons = []
        forbidden = sorted(set(candidate) - _FIELDS)
        if forbidden:
            reasons.append("unexpected_selection_fields:" + ",".join(forbidden))
        if candidate.get("stage") != 5 or not _integer(candidate.get("stage")):
            reasons.append("not_stage_5")
        if not _integer(candidate.get("step")) or candidate.get("step", 0) < 1:
            reasons.append("invalid_checkpoint_step")
        if candidate.get("split") != "validation":
            reasons.append("not_validation_split")
        if candidate.get("native_validation") is not True:
            reasons.append("native_validation_not_asserted")
        try:
            candidate_manifest = _freeze_validation_manifest(candidate.get("validation_manifest"))
            for horizon in VALIDATION_HORIZONS:
                for name in ("batch_sha256", "sampling_seed"):
                    if candidate_manifest[horizon][name] != frozen_manifest[horizon][name]:
                        reasons.append("validation_manifest_mismatch:" + horizon + ":" + name)
        except (ValueError, TypeError, KeyError) as error:
            reasons.append("invalid_candidate_validation_manifest:" + str(error))
        records = candidate.get("short_horizon_records")
        try:
            if not isinstance(records, list) or len(records) != 3 or any(not isinstance(r, dict) for r in records):
                raise ValueError("exactly R0/R1/R2 records required")
            if {r.get("assay") for r in records} != {R0, R1, R2}:
                raise ValueError("exactly R0/R1/R2 assays required")
            for record in records:
                for name in ("model", "task", "trained_seed", "checkpoint_id"):
                    if record.get(name) != candidate[name]:
                        raise ValueError("record/candidate identity mismatch:" + name)
                if record.get("fixture_only") is not False or record.get("trained_seed_units") != 1:
                    raise ValueError("reference fixtures cannot qualify a trained checkpoint")
                if record.get("validation_eligibility_met") is not record.get("coordinate_gate_passed"):
                    raise ValueError("record validation flag disagrees with coordinate gate")
                horizon = _ASSAY_HORIZONS[record["assay"]]
                binding = frozen_manifest[horizon]
                digest = record.get("batch_sha256")
                if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest != binding["batch_sha256"]:
                    raise ValueError("record/frozen validation batch hash mismatch:" + horizon)
                sampling_seed = record.get("sampling_seed")
                if not _integer(sampling_seed) or sampling_seed != binding["sampling_seed"]:
                    raise ValueError("record/frozen validation sampling seed mismatch:" + horizon)
            gate = assess_stage_competence(records, 5)
            reasons.extend("short_horizon_gate_failed:" + assay for assay in gate["failed_assays"])
            reasons.extend("short_horizon_gate_missing:" + assay for assay in gate["missing_assays"])
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            reasons.append("invalid_short_horizon_records:" + str(error))
        balanced_loss = None
        try:
            losses = candidate.get("validation_loss_by_horizon")
            if not isinstance(losses, dict) or set(losses) != set(HORIZONS):
                raise ValueError("exactly R1/R2/R4/R8/R16 validation losses required")
            values = []
            for horizon in HORIZONS:
                arm_losses = losses[horizon]
                if not isinstance(arm_losses, dict) or set(arm_losses) != set(ARMS):
                    raise ValueError("each validation horizon requires exactly periodic/control losses")
                for arm in ARMS:
                    value = arm_losses[arm]
                    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                        raise ValueError("validation task losses must be finite nonnegative CE/raw MSE")
                    values.append(float(value))
            # Scale before summing: even a finite mean of large finite losses
            # must not overflow an intermediate unnormalized accumulation.
            balanced_loss = math.fsum(value / len(values) for value in values)
            if not math.isfinite(balanced_loss):
                raise ValueError("balanced validation task loss is nonfinite")
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            reasons.append("invalid_validation_losses:" + str(error))
        reviews.append({"index": index, "checkpoint_id": candidate["checkpoint_id"],
                        "eligible": not reasons, "ineligibility_reasons": reasons,
                        "balanced_validation_loss": balanced_loss})
    eligible = [review for review in reviews if review["eligible"]]
    selected = None
    selected_index = None
    if eligible:
        best = min(eligible, key=lambda r: (r["balanced_validation_loss"], candidates[r["index"]]["step"],
                                          grid.index(float(candidates[r["index"]]["learning_rate"]))))
        selected_index = best["index"]
        candidate = candidates[selected_index]
        selected = {name: candidate[name] for name in ("model", "task", "trained_seed", "learning_rate", "step", "checkpoint_id")}
        selected["balanced_validation_loss"] = best["balanced_validation_loss"]
    return {"schema_version": 1, "kind": "gate_b_validation_checkpoint_selection",
            "scientific_acceptance": False, "selected_index": selected_index,
            "selected_checkpoint_id": selected["checkpoint_id"] if selected else None,
            "selected": selected, "candidate_reviews": reviews, "frozen_lr_grid": grid,
            "validation_manifest": frozen_manifest,
            "validation_manifest_sha256": hashlib.sha256(json.dumps(frozen_manifest, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "selection_units": "equally weighted periodic/control arms and R1/R2/R4/R8/R16 horizons",
            "loss_units": "cross_entropy" if identity[1] == "categorical" else "raw_squared_error",
            "tie_breakers": ["earliest_step", "frozen_lr_grid_order"],
            "native_execution_independently_verified": False,
            "status": "eligible_candidate_selected" if selected else "acquisition_or_optimization_unresolved"}
