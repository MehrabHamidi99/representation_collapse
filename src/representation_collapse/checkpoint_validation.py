"""Check S04 serialized validation-row consistency for checkpoint schema v1.

The accepted schema is pinned by checkpoint_json.SCHEMA_SHA256. This checker
accepts an already-decoded tree; it performs no I/O, parsing, model calls, or
mutation. Callers separately establish canonical bytes, hashes, admission, and
cross-artifact identities. Consistent rows do not establish raw-data
compatibility, model competence, or representation collapse.
"""
from __future__ import annotations

import math

from .checkpoint_json import CheckpointJSONError, require_int, require_keys


_TOP_KEYS = ("kind", "schema_version", "split", "epoch", "rows", "time_mean_f1")
_ROW_KEYS = (
    "target_timestamp", "source_timestamp", "target_phase", "source_phase",
    "phase_transition_expected", "change_point", "pairs", "tp", "fp", "fn",
    "tn", "precision", "recall", "f1", "accuracy",
)
_FIRST_TARGET = 2560
_ROWS = 256
_PAIRS = 9900


def _metric(value: object) -> int | float:
    # Check the range before isfinite so even an arbitrarily large caller-owned
    # int is rejected without coercion to float or an OverflowError.
    if (type(value) not in (int, float) or not 0 <= value <= 1
            or not math.isfinite(value)):
        raise CheckpointJSONError("finite built-in metric in [0,1] required")
    return value


def validate_validation_artifact(value: object) -> None:
    """Require the complete closed validation artifact and exact GR formulas.

    All integer fields are exact built-in ints, flags are exact bools, and
    metrics are built-in int/float numbers excluding bool. Numeric equality is
    exact, with no tolerance: the same-runtime GR zero-division formulas and
    math.fsum(F1)/256 must agree. This is not a full JSON Schema validator or a
    validation-result capability. The caller retains ownership of the tree.
    """
    artifact = require_keys(value, _TOP_KEYS)
    for key, expected in (("kind", "tgrab_complete_validation_rows_v1"),
                          ("split", "validation")):
        if type(artifact[key]) is not str or artifact[key] != expected:
            raise CheckpointJSONError("validation kind/split mismatch")
    require_int(artifact["schema_version"], minimum=1, maximum=1)
    require_int(artifact["epoch"], minimum=1, maximum=200)
    rows = artifact["rows"]
    if type(rows) is not list or len(rows) != _ROWS:
        raise CheckpointJSONError("exactly 256 built-in validation rows required")
    declared_mean = _metric(artifact["time_mean_f1"])
    f1_values = []
    for target, item in enumerate(rows, start=_FIRST_TARGET):
        row = require_keys(item, _ROW_KEYS)
        expected_integers = {
            "target_timestamp": target, "source_timestamp": target - 1,
            "target_phase": target % 64, "source_phase": (target - 1) % 64,
            "pairs": _PAIRS,
        }
        for key, expected in expected_integers.items():
            require_int(row[key], minimum=expected, maximum=expected)
        boundary = target % 64 in (0, 32)
        for key in ("phase_transition_expected", "change_point"):
            if type(row[key]) is not bool or row[key] is not boundary:
                raise CheckpointJSONError("validation phase/boundary flag mismatch")
        tp, fp, fn, tn = (
            require_int(row[key], minimum=0, maximum=_PAIRS)
            for key in ("tp", "fp", "fn", "tn")
        )
        if tp + fp + fn + tn != _PAIRS:
            raise CheckpointJSONError("confusion counts must sum to 9900")
        # Match graph_reconstruction.snapshot_metrics operation order exactly.
        expected_metrics = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "accuracy": (tp + tn) / _PAIRS,
        }
        for key, expected in expected_metrics.items():
            if _metric(row[key]) != expected:
                raise CheckpointJSONError("serialized validation formula mismatch")
        f1_values.append(row["f1"])
    if declared_mean != math.fsum(f1_values) / _ROWS:
        raise CheckpointJSONError("serialized time-mean F1 mismatch")
