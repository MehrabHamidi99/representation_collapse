"""Fail-closed prospective T-GRAB Figure 10 GC-LSTM reconstruction gate.

This module deliberately does not train or launch a model. The frozen config
still disables launch, and reviewed native checkpoint/reporting integration is
missing. It provides auditable input checks and the exact per-target F1
reducer needed by a future admitted runner. Nothing here is historical replay
or evidence of representation collapse.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Mapping, Sequence

import numpy as np

from . import graph_data
from . import graph_reconstruction as gr


class Figure10Blocked(RuntimeError):
    """Execution is not admitted by verified project artifacts."""


EXPECTED_SPLITS = {"train": [0, 2560], "validation": [2560, 2816], "test": [2816, 3072]}
EXPECTED_TEST_TARGETS = 256
EXPECTED_PAIRS = 9900


@dataclass(frozen=True)
class FilePin:
    path: Path
    sha256: str

    def verify(self) -> bytes:
        p = Path(self.path)
        if p.is_symlink() or not p.is_file():
            raise Figure10Blocked(f"explicit regular file required: {p}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise Figure10Blocked("explicit lowercase SHA-256 pin required")
        if p.stat().st_nlink != 1 or p.stat().st_size > 2 * 1024 * 1024:
            raise Figure10Blocked(f"single-link bounded input required: {p}")
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise Figure10Blocked(f"SHA-256 pin mismatch: {p}")
        return raw


def validate_primary_contract(config: Mapping, manifest: Mapping) -> None:
    """Check frozen candidate identity and the exact chronological 40/4/4 split."""
    gr.validate_frozen_config(config)
    if manifest.get("kind") != "tgrab_primary_converted_v1":
        raise Figure10Blocked("verified primary T-GRAB conversion manifest required")
    if manifest.get("config_sha256") != graph_data.CONFIG_SHA256:
        raise Figure10Blocked("data/config identity mismatch")
    if manifest.get("generation_performed") is not False:
        raise Figure10Blocked("converted source data must not be regenerated")
    spec = manifest.get("spec", {})
    if spec != {"nodes": 100, "block": 32, "train_periods": 40,
                "validation_periods": 4, "test_periods": 4}:
        raise Figure10Blocked("primary Pdet(2,32) 40/4/4 spec mismatch")
    intervals = {k: manifest.get("splits", {}).get(k, {}).get("interval")
                 for k in EXPECTED_SPLITS}
    if intervals != EXPECTED_SPLITS:
        raise Figure10Blocked("expected source intervals [0,2560)/[2560,2816)/[2816,3072)")
    if len(manifest.get("snapshot_semantic_sha256", ())) != 3072:
        raise Figure10Blocked("exactly 3072 source snapshots required")
    if len(manifest.get("pattern_pairs", ())) != 2 or manifest["pattern_pairs"][0] == manifest["pattern_pairs"][1]:
        raise Figure10Blocked("two distinct source motifs required")


def per_target_f1_with_predecessor(predecessor: gr.Snapshot,
                                  targets: Sequence[gr.Snapshot],
                                  probabilities: np.ndarray) -> list[dict]:
    """Same reducer with the required t=2815 predecessor made explicit."""
    if predecessor.timestamp != 2815 or predecessor.nodes != 100:
        raise Figure10Blocked("explicit timestamp-2815 predecessor required")
    if predecessor.node_ids != tuple(range(100)) or len(targets) != EXPECTED_TEST_TARGETS:
        raise Figure10Blocked("test state chronology contract mismatch")
    values = np.asarray(probabilities)
    if values.shape != (EXPECTED_TEST_TARGETS, EXPECTED_PAIRS) or values.dtype.kind not in "fiu":
        raise Figure10Blocked("probability array must be numeric (256,9900)")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise Figure10Blocked("finite probabilities in [0,1] required")
    rows = []
    previous = predecessor
    for i, target in enumerate(targets):
        if target.timestamp != 2816 + i or target.nodes != 100 or target.node_ids != tuple(range(100)):
            raise Figure10Blocked("test timestamps must be contiguous 2816..3071")
        rows.append(gr.snapshot_metrics(target, values[i], previous))
        previous = target
    return rows


def test_reduction(rows: Sequence[Mapping]) -> dict:
    """Summarize per-target rows; primary is the unweighted temporal F1 mean."""
    if len(rows) != EXPECTED_TEST_TARGETS:
        raise Figure10Blocked("exactly 256 per-target metric rows required")
    f1 = [float(r["f1"]) for r in rows]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in f1):
        raise Figure10Blocked("invalid per-target F1 values")
    for row in rows:
        expected = row.get("phase_transition_expected")
        observed = row.get("change_point")
        if type(expected) is not bool or type(observed) is not bool:
            raise Figure10Blocked("transition and graph-change flags must be booleans")
        if expected != observed:
            raise Figure10Blocked("phase transition disagrees with actual adjacent-graph change")
    transition = [x for x, r in zip(f1, rows) if r["change_point"]]
    nontransition = [x for x, r in zip(f1, rows) if not r["change_point"]]
    if len(transition) != 8 or len(nontransition) != 248:
        raise Figure10Blocked("expected 8 transition and 248 nontransition targets")
    return {"targets": 256, "primary_time_mean_f1": math.fsum(f1) / 256,
            "transition_time_mean_f1": math.fsum(transition) / 8,
            "nontransition_time_mean_f1": math.fsum(nontransition) / 248,
            "per_target_f1": f1}


def run(*, config_pin: FilePin, data_manifest_pin: FilePin,
        stop_file: Path, deadline_unix: float, source_root: Path,
        readiness: gr.Readiness) -> None:
    """Fail closed before constructing a model or opening the sealed test split.

    This release intentionally cannot reach training: frozen config has
    launch.enabled=false and graph_reconstruction preflight records the native
    checkpoint/reporting runner gap. It makes the residual gates explicit.
    """
    import time
    if config_pin is None or data_manifest_pin is None or stop_file is None or deadline_unix is None or source_root is None or readiness is None:
        raise Figure10Blocked("explicit config/data/source/receipt/stop/deadline pins required")
    if not math.isfinite(float(deadline_unix)) or time.time() >= deadline_unix:
        raise Figure10Blocked("deadline expired or invalid")
    if Path(stop_file).exists():
        raise Figure10Blocked("STOP file present")
    config_bytes = config_pin.verify()
    data_manifest_pin.verify()
    config = json.loads(config_bytes)
    gr.validate_frozen_config(config)
    report = gr.preflight(config_pin.path, source_root, readiness)
    raise Figure10Blocked("prospective reconstruction blocked: " + "; ".join(report["blockers"]))


def _atomic_new_json(path: Path, value: Mapping) -> None:
    """Publish a small status receipt atomically without overwriting evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path, follow_symlinks=False)
        os.unlink(temp)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Readiness-only CLI; it cannot launch training in frozen v1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--receipt", action="append", default=[], metavar="NAME=PATH=SHA256",
                        help="pinned readiness receipt; names: environment,conversion,engineering_review,scientific_review,reporting,admission")
    parser.add_argument("--report", type=Path, required=True,
                        help="new path for an atomic blocked-readiness receipt")
    args = parser.parse_args(argv)
    allowed = {"environment", "conversion", "engineering_review", "scientific_review", "reporting", "admission"}
    pins = {}
    for token in args.receipt:
        parts = token.split("=", 2)
        if len(parts) != 3 or parts[0] not in allowed or parts[0] in pins:
            parser.error("invalid or duplicate --receipt")
        pins[parts[0]] = gr.PinnedJSON(Path(parts[1]), parts[2])
    readiness = gr.Readiness(**pins)
    config_pin = FilePin(args.config, args.config_sha256)
    data_pin = FilePin(args.data_manifest, args.data_manifest_sha256)
    status = {"kind": "tgrab_figure10_reconstruction_readiness_v1",
              "status": "blocked", "scientific_status": "no_run_no_result",
              "interpretation": "prospective reconstruction only; not historical Figure 10 reproduction",
              "created_unix": time.time(), "config_path": str(args.config),
              "data_manifest_path": str(args.data_manifest),
              "source_root": str(args.source_root), "report_path": str(args.report),
              "training_started": False, "test_payload_opened": False,
              "model_constructed": False, "job_submitted": False}
    try:
        run(config_pin=config_pin, data_manifest_pin=data_pin, stop_file=args.stop_file,
            deadline_unix=args.deadline_unix, source_root=args.source_root,
            readiness=readiness)
    except Figure10Blocked as exc:
        status["blocker"] = str(exc)[:2048]
        _atomic_new_json(args.report, status)
        print(json.dumps(status, sort_keys=True), file=sys.stderr)
        return 2
    status.update(status="unexpected_ready_return")
    _atomic_new_json(args.report, status)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

