#!/usr/bin/env python3
"""Dry-prepare the one-shot synthetic Figure 10 node2a Mila CPU job.

This tool never submits, connects to a cluster, stages files, or runs the
fixture. It emits a reviewed-input-bound plan and exact cluv command only when
the local source commit is clean and all externally supplied receipts exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

SCRATCH = Path("/network/scratch/m/mehrab.hamidi/representation-collapse")
INTERPRETER = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/venv/bin/python"
ENV_RECEIPT = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/preparation-receipt.json"
ENV_INVENTORY = SCRATCH / "environments/graph-native-cpu-20261015-calibration01/node2a-environment-identity.json"
INTERPRETER_SHA256 = "b081289003128d701e48880da11800d7e4410a5777fec6afaa9e31d07419c52f"
SOURCE_ROOT = SCRATCH / "source"
RUN_ROOT = SCRATCH / "runs/figure10-node2a"
JOB = "code/ops/tgrab_node2a_job.sh"
BOOTSTRAP = "src/representation_collapse/tgrab_figure10_bootstrap.py"
PINNED_IMPORT = "src/representation_collapse/pinned_graph_import.py"
PINNED_IMPORT_SHA256 = "2f14b3683f67d33f4a5f6325b5b204aee3a086e1fad1966acb9810a302d04344"
PINNED_CELL_SHA256 = "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
PINNED_GRAPH_RECONSTRUCTION_SHA256 = "1e8212b9589709d1d4654d0ebfe98179a41ea5a307dbc2083fe84f06c1aead13"
PINNED_WRAPPER_SHA256 = {
    "model/gclstm.py": "86069a32fced5f09c60cd795e4f0abd82674dafae250ef3673ac8c3650504609",
    "model/link_predictor.py": "cca12cfc73b8660c92236f530504da08ba0b606ec13a1a681ffabcae43b7ca86",
    "model/node_emb.py": "4be4450730f33949e0354348b3bd8f69ba0434e4aa911abec0fc5f798e8c2d29",
}
EXPECTED_MODULE_DISTRIBUTIONS = {
    "numpy": "numpy", "numpy.core._multiarray_umath": "numpy",
    "torch": "torch", "torch._C": "torch", "torch.nn": "torch",
    "torch.nn.functional": "torch", "torch.optim": "torch",
    "torch_geometric": "torch-geometric", "torch_geometric.nn": "torch-geometric",
    "torch_geometric_temporal.nn.recurrent.gc_lstm": "torch-geometric-temporal",
    "torch_cluster": "torch-cluster", "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse", "torch_spline_conv": "torch-spline-conv",
}
FIXED = {
    "cluster": "mila", "account": "mila", "qos": "normal", "partition": "main-cpu",
    "nodes": 1, "ntasks": 1, "cpus_per_task": 1, "memory": "2G",
    "walltime": "00:10:00", "gpus": 0, "submissions": 1,
    "retry": False, "requeue": False, "scientific_acceptance": False,
}


class PreparationError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], check=True,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreparationError(f"git verification failed: {args[0]}") from exc


def validate_staged_environment_receipt(path: Path, expected_sha256: str) -> None:
    path = Path(path)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            raise PreparationError("staged environment receipt must be a bounded regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise PreparationError("authenticated remote environment receipt must first be staged locally") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PreparationError("staged environment receipt digest mismatch")
    try:
        receipt = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError("staged environment receipt is invalid JSON") from exc
    if (type(receipt) is not dict
            or receipt.get("state") != "installed_metadata_verified_native_smoke_pending"
            or receipt.get("installed_metadata_verified") is not True
            or receipt.get("requirements_sha256") != "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
            or receipt.get("artifact_manifest_sha256") != "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"
            or receipt.get("installed_cell_sha256") != "c3568d2c1ba8926ab43e11c840213f90ebe782ad158dd76809d1dc1c255797af"
            or receipt.get("venv") != str(INTERPRETER.parent.parent)):
        raise PreparationError("staged environment receipt is not the exact pinned install receipt")


def validate_staged_environment_inventory(path: Path, expected_sha256: str) -> None:
    path = Path(path)
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 262_144:
            raise PreparationError("staged environment inventory must be a bounded regular file")
        raw = path.read_bytes()
    except OSError as exc:
        raise PreparationError("externally pinned environment identity inventory must be staged locally") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PreparationError("staged environment inventory digest mismatch")
    try:
        inventory = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError("staged environment inventory is invalid JSON") from exc
    identity = inventory.get("identity") if type(inventory) is dict else None
    if (type(inventory) is not dict or set(inventory) != {"schema", "identity"}
            or inventory.get("schema") != "tgrab-node2a-environment-identity-v1"
            or type(identity) is not dict
            or set(identity) != {"python", "python_version", "platform", "machine", "prefix",
                                 "lock_sha256", "wheel_manifest_sha256", "installed_versions",
                                 "installed_module_files", "source_identity"}
            or identity.get("python_version") != [3, 10, 19]
            or "3.10.19" not in str(identity.get("python", ""))
            or "Linux" not in str(identity.get("platform", ""))
            or identity.get("machine") != "x86_64"
            or identity.get("prefix") != str(INTERPRETER.parent.parent)
            or identity.get("lock_sha256") != "ea2f58d2a12f5d8562b31d5fa7b5c646121784867c134afd4a5cf6a8108b8c06"
            or identity.get("wheel_manifest_sha256") != "67e27963f2166f8a7e26dba301c9020d2bd7d4c012010fb859e6e43b15ce9842"):
        raise PreparationError("environment identity inventory does not match the closed pinned runtime schema")
    source = identity["source_identity"]
    if (type(source) is not dict
            or set(source) != {"wrappers", "cell_sha256", "graph_reconstruction_sha256",
                               "pinned_graph_import_sha256"}
            or source.get("wrappers") != PINNED_WRAPPER_SHA256
            or source.get("cell_sha256") != PINNED_CELL_SHA256
            or source.get("graph_reconstruction_sha256") != PINNED_GRAPH_RECONSTRUCTION_SHA256
            or source.get("pinned_graph_import_sha256") != PINNED_IMPORT_SHA256):
        raise PreparationError("environment inventory source identity differs from pinned graph sources")
    module_files = identity["installed_module_files"]
    if type(module_files) is not dict or set(module_files) != set(EXPECTED_MODULE_DISTRIBUTIONS):
        raise PreparationError("environment inventory has incomplete native module coverage")
    if type(identity.get("installed_versions")) is not dict or not identity["installed_versions"]:
        raise PreparationError("environment inventory lacks installed package versions")
    for module_name, record in module_files.items():
        if (type(record) is not dict or set(record) != {"path", "sha256", "distribution", "record_sha256"}
                or type(record["path"]) is not str
                or not record["path"].startswith(str(INTERPRETER.parent.parent) + "/")
                or record["distribution"] != EXPECTED_MODULE_DISTRIBUTIONS[module_name]
                or not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))
                or not re.fullmatch(r"[0-9a-f]{64}", str(record["record_sha256"]))):
            raise PreparationError("environment inventory contains invalid module/RECORD identity")


def validate(repo: Path, manifest: dict) -> dict:
    """Validate pins; manifest receipts are inputs, never inferred or fetched."""
    if set(manifest) != {"source_commit", "source_tree", "bootstrap_sha256",
                         "pinned_graph_import_sha256",
                         "job_sha256", "supervisor_sha256", "environment_receipt_sha256",
                         "environment_receipt_local_path",
                         "environment_inventory_sha256", "environment_inventory_local_path",
                         "independent_review_receipt", "independent_review_sha256",
                         "run_id"}:
        raise PreparationError("manifest fields must exactly match the closed schema")
    for key in ("bootstrap_sha256", "pinned_graph_import_sha256", "job_sha256", "supervisor_sha256",
                "environment_receipt_sha256", "independent_review_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest[key])):
            raise PreparationError(f"invalid SHA-256 pin: {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["environment_inventory_sha256"])):
        raise PreparationError("invalid SHA-256 pin: environment_inventory_sha256")
    if manifest["pinned_graph_import_sha256"] != PINNED_IMPORT_SHA256:
        raise PreparationError("pinned graph importer differs from independently reviewed source pin")
    commit = manifest["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise PreparationError("source_commit must be a full immutable Git SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest["source_tree"])):
        raise PreparationError("source_tree must be a full immutable Git tree SHA")
    run_id = manifest["run_id"]
    if not re.fullmatch(r"node2a-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}", str(run_id)):
        raise PreparationError("run_id must be unique UTC node2a-... identity")
    review = Path(manifest["independent_review_receipt"])
    if not review.is_file() or review.is_symlink() or digest(review) != manifest["independent_review_sha256"]:
        raise PreparationError("independent code/job review receipt missing or hash mismatch")
    head = git(repo, "rev-parse", "HEAD")
    if head != commit or git(repo, "status", "--porcelain"):
        raise PreparationError("source must be the requested clean committed tree")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    # Git object identity binds the entire committed tree; the explicit digest
    # pins are additional runtime checks for the two executable entrypoints.
    if digest(repo / BOOTSTRAP) != manifest["bootstrap_sha256"]:
        raise PreparationError("bootstrap source digest mismatch")
    if (manifest["pinned_graph_import_sha256"] != PINNED_IMPORT_SHA256
            or digest(repo / PINNED_IMPORT) != PINNED_IMPORT_SHA256):
        raise PreparationError("pinned graph importer differs from independently reviewed source pin")
    if tree != manifest["source_tree"]:
        raise PreparationError("committed Git tree identity mismatch")
    if digest(repo / JOB) != manifest["job_sha256"]:
        raise PreparationError("job wrapper digest mismatch")
    if digest(repo / "code/ops/tgrab_node2a_supervisor.py") != manifest["supervisor_sha256"]:
        raise PreparationError("supervisor digest mismatch")
    staged_receipt = Path(manifest["environment_receipt_local_path"])
    if not staged_receipt.is_absolute():
        raise PreparationError("authenticated remote environment receipt must first be staged locally as a regular file")
    validate_staged_environment_receipt(staged_receipt, manifest["environment_receipt_sha256"])
    staged_inventory = Path(manifest["environment_inventory_local_path"])
    if not staged_inventory.is_absolute():
        raise PreparationError("environment inventory local path must be absolute")
    validate_staged_environment_inventory(staged_inventory, manifest["environment_inventory_sha256"])
    return {"source_commit": commit, "source_tree": tree,
            "run_id": run_id, "source_root": str(SOURCE_ROOT),
            "interpreter": str(INTERPRETER), "environment_receipt": str(ENV_RECEIPT),
            "environment_receipt_sha256": manifest["environment_receipt_sha256"],
            "environment_inventory": str(ENV_INVENTORY),
            "environment_inventory_sha256": manifest["environment_inventory_sha256"],
            "bootstrap_sha256": manifest["bootstrap_sha256"],
            "pinned_graph_import_sha256": PINNED_IMPORT_SHA256,
            "job_sha256": manifest["job_sha256"],
            "supervisor_sha256": manifest["supervisor_sha256"],
            "independent_review_sha256": manifest["independent_review_sha256"],
            "output": str(RUN_ROOT / run_id), "stop_file": str(RUN_ROOT / run_id / "STOP"),
            "deadline_seconds": 570, "fixture": "synthetic-node2a-zero-update-only",
            "argv": [str(INTERPRETER), "-B", "-m", "representation_collapse.tgrab_figure10_bootstrap",
                     "--source-root", str(SOURCE_ROOT), "--output", str(RUN_ROOT / run_id),
                     "--environment-identity", str(ENV_INVENTORY),
                     "--environment-identity-sha256", manifest["environment_inventory_sha256"]],
            "allocation": FIXED, "scientific_acceptance": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        plan = validate(args.repo.resolve(), manifest)
        if args.output.exists() or args.output.is_symlink():
            raise PreparationError("plan output must be new and absent")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()
        tmp = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
        with tmp.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, args.output)
        print(json.dumps({"plan": str(args.output), "plan_sha256": hashlib.sha256(raw).hexdigest(),
                          "cluv_command": f"uv run --group ops cluv submit mila {JOB} --no-sync-datasets -- --plan {args.output} --plan-sha256 {hashlib.sha256(raw).hexdigest()}",
                          "submitted": False}, sort_keys=True))
        return 0
    except (PreparationError, OSError, json.JSONDecodeError) as exc:
        print(f"PREPARATION BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
