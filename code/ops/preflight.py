"""Validate native cluv config and preview bounded smoke commands, without SSH.

Usage: uv run --group ops python code/ops/preflight.py --cluster tamia
This is the project's dry run. `cluv submit` has no dry-run flag in 0.1.5.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shlex
import subprocess
import tomllib

from cluv.config import CluvConfig
from cluv.cli.submit import add_cluv_sbatch_args, get_sbatch_command, get_job_env_vars

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--cluster", help="Preview one cluster, never submit or connect")
    parser.add_argument("--source-commit", help="Full SHA; defaults to committed local HEAD")
    parser.add_argument("--run-id", default="cpu-smoke-001")
    args = parser.parse_args()
    version = importlib.metadata.version("cluster-uv")
    if version != "0.1.5":
        parser.error(f"expected cluster-uv 0.1.5, found {version}")
    with args.config.open("rb") as handle:
        config = CluvConfig.model_validate(tomllib.load(handle)["tool"]["cluv"])
    if not args.cluster:
        print(json.dumps({"schema_valid": True, "cluster_uv": version,
                          "clusters": config.clusters_names, "remote_actions": 0}, indent=2))
        return
    if args.config.resolve() != (ROOT / "pyproject.toml").resolve():
        parser.error("cluster previews require root pyproject.toml; alternative configs are schema-only")
    if args.cluster == "first":
        parser.error("first races all active clusters and all configurations; approve an explicit race separately")
    if args.cluster not in config.clusters_names:
        parser.error(f"unknown configured cluster {args.cluster}")
    source_commit = args.source_commit
    if source_commit is None:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True)
        if result.returncode:
            parser.error("no committed HEAD; schema-only validation works without --cluster")
        source_commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        parser.error("require a full 40-character source SHA")
    commit_check = subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
                                  cwd=ROOT, text=True, capture_output=True)
    if commit_check.returncode:
        parser.error("source SHA does not exist as a local commit")
    required = ["pyproject.toml", "uv.lock", ".python-version", "README.md", "code/ops/calibration_job.sh",
                "src/representation_collapse/__init__.py",
                "src/representation_collapse/calibrate.py", "src/representation_collapse/metrics.py"]
    for required_path in required:
        content = subprocess.run(["git", "show", f"{source_commit}:{required_path}"],
                                 cwd=ROOT, capture_output=True)
        if content.returncode or not content.stdout:
            parser.error(f"source commit is missing required file {required_path}")
        if required_path in {"pyproject.toml", "uv.lock", "code/ops/calibration_job.sh"}:
            current = ROOT / required_path
            if not current.is_file() or current.read_bytes() != content.stdout:
                parser.error(f"working {required_path} differs from pinned source; commit/review before preview")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", args.run_id):
        parser.error("invalid run ID")
    cluster_config = config.get_cluster_config(args.cluster)
    if len(cluster_config.sbatch_args) != 1:
        parser.error("bounded smoke requires exactly one configuration; lists trigger races")
    resources = cluster_config.sbatch_args[0]
    expected = {"nodes": 1, "ntasks": 1, "cpus-per-task": 1, "mem": "2G", "time": "00:10:00"}
    for key, value in expected.items():
        if resources.get(key) != value:
            parser.error(f"smoke requires {key}={value!r}, found {resources.get(key)!r}")
    allowed = set(expected) | {"account", "partition", "qos", "job-name"}
    if unknown := set(resources) - allowed:
        parser.error(f"unreviewed resource flags: {sorted(unknown)}")
    if args.cluster != "mila" and not resources.get("account"):
        parser.error("verify and configure this cluster's account before previewing a submission")
    if cluster_config.job_script_path is None:
        parser.error("configure job_script_path before preview")
    script = ROOT / str(cluster_config.job_script_path)
    if not script.is_file():
        parser.error(f"missing job script: {script}")
    forwarded = ["--source-commit", source_commit, "--run-id", args.run_id,
                 "--cluster", args.cluster]
    sbatch = add_cluv_sbatch_args(resources, script, args.cluster, cluster_config)
    command = get_sbatch_command(cluster_config.job_script_path, sbatch, forwarded,
                                get_job_env_vars(args.cluster, source_commit, cluster_config),
                                cluster_config.project_dir)
    client = ["uv", "run", "--group", "ops", "cluv", "submit", args.cluster,
              "code/ops/calibration_job.sh", "--no-sync-datasets", "--", *forwarded]
    print(json.dumps({"schema_valid": True, "remote_actions": 0,
                      "approval_required_before_execution": True, "cluster": args.cluster,
                      "max_cpu_hours": 1 / 6, "max_gpu_hours": 0, "resources": resources,
                      "source_commit": source_commit, "cluv_command_after_approval": shlex.join(client),
                      "remote_sbatch_preview": command}, indent=2))


if __name__ == "__main__":
    main()
