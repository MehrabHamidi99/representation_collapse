#!/usr/bin/env bash
# Resource requests are explicit in the reviewed, single-target cluv submission.
set -euo pipefail
[[ $# -eq 4 && "$1" == --plan && "$3" == --plan-sha256 ]] || exit 2
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_CPUS_PER_TASK:-}" == 1 ]] || exit 2
[[ "$4" =~ ^[0-9a-f]{64}$ ]] || exit 2
# Only a prepared scratch environment is used. No install or package fetch here.
exec python3 -B code/ops/native_graph_calibration_job.py --plan "$2" --plan-sha256 "$4"
