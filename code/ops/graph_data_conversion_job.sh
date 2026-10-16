#!/usr/bin/env bash
# Exactly one separately approved cluv CPU submission; no install or retry.
set -euo pipefail
[[ $# -eq 4 && "$1" == --plan && "$3" == --plan-sha256 ]] || exit 2
[[ "$4" =~ ^[0-9a-f]{64}$ && "${SLURM_CPUS_PER_TASK:-}" == 1 && -n "${SLURM_JOB_ID:-}" ]] || exit 2
exec /network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/bin/python -B code/ops/graph_data_conversion_job.py --plan "$2" --plan-sha256 "$4"
