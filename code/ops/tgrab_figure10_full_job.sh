#!/bin/bash
set -euo pipefail
[[ $# -eq 6 && "$1" = --admission && "$3" = --sha256 && "$5" = --wall-seconds ]] || {
  echo 'usage: full_job.sh --admission PATH --sha256 SHA --wall-seconds 3570' >&2; exit 2;
}
export RC_ADMISSION="$2" RC_ADMISSION_SHA256="$4" RC_JOB_WALL_SECONDS="$6"
[[ "$RC_ADMISSION" = /* && "$RC_ADMISSION_SHA256" =~ ^[0-9a-f]{64}$ && "$RC_JOB_WALL_SECONDS" = 3570 ]] || exit 2
export RC_WALL_START_EPOCH_NS="$(date +%s%N)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PINNED_ROOT=/network/scratch/m/mehrab.hamidi/representation-collapse
exec timeout --signal=TERM --kill-after=10 3570s \
  "$PINNED_ROOT/environments/graph-native-cpu-20261015-calibration01/venv/bin/python" -B \
  "$PINNED_ROOT/source/code/ops/tgrab_figure10_full_supervisor.py" \
  --admission "$RC_ADMISSION" --sha256 "$RC_ADMISSION_SHA256"
