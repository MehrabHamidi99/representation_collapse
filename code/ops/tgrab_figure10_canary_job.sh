#!/bin/bash
# Slurm job entry point. The elapsed clock and thread limits are set before
# supervisor/Python imports; GNU timeout is a final import-inclusive wall guard.
set -euo pipefail
# cluv does not reliably export newly assigned submit-shell variables. Prefer
# these explicit job arguments; retain environment compatibility for old plans.
seen_admission=0 seen_sha=0 seen_wall=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --admission)
      [[ $# -ge 2 && $seen_admission -eq 0 ]] || { echo 'invalid --admission' >&2; exit 2; }
      RC_ADMISSION="$2"; seen_admission=1; shift 2 ;;
    --sha256)
      [[ $# -ge 2 && $seen_sha -eq 0 ]] || { echo 'invalid --sha256' >&2; exit 2; }
      RC_ADMISSION_SHA256="$2"; seen_sha=1; shift 2 ;;
    --wall-seconds)
      [[ $# -ge 2 && $seen_wall -eq 0 ]] || { echo 'invalid --wall-seconds' >&2; exit 2; }
      RC_JOB_WALL_SECONDS="$2"; seen_wall=1; shift 2 ;;
    *) echo 'usage: canary_job.sh --admission PATH --sha256 SHA --wall-seconds N' >&2; exit 2 ;;
  esac
done
: "${RC_ADMISSION:?set admission manifest}"
: "${RC_ADMISSION_SHA256:?set admission SHA-256}"
: "${RC_JOB_WALL_SECONDS:?set admitted outer wall limit}"
[[ "$RC_ADMISSION" = /* && "$RC_ADMISSION_SHA256" =~ ^[0-9a-f]{64}$ && "$RC_JOB_WALL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo 'invalid admission arguments' >&2; exit 2;
}
export RC_ADMISSION RC_ADMISSION_SHA256 RC_JOB_WALL_SECONDS
export RC_WALL_START_EPOCH_NS="$(date +%s%N)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PINNED_PYTHON=/network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/bin/python
[[ -x "$PINNED_PYTHON" ]] || { echo 'pinned environment interpreter missing' >&2; exit 2; }
exec timeout --signal=TERM --kill-after=10 "${RC_JOB_WALL_SECONDS}s" \
  "$PINNED_PYTHON" -B "$(dirname "$0")/tgrab_figure10_canary_supervisor.py" \
  --admission "$RC_ADMISSION" --sha256 "$RC_ADMISSION_SHA256"
