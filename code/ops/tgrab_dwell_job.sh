#!/bin/bash
# Fixed-checkpoint, generated-forcing diagnostic; no training or test payload.
set -euo pipefail
[[ $# -eq 2 ]] || exit 2
revision="$1"
run_id="$2"
[[ "$revision" =~ ^[0-9a-f]{40}$ && "$run_id" =~ ^dwell-[A-Za-z0-9-]+$ ]] || exit 2
base=/network/scratch/m/mehrab.hamidi/representation-collapse
source_dir="$base/source-dwell-$revision"
python="$base/environments/graph-native-cpu-20261015-calibration01/venv/bin/python"
[[ "$(git -C "$source_dir" rev-parse HEAD)" = "$revision" ]] || exit 3
[[ -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]] || exit 3
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
run_dir="$base/runs/repetition-dwell/$run_id"
[[ ! -e "$run_dir" ]] || exit 4
mkdir -p "$(dirname "$run_dir")"
stop_file="$run_dir.STOP"
deadline="$(( $(date +%s) + 7080 ))"
exec timeout --signal=TERM --kill-after=15 7140s "$python" -B -m representation_collapse.tgrab_dwell \
  --candidate-dir "$base/runs/figure10-long/long-20261019T215117Z-1891bdd5/candidates/epoch-160" \
  --data-manifest "$base/runs/primary-conversion-verified-20261016/attempt001/worker/converted/manifest.json" \
  --source-root "$base/runs/graph/upstream-pin-20261012/source" \
  --cell-path "$base/environments/graph-native-cpu-20261015-calibration01/venv/lib/python3.10/site-packages/torch_geometric_temporal/nn/recurrent/gc_lstm.py" \
  --output "$run_dir" --run-id "$run_id" --deadline-unix "$deadline" --stop-file "$stop_file"
