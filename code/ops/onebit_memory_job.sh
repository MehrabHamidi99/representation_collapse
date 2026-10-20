#!/bin/bash
set -euo pipefail
[[ $# -eq 2 ]] || exit 2
revision="$1"
run_id="$2"
[[ "$revision" =~ ^[0-9a-f]{40}$ && "$run_id" =~ ^onebit-[A-Za-z0-9-]+$ ]] || exit 2
base=/scratch/m/mehrab/representation-collapse
source_dir="$base/source-onebit-$revision"
python="$base/environments/onebit-cpu-20261020/venv/bin/python"
[[ "$(git -C "$source_dir" rev-parse HEAD)" = "$revision" ]] || exit 3
[[ -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]] || exit 3
cd "$source_dir"
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
[[ "$(sha256sum "$base/environments/onebit-cpu-20261020/requirements.lock" | cut -d' ' -f1)" = 5069f2d7da903f16a485edc43b24e67a90d6278af25b97770a75f9f3f4cecfd4 ]] || exit 3
"$python" -B -m unittest tests.test_onebit_memory tests.test_vector_task -q
run_dir="$base/runs/onebit-memory/$run_id"
[[ ! -e "$run_dir" ]] || exit 4
mkdir -p "$(dirname "$run_dir")"
deadline="$(( $(date +%s) + 7080 ))"
exec timeout --signal=TERM --kill-after=15 7140s "$python" -B -m representation_collapse.onebit_memory \
  --protocol "$source_dir/doc/auto/onebit-memory-discovery-v1.md" \
  --output "$run_dir" --run-id "$run_id" --deadline-unix "$deadline" --stop-file "$run_dir.STOP"
