#!/bin/bash
# The scientific worker runs only after cluv has staged the committed project.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: job.sh SOURCE_COMMIT RUN_ID' >&2; exit 2; }
revision="$1"
run_id="$2"
[[ "$revision" =~ ^[0-9a-f]{40}$ && "$run_id" =~ ^multibit-[A-Za-z0-9-]+$ ]] || exit 2
launch_dir="$(pwd)"
source_dir="$(git rev-parse --show-toplevel)"
[[ "$(git rev-parse HEAD)" = "$revision" ]] || exit 3
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || exit 3
[[ -n "${SCRATCH:-}" && -n "${SLURM_JOB_ID:-}" ]] || exit 3
run_dir="$SCRATCH/representation-collapse/runs/multibit-memory/$run_id"
[[ ! -e "$run_dir" ]] || exit 4
mkdir -p "$(dirname "$run_dir")"
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline UV_OFFLINE=1
python="$launch_dir/.venv/bin/python"
[[ -x "$python" ]] || exit 3
cd "$source_dir"
deadline="$(( $(date +%s) + 10500 ))"
exec srun --ntasks=1 --cpus-per-task=1 timeout --signal=TERM --kill-after=20 10620s \
  "$python" -B -m representation_collapse.multibit_memory \
  --config "$source_dir/configs/experiments/multibit-memory-v1.json" \
  --protocol "$source_dir/docs/November/2026-11-16/cross-cluster/experiment/multibit-memory-v1.md" \
  --output "$run_dir" --run-id "$run_id" --deadline-unix "$deadline" \
  --stop-file "$run_dir.STOP"
