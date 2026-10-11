#!/usr/bin/env bash
# Unified pinned-source calibration/pilot runner; resources come from cluv.
set -euo pipefail
fail() { printf 'phase1_job: %s\n' "$*" >&2; exit 2; }
source_commit='' run_id='' cluster='' kind='calibration' model='' task='' seed='0'
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || fail 'missing argument value'
    case "$1" in
        --source-commit) source_commit="$2";; --run-id) run_id="$2";;
        --cluster) cluster="$2";; --kind) kind="$2";; --model) model="$2";;
        --task) task="$2";; --seed) seed="$2";; *) fail 'unknown argument';;
    esac
    shift 2
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'full source SHA required'
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid run ID'
[[ "$cluster" =~ ^[a-z][a-z0-9-]*$ && "$seed" =~ ^[0-9]+$ ]] || fail 'invalid cluster/seed'
[[ "$kind" == calibration || "$kind" == pilot ]] || fail 'invalid kind'
[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'SLURM allocation required'
if [[ "$kind" == pilot ]]; then
    [[ "$model" == gru || "$model" == lstm ]] || fail 'invalid model'
    [[ "$task" == categorical || "$task" == continuous ]] || fail 'invalid task'
fi
export PATH="$HOME/.local/bin:$PATH"
scratch_mount="${SCRATCH:-$HOME/scratch}"
[[ -d "$scratch_mount" ]] || fail 'scratch mount missing'
scratch_mount="$(cd "$scratch_mount" && pwd -P)"
[[ "$scratch_mount" != "$HOME" && "$scratch_mount" != / ]] || fail 'unsafe scratch'
project_root="$scratch_mount/representation-collapse"
results_root="$project_root/logs/cluv/${cluster}_$SLURM_JOB_ID"
job_root="$project_root/runs/$run_id/job-$SLURM_JOB_ID"
mkdir -p "$project_root/runs/$run_id" "$project_root/cache/uv" "$project_root/cache/python" "$project_root/venvs" "$results_root"
mkdir "$job_root" || fail 'attempt already exists'
checkout="$(pwd -P)"
git cat-file -e "${source_commit}^{commit}" || fail 'source SHA absent'
snapshot="${SLURM_TMPDIR:-$job_root}/rc-source-$SLURM_JOB_ID"
git clone --quiet --no-hardlinks --no-checkout "$checkout" "$snapshot"
git -C "$snapshot" checkout --quiet --detach "$source_commit"
cd "$snapshot"
export GIT_COMMIT="$source_commit" CLUV_CLUSTER_NAME="$cluster" UV_OFFLINE=1 WANDB_MODE=offline
export UV_CACHE_DIR="$project_root/cache/uv" UV_PYTHON_INSTALL_DIR="$project_root/cache/python"
export UV_PROJECT_ENVIRONMENT="$project_root/venvs/job-$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'source_commit=%s\nrun_id=%s\ncluster=%s\nslurm_job_id=%s\nkind=%s\n' "$source_commit" "$run_id" "$cluster" "$SLURM_JOB_ID" "$kind" > "$results_root/provenance.txt"
sha256sum uv.lock >> "$results_root/provenance.txt"
uv --version >> "$results_root/provenance.txt"
# cluv's uv-run controller can export VIRTUAL_ENV; ignore that interpreter.
python_path="$(uv python find --offline --no-project --system --managed-python "$(cat .python-version)")"
[[ "$python_path" == "$project_root/cache/python"/* ]] || fail 'scratch Python absent'
if [[ "$kind" == calibration ]]; then
    uv sync --locked --offline --no-dev --python "$python_path"
    srun --ntasks=1 --cpus-per-task=1 uv run --locked --offline --no-sync python -m representation_collapse.calibrate --output "$results_root/calibration" --seed "$seed"
    srun --ntasks=1 --cpus-per-task=1 uv run --locked --offline --no-sync python -m representation_collapse.mechanism_calibration --output "$results_root/mechanisms" --seed "$seed"
else
    uv sync --locked --offline --no-dev --group experiments --python "$python_path"
    srun --ntasks=1 uv run --locked --offline --no-sync python -m representation_collapse.pilot --config configs/experiments/vector-pilot-v1.json --output "$results_root/pilot" --model "$model" --task "$task" --seed "$seed" --device cuda
fi
printf 'results_root=%s\n' "$results_root"
