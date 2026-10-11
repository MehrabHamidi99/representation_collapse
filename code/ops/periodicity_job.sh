#!/usr/bin/env bash
# Pinned-source checkpoint-only diagnostic. No training or network package fetch.
set -euo pipefail
fail() { printf 'periodicity_job: %s\n' "$*" >&2; exit 2; }
source_commit='' run_id='' cluster='' model='' task='' seed='0'
checkpoint='' checkpoint_sha256=''
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || fail 'missing argument value'
    case "$1" in
        --source-commit) source_commit="$2";; --run-id) run_id="$2";;
        --cluster) cluster="$2";; --model) model="$2";;
        --task) task="$2";; --seed) seed="$2";;
        --checkpoint) checkpoint="$2";; --checkpoint-sha256) checkpoint_sha256="$2";;
        *) fail 'unknown argument';;
    esac
    shift 2
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ && "$checkpoint_sha256" =~ ^[0-9a-f]{64}$ ]] || fail 'full source and checkpoint hashes required'
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid run ID'
[[ "$cluster" == narval || "$cluster" == rorqual ]] || fail 'unsupported cluster'
[[ "$model" == gru || "$model" == lstm ]] || fail 'invalid model'
[[ "$task" == categorical || "$task" == continuous ]] || fail 'invalid task'
[[ "$seed" == 0 && -n "${SLURM_JOB_ID:-}" ]] || fail 'seed0 and Slurm allocation required'
[[ "$checkpoint" == /scratch/mehrab/representation-collapse/logs/cluv/*/pilot/checkpoint.pt ]] || fail 'checkpoint must be an original project scratch artifact'
[[ -f "$checkpoint" && ! -L "$checkpoint" ]] || fail 'checkpoint missing or symlink'
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == "$checkpoint_sha256" ]] || fail 'checkpoint hash mismatch'
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
snapshot="${SLURM_TMPDIR:-$job_root}/rc-periodicity-source-$SLURM_JOB_ID"
git clone --quiet --no-hardlinks --no-checkout "$checkout" "$snapshot"
git -C "$snapshot" checkout --quiet --detach "$source_commit"
cd "$snapshot"
export GIT_COMMIT="$source_commit" CLUV_CLUSTER_NAME="$cluster" UV_OFFLINE=1 WANDB_MODE=offline
export UV_CACHE_DIR="$project_root/cache/uv" UV_PYTHON_INSTALL_DIR="$project_root/cache/python"
export UV_PROJECT_ENVIRONMENT="$project_root/venvs/job-$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'source_commit=%s\nrun_id=%s\ncluster=%s\nslurm_job_id=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' "$source_commit" "$run_id" "$cluster" "$SLURM_JOB_ID" "$checkpoint" "$checkpoint_sha256" > "$results_root/provenance.txt"
sha256sum uv.lock >> "$results_root/provenance.txt"
uv --version >> "$results_root/provenance.txt"
python_path="$(uv python find --offline --no-project --system --managed-python "$(cat .python-version)")"
[[ "$python_path" == "$project_root/cache/python"/* ]] || fail 'scratch Python absent'
uv sync --locked --offline --no-dev --group experiments --python "$python_path"
srun --ntasks=1 uv run --locked --offline --no-sync python -m representation_collapse.periodicity \
    --config configs/experiments/periodicity-analysis-v1.json --output "$results_root/periodicity" \
    --model "$model" --task "$task" --seed "$seed" --device cuda \
    --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha256" \
    --source-commit "$source_commit" --cluster "$cluster"
[[ "$(sha256sum "$checkpoint" | cut -d' ' -f1)" == "$checkpoint_sha256" ]] || fail 'checkpoint changed during analysis'
printf 'results_root=%s\n' "$results_root"
