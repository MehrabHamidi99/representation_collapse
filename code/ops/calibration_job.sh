#!/usr/bin/env bash
# SLURM resources are provided by [tool.cluv.sbatch_args], not hidden here.
set -euo pipefail

fail() { printf '%s\n' "calibration_job: $*" >&2; exit 2; }
[[ "${1:-}" == "--source-commit" && $# -eq 6 && "${3:-}" == "--run-id" && "${5:-}" == "--cluster" ]] ||
    fail 'usage: calibration_job.sh --source-commit FULL_SHA --run-id ID --cluster NAME'
source_commit="$2"
run_id="$4"
cluster="$6"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'require a full 40-character commit SHA'
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid run ID'
[[ "$cluster" =~ ^[a-z][a-z0-9-]*$ ]] || fail 'invalid cluster name'
[[ -n "${SLURM_JOB_ID:-}" ]] || fail 'run this script in a SLURM allocation'

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || fail 'uv must be installed during approved staging'
source_checkout="$(pwd -P)"
git cat-file -e "${source_commit}^{commit}" || fail 'approved source commit is absent'
scratch_mount="${SCRATCH:-$HOME/scratch}"
[[ -d "$scratch_mount" ]] || fail 'scratch mount is absent; verify SCRATCH first'
scratch_mount="$(cd "$scratch_mount" && pwd -P)"
[[ "$scratch_mount" != "$HOME" && "$scratch_mount" != / ]] || fail 'unsafe scratch mount'
scratch_root="$scratch_mount/representation-collapse"
job_root="$scratch_root/runs/$run_id/job-$SLURM_JOB_ID"
results_root="$scratch_root/logs/cluv/${cluster}_$SLURM_JOB_ID"
mkdir -p "$scratch_root/cache/uv" "$scratch_root/cache/python" "$scratch_root/venvs"
mkdir -p "$scratch_root/runs/$run_id"
mkdir "$job_root" || fail 'job output already exists; choose a new run ID'
mkdir -p "$results_root"
snapshot_parent="${SLURM_TMPDIR:-$job_root}"
[[ -d "$snapshot_parent" ]] || fail 'SLURM_TMPDIR is not a directory'
snapshot="$snapshot_parent/rc-source-$SLURM_JOB_ID"
git clone --quiet --no-hardlinks --no-checkout "$source_checkout" "$snapshot"
git -C "$snapshot" checkout --quiet --detach "$source_commit"
cd "$snapshot"

# Export here explicitly: cluv 0.1.5 does not export every newly defined
# environment variable in its submission shell. SHA arrives as a script arg.
export GIT_COMMIT="$source_commit"
export UV_CACHE_DIR="$scratch_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$scratch_root/cache/python"
export UV_PROJECT_ENVIRONMENT="$scratch_root/venvs/job-$SLURM_JOB_ID"
export UV_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'source_commit=%s\nrun_id=%s\nslurm_job_id=%s\nsource_checkout=%s\n' \
    "$source_commit" "$run_id" "$SLURM_JOB_ID" "$source_checkout" > "$job_root/provenance.txt"
sha256sum uv.lock >> "$job_root/provenance.txt"
uv --version >> "$job_root/provenance.txt"
cp "$job_root/provenance.txt" "$results_root/provenance.txt"
# Dependencies must have been cached on a login node before this allocation.
# Compute nodes are never used to fetch dependencies or train in this smoke.
python_path="$(uv python find --offline --no-project --managed-python "$(cat .python-version)")"
[[ "$python_path" == "$scratch_root/cache/python"/* ]] || fail 'prepared scratch Python is absent'
uv sync --locked --offline --no-dev --python "$python_path"
srun --ntasks=1 --cpus-per-task=1 uv run --locked --offline --no-sync \
    python -m representation_collapse.calibrate --output "$results_root/calibration"
printf 'calibration_output=%s\n' "$results_root/calibration"
