#!/usr/bin/env bash
# One pinned-source CPU reduction, submitted through cluv. No training/network.
#SBATCH --job-name=rc-periodicity-reduction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail

fail() { printf 'periodicity_reduction_job: %s\n' "$*" >&2; exit 2; }
source_commit='' run_id='' cluster=''
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 && "$2" != --* ]] || fail 'missing argument value'
    case "$1" in
        --source-commit) [[ -z "$source_commit" ]] || fail 'duplicate source'; source_commit="$2";;
        --run-id) [[ -z "$run_id" ]] || fail 'duplicate run ID'; run_id="$2";;
        --cluster) [[ -z "$cluster" ]] || fail 'duplicate cluster'; cluster="$2";;
        *) fail 'unknown argument';;
    esac
    shift 2
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'full lowercase source SHA required'
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid run ID'
[[ "$cluster" == tamia ]] || fail 'only Tamia is allowed'
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || fail 'Slurm allocation required'

# Check every existing ancestor before using a path; never silently resolve links.
plain_path() {
    local cursor="$1"
    [[ "$cursor" == /* && "$cursor" != *'/../'* && "$cursor" != *'/./'* && "$cursor" != */.. && "$cursor" != */. && "$cursor" != *'//'* ]] || fail 'invalid absolute path'
    while [[ "$cursor" != / ]]; do
        [[ ! -L "$cursor" ]] || fail 'symlinked path component rejected'
        cursor="${cursor%/*}"
        [[ -n "$cursor" ]] || cursor=/
    done
}

[[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]] || fail 'explicit SCRATCH directory required'
plain_path "$SCRATCH"
scratch_mount="$(cd "$SCRATCH" && pwd -P)"
[[ "$scratch_mount" != / && "$scratch_mount" != "$HOME" ]] || fail 'unsafe scratch root'
project_root="$scratch_mount/representation-collapse"
results_root="$project_root/logs/cluv/tamia_$SLURM_JOB_ID"
output="$results_root/reduction"
job_root="$project_root/runs/$run_id/job-$SLURM_JOB_ID"
venv="$project_root/venvs/periodicity-reduction-tamia-$SLURM_JOB_ID"
for path in "$project_root" "$results_root" "$output" "$job_root" "$venv"; do plain_path "$path"; done
[[ ! -e "$output" && ! -e "$job_root" && ! -e "$venv" ]] || fail 'attempt/output/environment already exists; no retry or overwrite'
# cluv owns results_root and may already have placed its Slurm log there.
mkdir -p "$results_root" "$project_root/runs/$run_id" "$project_root/venvs"
mkdir "$output"
stage='initializing'
completed=0
write_status() {
    local state="$1" code="$2" temporary="$output/status.json.tmp"
    [[ ! -L "$output/status.json" && ! -L "$temporary" ]] || return 2
    printf '{"schema_version":1,"state":"%s","exit_code":%s,"stage":"%s","source_commit":"%s","run_id":"%s","cluster":"tamia","slurm_job_id":"%s","scientific_acceptance":false,"updated_at":"%s"}\n' \
        "$state" "$code" "$stage" "$source_commit" "$run_id" "$SLURM_JOB_ID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
    mv "$temporary" "$output/status.json"
}
on_exit() {
    local code="$?"
    trap - EXIT
    if [[ "$completed" != 1 ]]; then
        [[ "$code" != 0 ]] || code=1
        write_status failed "$code" || true
    fi
    exit "$code"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running 0
printf '{"schema_version":1,"kind":"pinned_cpu_periodicity_reduction","source_validation":"pending","requested_source_commit":"%s","run_id":"%s","cluster":"tamia","slurm_job_id":"%s","scientific_acceptance":false}\n' \
    "$source_commit" "$run_id" "$SLURM_JOB_ID" > "$output/provenance.json.tmp"
mv "$output/provenance.json.tmp" "$output/provenance.json"
mkdir "$job_root"

stage='allocation_validation'
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_NTASKS:-}" == 1 && "${SLURM_JOB_NUM_NODES:-}" == 1 ]] || fail 'exactly one node/task/CPU is required'
[[ "${SLURM_MEM_PER_NODE:-}" == 2048 || "${SLURM_MEM_PER_CPU:-}" == 2048 ]] || fail 'exactly 2 GiB requested memory is required'
for key in SLURM_GPUS SLURM_GPUS_ON_NODE SLURM_GPUS_PER_NODE SLURM_GPUS_PER_TASK; do
    [[ -z "${!key:-}" || "${!key}" == 0 ]] || fail 'GPU allocation rejected'
done
for key in SLURM_JOB_GPUS SLURM_STEP_GPUS CUDA_VISIBLE_DEVICES; do
    [[ -z "${!key:-}" || "${!key}" == NoDevFiles || "${!key}" == none || "${!key}" == -1 ]] || fail 'GPU device assignment rejected'
done
allocation="$(scontrol show job --oneliner "$SLURM_JOB_ID")"
[[ " $allocation " == *' TimeLimit=00:10:00 '* ]] || fail 'ten-minute walltime required'
[[ " $allocation " != *' gres/gpu='[1-9]* && " $allocation " != *'gres/gpu:'[1-9]* ]] || fail 'GPU TRES rejected'

stage='source_validation'
checkout="$(pwd -P)"
plain_path "$checkout"
[[ "$(git rev-parse --show-toplevel)" == "$checkout" ]] || fail 'controller CWD must be repository root'
[[ "$(git rev-parse HEAD)" == "$source_commit" ]] || fail 'controller checkout does not match source SHA'
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail 'controller checkout must be clean'
[[ -z "$(git ls-files -s | awk '$1 == "120000" {print $4}')" ]] || fail 'tracked source symlinks rejected'
git cat-file -e "${source_commit}^{commit}" || fail 'source SHA absent'
snapshot_base="${SLURM_TMPDIR:-$job_root}"
plain_path "$snapshot_base"
[[ -d "$snapshot_base" ]] || fail 'snapshot parent missing'
snapshot="$snapshot_base/rc-periodicity-reduction-source-$SLURM_JOB_ID"
plain_path "$snapshot"
[[ ! -e "$snapshot" ]] || fail 'snapshot already exists'
git clone --quiet --no-hardlinks --no-checkout -- "$checkout" "$snapshot"
git -C "$snapshot" checkout --quiet --detach "$source_commit"
cd "$snapshot"
[[ "$(git rev-parse HEAD)" == "$source_commit" && -z "$(git status --porcelain)" ]] || fail 'snapshot verification failed'
[[ -f code/analysis/reduce_periodicity.py && -f uv.lock && -f .python-version ]] || fail 'pinned reduction source missing'

stage='input_validation'
artifacts=()
for directory in narval_5254858 narval_5254871 rorqual_22984956 rorqual_22984963; do
    artifact="$project_root/logs/cluv/$directory/periodicity"
    plain_path "$artifact"
    [[ -d "$artifact" ]] || fail 'required mirrored periodicity directory missing'
    shopt -s nullglob
    entries=("$artifact"/* "$artifact"/.[!.]* "$artifact"/..?*)
    [[ "${#entries[@]}" == 8 ]] || fail 'artifact directory must contain the eight frozen files only'
    for path in "${entries[@]}"; do [[ -f "$path" && ! -L "$path" ]] || fail 'nonregular artifact entry rejected'; done
    for filename in config.json provenance.json status.json metrics.json inputs.npz trajectories.npz tangents.npz manifest.json; do
        [[ -f "$artifact/$filename" && ! -L "$artifact/$filename" ]] || fail 'required frozen artifact file missing'
    done
    artifacts+=("$artifact")
done

stage='offline_environment'
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$project_root/cache/uv" UV_PYTHON_INSTALL_DIR="$project_root/cache/python"
export UV_PROJECT_ENVIRONMENT="$venv" UV_OFFLINE=1 UV_NO_PROGRESS=1
export GIT_COMMIT="$source_commit" CLUV_CLUSTER_NAME=tamia WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
for path in "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"; do
    plain_path "$path"
    [[ -d "$path" ]] || fail 'pre-staged scratch cache/Python directory missing'
done
# Ignore cluv's inherited environment; require the project-managed scratch Python.
python_path="$(uv python find --offline --no-project --system --managed-python "$(cat .python-version)")"
[[ "$python_path" == "$UV_PYTHON_INSTALL_DIR"/* && -x "$python_path" ]] || fail 'scratch-managed Python missing'
plain_path "$python_path"
export RC_REDUCTION_SOURCE_COMMIT="$source_commit" RC_REDUCTION_RUN_ID="$run_id" RC_REDUCTION_OUTPUT="$output"
"$python_path" - "$output/provenance.json" "${artifacts[@]}" <<'PY'
import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
packet = {"schema_version": 1, "kind": "pinned_cpu_periodicity_reduction", "scientific_acceptance": False,
          "source_commit": os.environ["RC_REDUCTION_SOURCE_COMMIT"], "run_id": os.environ["RC_REDUCTION_RUN_ID"],
          "cluster": "tamia", "python": platform.python_version(), "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
          "source_sha256": {name: digest(name) for name in ("uv.lock", "pyproject.toml", ".python-version", "code/analysis/reduce_periodicity.py", "code/ops/periodicity_reduction_job.sh", "configs/experiments/periodicity-analysis-v1.json")},
          "input_manifests_sha256": {path: digest(Path(path) / "manifest.json") for path in sys.argv[2:]},
          "requested_resources": {"nodes": 1, "tasks": 1, "cpus": 1, "memory_gib": 2, "walltime": "00:10:00", "gpus": 0},
          "allocation": {name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS")},
          "blas_threads": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
          "environment": os.environ["UV_PROJECT_ENVIRONMENT"], "offline": True, "retries": 0}
target = Path(sys.argv[1])
temporary = target.with_name(target.name + ".tmp")
with temporary.open("x") as stream:
    json.dump(packet, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
temporary.rename(target)
PY
uv sync --locked --offline --no-dev --python "$python_path"

stage='independent_reduction'
srun --ntasks=1 --cpus-per-task=1 uv run --locked --offline --no-sync python code/analysis/reduce_periodicity.py \
    --artifact-dir "${artifacts[0]}" --artifact-dir "${artifacts[1]}" \
    --artifact-dir "${artifacts[2]}" --artifact-dir "${artifacts[3]}" \
    --expected-source-commit 458bc9cfce474b8e11e433e33bf132faf5032387 \
    --expected-module-sha256 5ff03bcddc5b0c3810eec8c02b4efbbd86da370567e876eaa150dd446bcc25e8 \
    --frozen-config configs/experiments/periodicity-analysis-v1.json --output "$output/report.json"
stage='report_validation'
"$venv/bin/python" - "$output/report.json" <<'PY'
import json, sys
from pathlib import Path
packet = json.loads(Path(sys.argv[1]).read_text())
expected = {("gru", "categorical", 0), ("gru", "continuous", 0), ("lstm", "categorical", 0), ("lstm", "continuous", 0)}
actual = {(cell["cell"]["model"], cell["cell"]["task"], cell["cell"]["seed"]) for cell in packet["cells"]}
if actual != expected or len(packet["cells"]) != 4 or packet["errors"] or packet.get("independent_raw_reduction_passed") is not True or packet.get("all_frozen_cells_present") is not True or packet.get("total_dense_windows") != 64 or packet.get("scientific_acceptance") is not False:
    raise SystemExit("four-cell reduction acceptance failed")
PY
stage='completed'
write_status completed 0
"$venv/bin/python" - "$output" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
packet = {"schema_version": 1, "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("report.json", "provenance.json", "status.json")}}
temporary = root / "manifest.json.tmp"
with temporary.open("x") as stream:
    json.dump(packet, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
temporary.rename(root / "manifest.json")
PY
completed=1
printf 'reduction_report=%s\n' "$output/report.json"
