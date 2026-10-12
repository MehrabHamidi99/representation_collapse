#!/usr/bin/env bash
# One immutable, approved discovery fit. No network, retry, requeue or race.
#SBATCH --job-name=rc-competence-canary-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
fail() { printf 'competence_job: %s\n' "$*" >&2; exit 2; }
source_commit='' run_id='' cluster='' model='' task='' seed='' learning_rate='' cell=''
config='' config_sha256='' canary_config='' canary_sha256=''
approval_record='' approval_sha256='' account='' partition='' qos='' gres=''
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 && "$2" != --* ]] || fail 'missing argument value'
    case "$1" in
        --source-commit) [[ -z "$source_commit" ]] || fail 'duplicate source'; source_commit="$2";;
        --run-id) [[ -z "$run_id" ]] || fail 'duplicate run'; run_id="$2";;
        --cluster) [[ -z "$cluster" ]] || fail 'duplicate cluster'; cluster="$2";;
        --model) [[ -z "$model" ]] || fail 'duplicate model'; model="$2";;
        --task) [[ -z "$task" ]] || fail 'duplicate task'; task="$2";;
        --seed) [[ -z "$seed" ]] || fail 'duplicate seed'; seed="$2";;
        --learning-rate) [[ -z "$learning_rate" ]] || fail 'duplicate learning rate'; learning_rate="$2";;
        --cell) [[ -z "$cell" ]] || fail 'duplicate cell'; cell="$2";;
        --config) [[ -z "$config" ]] || fail 'duplicate config'; config="$2";;
        --config-sha256) [[ -z "$config_sha256" ]] || fail 'duplicate config hash'; config_sha256="$2";;
        --canary-config) [[ -z "$canary_config" ]] || fail 'duplicate canary'; canary_config="$2";;
        --canary-sha256) [[ -z "$canary_sha256" ]] || fail 'duplicate canary hash'; canary_sha256="$2";;
        --approval-record) [[ -z "$approval_record" ]] || fail 'duplicate approval'; approval_record="$2";;
        --approval-sha256) [[ -z "$approval_sha256" ]] || fail 'duplicate approval hash'; approval_sha256="$2";;
        --account) [[ -z "$account" ]] || fail 'duplicate account'; account="$2";;
        --partition) [[ -z "$partition" ]] || fail 'duplicate partition'; partition="$2";;
        --qos) [[ -z "$qos" ]] || fail 'duplicate QOS'; qos="$2";;
        --gres) [[ -z "$gres" ]] || fail 'duplicate GRES'; gres="$2";;
        *) fail 'unknown argument';;
    esac
    shift 2
done
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail 'full lowercase source SHA required'
for value in "$config_sha256" "$canary_sha256" "$approval_sha256"; do
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail 'full config/canary/approval hashes required'
done
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid run ID'
[[ "$cluster" == fir || "$cluster" == nibi || "$cluster" == narval || "$cluster" == rorqual ]] || fail 'unsupported explicit target; no fallback'
[[ "$model" == gru || "$model" == lstm ]] || fail 'invalid model'
[[ "$task" == categorical || "$task" == continuous ]] || fail 'invalid task'
[[ "$seed" == 0 && "$learning_rate" == 0.001 && "$cell" =~ ^[0-3]$ ]] || fail 'only the frozen seed0/LR0.001 four-fit canary is allowed'
[[ "$config" == configs/experiments/competence-recovery-v2.json && "$canary_config" == configs/experiments/competence-canary-v2.json ]] || fail 'versioned protocol/canary paths required'
for value in "$account" "$partition" "$qos"; do
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || fail 'invalid scheduler identifier'
done
[[ "$gres" =~ ^gpu:[A-Za-z0-9_.-]+:1$ ]] || fail 'exactly one typed GPU is required'
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ && -z "${SLURM_ARRAY_JOB_ID:-}" && "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail 'one unrequeued nonarray Slurm allocation is required'

plain_path() {
    local cursor="$1"
    [[ "$cursor" == /* && "$cursor" != *'/../'* && "$cursor" != *'/./'* && "$cursor" != */.. && "$cursor" != */. && "$cursor" != *'//'* ]] || fail 'invalid absolute path'
    while [[ "$cursor" != / ]]; do
        [[ ! -L "$cursor" ]] || fail 'symlinked path component rejected'
        cursor="${cursor%/*}"; [[ -n "$cursor" ]] || cursor=/
    done
}
[[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]] || fail 'explicit existing SCRATCH required'
plain_path "$SCRATCH"
scratch_mount="$(cd "$SCRATCH" && pwd -P)"
[[ "$scratch_mount" == /scratch/* && "$scratch_mount" != "$HOME" ]] || fail 'approved cluster scratch root required'
project_root="$scratch_mount/representation-collapse"
checkout="$(pwd -P)"
plain_path "$checkout"; plain_path "$approval_record"
[[ "$checkout" == "$project_root/"* && "$approval_record" == "$project_root/runs/ops/"*.json ]] || fail 'source and approval must be on this project scratch'
[[ -f "$approval_record" && ! -L "$approval_record" ]] || fail 'exact owner-approval receipt absent'
[[ "$(sha256sum "$approval_record" | cut -d' ' -f1)" == "$approval_sha256" ]] || fail 'approval receipt hash mismatch'
[[ "$(git rev-parse --show-toplevel)" == "$checkout" && "$(git rev-parse HEAD)" == "$source_commit" ]] || fail 'clean pinned repository root required'
[[ -z "$(git status --porcelain --untracked-files=all)" && -z "$(git ls-files -s | awk '$1 == "120000" {print $4}')" ]] || fail 'dirty or symlinked source rejected'
[[ "$(sha256sum "$config" | cut -d' ' -f1)" == "$config_sha256" && "$(sha256sum "$canary_config" | cut -d' ' -f1)" == "$canary_sha256" ]] || fail 'frozen config/canary hash mismatch'
allocation="$(scontrol show job --oneliner "$SLURM_JOB_ID")"
python3 - "$approval_record" "$canary_config" "$config" "$source_commit" "$cluster" "$model" "$task" "$cell" "$run_id" "$account" "$partition" "$qos" "$gres" "$checkout" "$scratch_mount" "$allocation" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
a,canary,protocol,sha,host,model,task,cell,run,account,partition,qos,gres,checkout,scratch,allocation=sys.argv[1:]
approval=json.loads(Path(a).read_text()); plan=json.loads(Path(canary).read_text()); frozen=json.loads(Path(protocol).read_text())
binding=approval.get('approval_binding', {}); expected=plan['cells'][int(cell)]; target=plan['targets'][host]
def require(condition, message):
    if not condition: raise SystemExit('competence_job: '+message)
require(approval.get('approved') is True and approval.get('approved_by')=='owner' and approval.get('scientific_acceptance') is False, 'explicit owner approval required')
require(binding.get('phase_id')=='competence-canary-v2' and binding.get('source_commit')==sha and binding.get('canary_sha256')==hashlib.sha256(Path(canary).read_bytes()).hexdigest() and binding.get('protocol_sha256')==hashlib.sha256(Path(protocol).read_bytes()).hexdigest(), 'approval source/config binding mismatch')
require(binding.get('cells')==plan['cells'] and binding.get('targets')==plan['targets'] and binding.get('resource_envelope')==plan['resource_envelope'], 'approval cells/targets/resources mismatch')
require(plan['resource_envelope']=={'max_submissions':4,'max_concurrent_gpu_jobs':2,'max_requested_gpu_hours':2,'gpus_per_job':1,'cpus_per_job':4,'memory_gib':16,'walltime':'00:30:00','nodes':1,'tasks':1,'race':False,'automatic_retries':False,'requeue':False} and plan.get('scientific_acceptance') is False and plan.get('confirmation') is False and plan.get('grid_selection') is False, 'bounded discovery scope mismatch')
require(expected['cluster']==host and expected['model']==model and expected['task']==task and expected['seed']==0 and plan['learning_rate']==0.001, 'wrong cell or learning rate')
require(run==binding.get('run_prefix','')+'-'+cell+'-'+model+'-'+task+'-s0-lr001', 'run prefix/cell binding mismatch')
require(target['scheduler_verified'] is True and target['offline_environment_verified'] is True and target['source_directory']==checkout and target['scratch_root']==scratch, 'approved staging evidence/source path mismatch')
require(all(target[k]==v for k,v in [('account',account),('partition',partition),('qos',qos),('gres',gres)]), 'scheduler target differs from approval')
require(frozen['protocol_id']=='competence-recovery-v2' and frozen['maximum_updates_per_fit']==3500 and frozen['diagnostics']['fixed_batch_updates']==100 and frozen['diagnostics']['iid_immediate_updates_after_fixed_batch']==400, 'frozen schedule mismatch')
fields=dict(word.split('=',1) for word in allocation.split() if '=' in word)
require(fields.get('TimeLimit')=='00:30:00' and fields.get('NumNodes')=='1' and fields.get('NumCPUs')=='4' and fields.get('NumTasks')=='1' and fields.get('Requeue')=='0', 'exact30minute/one-node/one-task/four-CPU unrequeued allocation required')
require(fields.get('Account')==account and fields.get('Partition')==partition and fields.get('QOS')==qos, 'allocated account/partition/QOS differs')
tres=dict(piece.split('=',1) for piece in fields.get('ReqTRES','').split(',') if '=' in piece)
require(tres.get('gres/gpu')=='1' and tres.get('mem') in ('16G','16384M') and tres.get('cpu')=='4' and tres.get('node')=='1', 'exactGPU1/16GiB/CPU4 resource request required')
require(fields.get('TresPerNode') in (gres,'gres/'+gres) or fields.get('Gres')==gres, 'allocated typed GPU differs from approval')
require(os.environ.get('SLURM_CPUS_PER_TASK')=='4' and os.environ.get('SLURM_NTASKS')=='1' and os.environ.get('SLURM_JOB_NUM_NODES')=='1', 'allocation environment differs')
require(os.environ.get('SLURM_MEM_PER_NODE')=='16384' or os.environ.get('SLURM_MEM_PER_CPU')=='4096', 'exact16GiB requested memory required')
PY

results_root="$project_root/logs/cluv/${cluster}_$SLURM_JOB_ID"
job_root="$project_root/runs/$run_id/job-$SLURM_JOB_ID"
output="$results_root/competence"
venv="$project_root/venvs/competence-${cluster}-$SLURM_JOB_ID"
for path in "$project_root" "$results_root" "$job_root" "$output" "$venv"; do plain_path "$path"; done
[[ ! -e "$job_root" && ! -e "$output" && ! -e "$venv" && ! -e "$results_root/job-status.json" && ! -e "$results_root/job-provenance.json" ]] || fail 'attempt/output/environment already exists; no retry or overwrite'
mkdir -p "$results_root" "$project_root/runs/$run_id" "$project_root/venvs"
mkdir "$job_root"
stage=source_snapshot completed=0
write_status() {
    local state="$1" code="$2" temporary="$results_root/job-status.json.tmp"
    plain_path "$temporary"
    [[ ! -e "$temporary" ]] || return 2
    printf '{"schema_version":1,"state":"%s","exit_code":%s,"stage":"%s","source_commit":"%s","run_id":"%s","cluster":"%s","slurm_job_id":"%s","model":"%s","task":"%s","seed":0,"learning_rate":0.001,"scientific_acceptance":false,"updated_at_utc":"%s"}\n' \
        "$state" "$code" "$stage" "$source_commit" "$run_id" "$cluster" "$SLURM_JOB_ID" "$model" "$task" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
    mv "$temporary" "$results_root/job-status.json"
}
on_exit() {
    local code="$?"; trap - EXIT
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
# Shared scratch avoids Slurm step/batch differences in node-local /tmp.
snapshot="$job_root/immutable-source"
git clone --quiet --no-hardlinks --no-checkout -- "$checkout" "$snapshot"
git -C "$snapshot" checkout --quiet --detach "$source_commit"
cd "$snapshot"
[[ "$(git rev-parse HEAD)" == "$source_commit" && -z "$(git status --porcelain)" ]] || fail 'snapshot does not match immutable source'
stage=offline_environment
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$project_root/cache/uv" UV_PYTHON_INSTALL_DIR="$project_root/cache/python"
export UV_PROJECT_ENVIRONMENT="$venv" UV_OFFLINE=1 UV_NO_PROGRESS=1 WANDB_MODE=offline
export GIT_COMMIT="$source_commit" CLUV_CLUSTER_NAME="$cluster"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for path in "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"; do plain_path "$path"; [[ -d "$path" ]] || fail 'pre-staged scratch cache absent'; done
[[ "$(uv --version)" == 'uv 0.9.18'* ]] || fail 'pinned uv0.9.18 required'
python_path="$(uv python find --offline --no-project --system --managed-python "$(cat .python-version)")"
[[ "$python_path" == "$UV_PYTHON_INSTALL_DIR/"* && -x "$python_path" ]] || fail 'scratch-managed Python absent'
plain_path "$python_path"
uv sync --locked --offline --no-dev --group experiments --python "$python_path"
stage=provenance
"$venv/bin/python" - "$results_root/job-provenance.json" "$approval_record" "$source_commit" "$run_id" "$cluster" "$model" "$task" "$cell" "$config" "$canary_config" "$allocation" <<'PY'
import hashlib,json,os,platform,subprocess,sys
from pathlib import Path
import torch
target,approval,sha,run,cluster,model,task,cell,config,canary,allocation=sys.argv[1:]
if torch.__version__.split('+')[0]!='2.6.0' or not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise SystemExit('competence_job: pinnedtorch2.6 and exactlyonevisibleGPU required')
files=('uv.lock','pyproject.toml','.python-version','code/ops/competence_job.sh','code/ops/submit_competence_recovery.py','src/representation_collapse/competence_training.py','src/representation_collapse/competence.py','src/representation_collapse/competence_selection.py','src/representation_collapse/native_state.py','src/representation_collapse/diagnostics.py','src/representation_collapse/vector_task.py','src/representation_collapse/pilot.py',config,canary)
packet={'schema_version':1,'kind':'pinned_competence_discovery_canary','source_commit':sha,'run_id':run,'cluster':cluster,'model':model,'task':task,'seed':0,'learning_rate':0.001,'cell':int(cell),'scientific_acceptance':False,'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files},'approval_receipt_sha256':hashlib.sha256(Path(approval).read_bytes()).hexdigest(),'requested_resources':{'nodes':1,'tasks':1,'cpus':4,'memory_gib':16,'walltime':'00:30:00','gpus':1},'allocation':allocation,'slurm_job_id':os.environ['SLURM_JOB_ID'],'python':platform.python_version(),'uv':subprocess.check_output(['uv','--version'],text=True).strip(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'offline':True,'retries':0,'environment':os.environ['UV_PROJECT_ENVIRONMENT'],'threads':{key:os.environ[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}
path=Path(target); temporary=path.with_name(path.name+'.tmp')
with temporary.open('x') as stream:
 json.dump(packet,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
temporary.rename(path)
PY
stage=competence_training
write_status running 0
srun --ntasks=1 --cpus-per-task=4 --chdir="$snapshot" uv run --locked --offline --no-sync python -m representation_collapse.competence_training \
    --config "$config" --output "$output" --model "$model" --task "$task" \
    --seed "$seed" --learning-rate "$learning_rate" --device cuda
stage=output_validation
"$venv/bin/python" - "$output/status.json" <<'PY'
import json,sys
from pathlib import Path
packet=json.loads(Path(sys.argv[1]).read_text())
if packet.get('state')!='completed' or packet.get('scientific_acceptance') is not False or packet.get('outcome') not in ('completed_schedule','stopped_fixed_batch_failure','stopped_coordinate_gate_failure'): raise SystemExit('competence_job: completed bounded discovery status required')
PY
stage=completed
write_status completed 0
completed=1
printf 'competence_output=%s\n' "$output"
