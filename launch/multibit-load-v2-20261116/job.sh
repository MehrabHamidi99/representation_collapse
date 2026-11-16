#!/bin/bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'usage: job.sh SOURCE_COMMIT RUN_ID' >&2; exit 2; }
revision="$1"
run_id="$2"
[[ "$revision" =~ ^[0-9a-f]{40}$ && "$run_id" =~ ^load-v2-[A-Za-z0-9-]+$ ]] || exit 2
launch_dir="$(pwd)"
source_dir="$(git rev-parse --show-toplevel)"
[[ "$(git rev-parse HEAD)" = "$revision" ]] || exit 3
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || exit 3
[[ -n "${SCRATCH:-}" && -n "${SLURM_JOB_ID:-}" ]] || exit 3
run_root="$SCRATCH/representation-collapse/runs/multibit-load-ladder/$run_id"
stop_file="$run_root.STOP"
[[ ! -e "$run_root" && ! -e "$stop_file" ]] || exit 4
export PYTHONPATH="$source_dir/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline UV_OFFLINE=1
python="$launch_dir/.venv/bin/python"
[[ -x "$python" ]] || exit 3
"$python" - "$run_root" "$revision" "$run_id" <<'PY'
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
expected = {'torch': '2.0.0+cpu', 'numpy': '1.24.4', 'wandb': '0.30.0'}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f'Pinned environment mismatch: {actual}')
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False)
record = {'source_commit': sys.argv[2], 'run_id': sys.argv[3],
          'slurm_job_id': os.environ['SLURM_JOB_ID'],
          'environment': actual, 'seeds': [0, 1, 2],
          'lock_sha256': hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest(),
          'cpus': 3, 'memory_gib': 12, 'gpus': 0}
(root / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
PY
cd "$source_dir"
deadline="$(( $(date +%s) + 10500 ))"
pids=()
request_stop() {
  touch "$stop_file"
  for child in "${pids[@]}"; do kill -TERM "$child" 2>/dev/null || true; done
}
trap request_stop TERM INT
for seed in 0 1 2; do
  timeout --signal=TERM --kill-after=20 10620s \
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=3G \
    "$python" -B -m representation_collapse.multibit_load_ladder \
    --config "$source_dir/configs/experiments/multibit-load-ladder-v2.json" \
    --protocol "$source_dir/docs/November/2026-11-16/cross-cluster/experiment/multibit-load-ladder-v2.md" \
    --seed "$seed" --output "$run_root/seed-$seed" --run-id "$run_id-s$seed" \
    --deadline-unix "$deadline" --stop-file "$stop_file" \
    >"$run_root/seed-$seed.stdout.log" 2>&1 &
  pids+=("$!")
done
codes=()
result=0
for child in "${pids[@]}"; do
  code=0
  wait "$child" || code="$?"
  if kill -0 "$child" 2>/dev/null; then
    code=0
    wait "$child" || code="$?"
  fi
  codes+=("$code")
  if [[ "$code" -ne 0 ]]; then result=1; fi
done
"$python" - "$run_root" "$stop_file" "${codes[@]}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
record = {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
          'stop_requested': Path(sys.argv[2]).exists(),
          'seed_exit_codes': {str(s): int(c) for s, c in enumerate(sys.argv[3:])}}
temporary = root / 'launcher-result.json.tmp'
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(root / 'launcher-result.json')
print(json.dumps(record), flush=True)
PY
exit "$result"
