#!/bin/bash
# Run on the login node after cluv sync has finished; never in a Slurm worker.
set -euo pipefail
[[ -n "${SCRATCH:-}" ]] || exit 2
launch_dir="$(pwd)"
snapshot="$SCRATCH/representation-collapse/envs/load-v2-repair-20261116"
[[ -f "$launch_dir/.venv/pyvenv.cfg" && -f "$launch_dir/uv.lock" ]] || exit 2
if [[ ! -e "$snapshot" ]]; then
  mkdir -p "$(dirname "$snapshot")"
  temporary="$snapshot.preparing.$$"
  # The snapshot uses new directory entries; later uv reinstalls may unlink
  # .venv entries without removing these links. Never write to this snapshot.
  cp -al "$launch_dir/.venv" "$temporary"
  "$temporary/bin/python" -c 'import numpy, torch, wandb'
  mv -T "$temporary" "$snapshot"
fi
"$snapshot/bin/python" - "$launch_dir/uv.lock" "$snapshot" <<'PY'
import datetime, hashlib, importlib.metadata, json, sys
from pathlib import Path
expected = {'torch': '2.0.0+cpu', 'numpy': '1.24.4', 'wandb': '0.30.0'}
actual = {n: importlib.metadata.version(n) for n in expected}
assert actual == expected, actual
marker = Path(sys.argv[2]) / 'runtime-snapshot.json'
record = {'prepared_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'lock_sha256': hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),
          'versions': actual, 'snapshot': sys.argv[2],
          'policy': 'published runtime snapshot; do not reinstall or modify'}
if marker.exists():
    saved = json.loads(marker.read_text())
    assert saved['lock_sha256'] == record['lock_sha256'] and saved['versions'] == actual
else:
    marker.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
PY
