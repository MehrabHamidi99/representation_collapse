#!/usr/bin/env bash
# Run in an approved cluster checkout on a login node, before allocating.
# Creates only project scratch directories and a locked project environment.
set -euo pipefail
sync_groups=()
case "${1:-}" in
    '') ;;
    --experiments) sync_groups=(--group experiments); shift ;;
    *) echo 'usage: prepare_environment.sh [--experiments]' >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { echo 'unexpected arguments' >&2; exit 2; }
[[ -f pyproject.toml && -f uv.lock ]] || { echo 'run from project root with uv.lock' >&2; exit 2; }
scratch_mount="${SCRATCH:-$HOME/scratch}"
[[ -d "$scratch_mount" ]] || { echo 'verify SCRATCH mount first' >&2; exit 2; }
scratch_mount="$(cd "$scratch_mount" && pwd -P)"
[[ "$scratch_mount" != "$HOME" && "$scratch_mount" != / ]] || exit 2
export PATH="$HOME/.local/bin:$PATH"
scratch_root="$scratch_mount/representation-collapse"
for name in cache/uv cache/python data runs logs/cluv venvs; do
    mkdir -p "$scratch_root/$name"
done
export UV_CACHE_DIR="$scratch_root/cache/uv"
export UV_PYTHON_INSTALL_DIR="$scratch_root/cache/python"
# cluv sync creates source/.venv: the native project_dir keeps this on scratch.
checkout="$(pwd -P)"
[[ "$checkout" == "$scratch_root"/* ]] || { echo 'checkout must be below project scratch root' >&2; exit 2; }
export UV_PROJECT_ENVIRONMENT="$checkout/.venv"
python_request="$(cat .python-version)"
[[ "$python_request" =~ ^3\.11(\.[0-9]+)?$ ]] || { echo 'review the project Python version before staging' >&2; exit 2; }
uv python install --no-bin "$python_request"
python_path="$(uv python find --no-project --system --managed-python "$python_request")"
[[ "$python_path" == "$scratch_root/cache/python"/* ]] || { echo 'Python is outside scratch' >&2; exit 2; }
uv sync --locked --no-dev "${sync_groups[@]}" --python "$python_path"
uv run --locked --offline --no-sync python -c 'import numpy, representation_collapse; print("locked calibration imports ready")'
if [[ ${#sync_groups[@]} -gt 0 ]]; then
    uv run --locked --offline --no-sync python -c 'import torch; assert torch.__version__.split("+")[0] == "2.6.0"; print("locked experiment imports ready", torch.__version__)'
fi
# A pre-existing .venv alone cannot prove a fresh compute job works offline.
validation_root="$(mktemp -d "$scratch_root/venvs/offline-preflight.XXXXXXXX")"
export UV_PROJECT_ENVIRONMENT="$validation_root/env"
uv sync --locked --offline --no-dev "${sync_groups[@]}" --python "$python_path"
uv run --locked --offline --no-sync python -c 'import numpy, representation_collapse; print("fresh offline environment ready")'
if [[ ${#sync_groups[@]} -gt 0 ]]; then
    uv run --locked --offline --no-sync python -c 'import torch; assert torch.__version__.split("+")[0] == "2.6.0"; print("fresh offline experiment environment ready", torch.__version__)'
fi
printf 'scratch_root=%s\ncheckout=%s\n' "$scratch_root" "$checkout"
printf 'python_path=%s\noffline_validation_environment=%s\n' "$python_path" "$UV_PROJECT_ENVIRONMENT"
