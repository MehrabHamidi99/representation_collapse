#!/usr/bin/env bash
# Read-only host/storage/scheduler probe. Never calls sbatch or cluv sync.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || { echo 'usage: preflight_remote.sh HOST [JUMP_HOST]' >&2; exit 2; }
host="$1"
[[ "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || exit 2
ssh_args=(-o BatchMode=yes -o ConnectTimeout=8)
remote_probe='hostname; printf "SCRATCH=%s\n" "$SCRATCH"; readlink -f "$HOME/scratch"; command -v uv || true; test ! -x "$HOME/.local/bin/uv" || "$HOME/.local/bin/uv" --version; sinfo -h -o "%P %G" | head -20; sacctmgr -n -P show assoc where user=$USER format=Account,Partition,QOS 2>/dev/null | head -20; squeue --me -h -o "%j|%i|%T|%P|%b|%l" | head -20'
if [[ $# -eq 2 ]]; then
    jump_host="$2"
    [[ "$jump_host" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]] || exit 2
    # Existing remote SSH credentials can be used without forwarding secrets.
    printf -v quoted_probe '%q' "$remote_probe"
    ssh "${ssh_args[@]}" "$jump_host" "ssh -o BatchMode=yes -o ConnectTimeout=8 $host $quoted_probe"
else
    ssh "${ssh_args[@]}" "$host" "$remote_probe"
fi
