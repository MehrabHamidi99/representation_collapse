#!/usr/bin/env bash
# One-shot Mila capture of the reviewed node2a runtime identity (no model/data).
#SBATCH --account=mila
#SBATCH --qos=normal
#SBATCH --partition=main-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec timeout --foreground --signal=TERM --kill-after=15s 575s \
  /network/scratch/m/mehrab.hamidi/representation-collapse/toolchains/python/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10 \
  -B /network/scratch/m/mehrab.hamidi/representation-collapse/source/code/ops/tgrab_node2a_capture_supervisor.py "$@"
