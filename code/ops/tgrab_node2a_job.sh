#!/usr/bin/env bash
# One-shot node2a execution wrapper; cluv submits this only after independent review.
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
exec timeout --foreground --signal=TERM --kill-after=5s 590s \
  /network/scratch/m/mehrab.hamidi/representation-collapse/environments/graph-native-cpu-20261015-calibration01/venv/bin/python \
  -B /network/scratch/m/mehrab.hamidi/representation-collapse/source/code/ops/tgrab_node2a_supervisor.py "$@"
