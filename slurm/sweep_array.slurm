#!/bin/bash
# HYAK (klone) SLURM array for the baseline MIPT sweep at larger L.
# One array task per (L, p) cell; edit the L_LIST/P_LIST below and the
# account/partition for your allocation (e.g. -A <lab> -p ckpt-all for
# checkpoint cycles, or your group's compute partition).
#
# Submit:  sbatch slurm/sweep_array.slurm
# Merge:   python scripts/merge_results.py results/cells/*.json --out results/baseline_hyak.json
#
#SBATCH --job-name=mipt-sweep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-83
#SBATCH --output=slurm/logs/%A_%a.out

set -euo pipefail
mkdir -p results/cells slurm/logs

L_LIST=(64 128 256 512)                       # 4 sizes
P_LIST=(0.05 0.08 0.10 0.12 0.13 0.14 0.145 0.15 0.155 0.16 0.165 0.17 0.175 0.18 0.19 0.20 0.22 0.24 0.27 0.30 0.35)  # 21 rates
SHOTS=(2000 1000 400 128)                     # per size

NP=${#P_LIST[@]}
i=$SLURM_ARRAY_TASK_ID
Li=$(( i / NP )); Pi=$(( i % NP ))
L=${L_LIST[$Li]}; P=${P_LIST[$Pi]}; S=${SHOTS[$Li]}

module load foster/python/miniconda/3.11 2>/dev/null || true
python scripts/run_sweep.py --L "$L" --shots "$S" --p "$P" \
    --out "results/cells/L${L}_p${P}.json" --workers 1 --seed 7
