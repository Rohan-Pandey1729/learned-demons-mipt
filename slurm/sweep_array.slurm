#!/bin/bash
# HYAK (klone) SLURM array for large-L sweeps, both placement modes.
#
# Submit:
#   sbatch --export=ALL,MODE=random slurm/sweep_array.slurm
#   sbatch --export=ALL,MODE=sweep  slurm/sweep_array.slurm
# Merge afterwards:
#   python scripts/merge_results.py results/cells/random_*.json --out results/baseline_hyak.json
#   python scripts/merge_results.py results/cells/sweep_*.json  --out results/lawnmower_hyak.json
#
# Edit --account/--partition for your allocation (ckpt-all runs on idle cycles).
#
#SBATCH --job-name=mipt-sweep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-83
#SBATCH --output=slurm/logs/%A_%a.out

set -euo pipefail
mkdir -p results/cells slurm/logs

MODE=${MODE:-random}
L_LIST=(64 128 256 512)                       # 4 sizes
SHOTS=(2000 1000 400 128)                     # per size

if [ "$MODE" = "sweep" ]; then
    # sweep placement: the action is at small p (crossing drift, 1/p saturation)
    P_LIST=(0.004 0.006 0.008 0.010 0.0125 0.015 0.0175 0.020 0.025 0.030 0.035 0.040 0.050 0.060 0.080 0.100 0.120 0.140 0.160 0.200 0.300)
else
    # random placement: dense near p_c ~ 0.16
    P_LIST=(0.05 0.08 0.10 0.12 0.13 0.14 0.145 0.15 0.155 0.16 0.165 0.17 0.175 0.18 0.19 0.20 0.22 0.24 0.27 0.30 0.35)
fi

NP=${#P_LIST[@]}                              # 21 rates -> array 0-83
i=$SLURM_ARRAY_TASK_ID
Li=$(( i / NP )); Pi=$(( i % NP ))
L=${L_LIST[$Li]}; P=${P_LIST[$Pi]}; S=${SHOTS[$Li]}

python scripts/run_sweep.py --L "$L" --shots "$S" --p "$P" --mode "$MODE" \
    --out "results/cells/${MODE}_L${L}_p${P}.json" --workers 1 --seed 7
