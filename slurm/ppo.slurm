#!/bin/bash
# HYAK (klone) PPO training run — the demon that has to beat the lawnmower.
# CPU is fine (the policy is tiny; episodes dominate). Bump cpus-per-task and
# episodes-per-update together if you widen the batch.
#
# Submit: sbatch slurm/ppo.slurm
#
#SBATCH --job-name=demon-ppo
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/ppo_%j.out

set -euo pipefail
mkdir -p results slurm/logs

python scripts/train_ppo.py --L 32 --budget 4 --T 128 \
    --updates 400 --episodes-per-update 64 --shaped \
    --out results/ppo_L32b4.json
