#!/bin/bash
#SBATCH -J seed_sweep_gan
#SBATCH --output=logs/seed_sweep_gan_%A_%a.out
#SBATCH --error=logs/seed_sweep_gan_%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-399%8

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate
unset COMET_API_KEY

python -u seed_sweep/run.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --tasks_file seed_sweep/tasks_gan.json \
    --out_root seed_sweep/outputs_gan
