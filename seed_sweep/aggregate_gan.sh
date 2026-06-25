#!/bin/bash
#SBATCH -J seed_sweep_gan_agg
#SBATCH --output=logs/seed_sweep_gan_agg_%j.out
#SBATCH --error=logs/seed_sweep_gan_agg_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate

python -u seed_sweep/aggregate.py --tag gan
