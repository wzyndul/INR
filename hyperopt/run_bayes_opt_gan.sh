#!/bin/bash
#SBATCH -J bayes_opt_gan
#SBATCH --output=logs/bayes_opt_gan_%j.out
#SBATCH --error=logs/bayes_opt_gan_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00

REPO_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
cd "$REPO_ROOT"
source venv/bin/activate

NETWORK=${1:-siren}
N_TRIALS=${2:-100}

python -u hyperopt/bayes_opt_gan.py \
    --network "$NETWORK" \
    --n_trials "$N_TRIALS"
