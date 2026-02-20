#!/usr/bin/env bash
#SBATCH --job-name=ood_rn18
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cd ~/cv_job   # ajusta si model.py está en otro lado

# activar venv
source .venv/bin/activate

export DATA_DIR="$HOME/data"
export OUT_DIR="$HOME/outputs_ood/${SLURM_JOB_ID}"
mkdir -p "$DATA_DIR" "$OUT_DIR"

nvidia-smi || true
python -u model.py
