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

cd "$HOME/cv_job"
source "$HOME/cv_job/.venv/bin/activate"

echo "python: $(which python)"
python -V
python -m pip -V

set +e
python - <<'PY'
import importlib.util, sys
mods = ["numpy", "matplotlib", "scipy", "torch", "torchvision"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing:", missing)
sys.exit(0 if not missing else 2)
PY
status=$?
set -e

if [[ $status -eq 2 ]]; then
  echo "[info] installing missing deps into this venv..."
  python -m pip install -q --upgrade pip setuptools wheel
  python -m pip install -q --no-cache-dir scipy numpy matplotlib
fi

python -c "import scipy, torch, torchvision; print('ok', scipy.__version__, torch.__version__, torchvision.__version__)"
export DATA_DIR="$HOME/data"
export OUT_DIR="$HOME/outputs_ood/12881"
mkdir -p "$DATA_DIR" "$OUT_DIR"
nvidia-smi || true
python -u calculate.py