#!/bin/bash
#SBATCH --job-name=aibo_cot_annotations
#SBATCH --nodelist=chi-gpu1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export LD_LIBRARY_PATH="/data/chi-gpu4/eihwadmin/cuda/lib64:${LD_LIBRARY_PATH:-}"
PYTHON=/data/chi-gpu4/ge94xov/lam4ser-env/bin/python

"$PYTHON" data/generate_aibo_cot_annotations.py \
  --encoder wavlm-large \
  --output annotations/aibo_label_blind_rationale_v1.jsonl
