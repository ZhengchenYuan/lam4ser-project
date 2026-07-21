#!/bin/bash
#SBATCH --job-name=aibo_audioqwen3_teacher_pilot
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export LD_LIBRARY_PATH="/data/chi-gpu4/eihwadmin/cuda/lib64:${LD_LIBRARY_PATH:-}"
PYTHON=/data/chi-gpu4/ge94xov/lam4ser-env/bin/python

"$PYTHON" scripts/generate_audioqwen3_teacher_rationales.py \
  --dataset aibo \
  --split test \
  --max_samples 20 \
  --output annotations/aibo_audioqwen3_teacher_rationales_pilot.jsonl \
  --device cuda \
  --trust_remote_code \
  --resume
