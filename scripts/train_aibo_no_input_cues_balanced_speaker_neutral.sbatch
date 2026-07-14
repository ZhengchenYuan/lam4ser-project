#!/bin/bash
#SBATCH --job-name=aibo_no_cues_neutral
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

"$PYTHON" training/train_generation_model.py \
  --dataset aibo \
  --encoder wavlm-large \
  --prompt_type speaker_feature_answer_generation \
  --speaker_baseline_mode neutral \
  --baseline_estimation_mode speaker_neutral \
  --class_weighted_answer_loss \
  --class_weight_mode balanced \
  --class_weight_power 0.5 \
  --class_weight_max 2.0 \
  --disable_input_cue_text
