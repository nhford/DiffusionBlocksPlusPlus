#!/bin/bash
#SBATCH --job-name=10-epoch-dblock-cifar100
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --time=28:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nford@andrew.cmu.edu

set -euo pipefail

cd /path/to/DiffusionBlocksPlusPlus
module load cuda
source .venv/bin/activate

nvidia-smi

BATCH_SIZE=128
NUM_BLOCKS=3
EPOCHS=10
WARMUP_STEPS=$((3900 * NUM_BLOCKS))

python main.py train cifar100 \
  --model_type dblock \
  --num_blocks "$NUM_BLOCKS" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$EPOCHS" \
  --postfix=-psc-rand-augment \
  --scheduler_type cosine_with_min_lr \
  --num_warmup_steps "$WARMUP_STEPS" \
  --lr 5e-4 \
  --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
  --add_rand_aug