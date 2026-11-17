# CIFAR-100 DiffusionBlocks full runs

These commands start fresh on all 50,000 training images. The paper and inverted-frequency runs use 500 actual data passes. The old-code CLI multiplies the requested epochs by three, so 167 requests 501 actual passes, the closest its unmodified CLI gets to 500. All three use ten actual warmup epochs, seed 42, and the same paper training recipe.

## Training parameter record

Recorded from the launch commands below. EC2 logs and checkpoint metadata were not checked independently.

- **Shared settings** — CIFAR-100, all 50,000 training images, fresh weights, seed 42, batch size 128, three four-layer blocks, overlap 0.05, AdamW with weight decay 0.01 and peak LR 0.0005, cosine decay, ten warmup epochs, horizontal flip plus RandAugment, padded crop disabled, Euler with four grid points, and automatic post-fit testing disabled.
- **`dblock-old`** — checkout `a53e57f`, postfix `-dblock-old-full`, 167 requested epochs for 501 actual passes, original loss broadcasting and training block selection, original timestep frequencies and legacy inference routing. Milestones at 20, 50, and 100, plus regular checkpointing at pass 500 with `--save_top_k -1`.
- **`dblock-paper-full`** — corrected per-example EDM loss and training block selection, `--dblock_literal_epochs --num_epochs 500`, original timestep frequencies, and `--dblock_paper_grid`. Milestones at 20, 50, 100, 250, and 500.
- **`dblock-inverted-full`** — same parameters as the paper run with `--invert_timestep_frequencies` enabled.

## Set up on EC2

Run from your current checkout, with its working `.venv`:

```bash
export DBLOCK_REPO="$(pwd)"
git worktree add --detach ../project-dblock-old a53e57f
export DBLOCK_OLD="$(cd ../project-dblock-old && pwd)"
nvidia-smi -L
```

The current checkout needs commit `72d0f92` or later with the inversion flag. Run each block from the SSH shell. The exported paths are available when invoking the blocks. With one GPU, launch the next run after the previous one finishes. With three GPUs, change the three device assignments to 0, 1, and 2 before launching concurrently.

## Old code

```bash
tmux new-session -d -s dblock-old -c "$DBLOCK_OLD" \
  "CUDA_VISIBLE_DEVICES=0 '$DBLOCK_REPO/.venv/bin/python' -m src.diffusion_blocks_plus_plus.deploy_vit train cifar100 \
  --model_type dblock --num_blocks 3 --gamma 0.05 \
  --num_epochs 167 --batch_size 128 --seed 42 --validation_fraction 0.0 \
  --lr 5e-4 --scheduler_type cosine --num_warmup_epochs 10 \
  --no_padded_crop --add_rand_aug --ode euler --num_inference_steps 4 \
  --checkpoint_epochs 20 50 100 --save_every_n_epochs 500 --save_top_k -1 --skip_test_after_fit \
  --postfix=-dblock-old-full > dblock-old-full.log 2>&1"
```

The old checkout uses the working Python environment from the current checkout to avoid a FlashAttention dependency sync.

## Paper recipe with original timestep frequencies

```bash
tmux new-session -d -s dblock-paper -c "$DBLOCK_REPO" \
  "CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.diffusion_blocks_plus_plus.deploy_vit train cifar100 \
  --model_type dblock --num_blocks 3 --gamma 0.05 \
  --dblock_literal_epochs --num_epochs 500 --batch_size 128 --seed 42 --validation_fraction 0.0 \
  --lr 5e-4 --scheduler_type cosine --num_warmup_epochs 10 \
  --no_padded_crop --add_rand_aug --ode euler --num_inference_steps 4 --dblock_paper_grid \
  --checkpoint_epochs 20 50 100 250 500 --skip_test_after_fit \
  --postfix=-dblock-paper-full > dblock-paper-full.log 2>&1"
```

## Paper recipe with inverted timestep frequencies

```bash
tmux new-session -d -s dblock-inverted -c "$DBLOCK_REPO" \
  "CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m src.diffusion_blocks_plus_plus.deploy_vit train cifar100 \
  --model_type dblock --num_blocks 3 --gamma 0.05 \
  --dblock_literal_epochs --num_epochs 500 --batch_size 128 --seed 42 --validation_fraction 0.0 \
  --lr 5e-4 --scheduler_type cosine --num_warmup_epochs 10 \
  --no_padded_crop --add_rand_aug --ode euler --num_inference_steps 4 --dblock_paper_grid \
  --invert_timestep_frequencies \
  --checkpoint_epochs 20 50 100 250 500 --skip_test_after_fit \
  --postfix=-dblock-inverted-full > dblock-inverted-full.log 2>&1"
```

## Check progress and evaluate

Follow the output with `tail -f dblock-paper-full.log` from the current checkout, or use the corresponding log in the old checkout. Stopping `tail` leaves training running. Each checkout saves checkpoints under its own `logs/` directory.

Use `milestone-500.ckpt` from the paper and inverted runs for the matched 500-pass comparison. For the old run, use the regular checkpoint written after pass 500, whose filename starts with `epoch=499-step=` because Lightning numbers epochs from zero. The old CLI rejects milestone arguments above 167, so the regular checkpoint callback saves that checkpoint instead. Run all evaluations from the current checkout so they share the same sampler implementation. Set `CKPT` to the actual checkpoint path, using the old checkout's absolute path for the old-code checkpoint.

```bash
uv run --no-sync python -m src.diffusion_blocks_plus_plus.deploy_vit test cifar100 \
  --model_type dblock --num_blocks 3 --seed 42 \
  --ode euler --num_inference_steps 4 --dblock_paper_grid \
  --ckpt_path "$CKPT"
```

With three blocks, `--dblock_paper_grid` requires at least four grid points. For a two- or three-point test, remove that flag to use legacy routing. For a sweep across 2, 3, 4, 5, 6, 8, 16, and 32 points, omit the flag for every count to keep routing consistent.

Add `--invert_timestep_frequencies` only for the inverted checkpoint. For the inference-point comparison, repeat with 8, 16, and 32 while keeping the checkpoint, seed, solver, and routing fixed. Confirm `test/examples` is 10,000.

## Regenerate the sweep figure

From the repository root, run the following with a Python environment that has Matplotlib installed.

```bash
python inference_results/plot_results.py
```

The script reads [the recorded sweep CSV](dblock-full-sweep-results.csv) and writes [the side-by-side figure](dblock-full-sweeps.png). Use `--legacy-logs` to regenerate the earlier plot from individual evaluation logs.
