# CIFAR-100 checkpoint continuation: training-only rental

Prepared November 8, 2026. This guide uses the training changes on the
`train-on-gpu` branch. Push that branch before cloning it on the rental machine.

## Which paper recipe applies?

This repository trains a CIFAR-100 classifier, not an image generator. The
[classification appendix in the current paper](https://arxiv.org/html/2506.14202#A5.SS1)
specifies a 12-layer ViT, width 128, four attention heads, 4x4 patches,
dropout 0.1, three blocks of four layers, AdamW, batch 128, learning rate
0.0005, cosine decay, 10-epoch warmup, 500 training epochs, RandAugment and
horizontal flips, and overlap gamma 0.05. It describes four denoising steps.
The [initial June 2025 version](https://arxiv.org/html/2506.14202v1) focuses on
generation and is not the classification recipe to copy into this project.

The repository multiplies `--num_epochs` by `--num_blocks`. Therefore
`--num_epochs 500 --num_blocks 3` schedules 1,500 actual data-loader epochs.
The current defaults also differ from the appendix: LR 0.001, constant schedule,
no RandAugment, and three inference sigma points. Paper prose and implementation
are not fully aligned; record exact commands rather than calling the defaults
an exact reproduction.

## Revised plan: official base defaults, 500 × 3 epochs

The requested baseline is the **official code's base recipe**: CIFAR-100,
all 50,000 training images, batch 128, three blocks, AdamW at LR 0.001,
weight decay 0.01, constant LR without warmup, gamma 0.05, no RandAugment,
seed 42, float32, and `--num_epochs 500`. This schedules 1,500 actual epochs
and 585,000 updates (390 updates per epoch). Keep the architecture unchanged.
Do not add a validation holdout to this reproduction run. W&B still logs
training CE/loss per block, LR and system metrics. Skip all testing on the
rental machine. Download the final checkpoint and evaluate it later locally.

Important distinction: these base defaults differ from the paper appendix's
augmented/cosine recipe. They reproduce the public default training procedure,
not a verified bit-for-bit copy of a particular published checkpoint.
[Official default settings](https://raw.githubusercontent.com/SakanaAI/DiffusionBlocks/main/main.py).

The existing `inference_results/dblock-checkpoint-150epochs.ckpt` contains
`epoch=149`, `global_step=58500`, batch 128, three blocks, LR 0.001,
constant schedule, no warmup and no RandAugment. It can be continued to the
500 × 3 target under the same base objective; 1,350 actual epochs and 526,500
updates remain. That saves 10% of the full training computation. Resume is
not bit-for-bit equivalent to an uninterrupted run: the current launcher
reseeds at startup and does not explicitly restore Python/NumPy/Torch RNG
states. The selected plan is continuation of this checkpoint, labeled as a resumed
official-base run. Do not start over.

### Runtime and cost

[Paper Appendix H](https://arxiv.org/html/2506.14202#A8) measures **0.0181 seconds
per block-training update on a single H100 80 GB**, averaged over 100 iterations.
For this continuation, **526,500 updates × 0.0181 seconds = 2.65 hours**
of training computation at the paper's measured throughput. The target remains
1,500 actual epochs; 150 are complete and **1,350 remain**. This does not include
setup, data loading, checkpoint saving or transfer. There will be no test or
solver sweep on the rental machine.

Allow **3.5–5.5 rented hours on an H100** as a provisional end-to-end planning
range. At Runpod's published H100 PCIe price of **$2.89/hour**, that is about
**$10–16 for compute**. Set aside **$20 total**, allowing some headroom for setup
and modest storage/transfer charges; this is not a fixed quote or a cost cap.
An H100 SXM at $3.99/hour gives about **$14–22** for the same assumed hours.
The paper does not specify enough details to guarantee the same throughput on
either rented configuration. [Current pricing](https://www.runpod.io/pricing).

A 4090 at $0.89/hour could be cheaper. A provisional **5.5–13.5 hours / $5–12**
compute scenario assumes roughly 2–5 times the H100 compute duration, without a
measured 4090 benchmark. Prefer the H100 for the more grounded timing estimate.

Measure the first 10–20 steady-state epochs of the resumed 500-target job,
including periodic checkpoint overhead. Estimate remaining training time as
`remaining_actual_epochs * average_epoch_seconds / 3600`. At the start, this is
`1350 * average_epoch_seconds / 3600`. For example, 10, 20, 30 and 60 seconds per
actual epoch imply 3.75, 7.5, 11.25 and 22.5 training hours, respectively, plus
setup and checkpoint download. Do not multiply by three again. Slow CPUs or data
loading can exceed the planning range.

Skipping the automatic final test saves some rental time, but training accounts
for most of the cost. The main saving versus a fresh run is reusing the completed
10% of training; deferring a large solver sweep avoids an additional workload.

## Rental options

1. **Runpod Pods — first choice for this project.** Prefer one H100 PCIe
   (80 GB); consider an RTX 4090 if minimizing cost takes priority. These are
   reasonable candidates for this small classifier, subject to the pilot's
   measured memory use. Published prices checked today: 4090 $0.89/hour,
   A40/A6000 $0.59/hour, A100 80 GB $1.79/hour. Select on-demand capacity.
   [Official pricing](https://www.runpod.io/pricing).
2. **Vast.ai — compare for the lowest total cost.** Look for a verified,
   high-reliability host with one 4090 or A6000, enough CPU/RAM, and an on-demand
   offer whose end date covers the run. Prices are live marketplace offers;
   include disk and transfer charges. Avoid interruptible capacity for the first
   long run. [Official pricing](https://vast.ai/pricing).
3. **Lambda — alternative if a conventional GPU VM is preferable.** Check
   single-GPU A6000/A10/A100 availability. Its public pricing page currently
   announces rates effective November 16, 2026, so those are not today's quotes;
   use the logged-in console for the current price. Do not rent an eight-GPU
   instance for this job. [Official pricing](https://lambda.com/pricing).

These are advertised prices, not confirmed capacity or an account-specific quote.
An H100 is not required just because the authors used one. Compare dollars per
completed training epoch rather than hourly rate alone.

## Runpod setup

Deploy a **Pod**, choose one GPU and a PyTorch/CUDA 12.4 development image,
and attach about 50 GB of persistent storage. Prefer at least 4 CPU cores and
16 GB system RAM. Keep the project, cache and checkpoints under `/workspace`.
A network volume survives Pod deletion; a regular volume disk survives stops
but is deleted with the Pod. Back up checkpoints before terminating the Pod.
[Storage documentation](https://docs.runpod.io/pods/storage/types).

On your local computer, publish the committed branch first:

```bash
git push -u origin train-on-gpu
```

In the rental machine's terminal or SSH connection, clone that branch explicitly:

```bash
cd /workspace
git clone --branch train-on-gpu --single-branch https://github.com/nhford/DiffusionBlocksPlusPlus.git
cd DiffusionBlocksPlusPlus
git branch --show-current
git rev-parse HEAD
```

The branch should print `train-on-gpu`. Record the commit hash in the run notes.
For an existing clean checkout, use `git fetch origin`, then
`git switch --track origin/train-on-gpu` (or `git switch train-on-gpu` if it already
exists). Do not use a checkout of `main` for these commands: it may lack the
training-only flag and other changes.

The checkpoint is not supplied by cloning the source branch. Upload your local
`inference_results/dblock-checkpoint-150epochs.ckpt` to the same relative path
under `/workspace/DiffusionBlocksPlusPlus` before starting the continuation.

Install into a dedicated environment:

```bash
cd /workspace/DiffusionBlocksPlusPlus
python -m pip install uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install lightning==2.5.0 wandb==0.30.0 transformers==4.52.4 \
  datasets==3.5.0 'huggingface-hub[hf-xet]>=0.31.2' \
  'hf-transfer>=0.1.9' 'scipy>=1.16.2'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
export USE_TF=0
export HF_HOME=/workspace/hf-cache
wandb login
```

These explicit installs avoid building `flash-attn`, which this ViT path does
not import. Use `python` from the activated environment for the commands below;
`uv run` would try to synchronize the full project dependency list again.
Public CIFAR-100 does not normally require Hugging Face login.

## Train and resume with W&B

Run inside a persistent terminal session, for example `tmux new -s dblock`
if tmux is available. Start the real 500-target job and check its first 10–20
epochs for stable loss, GPU use and throughput; keep it running if healthy.

```bash
export WANDB_PROJECT=dblocks-cifar100-solvers
export WANDB_RUN_GROUP=official-base-500
export WANDB_RUN_ID=dblock-official-base500-resumed150-seed42-v1
python -u -m src.diffusion_blocks_plus_plus.deploy_vit train cifar100 \
  --model_type dblock --num_blocks 3 --num_epochs 500 \
  --batch_size 128 --eval_batch_size 128 --num_workers 4 \
  --lr 0.001 --weight_decay 0.01 --gamma 0.05 \
  --scheduler_type constant_with_warmup --num_warmup_steps 0 \
  --validation_fraction 0 \
  --num_inference_steps 3 --save_every_n_epochs 5 \
  --precision 32-true --seed 42 --skip_test_after_fit \
  --ckpt_path /workspace/DiffusionBlocksPlusPlus/inference_results/dblock-checkpoint-150epochs.ckpt
```

Keep `--num_epochs 500`: it is the total target, not 500 additional epochs.
The new W&B run begins at the restored global training step; it does not recover
older W&B history. To resume a later interruption of this new job, use its
`last.ckpt`, keep the same W&B ID, and set `WANDB_RESUME=must`.

W&B displays the run link. Inspect `train/ce_loss`, block-specific CE losses,
`train/loss`, learning rate, and GPU utilization/memory. Weighted loss is noisy
because noise scales vary. Matching the base training dataset means no validation
accuracy curve; training curves are available and test accuracy will be measured
later on your own machine. Adding
`--validation_fraction 0.1` is a separate experiment, not this base reproduction.
Checkpoint artifacts upload when the logger finishes successfully; keep local
checkpoints on persistent storage for interruption recovery. Back up the final
checkpoint externally before terminating the rental machine.

When training ends, download `last.ckpt` from the run directory printed by the
launcher, along with the exact source revision and W&B run link. The source
checkpoint is about 121 MB; the final file should be of a similar order of size.
Verify its SHA-256 checksum matches on both machines. Inspect its metadata on
your own machine: the zero-based epoch should be 1499 and global step 585000.
Then terminate the Pod and remove unneeded paid storage. No inference is needed
to verify the file transfer or checkpoint metadata.

## Later: compare Euler, Heun and midpoint on your own machine

Use the final 1,500-epoch checkpoint and freeze it for all solvers. Predeclare
the solver grid and report it on all 10,000 official test images. Avoid selecting
training checkpoints or tuning the grid using repeated test results.

`--num_inference_steps S` currently means **S sigma points**, giving S-1
integration intervals and one final classification evaluation. Consequently,
Euler uses S denoiser calls, while Heun and midpoint use 2S-1. This is the code's
convention; do not equate it silently with the paper's step terminology.

```bash
unset WANDB_RUN_ID WANDB_RESUME
export WANDB_RUN_GROUP=cifar100-frozen-checkpoint-solver-comparison
CKPT=/absolute/local/path/to/downloaded-final-checkpoint.ckpt
for seed in 42 43 44; do
  for method in euler heun midpoint; do
    for points in 4 8 16 32; do
      WANDB_TAGS="$method,points-$points,seed-$seed" \
      python -u -m src.diffusion_blocks_plus_plus.deploy_vit test cifar100 \
        --model_type dblock --ckpt_path "$CKPT" \
        --num_blocks 3 --num_inference_steps "$points" --ode "$method" \
        --eval_batch_size 128 --num_workers 4 --precision 32-true --seed "$seed"
    done
  done
done
```

These seeds repeat inference noise, not training. Use identical seeds, batch
size, data order and precision across solvers. Report mean accuracy and its
variation over inference seeds. For stronger claims, repeat training with
independent seeds too. At equal point count, Heun and midpoint have equal call
counts; Euler is cheaper. Add compute-matched comparisons: Euler with 7/15/31
points versus Heun/midpoint with 4/8/16 points respectively. Measure latency
after GPU warmup, separately from process startup, downloads and W&B uploads.

Higher order methods are not guaranteed to improve accuracy: the denoising field
switches between independently trained blocks and may be insufficiently smooth,
especially on coarse grids. Treat the comparison as an experiment.

## Local fixes and verification

The companion source changes retain all validation/test examples; add an optional stratified training holdout; save
validation-selected checkpoints after validation; use the last checkpoint for
post-training evaluation when no validation set exists; support
`--skip_test_after_fit` for this training-only rental; expose precision; and
honor `WANDB_PROJECT`/`WANDB_RUN_ID` with checkpoint artifact logging.

Verified locally: edited files parse and evaluation loaders retain partial batches. Full data-loading, Lightning/W&B and
CUDA integration were not run locally because Lightning is absent from the local
Python environment. The rental-machine pilot is required before a long run.
